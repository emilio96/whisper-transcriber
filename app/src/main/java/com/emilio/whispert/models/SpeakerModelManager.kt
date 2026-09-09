package com.emilio.whispert.models

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

private const val TAG = "SpeakerModelManager"

/** Current on-disk name. Bumped when the shipped model changes. */
private const val SPEAKER_MODEL_FILE = "campplus_zh_en_fp16.onnx"

/** Files from earlier versions, deleted on sight so they stop wasting storage. */
private val LEGACY_MODEL_FILES = listOf(
    "speaker_campplus.onnx",
    "speaker_campplus_int8.onnx",
)

/** Exact size of the fp16 model, verified against the publisher's manifest. */
const val SPEAKER_MODEL_SIZE_BYTES = 14_695_579L

/** Pinned digest of that exact file. */
private const val SPEAKER_MODEL_SHA256 =
    "62eb2d79d363c1fd5ee093a4b0dcb5470d5ad3b7452612b67cce9b89f36c8ef3"

/** Human-readable size for the UI. */
const val SPEAKER_MODEL_SIZE_LABEL = "14 MB"

/**
 * CAM++ (D-TDNN) speaker embedding, 192-dim, 80-bin fbank at 16 kHz.
 *
 * This is the half-precision build of 3D-Speaker's
 * `speech_campplus_sv_zh_en_16k-common_advanced`, the code-switched
 * Mandarin+English model. Two reasons for this exact artifact:
 *
 *  - **fp16 rather than int8.** On ARM the int8 build is ~2.8x *slower* than
 *    fp32 (287 ms vs 102 ms per utterance on a Pixel 6) because ONNX Runtime's
 *    mobile build has no optimised MatMulInteger kernel for this graph, and it
 *    loses accuracy on top. fp16 is ~1.45x faster than fp32 (70 ms), half the
 *    size, and scores within noise of it. Quantisation is not automatically a
 *    speedup.
 *  - **zh_en rather than zh-cn.** The previous model was the Mandarin-only one;
 *    this one is trained on code-switched Mandarin+English and behaves better on
 *    English speech.
 *
 * Provenance: the fp32 file in the same publisher repository is byte-identical
 * (SHA-256 aa3cfc16…) to the official sherpa-onnx release artifact, so the fp16
 * is a straight cast of the upstream model rather than an unrelated retrain. The
 * digest below is pinned regardless.
 */
private const val PRIMARY_URL =
    "https://huggingface.co/Luigi/campplus-zh-en-onnx/resolve/main/campplus_zh_en_fp16.onnx"

/**
 * Full-precision fallback, straight from the sherpa-onnx release. Larger and
 * slower but from the upstream project, used when the fp16 mirror is
 * unreachable.
 */
private const val FALLBACK_URL =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
private const val FALLBACK_SIZE_BYTES = 28_281_164L
private const val FALLBACK_SHA256 =
    "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2"

private class Mirror(val url: String, val sizeBytes: Long, val sha256: String)

private val MIRRORS = listOf(
    Mirror(PRIMARY_URL, SPEAKER_MODEL_SIZE_BYTES, SPEAKER_MODEL_SHA256),
    Mirror(FALLBACK_URL, FALLBACK_SIZE_BYTES, FALLBACK_SHA256),
)

/**
 * Manages on-device storage and download of the neural speaker embedding model.
 */
class SpeakerModelManager(private val context: Context) {

    val modelDir: File = File(context.filesDir, "models").apply { mkdirs() }

    private val currentModel: File get() = File(modelDir, SPEAKER_MODEL_FILE)

    fun getModelFile(): File? {
        val f = currentModel
        return if (f.exists() && f.length() > 1_000_000L) f else null
    }

    fun isDownloaded(): Boolean = getModelFile() != null

    /** Removes files left behind by previous versions of the app. */
    fun purgeLegacyModels() {
        for (name in LEGACY_MODEL_FILES) {
            val f = File(modelDir, name)
            if (f.exists() && f.delete()) {
                Log.i(TAG, "Removed superseded speaker model $name")
            }
        }
    }

    fun deleteModel(): Boolean {
        purgeLegacyModels()
        File(modelDir, "$SPEAKER_MODEL_FILE.part").takeIf { it.exists() }?.delete()
        val f = currentModel
        return if (f.exists()) f.delete() else true
    }

    private fun openConnectionWithRedirects(initialUrl: String): HttpURLConnection {
        var currentUrl = initialUrl
        var redirects = 0
        while (redirects < 6) {
            val url = URL(currentUrl)
            val conn = (url.openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = true
                connectTimeout = 20_000
                readTimeout = 60_000
                setRequestProperty("User-Agent", "Mozilla/5.0 (Android; WhisperTranscriber/1.0)")
            }
            val status = conn.responseCode
            if (status == HttpURLConnection.HTTP_MOVED_TEMP ||
                status == HttpURLConnection.HTTP_MOVED_PERM ||
                status == HttpURLConnection.HTTP_SEE_OTHER ||
                status == 307 || status == 308
            ) {
                val newUrl = conn.getHeaderField("Location")
                if (newUrl == null) {
                    // A redirect status with no Location header: bail out, but
                    // not before releasing the connection like every other path.
                    conn.disconnect()
                    throw IllegalStateException("Redirect from $currentUrl carried no Location header")
                }
                conn.disconnect()
                currentUrl = if (newUrl.startsWith("http://") || newUrl.startsWith("https://")) {
                    newUrl
                } else {
                    URL(url, newUrl).toString()
                }
                redirects++
                continue
            }
            return conn
        }
        throw IllegalStateException("Too many redirects for $initialUrl")
    }

    suspend fun download(onProgress: (Float) -> Unit = {}): File = withContext(Dispatchers.IO) {
        val target = currentModel
        val tmp = File(modelDir, "$SPEAKER_MODEL_FILE.part")
        if (tmp.exists()) tmp.delete()

        Log.i(TAG, "Downloading neural speaker model -> ${target.absolutePath}")

        var lastError: Throwable? = null
        for (mirror in MIRRORS) {
            var conn: HttpURLConnection? = null
            try {
                conn = openConnectionWithRedirects(mirror.url)
                val code = conn.responseCode
                if (code !in 200..299) {
                    throw IllegalStateException("HTTP $code from ${mirror.url}")
                }

                val expectedLen = conn.contentLengthLong.takeIf { it > 0 } ?: mirror.sizeBytes
                val total = expectedLen.toFloat()
                val digest = MessageDigest.getInstance("SHA-256")

                FileOutputStream(tmp).use { out ->
                    conn.inputStream.use { input ->
                        val buf = ByteArray(64 * 1024)
                        var read = 0L
                        while (true) {
                            ensureActive()
                            val n = input.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            digest.update(buf, 0, n)
                            read += n
                            onProgress((read / total).coerceIn(0f, 1f))
                        }
                        out.flush()
                    }
                }

                // Verify before the file can be promoted: a truncated or altered
                // download must never reach the path ONNX Runtime will parse.
                check(tmp.length() == mirror.sizeBytes) {
                    "Size mismatch: got ${tmp.length()} / expected ${mirror.sizeBytes} B"
                }
                val got = digest.digest().joinToString("") { "%02x".format(it) }
                check(got.equals(mirror.sha256, ignoreCase = true)) {
                    "SHA-256 mismatch: expected ${mirror.sha256}, got $got"
                }

                if (target.exists()) target.delete()
                if (!tmp.renameTo(target)) {
                    tmp.copyTo(target, overwrite = true)
                    tmp.delete()
                }
                purgeLegacyModels()

                Log.i(TAG, "Downloaded and verified speaker model: ${target.length()} bytes")
                return@withContext target
            } catch (t: Throwable) {
                Log.w(TAG, "Download attempt failed from ${mirror.url}: ${t.message}", t)
                lastError = t
                if (tmp.exists()) tmp.delete()
            } finally {
                conn?.disconnect()
            }
        }

        throw lastError ?: IllegalStateException("Failed to download speaker model from all mirrors")
    }
}
