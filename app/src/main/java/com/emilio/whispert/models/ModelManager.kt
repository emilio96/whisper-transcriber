package com.emilio.whispert.models

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Owns the on-device model cache and orchestrates downloads.
 *
 * - Models live in `context.filesDir/models/ggml-<id>.bin` (private internal
 *   storage, no permissions, survives across launches).
 * - Downloads stream with a progress callback. HttpURLConnection follows the
 *   huggingface.co -> CDN redirect transparently (`instanceFollowRedirects`).
 * - On completion, if [ModelDescriptor.sha256] is set we verify the digest and
 *   delete the partial file on mismatch. For MVP most descriptors have null,
 *   which skips the check (acceptable: HTTPS already protects integrity).
 * - The "active" model id is persisted in SharedPreferences so the ViewModel
 *   reattaches to the same model on relaunch.
 *
 * Designed to be instantiated per-ViewModel (AndroidViewModel holds the app
 * context), or as a singleton bound to application context if needed later.
 */
class ModelManager(private val context: Context) {

    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    val modelDir: File = File(context.filesDir, "models").apply { mkdirs() }

    /** Returns the cached model file if present (fully downloaded), else null. */
    fun cachedFile(desc: ModelDescriptor): File? {
        val f = File(modelDir, desc.localFilename())
        return if (f.exists() && f.length() == desc.sizeBytes) f else null
    }

    fun isDownloaded(desc: ModelDescriptor): Boolean = cachedFile(desc) != null

    fun activeModelId(): String? = prefs.getString(KEY_ACTIVE, null)

    fun setActiveModelId(id: String?) {
        prefs.edit().apply {
            if (id == null) remove(KEY_ACTIVE) else putString(KEY_ACTIVE, id)
        }.apply()
    }

    /** Returns the descriptor for the active (persisted) model, or null if none/unavailable. */
    fun activeModel(): ModelDescriptor? {
        val id = activeModelId() ?: return null
        return runCatching { ModelRegistry.byId(id) }.getOrNull()?.takeIf { isDownloaded(it) }
    }

    /** Deletes a cached model file (and clears active selection if it was active). */
    fun deleteModel(desc: ModelDescriptor) {
        cachedFile(desc)?.delete()
        if (activeModelId() == desc.id) setActiveModelId(null)
    }

    /**
     * Streams [desc] to disk, calling [onProgress] with a fraction in [0,1].
     * Throws on failure (caller surfaces the message). On success the file is
     * complete on disk and the active selection is set.
     */
    suspend fun download(
        desc: ModelDescriptor,
        onProgress: (Float) -> Unit = {},
    ): File = withContext(Dispatchers.IO) {
        val target = File(modelDir, desc.localFilename())
        Log.i(TAG, "Downloading ${desc.id} (${desc.sizeBytes} B) -> ${target.absolutePath}")

        var conn: HttpURLConnection? = null
        val tmp = File(modelDir, desc.localFilename() + ".part")
        try {
            conn = (URL(desc.url).openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = true
                connectTimeout = 20_000
                readTimeout = 60_000
            }
            val code = conn.responseCode
            require(code in 200..299) { "HTTP $code for ${desc.url}" }

            val total = (conn.contentLengthLong.takeIf { it > 0 } ?: desc.sizeBytes).toFloat()
            val sha = desc.sha256?.let { MessageDigest.getInstance("SHA-256") }
            FileOutputStream(tmp).use { out ->
                conn.inputStream.use { input ->
                    val buf = ByteArray(64 * 1024)
                    var read = 0L
                    while (true) {
                        val n = input.read(buf)
                        if (n <= 0) break
                        out.write(buf, 0, n)
                        sha?.update(buf, 0, n)
                        read += n
                        ensureActive()
                        onProgress((read.toFloat() / total).coerceIn(0f, 0.999f))
                    }
                }
            }

            // Integrity check (only when the descriptor tracks a digest).
            if (sha != null && desc.sha256 != null) {
                val got = sha.digest().joinToString("") { "%02x".format(it) }
                check(got.equals(desc.sha256, ignoreCase = true)) {
                    "SHA-256 mismatch: expected ${desc.sha256}, got $got"
                }
            }

            // Atomic-ish rename over the final path. Validates final length.
            if (target.exists()) target.delete()
            check(tmp.renameTo(target)) { "Failed to rename ${tmp.name} -> ${target.name}" }
            require(target.length() == desc.sizeBytes) {
                "Truncated model: ${target.length()} / ${desc.sizeBytes} B"
            }

            setActiveModelId(desc.id)
            onProgress(1f)
            Log.i(TAG, "Download ${desc.id} done (${target.length()} B)")
            target
        } catch (t: Throwable) {
            tmp.delete()
            target.delete()
            Log.e(TAG, "Download failed: ${t.message}", t)
            throw t
        } finally {
            conn?.disconnect()
        }
    }

    companion object {
        private const val TAG = "ModelManager"
        private const val PREFS = "whispert_models"
        private const val KEY_ACTIVE = "active_model_id"
    }
}