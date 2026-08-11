package com.emilio.whispert.models

import android.content.Context
import android.net.Uri
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

    /** Returns the cached model file if present (fully downloaded/imported), else null. */
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
            ?: importedById(id)?.takeIf { isDownloaded(it) }
    }

    /** Deletes a cached model file (and clears active selection if it was active). */
    fun deleteModel(desc: ModelDescriptor) {
        cachedFile(desc)?.delete()
        if (activeModelId() == desc.id) setActiveModelId(null)
        if (desc.id.startsWith(IMPORTED_PREFIX)) removeImported(desc.id)
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

            // The server's Content-Length is the truth about the file. Fall back to
            // the descriptor's hardcoded size only if the server is using chunked
            // encoding (chunked => -1). This makes the truncation check robust to
            // upstream re-uploads that slightly shift the file size.
            val expectedLen: Long = conn.contentLengthLong.takeIf { it > 0 } ?: desc.sizeBytes
            val total = expectedLen.toFloat()
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

            // Atomic-ish rename over the final path. Validates final length
            // against the server-declared Content-Length (or the descriptor size as
            // a fallback), not the static hardcoded number.
            if (target.exists()) target.delete()
            check(tmp.renameTo(target)) { "Failed to rename ${tmp.name} -> ${target.name}" }
            require(target.length() == expectedLen) {
                "Truncated model: got ${target.length()} / expected $expectedLen B"
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

    /**
     * Reads a previously-downloaded / user-imported .bin model pointed at by a
     * SAF [uri] (any file the document picker can reach). Copies it into
     * filesDir/models/ggml-imported-<N>.bin so whisper_init_from_file can mmap it,
     * and registers a runtime [ModelDescriptor] with id=`imported-<N>` so it
     * shows up in the Models tab alongside the registry entries.
     *
     * The descriptor is conservatively flagged multilingual=true so the language
     * selector stays enabled in the Transcribe tab. English-only .bin files work
     * too; the user just has to pick 'en' explicitely.
     *
     * Returns the descriptor of the imported model.
     */
    suspend fun importFromFile(uri: Uri): ModelDescriptor = withContext(Dispatchers.IO) {
        val sequence = nextImportedSequence()
        val id = "$IMPORTED_PREFIX$sequence"
        val target = File(modelDir, "ggml-$id.bin")
        Log.i(TAG, "Importing model from $uri -> ${target.absolutePath}")

        var total = 0L
        val sha = MessageDigest.getInstance("SHA-256")
        try {
            context.contentResolver.openInputStream(uri).use { input ->
                requireNotNull(input) { "Cannot open InputStream for $uri" }
                FileOutputStream(target).use { out ->
                    val buf = ByteArray(64 * 1024)
                    while (true) {
                        val n = input.read(buf)
                        if (n <= 0) break
                        out.write(buf, 0, n)
                        sha.update(buf, 0, n)
                        total += n
                        ensureActive()
                    }
                }
            }
        } catch (t: Throwable) {
            target.delete()
            Log.e(TAG, "Import failed: ${t.message}", t)
            throw t
        }
        Log.i(TAG, "Imported $id done ($total B)")

        val displayName = "Imported (#$sequence)"
        val desc = ModelDescriptor(
            id = id,
            displayName = displayName,
            sizeBytes = total,
            url = "",
            multilingual = true,
            sha256 = null,
            description = "Loaded from a file you picked with SAF. You can pick language in the Transcribe tab; for English-only .bin choose 'en'.",
            recommendedFor = emptySet(),
        )
        saveImported(desc)
        setActiveModelId(id)
        desc
    }

    // -- imported-models runtime registry ---------------------------------

    /** Lists every imported model whose file still exists on disk. */
    fun listImported(): List<ModelDescriptor> {
        val ids = prefs.getStringSet(KEY_IMPORTED_IDS, emptySet()) ?: emptySet()
        return ids.mapNotNull { id -> importedById(id)?.takeIf { cachedFile(it) != null } }
    }

    private fun importedById(id: String): ModelDescriptor? {
        val str = prefs.getString("imp#$id", null) ?: return null
        val parts = str.split(',')
        if (parts.size < 4) return null
        return runCatching {
            ModelDescriptor(
                id = id,
                displayName = parts[0],
                sizeBytes = parts[1].toLong(),
                url = parts[2],
                multilingual = parts[3] == "1",
                sha256 = null,
                description = "Imported from file.",
                recommendedFor = emptySet(),
            )
        }.getOrNull()
    }

    private fun saveImported(desc: ModelDescriptor) {
        val ids = (prefs.getStringSet(KEY_IMPORTED_IDS, emptySet()) ?: emptySet()).toMutableSet()
        ids += desc.id
        prefs.edit()
            .putStringSet(KEY_IMPORTED_IDS, ids)
            .putString("imp#${desc.id}", "${desc.displayName},${desc.sizeBytes},${desc.url},${if (desc.multilingual) "1" else "0"}")
            .apply()
        // Keep our private counter monotonic.
        prefs.edit().putInt(KEY_IMPORTED_SEQ, desc.id.substringAfter(IMPORTED_PREFIX).toIntOrNull() ?: prefs.getInt(KEY_IMPORTED_SEQ, 0)).apply()
    }

    private fun removeImported(id: String) {
        val ids = (prefs.getStringSet(KEY_IMPORTED_IDS, emptySet()) ?: emptySet()).toMutableSet()
        ids.remove(id)
        prefs.edit()
            .putStringSet(KEY_IMPORTED_IDS, ids)
            .remove("imp#$id")
            .apply()
    }

    private fun nextImportedSequence(): Int {
        val seq = prefs.getInt(KEY_IMPORTED_SEQ, 0) + 1
        prefs.edit().putInt(KEY_IMPORTED_SEQ, seq).apply()
        return seq
    }

    companion object {
        private const val TAG = "ModelManager"
        private const val PREFS = "whispert_models"
        private const val KEY_ACTIVE = "active_model_id"
        private const val IMPORTED_PREFIX = "imported-"
        private const val KEY_IMPORTED_IDS = "imported_ids"
        private const val KEY_IMPORTED_SEQ = "imported_seq"
    }
}