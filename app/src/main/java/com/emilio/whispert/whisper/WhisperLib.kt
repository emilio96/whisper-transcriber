package com.emilio.whispert.whisper

import android.os.Build
import android.util.Log

private const val LOG_TAG = "WhisperLib"

/** Raised when the device CPU cannot run the kernels libwhisper.so was built with. */
class UnsupportedCpuException(message: String) : RuntimeException(message)

/**
 * Thin Kotlin binding to the whisper.cpp native library loaded via JNI.
 *
 * WARNING: every function here touches whisper.cpp, which is **not thread-safe**.
 * All calls are funneled through [WhisperContext]'s single-thread dispatcher.
 *
 * ggml is compiled with `-march=armv8.2-a+dotprod+fp16` (see CMakeLists.txt), so
 * SDOT and fp16 vector arithmetic are hard requirements. An earlier revision
 * shipped a second "generic" .so meant as a fallback, but both variants linked
 * the same ggml and therefore the same instructions: the fallback would have
 * crashed with SIGILL just the same. We now ship one library and check the CPU
 * features before loading it, so an unsupported device gets a readable error
 * instead of an illegal instruction deep inside a matmul.
 */
internal class WhisperLib {
    companion object {
        /**
         * Requests the native whisper_full loop to abort at the next graph
         * boundary (every decode batch / encode). Set atomically; reset at the
         * start of the next fullTranscribe run. No-op when idle.
         */
        external fun cancelTranscription()

        // Set once the native library has been bound. Logged at startup under
        // the "WhisperLib" tag so you can verify which build is active without
        // digging through the APK.
        var loadedLibName: String? = null
            private set

        init {
            Log.d(LOG_TAG, "Primary ABI: ${Build.SUPPORTED_ABIS.joinToString()}")

            val info = cpuInfo()
            if (info != null) {
                Log.d(LOG_TAG, "CPU info:\n$info")
                val hasFp16 = info.contains("fphp")
                val hasDotprod = info.contains("asimddp")
                if (!hasFp16 || !hasDotprod) {
                    val missing = buildList {
                        if (!hasDotprod) add("asimddp (SDOT)")
                        if (!hasFp16) add("fphp (fp16 arithmetic)")
                    }.joinToString(", ")
                    throw UnsupportedCpuException(
                        "This device's CPU lacks $missing. Whisper Transcriber requires an " +
                            "ARMv8.2-A processor with dot-product and fp16 support."
                    )
                }
            } else {
                Log.w(LOG_TAG, "Could not read /proc/cpuinfo; loading the library optimistically")
            }

            Log.d(LOG_TAG, "Loading libwhisper.so (armv8.2-a+dotprod+fp16)")
            System.loadLibrary("whisper")
            loadedLibName = "libwhisper.so (armv8.2-a+dotprod+fp16)"
        }

        // ---- JNI methods -----------------------------------------------------
        external fun initContext(modelPath: String): Long
        external fun freeContext(contextPtr: Long)

        /**
         * In-memory transcription for short clips (the bundled smoke test).
         * Returns 0 on success, 1 when the user aborted, negative on failure.
         */
        external fun fullTranscribe(
            contextPtr: Long,
            numThreads: Int,
            language: String?,
            initialPrompt: String?,
            temperature: Float,
            temperatureInc: Float,
            entropyThold: Float,
            logprobThold: Float,
            noSpeechThold: Float,
            noContext: Boolean,
            singleSegment: Boolean,
            translate: Boolean,
            suppressBlank: Boolean,
            suppressNonSpeechTokens: Boolean,
            audioData: FloatArray,
        ): Int

        /**
         * Streaming transcription: consumes a WAV file written by
         * AudioDecoder.decodeToWavFile (mono PCM16 16 kHz). The native side
         * parses the header, converts PCM->float into malloc'd memory and runs
         * whisper_full, so the Java heap never holds the whole signal.
         *
         * Returns 0 on success, 1 when the user aborted, negative on failure.
         */
        external fun fullTranscribeFile(
            contextPtr: Long,
            numThreads: Int,
            language: String?,
            initialPrompt: String?,
            temperature: Float,
            temperatureInc: Float,
            entropyThold: Float,
            logprobThold: Float,
            noSpeechThold: Float,
            noContext: Boolean,
            singleSegment: Boolean,
            translate: Boolean,
            suppressBlank: Boolean,
            suppressNonSpeechTokens: Boolean,
            wavPath: String,
        ): Int

        external fun getTextSegmentCount(contextPtr: Long): Int
        external fun getTextSegment(contextPtr: Long, index: Int): String
        external fun getTextSegmentT0(contextPtr: Long, index: Int): Long
        external fun getTextSegmentT1(contextPtr: Long, index: Int): Long
        external fun getTimings(contextPtr: Long): String
        external fun getModelInfo(contextPtr: Long): String

        external fun nativeDecodeAudioFd(
            fd: Int,
            offset: Long,
            length: Long,
            outWavPath: String
        ): Int

        /** DSP voiceprint for one segment. Returns an empty array when the segment is too short. */
        external fun extractSegmentEmbedding(
            audioData: FloatArray,
            segT0: Long,
            segT1: Long
        ): FloatArray
    }
}

/**
 * Bridge for native callbacks:
 * - whisper_full invokes onProgress(int) during encoder/decoder steps
 * - whisper_full invokes onNewSegment(text, t0, t1) whenever a segment is ready
 * - nativeDecodeAudioFd invokes onDecodeProgress(int) during audio decoding
 *
 * These are process-wide singletons because the native side caches static method
 * IDs. Every setter site must save the previous value and restore it in a
 * `finally`, so nested users (decode inside a transcription run) do not clobber
 * each other. [withListeners] does that for you.
 */
object WhisperProgress {
    @Volatile
    var progressListener: ((Int) -> Unit)? = null

    @Volatile
    var decodeListener: ((Int) -> Unit)? = null

    @Volatile
    var segmentListener: ((text: String, t0: Long, t1: Long) -> Unit)? = null

    /**
     * Installs the given listeners for the duration of [block], restoring
     * whatever was set before on the way out. Passing null for a listener
     * clears it for the duration of the block.
     */
    inline fun <T> withListeners(
        noinline progress: ((Int) -> Unit)? = null,
        noinline decode: ((Int) -> Unit)? = null,
        noinline segment: ((text: String, t0: Long, t1: Long) -> Unit)? = null,
        block: () -> T,
    ): T {
        val prevProgress = progressListener
        val prevDecode = decodeListener
        val prevSegment = segmentListener
        progressListener = progress
        decodeListener = decode
        segmentListener = segment
        try {
            return block()
        } finally {
            progressListener = prevProgress
            decodeListener = prevDecode
            segmentListener = prevSegment
        }
    }

    @JvmStatic
    fun onProgress(progress: Int) {
        progressListener?.invoke(progress)
    }

    @JvmStatic
    fun onDecodeProgress(progress: Int) {
        decodeListener?.invoke(progress)
    }

    @JvmStatic
    fun onNewSegment(text: String, t0: Long, t1: Long) {
        segmentListener?.invoke(text, t0, t1)
    }
}

private fun cpuInfo(): String? = try {
    java.io.File("/proc/cpuinfo").inputStream().bufferedReader().use { it.readText() }
} catch (e: Exception) {
    Log.w(LOG_TAG, "Couldn't read /proc/cpuinfo", e)
    null
}
