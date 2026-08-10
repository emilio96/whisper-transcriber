package com.emilio.whispert.whisper

import android.content.res.AssetManager
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.withContext
import java.io.InputStream
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

private const val LOG_TAG = "WhisperContext"

/**
 * Owns a whisper cpp context handle ([ptr]) and funnels every native call onto a
 * single dedicated worker thread (whisper.cpp is not thread-safe).
 *
 * Holds native heap memory (~size of the loaded model) until [release] is called;
 * the owner (e.g. a ViewModel) **must** call [release] in its onCleared to avoid
 * leaking native memory when the screen is destroyed.
 */
class WhisperContext private constructor(private var ptr: Long) {

    private val inferenceExecutor = Executors
        .newSingleThreadExecutor { r -> Thread(r, "whisper-inference") }
    private val singleThreadDispatcher = inferenceExecutor.asCoroutineDispatcher()
    private val singleThreadScope = CoroutineScope(singleThreadDispatcher)

    /**
     * Transcribes [audioData] (16 kHz mono float32 samples in [-1, 1]).
     * Returns the full transcript, optionally prefixed with segment timestamps like
     * [00:00:00.000 --> 00:00:01.000] testo.
     */
    suspend fun transcribe(
        audioData: FloatArray,
        language: String = "en",
        printTimestamps: Boolean = true,
    ): String = withContext(singleThreadDispatcher) {
        check(ptr != 0L) { "WhisperContext has already been released" }
        val numThreads = WhisperCpuConfig.preferredThreadCount
        Log.d(LOG_TAG, "Transcribing ${audioData.size} samples with $numThreads threads")
        WhisperLib.fullTranscribe(ptr, numThreads, language, audioData)

        val segmentCount = WhisperLib.getTextSegmentCount(ptr)
        buildString {
            for (i in 0 until segmentCount) {
                val text = WhisperLib.getTextSegment(ptr, i)
                if (printTimestamps) {
                    val t0 = WhisperLib.getTextSegmentT0(ptr, i)
                    val t1 = WhisperLib.getTextSegmentT1(ptr, i)
                    append("[${toTimestamp(t0)} --> ${toTimestamp(t1)}] $text\n")
                } else {
                    append(text)
                }
            }
        }
    }

    /** Graceful async release. Use from coroutines/UI code. */
    suspend fun release() = withContext(singleThreadDispatcher) {
        freeNativeAndShutdown()
    }

    /**
     * Synchronous release meant to be called from a non-coroutine teardown path
     * such as ViewModel.onCleared(). Blocks (briefly) on the inference thread to
     * guarantee the native context is freed before the process lets go of it,
     * which avoids leaking native heap when the screen goes away.
     */
    fun releaseBlocking() {
        if (inferenceExecutor.isShutdown) return
        try {
            val future = inferenceExecutor.submit { freeNativeAndShutdownInline() }
            // whisper_free for a small model is sub-millisecond; allow a safety margin.
            runCatching { future.get(2, TimeUnit.SECONDS) }
        } catch (_: Exception) {
            // executor rejected (e.g., already shutting down) -> still try native free inline
            freeNativeAndShutdownInline()
        }
        inferenceExecutor.shutdown()
    }

    private fun freeNativeAndShutdownInline() {
        if (ptr != 0L) {
            WhisperLib.freeContext(ptr)
            ptr = 0L
            Log.d(LOG_TAG, "Native context released (blocking)")
        }
        singleThreadScope.cancel()
    }

    private suspend fun freeNativeAndShutdown() {
        freeNativeAndShutdownInline()
    }

    protected fun finalize() {
        if (ptr != 0L) {
            Log.w(LOG_TAG, "WhisperContext was not released before GC; freeing native memory in finalizer")
            WhisperLib.freeContext(ptr)
            ptr = 0L
        }
    }

    companion object {
        /** Loads a model from an absolute filesystem path (the main path for this app). */
        fun createContextFromFile(filePath: String): WhisperContext {
            val ptr = WhisperLib.initContext(filePath)
            require(ptr != 0L) { "Couldn't create context from file: $filePath" }
            return WhisperContext(ptr)
        }

        /** Loads a model bundled as an APK asset (path relative to assets/). */
        fun createContextFromAsset(assetManager: AssetManager, assetPath: String): WhisperContext {
            val ptr = WhisperLib.initContextFromAsset(assetManager, assetPath)
            require(ptr != 0L) { "Couldn't create context from asset: $assetPath" }
            return WhisperContext(ptr)
        }

        /** Loads a model from any Java InputStream (e.g. a SAF content Uri). */
        fun createContextFromInputStream(stream: InputStream): WhisperContext {
            val ptr = WhisperLib.initContextFromInputStream(stream)
            require(ptr != 0L) { "Couldn't create context from input stream" }
            return WhisperContext(ptr)
        }

        fun getSystemInfo(): String = WhisperLib.getSystemInfo()
    }
}

//  500 -> 00:05.000 ; t is in units of 10 ms (whisper t0/t1)
private fun toTimestamp(t: Long): String {
    var msec = t * 10
    val hr = msec / (1000 * 60 * 60)
    msec -= hr * (1000 * 60 * 60)
    val min = msec / (1000 * 60)
    msec -= min * (1000 * 60)
    val sec = msec / 1000
    msec -= sec * 1000
    return String.format("%02d:%02d:%02d.%03d", hr, min, sec, msec)
}