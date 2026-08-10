package com.emilio.whispert.whisper

import android.content.res.AssetManager
import android.os.Build
import android.util.Log
import java.io.InputStream

private const val LOG_TAG = "WhisperLib"

/**
 * Thin Kotlin binding to the whisper.cpp native library loaded via JNI.
 *
 * WARNING: every function here touches whisper.cpp, which is **not thread-safe**.
 * All calls are funneled through [WhisperContext]'s single-thread dispatcher.
 *
 * The library is actually shipped in two flavours:
 *   - `libwhisper_v8fp16_va.so` : built with -march=armv8.2-a+fp16, used on SoCs
 *     whose /proc/cpuinfo advertises "fphp" (fp16 arithmethic), e.g. Snapdragon
 *     8 Elite in the Xiaomi 17.
 *   - `libwhisper.so`            : the generic arm64-v8a fallback.
 * The right one is picked at class-init time, mirroring the upstream example.
 */
internal class WhisperLib {
    companion object {
        init {
            Log.d(LOG_TAG, "Primary ABI: ${Build.SUPPORTED_ABIS.joinToString()}")
            var loadV8fp16 = false
            if (isArmEabiV8a()) {
                cpuInfo()?.let {
                    Log.d(LOG_TAG, "CPU info:\n$it")
                    if (it.contains("fphp")) {
                        Log.d(LOG_TAG, "CPU supports fp16 arithmetic -> will use libwhisper_v8fp16_va.so")
                        loadV8fp16 = true
                    }
                }
            }

            if (loadV8fp16) {
                System.loadLibrary("whisper_v8fp16_va")
            } else {
                System.loadLibrary("whisper")
            }
        }

        // ---- JNI methods -----------------------------------------------------
        external fun initContext(modelPath: String): Long
        external fun initContextFromAsset(assetManager: AssetManager, assetPath: String): Long
        external fun initContextFromInputStream(inputStream: InputStream): Long
        external fun freeContext(contextPtr: Long)
        external fun fullTranscribe(contextPtr: Long, numThreads: Int, language: String?, audioData: FloatArray)
        external fun getTextSegmentCount(contextPtr: Long): Int
        external fun getTextSegment(contextPtr: Long, index: Int): String
        external fun getTextSegmentT0(contextPtr: Long, index: Int): Long
        external fun getTextSegmentT1(contextPtr: Long, index: Int): Long
        external fun getSystemInfo(): String
    }
}

private fun isArmEabiV8a(): Boolean =
    Build.SUPPORTED_ABIS.isNotEmpty() && Build.SUPPORTED_ABIS[0] == "arm64-v8a"

private fun cpuInfo(): String? = try {
    java.io.File("/proc/cpuinfo").inputStream().bufferedReader().use { it.readText() }
} catch (e: Exception) {
    Log.w(LOG_TAG, "Couldn't read /proc/cpuinfo", e)
    null
}