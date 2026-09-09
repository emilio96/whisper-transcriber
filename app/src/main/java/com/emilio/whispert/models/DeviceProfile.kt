package com.emilio.whispert.models

import android.app.ActivityManager
import android.content.Context
import android.util.Log

/**
 * Lightweight hardware probe used to recommend an appropriate Whisper model
 * (size / quality trade-off) at first launch. Doesn't drive correctness — the
 * user can always override and download any model — it just highlights the
 * sensible default for the device class.
 *
 * Signal: total physical RAM (ActivityManager.MemoryInfo.totalMem) crossed with
 * the number of CPU cores (Runtime.availableProcessors). We deliberately avoid
 * parsing SoC marketing strings, which are brittle across vendors.
 */
enum class DeviceClass { LOW, MID, HIGH, FLAGSHIP }

data class DeviceProfile(
    val deviceClass: DeviceClass,
    val totalMemBytes: Long,
    val coreCount: Int,
) {
    /** Pretty string for the UI ("Flagship · 12 GB · 8 cores"). */
    fun describe(): String {
        val gb = totalMemBytes / (1024L * 1024L * 1024L)
        val cls = when (deviceClass) {
            DeviceClass.LOW -> "Low"
            DeviceClass.MID -> "Mid"
            DeviceClass.HIGH -> "High"
            DeviceClass.FLAGSHIP -> "Flagship"
        }
        return "$cls · ${gb} GB · ${coreCount} cores"
    }

    companion object {
        private const val TAG = "DeviceProfile"

        fun getHardwareInfoSummary(cores: Int, autoThreads: Int): String {
            val soc = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
                android.os.Build.SOC_MODEL.ifBlank { android.os.Build.HARDWARE }
            } else {
                android.os.Build.HARDWARE
            }
            val manufacturer = android.os.Build.MANUFACTURER
            val model = android.os.Build.MODEL
            val board = android.os.Build.BOARD
            return buildString {
                appendLine("• Device: $manufacturer $model")
                appendLine("• SoC / Hardware: $soc (Board: $board)")
                appendLine("• Total Physical Cores: $cores")
                appendLine("• Auto-selected Worker Threads: $autoThreads (Big-Cores)")
            }
        }

        fun probe(context: Context): DeviceProfile {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val mem = ActivityManager.MemoryInfo()
            am.getMemoryInfo(mem)
            val totalMem = mem.totalMem
            val cores = Runtime.getRuntime().availableProcessors()

            // RAM takes precedence; core count acts as a tie-breaker/coarse gate.
            val deviceClass = when {
                totalMem >= 8L * 1024 * 1024 * 1024 -> DeviceClass.FLAGSHIP
                totalMem >= 6L * 1024 * 1024 * 1024 -> DeviceClass.HIGH
                totalMem >= 4L * 1024 * 1024 * 1024 -> DeviceClass.MID
                else -> DeviceClass.LOW
            }

            Log.i(TAG, "Profiled device: $deviceClass (RAM=${totalMem / (1024*1024)} MB, cores=$cores)")
            return DeviceProfile(deviceClass, totalMem, cores)
        }

        /** A small multilingual model that's almost always safe to start with on a class. */
        fun defaultModelFor(clazz: DeviceClass): String = when (clazz) {
            DeviceClass.LOW -> "tiny"          // 75 MB multilingual
            DeviceClass.MID -> "base"          // 145 MB multilingual
            DeviceClass.HIGH -> "small"        // 488 MB multilingual, good quality
            DeviceClass.FLAGSHIP -> "small"    // Comfortable on Snapdragon 8 Elite tier; turbo reserved as opt-in
        }
    }
}