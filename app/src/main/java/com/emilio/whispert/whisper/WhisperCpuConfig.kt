package com.emilio.whispert.whisper

import android.util.Log
import java.io.BufferedReader
import java.io.FileReader

/**
 * Decides how many threads to give whisper_full.
 *
 * Mobile SoCs are heterogeneous: classical big.LITTLE (1+3+4) has a slow
 * efficiency cluster that we want to skip, whereas newer flagship designs like
 * the Snapdragon 8 Elite ("2+6 Oryon": 2 prime + 6 performance, no real E-core)
 * have two fast clusters where both are worth using. Picking N threads badly
 * here is a 3x performance trap (we'd otherwise run on only the 2 prime cores).
 *
 * Heuristic:
 *   1) Read max clock frequency of every CPU.
 *   2) Compute ratio = lowestFreq / highestFreq.
 *      - ratio >= 0.70  -> both clusters are fast (e.g., 2+6 Oryon) -> use all
 *      - ratio <  0.70  -> lowest cluster is a genuine efficiency tier -> skip it
 *   3) Reserve 1 core for the UI/Binder/RenderThread so the app stays responsive
 *      while inference runs, but never go below 2 threads.
 *   4) If anything goes wrong reading sysfs, fall back to availableProcessors-1.
 */
object WhisperCpuConfig {
    val preferredThreadCount: Int
        get() = CpuInfo.computeThreadCount().coerceAtLeast(2)
}

private class CpuInfo(private val lines: List<String>) {

    fun computeThreadCount(): Int = try {
        computeByFrequencies()
    } catch (e: Exception) {
        Log.d(LOG_TAG, "Frequency probe failed, falling back to variant heuristic", e)
        runCatching { computeByVariant() }.getOrElse { fallback() }
    }

    /**
     * Reads cpuinfo_max_freq of every listed "processor" and applies the ratio
     * heuristic described above.
     */
    private fun computeByFrequencies(): Int {
        val freqs = lines
            .asSequence()
            .filter { it.startsWith("processor") }
            .map { it.substringAfter(':').trim().toInt() }
            .map { getMaxCpuFrequency(it) }
            .sorted()
            .toList()

        check(freqs.isNotEmpty()) { "No CPU cores found in /proc/cpuinfo" }

        val lo = freqs.first()
        val hi = freqs.last()
        val ratio = lo.toDouble() / hi.toDouble()
        val keepAll = ratio >= LOW_HIGH_RATIO_THRESHOLD

        val qualified = if (keepAll) freqs.size else freqs.count { it > lo }
        val binned = freqs.groupingBy { it }.eachCount()

        Log.d(LOG_TAG, "CPU max MHz (sorted): $freqs")
        Log.d(LOG_TAG, "Binned MHz -> count: $binned ; lo=$lo hi=$hi ratio=${
            "%.2f".format(ratio)} keepAll=$keepAll qualified=$qualified")

        // Reserve 1 core for the UI/binder/render threads.
        val threads = (qualified - 1).coerceAtLeast(2)
        Log.d(LOG_TAG, "Computed thread count = $threads (reserved 1 for UI)")
        return threads
    }

    /**
     * Fallback when sysfs freq files aren't readable: use the "CPU variant"
     * hex field from /proc/cpuinfo and keep only the highest-variant cluster.
     * Works on older ARMeabi designs where variant encodes the core tier.
     */
    private fun computeByVariant(): Int {
        val variants = lines
            .asSequence()
            .filter { it.startsWith("CPU variant") }
            .map { it.substringAfter("0x").trim().toInt(radix = 16) }
            .sorted()
            .toList()
        check(variants.isNotEmpty()) { "No CPU variant entries in /proc/cpuinfo" }
        val kept = variants.count { it == variants.max() }
        Log.d(LOG_TAG, "Variant heuristic kept $kept cores")
        return (kept - 1).coerceAtLeast(2)
    }

    private fun fallback(): Int {
        val n = Runtime.getRuntime().availableProcessors()
        Log.d(LOG_TAG, "Using naive fallback: availableProcessors($n) - 1")
        return (n - 1).coerceAtLeast(2)
    }

    companion object {
        private const val LOG_TAG = "WhisperCpuConfig"
        private const val LOW_HIGH_RATIO_THRESHOLD = 0.70

        fun computeThreadCount(): Int = try {
            readCpuInfo().computeThreadCount()
        } catch (e: Exception) {
            Log.d(LOG_TAG, "Couldn't read /proc/cpuinfo", e)
            CpuInfo(emptyList()).let {
                runCatching { it.computeByVariant() }.getOrElse { it.fallback() }
            }
        }

        private fun readCpuInfo() = CpuInfo(
            BufferedReader(FileReader("/proc/cpuinfo")).useLines { it.toList() }
        )

        private fun getMaxCpuFrequency(cpuIndex: Int): Int {
            val path = "/sys/devices/system/cpu/cpu${cpuIndex}/cpufreq/cpuinfo_max_freq"
            return BufferedReader(FileReader(path)).use { it.readLine() }.toInt() / 1000 // kHz -> MHz
        }
    }
}