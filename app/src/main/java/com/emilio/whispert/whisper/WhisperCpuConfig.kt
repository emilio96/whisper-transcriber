package com.emilio.whispert.whisper

import android.util.Log
import java.io.BufferedReader
import java.io.File
import java.io.FileReader

/**
 * Decides how many threads to give whisper_full.
 *
 * Mobile SoCs are heterogeneous. Picking N threads badly here is a 3x perf trap:
 *   - On a 2+6 Oryon Snapdragon 8 Elite (no real E-cores) we want ALL 8 cores.
 *   - On a 1+3+4 big.LITTLE (4× A510 E-cores) we want only the 4 big ones, or
 *     the OS may schedule heavy matmul onto the slow cluster and lose ~3x.
 *
 * Sources we try, in order of robustness:
 *   1) /sys/devices/system/cpu/cpuN/capacity  — EAS-normalised "capacity"
 *      (0..1024) used by the kernel scheduler itself to pick cores. This is the
 *      most reliable way to separate true performance cores from efficiency
 *      cores, because unlike max clock frequency it accounts for µarch/slower
 *      IPC and is *not* fooled by an E-core clocked near 2 GHz.
 *   2) /sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq — fallback when
 *      the kernel doesn't expose capacity (some older vendor kernels).
 *   3) "CPU variant" lines in /proc/cpuinfo — legacy ARMv8 ARM bidding.
 *   4) Runtime.availableProcessors() - 1 — last-resort, naive.
 *
 * For (1) and (2) we drop any core whose value is < 70% of the max, which on:
 *   - 2+6 Oryon (capacity 1024 / ~1024): ratio high -> keep all -> 7 threads
 *   - 1+3+4 with A510 (capacity ~300 / 1024): ratio low -> drop E-cores
 *
 * Always reserve 1 core for the UI/Binder so the app keeps rendering during the
 * (long) whisper inference. Floor at 2 threads.
 */
object WhisperCpuConfig {
    /**
     * Computed once per process. The probe reads /proc/cpuinfo plus one sysfs
     * file per core, and this value is also surfaced in the settings UI: as a
     * recomputing getter it ran that disk I/O on the main thread on every
     * recomposition.
     */
    val preferredThreadCount: Int by lazy {
        CpuInfo.computeThreadCount().coerceAtLeast(2)
    }
}

private class CpuInfo(private val lines: List<String>) {

    fun computeThreadCount(): Int {
        // 1) capacity sysfs (EAS-aware, preferred)
        try {
            val coresFromCapacity = computeByCapacity()
            if (coresFromCapacity != null) return coresFromCapacity
        } catch (e: Exception) {
            Log.d(LOG_TAG, "capacity probe failed", e)
        }
        // 2) max_freq sysfs (fallback)
        try {
            return computeByMaxFreq()
        } catch (e: Exception) {
            Log.d(LOG_TAG, "max_freq probe failed", e)
        }
        // 3) /proc/cpuinfo CPU variant (legacy)
        try {
            return computeByVariant()
        } catch (e: Exception) {
            Log.d(LOG_TAG, "variant probe failed", e)
        }
        // 4) naive
        return fallback()
    }

    /**
     * Reads /sys/devices/system/cpu/cpuN/capacity for N = processor ids in
     * /proc/cpuinfo. Returns null only if no capacity file is reachable at all
     * (kernel without EAS), so the caller can fall back to max_freq.
     */
    private fun computeByCapacity(): Int? {
        val processorIds = lines
            .asSequence()
            .filter { it.startsWith("processor") }
            .map { it.substringAfter(':').trim().toInt() }
            .toList()
        if (processorIds.isEmpty()) return null

        data class Cap(val cpu: Int, val value: Int)
        val caps = mutableListOf<Cap>()
        for (cpu in processorIds) {
            val path = "/sys/devices/system/cpu/cpu${cpu}/capacity"
            val f = File(path)
            if (!f.exists()) return null
            val cap = BufferedReader(FileReader(f)).use { it.readLine() }.trim().toIntOrNull()
                ?: return null
            caps += Cap(cpu, cap)
        }
        if (caps.isEmpty()) return null

        val values = caps.map { it.value }.sorted()
        val lo = values.first()
        val hi = values.last()
        if (hi <= 0) return null
        val ratio = lo.toDouble() / hi.toDouble()
        val keepAll = ratio >= LOW_HIGH_RATIO_THRESHOLD
        val qualified = if (keepAll) values.size else values.count { it > lo }
        val binned = values.groupingBy { it }.eachCount()

        Log.d(LOG_TAG, "Capacity per CPU: ${caps.map { "cpu${it.cpu}=${it.value}" }}")
        Log.d(LOG_TAG, "Binned capacity -> count: $binned ; lo=$lo hi=$hi ratio=${
            "%.2f".format(ratio)} keepAll=$keepAll qualified=$qualified (via capacity)")

        return reserveUi(qualified)
    }

    /**
     * Reads /sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq (kHz) for each
     * processor id and applies the same ratio heuristic as capacity. Falls back
     * to throwing so the caller moves on.
     */
    private fun computeByMaxFreq(): Int {
        val processorIds = lines
            .asSequence()
            .filter { it.startsWith("processor") }
            .map { it.substringAfter(':').trim().toInt() }
            .toList()
        check(processorIds.isNotEmpty()) { "No 'processor' entries in /proc/cpuinfo" }

        val freqs = processorIds.map { getMaxCpuFrequency(it) }.sorted()
        val lo = freqs.first()
        val hi = freqs.last()
        check(hi > 0) { "All max_freq values zero" }
        val ratio = lo.toDouble() / hi.toDouble()
        val keepAll = ratio >= LOW_HIGH_RATIO_THRESHOLD
        val qualified = if (keepAll) freqs.size else freqs.count { it > lo }
        val binned = freqs.groupingBy { it }.eachCount()

        Log.d(LOG_TAG, "Max MHz per CPU (sorted): $freqs")
        Log.d(LOG_TAG, "Binned MHz -> count: $binned ; lo=$lo hi=$hi ratio=${
            "%.2f".format(ratio)} keepAll=$keepAll qualified=$qualified (via max_freq)")

        return reserveUi(qualified)
    }

    /**
     * Legacy probe: classify by the "CPU variant" hex field in /proc/cpuinfo
     * (older ARMeabi designs). Keeps only cores in the highest variant cluster.
     */
    private fun computeByVariant(): Int {
        val variants = lines
            .asSequence()
            .filter { it.startsWith("CPU variant") }
            .map { it.substringAfter("0x").trim().toInt(radix = 16) }
            .sorted()
            .toList()
        check(variants.isNotEmpty()) { "No 'CPU variant' entries in /proc/cpuinfo" }
        val kept = variants.count { it == variants.max() }
        Log.d(LOG_TAG, "Variant heuristic kept $kept cores")
        return reserveUi(kept)
    }

    fun fallback(): Int {
        val n = Runtime.getRuntime().availableProcessors()
        Log.d(LOG_TAG, "Using naive fallback: availableProcessors($n) - 1")
        return (n - 1).coerceAtLeast(2)
    }

    private fun reserveUi(qualified: Int): Int {
        val threads = (qualified - 1).coerceAtLeast(2)
        Log.d(LOG_TAG, "Computed thread count = $threads (reserved 1 for UI)")
        return threads
    }

    companion object {
        private const val LOG_TAG = "WhisperCpuConfig"
        private const val LOW_HIGH_RATIO_THRESHOLD = 0.70

        fun computeThreadCount(): Int {
            return try {
                readCpuInfo().computeThreadCount()
            } catch (e: Exception) {
                Log.d(LOG_TAG, "Couldn't read /proc/cpuinfo", e)
                CpuInfo(emptyList()).fallback()
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