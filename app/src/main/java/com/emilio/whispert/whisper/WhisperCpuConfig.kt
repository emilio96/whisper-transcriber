package com.emilio.whispert.whisper

import android.util.Log
import java.io.BufferedReader
import java.io.FileReader

/**
 * Decides how many threads to give whisper_full.
 *
 * On big.LITTLE mobile SoCs handing whisper_full every available CPU can be a
 * trap: the OS scheduler may put the heavy matmul onto the E-cores, making the
 * transcription several times slower. Instead we count only the **high-perf**
 * cores (by reading their max clock frequencies, falling back to the "CPU
 * variant" field) and use that count.
 *
 * Xiaomi 17 (Snapdragon 8 Elite) reports 2 Phoenix-L prime + 4 Phoenix-M + 2
 * Phoenix-E cores: the prime + medium cluster is what we want.
 */
object WhisperCpuConfig {
    val preferredThreadCount: Int
        get() = CpuInfo.getHighPerfCpuCount().coerceAtLeast(2)
}

private class CpuInfo(private val lines: List<String>) {

    private fun getHighPerfCpuCount(): Int = try {
        getHighPerfCpuCountByFrequencies()
    } catch (e: Exception) {
        Log.d(LOG_TAG, "Couldn't read CPU frequencies", e)
        getHighPerfCpuCountByVariant()
    }

    private fun getHighPerfCpuCountByFrequencies(): Int =
        getCpuValues(property = "processor") { getMaxCpuFrequency(it.toInt()) }
            .also { Log.d(LOG_TAG, "Binned cpu frequencies (max MHz, count): ${it.binnedValues()}") }
            .countDroppingMin()

    private fun getHighPerfCpuCountByVariant(): Int =
        getCpuValues(property = "CPU variant") { it.substringAfter("0x").toInt(radix = 16) }
            .also { Log.d(LOG_TAG, "Binned cpu variants (variant, count): ${it.binnedValues()}") }
            .countKeepingMin()

    private fun List<Int>.binnedValues() = groupingBy { it }.eachCount()

    private fun getCpuValues(property: String, mapper: (String) -> Int) = lines
        .asSequence()
        .filter { it.startsWith(property) }
        .map { mapper(it.substringAfter(':').trim()) }
        .sorted()
        .toList()

    private fun List<Int>.countDroppingMin(): Int {
        val min = min()
        return count { it > min }
    }

    private fun List<Int>.countKeepingMin(): Int {
        val min = min()
        return count { it == min }
    }

    companion object {
        private const val LOG_TAG = "WhisperCpuConfig"

        fun getHighPerfCpuCount(): Int = try {
            readCpuInfo().getHighPerfCpuCount()
        } catch (e: Exception) {
            Log.d(LOG_TAG, "Couldn't read CPU info", e)
            // Last-resort: leave a couple of cores for the UI thread binder.
            (Runtime.getRuntime().availableProcessors() - 2).coerceAtLeast(1)
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