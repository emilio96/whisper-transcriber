package com.emilio.whispert.whisper

import java.io.Closeable
import java.io.File
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel

/**
 * Tiny RIFF/WAVE PCM16 reader.
 *
 * Two entry points:
 *  - [decodePcm16ToMonoFloat] loads a whole file into a FloatArray. Only use it
 *    for short clips (the bundled jfk.wav smoke test): an hour of 16 kHz mono
 *    costs ~115 MB for the byte buffer plus ~230 MB for the floats, on the Java
 *    heap, which is an OOM on most devices.
 *  - [MappedPcm16] memory-maps the payload and hands out one segment at a time.
 *    That is what the diarization pass uses, so the Java heap stays flat no
 *    matter how long the recording is while random access stays O(1).
 */
object WavHelper {

    data class WavInfo(
        val sampleRate: Int,
        val channels: Int,
        val bitsPerSample: Int,
        val audioFormat: Int, // 1 = PCM
    )

    /** Byte layout of the PCM payload inside a RIFF/WAVE container. */
    private data class PcmLayout(
        val dataOffset: Long,
        val dataSize: Long,
        val channels: Int,
        val sampleRate: Int,
    )

    /**
     * Reads [stream] from its current position and returns the PCM16 payload as
     * mono Float32 samples in [-1, 1]. The stream must point at the very start
     * of a RIFF/WAVE file.
     */
    fun decodePcm16ToMonoFloat(stream: InputStream): FloatArray {
        // 12-byte RIFF header: "RIFF" <4 size> "WAVE".
        val riff = ByteArray(12)
        readFully(stream, riff, 0, 12)
        check(String(riff, 0, 4) == "RIFF") { "Missing RIFF tag - not a WAVE file" }
        check(String(riff, 8, 4) == "WAVE") { "Missing WAVE tag - not a WAVE file" }

        var info: WavInfo? = null
        val chunkId = ByteArray(4)
        val chunkSizeBuf = ByteArray(4)

        // Walk chunks: "fmt " (then optional chunks) then "data".
        while (true) {
            if (readFully(stream, chunkId, 0, 4) < 4) error("Reached EOF before 'data' chunk")
            if (readFully(stream, chunkSizeBuf, 0, 4) < 4) error("Truncated chunk size")
            val sz = readU32LE(chunkSizeBuf).toInt()

            when (String(chunkId)) {
                "fmt " -> {
                    val body = ByteArray(sz)
                    readFully(stream, body, 0, sz)
                    info = parseFmt(body)
                }
                "data" -> {
                    val fmt = info ?: error("'fmt ' chunk must come before 'data' chunk")
                    return readPayloadAsMonoFloat(stream, sz, fmt.channels)
                }
                else -> {
                    // Skip unknown chunk (LIST/INFO, fact, ...).
                    var s = sz.toLong()
                    while (s > 0) {
                        val skipped = stream.skip(s)
                        if (skipped <= 0) break
                        s -= skipped
                    }
                }
            }
            // WAVE pads odd-sized chunks to an even boundary. Exactly once per
            // chunk: an earlier revision skipped the pad byte twice for unknown
            // chunks, which desynchronised the parser for the rest of the file.
            if (sz % 2 != 0) stream.skip(1L)
        }
    }

    private fun parseFmt(body: ByteArray): WavInfo {
        val audioFormat = readU16LE(body, 0)
        val channels = readU16LE(body, 2)
        val sampleRate = readU32LE(body, 4).toInt()
        val bitsPerSample = readU16LE(body, 14)
        check(audioFormat == 1) { "Only PCM (format 1) supported, got $audioFormat" }
        check(bitsPerSample == 16) { "Only 16-bit PCM supported, got $bitsPerSample" }
        check(sampleRate == 16_000) {
            "Whisper needs 16 kHz; this file is $sampleRate Hz. Use the MediaExtractor path."
        }
        check(channels >= 1) { "Invalid channel count $channels" }
        return WavInfo(sampleRate, channels, bitsPerSample, audioFormat)
    }

    private fun readPayloadAsMonoFloat(stream: InputStream, byteSize: Int, channels: Int): FloatArray {
        val len = byteSize and 0x7FFFFFFF
        val raw = ByteArray(len)
        var off = 0
        while (off < len) {
            val n = stream.read(raw, off, len - off)
            if (n < 0) break
            off += n
        }

        val sb: ShortBuffer = ByteBuffer.wrap(raw, 0, off)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asShortBuffer()

        val totalFrames = off / (2 * channels)
        val out = FloatArray(totalFrames)
        var si = 0
        for (i in 0 until totalFrames) {
            var sum = 0
            for (c in 0 until channels) sum += sb.get(si++)
            out[i] = (sum / channels) / 32_768.0f
        }
        return out
    }

    /** Locates the PCM payload of [file] without reading it. */
    private fun readLayout(file: File): PcmLayout {
        RandomAccessFile(file, "r").use { raf ->
            val header = ByteArray(12)
            raf.readFully(header)
            check(String(header, 0, 4) == "RIFF") { "Missing RIFF tag - not a WAVE file" }
            check(String(header, 8, 4) == "WAVE") { "Missing WAVE tag - not a WAVE file" }

            var info: WavInfo? = null
            val chunkId = ByteArray(4)
            val chunkSizeBuf = ByteArray(4)
            val fileLength = raf.length()

            while (raf.filePointer + 8 <= fileLength) {
                raf.readFully(chunkId)
                raf.readFully(chunkSizeBuf)
                val sz = readU32LE(chunkSizeBuf)
                val bodyStart = raf.filePointer

                when (String(chunkId)) {
                    "fmt " -> {
                        val body = ByteArray(minOf(sz, 64L).toInt())
                        raf.readFully(body)
                        info = parseFmt(body)
                    }
                    "data" -> {
                        val fmt = info ?: error("'fmt ' chunk must come before 'data' chunk")
                        // Trust the file, not the header: a truncated write (disk
                        // full) leaves a data size larger than what is on disk.
                        val available = fileLength - bodyStart
                        val size = if (sz in 1..available) sz else available
                        check(size > 0) { "Empty PCM payload in ${file.name}" }
                        return PcmLayout(bodyStart, size, fmt.channels, fmt.sampleRate)
                    }
                }
                raf.seek(bodyStart + sz + (sz and 1L))
            }
            error("Reached EOF before 'data' chunk in ${file.name}")
        }
    }

    /**
     * Memory-mapped view over the PCM payload of a mono 16 kHz WAV file.
     *
     * The mapping is virtual memory backed by the page cache, not Java heap, so
     * a three-hour recording costs a few kB of Java objects instead of hundreds
     * of megabytes. Callers pull one segment at a time into a reusable buffer.
     */
    class MappedPcm16 internal constructor(
        private val channel: FileChannel,
        private val shorts: ShortBuffer,
        val channels: Int,
        val sampleRate: Int,
    ) : Closeable {

        /** Number of mono frames available. */
        val frameCount: Int = shorts.limit() / channels

        /**
         * Copies frames [startFrame, endFrame) into [out] as mono floats in
         * [-1, 1], downmixing if needed. Reads at most [out].size frames and
         * returns how many were actually written.
         */
        fun readInto(startFrame: Int, endFrame: Int, out: FloatArray): Int {
            val from = startFrame.coerceIn(0, frameCount)
            val to = endFrame.coerceIn(from, frameCount)
            val count = minOf(to - from, out.size)
            if (count <= 0) return 0

            if (channels == 1) {
                var idx = from
                for (i in 0 until count) {
                    out[i] = shorts.get(idx++) / 32_768.0f
                }
            } else {
                var idx = from * channels
                for (i in 0 until count) {
                    var sum = 0
                    for (c in 0 until channels) sum += shorts.get(idx++)
                    out[i] = (sum / channels) / 32_768.0f
                }
            }
            return count
        }

        override fun close() {
            runCatching { channel.close() }
        }
    }

    /**
     * Memory-maps the PCM payload of [file]. The caller owns the returned handle
     * and must close it.
     */
    fun mapPcm16(file: File): MappedPcm16 {
        val layout = readLayout(file)
        val channel = RandomAccessFile(file, "r").channel
        try {
            val mapped = channel.map(FileChannel.MapMode.READ_ONLY, layout.dataOffset, layout.dataSize)
            mapped.order(ByteOrder.LITTLE_ENDIAN)
            return MappedPcm16(channel, mapped.asShortBuffer(), layout.channels, layout.sampleRate)
        } catch (t: Throwable) {
            runCatching { channel.close() }
            throw t
        }
    }

    // -- helpers --------------------------------------------------------------
    private fun readU16LE(b: ByteArray, off: Int): Int =
        (b[off].toInt() and 0xFF) or ((b[off + 1].toInt() and 0xFF) shl 8)

    private fun readU32LE(b: ByteArray, off: Int = 0): Long =
        ((b[off].toLong() and 0xFF)) or
        ((b[off + 1].toLong() and 0xFF) shl 8) or
        ((b[off + 2].toLong() and 0xFF) shl 16) or
        ((b[off + 3].toLong() and 0xFF) shl 24)

    private fun readFully(s: InputStream, b: ByteArray, off: Int, len: Int): Int {
        var got = 0
        while (got < len) {
            val n = s.read(b, off + got, len - got)
            if (n < 0) return got
            got += n
        }
        return got
    }
}
