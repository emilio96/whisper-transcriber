package com.emilio.whispert.whisper

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.ShortBuffer

/**
 * Tiny RIFF/WAVE PCM16 decoder.
 *
 * Walks the chunked WAVE format header-to-header (skipping non-"data" chunks
 * such as the "LIST"/"INFO" block that ffmpeg writes), reads the PCM16 little-
 * endian payload, and converts to the FloatArray in [-1, 1] that whisper_full
 * expects.
 *
 * Multi-channel input is down-mixed to mono by averaging. Sample-rate must be
 * 16 kHz (the only rate whisper accepts) — an IllegalStateException is thrown
 * otherwise, so the caller can handle upstream resampling (MediaExtractor path
 * used later for mp3/m4a). This helper is intentionally minimal: it is meant to
 * read the bundled 16 kHz test sample (jfk.wav). For arbitrary audio files
 * we will route through MediaExtractor + MediaCodec in a later step.
 */
object WavHelper {

    data class WavInfo(
        val sampleRate: Int,
        val channels: Int,
        val bitsPerSample: Int,
        val audioFormat: Int, // 1 = PCM
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
        check(String(riff, 0, 4) == "RIFF") { "Missing RIFF tag — not a WAVE file" }
        check(String(riff, 8, 4) == "WAVE") { "Missing WAVE tag — not a WAVE file" }

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
                    val audioFormat = readU16LE(body, 0)
                    val channels = readU16LE(body, 2)
                    val sampleRate = readU32LE(body, 4).toInt()
                    val bitsPerSample = readU16LE(body, 14)
                    check(audioFormat == 1) { "Only PCM (format 1) supported, got $audioFormat" }
                    check(bitsPerSample == 16) { "Only 16-bit PCM supported, got $bitsPerSample" }
                    check(sampleRate == 16_000) {
                        "Whisper needs 16 kHz; this file is ${sampleRate} Hz. Use the MediaExtractor path."
                    }
                    info = WavInfo(sampleRate, channels, bitsPerSample, audioFormat)
                }
                "data" -> {
                    val fmt = info ?: error("'fmt ' chunk must come before 'data' chunk")
                    return readPayloadAsMonoFloat(stream, sz, fmt.channels)
                }
                else -> {
                    // Skip unknown chunk (LIST/INFO, fact, ...). WAVE odd-pads chunks to even.
                    var s = sz.toLong()
                    while (s > 0) s -= stream.skip(s)
                    if (sz % 2 != 0) stream.skip(1L)
                }
            }
            // Pad to even after every chunk (in particular after 'fmt ').
            if (sz % 2 != 0) stream.skip(1L)
        }
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