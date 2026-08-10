package com.emilio.whispert.whisper

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import java.nio.ByteOrder

/**
 * Decodes any audio file reachable via a SAF [Uri] (mp3, m4a, opus, wav, ...) down
 * to a mono 16 kHz FloatArray in [-1, 1] — the exact input whisper.cpp expects.
 *
 * Pipeline:
 *   1) MediaExtractor picks the first audio track and gives sample rate/channels.
 *   2) MediaCodec decodes the compressed stream to raw PCM16 little-endian
 *      (interleaved for multi-channel) into in-memory chunks.
 *   3) Down-mix to mono by averaging channels.
 *   4) Resample to 16 kHz with linear interpolation when the source rate
 *      differs (e.g. 44.1/48 kHz).
 *
 * The resampling is intentionally simple — Whisper has its own mel filterbank on
 * top, so we don't need broadcast-grade anti-aliasing for transcription-level
 * quality. For a 48 kHz -> 16 kHz step this is a factor-3 decimation with naive
 * interpolation; perceptually lossless for speech SR up to ~32 kHz. Higher rates
 * (e.g. 96 kHz) would warrant a real FIR lowpass — out of scope for MVP.
 */
object AudioDecoder {

    private const val TAG = "AudioDecoder"
    private const val TARGET_SAMPLE_RATE = 16_000
    private const val TIMEOUT_US = 10_000L

    data class AudioInfo(val sampleRate: Int, val channels: Int, val mime: String, val durationUs: Long)

    /** Quick metadata probe without fully decoding (for the UI / logging). */
    fun probe(context: Context, uri: Uri): AudioInfo {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(context, uri, null)
            val trackIdx = firstAudioTrack(extractor)
                ?: error("No audio track found in $uri")
            extractor.selectTrack(trackIdx)
            val fmt = extractor.getTrackFormat(trackIdx)
            val mime = fmt.getString(MediaFormat.KEY_MIME) ?: "audio/unknown"
            val sr = fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val ch = if (fmt.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT) else 1
            val dur = if (fmt.containsKey(MediaFormat.KEY_DURATION)) fmt.getLong(MediaFormat.KEY_DURATION) else -1L
            return AudioInfo(sr, ch, mime, dur)
        } finally {
            runCatching { extractor.release() }
        }
    }

    /** Total length of the [uri] content in bytes, when ContentResolver knows it. */
    fun fileBytes(context: Context, uri: Uri): Long? = try {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { c ->
            if (c.moveToFirst() && !c.isNull(0)) c.getLong(0) else null
        }
    } catch (e: Exception) {
        Log.w(TAG, "Couldn't query SIZE", e); null
    }

    /**
     * Fully decodes [uri] and returns mono 16 kHz float32. [onProgress] is called
     * with a value in [0,1] approximated from input-bytes-read vs total file size.
     * Must be invoked off the main thread.
     */
    fun decode(context: Context, uri: Uri, onProgress: ((Float) -> Unit)? = null): FloatArray {
        val totalBytes = fileBytes(context, uri)
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            val trackIdx = firstAudioTrack(extractor)
                ?: error("No audio track found in $uri")
            extractor.selectTrack(trackIdx)
            val format = extractor.getTrackFormat(trackIdx)
            val mime = format.getString(MediaFormat.KEY_MIME)
                ?: error("Track has no MIME type")
            val inputSR = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val inputChannels = if (format.containsKey(MediaFormat.KEY_CHANNEL_COUNT))
                format.getInteger(MediaFormat.KEY_CHANNEL_COUNT) else 1
            Log.i(TAG, "Decoding: mime=$mime sr=$inputSR ch=$inputChannels uri=$uri")

            codec = MediaCodec.createDecoderByType(mime).also { it.configure(format, null, null, 0); it.start() }

            val chunks = ArrayList<ShortArray>(64)
            val info = MediaCodec.BufferInfo()
            var sawInputEOS = false
            var sawOutputEOS = false
            var totalBytesRead = 0L

            while (!sawOutputEOS) {
                if (!sawInputEOS) {
                    val inIdx = codec.dequeueInputBuffer(TIMEOUT_US)
                    if (inIdx >= 0) {
                        val inBuf = codec.getInputBuffer(inIdx)!!
                        val sampleSize = extractor.readSampleData(inBuf, 0)
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            sawInputEOS = true
                        } else {
                            val pts = extractor.sampleTime ?: 0L
                            codec.queueInputBuffer(inIdx, 0, sampleSize, pts, 0)
                            extractor.advance()
                            totalBytesRead += sampleSize
                            if (totalBytes > 0) onProgress?.invoke((totalBytesRead.toFloat() / totalBytes).coerceIn(0f, 0.95f))
                        }
                    }
                }

                val outIdx = codec.dequeueOutputBuffer(info, TIMEOUT_US)
                if (outIdx >= 0) {
                    val outBuf = codec.getOutputBuffer(outIdx)
                    if (outBuf != null && info.size > 0) {
                        outBuf.order(ByteOrder.LITTLE_ENDIAN)
                        val sb = outBuf.asShortBuffer()
                        val chunk = ShortArray(sb.remaining())
                        sb.get(chunk)
                        chunks += chunk
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) sawOutputEOS = true
                }
            }

            onProgress?.invoke(0.98f)

            // Concatenate all chunks into one PCM16 array.
            val totalShorts = chunks.sumOf { it.size }
            if (totalShorts == 0) error("Decoder produced no PCM data")
            val pcm = ShortArray(totalShorts)
            var pos = 0
            for (c in chunks) { System.arraycopy(c, 0, pcm, pos, c.size); pos += c.size }

            onProgress?.invoke(1f)
            return downmixAndResample(pcm, inputSR, inputChannels)
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            runCatching { extractor.release() }
        }
    }

    private fun firstAudioTrack(extractor: MediaExtractor): Int? {
        for (i in 0 until extractor.trackCount) {
            val fmt = extractor.getTrackFormat(i)
            val m = fmt.getString(MediaFormat.KEY_MIME) ?: continue
            if (m.startsWith("audio/")) return i
        }
        return null
    }

    /** Down-mix to mono Float32 (in [-1,1]) and resample to 16 kHz. */
    private fun downmixAndResample(pcm: ShortArray, inputSR: Int, channels: Int): FloatArray {
        check(channels >= 1) { "Channel count must be >= 1, was $channels" }
        val frames = pcm.size / channels

        // Step 1: down-mix to mono Float32 normalised to [-1, 1].
        val mono = FloatArray(frames)
        var maxAbs = 0f
        for (f in 0 until frames) {
            var sum = 0
            for (c in 0 until channels) sum += pcm[f * channels + c].toInt()
            val v = (sum.toFloat() / channels) / 32768f
            mono[f] = v
            val a = if (v < 0f) -v else v
            if (a > maxAbs) maxAbs = a
        }
        Log.d(TAG, "Down-mix: frames=$frames maxAbs=${"%.3f".format(maxAbs)} target=$TARGET_SAMPLE_RATE sr=$inputSR")

        // Step 2: resample (linear) to 16 kHz when needed.
        if (inputSR == TARGET_SAMPLE_RATE) return mono

        val ratio = inputSR.toDouble() / TARGET_SAMPLE_RATE.toDouble()
        val outFrames = (frames / ratio).toInt().coerceAtLeast(1)
        val out = FloatArray(outFrames)
        for (o in 0 until outFrames) {
            val srcPos = o * ratio
            val i0 = srcPos.toInt()
            val i1 = (i0 + 1).coerceAtMost(frames - 1)
            val frac = (srcPos - i0).toFloat()
            out[o] = mono[i0] + (mono[i1] - mono[i0]) * frac
        }
        return out
    }
}