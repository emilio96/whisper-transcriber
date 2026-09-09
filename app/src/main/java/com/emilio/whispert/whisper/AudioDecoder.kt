package com.emilio.whispert.whisper

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Audio decoder producing 16 kHz mono PCM16 WAV files for whisper.
 *
 * Dual engine, selected by the user in Model settings:
 *   1) Native NDK engine (`NdkMediaCodec` + ARM NEON): demux, decode, downmix
 *      and resample entirely in C with 256 KB buffered I/O.
 *   2) Kotlin MediaCodec engine: the framework path, kept as a portable
 *      reference implementation and an escape hatch when a device's native
 *      codec misbehaves. Note this is a *manual* choice - a native failure
 *      surfaces as an error, it does not silently fall back.
 */
object AudioDecoder {

    private const val TAG = "AudioDecoder"
    private const val TARGET_SAMPLE_RATE = 16_000
    private const val WAV_HEADER_SIZE = 44
    private const val IO_BUFFER_SIZE = 128 * 1024 // 128 KB write buffer

    /** 32-bit RIFF size fields cap a WAV at 4 GB (~37 h of 16 kHz mono). */
    private const val MAX_WAV_DATA_BYTES = 0xFFFFFFFFL

    /** Total length of the [uri] content in bytes, when ContentResolver knows it. */
    fun fileBytes(context: Context, uri: Uri): Long? = try {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { c ->
            if (c.moveToFirst() && !c.isNull(0)) c.getLong(0) else null
        }
    } catch (e: Exception) {
        Log.w(TAG, "Couldn't query SIZE", e)
        null
    }

    data class DecodeResult(
        val engine: String,
        val decodeMs: Long,
        val sampleCount: Int,
        val durationSec: Float,
        val speedX: Float
    )

    /**
     * Decodes [uri] into [outFile] as 16 kHz mono PCM16 WAV using the requested
     * [engine]. Throws on failure; [isCancelled] is polled by both engines.
     */
    fun decodeToWavFile(
        context: Context,
        uri: Uri,
        outFile: File,
        engine: DecoderEngine = DecoderEngine.NATIVE,
        isCancelled: () -> Boolean = { false },
        onProgress: ((Float) -> Unit)? = null
    ): DecodeResult {
        val startMs = System.currentTimeMillis()

        if (engine == DecoderEngine.KOTLIN) {
            Log.i(TAG, "Running user-selected Kotlin MediaCodec decoder for $uri")
            decodeToWavFileKotlin(context, uri, outFile, isCancelled, onProgress)
            return summarise("Kotlin MediaCodec", outFile, startMs, onProgress)
        }

        // Native C++ (NDK/NEON) engine
        val pfd = runCatching {
            context.contentResolver.openFileDescriptor(uri, "r")
                ?: context.contentResolver.openAssetFileDescriptor(uri, "r")?.parcelFileDescriptor
        }.getOrNull() ?: error("Could not open FileDescriptor for uri: $uri")

        val fd = pfd.fd
        var length = if (pfd.statSize > 0) pfd.statSize else -1L
        if (length <= 0) {
            runCatching {
                context.contentResolver.openAssetFileDescriptor(uri, "r")?.use { afd ->
                    if (afd.length > 0) length = afd.length
                }
            }
        }
        Log.i(TAG, "Running user-selected Native C++ decode: fd=$fd len=$length uri=$uri")

        try {
            // The native decoder polls the shared abort flag rather than a Kotlin
            // lambda, so bridge the two: every progress tick is a chance to see
            // that the caller's coroutine has been cancelled and raise the flag.
            val decodeProgress: (Int) -> Unit = { p ->
                if (isCancelled()) WhisperLib.cancelTranscription()
                onProgress?.invoke((p / 100f).coerceIn(0f, 1f))
            }
            val res = WhisperProgress.withListeners(progress = decodeProgress, decode = decodeProgress) {
                WhisperLib.nativeDecodeAudioFd(
                    fd = fd,
                    offset = 0L,
                    length = length,
                    outWavPath = outFile.absolutePath
                )
            }
            val elapsed = System.currentTimeMillis() - startMs

            if (res == NATIVE_DECODE_CANCELLED) {
                throw kotlinx.coroutines.CancellationException("Decode cancelled by user")
            }
            if (res != 0) {
                error("Native C decoder failed with error code: $res (after ${elapsed}ms)")
            }
            if (!outFile.exists() || outFile.length() <= WAV_HEADER_SIZE) {
                error("Native C decoder completed with code 0 but output file is invalid (${outFile.length()} bytes)")
            }

            Log.i(TAG, "Native C decode SUCCEEDED in ${elapsed}ms (${outFile.length()} bytes)")
        } finally {
            runCatching { pfd.close() }
        }

        return summarise("Native C++ (NDK/NEON)", outFile, startMs, onProgress)
    }

    private const val NATIVE_DECODE_CANCELLED = -99

    private fun summarise(
        engine: String,
        outFile: File,
        startMs: Long,
        onProgress: ((Float) -> Unit)?
    ): DecodeResult {
        val totalMs = (System.currentTimeMillis() - startMs).coerceAtLeast(1L)
        val sampleCount = if (outFile.exists() && outFile.length() > WAV_HEADER_SIZE) {
            ((outFile.length() - WAV_HEADER_SIZE) / 2).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
        } else 0
        val durationSec = sampleCount / TARGET_SAMPLE_RATE.toFloat()
        val speedX = durationSec / (totalMs / 1000f)
        onProgress?.invoke(1f)
        return DecodeResult(engine, totalMs, sampleCount, durationSec, speedX)
    }

    /**
     * Kotlin streaming decoder (portable reference implementation).
     */
    fun decodeToWavFileKotlin(
        context: Context,
        uri: Uri,
        outFile: File,
        isCancelled: () -> Boolean = { false },
        onProgress: ((Float) -> Unit)? = null
    ) {
        val totalBytes: Long = fileBytes(context, uri) ?: -1L
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null

        val pfd = runCatching { context.contentResolver.openFileDescriptor(uri, "r") }.getOrNull()
        try {
            if (pfd != null) {
                extractor.setDataSource(pfd.fileDescriptor)
            } else {
                extractor.setDataSource(context, uri, null)
            }
            val trackIdx = firstAudioTrack(extractor)
                ?: error("No audio track found in $uri")
            extractor.selectTrack(trackIdx)

            val format = extractor.getTrackFormat(trackIdx)
            val mime = format.getString(MediaFormat.KEY_MIME)
                ?: error("Track has no MIME type")

            var inputSR = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var inputChannels = if (format.containsKey(MediaFormat.KEY_CHANNEL_COUNT))
                format.getInteger(MediaFormat.KEY_CHANNEL_COUNT) else 1

            Log.i(TAG, "Starting fast stream decode (Kotlin): mime=$mime sr=$inputSR ch=$inputChannels uri=$uri")

            codec = MediaCodec.createDecoderByType(mime).apply {
                configure(format, null, null, 0)
                start()
            }

            val pcmProcessor = PcmStreamProcessor(inputSR)
            var rawShorts = ShortArray(16384)
            var totalFrames = 0L

            val fos = FileOutputStream(outFile)
            val bos = BufferedOutputStream(fos, IO_BUFFER_SIZE)

            try {
                // Write placeholder 44-byte WAV header (patched at the end)
                bos.write(ByteArray(WAV_HEADER_SIZE))

                val info = MediaCodec.BufferInfo()
                var sawInputEOS = false
                var sawOutputEOS = false
                var totalBytesRead = 0L

                /** Shared drain body for both the polling and the blocking path. */
                fun consume(outIdx: Int): Boolean {
                    val outBuf = codec.getOutputBuffer(outIdx)
                    if (outBuf != null && info.size > 0) {
                        outBuf.position(info.offset)
                        outBuf.limit(info.offset + info.size)
                        outBuf.order(ByteOrder.LITTLE_ENDIAN)

                        val shortBuf = outBuf.asShortBuffer()
                        val count = shortBuf.remaining()
                        if (rawShorts.size < count) rawShorts = ShortArray(count * 2)
                        shortBuf.get(rawShorts, 0, count)

                        totalFrames += pcmProcessor.processAndWrite(
                            bos = bos,
                            rawShorts = rawShorts,
                            sampleCount = count,
                            channels = inputChannels,
                            inputSR = inputSR
                        )
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                    return (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                }

                while (!sawOutputEOS) {
                    if (isCancelled()) throw kotlinx.coroutines.CancellationException("Decode cancelled by user")

                    // 1. Feed input buffers as long as the codec accepts them
                    while (!sawInputEOS) {
                        val inIdx = codec.dequeueInputBuffer(0L)
                        if (inIdx < 0) break

                        val inBuf = codec.getInputBuffer(inIdx)!!
                        val sampleSize = extractor.readSampleData(inBuf, 0)
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            sawInputEOS = true
                            break
                        } else {
                            val pts = extractor.sampleTime
                            codec.queueInputBuffer(inIdx, 0, sampleSize, pts, 0)
                            extractor.advance()
                            totalBytesRead += sampleSize
                            if (totalBytes > 0) {
                                onProgress?.invoke((totalBytesRead.toFloat() / totalBytes).coerceIn(0f, 0.95f))
                            }
                        }
                    }

                    // 2. Drain all available output buffers in a tight loop
                    var drainedCount = 0
                    while (true) {
                        val outIdx = codec.dequeueOutputBuffer(info, 0L)
                        if (outIdx == MediaCodec.INFO_TRY_AGAIN_LATER) {
                            break
                        } else if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                            val newFormat = codec.outputFormat
                            if (newFormat.containsKey(MediaFormat.KEY_SAMPLE_RATE)) {
                                val sr = newFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                                if (sr > 0) inputSR = sr
                            }
                            if (newFormat.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) {
                                val ch = newFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                                if (ch > 0) inputChannels = ch
                            }
                            Log.d(TAG, "Format changed dynamically: sr=$inputSR ch=$inputChannels")
                            continue
                        } else if (outIdx >= 0) {
                            if (consume(outIdx)) {
                                sawOutputEOS = true
                                break
                            }
                            drainedCount++
                        }
                    }

                    // 3. Nothing moved: block briefly on the hardware DSP
                    if (drainedCount == 0 && !sawOutputEOS) {
                        val outIdx = codec.dequeueOutputBuffer(info, 1000L)
                        if (outIdx >= 0 && consume(outIdx)) {
                            sawOutputEOS = true
                        }
                    }
                }

                bos.flush()
                if (totalFrames == 0L) error("Decoder produced no PCM audio data")
            } finally {
                runCatching { bos.close() }
                runCatching { fos.close() }
            }

            // Patch standard WAV header with real frame count
            val dataSize = totalFrames * 2
            check(dataSize <= MAX_WAV_DATA_BYTES) {
                "Recording too long for a 32-bit WAV data chunk ($totalFrames frames)"
            }
            RandomAccessFile(outFile, "rw").use { raf ->
                raf.seek(0)
                raf.write(wavHeader(dataSize))
                raf.setLength(WAV_HEADER_SIZE + dataSize)
            }

            onProgress?.invoke(1f)
            Log.i(TAG, "Streaming decode completed (Kotlin): $totalFrames frames (WAV dataSize=$dataSize bytes)")
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            runCatching { extractor.release() }
            runCatching { pfd?.close() }
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

    private fun wavHeader(dataSize: Long, sampleRate: Int = TARGET_SAMPLE_RATE): ByteArray {
        val size = dataSize.toInt() // caller has already range-checked
        val bb = ByteBuffer.allocate(WAV_HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray(Charsets.US_ASCII)); bb.putInt(36 + size)
        bb.put("WAVE".toByteArray(Charsets.US_ASCII))
        bb.put("fmt ".toByteArray(Charsets.US_ASCII)); bb.putInt(16)
        bb.putShort(1); bb.putShort(1)                 // PCM, mono
        bb.putInt(sampleRate); bb.putInt(sampleRate * 2); bb.putShort(2); bb.putShort(16)
        bb.put("data".toByteArray(Charsets.US_ASCII)); bb.putInt(size)
        return bb.array()
    }

    /**
     * Streaming downmix + anti-aliased resample to 16 kHz.
     *
     * The phase accumulator, the last input sample and the low-pass filter state
     * all persist across codec buffers. An earlier revision reset the phase to 0
     * on every call, which inserted a discontinuity at each of the hundreds of
     * buffer boundaries per second, and decimated without any anti-alias filter
     * so everything above 8 kHz folded straight back into the speech band.
     */
    private class PcmStreamProcessor(initialSampleRate: Int) {
        private var monoBuffer = ShortArray(16384)
        private var resampledBuffer = ShortArray(16384)
        private var byteOutputBuffer = ByteArray(32768)

        private var configuredSR = 0
        private var ratioQ16 = 1 shl 16
        private var posQ16 = 0
        private var hist: Short = 0
        private var passthrough = true
        private var antialias = false
        private val lp = arrayOf(Biquad(), Biquad())

        init {
            configure(initialSampleRate)
        }

        private fun configure(inputSR: Int) {
            val sr = inputSR.coerceAtLeast(MIN_INPUT_SAMPLE_RATE)
            configuredSR = sr
            ratioQ16 = ((sr.toLong() shl 16) / TARGET_SAMPLE_RATE).toInt().coerceAtLeast(1)
            posQ16 = 0
            hist = 0
            passthrough = sr == TARGET_SAMPLE_RATE
            antialias = sr > TARGET_SAMPLE_RATE
            if (antialias) {
                val fc = (TARGET_SAMPLE_RATE / 2.0) * 0.9 // 7.2 kHz
                lp[0].setLowPass(fc, sr.toDouble(), 0.54119610)
                lp[1].setLowPass(fc, sr.toDouble(), 1.30656296)
            }
        }

        fun processAndWrite(
            bos: BufferedOutputStream,
            rawShorts: ShortArray,
            sampleCount: Int,
            channels: Int,
            inputSR: Int
        ): Int {
            if (inputSR != configuredSR) {
                Log.i(TAG, "Input rate changed $configuredSR -> $inputSR, reconfiguring resampler")
                configure(inputSR)
            }

            val safeChannels = if (channels >= 1) channels else 1
            val inFrames = sampleCount / safeChannels
            if (inFrames <= 0) return 0

            // 1. Downmix to mono
            if (monoBuffer.size < inFrames) monoBuffer = ShortArray(inFrames * 2)
            when (safeChannels) {
                1 -> System.arraycopy(rawShorts, 0, monoBuffer, 0, inFrames)
                2 -> for (i in 0 until inFrames) {
                    val base = i shl 1
                    val sum = rawShorts[base].toInt() + rawShorts[base + 1].toInt()
                    monoBuffer[i] = (sum shr 1).toShort()
                }
                else -> for (i in 0 until inFrames) {
                    var sum = 0
                    val base = i * safeChannels
                    for (c in 0 until safeChannels) sum += rawShorts[base + c].toInt()
                    monoBuffer[i] = (sum / safeChannels).toShort()
                }
            }

            // 2. Anti-alias before decimating
            if (antialias) {
                for (i in 0 until inFrames) {
                    var x = monoBuffer[i].toFloat()
                    x = lp[0].step(x)
                    x = lp[1].step(x)
                    monoBuffer[i] = x.roundToInt().coerceIn(-32768, 32767).toShort()
                }
            }

            // 3. Resample with carried-over phase
            val finalMono: ShortArray
            val outFrames: Int
            if (passthrough) {
                finalMono = monoBuffer
                outFrames = inFrames
                hist = monoBuffer[inFrames - 1]
                posQ16 = 0
            } else {
                val maxOut = (((inFrames + 2).toLong() shl 16) / ratioQ16).toInt() + 4
                if (resampledBuffer.size < maxOut) resampledBuffer = ShortArray(maxOut * 2)

                var out = 0
                while (true) {
                    val i0 = posQ16 shr 16 // arithmetic shift floors negatives too
                    if (i0 + 1 >= inFrames) break
                    val frac = posQ16 and 0xFFFF
                    val s0 = if (i0 < 0) hist.toInt() else monoBuffer[i0].toInt()
                    val s1 = monoBuffer[i0 + 1].toInt()
                    resampledBuffer[out++] = (s0 + (((s1 - s0) * frac) shr 16)).toShort()
                    posQ16 += ratioQ16
                }
                hist = monoBuffer[inFrames - 1]
                posQ16 -= inFrames shl 16
                finalMono = resampledBuffer
                outFrames = out
            }

            if (outFrames <= 0) return 0

            // 4. Little-endian bytes straight into the write buffer
            val bytesToWrite = outFrames shl 1
            if (byteOutputBuffer.size < bytesToWrite) byteOutputBuffer = ByteArray(bytesToWrite * 2)

            var bIdx = 0
            for (i in 0 until outFrames) {
                val sample = finalMono[i].toInt()
                byteOutputBuffer[bIdx++] = (sample and 0xFF).toByte()
                byteOutputBuffer[bIdx++] = ((sample ushr 8) and 0xFF).toByte()
            }

            bos.write(byteOutputBuffer, 0, bytesToWrite)
            return outFrames
        }

        companion object {
            private const val MIN_INPUT_SAMPLE_RATE = 4000
        }
    }

    /** RBJ low-pass biquad, transposed direct form II. */
    private class Biquad {
        private var b0 = 1f
        private var b1 = 0f
        private var b2 = 0f
        private var a1 = 0f
        private var a2 = 0f
        private var z1 = 0f
        private var z2 = 0f

        fun setLowPass(fc: Double, fs: Double, q: Double) {
            val w0 = (2.0 * Math.PI * fc / fs).coerceIn(1e-6, 3.0)
            val cw = cos(w0)
            val sw = sin(w0)
            val alpha = sw / (2.0 * q)
            val a0 = 1.0 + alpha
            b0 = (((1.0 - cw) / 2.0) / a0).toFloat()
            b1 = ((1.0 - cw) / a0).toFloat()
            b2 = b0
            a1 = ((-2.0 * cw) / a0).toFloat()
            a2 = ((1.0 - alpha) / a0).toFloat()
            z1 = 0f
            z2 = 0f
        }

        fun step(x: Float): Float {
            val y = b0 * x + z1
            z1 = b1 * x - a1 * y + z2
            z2 = b2 * x - a2 * y
            return y
        }
    }
}
