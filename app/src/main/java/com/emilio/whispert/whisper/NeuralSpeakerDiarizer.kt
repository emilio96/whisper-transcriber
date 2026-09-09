package com.emilio.whispert.whisper

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.util.Half
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

private const val TAG = "NeuralSpeakerDiarizer"
private const val SAMPLE_RATE = 16000
private const val FRAME_LEN = 400 // 25 ms
private const val FRAME_SHIFT = 160 // 10 ms
private const val FFT_SIZE = 512
private const val NUM_MEL = 80
private const val PI = Math.PI
private const val PREEMPH = 0.97f
private const val MEL_LOW_HZ = 20.0f
private const val MEL_HIGH_HZ = 8000.0f

/** Kaldi's fbank operates on int16-range samples; ours arrive normalised to [-1, 1]. */
private const val INT16_SCALE = 32768.0f

/** torchaudio's kaldi.fbank floors the mel energies at float epsilon before the log. */
private const val MEL_ENERGY_FLOOR = 1.1920929e-7f

/** 3.5 s is enough for a stable d-vector and keeps the ONNX call short. */
const val MAX_SAMPLES_PER_SEGMENT = 56000

private class MelFilter(val startBin: Int, val weights: FloatArray)

/**
 * Neural speaker embedding engine (CAM++ d-vectors) on ONNX Runtime.
 *
 * The front-end reproduces Kaldi's `fbank` because that is what the model was
 * trained on: int16-scaled samples, Hamming window, DC removal, pre-emphasis,
 * raw power spectrum, and an 80-channel mel filterbank whose triangles are
 * evaluated in the mel domain at each FFT bin's real centre frequency.
 *
 * ## Concurrency
 *
 * [extractEmbedding] is safe to call from several threads at once. `OrtSession.run`
 * is thread-safe by contract, the precomputed tables are immutable, and the DSP
 * scratch buffers are thread-local. Running N segments concurrently with a small
 * intra-op thread count beats one segment at a time with a large one: CAM++
 * layers over a 350-frame input are small enough that intra-op parallelism
 * mostly buys synchronisation overhead.
 */
class NeuralSpeakerDiarizer(
    modelFile: File,
    intraOpThreads: Int = 1,
) : AutoCloseable {

    private val ortEnv = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession

    private val inputName: String
    private val inputRank: Int
    private val inputIsFloat16: Boolean

    // Immutable, shared across threads.
    private val window = FloatArray(FRAME_LEN)
    private val melFilters: Array<MelFilter>
    private val bitRevTable = IntArray(FFT_SIZE)
    private val cosTable = FloatArray(FFT_SIZE / 2)
    private val sinTable = FloatArray(FFT_SIZE / 2)

    /** Per-thread DSP working set, so extraction can run concurrently. */
    private class Scratch {
        val real = FloatArray(FFT_SIZE)
        val imag = FloatArray(FFT_SIZE)
        val power = FloatArray(FFT_SIZE / 2 + 1)
        val frame = FloatArray(FRAME_LEN)
        val means = FloatArray(NUM_MEL)
        var fbank = FloatArray(0)
        var half = ShortArray(0)
    }

    private val scratchLocal = ThreadLocal.withInitial { Scratch() }

    init {
        val threads = intraOpThreads.coerceAtLeast(1)
        Log.i(TAG, "Initializing NeuralSpeakerDiarizer: ${modelFile.name}, intraOpThreads=$threads")
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        ortSession = ortEnv.createSession(modelFile.absolutePath, sessionOptions)

        inputName = ortSession.inputNames.firstOrNull() ?: "speech"
        val info = ortSession.inputInfo[inputName]?.info as? TensorInfo
        inputRank = info?.shape?.size ?: 3
        inputIsFloat16 = info?.type == OnnxJavaType.FLOAT16
        Log.i(TAG, "Model input '$inputName': rank=$inputRank type=${info?.type} (fp16=$inputIsFloat16)")

        // 1. Hamming window. Povey is torchaudio's default, but wespeaker passes
        // window_type='hamming' explicitly, so that is what CAM++ was trained on.
        for (i in 0 until FRAME_LEN) {
            window[i] = (0.54 - 0.46 * cos(2.0 * PI * i / (FRAME_LEN - 1))).toFloat()
        }

        // 2. FFT bit-reversal table
        for (i in 0 until FFT_SIZE) {
            var rev = 0
            var temp = i
            for (bit in 0 until 9) { // 2^9 = 512
                rev = (rev shl 1) or (temp and 1)
                temp = temp shr 1
            }
            bitRevTable[i] = rev
        }

        // 3. Twiddle factors
        for (k in 0 until FFT_SIZE / 2) {
            val angle = -2.0 * PI * k / FFT_SIZE
            cosTable[k] = cos(angle).toFloat()
            sinTable[k] = sin(angle).toFloat()
        }

        // 4. Mel filterbank with fractional triangles evaluated per FFT bin.
        val minMel = hzToMel(MEL_LOW_HZ)
        val maxMel = hzToMel(MEL_HIGH_HZ)
        val binHz = SAMPLE_RATE.toFloat() / FFT_SIZE
        val numBins = FFT_SIZE / 2 + 1

        melFilters = Array(NUM_MEL) { m ->
            val melLeft = minMel + (maxMel - minMel) * m / (NUM_MEL + 1)
            val melCenter = minMel + (maxMel - minMel) * (m + 1) / (NUM_MEL + 1)
            val melRight = minMel + (maxMel - minMel) * (m + 2) / (NUM_MEL + 1)

            var start = -1
            var end = -1
            val full = FloatArray(numBins)
            for (k in 0 until numBins) {
                val melK = hzToMel(k * binHz)
                if (melK <= melLeft || melK >= melRight) continue
                val w = if (melK <= melCenter) {
                    (melK - melLeft) / (melCenter - melLeft)
                } else {
                    (melRight - melK) / (melRight - melCenter)
                }
                if (w > 0f) {
                    full[k] = w
                    if (start < 0) start = k
                    end = k
                }
            }
            if (start < 0) {
                // Triangle narrower than one FFT bin: fall back to its centre bin.
                var k = Math.round(melToHz(melCenter) / binHz)
                if (k < 0) k = 0
                if (k >= numBins) k = numBins - 1
                full[k] = 1f
                start = k
                end = k
            }
            MelFilter(startBin = start, weights = full.copyOfRange(start, end + 1))
        }
    }

    private fun hzToMel(hz: Float): Float = (1127.0 * ln(1.0 + hz / 700.0)).toFloat()
    private fun melToHz(mel: Float): Float = (700.0 * (Math.exp(mel / 1127.0) - 1.0)).toFloat()

    /**
     * Kaldi-compatible 80-bin log-mel filterbank over [length] samples at the
     * front of [samples]. Returns the frame count and fills `s.fbank`.
     */
    private fun extractFbank(s: Scratch, samples: FloatArray, length: Int): Int {
        if (length < FRAME_LEN) return 0

        val numFrames = (length - FRAME_LEN) / FRAME_SHIFT + 1
        if (numFrames < 8) return 0 // too short for a meaningful embedding

        val needed = numFrames * NUM_MEL
        if (s.fbank.size < needed) s.fbank = FloatArray(needed)
        val fbank = s.fbank
        s.means.fill(0f)

        for (f in 0 until numFrames) {
            val offset = f * FRAME_SHIFT

            // Kaldi order: copy -> remove DC offset -> pre-emphasis -> window.
            // The samples arrive in [-1, 1]; Kaldi's fbank is fed int16-range
            // values. A pure gain cancels out in the per-channel mean removal
            // below, but it does decide whether the log floor bites: at [-1, 1]
            // scale the quiet mel bins land near 1e-10 and get clamped to a
            // constant, which is exactly where the speaker detail lives.
            var mean = 0f
            for (i in 0 until FRAME_LEN) {
                val v = samples[offset + i] * INT16_SCALE
                s.frame[i] = v
                mean += v
            }
            mean /= FRAME_LEN
            for (i in 0 until FRAME_LEN) s.frame[i] -= mean

            s.real.fill(0f)
            s.imag.fill(0f)
            // Kaldi treats x[-1] as x[0] for the first sample.
            var prev = s.frame[0]
            for (i in 0 until FRAME_LEN) {
                val cur = s.frame[i]
                val emphasised = cur - PREEMPH * prev
                prev = cur
                s.real[bitRevTable[i]] = emphasised * window[i]
            }

            // Table-driven radix-2 FFT
            var halfSize = 1
            while (halfSize < FFT_SIZE) {
                val step = halfSize shl 1
                val twiddleStep = FFT_SIZE / step
                for (k in 0 until halfSize) {
                    val twiddleIdx = k * twiddleStep
                    val wR = cosTable[twiddleIdx]
                    val wI = sinTable[twiddleIdx]
                    var i = k
                    while (i < FFT_SIZE) {
                        val j = i + halfSize
                        val tr = s.real[j] * wR - s.imag[j] * wI
                        val ti = s.real[j] * wI + s.imag[j] * wR
                        s.real[j] = s.real[i] - tr
                        s.imag[j] = s.imag[i] - ti
                        s.real[i] += tr
                        s.imag[i] += ti
                        i += step
                    }
                }
                halfSize = step
            }

            // Raw power spectrum (Kaldi does not normalise by the FFT size).
            for (i in 0..FFT_SIZE / 2) {
                s.power[i] = s.real[i] * s.real[i] + s.imag[i] * s.imag[i]
            }

            val frameOffset = f * NUM_MEL
            for (m in 0 until NUM_MEL) {
                val filter = melFilters[m]
                val weights = filter.weights
                val start = filter.startBin
                var sum = 0.0f
                for (k in weights.indices) {
                    sum += s.power[start + k] * weights[k]
                }
                val logEnergy = ln(if (sum > MEL_ENERGY_FLOOR) sum else MEL_ENERGY_FLOOR)
                fbank[frameOffset + m] = logEnergy
                s.means[m] += logEnergy
            }
        }

        // Per-channel mean normalisation across frames (CMN), as wespeaker does
        // before feeding CAM++.
        val invNumFrames = 1.0f / numFrames.toFloat()
        for (m in 0 until NUM_MEL) {
            val mean = s.means[m] * invNumFrames
            for (f in 0 until numFrames) {
                fbank[f * NUM_MEL + m] -= mean
            }
        }

        return numFrames
    }

    /**
     * Extracts a 192-dimensional L2-normalized speaker embedding from [length]
     * mono 16 kHz samples at the front of [samples].
     * Returns an empty array when the slice is too short or inference fails.
     */
    fun extractEmbedding(samples: FloatArray, length: Int): FloatArray {
        val usable = minOf(length, samples.size, MAX_SAMPLES_PER_SEGMENT)
        if (usable < FRAME_LEN) return FloatArray(0)

        val s = scratchLocal.get()!!
        val numFrames = extractFbank(s, samples, usable)
        if (numFrames == 0) return FloatArray(0)

        val count = numFrames * NUM_MEL
        val shape = if (inputRank == 2) {
            longArrayOf(numFrames.toLong(), NUM_MEL.toLong())
        } else {
            longArrayOf(1, numFrames.toLong(), NUM_MEL.toLong())
        }

        return try {
            makeInputTensor(s, count, shape).use { tensor ->
                ortSession.run(mapOf(inputName to tensor)).use { outputs ->
                    val out = outputs.firstOrNull()?.value as? OnnxTensor ?: return FloatArray(0)
                    normalise(readEmbedding(out))
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Neural embedding extraction failed: ${t.message}", t)
            FloatArray(0)
        }
    }

    /**
     * Wraps the fbank in an input tensor. Half-precision builds of this model
     * may expose fp16 graph inputs rather than keeping fp32 I/O with internal
     * casts, so both are handled rather than assumed.
     */
    // Half.toHalf returns raw fp16 bits in a short, which is exactly the payload
    // ONNX Runtime wants for a FLOAT16 tensor. Lint's HalfFloat check flags the
    // plain ShortArray because it drops the @HalfFloat annotation; storing the
    // bits verbatim is the point here.
    @Suppress("HalfFloat")
    private fun makeInputTensor(s: Scratch, count: Int, shape: LongArray): OnnxTensor {
        if (!inputIsFloat16) {
            return OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(s.fbank, 0, count), shape)
        }
        if (s.half.size < count) s.half = ShortArray(count)
        for (i in 0 until count) s.half[i] = Half.toHalf(s.fbank[i])
        return OnnxTensor.createTensor(
            ortEnv,
            ShortBuffer.wrap(s.half, 0, count),
            shape,
            OnnxJavaType.FLOAT16
        )
    }

    // Same as makeInputTensor: the shorts coming back from a FLOAT16 tensor are
    // raw fp16 bits and Half.toFloat is exactly how they are meant to be widened.
    @Suppress("HalfFloat")
    private fun readEmbedding(out: OnnxTensor): FloatArray {
        val info = out.info
        if (info.type == OnnxJavaType.FLOAT16) {
            val raw = out.shortBuffer
            val dim = raw.remaining()
            val embed = FloatArray(dim)
            for (i in 0 until dim) embed[i] = Half.toFloat(raw.get())
            return embed
        }
        val raw = out.floatBuffer
        val embed = FloatArray(raw.remaining())
        raw.get(embed)
        return embed
    }

    private fun normalise(embed: FloatArray): FloatArray {
        var sumSq = 0f
        for (x in embed) sumSq += x * x
        val norm = sqrt(sumSq.toDouble()).toFloat()
        if (norm > 1e-6f) {
            val invNorm = 1.0f / norm
            for (i in embed.indices) embed[i] *= invNorm
        }
        return embed
    }

    override fun close() {
        try {
            ortSession.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing ONNX session", e)
        }
    }
}
