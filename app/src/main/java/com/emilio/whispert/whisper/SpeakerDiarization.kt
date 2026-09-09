package com.emilio.whispert.whisper

import android.util.Log
import java.io.Closeable
import java.io.File
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

private const val TAG = "SpeakerDiarization"

/** 16 kHz mono PCM that can be read one segment at a time without loading it all. */
interface PcmSource : Closeable {
    /** Total number of mono frames. */
    val frameCount: Int

    /**
     * Copies frames [startFrame, endFrame) into [out]; returns how many were
     * written. Implementations must be safe to call from several threads at
     * once (they read by absolute index and hold no cursor).
     */
    fun readInto(startFrame: Int, endFrame: Int, out: FloatArray): Int
}

/** Adapter for audio that genuinely is small enough to live in RAM (the smoke test clip). */
class InMemoryPcmSource(private val samples: FloatArray) : PcmSource {
    override val frameCount: Int get() = samples.size

    override fun readInto(startFrame: Int, endFrame: Int, out: FloatArray): Int {
        val from = startFrame.coerceIn(0, samples.size)
        val to = endFrame.coerceIn(from, samples.size)
        val count = minOf(to - from, out.size)
        if (count <= 0) return 0
        System.arraycopy(samples, from, out, 0, count)
        return count
    }

    override fun close() = Unit
}

/** Memory-mapped WAV on disk: the path used for real recordings. */
fun mappedPcmSource(wavFile: File): PcmSource {
    val mapped = WavHelper.mapPcm16(wavFile)
    return object : PcmSource {
        override val frameCount: Int get() = mapped.frameCount
        override fun readInto(startFrame: Int, endFrame: Int, out: FloatArray) =
            mapped.readInto(startFrame, endFrame, out)
        override fun close() = mapped.close()
    }
}

/**
 * Runs speaker diarization over an already-transcribed set of [segments].
 *
 * This is deliberately a post-processing pass: the ASR decode runs at full speed
 * with no embedding work competing for CPU, and clustering then sees the whole
 * recording at once, which is what makes the speaker assignment stable.
 *
 * Audio is pulled from the [PcmSource] one segment at a time into per-worker
 * buffers, so peak Java heap stays under a megabyte regardless of how long the
 * recording is.
 */
object SpeakerDiarization {

    /** Outcome of a diarization pass, including where the time went. */
    data class Outcome(
        val segments: List<Segment>,
        val engineLabel: String?,
        val speakerCount: Int,
        val extractMs: Long = 0L,
        val clusterMs: Long = 0L,
        val mergeThreshold: Float = 0f,
        val stats: DiarizationStats = DiarizationStats(),
    ) {
        val totalMs: Long get() = extractMs + clusterMs
    }

    /** Whisper t0/t1 are centiseconds; 160 samples per centisecond at 16 kHz. */
    private const val SAMPLES_PER_CENTISECOND = 160

    /** Merge thresholds, calibrated per engine (cosine distance on unit vectors). */
    private const val NEURAL_MERGE_THRESHOLD = 0.40f
    private const val DSP_MERGE_THRESHOLD = 0.64f

    /**
     * How many segments to embed concurrently, and how many intra-op threads each
     * inference gets.
     *
     * CAM++ over a ~350-frame input is a small graph: most of its ops are too
     * small for intra-op parallelism to pay for its own synchronisation, so the
     * useful axis is whole segments, not slices of one inference.
     *
     * intra-op is deliberately pinned to 1. ONNX Runtime's intra-op thread pool
     * belongs to the *session* and is shared by every concurrent `Run` on it, so
     * asking for more than one thread makes parallel workers queue against each
     * other on that one pool instead of running independently — an earlier
     * revision used workers = threads/2 with intra-op = 2 and got only ~1.2x out
     * of it for exactly that reason. At 1 the graph executes entirely on the
     * calling worker thread: no pool, no contention, N workers means N real
     * concurrent inferences.
     *
     * The DSP engine is single-threaded native code, so the same shape applies.
     */
    private fun neuralWorkers(threads: Int): Int = threads.coerceIn(1, 8)
    private const val NEURAL_INTRA_OP_THREADS = 1
    private fun dspWorkers(threads: Int): Int = threads.coerceIn(1, 8)

    suspend fun run(
        pcm: PcmSource,
        segments: List<Segment>,
        speakerModelFile: File?,
        threads: Int,
        /** 0 keeps the per-engine default; anything else overrides it. */
        mergeThresholdOverride: Float = 0f,
        /** Known speaker count; 0 lets the threshold decide how many there are. */
        expectedSpeakers: Int = 0,
        onProgress: (Float) -> Unit = {},
    ): Outcome {
        if (segments.isEmpty()) {
            return Outcome(segments, null, 0)
        }

        val useNeural = speakerModelFile != null
        val workers = if (useNeural) neuralWorkers(threads) else dspWorkers(threads)
        val intraOp = NEURAL_INTRA_OP_THREADS
        val engineLabel = if (useNeural) {
            "Neural ONNX (CAM++ fp16, $workers parallel segments)"
        } else {
            "Acoustic DSP (Native C, $workers parallel segments)"
        }

        val extractStart = System.currentTimeMillis()
        val embeddings = arrayOfNulls<FloatArray>(segments.size)
        val done = AtomicInteger(0)
        val total = segments.size
        val lastPct = AtomicInteger(-1)

        fun tick() {
            val pct = done.incrementAndGet() * 100 / total
            // Only report on a whole percentage point: this runs on every worker
            // and each call ends up writing a StateFlow.
            val prev = lastPct.get()
            if (pct != prev && lastPct.compareAndSet(prev, pct)) {
                onProgress(pct / 100f)
            }
        }

        val diarizer = if (useNeural) {
            NeuralSpeakerDiarizer(speakerModelFile!!, intraOpThreads = intraOp)
        } else null

        try {
            coroutineScope {
                (0 until workers).map { worker ->
                    async(Dispatchers.Default) {
                        val scratch = FloatArray(MAX_SAMPLES_PER_SEGMENT)
                        var i = worker
                        while (i < segments.size) {
                            ensureActive()
                            embeddings[i] = embed(pcm, segments[i], scratch, diarizer)
                            tick()
                            i += workers
                        }
                    }
                }.awaitAll()
            }
        } finally {
            diarizer?.close()
        }

        val extractMs = System.currentTimeMillis() - extractStart

        val clusterStart = System.currentTimeMillis()
        val threshold = if (mergeThresholdOverride > 0f) {
            mergeThresholdOverride
        } else if (useNeural) {
            NEURAL_MERGE_THRESHOLD
        } else {
            DSP_MERGE_THRESHOLD
        }
        val clustered = withContext(Dispatchers.Default) {
            performPostDiarization(
                segments,
                embeddings.map { it ?: FloatArray(0) },
                mergeThreshold = threshold,
                expectedSpeakers = expectedSpeakers,
            )
        }
        val clusterMs = System.currentTimeMillis() - clusterStart

        val labelled = clustered.segments
        val stats = clustered.stats
        val speakerCount = labelled.mapNotNull { it.speaker }.distinct().size
        Log.i(
            TAG,
            "Diarized $total segments into $speakerCount speakers via $engineLabel " +
                "(extract ${extractMs}ms, cluster ${clusterMs}ms); " +
                "threshold=$threshold clusters ${stats.clustersAtCut}->${stats.clustersFinal} " +
                "closestPair=${stats.closestClusterDistance} widestFold=${stats.widestFoldDistance} " +
                "medianNN=${stats.medianNearestDistance} medianSegMs=${stats.medianSegmentMs} " +
                "shortFrac=${stats.shortSegmentFraction}"
        )
        return Outcome(labelled, engineLabel, speakerCount, extractMs, clusterMs, threshold, stats)
    }

    /** Embeds one segment with whichever engine is active. */
    private fun embed(
        pcm: PcmSource,
        seg: Segment,
        scratch: FloatArray,
        diarizer: NeuralSpeakerDiarizer?,
    ): FloatArray {
        val n = fill(pcm, seg, scratch)
        if (n <= 0) return FloatArray(0)
        return if (diarizer != null) {
            diarizer.extractEmbedding(scratch, n)
        } else {
            // The native side takes t0/t1 in centiseconds; the slice we just
            // filled starts at 0, so the window is the whole slice.
            WhisperLib.extractSegmentEmbedding(
                audioData = if (n == scratch.size) scratch else scratch.copyOf(n),
                segT0 = 0L,
                segT1 = (n / SAMPLES_PER_CENTISECOND).toLong(),
            )
        }
    }

    /** Reads the audio window of [seg] into [scratch]; returns the frame count. */
    private fun fill(pcm: PcmSource, seg: Segment, scratch: FloatArray): Int {
        val start = (seg.t0 * SAMPLES_PER_CENTISECOND).coerceAtLeast(0L)
        val end = (seg.t1 * SAMPLES_PER_CENTISECOND).coerceAtLeast(start)
        if (start >= Int.MAX_VALUE) return 0
        val startFrame = start.toInt()
        val endFrame = minOf(end, start + scratch.size).toInt()
        return pcm.readInto(startFrame, endFrame, scratch)
    }
}
