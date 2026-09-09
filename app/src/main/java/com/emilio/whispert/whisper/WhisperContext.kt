package com.emilio.whispert.whisper

import android.util.Log
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.concurrent.Executors

private const val LOG_TAG = "WhisperContext"

/**
 * Owns a whisper cpp context handle ([ptr]) and funnels every native call onto a
 * single dedicated worker thread (whisper.cpp is not thread-safe).
 *
 * Holds native heap memory (~size of the loaded model) until it is released;
 * the owner (e.g. a ViewModel) **must** call [release] (suspending) or
 * [releaseAsync] (fire-and-forget, for onCleared) to avoid leaking native memory
 * when the screen is destroyed.
 */
class WhisperContext private constructor(private var ptr: Long) {

    private val inferenceExecutor = Executors
        .newSingleThreadExecutor { r -> Thread(r, "whisper-inference") }
    private val singleThreadDispatcher = inferenceExecutor.asCoroutineDispatcher()

    /**
     * Transcribes [audioData] (16 kHz mono float32 samples in [-1, 1]).
     * Throws [TranscriptionFailedException] when whisper_full reports an error.
     */
    suspend fun transcribe(
        audioData: FloatArray,
        language: String = "en",
        config: WhisperConfig = WhisperConfig(),
        printTimestamps: Boolean = true,
        onProgress: ((Int) -> Unit)? = null,
        onSegment: ((Segment) -> Unit)? = null,
    ): TranscriptionResult = withContext(singleThreadDispatcher) {
        check(ptr != 0L) { "WhisperContext has already been released" }
        val numThreads = resolveThreads(config)
        Log.d(LOG_TAG, "Transcribing ${audioData.size} samples with $numThreads threads (config: $config)")

        val rc = WhisperProgress.withListeners(
            progress = onProgress,
            segment = onSegment?.let { cb -> { text, t0, t1 -> cb(Segment(text, t0, t1)) } },
        ) {
            WhisperLib.fullTranscribe(
                contextPtr = ptr,
                numThreads = numThreads,
                language = language,
                initialPrompt = config.initialPrompt,
                temperature = config.temperature,
                temperatureInc = config.temperatureInc,
                entropyThold = config.entropyThold,
                logprobThold = config.logprobThold,
                noSpeechThold = config.noSpeechThold,
                noContext = config.noContext,
                singleSegment = config.singleSegment,
                translate = config.translate,
                suppressBlank = config.suppressBlank,
                suppressNonSpeechTokens = config.suppressNonSpeechTokens,
                audioData = audioData,
            )
        }
        checkTranscribeResultCode(rc)
        transcribeResult(printTimestamps)
    }

    /**
     * Streaming variant for long recordings: the audio lives in [wavPath]
     * (mono PCM16 16 kHz WAV produced by AudioDecoder.decodeToWavFile) and is
     * read entirely in native memory, keeping the Java heap flat regardless of
     * the recording length.
     */
    suspend fun transcribeFile(
        wavPath: String,
        language: String = "en",
        config: WhisperConfig = WhisperConfig(),
        printTimestamps: Boolean = true,
        onProgress: ((Int) -> Unit)? = null,
        onSegment: ((Segment) -> Unit)? = null,
    ): TranscriptionResult = withContext(singleThreadDispatcher) {
        check(ptr != 0L) { "WhisperContext has already been released" }
        val numThreads = resolveThreads(config)
        Log.d(LOG_TAG, "Transcribing file $wavPath with $numThreads threads (config: $config)")

        val rc = WhisperProgress.withListeners(
            progress = onProgress,
            segment = onSegment?.let { cb -> { text, t0, t1 -> cb(Segment(text, t0, t1)) } },
        ) {
            WhisperLib.fullTranscribeFile(
                contextPtr = ptr,
                numThreads = numThreads,
                language = language,
                initialPrompt = config.initialPrompt,
                temperature = config.temperature,
                temperatureInc = config.temperatureInc,
                entropyThold = config.entropyThold,
                logprobThold = config.logprobThold,
                noSpeechThold = config.noSpeechThold,
                noContext = config.noContext,
                singleSegment = config.singleSegment,
                translate = config.translate,
                suppressBlank = config.suppressBlank,
                suppressNonSpeechTokens = config.suppressNonSpeechTokens,
                wavPath = wavPath,
            )
        }
        checkTranscribeResultCode(rc)
        transcribeResult(printTimestamps)
    }

    private fun resolveThreads(config: WhisperConfig): Int =
        if (config.customThreads > 0) config.customThreads else WhisperCpuConfig.preferredThreadCount

    /**
     * 0 = success, 1 = aborted by the user (partial segments are still valid and
     * are kept), anything else is a hard failure that must not be reported to
     * the user as an empty transcript.
     */
    private fun checkTranscribeResultCode(rc: Int) {
        when {
            rc == 0 -> Unit
            rc == 1 -> Log.i(LOG_TAG, "Transcription aborted by user; keeping partial segments")
            else -> throw TranscriptionFailedException(rc)
        }
    }

    /** Reads all segments back from the native context into [TranscriptionResult]. */
    private fun transcribeResult(printTimestamps: Boolean): TranscriptionResult {
        val segmentCount = WhisperLib.getTextSegmentCount(ptr)
        val segments = ArrayList<Segment>(segmentCount)
        val text = buildString {
            for (i in 0 until segmentCount) {
                val seg = WhisperLib.getTextSegment(ptr, i)
                val t0 = WhisperLib.getTextSegmentT0(ptr, i)
                val t1 = WhisperLib.getTextSegmentT1(ptr, i)
                segments += Segment(seg, t0, t1)
                if (printTimestamps) {
                    append("[${toTimestamp(t0)} --> ${toTimestamp(t1)}] $seg\n")
                } else {
                    if (isNotEmpty()) append(" ")
                    append(seg.trim())
                }
            }
        }
        val timings = WhisperTimings.parse(WhisperLib.getTimings(ptr))
        return TranscriptionResult(text = text, segments = segments, timings = timings)
    }

    /**
     * Diagnostic: which ggml CPU kernels are active + the real architecture of
     * the loaded model (see whisper_jni.c getModelInfo). Runs on the inference
     * thread to stay whisper-safe.
     */
    suspend fun modelInfo(): String = withContext(singleThreadDispatcher) {
        check(ptr != 0L) { "WhisperContext has already been released" }
        WhisperLib.getModelInfo(ptr)
    }

    /**
     * Frees the native context on the inference thread and shuts the thread down.
     * Suspends until the memory is actually gone, so a caller can immediately
     * load a bigger model without transiently holding both in RAM.
     */
    suspend fun release() {
        if (inferenceExecutor.isShutdown) return
        withContext(singleThreadDispatcher) { freeNativeInline() }
        inferenceExecutor.shutdown()
    }

    /**
     * Fire-and-forget release for non-suspending teardown paths such as
     * ViewModel.onCleared(). The free still runs on the inference thread (the
     * executor keeps it alive until the task completes), so nothing blocks the
     * main thread and nothing touches whisper.cpp from the wrong thread.
     */
    fun releaseAsync() {
        if (inferenceExecutor.isShutdown) return
        runCatching { inferenceExecutor.submit { freeNativeInline() } }
        inferenceExecutor.shutdown()
    }

    private fun freeNativeInline() {
        if (ptr != 0L) {
            WhisperLib.freeContext(ptr)
            ptr = 0L
            Log.d(LOG_TAG, "Native context released")
        }
    }

    companion object {
        /** Loads a model from an absolute filesystem path (the only path this app uses). */
        fun createContextFromFile(filePath: String): WhisperContext {
            val ptr = WhisperLib.initContext(filePath)
            require(ptr != 0L) { "Couldn't create context from file: $filePath" }
            return WhisperContext(ptr)
        }
    }
}

/** whisper_full returned a hard error (not a user abort). */
class TranscriptionFailedException(val code: Int) :
    RuntimeException("whisper_full failed with code $code")

//  500 -> 00:05.000 ; t is in units of 10 ms (whisper t0/t1)
internal fun toTimestamp(t: Long): String {
    var msec = t * 10
    val hr = msec / (1000 * 60 * 60)
    msec -= hr * (1000 * 60 * 60)
    val min = msec / (1000 * 60)
    msec -= min * (1000 * 60)
    val sec = msec / 1000
    msec -= sec * 1000
    return String.format(Locale.US, "%02d:%02d:%02d.%03d", hr, min, sec, msec)
}

/** Whisper internal phase timings extracted from the native context. */
data class WhisperTimings(
    val sampleMs: Float,
    val encodeMs: Float,
    val decodeMs: Float,
    val batchdMs: Float,
    val promptMs: Float,
) {
    companion object {
        fun parse(csv: String): WhisperTimings {
            val parts = csv.split(',')
            fun f(i: Int) = parts.getOrElse(i) { "0" }.toFloatOrNull() ?: 0f
            return WhisperTimings(f(0), f(1), f(2), f(3), f(4))
        }
    }
}

data class Segment(
    val text: String,
    val t0: Long,
    val t1: Long,
    val speaker: String? = null,
)

data class TranscriptionResult(
    val text: String,
    val segments: List<Segment>,
    val timings: WhisperTimings,
)

// ---------------------------------------------------------------------------
// Speaker diarization: global agglomerative clustering over segment embeddings
// ---------------------------------------------------------------------------

/**
 * Groups [segments] by speaker using average-linkage agglomerative clustering
 * over their [embeddings] (unit-norm voiceprints, one per segment; an empty
 * array marks a segment that was too short to embed).
 *
 * ## Why this shape
 *
 * Average linkage on cosine distance has a closed form that makes the pairwise
 * distance matrix unnecessary:
 *
 *     d(A, B) = mean over i in A, j in B of (1 - x_i . x_j)
 *             = 1 - (sum_A / |A|) . (sum_B / |B|)
 *
 * so a cluster is fully described by the *sum* of its member vectors. Keeping
 * those sums instead of an n x n matrix takes the memory from O(n^2) (81 MB at
 * 4500 segments, i.e. an OOM waiting to happen on a three-hour recording) down
 * to O(n * dim) — about 3 MB — and lets us use the nearest-neighbour chain
 * algorithm, which is O(n^2) instead of the O(n^3) repeated full-matrix scan
 * this used to do.
 *
 * NN-chain yields exactly the same dendrogram as naive greedy merging because
 * average linkage is a reducible metric; the threshold is then applied by
 * cutting that dendrogram, which is equivalent to stopping the greedy loop.
 */
fun performPostDiarization(
    segments: List<Segment>,
    embeddings: List<FloatArray>,
    mergeThreshold: Float = 0.42f,
    /** Known speaker count; 0 cuts the dendrogram by distance instead. */
    expectedSpeakers: Int = 0,
): PostDiarizationResult {
    if (segments.isEmpty() || embeddings.isEmpty()) return PostDiarizationResult(segments)
    val n = minOf(segments.size, embeddings.size)
    if (n <= 0) return PostDiarizationResult(segments)

    val valid = (0 until n).filter { embeddings[it].isNotEmpty() }
    if (valid.size <= 1) {
        return PostDiarizationResult(
            segments.take(n).map { it.copy(speaker = SPEAKER_PREFIX_1) },
            DiarizationStats(clustersAtCut = 1, clustersFinal = 1),
        )
    }

    val dim = embeddings[valid[0]].size
    val m = valid.size

    // Threshold-independent measure of embedding quality: for each segment, the
    // distance to its single closest neighbour. Whatever the speaker layout is,
    // most segments have at least one other segment from the same speaker, so
    // the median of this should sit low (~0.1-0.2 for usable d-vectors). A high
    // median means the embeddings carry no speaker structure at all, and no
    // choice of merge threshold can recover one.
    val medianNearest = run {
        val nearest = FloatArray(m) { Float.MAX_VALUE }
        for (a in 0 until m) {
            val ea = embeddings[valid[a]]
            for (b in a + 1 until m) {
                val eb = embeddings[valid[b]]
                var dot = 0f
                for (k in 0 until dim) dot += ea[k] * eb[k]
                val d = (1f - dot).coerceAtLeast(0f)
                if (d < nearest[a]) nearest[a] = d
                if (d < nearest[b]) nearest[b] = d
            }
        }
        nearest.sort()
        nearest[m / 2]
    }

    // Cluster state: sum of member vectors + member count. Index space is the
    // compacted "valid" space; validRoot maps back to original segment indices.
    val sums = Array(m) { embeddings[valid[it]].copyOf(dim) }
    val sizes = IntArray(m) { 1 }
    val active = BooleanArray(m) { true }

    fun clusterDistance(a: Int, b: Int): Float {
        val sa = sums[a]
        val sb = sums[b]
        var dot = 0f
        for (k in 0 until dim) dot += sa[k] * sb[k]
        val d = 1f - dot / (sizes[a].toFloat() * sizes[b].toFloat())
        return if (d < 0f) 0f else d
    }

    /** Nearest active cluster to [a]; ties break on the lowest index for determinism. */
    fun nearestTo(a: Int): Int {
        var best = -1
        var bestDist = Float.MAX_VALUE
        for (c in 0 until m) {
            if (!active[c] || c == a) continue
            val d = clusterDistance(a, c)
            if (d < bestDist) {
                bestDist = d
                best = c
            }
        }
        return best
    }

    // --- 1. Nearest-neighbour chain: build the full dendrogram, O(m^2 * dim) ---
    data class Merge(val into: Int, val from: Int, val dist: Float)

    val merges = ArrayList<Merge>(m)
    val chain = IntArray(m + 1)
    var chainLen = 0
    var activeCount = m

    while (activeCount > 1) {
        if (chainLen == 0) {
            // Restart from the first still-active cluster. Scanning from 0 every
            // time matters: a merge writes the combined cluster back into the
            // *lower* of the two indices, so a monotonically advancing cursor
            // would skip past clusters that are still alive.
            var seed = 0
            while (seed < m && !active[seed]) seed++
            if (seed >= m) break
            chain[chainLen++] = seed
        }

        val a = chain[chainLen - 1]
        val b = nearestTo(a)
        if (b < 0) break

        if (chainLen >= 2 && b == chain[chainLen - 2]) {
            // a and b are mutual nearest neighbours: merge them.
            val d = clusterDistance(a, b)
            val sa = sums[a]
            val sb = sums[b]
            for (k in 0 until dim) sa[k] += sb[k]
            sizes[a] += sizes[b]
            active[b] = false
            activeCount--
            merges += Merge(into = a, from = b, dist = d)
            chainLen -= 2
        } else {
            chain[chainLen++] = b
        }
    }

    // --- 2. Cut the dendrogram at mergeThreshold via union-find ---
    val parent = IntArray(m) { it }
    fun find(x: Int): Int {
        var r = x
        while (parent[r] != r) {
            parent[r] = parent[parent[r]]
            r = parent[r]
        }
        return r
    }
    // With a known speaker count the dendrogram is cut by *rank* instead: apply
    // the (m - k) shortest merges and stop. Average linkage is reducible, so
    // sorting the NN-chain merges by height recovers the greedy order, and the
    // first m - k of them leave exactly k clusters. This asks far less of the
    // embeddings than an absolute threshold does — the two speakers only have to
    // be the last pair to join, not to sit on opposite sides of a fixed distance.
    if (expectedSpeakers in 1 until m) {
        merges.sortedBy { it.dist }.take(m - expectedSpeakers).forEach { mg ->
            val ra = find(mg.into)
            val rb = find(mg.from)
            if (ra != rb) parent[rb] = ra
        }
    } else {
        for (mg in merges) {
            if (mg.dist > mergeThreshold) continue
            val ra = find(mg.into)
            val rb = find(mg.from)
            if (ra != rb) parent[rb] = ra
        }
    }

    // Rebuild per-cluster mean vectors for the cut we actually chose, so the
    // minority-merge step below compares real cluster centroids. (The previous
    // implementation read stale rows of the distance matrix belonging to
    // clusters that had already been absorbed.)
    val clusterSum = HashMap<Int, FloatArray>()
    val clusterCount = HashMap<Int, Int>()
    for (i in 0 until m) {
        val root = find(i)
        val acc = clusterSum.getOrPut(root) { FloatArray(dim) }
        val src = embeddings[valid[i]]
        val lim = minOf(dim, src.size)
        for (k in 0 until lim) acc[k] += src[k]
        clusterCount[root] = (clusterCount[root] ?: 0) + 1
    }

    fun meanDistance(rootA: Int, rootB: Int): Float {
        val sa = clusterSum[rootA] ?: return Float.MAX_VALUE
        val sb = clusterSum[rootB] ?: return Float.MAX_VALUE
        val ca = (clusterCount[rootA] ?: 1).toFloat()
        val cb = (clusterCount[rootB] ?: 1).toFloat()
        var dot = 0f
        for (k in 0 until dim) dot += sa[k] * sb[k]
        val d = 1f - dot / (ca * cb)
        return if (d < 0f) 0f else d
    }

    // Closest surviving pair at the cut: the number that says whether the
    // threshold is the thing separating the speakers, or nowhere near it.
    var closestPair = -1f
    val roots = clusterCount.keys.toList()
    for (i in roots.indices) {
        for (j in i + 1 until roots.size) {
            val d = meanDistance(roots[i], roots[j])
            if (closestPair < 0f || d < closestPair) closestPair = d
        }
    }

    // --- 3. Fold minority clusters (< 15% of segments) into the closest major one ---
    // The size test alone is not enough: a speaker who simply talks little is a
    // minority cluster too, and absorbing one that sits 0.8 away from every major
    // cluster erases a real voice. Only fold what is also acoustically close.
    // Skipped entirely when the speaker count is pinned: the cut already
    // produced exactly the requested number, and folding would undershoot it.
    val minSegments = (m * 0.15f).coerceAtLeast(1f)
    val majorRoots = clusterCount.filterValues { it >= minSegments }.keys
    val reassign = HashMap<Int, Int>()
    var widestFold = -1f
    if (expectedSpeakers <= 0 && majorRoots.isNotEmpty() && majorRoots.size < clusterCount.size) {
        for ((root, _) in clusterCount) {
            if (root in majorRoots) continue
            var nearest = majorRoots.first()
            var bestDist = Float.MAX_VALUE
            for (major in majorRoots) {
                val d = meanDistance(root, major)
                if (d < bestDist) {
                    bestDist = d
                    nearest = major
                }
            }
            if (bestDist <= mergeThreshold * MINORITY_FOLD_SLACK) {
                reassign[root] = nearest
                if (bestDist > widestFold) widestFold = bestDist
            }
        }
    }
    // How much speech each embedding actually saw. A d-vector from under a
    // second of audio is mostly noise, and a transcript made of short segments
    // will over-split no matter how good the model is.
    val durationsMs = valid.map { (segments[it].t1 - segments[it].t0) * 10 }.sorted()
    val stats = DiarizationStats(
        clustersAtCut = clusterCount.size,
        clustersFinal = clusterCount.size - reassign.size,
        closestClusterDistance = closestPair,
        widestFoldDistance = widestFold,
        medianNearestDistance = medianNearest,
        medianSegmentMs = durationsMs[durationsMs.size / 2],
        shortSegmentFraction = durationsMs.count { it < 1000 }.toFloat() / m,
        expectedSpeakers = expectedSpeakers,
    )

    // --- 4. Project cluster ids back onto every segment ---
    val speakerOf = IntArray(n) { -1 }
    for (i in 0 until m) {
        val root = find(i)
        speakerOf[valid[i]] = reassign[root] ?: root
    }
    // Segments too short to embed inherit their nearest labelled neighbour in time.
    for (i in 0 until n) {
        if (speakerOf[i] >= 0) continue
        var back = i - 1
        while (back >= 0 && speakerOf[back] < 0) back--
        if (back >= 0) {
            speakerOf[i] = speakerOf[back]
        } else {
            var fwd = i + 1
            while (fwd < n && speakerOf[fwd] < 0) fwd++
            speakerOf[i] = if (fwd < n) speakerOf[fwd] else 0
        }
    }

    // --- 5. Temporal smoothing of single-segment spikes (< 1.5 s) ---
    for (k in 1 until n - 1) {
        if (speakerOf[k - 1] == speakerOf[k + 1] && speakerOf[k] != speakerOf[k - 1]) {
            val segLenMs = (segments[k].t1 - segments[k].t0) * 10
            if (segLenMs < 1500) {
                speakerOf[k] = speakerOf[k - 1]
            }
        }
    }

    // --- 6. Sequential speaker numbering in order of first appearance ---
    val labelMap = HashMap<Int, Int>()
    var nextLabel = 1
    val labelled = segments.take(n).mapIndexed { idx, seg ->
        val label = labelMap.getOrPut(speakerOf[idx]) { nextLabel++ }
        seg.copy(speaker = "Speaker $label")
    }
    return PostDiarizationResult(labelled, stats)
}

private const val SPEAKER_PREFIX_1 = "Speaker 1"

/**
 * How far a minority cluster may sit from the major cluster that absorbs it,
 * as a multiple of the merge threshold. Above this it is kept as its own
 * speaker no matter how little it talks.
 */
private const val MINORITY_FOLD_SLACK = 1.5f

/** Labelled segments plus the cluster-level numbers used to calibrate the thresholds. */
data class PostDiarizationResult(
    val segments: List<Segment>,
    val stats: DiarizationStats = DiarizationStats(),
)

/**
 * What the clustering actually saw. [closestClusterDistance] against the merge
 * threshold tells whether the speakers were separated by a wide margin or by a
 * hair, and [widestFoldDistance] exposes how much the minority fold moved.
 * Both are -1 when they do not apply (a single cluster, nothing folded).
 */
data class DiarizationStats(
    val clustersAtCut: Int = 0,
    val clustersFinal: Int = 0,
    val closestClusterDistance: Float = -1f,
    val widestFoldDistance: Float = -1f,
    /** Median distance from a segment to its closest neighbour; independent of the threshold. */
    val medianNearestDistance: Float = -1f,
    val medianSegmentMs: Long = 0L,
    val shortSegmentFraction: Float = 0f,
    /** Speaker count the cut was pinned to; 0 when the threshold decided. */
    val expectedSpeakers: Int = 0,
)

/** Formats transcript segments into a clean dialogue text grouped by speaker. */
fun formatTranscriptWithSpeakers(segments: List<Segment>): String {
    if (segments.isEmpty()) return ""
    val sb = StringBuilder()
    var currentSpeaker: String? = null
    for (seg in segments) {
        val spk = seg.speaker ?: "Speaker"
        if (spk != currentSpeaker) {
            if (sb.isNotEmpty()) sb.append("\n\n")
            sb.append("$spk: ").append(seg.text.trim())
            currentSpeaker = spk
        } else {
            sb.append(" ").append(seg.text.trim())
        }
    }
    return sb.toString()
}
