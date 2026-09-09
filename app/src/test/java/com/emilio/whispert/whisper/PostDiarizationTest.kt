package com.emilio.whispert.whisper

import kotlin.math.sqrt
import kotlin.random.Random
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Covers the agglomerative clustering rewrite (nearest-neighbour chain over
 * cluster mean vectors). The previous implementation was a repeated full scan of
 * an n x n distance matrix; these tests pin down that the new one still recovers
 * the right speakers and that it stays tractable at realistic segment counts.
 */
class PostDiarizationTest {

    private fun unit(vararg v: Float): FloatArray {
        var sumSq = 0f
        for (x in v) sumSq += x * x
        val norm = sqrt(sumSq.toDouble()).toFloat()
        return FloatArray(v.size) { v[it] / norm }
    }

    private fun segment(index: Int, lengthCs: Long = 300): Segment =
        Segment(text = "seg$index", t0 = index * lengthCs, t1 = (index + 1) * lengthCs)

    /** Random unit vector near [center], with [jitter] controlling the spread. */
    private fun near(center: FloatArray, jitter: Float, rnd: Random): FloatArray {
        val v = FloatArray(center.size) { center[it] + (rnd.nextFloat() - 0.5f) * 2f * jitter }
        var sumSq = 0f
        for (x in v) sumSq += x * x
        val norm = sqrt(sumSq.toDouble()).toFloat()
        for (i in v.indices) v[i] /= norm
        return v
    }

    @Test
    fun `two well separated speakers are recovered`() {
        val a = unit(1f, 0f, 0f)
        val b = unit(0f, 1f, 0f)
        // Alternate in long blocks so temporal smoothing has nothing to chew on.
        val embeddings = List(20) { if (it < 10) a else b }
        val segments = List(20) { segment(it) }

        val out = performPostDiarization(segments, embeddings, mergeThreshold = 0.42f).segments

        assertEquals(20, out.size)
        assertEquals(2, out.mapNotNull { it.speaker }.distinct().size)
        assertEquals("Speaker 1", out[0].speaker)
        assertEquals("Speaker 2", out[19].speaker)
    }

    @Test
    fun `a single speaker stays a single speaker`() {
        val a = unit(1f, 0f, 0f)
        val rnd = Random(7)
        val embeddings = List(30) { near(a, 0.05f, rnd) }
        val segments = List(30) { segment(it) }

        val out = performPostDiarization(segments, embeddings, mergeThreshold = 0.42f).segments

        assertEquals(1, out.mapNotNull { it.speaker }.distinct().size)
    }

    @Test
    fun `noisy clusters still converge to the two real speakers`() {
        val rnd = Random(42)
        val a = unit(1f, 0.1f, 0f, 0f)
        val b = unit(0f, 0.1f, 1f, 0f)
        val embeddings = ArrayList<FloatArray>()
        val segments = ArrayList<Segment>()
        // Six alternating turns of ten segments each.
        for (turn in 0 until 6) {
            val centre = if (turn % 2 == 0) a else b
            repeat(10) {
                embeddings += near(centre, 0.25f, rnd)
                segments += segment(segments.size)
            }
        }

        val out = performPostDiarization(segments, embeddings, mergeThreshold = 0.42f).segments

        assertEquals(2, out.mapNotNull { it.speaker }.distinct().size)
        // Every turn must be internally consistent.
        for (turn in 0 until 6) {
            val labels = (0 until 10).map { out[turn * 10 + it].speaker }.distinct()
            assertEquals("turn $turn should be one speaker", 1, labels.size)
        }
    }

    /**
     * A speaker who barely talks is a minority cluster by size, but folding it
     * away is only legitimate when it is also acoustically close to the cluster
     * absorbing it. Two orthogonal voices must stay two speakers no matter how
     * lopsided the split is.
     */
    @Test
    fun `a distant minority speaker is not folded away`() {
        val a = unit(1f, 0f, 0f)
        val b = unit(0f, 1f, 0f)
        // 20 segments of A then 2 of B: 9% of the segments, well under the 15% floor.
        val embeddings = List(22) { if (it < 20) a else b }
        val segments = List(22) { segment(it) }

        val out = performPostDiarization(segments, embeddings, mergeThreshold = 0.42f)

        assertEquals(2, out.segments.mapNotNull { it.speaker }.distinct().size)
        assertEquals(2, out.stats.clustersAtCut)
        assertEquals(2, out.stats.clustersFinal)
    }

    /** A minority cluster that really is the same voice still gets absorbed. */
    @Test
    fun `a close minority cluster is still folded`() {
        val rnd = Random(11)
        val a = unit(1f, 0f, 0f, 0f)
        // A tight main cluster plus a splinter that survives the cut at 0.42
        // (its distance is 0.5) but stays inside the fold slack of 0.63.
        val splinter = unit(0.5f, 0.866f, 0f, 0f)
        val embeddings = List(22) { if (it < 20) near(a, 0.02f, rnd) else near(splinter, 0.02f, rnd) }
        val segments = List(22) { segment(it) }

        val out = performPostDiarization(segments, embeddings, mergeThreshold = 0.42f)

        assertEquals(1, out.segments.mapNotNull { it.speaker }.distinct().size)
        assertEquals(1, out.stats.clustersFinal)
        assertTrue("expected a recorded fold distance", out.stats.widestFoldDistance > 0f)
    }

    /**
     * The point of pinning the count: with embeddings too noisy for any single
     * threshold, cutting by rank still recovers the speakers. Here the three
     * groups are close enough that a 0.42 cut collapses them all into one.
     */
    @Test
    fun `a pinned speaker count is honoured whatever the threshold would do`() {
        val rnd = Random(3)
        val centres = listOf(
            unit(1f, 0.2f, 0f),
            unit(0.9f, 0.45f, 0f),
            unit(0.8f, 0.6f, 0f),
        )
        val embeddings = List(30) { near(centres[it / 10], 0.03f, rnd) }
        val segments = List(30) { segment(it) }

        val auto = performPostDiarization(segments, embeddings, mergeThreshold = 0.42f)
        assertEquals(1, auto.segments.mapNotNull { it.speaker }.distinct().size)

        for (k in 2..3) {
            val pinned = performPostDiarization(segments, embeddings, 0.42f, expectedSpeakers = k)
            assertEquals(k, pinned.segments.mapNotNull { it.speaker }.distinct().size)
            assertEquals(k, pinned.stats.expectedSpeakers)
        }
    }

    @Test
    fun `segments without an embedding inherit their neighbour`() {
        val a = unit(1f, 0f, 0f)
        val b = unit(0f, 1f, 0f)
        val embeddings = listOf(a, FloatArray(0), a, b, FloatArray(0), b)
        val segments = List(6) { segment(it) }

        val out = performPostDiarization(segments, embeddings, mergeThreshold = 0.42f).segments

        assertEquals(6, out.size)
        assertTrue(out.all { it.speaker != null })
        assertEquals(out[0].speaker, out[1].speaker)
        assertEquals(out[3].speaker, out[4].speaker)
    }

    @Test
    fun `short spikes between two identical neighbours are smoothed away`() {
        val a = unit(1f, 0f, 0f)
        val b = unit(0f, 1f, 0f)
        // index 5 is a 0.5 s blip of speaker B surrounded by speaker A.
        val embeddings = List(11) { if (it == 5) b else a }
        val segments = List(11) { i ->
            if (i == 5) Segment("seg$i", t0 = i * 300L, t1 = i * 300L + 50) else segment(i)
        }

        val out = performPostDiarization(segments, embeddings, mergeThreshold = 0.42f).segments

        assertEquals(1, out.mapNotNull { it.speaker }.distinct().size)
    }

    @Test
    fun `empty input is passed through untouched`() {
        assertTrue(performPostDiarization(emptyList(), emptyList()).segments.isEmpty())
        val one = listOf(segment(0))
        assertEquals(1, performPostDiarization(one, listOf(unit(1f, 0f))).segments.size)
    }

    /**
     * Guards the complexity fix. 1500 segments is about an hour of speech; the
     * old O(n^3) loop needed ~10^9 float comparisons plus a 9 MB distance matrix
     * for this, which is the "instant post-processing" that was not instant.
     */
    @Test
    fun `clustering an hour of segments stays fast`() {
        val rnd = Random(1)
        val dim = 192
        val centres = List(3) {
            val v = FloatArray(dim) { rnd.nextFloat() - 0.5f }
            var s = 0f
            for (x in v) s += x * x
            val n = sqrt(s.toDouble()).toFloat()
            FloatArray(dim) { i -> v[i] / n }
        }
        val n = 1500
        val embeddings = List(n) { near(centres[(it / 50) % 3], 0.02f, rnd) }
        val segments = List(n) { segment(it) }

        val start = System.nanoTime()
        val out = performPostDiarization(segments, embeddings, mergeThreshold = 0.42f).segments
        val elapsedMs = (System.nanoTime() - start) / 1_000_000

        assertEquals(n, out.size)
        println("performPostDiarization: n=$n dim=$dim took ${elapsedMs}ms")
        assertTrue("clustering took ${elapsedMs}ms, expected well under 20s", elapsedMs < 20_000)
    }
}
