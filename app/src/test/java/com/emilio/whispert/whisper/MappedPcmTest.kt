package com.emilio.whispert.whisper

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/**
 * Covers [WavHelper.mapPcm16], which replaced loading the whole PCM payload onto
 * the Java heap, and which the parallel diarization workers now read from
 * concurrently.
 */
class MappedPcmTest {

    @get:Rule
    val tmp = TemporaryFolder()

    /** Writes a mono 16 kHz PCM16 WAV whose sample i holds the value i. */
    private fun writeWav(frames: Int, channels: Int = 1, extraChunk: Boolean = false): File {
        val dataSize = frames * channels * 2
        val bb = ByteBuffer.allocate(1024 + dataSize).order(ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray()); bb.putInt(0) // patched implicitly, unused by the reader
        bb.put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray()); bb.putInt(16)
        bb.putShort(1); bb.putShort(channels.toShort())
        bb.putInt(16000); bb.putInt(16000 * 2 * channels)
        bb.putShort((2 * channels).toShort()); bb.putShort(16)
        if (extraChunk) {
            // Odd-sized unknown chunk: exercises the pad-byte handling.
            bb.put("LIST".toByteArray()); bb.putInt(3)
            bb.put(byteArrayOf(1, 2, 3)); bb.put(0)
        }
        bb.put("data".toByteArray()); bb.putInt(dataSize)
        for (i in 0 until frames) {
            for (c in 0 until channels) bb.putShort((i + c).toShort())
        }
        val f = tmp.newFile("test_${frames}_${channels}_$extraChunk.wav")
        f.writeBytes(bb.array().copyOf(bb.position()))
        return f
    }

    @Test
    fun `maps a mono payload and reads exact windows`() {
        val f = writeWav(frames = 5000)
        WavHelper.mapPcm16(f).use { pcm ->
            assertEquals(5000, pcm.frameCount)
            val out = FloatArray(100)
            assertEquals(100, pcm.readInto(1000, 1100, out))
            for (i in 0 until 100) {
                assertEquals((1000 + i) / 32768.0f, out[i], 1e-7f)
            }
        }
    }

    @Test
    fun `skips an odd sized unknown chunk without desyncing`() {
        val f = writeWav(frames = 300, extraChunk = true)
        WavHelper.mapPcm16(f).use { pcm ->
            assertEquals(300, pcm.frameCount)
            val out = FloatArray(10)
            pcm.readInto(0, 10, out)
            for (i in 0 until 10) assertEquals(i / 32768.0f, out[i], 1e-7f)
        }
    }

    @Test
    fun `downmixes stereo to mono`() {
        // Sample i of channel c holds (i + c), so the mono average is i + 0.5 -> i.
        val f = writeWav(frames = 200, channels = 2)
        WavHelper.mapPcm16(f).use { pcm ->
            assertEquals(200, pcm.frameCount)
            val out = FloatArray(5)
            pcm.readInto(10, 15, out)
            for (i in 0 until 5) {
                val expected = ((10 + i) + (10 + i + 1)) / 2 / 32768.0f
                assertEquals(expected, out[i], 1e-7f)
            }
        }
    }

    @Test
    fun `clamps out of range windows instead of throwing`() {
        val f = writeWav(frames = 100)
        WavHelper.mapPcm16(f).use { pcm ->
            val out = FloatArray(50)
            assertEquals(10, pcm.readInto(90, 200, out))   // past the end
            assertEquals(0, pcm.readInto(500, 600, out))   // entirely past the end
            assertEquals(0, pcm.readInto(50, 40, out))     // inverted
            assertEquals(20, pcm.readInto(-30, 20, out))   // negative start
        }
    }

    @Test
    fun `output buffer size caps the read`() {
        val f = writeWav(frames = 1000)
        WavHelper.mapPcm16(f).use { pcm ->
            val out = FloatArray(16)
            assertEquals(16, pcm.readInto(0, 1000, out))
        }
    }

    /**
     * The diarization workers all read from one mapped source at the same time.
     * That is only sound because reads go through absolute-index gets and keep no
     * shared cursor; this pins that down.
     */
    @Test
    fun `concurrent readers see consistent data`() {
        val frames = 40_000
        val f = writeWav(frames = frames)
        WavHelper.mapPcm16(f).use { pcm ->
            val workers = 8
            val pool = Executors.newFixedThreadPool(workers)
            try {
                val tasks = (0 until workers).map { w ->
                    Callable {
                        val out = FloatArray(512)
                        var ok = true
                        var start = w * 512
                        while (start + 512 <= frames) {
                            val n = pcm.readInto(start, start + 512, out)
                            if (n != 512) { ok = false; break }
                            for (i in 0 until 512) {
                                // The writer stored (start+i) truncated to a short,
                                // which wraps past 32767 at this frame count.
                                val expected = (start + i).toShort() / 32768.0f
                                if (out[i] != expected) { ok = false; break }
                            }
                            if (!ok) break
                            start += workers * 512
                        }
                        ok
                    }
                }
                val results = pool.invokeAll(tasks, 30, TimeUnit.SECONDS)
                assertTrue("every worker must read consistent data", results.all { it.get() })
            } finally {
                pool.shutdownNow()
            }
        }
    }

    @Test
    fun `trusts the file over a data size that overruns it`() {
        // Simulates a decode interrupted by a full disk: the header claims more
        // PCM than the file actually holds.
        val f = writeWav(frames = 1000)
        val bytes = f.readBytes()
        f.writeBytes(bytes.copyOf(bytes.size - 800)) // drop 400 frames
        WavHelper.mapPcm16(f).use { pcm ->
            assertEquals(600, pcm.frameCount)
            val out = FloatArray(4)
            assertEquals(4, pcm.readInto(0, 4, out))
            assertArrayEquals(
                floatArrayOf(0f, 1 / 32768.0f, 2 / 32768.0f, 3 / 32768.0f),
                out, 1e-7f
            )
        }
    }
}
