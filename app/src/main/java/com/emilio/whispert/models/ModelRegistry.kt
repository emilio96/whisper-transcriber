package com.emilio.whispert.models

/**
 * Static catalogue of Whisper ggml models offered by the app's downloader.
 *
 * Sizes and URLs are the official ones from ggerganov/whisper.cpp on Hugging Face:
 *   https://huggingface.co/ggerganov/whisper.cpp/tree/main
 *
 * Notes:
 *   - `*.en.bin` files are English-only and should always run with language="en".
 *   - Multilingual files accept "auto" / "it" / "en" / etc.
 *   - We exclude medium/large-v3 from the default catalogue to keep MVP simple:
 *     they are 1.5-3 GB and would be painfully slow on CPU for clips longer than
 *     a couple of minutes; the viper filenames and URLs stay stable upstream if
 *     we want to add them later.
 */
object ModelRegistry {

    private const val HF_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

    val ALL: List<ModelDescriptor> = listOf(
        ModelDescriptor(
            id = "tiny.en",
            displayName = "Tiny (English only)",
            sizeBytes = 75L * 1024 * 1024,
            url = "$HF_BASE/ggml-tiny.en.bin",
            multilingual = false,
            sha256 = null,
            description = "Smallest and fastest. English-only. Great for short voice notes on any device.",
            recommendedFor = setOf(DeviceClass.LOW),
        ),
        ModelDescriptor(
            id = "tiny",
            displayName = "Tiny (multilingual)",
            sizeBytes = 75L * 1024 * 1024,
            url = "$HF_BASE/ggml-tiny.bin",
            multilingual = true,
            sha256 = null,
            description = "Smallest multilingual. Supports auto-language detection. Demanding audio mishears.",
            recommendedFor = setOf(),
        ),
        ModelDescriptor(
            id = "base.en",
            displayName = "Base (English only)",
            sizeBytes = 142L * 1024 * 1024,
            url = "$HF_BASE/ggml-base.en.bin",
            multilingual = false,
            sha256 = null,
            description = "Decent English quality for ~2.5x realtime. Good for dictations on mid-tier phones.",
            recommendedFor = setOf(),
        ),
        ModelDescriptor(
            id = "base",
            displayName = "Base (multilingual)",
            sizeBytes = 145L * 1024 * 1024,
            url = "$HF_BASE/ggml-base.bin",
            multilingual = true,
            sha256 = null,
            description = "Decent multilingual quality for ~2.5-3x realtime. Solid all-rounder on mid devices.",
            recommendedFor = setOf(DeviceClass.MID),
        ),
        ModelDescriptor(
            id = "small.en",
            displayName = "Small (English only)",
            sizeBytes = 466L * 1024 * 1024,
            url = "$HF_BASE/ggml-small.en.bin",
            multilingual = false,
            sha256 = null,
            description = "Good English quality at ~5x realtime. Worth the extra ~320 MB on flagship tiers.",
            recommendedFor = setOf(),
        ),
        ModelDescriptor(
            id = "small",
            displayName = "Small (multilingual)",
            sizeBytes = 488L * 1024 * 1024,
            url = "$HF_BASE/ggml-small.bin",
            multilingual = true,
            sha256 = null,
            description = "Good multilingual quality at ~5-6x realtime. Default for modern flagships; usable on mid.",
            recommendedFor = setOf(DeviceClass.HIGH, DeviceClass.FLAGSHIP),
        ),
    )

    fun byId(id: String): ModelDescriptor =
        ALL.firstOrNull { it.id == id } ?: error("Unknown model id: $id")

    /** Models shown by default for a given device class (recommended + everything allowed). */
    fun visibleFor(clazz: DeviceClass): List<ModelDescriptor> = ALL

    fun recommendedFor(clazz: DeviceClass): List<ModelDescriptor> =
        ALL.filter { clazz in it.recommendedFor }
}