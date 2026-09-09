package com.emilio.whispert.models

/**
 * Static catalogue of Whisper ggml models offered by the app's downloader.
 *
 * Sizes and URLs are the official ones from ggerganov/whisper.cpp on Hugging Face:
 *   https://huggingface.co/ggerganov/whisper.cpp/tree/main
 */
object ModelRegistry {

    private const val HF_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

    val ALL: List<ModelDescriptor> = listOf(
        // --- TINY FAMILY ---
        ModelDescriptor(
            id = "tiny",
            family = ModelFamily.TINY,
            displayName = "Tiny (Multilingual)",
            sizeBytes = 43_537_433L,
            url = "$HF_BASE/ggml-tiny-q8_0.bin",
            multilingual = true,
            sha256 = null,
            description = "Smallest multilingual. Supports auto-language detection. Very low RAM usage.",
            recommendedFor = setOf(DeviceClass.LOW),
        ),
        ModelDescriptor(
            id = "tiny.en",
            family = ModelFamily.TINY,
            displayName = "Tiny (English only)",
            sizeBytes = 43_550_795L,
            url = "$HF_BASE/ggml-tiny.en-q8_0.bin",
            multilingual = false,
            sha256 = null,
            description = "Smallest and fastest. English-only. Great for short voice notes on any device.",
            recommendedFor = setOf(),
        ),

        // --- BASE FAMILY ---
        ModelDescriptor(
            id = "base",
            family = ModelFamily.BASE,
            displayName = "Base (Multilingual)",
            sizeBytes = 81_768_585L,
            url = "$HF_BASE/ggml-base-q8_0.bin",
            multilingual = true,
            sha256 = null,
            description = "Decent multilingual quality, typically faster than realtime on mid devices.",
            recommendedFor = setOf(DeviceClass.MID),
        ),
        ModelDescriptor(
            id = "base.en",
            family = ModelFamily.BASE,
            displayName = "Base (English only)",
            sizeBytes = 81_781_811L,
            url = "$HF_BASE/ggml-base.en-q8_0.bin",
            multilingual = false,
            sha256 = null,
            description = "Decent English quality for well under realtime on mid-tier phones.",
            recommendedFor = setOf(),
        ),

        // --- SMALL FAMILY ---
        ModelDescriptor(
            id = "small",
            family = ModelFamily.SMALL,
            displayName = "Small (Multilingual)",
            sizeBytes = 264_464_607L,
            url = "$HF_BASE/ggml-small-q8_0.bin",
            multilingual = true,
            sha256 = null,
            description = "Good multilingual quality. Default for modern flagships; usable on mid devices.",
            recommendedFor = setOf(DeviceClass.HIGH, DeviceClass.FLAGSHIP),
        ),
        ModelDescriptor(
            id = "small.en",
            family = ModelFamily.SMALL,
            displayName = "Small (English only)",
            sizeBytes = 264_477_561L,
            url = "$HF_BASE/ggml-small.en-q8_0.bin",
            multilingual = false,
            sha256 = null,
            description = "High English accuracy. Optimal for meetings and podcast audio.",
            recommendedFor = setOf(),
        ),

        // --- MEDIUM FAMILY ---
        ModelDescriptor(
            id = "medium",
            family = ModelFamily.MEDIUM,
            displayName = "Medium (Multilingual)",
            sizeBytes = 815_483_430L,
            url = "$HF_BASE/ggml-medium-q8_0.bin",
            multilingual = true,
            sha256 = null,
            description = "High-capacity multilingual model with native ARM SDOT int8 acceleration.",
            recommendedFor = setOf(),
        ),
        ModelDescriptor(
            id = "medium.en",
            family = ModelFamily.MEDIUM,
            displayName = "Medium (English only)",
            sizeBytes = 815_483_430L,
            url = "$HF_BASE/ggml-medium.en-q8_0.bin",
            multilingual = false,
            sha256 = null,
            description = "High-capacity English model with native ARM SDOT int8 acceleration.",
            recommendedFor = setOf(),
        ),

        // --- LARGE V3 TURBO FAMILY ---
        ModelDescriptor(
            id = "large-v3-turbo",
            family = ModelFamily.LARGE_V3_TURBO,
            displayName = "Large v3 Turbo (Multilingual)",
            sizeBytes = 858_485_222L,
            url = "$HF_BASE/ggml-large-v3-turbo-q8_0.bin",
            multilingual = true,
            sha256 = null,
            description = "State-of-the-art OpenAI model with 4-layer fast decoder and ARM SDOT int8 acceleration.",
            recommendedFor = setOf(DeviceClass.FLAGSHIP),
        ),

        // --- LARGE V3 FAMILY ---
        ModelDescriptor(
            id = "large-v3",
            family = ModelFamily.LARGE_V3,
            displayName = "Large v3 (Multilingual)",
            sizeBytes = 1_080_353_190L,
            url = "$HF_BASE/ggml-large-v3-q5_0.bin",
            multilingual = true,
            sha256 = null,
            description = "Full 32-layer OpenAI Large v3. Maximum possible precision, requires 6+ GB RAM.",
            recommendedFor = setOf(),
        ),
    )

    fun byId(id: String): ModelDescriptor =
        ALL.firstOrNull { it.id == id } ?: error("Unknown model id: $id")
}