package com.emilio.whispert.models

enum class ModelFamily(
    val id: String,
    val displayName: String,
    val description: String,
    val minRamGb: Int,
    val recommendedRamGb: Int,
    val supportsEnglishOnly: Boolean,
    val speedBadge: String,
    val qualityBadge: String
) {
    TINY(
        id = "tiny",
        displayName = "Tiny",
        description = "Smallest and fastest. Perfect for quick voice notes or low-spec devices.",
        minRamGb = 1,
        recommendedRamGb = 2,
        supportsEnglishOnly = true,
        speedBadge = "⚡⚡⚡⚡ Ultra Fast (~30x+)",
        qualityBadge = "Basic"
    ),
    BASE(
        id = "base",
        displayName = "Base",
        description = "Great balance of low memory footprint and reliable everyday transcription.",
        minRamGb = 2,
        recommendedRamGb = 3,
        supportsEnglishOnly = true,
        speedBadge = "⚡⚡⚡ Fast (~20x)",
        qualityBadge = "Good"
    ),
    SMALL(
        id = "small",
        displayName = "Small",
        description = "High accuracy for podcasts, meetings, and multitrack audio. The recommended standard.",
        minRamGb = 3,
        recommendedRamGb = 4,
        supportsEnglishOnly = true,
        speedBadge = "⚡⚡ Balanced (~10x)",
        qualityBadge = "Very High"
    ),
    MEDIUM(
        id = "medium",
        displayName = "Medium",
        description = "High-capacity model for difficult accents, background noise, and technical audio.",
        minRamGb = 4,
        recommendedRamGb = 6,
        supportsEnglishOnly = true,
        speedBadge = "⚡ Moderate (~4x)",
        qualityBadge = "Excellent"
    ),
    LARGE_V3_TURBO(
        id = "large-v3-turbo",
        displayName = "Large v3 Turbo",
        description = "State-of-the-art accuracy with optimized 4-layer decoder. Up to 4x faster than standard Large.",
        minRamGb = 4,
        recommendedRamGb = 6,
        supportsEnglishOnly = false,
        speedBadge = "⚡ Fast Large (~4-8x)",
        qualityBadge = "Maximum (SOTA)"
    ),
    LARGE_V3(
        id = "large-v3",
        displayName = "Large v3",
        description = "Full 32-layer OpenAI Large v3 model. Maximum possible transcription precision.",
        minRamGb = 6,
        recommendedRamGb = 8,
        supportsEnglishOnly = false,
        speedBadge = "⏳ Heavy (~1-2x)",
        qualityBadge = "Absolute Maximum"
    ),
    CUSTOM(
        id = "custom",
        displayName = "Custom / Imported",
        description = "User-imported GGML model loaded from local storage.",
        minRamGb = 1,
        recommendedRamGb = 2,
        supportsEnglishOnly = false,
        speedBadge = "Custom",
        qualityBadge = "Custom"
    )
}

/**
 * Describes one downloadable Whisper model.
 */
data class ModelDescriptor(
    val id: String,
    val family: ModelFamily,
    val displayName: String,
    val sizeBytes: Long,
    val url: String,
    val multilingual: Boolean,
    /** Optional SHA-256 of the downloaded file; null skips the integrity check. */
    val sha256: String?,
    val description: String,
    val recommendedFor: Set<DeviceClass>,
) {
    val isEnglishOnly: Boolean get() = !multilingual

    /** Filename used inside the on-device cache directory. */
    fun localFilename(): String = "ggml-$id.bin"
}