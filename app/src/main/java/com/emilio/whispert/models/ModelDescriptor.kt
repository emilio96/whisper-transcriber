package com.emilio.whispert.models

/**
 * Describes one downloadable Whisper model.
 *
 * `url` points to the ggerganov/whisper.cpp host on huggingface.co, which
 * serves the official ggml-converted `.bin` files. We stream-download these
 * once into internal storage and load them with `whisper_init_from_file_with_params`.
 *
 * `multilingual` controls whether the UI offers "auto/it" alongside "en":
 * English-only models (`*.en.bin`) MUST run with language="en" regardless of
 * the audio's true language, otherwise Whisper produces garbage transcription.
 */
data class ModelDescriptor(
    val id: String,
    val displayName: String,
    val sizeBytes: Long,
    val url: String,
    val multilingual: Boolean,
    /** Optional SHA-256 of the downloaded file; null skips the integrity check (MVP). */
    val sha256: String?,
    val description: String,
    /**
     * Set of device classes for which this model is the recommended sensible
     * pick (highlighted with a star in the picker). Any class may still download
     * it — this is purely a UI hint.
     */
    val recommendedFor: Set<DeviceClass>,
) {
    val isEnglishOnly: Boolean get() = !multilingual

    /** Filename used inside the on-device cache directory. */
    fun localFilename(): String = "ggml-$id.bin"
}