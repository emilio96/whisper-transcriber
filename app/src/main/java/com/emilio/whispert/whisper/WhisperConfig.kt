package com.emilio.whispert.whisper

enum class DecoderEngine(val label: String, val description: String) {
    NATIVE("Native C++ (NDK/NEON)", "High-performance C++ demuxer with ARM NEON SIMD stereo downmix."),
    KOTLIN("Kotlin MediaCodec", "Standard Android Java/Kotlin framework hardware decoder.")
}

enum class DiarizationEngine(val label: String, val description: String) {
    ACOUSTIC_DSP("Acoustic DSP (Built-in, 0 MB)", "Zero-memory classical acoustic pipeline (MFCC + Liftering + Octave Pitch)."),
    NEURAL_ONNX("Neural Voiceprint (CAM++, 14 MB)", "Deep neural embedding for maximum speaker separation precision.")
}

/**
 * Full tuning configuration for whisper.cpp execution.
 *
 * Can be customized from the "Model settings" screen and persisted to SharedPreferences.
 */
data class WhisperConfig(
    /** Audio decoder engine: Native C++ (NDK/NEON) vs Kotlin MediaCodec. */
    val decoderEngine: DecoderEngine = DecoderEngine.NATIVE,

    /** Speaker diarization engine: Acoustic DSP vs Neural ONNX. */
    val diarizationEngine: DiarizationEngine = DiarizationEngine.ACOUSTIC_DSP,

    /**
     * Cosine-distance cut of the speaker dendrogram; 0 keeps each engine's own
     * default. Lower splits more readily, higher merges more readily.
     */
    val speakerMergeThreshold: Float = 0f,

    /**
     * Known number of speakers; 0 lets the merge threshold decide. Pinning it
     * cuts the speaker dendrogram by rank instead of by distance, which is far
     * more robust on noisy recordings where no single threshold separates the
     * voices.
     */
    val expectedSpeakers: Int = 0,

    /** Custom thread count (0 = Auto-detect based on CPU EAS Big-Cores). */
    val customThreads: Int = 0,

    /** Base sampling temperature (default 0.0f). */
    val temperature: Float = 0.0f,

    /**
     * Temperature increment on fallback (default 0.0f).
     *
     * IMPORTANT: Setting this to 0.0f disables multi-temperature fallback retries,
     * which prevents whisper from getting stuck in slow multi-pass loops on difficult audio.
     */
    val temperatureInc: Float = 0.0f,

    /** Threshold for high entropy (hallucination detection trigger; default 2.4f, 0.0 to disable). */
    val entropyThold: Float = 2.4f,

    /** Threshold for average log probability (default -1.0f, 0.0 to disable). */
    val logprobThold: Float = -1.0f,

    /** VAD silence probability threshold (default 0.6f). */
    val noSpeechThold: Float = 0.6f,

    /**
     * When true (recommended), does NOT feed previous segment text as prompt to the next segment.
     * Prevents autoregressive repetition loops (e.g. repeating the same phrase 10+ times).
     */
    val noContext: Boolean = true,

    /** When true, transcribes only a single segment. */
    val singleSegment: Boolean = false,

    /** When true, translates to English instead of transcribing in source language. */
    val translate: Boolean = false,

    /** Suppress blank output tokens. */
    val suppressBlank: Boolean = true,

    /** Suppress non-speech tokens. */
    val suppressNonSpeechTokens: Boolean = true,

    /**
     * Initial prompt / vocabulary hint.
     * Use to provide technical jargon, abbreviations or proper nouns (e.g., "BDocs, Combi stopper, SOP, RMM, 3DSE").
     */
    val initialPrompt: String = "",
)
