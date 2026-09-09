#ifndef NATIVE_AUDIO_DECODER_H
#define NATIVE_AUDIO_DECODER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*decode_progress_callback)(float progress, void* user_data);
typedef bool (*decode_cancel_callback)(void* user_data);

/**
 * Decodes audio from an open file descriptor directly into a 16 kHz mono WAV file at out_wav_path.
 *
 * Uses NdkMediaExtractor + NdkMediaCodec in C with ARM NEON SIMD stereo downmix and Q16 resampler.
 * Returns 0 on success, negative error code on failure.
 */
int native_decode_to_wav(
    int fd,
    int64_t offset,
    int64_t length,
    const char* out_wav_path,
    decode_progress_callback on_progress,
    decode_cancel_callback is_cancelled,
    void* user_data
);

#ifdef __cplusplus
}
#endif

#endif // NATIVE_AUDIO_DECODER_H
