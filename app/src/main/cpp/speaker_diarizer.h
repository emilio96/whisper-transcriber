#ifndef SPEAKER_DIARIZER_H
#define SPEAKER_DIARIZER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPEAKER_EMBED_DIM 43

/**
 * Extracts a 43-dimensional L2-normalized voiceprint embedding for a single
 * segment of 16 kHz mono float samples.
 *
 * Clustering is done on the Kotlin side (performPostDiarization) so that both
 * this DSP engine and the neural CAMPPlus engine share one implementation.
 */
bool extract_segment_embedding(
    const float *samples,
    int sample_count,
    int64_t seg_t0,
    int64_t seg_t1,
    float *out_embedding
);

#ifdef __cplusplus
}
#endif

#endif // SPEAKER_DIARIZER_H
