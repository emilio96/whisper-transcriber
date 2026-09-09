#include "speaker_diarizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <android/log.h>
#include <pthread.h>

#define TAG "SpeakerDiarizer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define SAMPLE_RATE 16000
#define FRAME_SIZE  400   // 25 ms
#define FRAME_STEP  160   // 10 ms
#define FFT_SIZE    512
#define NUM_MEL     26
#define NUM_MFCC    13
#define USE_MFCC    12    // c1..c12 (discard c0 energy offset)
#define EMBED_DIM   SPEAKER_EMBED_DIM // 43

// Cap the analysis window. Segment-level statistics (means/stds over frames)
// are stable well before 15 s of speech, and Whisper hands us windows of up to
// 30 s: without the cap the per-segment cost doubles for no measurable gain.
#define MAX_ANALYSIS_FRAMES 1500 // 15 s at a 10 ms hop

// F0 moves slowly compared to the 10 ms hop, so estimating it on every frame is
// wasted work: the autocorrelation is by far the most expensive part of this
// pipeline (O(lags * frame) per frame). One estimate every 40 ms, computed on a
// 2x-decimated frame, keeps the pitch statistics intact for ~16x less work.
#define PITCH_FRAME_STRIDE 4
#define PITCH_DECIMATION   2

#define PI 3.14159265358979323846

static float g_hann_window[FRAME_SIZE];
static float g_mel_weights[NUM_MEL][FFT_SIZE / 2 + 1];
static int   g_mel_start[NUM_MEL];
static int   g_mel_end[NUM_MEL];
static float g_dct_matrix[NUM_MFCC][NUM_MEL];
static float g_lifter[USE_MFCC];

// The Kotlin side embeds segments on several worker threads at once, so table
// setup has to happen exactly once and be visible to all of them before any
// reader proceeds. pthread_once gives both properties.
static pthread_once_t g_tables_once = PTHREAD_ONCE_INIT;

static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void build_tables(void) {

    // 1. Hann Window
    for (int i = 0; i < FRAME_SIZE; i++) {
        g_hann_window[i] = 0.5f * (1.0f - cosf(2.0f * (float)PI * (float)i / (float)(FRAME_SIZE - 1)));
    }

    // 2. Mel filterbank, 100 Hz - 7800 Hz.
    //
    // The triangles are evaluated in the mel domain at the *actual* centre
    // frequency of each FFT bin. Quantising the triangle vertices to integer
    // bin indices first (as an earlier revision did) collapses adjacent low
    // filters onto the same bin, which makes several of the lowest channels
    // return a constant instead of any signal.
    const float min_mel = hz_to_mel(100.0f);
    const float max_mel = hz_to_mel(7800.0f);
    const float bin_hz = (float)SAMPLE_RATE / (float)FFT_SIZE;

    for (int m = 0; m < NUM_MEL; m++) {
        const float mel_left   = min_mel + (max_mel - min_mel) * (float)m       / (float)(NUM_MEL + 1);
        const float mel_center = min_mel + (max_mel - min_mel) * (float)(m + 1) / (float)(NUM_MEL + 1);
        const float mel_right  = min_mel + (max_mel - min_mel) * (float)(m + 2) / (float)(NUM_MEL + 1);

        int start = -1, end = -1;
        for (int k = 0; k <= FFT_SIZE / 2; k++) {
            const float mel_k = hz_to_mel((float)k * bin_hz);
            float w = 0.0f;
            if (mel_k > mel_left && mel_k < mel_right) {
                w = (mel_k <= mel_center)
                    ? (mel_k - mel_left) / (mel_center - mel_left)
                    : (mel_right - mel_k) / (mel_right - mel_center);
            }
            g_mel_weights[m][k] = w;
            if (w > 0.0f) {
                if (start < 0) start = k;
                end = k;
            }
        }
        // A triangle narrower than one bin still has to see something: fall back
        // to the single bin closest to its centre.
        if (start < 0) {
            int k = (int)lrintf(mel_to_hz(mel_center) / bin_hz);
            if (k < 0) k = 0;
            if (k > FFT_SIZE / 2) k = FFT_SIZE / 2;
            g_mel_weights[m][k] = 1.0f;
            start = k;
            end = k;
        }
        g_mel_start[m] = start;
        g_mel_end[m] = end;
    }

    // 3. DCT-II Matrix
    for (int m = 0; m < NUM_MFCC; m++) {
        for (int k = 0; k < NUM_MEL; k++) {
            g_dct_matrix[m][k] = cosf((float)PI * (float)m * ((float)k + 0.5f) / (float)NUM_MEL);
        }
    }

    // 4. Cepstral Lifter for c1..c12
    for (int m = 1; m <= USE_MFCC; m++) {
        g_lifter[m - 1] = 1.0f + 11.0f * sinf((float)PI * (float)m / 22.0f);
    }

}

static inline void init_tables(void) {
    pthread_once(&g_tables_once, build_tables);
}

// Simple Radix-2 Real FFT (in-place complex representation)
static void fft_radix2(float *real, float *imag, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            float tr = real[j]; real[j] = real[i]; real[i] = tr;
            float ti = imag[j]; imag[j] = imag[i]; imag[i] = ti;
        }
        int k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * (float)PI / (float)len;
        float wlen_r = cosf(angle);
        float wlen_i = sinf(angle);
        for (int i = 0; i < n; i += len) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (int k = 0; k < (len >> 1); k++) {
                int u = i + k;
                int v = i + k + (len >> 1);
                float vr = real[v] * w_r - imag[v] * w_i;
                float vi = real[v] * w_i + imag[v] * w_r;
                real[v] = real[u] - vr;
                imag[v] = imag[u] - vi;
                real[u] += vr;
                imag[u] += vi;
                float next_w_r = w_r * wlen_r - w_i * wlen_i;
                float next_w_i = w_r * wlen_i + w_i * wlen_r;
                w_r = next_w_r;
                w_i = next_w_i;
            }
        }
    }
}

/**
 * Estimates the fundamental frequency (F0, Hz) via normalized autocorrelation.
 *
 * The frame is decimated by PITCH_DECIMATION first: the search only covers
 * 60-400 Hz, so an 8 kHz effective rate is far above what Nyquist requires here
 * and it cuts the work by 4x (half the lags over half the samples).
 */
static float estimate_pitch(const float *frame) {
    const int n = FRAME_SIZE / PITCH_DECIMATION; // 200 samples at 8 kHz
    float dec[FRAME_SIZE / PITCH_DECIMATION];

    // Decimate with a 2-tap average so we do not alias the harmonics we search for.
    for (int i = 0; i < n; i++) {
        const int s = i * PITCH_DECIMATION;
        dec[i] = 0.5f * (frame[s] + frame[s + 1]);
    }

    float r0 = 0.0f;
    for (int i = 0; i < n; i++) r0 += dec[i] * dec[i];
    if (r0 < 1e-4f) return 0.0f;

    // 60 Hz .. 400 Hz at the decimated 8 kHz rate: lag 20 .. 133.
    const int decimated_sr = SAMPLE_RATE / PITCH_DECIMATION;
    int min_lag = decimated_sr / 400;
    int max_lag = decimated_sr / 60;
    if (max_lag >= n) max_lag = n - 1;
    if (min_lag < 1) min_lag = 1;
    if (min_lag > max_lag) return 0.0f;

    float max_corr = 0.0f;
    int best_lag = 0;
    for (int lag = min_lag; lag <= max_lag; lag++) {
        float sum = 0.0f;
        for (int i = 0; i < n - lag; i++) {
            sum += dec[i] * dec[i + lag];
        }
        const float norm_corr = sum / r0;
        if (norm_corr > max_corr) {
            max_corr = norm_corr;
            best_lag = lag;
        }
    }

    if (max_corr >= 0.30f && best_lag > 0) {
        return (float)decimated_sr / (float)best_lag;
    }
    return 0.0f;
}

// Compute 12 Liftered MFCCs, Spectral Centroid, Subband Ratios and (optionally) Pitch
static void process_frame(
    const float *raw_frame,
    bool want_pitch,
    float *out_mfcc,
    float *out_centroid,
    float *out_pitch,
    float *out_subbands
) {
    float real[FFT_SIZE];
    float imag[FFT_SIZE];
    memset(real, 0, sizeof(real));
    memset(imag, 0, sizeof(imag));

    // Pre-emphasis & Hann window
    real[0] = raw_frame[0] * g_hann_window[0];
    for (int i = 1; i < FRAME_SIZE; i++) {
        float sample = raw_frame[i] - 0.97f * raw_frame[i - 1];
        real[i] = sample * g_hann_window[i];
    }

    *out_pitch = want_pitch ? estimate_pitch(raw_frame) : -1.0f;

    // FFT
    fft_radix2(real, imag, FFT_SIZE);

    // Power Spectrum & Spectral Centroid
    float power[FFT_SIZE / 2 + 1];
    float sum_power = 0.0f;
    float weighted_sum = 0.0f;

    float e_low = 0.0f;   // 100 Hz - 800 Hz (bins 3 to 25)
    float e_mid = 0.0f;   // 800 Hz - 3200 Hz (bins 26 to 102)
    float e_high = 0.0f;  // 3200 Hz - 7800 Hz (bins 103 to 250)

    for (int i = 0; i <= FFT_SIZE / 2; i++) {
        power[i] = (real[i] * real[i] + imag[i] * imag[i]) / (float)FFT_SIZE;
        sum_power += power[i];
        weighted_sum += (float)i * power[i];

        if (i >= 3 && i <= 25) e_low += power[i];
        else if (i > 25 && i <= 102) e_mid += power[i];
        else if (i > 102 && i <= 250) e_high += power[i];
    }

    if (sum_power > 1e-6f) {
        *out_centroid = (weighted_sum / sum_power) * ((float)SAMPLE_RATE / (float)FFT_SIZE);
        out_subbands[0] = e_low / sum_power;
        out_subbands[1] = e_mid / sum_power;
        out_subbands[2] = e_high / sum_power;
    } else {
        *out_centroid = 0.0f;
        out_subbands[0] = 0.33f;
        out_subbands[1] = 0.33f;
        out_subbands[2] = 0.33f;
    }

    // Mel Filterbank Energies (fractional triangular weights, precomputed)
    float mel_energy[NUM_MEL];
    for (int m = 0; m < NUM_MEL; m++) {
        const float *w = g_mel_weights[m];
        float sum = 0.0f;
        for (int k = g_mel_start[m]; k <= g_mel_end[m]; k++) {
            sum += power[k] * w[k];
        }
        mel_energy[m] = logf(sum > 1e-6f ? sum : 1e-6f);
    }

    // DCT-II: Extract c1..c12 with liftering (discard c0 energy)
    for (int m = 1; m <= USE_MFCC; m++) {
        float sum = 0.0f;
        for (int k = 0; k < NUM_MEL; k++) {
            sum += mel_energy[k] * g_dct_matrix[m][k];
        }
        out_mfcc[m - 1] = sum * g_lifter[m - 1];
    }
}

// Compute an L2-normalized 43-dimensional acoustic embedding
static bool compute_segment_embedding(
    const float *samples,
    int sample_count,
    int start_sample,
    int end_sample,
    float *out_embedding
) {
    if (!samples || sample_count <= 0 || !out_embedding) return false;
    if (start_sample < 0) start_sample = 0;
    if (end_sample > sample_count) end_sample = sample_count;
    int seg_len = end_sample - start_sample;

    if (seg_len < FRAME_SIZE) {
        memset(out_embedding, 0, sizeof(float) * EMBED_DIM);
        return false;
    }

    int num_frames = (seg_len - FRAME_SIZE) / FRAME_STEP + 1;
    if (num_frames > MAX_ANALYSIS_FRAMES) num_frames = MAX_ANALYSIS_FRAMES;
    if (num_frames <= 0) {
        memset(out_embedding, 0, sizeof(float) * EMBED_DIM);
        return false;
    }

    float *mfccs = (float *)malloc(sizeof(float) * (size_t)num_frames * USE_MFCC);
    float *centroids = (float *)malloc(sizeof(float) * (size_t)num_frames);
    float *pitches = (float *)malloc(sizeof(float) * (size_t)num_frames);
    float *subbands = (float *)malloc(sizeof(float) * (size_t)num_frames * 3);

    if (!mfccs || !centroids || !pitches || !subbands) {
        free(mfccs);
        free(centroids);
        free(pitches);
        free(subbands);
        memset(out_embedding, 0, sizeof(float) * EMBED_DIM);
        return false;
    }

    for (int f = 0; f < num_frames; f++) {
        int offset = start_sample + f * FRAME_STEP;
        process_frame(
            &samples[offset],
            (f % PITCH_FRAME_STRIDE) == 0,
            &mfccs[f * USE_MFCC],
            &centroids[f],
            &pitches[f],
            &subbands[f * 3]
        );
    }

    // 1. MFCC Means (12) & StdDevs (12)
    float mfcc_means[USE_MFCC];
    float mfcc_stds[USE_MFCC];
    for (int m = 0; m < USE_MFCC; m++) {
        float sum = 0.0f;
        for (int f = 0; f < num_frames; f++) {
            sum += mfccs[f * USE_MFCC + m];
        }
        mfcc_means[m] = sum / (float)num_frames;

        float var_sum = 0.0f;
        for (int f = 0; f < num_frames; f++) {
            float diff = mfccs[f * USE_MFCC + m] - mfcc_means[m];
            var_sum += diff * diff;
        }
        mfcc_stds[m] = sqrtf(var_sum / (float)num_frames);
    }

    // 2. Delta MFCC Means (12)
    float delta_means[USE_MFCC];
    for (int m = 0; m < USE_MFCC; m++) {
        float sum = 0.0f;
        int delta_count = 0;
        for (int f = 1; f < num_frames; f++) {
            float d = mfccs[f * USE_MFCC + m] - mfccs[(f - 1) * USE_MFCC + m];
            sum += fabsf(d);
            delta_count++;
        }
        delta_means[m] = delta_count > 0 ? (sum / (float)delta_count) : 0.0f;
    }

    // 3. Pitch Mean & Std (voiced frames only; unestimated frames carry -1)
    float pitch_sum = 0.0f;
    int voiced_count = 0;
    for (int f = 0; f < num_frames; f++) {
        if (pitches[f] > 0.0f) {
            pitch_sum += pitches[f];
            voiced_count++;
        }
    }
    float pitch_mean = voiced_count > 0 ? (pitch_sum / (float)voiced_count) : 150.0f;
    float pitch_var = 0.0f;
    for (int f = 0; f < num_frames; f++) {
        if (pitches[f] > 0.0f) {
            float diff = pitches[f] - pitch_mean;
            pitch_var += diff * diff;
        }
    }
    float pitch_std = voiced_count > 0 ? sqrtf(pitch_var / (float)voiced_count) : 20.0f;

    // 4. Centroid Mean & Std
    float centroid_sum = 0.0f;
    for (int f = 0; f < num_frames; f++) {
        centroid_sum += centroids[f];
    }
    float centroid_mean = centroid_sum / (float)num_frames;
    float centroid_var = 0.0f;
    for (int f = 0; f < num_frames; f++) {
        float diff = centroids[f] - centroid_mean;
        centroid_var += diff * diff;
    }
    float centroid_std = sqrtf(centroid_var / (float)num_frames);

    // 5. Subband ratios Mean (3)
    float subband_means[3] = {0.0f, 0.0f, 0.0f};
    for (int f = 0; f < num_frames; f++) {
        subband_means[0] += subbands[f * 3 + 0];
        subband_means[1] += subbands[f * 3 + 1];
        subband_means[2] += subbands[f * 3 + 2];
    }
    subband_means[0] /= (float)num_frames;
    subband_means[1] /= (float)num_frames;
    subband_means[2] /= (float)num_frames;

    free(mfccs);
    free(centroids);
    free(pitches);
    free(subbands);

    // Cepstral centering: removes the component shared by every coefficient of a
    // block, which is dominated by the overall loudness of the segment.
    //
    // NOTE: this is deliberately NOT Cepstral Mean Subtraction. Real CMS
    // subtracts, per coefficient, the mean *across frames* — but the features
    // below are built from exactly those per-coefficient frame means, so CMS
    // would zero them out. Channel compensation is instead applied once across
    // the whole recording in performPostDiarization (Kotlin), where the full set
    // of segment embeddings is available.
    float mfcc_block_mean = 0.0f;
    for (int i = 0; i < USE_MFCC; i++) mfcc_block_mean += mfcc_means[i];
    mfcc_block_mean /= (float)USE_MFCC;

    float std_block_mean = 0.0f;
    for (int i = 0; i < USE_MFCC; i++) std_block_mean += mfcc_stds[i];
    std_block_mean /= (float)USE_MFCC;

    float delta_block_mean = 0.0f;
    for (int i = 0; i < USE_MFCC; i++) delta_block_mean += delta_means[i];
    delta_block_mean /= (float)USE_MFCC;

    // Assemble 43-dim embedding (12 + 12 + 12 + 2 + 2 + 3 = 43)
    int idx = 0;
    for (int i = 0; i < USE_MFCC; i++) out_embedding[idx++] = (mfcc_means[i] - mfcc_block_mean) * 1.0f;
    for (int i = 0; i < USE_MFCC; i++) out_embedding[idx++] = (mfcc_stds[i] - std_block_mean) * 0.5f;
    for (int i = 0; i < USE_MFCC; i++) out_embedding[idx++] = (delta_means[i] - delta_block_mean) * 0.5f;

    // Octave-scale pitch (log2) for natural acoustic human vocal range
    float log_pitch = (pitch_mean > 50.0f) ? (logf(pitch_mean / 100.0f) / 0.693147f) : 0.5f; // log2(pitch / 100Hz)
    out_embedding[idx++] = log_pitch * 1.0f;
    out_embedding[idx++] = (pitch_std / 100.0f) * 0.3f;
    out_embedding[idx++] = ((centroid_mean - 1800.0f) / 2000.0f) * 0.8f;
    out_embedding[idx++] = (centroid_std / 1000.0f) * 0.3f;
    out_embedding[idx++] = (subband_means[0] - 0.33f) * 1.0f;
    out_embedding[idx++] = (subband_means[1] - 0.33f) * 1.0f;
    out_embedding[idx++] = (subband_means[2] - 0.33f) * 1.0f;

    // L2 Normalize
    float norm_sq = 0.0f;
    for (int i = 0; i < EMBED_DIM; i++) {
        norm_sq += out_embedding[i] * out_embedding[i];
    }
    float norm = sqrtf(norm_sq);
    if (norm > 1e-6f) {
        for (int i = 0; i < EMBED_DIM; i++) {
            out_embedding[i] /= norm;
        }
    }

    return true;
}

bool extract_segment_embedding(
    const float *samples,
    int sample_count,
    int64_t seg_t0,
    int64_t seg_t1,
    float *out_embedding
) {
    if (!samples || sample_count <= 0 || !out_embedding || seg_t1 <= seg_t0) return false;
    init_tables();

    // t0/t1 are Whisper centiseconds; 160 samples per centisecond at 16 kHz.
    int64_t start64 = seg_t0 * 160;
    int64_t end64   = seg_t1 * 160;
    if (start64 < 0) start64 = 0;
    if (end64 > (int64_t)sample_count) end64 = (int64_t)sample_count;
    if (end64 <= start64) return false;

    return compute_segment_embedding(
        samples,
        sample_count,
        (int)start64,
        (int)end64,
        out_embedding
    );
}
