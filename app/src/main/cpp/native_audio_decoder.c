#include "native_audio_decoder.h"

#include <android/log.h>
#include <inttypes.h>
#include <math.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_ARM_NEON 1
#else
#define HAVE_ARM_NEON 0
#endif

#define TAG "NativeAudioDecoder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define TARGET_SAMPLE_RATE 16000
#define WAV_HEADER_SIZE 44
#define IO_BUFFER_SIZE (256 * 1024)
#define WRITE_CHUNK_SAMPLES 65536

// Defensive floor on the declared input rate. Everything below this would make
// the resampler produce more than 4x the input frames per buffer, and no real
// codec reports it; clamping keeps the output-capacity math provably safe.
#define MIN_INPUT_SAMPLE_RATE 4000

// Anti-alias cutoff. Anything above 8 kHz folds back into the 16 kHz output
// band, so the low-pass has to sit safely below the output Nyquist.
#define ANTIALIAS_CUTOFF_HZ 7000.0

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int64_t get_now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000LL + (int64_t)ts.tv_nsec / 1000000LL;
}

// ---------------------------------------------------------------------------
// Biquad low-pass (RBJ cookbook, transposed direct form II)
// ---------------------------------------------------------------------------
typedef struct {
    float b0, b1, b2, a1, a2;
    float z1, z2;
} biquad_t;

static void biquad_set_lowpass(biquad_t *f, double fc, double fs, double q) {
    double w0 = 2.0 * M_PI * fc / fs;
    if (w0 > 3.0) w0 = 3.0; // stay below pi
    if (w0 < 1e-6) w0 = 1e-6;
    const double cw = cos(w0);
    const double sw = sin(w0);
    const double alpha = sw / (2.0 * q);
    const double a0 = 1.0 + alpha;

    f->b0 = (float)(((1.0 - cw) / 2.0) / a0);
    f->b1 = (float)((1.0 - cw) / a0);
    f->b2 = f->b0;
    f->a1 = (float)((-2.0 * cw) / a0);
    f->a2 = (float)((1.0 - alpha) / a0);
    f->z1 = 0.0f;
    f->z2 = 0.0f;
}

static inline float biquad_step(biquad_t *f, float x) {
    const float y = f->b0 * x + f->z1;
    f->z1 = f->b1 * x - f->a1 * y + f->z2;
    f->z2 = f->b2 * x - f->a2 * y;
    return y;
}

// ---------------------------------------------------------------------------
// Stateful Q16 resampler.
//
// The phase accumulator and the last input sample persist across codec output
// buffers. Resetting them per buffer (as an earlier revision did) restarted the
// interpolation at phase 0 hundreds of times per second, which put a hard
// discontinuity at every buffer boundary and slowly drifted the total duration.
// ---------------------------------------------------------------------------
typedef struct {
    int in_sr;
    int32_t ratio_q16;
    int32_t pos_q16; // phase of the next output sample, relative to src[0]; >= -65536
    int16_t hist;    // last sample of the previous buffer, i.e. virtual index -1
    bool passthrough;
    bool antialias;
    biquad_t lp[2]; // 4th-order Butterworth = two cascaded biquads
} resampler_t;

static void resampler_configure(resampler_t *r, int in_sr) {
    if (in_sr < MIN_INPUT_SAMPLE_RATE) in_sr = MIN_INPUT_SAMPLE_RATE;
    r->in_sr = in_sr;
    r->ratio_q16 = (int32_t)(((int64_t)in_sr << 16) / TARGET_SAMPLE_RATE);
    if (r->ratio_q16 < 1) r->ratio_q16 = 1;
    r->pos_q16 = 0;
    r->hist = 0;
    r->passthrough = (in_sr == TARGET_SAMPLE_RATE);
    r->antialias = (in_sr > TARGET_SAMPLE_RATE);

    if (r->antialias) {
        double fc = ANTIALIAS_CUTOFF_HZ;
        const double out_nyquist = TARGET_SAMPLE_RATE / 2.0;
        if (fc > out_nyquist * 0.9) fc = out_nyquist * 0.9;
        // Butterworth 4th order Q factors.
        biquad_set_lowpass(&r->lp[0], fc, (double)in_sr, 0.54119610);
        biquad_set_lowpass(&r->lp[1], fc, (double)in_sr, 1.30656296);
    }
}

/** Upper bound on the frames [resampler_process] can emit for [in_frames] input frames. */
static int resampler_max_out(const resampler_t *r, int in_frames) {
    if (r->passthrough) return in_frames;
    const int64_t n = (((int64_t)(in_frames + 2)) << 16) / (int64_t)r->ratio_q16;
    return (int)n + 4;
}

/** Low-pass the input buffer in place before decimation (downsampling only). */
static void resampler_antialias(resampler_t *r, int16_t *buf, int n) {
    if (!r->antialias) return;
    for (int i = 0; i < n; i++) {
        float x = (float)buf[i];
        x = biquad_step(&r->lp[0], x);
        x = biquad_step(&r->lp[1], x);
        int v = (int)lrintf(x);
        if (v > 32767) v = 32767;
        else if (v < -32768) v = -32768;
        buf[i] = (int16_t)v;
    }
}

static int resampler_process(resampler_t *r, const int16_t *src, int in_frames, int16_t *dst) {
    if (in_frames <= 0) return 0;

    if (r->passthrough) {
        memcpy(dst, src, (size_t)in_frames * sizeof(int16_t));
        r->hist = src[in_frames - 1];
        r->pos_q16 = 0;
        return in_frames;
    }

    int out = 0;
    while (1) {
        const int i0 = r->pos_q16 >> 16; // arithmetic shift: floors for negatives too
        if (i0 + 1 >= in_frames) break;  // need one more input sample to interpolate

        const int32_t frac_q16 = r->pos_q16 & 0xFFFF;
        const int32_t s0 = (i0 < 0) ? (int32_t)r->hist : (int32_t)src[i0];
        const int32_t s1 = (int32_t)src[i0 + 1];
        dst[out++] = (int16_t)(s0 + (((s1 - s0) * frac_q16) >> 16));
        r->pos_q16 += r->ratio_q16;
    }

    r->hist = src[in_frames - 1];
    r->pos_q16 -= (int32_t)in_frames << 16; // rebase onto the next buffer
    return out;
}

// ---------------------------------------------------------------------------
// Downmix raw multi-channel PCM16 samples to mono PCM16
// ---------------------------------------------------------------------------
static inline void downmix_pcm16(const int16_t *src, int16_t *dst, int in_frames, int channels) {
    if (channels == 1) {
        memcpy(dst, src, (size_t)in_frames * sizeof(int16_t));
        return;
    }

    if (channels == 2) {
        int i = 0;
#if HAVE_ARM_NEON
        for (; i + 7 < in_frames; i += 8) {
            int16x8x2_t stereo = vld2q_s16(src + i * 2);
            int16x8_t mono = vrhaddq_s16(stereo.val[0], stereo.val[1]);
            vst1q_s16(dst + i, mono);
        }
#endif
        for (; i < in_frames; i++) {
            int s = (int)src[i * 2] + (int)src[i * 2 + 1];
            dst[i] = (int16_t)(s >> 1);
        }
        return;
    }

    for (int i = 0; i < in_frames; i++) {
        int sum = 0;
        const int16_t *base = src + (i * channels);
        for (int c = 0; c < channels; c++) {
            sum += (int)base[c];
        }
        dst[i] = (int16_t)(sum / channels);
    }
}

// ---------------------------------------------------------------------------
// PCM pipeline: downmix -> anti-alias -> resample -> staged write.
// Owns every scratch buffer so the two drain sites in the decode loop share one
// implementation and one set of allocation/IO error checks.
// ---------------------------------------------------------------------------
typedef struct {
    FILE *f;
    int16_t *mono;
    size_t mono_cap;
    int16_t *res;
    size_t res_cap;
    int16_t *stage;
    size_t stage_count;
    resampler_t rs;
    uint64_t frames_written;
    bool io_error;
    bool alloc_error;
} pipeline_t;

static bool pipeline_ensure(int16_t **buf, size_t *cap, size_t needed) {
    if (*cap >= needed && *buf != NULL) return true;
    size_t new_cap = (needed * 3) / 2 + 1024;
    int16_t *grown = (int16_t *)realloc(*buf, new_cap * sizeof(int16_t));
    if (grown == NULL) {
        LOGE("[C-NDK] realloc of %zu samples failed", new_cap);
        return false; // *buf stays valid and is freed by the caller
    }
    *buf = grown;
    *cap = new_cap;
    return true;
}

static bool pipeline_flush_stage(pipeline_t *p) {
    if (p->stage_count == 0) return true;
    const size_t wrote = fwrite(p->stage, sizeof(int16_t), p->stage_count, p->f);
    if (wrote != p->stage_count) {
        LOGE("[C-NDK] short write: %zu/%zu samples (disk full?)", wrote, p->stage_count);
        p->io_error = true;
        return false;
    }
    p->stage_count = 0;
    return true;
}

static bool pipeline_write(pipeline_t *p, const int16_t *samples, size_t n) {
    if (n == 0) return true;
    if (p->stage_count + n > WRITE_CHUNK_SAMPLES) {
        if (!pipeline_flush_stage(p)) return false;
    }
    if (n >= WRITE_CHUNK_SAMPLES) {
        const size_t wrote = fwrite(samples, sizeof(int16_t), n, p->f);
        if (wrote != n) {
            LOGE("[C-NDK] short write: %zu/%zu samples (disk full?)", wrote, n);
            p->io_error = true;
            return false;
        }
    } else {
        memcpy(p->stage + p->stage_count, samples, n * sizeof(int16_t));
        p->stage_count += n;
    }
    return true;
}

/** Feeds one codec output buffer through the whole chain. Returns false on a fatal error. */
static bool pipeline_feed(pipeline_t *p, const int16_t *raw_pcm, int sample_count,
                          int channels, int in_sr) {
    if (channels <= 0) channels = 1;
    const int in_frames = sample_count / channels;
    if (in_frames <= 0) return true;

    if (in_sr != p->rs.in_sr) {
        LOGI("[C-NDK] Input rate changed %d -> %d, reconfiguring resampler", p->rs.in_sr, in_sr);
        resampler_configure(&p->rs, in_sr);
    }

    if (!pipeline_ensure(&p->mono, &p->mono_cap, (size_t)in_frames)) {
        p->alloc_error = true;
        return false;
    }
    const int max_out = resampler_max_out(&p->rs, in_frames);
    if (!pipeline_ensure(&p->res, &p->res_cap, (size_t)max_out)) {
        p->alloc_error = true;
        return false;
    }

    downmix_pcm16(raw_pcm, p->mono, in_frames, channels);
    resampler_antialias(&p->rs, p->mono, in_frames);
    const int out_frames = resampler_process(&p->rs, p->mono, in_frames, p->res);

    if (out_frames > 0) {
        if (!pipeline_write(p, p->res, (size_t)out_frames)) return false;
        p->frames_written += (uint64_t)out_frames;
    }
    return true;
}

static void pipeline_release(pipeline_t *p) {
    free(p->mono);
    free(p->res);
    free(p->stage);
    p->mono = NULL;
    p->res = NULL;
    p->stage = NULL;
}

// ---------------------------------------------------------------------------
// WAV header
// ---------------------------------------------------------------------------
static bool write_wav_header(FILE *f, uint32_t data_bytes) {
    uint8_t header[WAV_HEADER_SIZE];
    uint32_t total_chunk_size = 36 + data_bytes;
    uint32_t byte_rate = TARGET_SAMPLE_RATE * 2; // 16 kHz * 1 channel * 2 bytes/sample

    // RIFF header
    header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
    header[4] = (uint8_t)(total_chunk_size & 0xFF);
    header[5] = (uint8_t)((total_chunk_size >> 8) & 0xFF);
    header[6] = (uint8_t)((total_chunk_size >> 16) & 0xFF);
    header[7] = (uint8_t)((total_chunk_size >> 24) & 0xFF);
    header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';

    // fmt subchunk
    header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
    header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0;
    header[20] = 1; header[21] = 0;
    header[22] = 1; header[23] = 0;
    header[24] = (uint8_t)(TARGET_SAMPLE_RATE & 0xFF);
    header[25] = (uint8_t)((TARGET_SAMPLE_RATE >> 8) & 0xFF);
    header[26] = (uint8_t)((TARGET_SAMPLE_RATE >> 16) & 0xFF);
    header[27] = (uint8_t)((TARGET_SAMPLE_RATE >> 24) & 0xFF);
    header[28] = (uint8_t)(byte_rate & 0xFF);
    header[29] = (uint8_t)((byte_rate >> 8) & 0xFF);
    header[30] = (uint8_t)((byte_rate >> 16) & 0xFF);
    header[31] = (uint8_t)((byte_rate >> 24) & 0xFF);
    header[32] = 2; header[33] = 0;
    header[34] = 16; header[35] = 0;

    // data subchunk
    header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
    header[40] = (uint8_t)(data_bytes & 0xFF);
    header[41] = (uint8_t)((data_bytes >> 8) & 0xFF);
    header[42] = (uint8_t)((data_bytes >> 16) & 0xFF);
    header[43] = (uint8_t)((data_bytes >> 24) & 0xFF);

    return fwrite(header, 1, WAV_HEADER_SIZE, f) == WAV_HEADER_SIZE;
}

// ---------------------------------------------------------------------------
int native_decode_to_wav(
    int fd,
    int64_t offset,
    int64_t length,
    const char* out_wav_path,
    decode_progress_callback on_progress,
    decode_cancel_callback is_cancelled,
    void* user_data
) {
    int64_t t_start = get_now_ms();

    if (fd < 0 || !out_wav_path) {
        LOGE("[C-NDK] Invalid arguments: fd=%d out_path=%s", fd, out_wav_path ? out_wav_path : "null");
        return -1;
    }

    AMediaExtractor* extractor = AMediaExtractor_new();
    if (!extractor) {
        LOGE("[C-NDK] AMediaExtractor_new() returned NULL");
        return -2;
    }

    off64_t safe_offset = (offset >= 0) ? (off64_t)offset : 0;
    off64_t safe_length = length;
    if (safe_length <= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > safe_offset) {
            safe_length = (off64_t)(st.st_size - safe_offset);
        } else {
            safe_length = 0x7FFFFFFFFFFFFFFFLL;
        }
    }

    LOGI("[C-NDK] Setting data source fd=%d offset=%" PRId64 " length=%" PRId64,
         fd, (int64_t)safe_offset, (int64_t)safe_length);

    media_status_t status = AMediaExtractor_setDataSourceFd(
        extractor,
        fd,
        safe_offset,
        safe_length
    );

    if (status != AMEDIA_OK) {
        LOGE("[C-NDK] AMediaExtractor_setDataSourceFd FAILED: status=%d", status);
        AMediaExtractor_delete(extractor);
        return -3;
    }

    size_t num_tracks = AMediaExtractor_getTrackCount(extractor);
    int audio_track_idx = -1;
    AMediaFormat* format = NULL;
    const char* mime = NULL;
    int32_t input_sr = 0;
    int32_t input_channels = 1;
    int64_t duration_us = 0;

    for (size_t i = 0; i < num_tracks; i++) {
        AMediaFormat* fmt = AMediaExtractor_getTrackFormat(extractor, i);
        const char* track_mime = NULL;
        if (AMediaFormat_getString(fmt, AMEDIAFORMAT_KEY_MIME, &track_mime) && track_mime) {
            if (strncmp(track_mime, "audio/", 6) == 0) {
                audio_track_idx = (int)i;
                format = fmt;
                mime = track_mime;
                AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_SAMPLE_RATE, &input_sr);
                AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_CHANNEL_COUNT, &input_channels);
                AMediaFormat_getInt64(format, AMEDIAFORMAT_KEY_DURATION, &duration_us);
                break;
            }
        }
        AMediaFormat_delete(fmt);
    }

    if (audio_track_idx < 0 || !format || !mime) {
        LOGE("[C-NDK] No audio track found in media source (total tracks=%zu)", num_tracks);
        AMediaExtractor_delete(extractor);
        return -4;
    }

    if (input_channels <= 0) input_channels = 1;
    if (input_sr <= 0) input_sr = 44100;

    LOGI("[C-NDK] Audio track %d found: mime=%s sr=%d ch=%d dur=%.2f s",
         audio_track_idx, mime, input_sr, input_channels, (double)duration_us / 1000000.0);

    AMediaExtractor_selectTrack(extractor, (size_t)audio_track_idx);

    AMediaCodec* codec = AMediaCodec_createDecoderByType(mime);
    if (!codec) {
        LOGE("[C-NDK] AMediaCodec_createDecoderByType(%s) failed", mime);
        AMediaFormat_delete(format);
        AMediaExtractor_delete(extractor);
        return -5;
    }

    status = AMediaCodec_configure(codec, format, NULL, NULL, 0);
    AMediaFormat_delete(format);

    if (status != AMEDIA_OK) {
        LOGE("[C-NDK] AMediaCodec_configure failed: status=%d", status);
        AMediaCodec_delete(codec);
        AMediaExtractor_delete(extractor);
        return -6;
    }

    status = AMediaCodec_start(codec);
    if (status != AMEDIA_OK) {
        LOGE("[C-NDK] AMediaCodec_start failed: status=%d", status);
        AMediaCodec_delete(codec);
        AMediaExtractor_delete(extractor);
        return -7;
    }

    FILE* f_out = fopen(out_wav_path, "wb");
    if (!f_out) {
        LOGE("[C-NDK] Failed to create output file: %s", out_wav_path);
        AMediaCodec_stop(codec);
        AMediaCodec_delete(codec);
        AMediaExtractor_delete(extractor);
        return -8;
    }

    char* io_buf = (char*)malloc(IO_BUFFER_SIZE);
    if (io_buf) {
        setvbuf(f_out, io_buf, _IOFBF, IO_BUFFER_SIZE);
    }

    int ret_code = 0;
    pipeline_t pipe;
    memset(&pipe, 0, sizeof(pipe));
    pipe.f = f_out;
    resampler_configure(&pipe.rs, input_sr);

    pipeline_ensure(&pipe.mono, &pipe.mono_cap, 65536);
    pipeline_ensure(&pipe.res, &pipe.res_cap, 65536);
    // The staging buffer never grows, so allocate it at its fixed size.
    pipe.stage = (int16_t*)malloc(WRITE_CHUNK_SAMPLES * sizeof(int16_t));

    if (!pipe.mono || !pipe.res || !pipe.stage) {
        LOGE("[C-NDK] Out of memory allocating decode scratch buffers");
        pipeline_release(&pipe);
        fclose(f_out);
        if (io_buf) free(io_buf);
        AMediaCodec_stop(codec);
        AMediaCodec_delete(codec);
        AMediaExtractor_delete(extractor);
        return -10;
    }

    if (!write_wav_header(f_out, 0)) {
        LOGE("[C-NDK] Failed to write placeholder WAV header");
        pipeline_release(&pipe);
        fclose(f_out);
        if (io_buf) free(io_buf);
        AMediaCodec_stop(codec);
        AMediaCodec_delete(codec);
        AMediaExtractor_delete(extractor);
        return -11;
    }

    AMediaCodecBufferInfo info;
    bool saw_input_eos = false;
    bool saw_output_eos = false;
    int64_t total_in_packets = 0;
    int64_t total_bytes_read = 0;
    int last_reported_pct = -1;

    LOGI("[C-NDK] Starting optimized decode loop...");

    while (!saw_output_eos) {
        if (is_cancelled && is_cancelled(user_data)) {
            LOGI("[C-NDK] Cancelled by user");
            ret_code = -99;
            break;
        }

        // 1. Feed input buffers (burst up to 32 packets per pass)
        int in_fed_this_pass = 0;
        while (!saw_input_eos && in_fed_this_pass < 32) {
            ssize_t in_idx = AMediaCodec_dequeueInputBuffer(codec, 0);
            if (in_idx < 0) break; // input queue full

            size_t buf_size = 0;
            uint8_t* in_buf = AMediaCodec_getInputBuffer(codec, (size_t)in_idx, &buf_size);
            if (!in_buf || buf_size == 0) {
                AMediaCodec_queueInputBuffer(codec, (size_t)in_idx, 0, 0, 0, 0);
                break;
            }

            ssize_t sample_size = AMediaExtractor_readSampleData(extractor, in_buf, buf_size);
            if (sample_size < 0) {
                AMediaCodec_queueInputBuffer(codec, (size_t)in_idx, 0, 0, 0, AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
                saw_input_eos = true;
                LOGI("[C-NDK] Input EOS reached (total packets: %" PRId64 ", total bytes: %" PRId64 ")",
                     total_in_packets, total_bytes_read);
                break;
            } else {
                int64_t pts = AMediaExtractor_getSampleTime(extractor);
                uint64_t safe_pts = (pts >= 0) ? (uint64_t)pts : 0;
                AMediaCodec_queueInputBuffer(codec, (size_t)in_idx, 0, (size_t)sample_size, safe_pts, 0);
                AMediaExtractor_advance(extractor);
                in_fed_this_pass++;
                total_in_packets++;
                total_bytes_read += sample_size;

                if (on_progress) {
                    float p = 0.0f;
                    if (duration_us > 0 && pts >= 0) {
                        p = (float)pts / (float)duration_us;
                    } else if (safe_length > 0 && safe_length < 0x7000000000000000LL) {
                        p = (float)total_bytes_read / (float)safe_length;
                    }
                    if (p < 0.0f) p = 0.0f;
                    if (p > 0.98f) p = 0.98f;
                    int pct = (int)(p * 100.0f);
                    if (pct != last_reported_pct) {
                        last_reported_pct = pct;
                        on_progress(p, user_data);
                    }
                }
            }
        }

        // 2. Drain all available output buffers in a non-blocking loop (timeout = 0)
        int out_drained_this_pass = 0;
        while (true) {
            ssize_t out_idx = AMediaCodec_dequeueOutputBuffer(codec, &info, 0);
            if (out_idx == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
                break;
            } else if (out_idx == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
                AMediaFormat* new_fmt = AMediaCodec_getOutputFormat(codec);
                if (new_fmt) {
                    AMediaFormat_getInt32(new_fmt, AMEDIAFORMAT_KEY_SAMPLE_RATE, &input_sr);
                    AMediaFormat_getInt32(new_fmt, AMEDIAFORMAT_KEY_CHANNEL_COUNT, &input_channels);
                    AMediaFormat_delete(new_fmt);
                    if (input_channels <= 0) input_channels = 1;
                    if (input_sr <= 0) input_sr = TARGET_SAMPLE_RATE;
                    LOGI("[C-NDK] Format updated: sr=%d ch=%d", input_sr, input_channels);
                }
                continue;
            } else if (out_idx >= 0) {
                size_t out_size = 0;
                uint8_t* out_buf = AMediaCodec_getOutputBuffer(codec, (size_t)out_idx, &out_size);
                if (out_buf && info.size > 0) {
                    const int16_t* raw_pcm = (const int16_t*)(out_buf + info.offset);
                    const int sample_count = info.size / (int)sizeof(int16_t);
                    if (!pipeline_feed(&pipe, raw_pcm, sample_count, input_channels, input_sr)) {
                        AMediaCodec_releaseOutputBuffer(codec, (size_t)out_idx, false);
                        ret_code = pipe.alloc_error ? -10 : -12;
                        saw_output_eos = true;
                        break;
                    }
                }

                AMediaCodec_releaseOutputBuffer(codec, (size_t)out_idx, false);
                out_drained_this_pass++;

                if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
                    saw_output_eos = true;
                    LOGI("[C-NDK] Output EOS reached (total written: %" PRIu64 " frames)", pipe.frames_written);
                    break;
                }
            }
        }

        if (ret_code != 0) break;

        // 3. If neither input was queued nor output drained, wait briefly on hardware DSP
        if (in_fed_this_pass == 0 && out_drained_this_pass == 0 && !saw_output_eos) {
            ssize_t out_idx = AMediaCodec_dequeueOutputBuffer(codec, &info, 100); // 100us DSP wait
            if (out_idx >= 0) {
                size_t out_size = 0;
                uint8_t* out_buf = AMediaCodec_getOutputBuffer(codec, (size_t)out_idx, &out_size);
                if (out_buf && info.size > 0) {
                    const int16_t* raw_pcm = (const int16_t*)(out_buf + info.offset);
                    const int sample_count = info.size / (int)sizeof(int16_t);
                    if (!pipeline_feed(&pipe, raw_pcm, sample_count, input_channels, input_sr)) {
                        AMediaCodec_releaseOutputBuffer(codec, (size_t)out_idx, false);
                        ret_code = pipe.alloc_error ? -10 : -12;
                        break;
                    }
                }
                AMediaCodec_releaseOutputBuffer(codec, (size_t)out_idx, false);
                if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
                    saw_output_eos = true;
                }
            }
        }
    }

    // Flush any remaining buffered samples
    if (ret_code == 0 && !pipeline_flush_stage(&pipe)) {
        ret_code = -12;
    }

    if (fflush(f_out) != 0) {
        LOGE("[C-NDK] fflush failed (disk full?)");
        if (ret_code == 0) ret_code = -12;
    }

    int64_t t_elapsed = get_now_ms() - t_start;

    if (ret_code == 0 && pipe.frames_written > 0) {
        // The int16 sample count must stay addressable by the 32-bit RIFF size
        // field; 4 GB of PCM16 is ~37 h of 16 kHz mono.
        if (pipe.frames_written > 0x7FFFFFFFULL) {
            LOGE("[C-NDK] Recording too long for a 32-bit WAV data chunk (%" PRIu64 " frames)", pipe.frames_written);
            ret_code = -13;
        } else {
            uint32_t data_bytes = (uint32_t)(pipe.frames_written * sizeof(int16_t));
            if (fseek(f_out, 0, SEEK_SET) != 0 || !write_wav_header(f_out, data_bytes) || fflush(f_out) != 0) {
                LOGE("[C-NDK] Failed to patch the final WAV header");
                ret_code = -12;
            } else {
                if (on_progress) on_progress(1.0f, user_data);
                LOGI("[C-NDK] SUCCESS: Decoded %" PRIu64 " frames (%" PRIu32 " bytes) in %" PRId64 " ms (%.1fx realtime)",
                     pipe.frames_written, data_bytes, t_elapsed,
                     (t_elapsed > 0) ? (double)pipe.frames_written / (16.0 * (double)t_elapsed) : 0.0);
            }
        }
    }

    if (ret_code != 0 || pipe.frames_written == 0) {
        if (ret_code == 0) ret_code = -9;
        LOGW("[C-NDK] FAILED with code %d (frames=%" PRIu64 ", elapsed=%" PRId64 " ms)",
             ret_code, pipe.frames_written, t_elapsed);
    }

    fclose(f_out);
    pipeline_release(&pipe);
    if (io_buf) free(io_buf);

    AMediaCodec_stop(codec);
    AMediaCodec_delete(codec);
    AMediaExtractor_delete(extractor);

    return ret_code;
}
