// JNI bridge between Kotlin (com.emilio.whispert.whisper.WhisperLib) and whisper.cpp.
//
// Adapted from the upstream whisper.cpp Android example (examples/whisper.android/lib/src/main/jni/whisper/jni.c)
// with JNI function names rewritten for our package: com.emilio.whispert.whisper.WhisperLib.
//
// Exposes:
//   - initContext(modelPath)            : load model from a filesystem path (downloaded to internal storage)
//   - freeContext(ptr)
//   - fullTranscribe(ptr, ..., audioData) : in-memory float samples (short clips)
//   - fullTranscribeFile(ptr, ..., wav)   : streams a mono PCM16 16 kHz WAV from native memory
//   - getTextSegmentCount/Text/T0/T1(ptr, i)
//   - getTimings / getModelInfo
//   - nativeDecodeAudioFd                 : NDK audio decode to WAV
//   - extractSegmentEmbedding             : DSP speaker voiceprint for one segment
//
// IMPORTANT: whisper.cpp is NOT thread-safe. Every native call must be dispatched
// from a single dedicated thread (handled in WhisperContext.kt).
#include <jni.h>
#include <android/log.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <string.h>
#include <pthread.h>
#include "whisper.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "native_audio_decoder.h"
#include "speaker_diarizer.h"

#define UNUSED(x) (void)(x)
#define TAG "whisper-jni"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// Cancellation + progress plumbing (whisper_full_params callbacks)
// ---------------------------------------------------------------------------
// One flag per phase, because the two do not run on the same thread and can
// overlap: the decode runs on the caller's thread while whisper_full is still
// finishing on the inference executor. With a single shared flag, cancelling a
// run in progress and immediately starting a new one made the new run's decode
// reset the flag that the old whisper_full had not yet observed, so the cancel
// was swallowed and the old transcription kept going to completion.
//
// Set from any thread by cancelTranscription(); each phase clears only its own
// flag at its own start, so a late cancel never leaks into the next run.
static volatile int g_abort_transcribe = 0;
static volatile int g_abort_decode = 0;

static bool abort_cb(void * user_data) {
    UNUSED(user_data);
    return __atomic_load_n(&g_abort_transcribe, __ATOMIC_RELAXED) != 0;
}

// Cached JVM + WhisperProgress methods
static JavaVM *g_jvm = NULL;
static jclass   g_progress_cls = NULL;
static jmethodID g_progress_mid = NULL;
static jmethodID g_decode_progress_mid = NULL;
static jmethodID g_new_segment_mid = NULL;

// Reachable from the inference thread, the decode thread and the callback
// threads, so the lazy init needs a lock: without one, two threads racing here
// each installed a NewGlobalRef and one of them leaked permanently.
static pthread_mutex_t g_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

static void cache_progress_methods(JNIEnv *env) {
    pthread_mutex_lock(&g_cache_mutex);
    if (g_progress_cls != NULL) {
        pthread_mutex_unlock(&g_cache_mutex);
        return;
    }

    jclass cls = (*env)->FindClass(env, "com/emilio/whispert/whisper/WhisperProgress");
    if (cls == NULL) {
        (*env)->ExceptionClear(env);
        pthread_mutex_unlock(&g_cache_mutex);
        return;
    }

    jclass global = (jclass)(*env)->NewGlobalRef(env, cls);
    (*env)->DeleteLocalRef(env, cls);
    if (global == NULL) {
        (*env)->ExceptionClear(env);
        pthread_mutex_unlock(&g_cache_mutex);
        return;
    }

    const jmethodID progress = (*env)->GetStaticMethodID(env, global, "onProgress", "(I)V");
    const jmethodID decode   = (*env)->GetStaticMethodID(env, global, "onDecodeProgress", "(I)V");
    const jmethodID segment  = (*env)->GetStaticMethodID(env, global, "onNewSegment", "(Ljava/lang/String;JJ)V");

    // A failed GetStaticMethodID leaves an exception pending; any further JNI
    // call on this env with one pending is undefined behaviour, so clear it
    // here rather than letting it escape into the caller.
    if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);

    if (progress == NULL || decode == NULL || segment == NULL) {
        LOGE("WhisperProgress found but its callback methods did not resolve; progress events disabled");
        (*env)->DeleteGlobalRef(env, global);
        pthread_mutex_unlock(&g_cache_mutex);
        return;
    }

    g_progress_mid = progress;
    g_decode_progress_mid = decode;
    g_new_segment_mid = segment;
    // Published last, with release semantics: a reader that sees a non-NULL
    // class is guaranteed to see the three method ids that go with it.
    __atomic_store_n(&g_progress_cls, global, __ATOMIC_RELEASE);

    pthread_mutex_unlock(&g_cache_mutex);
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    UNUSED(reserved);
    g_jvm = vm;
    JNIEnv *env = NULL;
    if ((*vm)->GetEnv(vm, (void **)&env, JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }

    cache_progress_methods(env);
    if (g_progress_cls != NULL) {
        LOGI("JNI_OnLoad: WhisperProgress class and callback methods successfully cached");
    } else {
        LOGW("JNI_OnLoad: WhisperProgress class not found yet; will lazy load");
    }
    return JNI_VERSION_1_6;
}

static void ensure_progress_class(JNIEnv *env) {
    if (env == NULL) return;
    if (__atomic_load_n(&g_progress_cls, __ATOMIC_ACQUIRE) == NULL) {
        cache_progress_methods(env);
    }
}

/**
 * Attaches the calling thread if needed and returns its JNIEnv.
 * Sets *needs_detach when the caller must detach again afterwards.
 */
static JNIEnv * acquire_env(int *needs_detach) {
    *needs_detach = 0;
    if (g_jvm == NULL) return NULL;

    JNIEnv * env = NULL;
    const int res = (*g_jvm)->GetEnv(g_jvm, (void **) &env, JNI_VERSION_1_6);
    if (res == JNI_EDETACHED) {
        if ((*g_jvm)->AttachCurrentThread(g_jvm, &env, NULL) != JNI_OK) return NULL;
        *needs_detach = 1;
    } else if (res != JNI_OK || env == NULL) {
        return NULL;
    }
    return env;
}

static void release_env(int needs_detach) {
    if (needs_detach) (*g_jvm)->DetachCurrentThread(g_jvm);
}

static void progress_cb(struct whisper_context * ctx, struct whisper_state * state, int progress, void * user_data) {
    UNUSED(ctx); UNUSED(state); UNUSED(user_data);

    int needs_detach = 0;
    JNIEnv * env = acquire_env(&needs_detach);
    if (env == NULL) return;

    ensure_progress_class(env);

    if (g_progress_cls != NULL && g_progress_mid != NULL) {
        (*env)->CallStaticVoidMethod(env, g_progress_cls, g_progress_mid, (jint) progress);
        if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
    }
    release_env(needs_detach);
}

static void new_segment_cb(struct whisper_context * ctx, struct whisper_state * state, int n_new, void * user_data) {
    UNUSED(ctx);
    UNUSED(user_data);
    if (n_new <= 0) return;

    int needs_detach = 0;
    JNIEnv * env = acquire_env(&needs_detach);
    if (env == NULL) return;

    ensure_progress_class(env);

    if (g_progress_cls != NULL && g_new_segment_mid != NULL) {
        int total_segments = whisper_full_n_segments_from_state(state);
        int start_idx = total_segments - n_new;
        if (start_idx < 0) start_idx = 0;

        for (int i = start_idx; i < total_segments; ++i) {
            const char * text = whisper_full_get_segment_text_from_state(state, i);
            int64_t t0 = whisper_full_get_segment_t0_from_state(state, i);
            int64_t t1 = whisper_full_get_segment_t1_from_state(state, i);
            if (text != NULL) {
                jstring jtext = (*env)->NewStringUTF(env, text);
                if (jtext == NULL) {
                    (*env)->ExceptionClear(env);
                    continue;
                }
                (*env)->CallStaticVoidMethod(env, g_progress_cls, g_new_segment_mid, jtext, (jlong)t0, (jlong)t1);
                if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
                (*env)->DeleteLocalRef(env, jtext);
            }
        }
    }

    release_env(needs_detach);
}

// ---------------------------------------------------------------------------
// CPU-only context params. ggml defaults to use_gpu=true and flash_attn=true,
// but on Android there is no GPU backend and the CPU flash-attention kernel is
// dramatically slower than the regular attention path (measured: 4.1 s of the
// 6.3 s tiny encode was FLASH_ATTN_EXT on Snapdragon 8 Elite).
// ---------------------------------------------------------------------------
static struct whisper_context_params default_cpu_params(void) {
    struct whisper_context_params p = whisper_context_default_params();
    p.use_gpu    = false;
    p.flash_attn = false;
    return p;
}

// ---------------------------------------------------------------------------
// Filepath-based model loader (the main path: downloaded model in internal storage)
// ---------------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_initContext(
        JNIEnv *env, jobject thiz, jstring model_path_str) {
    UNUSED(thiz);
    const char *model_path_chars = (*env)->GetStringUTFChars(env, model_path_str, NULL);
    if (model_path_chars == NULL) return 0;
    LOGI("Loading model from path '%s'", model_path_chars);
    struct whisper_context *context = whisper_init_from_file_with_params(
            model_path_chars, default_cpu_params());
    (*env)->ReleaseStringUTFChars(env, model_path_str, model_path_chars);
    return (jlong) context;
}

JNIEXPORT void JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_freeContext(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    whisper_free(context);
}

struct jni_whisper_params {
    int num_threads;
    const char *language;
    const char *initial_prompt;
    float temperature;
    float temperature_inc;
    float entropy_thold;
    float logprob_thold;
    float no_speech_thold;
    bool no_context;
    bool single_segment;
    bool translate;
    bool suppress_blank;
    bool suppress_non_speech_tokens;
};

/**
 * Shared whisper_full invocation: params, timings reset/dump.
 * Returns 0 on success, the whisper_full return code on failure, or 1 when the
 * run was aborted by the user (partial segments are still readable).
 */
static int run_whisper_full(struct whisper_context *context, const float *samples, int n_samples,
                            const struct jni_whisper_params *p) {
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime = false;        // we render in UI instead
    params.print_progress = false;
    params.print_timestamps = false;      // segment timestamps live in the JNI query APIs
    params.print_special = false;
    params.translate = p->translate;
    params.language = (p->language && p->language[0]) ? p->language : "en";
    params.n_threads = p->num_threads > 0 ? p->num_threads : 4;
    params.temperature = p->temperature;
    params.temperature_inc = p->temperature_inc;
    params.entropy_thold = p->entropy_thold;
    params.logprob_thold = p->logprob_thold;
    params.no_speech_thold = p->no_speech_thold;
    params.no_context = p->no_context;
    params.n_max_text_ctx = p->no_context ? 0 : 16384;
    params.single_segment = p->single_segment;
    params.suppress_blank = p->suppress_blank;
    params.suppress_nst = p->suppress_non_speech_tokens;
    params.initial_prompt = (p->initial_prompt && p->initial_prompt[0]) ? p->initial_prompt : NULL;

    // Fresh run: clear any stale cancel request and wire the callbacks.
    __atomic_store_n(&g_abort_transcribe, 0, __ATOMIC_RELAXED);
    params.progress_callback = progress_cb;
    params.progress_callback_user_data = NULL;
    params.new_segment_callback = new_segment_cb;
    params.new_segment_callback_user_data = NULL;
    params.abort_callback = abort_cb;
    params.abort_callback_user_data = NULL;

    // Use standard full encoder context (0 = default 1500) to ensure perfect
    // positional embeddings and timestamp decoding without repetition loops.
    params.audio_ctx = 0;
    params.offset_ms = 0;

    whisper_reset_timings(context);

    LOGI("About to run whisper_full (samples=%d, threads=%d, lang=%s, temp=%.2f, temp_inc=%.2f, no_ctx=%d, prompt='%s')",
         n_samples, params.n_threads, params.language, params.temperature, params.temperature_inc,
         (int)params.no_context, params.initial_prompt ? params.initial_prompt : "");
    int rc = whisper_full(context, params, samples, n_samples);
    const int aborted = __atomic_exchange_n(&g_abort_transcribe, 0, __ATOMIC_RELAXED);
    if (rc != 0) {
        if (aborted) {
            LOGI("whisper_full aborted by user (rc=%d, partial segments kept)", rc);
            return 1;
        }
        LOGE("whisper_full failed with code %d", rc);
        return rc;
    }
    whisper_print_timings(context);
    return 0;
}

JNIEXPORT void JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_cancelTranscription(
        JNIEnv *env, jobject thiz) {
    UNUSED(env); UNUSED(thiz);
    // Cancels whichever phase is running: the caller does not know whether the
    // run it is aborting is still decoding or already transcribing.
    __atomic_store_n(&g_abort_transcribe, 1, __ATOMIC_RELAXED);
    __atomic_store_n(&g_abort_decode, 1, __ATOMIC_RELAXED);
    LOGI("Cancel requested - aborting decode loop / next graph boundary");
}

JNIEXPORT jint JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_fullTranscribe(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint num_threads, jstring language,
        jstring initial_prompt, jfloat temperature, jfloat temperature_inc, jfloat entropy_thold,
        jfloat logprob_thold, jfloat no_speech_thold, jboolean no_context, jboolean single_segment,
        jboolean translate, jboolean suppress_blank, jboolean suppress_non_speech_tokens,
        jfloatArray audio_data) {
    UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    if (context == NULL || audio_data == NULL) return -1;

    jfloat *audio_data_arr = (*env)->GetFloatArrayElements(env, audio_data, NULL);
    if (audio_data_arr == NULL) return -1;
    const jsize audio_data_length = (*env)->GetArrayLength(env, audio_data);
    const char *lang_chars = language != NULL ? (*env)->GetStringUTFChars(env, language, NULL) : NULL;
    const char *prompt_chars = initial_prompt != NULL ? (*env)->GetStringUTFChars(env, initial_prompt, NULL) : NULL;

    struct jni_whisper_params p = {
        .num_threads = num_threads,
        .language = lang_chars,
        .initial_prompt = prompt_chars,
        .temperature = temperature,
        .temperature_inc = temperature_inc,
        .entropy_thold = entropy_thold,
        .logprob_thold = logprob_thold,
        .no_speech_thold = no_speech_thold,
        .no_context = (bool) no_context,
        .single_segment = (bool) single_segment,
        .translate = (bool) translate,
        .suppress_blank = (bool) suppress_blank,
        .suppress_non_speech_tokens = (bool) suppress_non_speech_tokens,
    };

    if (g_jvm == NULL) (*env)->GetJavaVM(env, &g_jvm);
    const int rc = run_whisper_full(context, audio_data_arr, audio_data_length, &p);

    if (prompt_chars) (*env)->ReleaseStringUTFChars(env, initial_prompt, prompt_chars);
    if (lang_chars) (*env)->ReleaseStringUTFChars(env, language, lang_chars);
    (*env)->ReleaseFloatArrayElements(env, audio_data, audio_data_arr, JNI_ABORT);
    return (jint) rc;
}

/**
 * Streaming entry point: reads a mono PCM16 16 kHz WAV file (as written by
 * AudioDecoder.decodeToWavFile) directly in native memory - the Java heap only
 * ever holds a path string, so hour-long recordings no longer OOM the VM.
 *
 * Returns 0 on success, 1 when aborted by the user, negative on failure.
 */
JNIEXPORT jint JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_fullTranscribeFile(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint num_threads, jstring language,
        jstring initial_prompt, jfloat temperature, jfloat temperature_inc, jfloat entropy_thold,
        jfloat logprob_thold, jfloat no_speech_thold, jboolean no_context, jboolean single_segment,
        jboolean translate, jboolean suppress_blank, jboolean suppress_non_speech_tokens,
        jstring wav_path) {
    UNUSED(thiz);

    // Declared up front so the error paths below never jump over an initializer.
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    FILE *f = NULL;
    float *samples = NULL;
    uint16_t *blk = NULL;
    char id[5] = {0};
    uint32_t chunk_size = 0;
    int channels = 1, sample_rate = 16000, bits = 16;
    bool fmt_ok = false, has_data = false;
    int64_t data_size = 0;
    int64_t file_size = 0;
    size_t n_samples = 0;
    size_t off = 0;
    int rc = -1;

    const char *lang_chars = language != NULL ? (*env)->GetStringUTFChars(env, language, NULL) : NULL;
    const char *prompt_chars = initial_prompt != NULL ? (*env)->GetStringUTFChars(env, initial_prompt, NULL) : NULL;
    const char *path = (*env)->GetStringUTFChars(env, wav_path, NULL);
    if (path == NULL) goto cleanup;
    if (context == NULL) { LOGE("fullTranscribeFile: null context"); goto cleanup; }

    if (g_jvm == NULL) (*env)->GetJavaVM(env, &g_jvm);

    f = fopen(path, "rb");
    if (f == NULL) {
        LOGE("fullTranscribeFile: cannot open %s", path);
        goto cleanup;
    }

    if (fseek(f, 0, SEEK_END) != 0) { LOGE("fullTranscribeFile: seek failed"); goto cleanup; }
    file_size = (int64_t) ftell(f);
    if (fseek(f, 0, SEEK_SET) != 0) { LOGE("fullTranscribeFile: rewind failed"); goto cleanup; }

    // ---- minimal RIFF/WAVE parser ----
    if (fread(id, 1, 4, f) != 4 || memcmp(id, "RIFF", 4) != 0) { LOGE("fullTranscribeFile: not a RIFF file"); goto cleanup; }
    if (fread(&chunk_size, 4, 1, f) != 1) { LOGE("fullTranscribeFile: truncated header"); goto cleanup; }
    if (fread(id, 1, 4, f) != 4 || memcmp(id, "WAVE", 4) != 0) { LOGE("fullTranscribeFile: not a WAVE file"); goto cleanup; }

    while (fread(id, 1, 4, f) == 4) {
        if (fread(&chunk_size, 4, 1, f) != 1) break;
        if (memcmp(id, "fmt ", 4) == 0) {
            long fmt_body = ftell(f);
            uint16_t audio_format = 0, num_ch = 1, bps = 16;
            uint32_t sr = 16000;
            if (chunk_size >= 16) {
                if (fread(&audio_format, 2, 1, f) != 1 ||
                    fread(&num_ch, 2, 1, f) != 1 ||
                    fread(&sr, 4, 1, f) != 1 ||
                    fseek(f, 6, SEEK_CUR) != 0 ||   // byteRate + blockAlign (unused)
                    fread(&bps, 2, 1, f) != 1) {
                    LOGE("fullTranscribeFile: truncated fmt chunk");
                    goto cleanup;
                }
                if (audio_format == 1) { // PCM
                    channels = num_ch;
                    sample_rate = (int) sr;
                    bits = bps;
                    fmt_ok = true;
                }
            }
            if (fseek(f, fmt_body + (long) chunk_size + (long)(chunk_size & 1), SEEK_SET) != 0) goto cleanup;
        } else if (memcmp(id, "data", 4) == 0) {
            data_size = (int64_t) chunk_size;
            has_data = true;
            break;
        } else {
            if (fseek(f, (long) chunk_size + (long)(chunk_size & 1), SEEK_CUR) != 0) break;
        }
    }

    if (!fmt_ok) { LOGE("fullTranscribeFile: missing/unsupported fmt chunk"); goto cleanup; }
    if (!has_data || data_size <= 0) { LOGE("fullTranscribeFile: empty data chunk"); goto cleanup; }
    if (bits != 16 || channels != 1) {
        LOGE("fullTranscribeFile: expected mono PCM16, got ch=%d bits=%d", channels, bits);
        goto cleanup;
    }
    if (sample_rate != 16000) {
        LOGW("fullTranscribeFile: sample rate %d != 16000 (decoder should have resampled)", sample_rate);
    }

    // A corrupt header must never drive the allocation: clamp the declared data
    // size to what is actually left in the file.
    {
        const int64_t remaining = file_size - (int64_t) ftell(f);
        if (remaining > 0 && data_size > remaining) {
            LOGW("fullTranscribeFile: data chunk declares %" PRId64 " B but only %" PRId64 " B remain; clamping",
                 data_size, remaining);
            data_size = remaining;
        }
    }

    // ---- PCM16 -> float, read in 1M-sample blocks so peak native memory stays ~audio size ----
    n_samples = (size_t) (data_size / 2);
    if (n_samples == 0) { LOGE("fullTranscribeFile: no samples"); goto cleanup; }

    samples = (float *) malloc(n_samples * sizeof(float));
    blk = (uint16_t *) malloc((1 << 20) * sizeof(uint16_t));
    if (samples == NULL || blk == NULL) {
        LOGE("fullTranscribeFile: out of native memory (%zu samples)", n_samples);
        goto cleanup;
    }

    while (off < n_samples) {
        const size_t want = ((size_t) (1 << 20) < n_samples - off) ? (size_t) (1 << 20) : (n_samples - off);
        const size_t got = fread(blk, sizeof(uint16_t), want, f);
        for (size_t i = 0; i < got; ++i) {
            samples[off + i] = (float) ((int16_t) blk[i]) / 32768.0f;
        }
        off += got;
        if (got < want) break; // truncated file: transcribe what we have
    }
    free(blk);
    blk = NULL;
    fclose(f);
    f = NULL;

    LOGI("fullTranscribeFile: %s sr=%d ch=%d samples=%zu", path, sample_rate, channels, off);
    {
        struct jni_whisper_params p = {
            .num_threads = num_threads,
            .language = lang_chars,
            .initial_prompt = prompt_chars,
            .temperature = temperature,
            .temperature_inc = temperature_inc,
            .entropy_thold = entropy_thold,
            .logprob_thold = logprob_thold,
            .no_speech_thold = no_speech_thold,
            .no_context = (bool) no_context,
            .single_segment = (bool) single_segment,
            .translate = (bool) translate,
            .suppress_blank = (bool) suppress_blank,
            .suppress_non_speech_tokens = (bool) suppress_non_speech_tokens,
        };
        rc = run_whisper_full(context, samples, (int) off, &p);
    }

cleanup:
    free(samples);
    free(blk);
    if (f) fclose(f);
    if (prompt_chars) (*env)->ReleaseStringUTFChars(env, initial_prompt, prompt_chars);
    if (path) (*env)->ReleaseStringUTFChars(env, wav_path, path);
    if (lang_chars) (*env)->ReleaseStringUTFChars(env, language, lang_chars);
    return (jint) rc;
}

JNIEXPORT jint JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_getTextSegmentCount(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    if (context == NULL) return 0;
    return whisper_full_n_segments(context);
}

JNIEXPORT jstring JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_getTextSegment(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    if (context == NULL) return (*env)->NewStringUTF(env, "");
    // whisper_full_get_segment_* index result_all with std::vector::operator[],
    // which does not bounds check: an out-of-range index is an OOB read feeding
    // a garbage pointer straight into NewStringUTF.
    if (index < 0 || index >= whisper_full_n_segments(context)) {
        LOGW("getTextSegment: index %d out of range (n=%d)", (int)index, whisper_full_n_segments(context));
        return (*env)->NewStringUTF(env, "");
    }
    const char *text = whisper_full_get_segment_text(context, index);
    return (*env)->NewStringUTF(env, text != NULL ? text : "");
}

JNIEXPORT jlong JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_getTextSegmentT0(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(env); UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    if (context == NULL) return 0;
    if (index < 0 || index >= whisper_full_n_segments(context)) return 0;
    return whisper_full_get_segment_t0(context, index);
}

JNIEXPORT jlong JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_getTextSegmentT1(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(env); UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    if (context == NULL) return 0;
    if (index < 0 || index >= whisper_full_n_segments(context)) return 0;
    return whisper_full_get_segment_t1(context, index);
}

// Diagnostic: returns which ggml CPU kernels are active on this device and the
// actual architecture of the loaded model (layers/heads/ctx/mels). Lets us see
// at a glance whether e.g. dotprod/SVE/llamafile paths are in use and whether
// the model really is a tiny (4 encoder layers) rather than something else.
JNIEXPORT jstring JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_getModelInfo(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    if (context == NULL) return (*env)->NewStringUTF(env, "");

    char buf[1024];
    snprintf(buf, sizeof(buf),
        "model: enc_layers=%d state=%d heads=%d a_ctx=%d mels=%d text_layers=%d mel_len=%d | "
        "cpu: neon=%d fp16_va=%d dotprod=%d matmul_int8=%d sve=%d sme=%d llamafile=%d",
        whisper_model_n_audio_layer(context),
        whisper_model_n_audio_state(context),
        whisper_model_n_audio_head(context),
        whisper_model_n_audio_ctx(context),
        whisper_model_n_mels(context),
        whisper_model_n_text_layer(context),
        whisper_n_len(context),
        ggml_cpu_has_neon(),
        ggml_cpu_has_fp16_va(),
        ggml_cpu_has_dotprod(),
        ggml_cpu_has_matmul_int8(),
        ggml_cpu_has_sve(),
        ggml_cpu_has_sme(),
        ggml_cpu_has_llamafile());

    // Append the per-op breakdown of the last slow graph compute (set by the
    // ggml-cpu instrumentation in ggml-cpu.c, empty until the first run).
    const char * perf = ggml_perf_last_dump();
    if (perf != NULL && perf[0] != '\0') {
        strncat(buf, "\n", sizeof(buf) - strlen(buf) - 1);
        strncat(buf, perf, sizeof(buf) - strlen(buf) - 1);
    }

    return (*env)->NewStringUTF(env, buf);
}

// Forward-declared in C linkage by whisper_timings_bridge.cpp. We use a tiny
// C++ wrapper around whisper_get_timings() because that function returns memory
// allocated with `new` and the public whisper.h does not expose a deleter; only
// the C++ side can safely `delete` it.
extern void whisper_timings_to_csv(struct whisper_context * ctx, char * out, int out_len);

// Returns a one-line CSV of phase timings (in ms) extracted from the last
// whisper_full run: "sample_ms,encode_ms,decode_ms,batchd_ms,prompt_ms". Caller
// parses it into a WhisperTimings data class for live UI rendering, so the user
// doesn't have to dig through logcat.
JNIEXPORT jstring JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_getTimings(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    if (context == NULL) return (*env)->NewStringUTF(env, "");
    char buf[160] = {0};
    whisper_timings_to_csv(context, buf, (int) sizeof(buf));
    return (*env)->NewStringUTF(env, buf);
}

static void native_decode_on_progress(float p, void* user_data) {
    UNUSED(user_data);
    int int_p = (int)(p * 100.0f);
    if (int_p < 0) int_p = 0;
    if (int_p > 100) int_p = 100;

    int needs_detach = 0;
    JNIEnv * env = acquire_env(&needs_detach);
    if (env == NULL) return;

    ensure_progress_class(env);

    if (g_progress_cls != NULL && g_decode_progress_mid != NULL) {
        (*env)->CallStaticVoidMethod(env, g_progress_cls, g_decode_progress_mid, (jint) int_p);
        if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
    } else if (g_progress_cls != NULL && g_progress_mid != NULL) {
        (*env)->CallStaticVoidMethod(env, g_progress_cls, g_progress_mid, (jint) int_p);
        if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
    }

    release_env(needs_detach);
}

static bool native_decode_is_cancelled(void* user_data) {
    UNUSED(user_data);
    return __atomic_load_n(&g_abort_decode, __ATOMIC_RELAXED) != 0;
}

JNIEXPORT jint JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_nativeDecodeAudioFd(
        JNIEnv *env, jobject thiz,
        jint fd, jlong offset, jlong length, jstring out_wav_path) {
    UNUSED(thiz);
    if (g_jvm == NULL) (*env)->GetJavaVM(env, &g_jvm);
    ensure_progress_class(env);
    __atomic_store_n(&g_abort_decode, 0, __ATOMIC_RELAXED);
    const char *c_out_wav = (*env)->GetStringUTFChars(env, out_wav_path, NULL);
    if (c_out_wav == NULL) return -1;

    LOGI("[C-NDK] Entering nativeDecodeAudioFd: fd=%d offset=%" PRId64 " length=%" PRId64 " out=%s",
         fd, (int64_t)offset, (int64_t)length, c_out_wav);

    int res = native_decode_to_wav(
        (int)fd,
        (int64_t)offset,
        (int64_t)length,
        c_out_wav,
        native_decode_on_progress,
        native_decode_is_cancelled,
        NULL
    );

    LOGI("[C-NDK] Exiting nativeDecodeAudioFd with return code: %d", res);

    (*env)->ReleaseStringUTFChars(env, out_wav_path, c_out_wav);
    return (jint)res;
}

JNIEXPORT jfloatArray JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_extractSegmentEmbedding(
        JNIEnv *env, jobject thiz,
        jfloatArray audio_data,
        jlong seg_t0,
        jlong seg_t1) {
    UNUSED(thiz);
    if (audio_data == NULL || seg_t1 <= seg_t0) {
        return (*env)->NewFloatArray(env, 0);
    }

    jsize sample_count = (*env)->GetArrayLength(env, audio_data);
    jfloat *c_samples = (*env)->GetFloatArrayElements(env, audio_data, NULL);
    if (c_samples == NULL) {
        return (*env)->NewFloatArray(env, 0);
    }

    float embedding[SPEAKER_EMBED_DIM];
    bool ok = extract_segment_embedding(
        (const float *)c_samples,
        (int)sample_count,
        (int64_t)seg_t0,
        (int64_t)seg_t1,
        embedding
    );

    (*env)->ReleaseFloatArrayElements(env, audio_data, c_samples, JNI_ABORT);

    if (!ok) {
        return (*env)->NewFloatArray(env, 0);
    }

    jfloatArray result = (*env)->NewFloatArray(env, SPEAKER_EMBED_DIM);
    if (result != NULL) {
        (*env)->SetFloatArrayRegion(env, result, 0, SPEAKER_EMBED_DIM, embedding);
    }
    return result;
}
