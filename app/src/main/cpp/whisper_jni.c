// JNI bridge between Kotlin (com.emilio.whispert.whisper.WhisperLib) and whisper.cpp.
//
// Adapted from the upstream whisper.cpp Android example (examples/whisper.android/lib/src/main/jni/whisper/jni.c)
// with JNI function names rewritten for our package: com.emilio.whispert.whisper.WhisperLib.
//
// Exposes:
//   - initContext(modelPath)            : load model from a filesystem path (downloaded to internal storage)
//   - initContextFromAsset(mgr, path)   : load model from a packaged asset
//   - initContextFromInputStream(stream): load model from a generic Java InputStream (e.g. SAF)
//   - freeContext(ptr)
//   - fullTranscribe(ptr, threads, audioData)
//   - getTextSegmentCount/T0/T1(ptr, i)
//   - getSystemInfo()
//
// IMPORTANT: whisper.cpp is NOT thread-safe. Every native call must be dispatched
// from a single dedicated thread (handled in WhisperContext.kt).
#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <stdlib.h>
#include <string.h>
#include "whisper.h"
#include "ggml.h"

#define UNUSED(x) (void)(x)
#define TAG "whisper-jni"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// InputStream-based model loader (lets us load a model from a SAF Uri)
// ---------------------------------------------------------------------------
struct input_stream_context {
    size_t offset;
    JNIEnv * env;
    jobject input_stream;

    jmethodID mid_available;
    jmethodID mid_read;
};

static size_t inputStreamRead(void * ctx, void * output, size_t read_size) {
    struct input_stream_context* is = (struct input_stream_context*)ctx;

    jint avail_size = (*is->env)->CallIntMethod(is->env, is->input_stream, is->mid_available);
    jint size_to_copy = read_size < (size_t)avail_size ? (jint)read_size : avail_size;
    if (size_to_copy < 0) size_to_copy = 0;

    jbyteArray byte_array = (*is->env)->NewByteArray(is->env, size_to_copy);
    if (byte_array == NULL) {
        return 0;
    }

    jint n_read = (*is->env)->CallIntMethod(is->env, is->input_stream, is->mid_read, byte_array, 0, size_to_copy);
    if (n_read < 0) n_read = 0;

    jbyte* bytes = (*is->env)->GetByteArrayElements(is->env, byte_array, NULL);
    memcpy(output, bytes, (size_t)n_read);
    (*is->env)->ReleaseByteArrayElements(is->env, byte_array, bytes, JNI_ABORT);
    (*is->env)->DeleteLocalRef(is->env, byte_array);

    is->offset += (size_t)n_read;
    return (size_t)n_read;
}

static bool inputStreamEof(void * ctx) {
    struct input_stream_context* is = (struct input_stream_context*)ctx;
    jint result = (*is->env)->CallIntMethod(is->env, is->input_stream, is->mid_available);
    return result <= 0;
}

static void inputStreamClose(void * ctx) {
    UNUSED(ctx);
}

JNIEXPORT jlong JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_initContextFromInputStream(
        JNIEnv *env, jobject thiz, jobject input_stream) {
    UNUSED(thiz);

    struct whisper_context *context = NULL;
    struct whisper_model_loader loader = {};
    struct input_stream_context inp_ctx = {};

    inp_ctx.offset = 0;
    inp_ctx.env = env;
    inp_ctx.input_stream = input_stream;

    jclass cls = (*env)->GetObjectClass(env, input_stream);
    inp_ctx.mid_available = (*env)->GetMethodID(env, cls, "available", "()I");
    inp_ctx.mid_read = (*env)->GetMethodID(env, cls, "read", "([BII)I");

    loader.context = &inp_ctx;
    loader.read = inputStreamRead;
    loader.eof = inputStreamEof;
    loader.close = inputStreamClose;

    if (inputStreamEof(loader.context)) {
        LOGE("Input stream already at EOF before loading model");
        return 0;
    }

    context = whisper_init(&loader);
    return (jlong) context;
}

// ---------------------------------------------------------------------------
// Asset-based model loader (for models bundled inside APK assets)
// ---------------------------------------------------------------------------
static size_t asset_read(void *ctx, void *output, size_t read_size) {
    return AAsset_read((AAsset *) ctx, output, read_size);
}

static bool asset_is_eof(void *ctx) {
    return AAsset_getRemainingLength64((AAsset *) ctx) <= 0;
}

static void asset_close(void *ctx) {
    AAsset_close((AAsset *) ctx);
}

static struct whisper_context *whisper_init_from_asset(
        JNIEnv *env,
        jobject assetManager,
        const char *asset_path
) {
    LOGI("Loading model from asset '%s'", asset_path);
    AAssetManager *asset_manager = AAssetManager_fromJava(env, assetManager);
    AAsset *asset = AAssetManager_open(asset_manager, asset_path, AASSET_MODE_STREAMING);
    if (!asset) {
        LOGW("Failed to open asset '%s'", asset_path);
        return NULL;
    }

    whisper_model_loader loader = {
            .context = asset,
            .read = &asset_read,
            .eof = &asset_is_eof,
            .close = &asset_close
    };

    return whisper_init_with_params(&loader, whisper_context_default_params());
}

JNIEXPORT jlong JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_initContextFromAsset(
        JNIEnv *env, jobject thiz, jobject assetManager, jstring asset_path_str) {
    UNUSED(thiz);
    const char *asset_path_chars = (*env)->GetStringUTFChars(env, asset_path_str, NULL);
    struct whisper_context *context = whisper_init_from_asset(env, assetManager, asset_path_chars);
    (*env)->ReleaseStringUTFChars(env, asset_path_str, asset_path_chars);
    return (jlong) context;
}

// ---------------------------------------------------------------------------
// Filepath-based model loader (the main path: downloaded model in internal storage)
// ---------------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_initContext(
        JNIEnv *env, jobject thiz, jstring model_path_str) {
    UNUSED(thiz);
    const char *model_path_chars = (*env)->GetStringUTFChars(env, model_path_str, NULL);
    LOGI("Loading model from path '%s'", model_path_chars);
    struct whisper_context *context = whisper_init_from_file_with_params(
            model_path_chars, whisper_context_default_params());
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

JNIEXPORT void JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_fullTranscribe(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint num_threads, jstring language, jfloatArray audio_data) {
    UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    jfloat *audio_data_arr = (*env)->GetFloatArrayElements(env, audio_data, NULL);
    const jsize audio_data_length = (*env)->GetArrayLength(env, audio_data);
    const char *lang_chars = language != NULL ? (*env)->GetStringUTFChars(env, language, NULL) : NULL;

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime = false;        // we render in UI instead
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.translate = false;
    params.language = lang_chars ? lang_chars : "en";
    params.n_threads = num_threads;
    params.offset_ms = 0;
    params.no_context = true;
    params.single_segment = false;

    whisper_reset_timings(context);

    LOGI("About to run whisper_full (samples=%d, threads=%d, lang=%s)",
         audio_data_length, num_threads, params.language);
    int rc = whisper_full(context, params, audio_data_arr, audio_data_length);
    if (rc != 0) {
        LOGE("whisper_full failed with code %d", rc);
    } else {
        whisper_print_timings(context);
    }

    if (lang_chars) (*env)->ReleaseStringUTFChars(env, language, lang_chars);
    (*env)->ReleaseFloatArrayElements(env, audio_data, audio_data_arr, JNI_ABORT);
}

JNIEXPORT jint JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_getTextSegmentCount(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    return whisper_full_n_segments(context);
}

JNIEXPORT jstring JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_getTextSegment(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    const char *text = whisper_full_get_segment_text(context, index);
    return (*env)->NewStringUTF(env, text);
}

JNIEXPORT jlong JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_getTextSegmentT0(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    return whisper_full_get_segment_t0(context, index);
}

JNIEXPORT jlong JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_getTextSegmentT1(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(thiz);
    struct whisper_context *context = (struct whisper_context *) context_ptr;
    return whisper_full_get_segment_t1(context, index);
}

JNIEXPORT jstring JNICALL
Java_com_emilio_whispert_whisper_WhisperLib_00024Companion_getSystemInfo(
        JNIEnv *env, jobject thiz) {
    UNUSED(thiz);
    const char *sysinfo = whisper_print_system_info();
    return (*env)->NewStringUTF(env, sysinfo);
}