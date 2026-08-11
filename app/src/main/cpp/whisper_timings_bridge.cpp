// whisper_timings_bridge.cpp
//
// Why this file exists:
//   whisper_get_timings() returns a `struct whisper_timings *` allocated with
//   C++ `new`, but the public whisper.h header does NOT expose a corresponding
//   `whisper_free_timings`. So a .c caller has no portable way to free the
//   returned struct: `free()` on `new`-allocated memory is UB, and `delete` is
//   not a C keyword.
//
// Solution: this tiny C++ translation unit wraps the call. It is compiled by
// the C++ compiler (because of the .cpp extension), can legally use `delete`,
// and exposes its single function with C linkage (`extern "C"`) so the .c
// JNI bridge can call it by simple name without mangling.
//
// All other JNI bridges stay in whisper_jni.c (C) — this file only owns the
// one spot that genuinely needs C++.

#include "whisper.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fills [out] with a CSV string of the per-phase timings collected by the
 * last whisper_full() call on [ctx]. Order:
 *   "sample_ms,encode_ms,decode_ms,batchd_ms,prompt_ms"
 *
 * If no run has produced timings yet, writes "0,0,0,0,0".
 *
 * No more than [out_len - 1] characters plus the null terminator are written.
 */
void whisper_timings_to_csv(struct whisper_context * ctx, char * out, int out_len) {
    if (out == nullptr || out_len <= 0) return;

    struct whisper_timings * t = whisper_get_timings(ctx);
    if (t == nullptr) {
        snprintf(out, (size_t) out_len, "0,0,0,0,0");
        return;
    }

    snprintf(out, (size_t) out_len, "%.2f,%.2f,%.2f,%.2f,%.2f",
             t->sample_ms, t->encode_ms, t->decode_ms, t->batchd_ms, t->prompt_ms);
    delete t;
}

#ifdef __cplusplus
} // extern "C"
#endif