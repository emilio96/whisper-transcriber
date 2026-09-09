# Local patches to the `whisper.cpp` submodule

The submodule `third_party/whisper.cpp` is pinned at `306c88f4` (v1.8.5-286),
but the working copy carries local edits that the pin alone does not describe.
The app depends on them directly: `app/src/main/cpp/whisper_jni.c` calls
`ggml_perf_last_dump()`, which exists only here.

So the real pin is "`306c88f4` + the patch in this directory". Without the
patch the native build fails to link.

## Reapplying after a fresh clone or a `submodule update`

```sh
git submodule update --init --recursive
git -C third_party/whisper.cpp apply ../patches/whisper-cpp-306c88f4-perf-dump.patch
```

`git apply` refuses to run twice on an already-patched tree, so a failure with
"patch does not apply" usually means it is already in place. Check with:

```sh
git -C third_party/whisper.cpp diff --stat
```

Three modified files and 133 added lines means the patch is applied.

## Regenerating the patch after editing the submodule

```sh
git -C third_party/whisper.cpp diff > third_party/patches/whisper-cpp-306c88f4-perf-dump.patch
```

Do this on every submodule edit: the checked-in patch is the only copy of that
work that lives in version control.

## What the patch contains

- `ggml/include/ggml-cpu.h` — declares `ggml_perf_last_dump()`.
- `ggml/src/ggml-cpu/ggml-cpu.c` — per-op timing instrumentation behind it.
- `src/whisper.cpp` — hooks the dump into the whisper graph compute.

All of it is diagnostic instrumentation, consumed by `getModelInfo` and shown
in the *Statistics & Diagnostics* panel. Nothing in the transcription path
depends on it for correctness, only for reporting.
