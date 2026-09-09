# CMake build for the whisper.cpp JNI bridge.
# Compiles whisper.cpp + ggml (vendored as a git submodule under third_party/)
# into a shared library loaded by the Kotlin WhisperLib wrapper via System.loadLibrary.
#
# Only arm64-v8a is targeted (Gradle abiFilters), which covers effectively all
# modern Android devices we care about and keeps build time / APK size low.
cmake_minimum_required(VERSION 3.10)

project(whisper.cpp)

set(CMAKE_CXX_STANDARD 17)
add_link_options("-Wl,-z,max-page-size=16384")

# Path to the whisper.cpp submodule from this CMakeLists.txt.
#   this file: <root>/app/src/main/cpp/CMakeLists.txt
#   submodule : <root>/third_party/whisper.cpp
#   => 4 levels up from CMAKE_SOURCE_DIR reaches the project root.
set(WHISPER_LIB_DIR ${CMAKE_SOURCE_DIR}/../../../../third_party/whisper.cpp)

# Read whisper.cpp version from its top-level CMakeLists (for logging / branding).
file(READ "${WHISPER_LIB_DIR}/CMakeLists.txt" MAIN_CMAKE_CONTENT)
string(REGEX MATCH "project\\(\"whisper\\.cpp\" VERSION ([0-9]+\\.[0-9]+\\.[0-9]+)\\)" VERSION_MATCH "${MAIN_CMAKE_CONTENT}")
if(CMAKE_MATCH_1)
    set(WHISPER_VERSION ${CMAKE_MATCH_1})
else()
    set(WHISPER_VERSION "unknown")
endif()

message(STATUS " Whisper version: ${WHISPER_VERSION}")

set(
    SOURCE_FILES
    ${WHISPER_LIB_DIR}/src/whisper.cpp
    ${CMAKE_SOURCE_DIR}/whisper_jni.c
    ${CMAKE_SOURCE_DIR}/whisper_timings_bridge.cpp
    ${CMAKE_SOURCE_DIR}/native_audio_decoder.c
    ${CMAKE_SOURCE_DIR}/speaker_diarizer.c
)

find_library(LOG_LIB log)

include(FetchContent)

# ============================================================================
# ggml CPU backend tuning (perf-critical). These cache vars are picked up when
# ggml's CMakeLists runs during FetchContent_MakeAvailable below.
#
# 1. GGML_OPENMP=OFF: ggml defaults to OpenMP, but on Android the libomp path
#    is drastically slower than ggml's native threadpool (per-op team spawns +
#    poor scaling). The pool is the fast path used by llama.cpp on desktop.
#    This was the main cause of the ~12s tiny encode on Snapdragon 8 Elite.
# 2. GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16: compiles the quantized matmul
#    kernels with SDOT (ASIMDDP) + fp16 vector arithmetic. Without this,
#    ggml-cpu is built as plain armv8-a and every q5_0 matmul runs the slow
#    non-dotprod NEON fallback.
# 3. GGML_LLAMAFILE=ON: enables the optimized llamafile matmul kernels for
#    quantized weights (q4/q5/q8) on ARM too.
#
# NOTE ON THE ARCH BASELINE: ggml is a single shared target, so every consumer
# links the same kernels. An earlier revision built two whisper variants
# ("generic" + fp16) and selected between them at runtime, but both linked this
# one ggml, which means the "generic" fallback also emitted SDOT/fp16 and would
# still SIGILL on a CPU lacking them. The fallback was therefore fiction; we now
# ship a single library and check the CPU features up front in WhisperLib.kt,
# failing with a readable message instead of an illegal instruction.
set(GGML_OPENMP         OFF                           CACHE BOOL   "" FORCE)
set(GGML_LLAMAFILE      ON                            CACHE BOOL   "" FORCE)
set(GGML_CPU_ARM_ARCH   "armv8.2-a+dotprod+fp16"      CACHE STRING "" FORCE)
# ============================================================================

# ggml is vendored inside the whisper.cpp submodule.
FetchContent_Declare(ggml SOURCE_DIR ${WHISPER_LIB_DIR}/ggml)
FetchContent_MakeAvailable(ggml)

add_library(whisper SHARED ${SOURCE_FILES})

target_compile_definitions(whisper PUBLIC GGML_USE_CPU)
target_compile_definitions(whisper PRIVATE WHISPER_VERSION="${WHISPER_VERSION}")

# The whisper.cpp sources themselves also benefit from fp16 vector arithmetic.
target_compile_options(whisper PRIVATE -march=armv8.2-a+dotprod+fp16)

# ALWAYS optimize, regardless of the Gradle variant: Android Studio installs
# Debug APKs, and at -O0 ggml's NEON intrinsics degrade into helper-function
# calls (10-50x slower matmuls). This was the single biggest performance bug:
# the whole encoder was running unoptimized.
target_compile_options(whisper PRIVATE -O3)
target_compile_options(whisper PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
target_compile_options(whisper PRIVATE -ffunction-sections -fdata-sections)

# ThinLTO has to be passed at COMPILE time as well, otherwise there is no
# bitcode for the linker to work with and the link-time -flto is a no-op (which
# is what happened before: the flag was set on the link line only).
target_compile_options(whisper PRIVATE -flto=thin)
target_link_options(whisper PRIVATE -flto=thin)

target_link_options(whisper PRIVATE -Wl,--gc-sections)
target_link_options(whisper PRIVATE -Wl,--exclude-libs,ALL)

# CRITICAL: the -O3 flags below must cover the ggml-cpu target too.
# ggml-cpu (libggml-cpu.so) holds ALL the matmul/vec_dot kernels
# (quants.c, ops.cpp) - with the Gradle Debug build it is compiled with no
# -O flag (i.e. -O0) unless we set it here, and at -O0 the NEON/SDOT
# intrinsics run 5-10x slower (register spills, no scheduling). This was
# the root cause of the ~12s tiny encode all along.
target_compile_options(ggml      PRIVATE -O3)
target_compile_options(ggml-base PRIVATE -O3)
target_compile_options(ggml-cpu  PRIVATE -O3)

target_link_libraries(whisper ${LOG_LIB} android mediandk ggml)

include_directories(${WHISPER_LIB_DIR})
include_directories(${WHISPER_LIB_DIR}/src)
include_directories(${WHISPER_LIB_DIR}/include)
include_directories(${WHISPER_LIB_DIR}/ggml/include)
include_directories(${WHISPER_LIB_DIR}/ggml/src)
include_directories(${WHISPER_LIB_DIR}/ggml/src/ggml-cpu)
