# Whisper Transcriber

> **Note:** this app was developed with AI coding agents (Google Antigravity, Claude Code, and GLM and DeepSeek models through the Pi coding agent). I directed the work, reviewed the changes and tested the app on my own phone.

Android app that transcribes audio files on the device with [whisper.cpp](https://github.com/ggml-org/whisper.cpp). The audio never leaves the phone: the only network access is the one-off download of the models.

What it does:

- Transcribes the common audio formats (mp3, m4a, ogg, aac, flac, wav), showing the text segment by segment while whisper runs.
- Identifies the speakers after the transcription, with two engines to choose from: a DSP one written in C that needs no download, and a neural one (CAM++ on ONNX Runtime, 14 MB) that is more accurate. If you know how many people are talking you can set it, and the clustering uses that number.
- Downloads the whisper models from Hugging Face, from tiny to large-v3, or imports a local `.bin` file.
- Keeps working with the screen off, through a foreground service, so long recordings don't get killed halfway.
- Exports the transcript as `.txt` and a report with the timing of each phase.

On a Snapdragon 8 Elite, 30 minutes of audio with two speakers take about 38 s to transcribe with the tiny model, and 6.5 s more to separate the speakers with the neural engine.

## Requirements

An arm64 phone with Android 9 or later. The native code is built for ARMv8.2 with the dot-product and fp16 extensions: on older CPUs the app tells you it can't run instead of crashing.

## Building

whisper.cpp is a git submodule, and the app needs a small patch on top of it (timing instrumentation, see [`third_party/patches/README.md`](third_party/patches/README.md)). Without the patch the native build fails to link.

```sh
git clone --recurse-submodules https://github.com/emilio96/whisper-transcriber.git
cd whisper-transcriber
git -C third_party/whisper.cpp apply ../patches/whisper-cpp-306c88f4-perf-dump.patch
./gradlew assembleDebug
```

Or open the folder in Android Studio once the patch is applied. The project pins NDK `28.2.13676358`, which you can install from the SDK Manager.

Release builds are signed only if there's a `keystore.properties` next to the root `build.gradle.kts`: copy `keystore.properties.template` and fill it in. Without it you still get an APK, just unsigned.

## License

MIT, see [`LICENSE`](LICENSE). whisper.cpp is MIT licensed as well.
