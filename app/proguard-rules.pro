# The JNI layer resolves these by name at runtime:
#   whisper_jni.c calls FindClass("com/emilio/whispert/whisper/WhisperProgress")
#   and GetStaticMethodID for onProgress / onDecodeProgress / onNewSegment.
# R8 has no way to see those references, so both the class and the methods must
# be kept verbatim, names included.
-keep class com.emilio.whispert.whisper.WhisperProgress {
    public static void onProgress(int);
    public static void onDecodeProgress(int);
    public static void onNewSegment(java.lang.String, long, long);
}

# `external fun` declarations live on WhisperLib's companion; the native symbol
# names encode the enclosing class, so neither may be renamed.
-keep class com.emilio.whispert.whisper.WhisperLib { *; }
-keep class com.emilio.whispert.whisper.WhisperLib$Companion { *; }

# Generic safety net for any other JNI entry point.
-keepclasseswithmembernames class * {
    native <methods>;
}

# ONNX Runtime resolves a good deal of its own surface reflectively.
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
