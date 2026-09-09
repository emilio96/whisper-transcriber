plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.emilio.whispert"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.emilio.whispert"
        minSdk = 28
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Compile the whisper.cpp native lib only for 64-bit ARM. Covers every
        // modern device we ship to; keeps build time and APK size small.
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    // Where the CMake build for whisper.cpp lives.
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            // Use the CMake installed via SDK Manager (currently 4.1.2).
            // Setting this explicitly avoids Gradle auto-downloading 3.22.x.
            version = "4.1.2"
            // NOTE: ggml tuning flags (GGML_OPENMP / GGML_LLAMAFILE /
            // GGML_CPU_ARM_ARCH) are set in src/main/cpp/CMakeLists.txt, not
            // here - AGP 9 removed the cmake.arguments DSL.
        }
    }

    buildTypes {
        release {
            // R8 is on: the JNI entry points and ONNX Runtime's reflective
            // surface are protected by explicit keep rules in proguard-rules.pro.
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }

    // Ship the .so files uncompressed and page-aligned so the loader maps them
    // straight out of the APK. Legacy packaging (the previous setting) extracts
    // a second copy to disk at install time and buys nothing here: the ggml
    // model .bin files live in filesDir and have nothing to do with how the
    // native libraries are packaged.
    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation("androidx.compose.material:material-icons-core")
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}