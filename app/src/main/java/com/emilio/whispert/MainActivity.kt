package com.emilio.whispert

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.core.content.ContextCompat
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import com.emilio.whispert.ui.MainScreen
import com.emilio.whispert.ui.theme.WhisperTranscriberTheme

class MainActivity : ComponentActivity() {
    private val vm: MainViewModel by viewModels()

    private val dimHandler = Handler(Looper.getMainLooper())
    private var isSmartDimmingActive = false
    private val dimRunnable = Runnable {
        dimScreen()
    }

    private val requestNotificationPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (!granted) {
                android.util.Log.w(
                    "MainActivity",
                    "POST_NOTIFICATIONS denied: transcription still runs, but its foreground " +
                        "notification will not be visible"
                )
            }
        }

    /**
     * The transcription foreground service posts an ongoing notification with
     * live progress. On Android 13+ that permission is granted at runtime, and
     * without it the user gets no feedback at all once the app is backgrounded —
     * which is exactly the long-running case the service exists for.
     */
    private fun ensureNotificationPermission() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) ==
            PackageManager.PERMISSION_GRANTED
        if (!granted) {
            requestNotificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        ensureNotificationPermission()
        enableEdgeToEdge()
        setContent {
            val themeMode by vm.themeMode.collectAsState()
            val transcribeState by vm.transcribeState.collectAsState()
            val keepScreenAwake by vm.keepScreenAwake.collectAsState()
            val shouldKeepScreenOn = (transcribeState is TranscribeUiState.Busy) && keepScreenAwake

            // Keep screen on if requested while transcribing / decoding with smart idle dimming
            DisposableEffect(shouldKeepScreenOn) {
                setSmartDimmingEnabled(shouldKeepScreenOn)
                onDispose {
                    setSmartDimmingEnabled(false)
                }
            }

            val isDark = when (themeMode) {
                ThemeMode.LIGHT -> false
                ThemeMode.DARK -> true
                ThemeMode.SYSTEM -> isSystemInDarkTheme()
            }
            WhisperTranscriberTheme(darkTheme = isDark) {
                Surface(modifier = Modifier.fillMaxSize()) {
                    MainScreen(vm)
                }
            }
        }
    }

    override fun onUserInteraction() {
        super.onUserInteraction()
        if (isSmartDimmingActive) {
            restoreScreenBrightness()
            resetDimTimer()
        }
    }

    override fun onPause() {
        super.onPause()
        dimHandler.removeCallbacks(dimRunnable)
        restoreScreenBrightness()
    }

    override fun onResume() {
        super.onResume()
        if (isSmartDimmingActive) {
            resetDimTimer()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        dimHandler.removeCallbacks(dimRunnable)
        restoreScreenBrightness()
    }

    private fun setSmartDimmingEnabled(enabled: Boolean) {
        isSmartDimmingActive = enabled
        dimHandler.removeCallbacks(dimRunnable)
        if (enabled) {
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            resetDimTimer()
        } else {
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            restoreScreenBrightness()
        }
    }

    private fun resetDimTimer() {
        dimHandler.removeCallbacks(dimRunnable)
        if (isSmartDimmingActive) {
            val timeoutMs = getSystemScreenOffTimeoutMs()
            dimHandler.postDelayed(dimRunnable, timeoutMs)
        }
    }

    private fun dimScreen() {
        if (!isSmartDimmingActive) return
        val lp = window.attributes
        lp.screenBrightness = 0.01f // Dim to lowest level allowed by display
        window.attributes = lp
    }

    private fun restoreScreenBrightness() {
        val lp = window.attributes
        if (lp.screenBrightness != WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE) {
            lp.screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
            window.attributes = lp
        }
    }

    private fun getSystemScreenOffTimeoutMs(): Long {
        return try {
            val timeout = Settings.System.getInt(
                contentResolver,
                Settings.System.SCREEN_OFF_TIMEOUT,
                30_000
            )
            // Clamp between 10 seconds and 10 minutes
            timeout.toLong().coerceIn(10_000L, 600_000L)
        } catch (e: Exception) {
            30_000L
        }
    }
}