package com.emilio.whispert

import android.app.Application
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.emilio.whispert.whisper.WavHelper
import com.emilio.whispert.whisper.WhisperContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Tiny UI-busy state machine for the "hello transcribe" test screen.
 *
 * It owns a [WhisperContext] and persists the SAF Uri of the chosen model so
 * the user doesn't have to re-pick it on every launch. Native memory is freed
 * in [onCleared] via [WhisperContext.releaseBlocking] (the ViewModel scope is
 * already being torn down, so a coroutine-based release would race the GC).
 */
class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val prefs = app.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val _state = MutableStateFlow<UiState>(UiState.Idle)
    val state: StateFlow<UiState> = _state.asStateFlow()

    private var whisperCtx: WhisperContext? = null

    init {
        // Try to restore the previously chosen model Uri on cold start.
        prefs.getString(KEY_MODEL_URI, null)?.let { uri ->
            _state.value = UiState.Ready(uri, "(not yet loaded — press Transcribe to use)")
        }
    }

    /** Called when the user picks a model file via the SAF picker. */
    fun onModelPicked(uri: Uri) {
        viewModelScope.launch {
            _state.value = UiState.Busy("Decoding model header…")
            try {
                // Persist permission so we can reopen on next launch without re-picking.
                runCatching {
                    val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION
                    getApplication<Application>().contentResolver.takePersistableUriPermission(uri, flags)
                }
                prefs.edit().putString(KEY_MODEL_URI, uri.toString()).apply()

                // Drop any previous context before creating a fresh one.
                whisperCtx?.releaseBlocking()
                whisperCtx = null

                _state.value = UiState.Busy("Loading model into native memory…")
                val ctx = withContext(Dispatchers.IO) {
                    val stream = getApplication<Application>()
                        .contentResolver.openInputStream(uri)
                        ?: error("Cannot open InputStream for $uri")
                    WhisperContext.createContextFromInputStream(stream).also { stream.close() }
                }
                whisperCtx = ctx
                _state.value = UiState.Ready(uri.toString(), "Model loaded — ready to transcribe.")
            } catch (t: Throwable) {
                _state.value = UiState.Error("Model load failed: ${t.message ?: t.toString()}")
            }
        }
    }

    /**
     * Transcribes the bundled 16 kHz mono test sample (assets/samples/jfk.wav)
     * with the currently loaded whisper model. This is the "hello transcribe"
     * smoke test that validates the entire native stack end-to-end.
     */
    fun transcribeTestSample(language: String = "en") {
        val ctx = whisperCtx
        if (ctx == null) {
            _state.value = UiState.Error("Pick a model file first.")
            return
        }
        viewModelScope.launch {
            _state.value = UiState.Busy("Reading test audio file from assets…")
            try {
                val start = System.currentTimeMillis()
                val samples = withContext(Dispatchers.IO) {
                    getApplication<Application>().assets.open(TEST_SAMPLE).use {
                        WavHelper.decodePcm16ToMonoFloat(it)
                    }
                }
                _state.value = UiState.Busy("Whisper running… (${samples.size} samples)")
                val text = ctx.transcribe(samples, language = language, printTimestamps = true)
                val elapsed = System.currentTimeMillis() - start
                _state.value = UiState.Done(
                    text = text,
                    elapsedMs = elapsed,
                    language = language,
                    sampleCount = samples.size,
                )
            } catch (t: Throwable) {
                _state.value = UiState.Error("Transcribe failed: ${t.message ?: t.toString()}")
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        whisperCtx?.releaseBlocking()
        whisperCtx = null
    }

    companion object {
        private const val PREFS_NAME = "whispert"
        private const val KEY_MODEL_URI = "model_uri"
        private const val TEST_SAMPLE = "samples/jfk.wav"
    }
}

sealed interface UiState {
    data object Idle : UiState
    data class Busy(val message: String) : UiState
    data class Ready(val modelUri: String, val note: String) : UiState
    data class Done(
        val text: String,
        val elapsedMs: Long,
        val language: String,
        val sampleCount: Int,
    ) : UiState
    data class Error(val message: String) : UiState
}