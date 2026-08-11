package com.emilio.whispert

import android.app.Application
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.emilio.whispert.models.DeviceProfile
import com.emilio.whispert.models.ModelDescriptor
import com.emilio.whispert.models.ModelManager
import com.emilio.whispert.models.ModelRegistry
import com.emilio.whispert.whisper.AudioDecoder
import com.emilio.whispert.whisper.WavHelper
import com.emilio.whispert.whisper.WhisperContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * State holder for the whole app. Holds two slices:
 *   - transcribe state (audio file, language, run result) — the "Transcribe" tab
 *   - model library state (registry + download progress + active) — the "Models" tab
 *
 * Long-lived resources:
 *   - [modelManager]  : downloads + caches ggml .bin files in filesDir/models
 *   - [whisperCtx]    : native whisper context; pinned to a single worker thread
 *     inside WhisperContext. Freed blocking in onCleared().
 *
 * The two slices talk through an explicit [ensureModelLoaded] gate before any
 * transcription: there must be a downloaded+activated model, and the model must
 * currently live in native memory; otherwise we materialise it from disk.
 */
class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    private val modelManager = ModelManager(app)
    private val deviceProfile = DeviceProfile.probe(app)

    private var whisperCtx: WhisperContext? = null
    private var loadedModelId: String? = null

    // ---------- transcribe slice ----------
    private val _transcribeState = MutableStateFlow<TranscribeUiState>(TranscribeUiState.Idle)
    val transcribeState: StateFlow<TranscribeUiState> = _transcribeState.asStateFlow()

    private val _audioUri = MutableStateFlow<Uri?>(prefs.getString(KEY_AUDIO_URI, null)?.let { Uri.parse(it) })
    val audioUri: StateFlow<Uri?> = _audioUri.asStateFlow()

    private val _language = MutableStateFlow(prefs.getString(KEY_LANGUAGE, "auto") ?: "auto")
    val language: StateFlow<String> = _language.asStateFlow()

    /**
     * Build the full visible list of models = the static registry + any imported
     * models the user has loaded via SAF. Imported models persist across launches
     * via the ModelManager's SharedPreferences-backed runtime registry.
     */
    private fun visibleModels(): List<ModelDescriptor> =
        ModelRegistry.ALL + modelManager.listImported()

    // ---------- models slice ----------
    private val _modelsState = MutableStateFlow(
        ModelsUiState(
            registry = visibleModels(),
            deviceClass = deviceProfile.deviceClass,
            deviceDescription = deviceProfile.describe(),
            modelStates = visibleModels().associate { it.id to stateFor(it) },
            activeModelId = modelManager.activeModelId(),
        )
    )
    val modelsState: StateFlow<ModelsUiState> = _modelsState.asStateFlow()

    private fun stateFor(desc: ModelDescriptor): ModelItemState {
        val cached = modelManager.isDownloaded(desc)
        val active = modelManager.activeModelId() == desc.id
        return ModelItemState(
            descriptor = desc,
            downloaded = cached,
            active = active,
            downloading = false,
            progress = 0f,
            error = null,
        )
    }

    private fun refreshModelsState() {
        val visible = visibleModels()
        val cur = _modelsState.value
        _modelsState.value = cur.copy(
            registry = visible,
            activeModelId = modelManager.activeModelId(),
            modelStates = visible.associate { desc ->
                desc.id to ModelItemState(
                    descriptor = desc,
                    downloaded = modelManager.isDownloaded(desc),
                    active = modelManager.activeModelId() == desc.id,
                    downloading = cur.modelStates[desc.id]?.downloading ?: false,
                    progress = cur.modelStates[desc.id]?.progress ?: 0f,
                    error = cur.modelStates[desc.id]?.error,
                )
            }
        )
    }

    // ---------- audio selection ----------
    fun onAudioPicked(uri: Uri) {
        viewModelScope.launch {
            // Persist permission so the Uri survives a relaunch without re-picking.
            runCatching {
                val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION
                getApplication<Application>().contentResolver.takePersistableUriPermission(uri, flags)
            }
            prefs.edit().putString(KEY_AUDIO_URI, uri.toString()).apply()
            _audioUri.value = uri
        }
    }

    fun clearAudio() {
        prefs.edit().remove(KEY_AUDIO_URI).apply()
        _audioUri.value = null
    }

    fun setLanguage(lang: String) {
        prefs.edit().putString(KEY_LANGUAGE, lang).apply()
        _language.value = lang
    }

    // ---------- model library actions ----------
    fun downloadModel(desc: ModelDescriptor) {
        // Avoid concurrent downloads of the same id.
        if (_modelsState.value.modelStates[desc.id]?.downloading == true) return
        updateItem(desc.id) { it.copy(downloading = true, progress = 0f, error = null) }

        viewModelScope.launch {
            try {
                modelManager.download(desc) { p -> updateItem(desc.id) { it.copy(progress = p) } }
                updateItem(desc.id) { it.copy(downloading = false, downloaded = true, progress = 1f, error = null) }
                refreshModelsState()
            } catch (t: Throwable) {
                updateItem(desc.id) { it.copy(downloading = false, progress = 0f, error = t.message ?: t.toString()) }
            }
        }
    }

    fun deleteModel(desc: ModelDescriptor) {
        modelManager.deleteModel(desc)
        refreshModelsState()
        // If this was the loaded model, drop the native context.
        if (loadedModelId == desc.id) {
            whisperCtx?.releaseBlocking()
            whisperCtx = null
            loadedModelId = null
        }
    }

    fun activateModel(desc: ModelDescriptor) {
        if (!modelManager.isDownloaded(desc)) return
        modelManager.setActiveModelId(desc.id)
        refreshModelsState()
    }

    /**
     * Imports a .bin model file picked with SAF (e.g. one the user already had on
     * the device, downloaded via a browser or copied via USB). Streams the SAF
     * stream into filesDir/models/ggml-imported-<N>.bin and activates it.
     */
    fun importModel(uri: Uri) {
        viewModelScope.launch {
            _transcribeState.value = TranscribeUiState.Busy("Importing model from file…")
            try {
                // Drop any currently-loaded native context because we are about to
                // switch the active model to the imported one.
                if (loadedModelId != null) {
                    whisperCtx?.releaseBlocking()
                    whisperCtx = null
                    loadedModelId = null
                }
                modelManager.importFromFile(uri)
                refreshModelsState()
                _transcribeState.value = TranscribeUiState.Idle
            } catch (t: Throwable) {
                refreshModelsState()
                _transcribeState.value = TranscribeUiState.Error("Import failed: ${t.message ?: t.toString()}")
            }
        }
    }


    private fun updateItem(id: String, block: (ModelItemState) -> ModelItemState) {
        val cur = _modelsState.value
        _modelsState.value = cur.copy(modelStates = cur.modelStates.mapValues { (k, v) ->
            if (k == id) block(v) else v
        })
    }

    // ---------- transcription ----------
    fun transcribe() {
        val uri = _audioUri.value
        if (uri == null) {
            _transcribeState.value = TranscribeUiState.Error("Pick an audio file first.")
            return
        }

        viewModelScope.launch {
            _transcribeState.value = TranscribeUiState.Busy("Checking model…")
            try {
                val ctxOrError = ensureModelLoaded()
                if (ctxOrError.err != null) {
                    _transcribeState.value = TranscribeUiState.Error(ctxOrError.err)
                    return@launch
                }
                val ctx = ctxOrError.ctx!!
                val activeDesc = ctxOrError.desc

                val language = effectiveLanguage(activeDesc)

                _transcribeState.value = TranscribeUiState.Busy("Decoding audio…")
                val start = System.currentTimeMillis()
                val samples = withContext(Dispatchers.IO) {
                    AudioDecoder.decode(getApplication(), uri) { p ->
                        _transcribeState.value = TranscribeUiState.Busy("Decoding audio… ${"%.0f".format(p * 100)}%")
                    }
                }

                _transcribeState.value = TranscribeUiState.Busy("Whisper running… (${samples.size} samples)")
                val text = ctx.transcribe(samples, language = language, printTimestamps = true)
                val elapsed = System.currentTimeMillis() - start
                _transcribeState.value = TranscribeUiState.Done(
                    text = text,
                    elapsedMs = elapsed,
                    language = language,
                    modelName = activeDesc.displayName,
                    modelId = activeDesc.id,
                    sampleCount = samples.size,
                )
            } catch (t: Throwable) {
                _transcribeState.value = TranscribeUiState.Error("Transcribe failed: ${t.message ?: t.toString()}")
            }
        }
    }

    /** Smoke test that always uses the bundled jfk.wav, regardless of picked file. */
    fun transcribeBundledJfk() {
        viewModelScope.launch {
            _transcribeState.value = TranscribeUiState.Busy("Checking model…")
            try {
                val ctxOrError = ensureModelLoaded()
                if (ctxOrError.err != null) {
                    _transcribeState.value = TranscribeUiState.Error(ctxOrError.err)
                    return@launch
                }
                val ctx = ctxOrError.ctx!!
                val activeDesc = ctxOrError.desc

                val language = effectiveLanguage(activeDesc)
                _transcribeState.value = TranscribeUiState.Busy("Reading test audio from assets…")
                val start = System.currentTimeMillis()
                val samples = withContext(Dispatchers.IO) {
                    getApplication<Application>().assets.open(TEST_SAMPLE).use {
                        WavHelper.decodePcm16ToMonoFloat(it)
                    }
                }
                _transcribeState.value = TranscribeUiState.Busy("Whisper running… (${samples.size} samples)")
                val text = ctx.transcribe(samples, language = language, printTimestamps = true)
                val elapsed = System.currentTimeMillis() - start
                _transcribeState.value = TranscribeUiState.Done(
                    text = text,
                    elapsedMs = elapsed,
                    language = language,
                    modelName = activeDesc.displayName,
                    modelId = activeDesc.id,
                    sampleCount = samples.size,
                )
            } catch (t: Throwable) {
                _transcribeState.value = TranscribeUiState.Error("Transcribe failed: ${t.message ?: t.toString()}")
            }
        }
    }

    private fun effectiveLanguage(desc: ModelDescriptor): String =
        if (desc.isEnglishOnly) "en" else _language.value

    private data class ModelReady(val ctx: WhisperContext?, val desc: ModelDescriptor, val err: String?)

    private suspend fun ensureModelLoaded(): ModelReady {
        val active = modelManager.activeModel() ?: return ModelReady(null, ModelRegistry.ALL.first(),
            "No model yet. Go to the Models tab and download one.")

        if (loadedModelId != active.id || whisperCtx == null) {
            _transcribeState.value = TranscribeUiState.Busy("Loading model '${active.displayName}'…")
            whisperCtx?.releaseBlocking(); whisperCtx = null; loadedModelId = null
            val file = modelManager.cachedFile(active)
                ?: return ModelReady(null, active, "Model '${active.displayName}' file is missing. Re-download it.")
            val ctx = withContext(Dispatchers.IO) {
                WhisperContext.createContextFromFile(file.absolutePath)
            }
            whisperCtx = ctx
            loadedModelId = active.id
        }
        return ModelReady(whisperCtx, active, null)
    }

    /**
     * Writes [content] to a SAF-selected destination [uri] (typically Downloads
     * or Documents). Returns true on success. The UI exposes this from the
     * 'Done' state via ActivityResultContracts.CreateDocument.
     */
    suspend fun saveTranscriptTo(content: String, uri: Uri): Boolean = withContext(Dispatchers.IO) {
        try {
            getApplication<Application>().contentResolver.openOutputStream(uri, "w")?.use { out ->
                out.write(content.toByteArray())
                true
            } ?: false
        } catch (t: Throwable) {
            Log.w(TAG_VM, "Export failed", t)
            false
        }
    }

    override fun onCleared() {
        super.onCleared()
        whisperCtx?.releaseBlocking()
        whisperCtx = null
        loadedModelId = null
    }

    companion object {
        private const val TAG_VM = "MainViewModel"
        private const val PREFS = "whispert"
        private const val KEY_AUDIO_URI = "audio_uri"
        private const val KEY_LANGUAGE = "language"
        private const val TEST_SAMPLE = "samples/jfk.wav"
    }
}

// ---------- UI state contracts ----------

sealed interface TranscribeUiState {
    data object Idle : TranscribeUiState
    data class Busy(val message: String) : TranscribeUiState
    data class Done(
        val text: String,
        val elapsedMs: Long,
        val language: String,
        val modelName: String,
        val modelId: String,
        val sampleCount: Int,
    ) : TranscribeUiState
    data class Error(val message: String) : TranscribeUiState
}

data class ModelItemState(
    val descriptor: ModelDescriptor,
    val downloaded: Boolean,
    val active: Boolean,
    val downloading: Boolean,
    val progress: Float,
    val error: String?,
)

data class ModelsUiState(
    val registry: List<ModelDescriptor>,
    val deviceClass: com.emilio.whispert.models.DeviceClass,
    val deviceDescription: String,
    val modelStates: Map<String, ModelItemState>,
    val activeModelId: String?,
)