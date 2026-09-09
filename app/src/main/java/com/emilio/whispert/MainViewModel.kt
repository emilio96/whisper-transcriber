package com.emilio.whispert

import android.app.Application
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.emilio.whispert.models.DeviceProfile
import com.emilio.whispert.models.ModelDescriptor
import com.emilio.whispert.models.ModelManager
import com.emilio.whispert.models.ModelRegistry
import com.emilio.whispert.models.SpeakerModelManager
import com.emilio.whispert.whisper.AudioDecoder
import com.emilio.whispert.whisper.DecoderEngine
import com.emilio.whispert.whisper.DiarizationEngine
import com.emilio.whispert.whisper.InMemoryPcmSource
import com.emilio.whispert.whisper.PcmSource
import com.emilio.whispert.whisper.Segment
import com.emilio.whispert.whisper.SpeakerDiarization
import com.emilio.whispert.whisper.TranscriptionService
import com.emilio.whispert.whisper.WavHelper
import com.emilio.whispert.whisper.WhisperConfig
import com.emilio.whispert.whisper.WhisperContext
import com.emilio.whispert.whisper.WhisperCpuConfig
import com.emilio.whispert.whisper.WhisperLib
import com.emilio.whispert.whisper.formatTranscriptWithSpeakers
import com.emilio.whispert.whisper.mappedPcmSource
import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * State holder for the whole app. Holds three slices:
 *   - transcribe state (audio file, language, run result, cached audio) — the "Transcribe" screen
 *   - model library state (registry + download progress + active) — the "Models" screen
 *   - model settings state (whisper parameters, threads, temperatures) — the "Model settings" screen
 *
 * Long-lived resources:
 *   - [modelManager]  : downloads + caches ggml .bin files in filesDir/models
 *   - [whisperCtx]    : native whisper context; pinned to a single worker thread
 *     inside WhisperContext. Released in onCleared().
 *
 * The slices talk through an explicit [ensureModelLoaded] gate before any
 * transcription: there must be a downloaded+activated model, and the model must
 * currently live in native memory; otherwise we materialise it from disk.
 */
enum class ThemeMode(val title: String) {
    SYSTEM("Auto"),
    LIGHT("Light"),
    DARK("Dark")
}

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    private val modelManager = ModelManager(app)
    private val speakerModelManager = SpeakerModelManager(app)
    private val deviceProfile = DeviceProfile.probe(app)

    private val _speakerModelDownloaded = MutableStateFlow(speakerModelManager.isDownloaded())
    val speakerModelDownloaded: StateFlow<Boolean> = _speakerModelDownloaded.asStateFlow()

    private val _speakerModelDownloading = MutableStateFlow(false)
    val speakerModelDownloading: StateFlow<Boolean> = _speakerModelDownloading.asStateFlow()

    private val _speakerModelProgress = MutableStateFlow(0f)
    val speakerModelProgress: StateFlow<Float> = _speakerModelProgress.asStateFlow()

    private val _speakerModelError = MutableStateFlow<String?>(null)
    val speakerModelError: StateFlow<String?> = _speakerModelError.asStateFlow()

    // Why a model-library action was refused, shown on the Models screen.
    private val _modelActionError = MutableStateFlow<String?>(null)
    val modelActionError: StateFlow<String?> = _modelActionError.asStateFlow()

    private var speakerDownloadJob: Job? = null

    fun downloadSpeakerModel() {
        if (_speakerModelDownloading.value) return
        speakerDownloadJob = viewModelScope.launch {
            _speakerModelDownloading.value = true
            _speakerModelProgress.value = 0f
            _speakerModelError.value = null
            try {
                speakerModelManager.download { p ->
                    _speakerModelProgress.value = p
                }
                _speakerModelDownloaded.value = true
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                Log.e(LOG_TAG, "Failed to download neural speaker model", t)
                _speakerModelError.value = t.message ?: t.toString()
            } finally {
                _speakerModelDownloading.value = false
            }
        }
    }

    /**
     * Deleting has to cancel an in-flight download and wait for it to unwind
     * before touching the file. Otherwise the download coroutine is still
     * writing while we delete, and on completion it sets downloaded = true for
     * a file the user just removed - leaving the UI claiming a model that is
     * not there, with the engine already switched back to DSP.
     */
    fun deleteSpeakerModel() {
        val inFlight = speakerDownloadJob
        speakerDownloadJob = null
        viewModelScope.launch {
            inFlight?.cancelAndJoin()
            withContext(Dispatchers.IO) { speakerModelManager.deleteModel() }
            _speakerModelDownloaded.value = false
            _speakerModelDownloading.value = false
            _speakerModelProgress.value = 0f
            _speakerModelError.value = null
            if (_settings.value.diarizationEngine == DiarizationEngine.NEURAL_ONNX) {
                updateSettings(_settings.value.copy(diarizationEngine = DiarizationEngine.ACOUSTIC_DSP))
            }
        }
    }

    private var whisperCtx: WhisperContext? = null
    private var loadedModelId: String? = null
    private var transcribeJob: Job? = null

    val deviceCoreCount: Int get() = deviceProfile.coreCount

    // Computed once: the underlying probe reads /proc/cpuinfo plus one sysfs file
    // per core, and both of these are read straight from composable bodies.
    val autoThreadCount: Int by lazy { WhisperCpuConfig.preferredThreadCount }
    val hardwareInfoSummary: String by lazy {
        DeviceProfile.getHardwareInfoSummary(deviceCoreCount, autoThreadCount)
    }

    // ---------- theme mode slice ----------
    private val _themeMode = MutableStateFlow(
        when (prefs.getString("app_theme_mode", "system")) {
            "light" -> ThemeMode.LIGHT
            "dark" -> ThemeMode.DARK
            else -> ThemeMode.SYSTEM
        }
    )
    val themeMode: StateFlow<ThemeMode> = _themeMode.asStateFlow()

    fun setThemeMode(mode: ThemeMode) {
        prefs.edit().putString("app_theme_mode", mode.name.lowercase()).apply()
        _themeMode.value = mode
    }

    // ---------- keep screen awake slice ----------
    private val _keepScreenAwake = MutableStateFlow(prefs.getBoolean("keep_screen_awake", false))
    val keepScreenAwake: StateFlow<Boolean> = _keepScreenAwake.asStateFlow()

    fun setKeepScreenAwake(enabled: Boolean) {
        prefs.edit().putBoolean("keep_screen_awake", enabled).apply()
        _keepScreenAwake.value = enabled
    }

    // ---------- speaker diarization slice ----------
    private val _identifySpeakers = MutableStateFlow(prefs.getBoolean("identify_speakers", false))
    val identifySpeakers: StateFlow<Boolean> = _identifySpeakers.asStateFlow()

    fun setIdentifySpeakers(enabled: Boolean) {
        prefs.edit().putBoolean("identify_speakers", enabled).apply()
        _identifySpeakers.value = enabled
    }

    // ---------- audio cache slice ----------
    private val audioCacheDir: File by lazy {
        File(getApplication<Application>().cacheDir, "audio_cache").apply { mkdirs() }
    }

    // Starts at 0 and is filled in off the main thread: creating the directory
    // and listing it is disk I/O, and this runs during ViewModel construction.
    private val _audioCacheSizeBytes = MutableStateFlow(0L)
    val audioCacheSizeBytes: StateFlow<Long> = _audioCacheSizeBytes.asStateFlow()

    init {
        refreshAudioCacheSize()
        // The shipped speaker model changed; drop any superseded file so it
        // stops occupying ~27 MB of internal storage for nothing.
        viewModelScope.launch {
            withContext(Dispatchers.IO) { speakerModelManager.purgeLegacyModels() }
        }
    }

    private fun refreshAudioCacheSize() {
        viewModelScope.launch {
            _audioCacheSizeBytes.value = withContext(Dispatchers.IO) { computeAudioCacheSize() }
        }
    }

    private val _lastDecodeInfo = MutableStateFlow<String?>(null)
    val lastDecodeInfo: StateFlow<String?> = _lastDecodeInfo.asStateFlow()

    private fun computeAudioCacheSize(): Long {
        return audioCacheDir.listFiles()?.sumOf { it.length() } ?: 0L
    }

    /**
     * Cache key for a decoded WAV.
     *
     * Uses SHA-256 over the Uri plus the document's size and last-modified stamp.
     * String.hashCode() (the previous key) is 32-bit, so two different files
     * could silently map to the same cache entry and the app would transcribe
     * the wrong audio; and without size/mtime in the key, editing the file
     * behind a stable content:// Uri kept serving the stale decode.
     */
    private suspend fun getDecodedCacheFile(uri: Uri): File = withContext(Dispatchers.IO) {
        val fingerprint = buildString {
            append(uri.toString())
            runCatching {
                getApplication<Application>().contentResolver.query(
                    uri,
                    arrayOf(OpenableColumns.SIZE, DocumentsContract.Document.COLUMN_LAST_MODIFIED),
                    null, null, null
                )?.use { c ->
                    if (c.moveToFirst()) {
                        append('|').append(if (c.isNull(0)) "?" else c.getLong(0).toString())
                        append('|').append(if (c.isNull(1)) "?" else c.getLong(1).toString())
                    }
                }
            }
        }
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(fingerprint.toByteArray())
            .take(16)
            .joinToString("") { "%02x".format(it) }
        File(audioCacheDir, "decoded_$digest.wav")
    }

    /**
     * Trims the decoded-audio cache down to [MAX_AUDIO_CACHE_BYTES], oldest
     * first. Without this the directory only ever grew (~115 MB per hour of
     * audio) until the user happened to notice the "Clear cache" button.
     */
    private fun evictAudioCache(keep: File?) {
        val files = audioCacheDir.listFiles()?.sortedBy { it.lastModified() } ?: return
        var total = files.sumOf { it.length() }
        for (f in files) {
            if (total <= MAX_AUDIO_CACHE_BYTES) break
            if (f == keep || f.name.endsWith(".part")) continue
            val size = f.length()
            if (f.delete()) {
                total -= size
                Log.i(LOG_TAG, "Evicted cached decode ${f.name} ($size B)")
            }
        }
    }

    fun clearAudioCache() {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { audioCacheDir.listFiles()?.forEach { it.delete() } }
            refreshAudioCacheSize()
            _lastDecodeInfo.value = null
        }
    }

    // ---------- settings slice ----------
    private val _settings = MutableStateFlow(loadSettings())
    val settings: StateFlow<WhisperConfig> = _settings.asStateFlow()

    fun updateSettings(config: WhisperConfig) {
        saveSettings(config)
        _settings.value = config
    }

    fun resetSettings() {
        updateSettings(WhisperConfig())
    }

    private fun loadSettings(): WhisperConfig {
        return WhisperConfig(
            decoderEngine = runCatching {
                DecoderEngine.valueOf(
                    prefs.getString("cfg_decoder_engine", DecoderEngine.NATIVE.name) ?: DecoderEngine.NATIVE.name
                )
            }.getOrDefault(DecoderEngine.NATIVE),
            diarizationEngine = runCatching {
                DiarizationEngine.valueOf(
                    prefs.getString("cfg_diarization_engine", DiarizationEngine.ACOUSTIC_DSP.name)
                        ?: DiarizationEngine.ACOUSTIC_DSP.name
                )
            }.getOrDefault(DiarizationEngine.ACOUSTIC_DSP),
            speakerMergeThreshold = prefs.getFloat("cfg_speaker_merge_thold", 0f),
            expectedSpeakers = prefs.getInt("cfg_expected_speakers", 0),
            customThreads = prefs.getInt("cfg_threads", 0),
            temperature = prefs.getFloat("cfg_temperature", 0.0f),
            temperatureInc = prefs.getFloat("cfg_temperature_inc", 0.0f),
            entropyThold = prefs.getFloat("cfg_entropy_thold", 2.4f),
            logprobThold = prefs.getFloat("cfg_logprob_thold", -1.0f),
            noSpeechThold = prefs.getFloat("cfg_no_speech_thold", 0.6f),
            noContext = prefs.getBoolean("cfg_no_context", true),
            singleSegment = prefs.getBoolean("cfg_single_segment", false),
            translate = prefs.getBoolean("cfg_translate", false),
            suppressBlank = prefs.getBoolean("cfg_suppress_blank", true),
            suppressNonSpeechTokens = prefs.getBoolean("cfg_suppress_non_speech_tokens", true),
            initialPrompt = prefs.getString("cfg_initial_prompt", "") ?: "",
        )
    }

    private fun saveSettings(c: WhisperConfig) {
        prefs.edit()
            .putString("cfg_decoder_engine", c.decoderEngine.name)
            .putString("cfg_diarization_engine", c.diarizationEngine.name)
            .putFloat("cfg_speaker_merge_thold", c.speakerMergeThreshold)
            .putInt("cfg_expected_speakers", c.expectedSpeakers)
            .putInt("cfg_threads", c.customThreads)
            .putFloat("cfg_temperature", c.temperature)
            .putFloat("cfg_temperature_inc", c.temperatureInc)
            .putFloat("cfg_entropy_thold", c.entropyThold)
            .putFloat("cfg_logprob_thold", c.logprobThold)
            .putFloat("cfg_no_speech_thold", c.noSpeechThold)
            .putBoolean("cfg_no_context", c.noContext)
            .putBoolean("cfg_single_segment", c.singleSegment)
            .putBoolean("cfg_translate", c.translate)
            .putBoolean("cfg_suppress_blank", c.suppressBlank)
            .putBoolean("cfg_suppress_non_speech_tokens", c.suppressNonSpeechTokens)
            .putString("cfg_initial_prompt", c.initialPrompt)
            .apply()
    }

    // ---------- transcribe slice ----------
    private val _transcribeState = MutableStateFlow<TranscribeUiState>(TranscribeUiState.Idle)
    val transcribeState: StateFlow<TranscribeUiState> = _transcribeState.asStateFlow()

    // Fresh start on every app launch: no stale audio Uri from a previous session.
    private val _audioUri = MutableStateFlow<Uri?>(null)
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
    private val totalRamGb = Math.round(deviceProfile.totalMemBytes.toDouble() / (1024.0 * 1024.0 * 1024.0)).toInt()

    private val _modelsState = MutableStateFlow(
        ModelsUiState(
            registry = visibleModels(),
            deviceClass = deviceProfile.deviceClass,
            deviceDescription = deviceProfile.describe(),
            totalRamGb = totalRamGb,
            modelStates = visibleModels().associate { it.id to stateFor(it) },
            activeModelId = modelManager.activeModelId(),
        )
    )
    val modelsState: StateFlow<ModelsUiState> = _modelsState.asStateFlow()

    private fun stateFor(desc: ModelDescriptor): ModelItemState = ModelItemState(
        descriptor = desc,
        downloaded = modelManager.isDownloaded(desc),
        active = modelManager.activeModelId() == desc.id,
        downloading = false,
        progress = 0f,
        error = null,
    )

    private fun refreshModelsState() {
        val visible = visibleModels()
        val activeId = modelManager.activeModelId()
        val downloaded = visible.associate { it.id to modelManager.isDownloaded(it) }
        _modelsState.update { cur ->
            cur.copy(
                registry = visible,
                activeModelId = activeId,
                modelStates = visible.associate { desc ->
                    desc.id to ModelItemState(
                        descriptor = desc,
                        downloaded = downloaded[desc.id] == true,
                        active = activeId == desc.id,
                        downloading = cur.modelStates[desc.id]?.downloading ?: false,
                        progress = cur.modelStates[desc.id]?.progress ?: 0f,
                        error = cur.modelStates[desc.id]?.error,
                    )
                }
            )
        }
    }

    // ---------- audio selection ----------
    fun onAudioPicked(uri: Uri) {
        viewModelScope.launch {
            // Persist permission so the Uri survives a relaunch without re-picking.
            runCatching {
                val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION
                getApplication<Application>().contentResolver.takePersistableUriPermission(uri, flags)
            }
            _audioUri.value = uri
            _lastDecodeInfo.value = null
            forgetLastRun()
            refreshAudioCacheSize()
        }
    }

    fun clearAudio() {
        _audioUri.value = null
        _lastDecodeInfo.value = null
        forgetLastRun()
    }

    /** Drops the replay handle: it points at audio that is no longer selected. */
    private fun forgetLastRun() {
        lastRun = null
        _canRediarize.value = false
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

    /**
     * Guards the model-library actions against a live run.
     *
     * Both deleting and importing tear down the native context, and release()
     * queues behind the in-flight whisper_full on the single-thread inference
     * executor: the button would simply hang until the transcription finished,
     * with no indication why. Import additionally overwrote the transcribe
     * state, wiping the running job's progress off the screen.
     */
    private fun blockedByActiveRun(action: String): Boolean {
        if (transcribeJob?.isActive != true) return false
        _modelActionError.value = "Cannot $action while a transcription is running. Stop it first."
        return true
    }

    fun clearModelActionError() {
        _modelActionError.value = null
    }

    fun deleteModel(desc: ModelDescriptor) {
        if (blockedByActiveRun("delete a model")) return
        _modelActionError.value = null
        viewModelScope.launch {
            // Drop the native context first if this is the loaded model, so the
            // memory is gone before the file is.
            if (loadedModelId == desc.id) {
                whisperCtx?.release()
                whisperCtx = null
                loadedModelId = null
            }
            modelManager.deleteModel(desc)
            refreshModelsState()
        }
    }

    fun activateModel(desc: ModelDescriptor) {
        if (!modelManager.isDownloaded(desc)) return
        if (blockedByActiveRun("switch models")) return
        _modelActionError.value = null
        modelManager.setActiveModelId(desc.id)
        refreshModelsState()
    }

    /**
     * Imports a .bin model file picked with SAF. Streams the SAF stream into
     * filesDir/models/ggml-imported-<N>.bin and activates it.
     */
    fun importModel(uri: Uri) {
        if (blockedByActiveRun("import a model")) return
        _modelActionError.value = null
        viewModelScope.launch {
            _transcribeState.value = TranscribeUiState.Busy("Importing model from file…")
            try {
                // Drop any currently-loaded native context because we are about to
                // switch the active model to the imported one.
                if (loadedModelId != null) {
                    whisperCtx?.release()
                    whisperCtx = null
                    loadedModelId = null
                }
                modelManager.importFromFile(uri)
                refreshModelsState()
                _transcribeState.value = TranscribeUiState.Idle
            } catch (t: Throwable) {
                refreshModelsState()
                _transcribeState.value = TranscribeUiState.Error("Import failed: ${describe(t)}")
            }
        }
    }

    /**
     * Download progress arrives on Dispatchers.IO while refreshModelsState()
     * runs on main, and both were doing a plain load-copy-store on the same
     * StateFlow: a progress tick could be lost, or could resurrect the state a
     * concurrent activate/delete had just written. update() retries on a
     * concurrent write instead.
     */
    private fun updateItem(id: String, block: (ModelItemState) -> ModelItemState) {
        _modelsState.update { cur ->
            cur.copy(modelStates = cur.modelStates.mapValues { (k, v) ->
                if (k == id) block(v) else v
            })
        }
    }

    // ---------- transcription ----------
    fun cancelTranscription() {
        Log.i(LOG_TAG, "Cancelling transcription on user request")
        WhisperLib.cancelTranscription()
        transcribeJob?.cancel()
        transcribeJob = null
        TranscriptionService.stop(getApplication())
        _transcribeState.value = TranscribeUiState.Idle
    }

    /**
     * Emits Busy states and notification updates at a bounded rate.
     *
     * These callbacks fire on the whisper inference thread. Rebuilding the whole
     * segment list and the whole accumulated transcript string on every single
     * one of them — plus a Binder round trip for the notification — was work
     * taken directly away from the decoder.
     */
    private inner class ProgressThrottle {
        private val segments = mutableListOf<Segment>()
        private val textBuilder = StringBuilder()
        private var lastUiEmitMs = 0L
        private var lastNotifyMs = 0L
        private var percent = 0

        fun addSegment(seg: Segment) {
            segments.add(seg)
            textBuilder.append(seg.text.trim()).append('\n')
            emit(force = false)
        }

        fun setPercent(p: Int) {
            percent = p
            emit(force = false)
            maybeNotify()
        }

        fun emit(force: Boolean) {
            val now = System.currentTimeMillis()
            if (!force && now - lastUiEmitMs < UI_EMIT_INTERVAL_MS) return
            lastUiEmitMs = now
            _transcribeState.value = TranscribeUiState.Busy(
                message = "Whisper running… $percent%",
                progress = percent / 100f,
                partialSegments = segments.toList(),
                partialText = textBuilder.toString(),
            )
        }

        private fun maybeNotify() {
            val now = System.currentTimeMillis()
            if (now - lastNotifyMs < NOTIFY_INTERVAL_MS) return
            lastNotifyMs = now
            TranscriptionService.update(
                getApplication(),
                "Whisper Transcriber",
                "Whisper transcribing… $percent%",
                percent
            )
        }
    }

    fun transcribe() {
        val uri = _audioUri.value
        if (uri == null) {
            _transcribeState.value = TranscribeUiState.Error("Pick an audio file first.")
            return
        }
        startTranscription { runFileTranscription(uri) }
    }

    /** Smoke test that always uses the bundled jfk.wav, regardless of picked file. */
    fun transcribeBundledJfk() {
        startTranscription { runSmokeTest() }
    }

    /**
     * Shared entry gate. Note the explicit native cancel: cancelling the
     * coroutine alone leaves whisper_full running on the inference thread, so
     * the new run would silently queue behind the old one and the UI would sit
     * on "Checking model…" until it finished.
     */
    private fun startTranscription(body: suspend () -> Unit) {
        if (transcribeJob?.isActive == true) {
            WhisperLib.cancelTranscription()
            transcribeJob?.cancel()
        }
        // The cleanup in the finally must only fire for the run that still owns
        // the field. A cancelled job resumes its finally on the main dispatcher
        // *after* its replacement has been assigned, so an unguarded
        // `transcribeJob = null` there would clear the new job's handle - letting
        // a third run start in parallel past the isActive guard above - and stop
        // the foreground service out from under it. Hence the identity check.
        //
        // LAZY so that the assignment below happens first: on
        // Dispatchers.Main.immediate an eagerly started coroutine runs up to its
        // first suspension point before launch() even returns, and the identity
        // check would then be comparing against the previous run's handle.
        val job = viewModelScope.launch(start = CoroutineStart.LAZY) {
            TranscriptionService.start(getApplication(), "Whisper Transcriber", "Initializing transcription…")
            try {
                body()
            } catch (e: CancellationException) {
                Log.i(LOG_TAG, "Transcription cancelled by user")
                _transcribeState.value = TranscribeUiState.Idle
                throw e
            } catch (t: Throwable) {
                Log.e(LOG_TAG, "Transcription failed", t)
                _transcribeState.value = TranscribeUiState.Error("Transcribe failed: ${describe(t)}")
            } finally {
                if (transcribeJob === coroutineContext[Job]) {
                    TranscriptionService.stop(getApplication())
                    transcribeJob = null
                }
            }
        }
        transcribeJob = job
        job.start()
    }

    private suspend fun runFileTranscription(uri: Uri) {
        _transcribeState.value = TranscribeUiState.Busy("Checking model…")
        val pcmFile = getDecodedCacheFile(uri)

        val ready = ensureModelLoaded()
        if (ready.err != null) {
            _transcribeState.value = TranscribeUiState.Error(ready.err)
            return
        }
        val ctx = ready.ctx!!
        val activeDesc = ready.desc
        val language = effectiveLanguage(activeDesc)
        val config = _settings.value

        var decodeMs = 0L
        val sampleCount: Int

        // stat() and the mtime touch are disk I/O; the rest of this method is
        // careful to keep that off the main dispatcher, and these were missed.
        val cachedLength = withContext(Dispatchers.IO) {
            if (pcmFile.exists()) pcmFile.length() else 0L
        }

        if (cachedLength > 44) {
            Log.i(LOG_TAG, "Using existing cached decoded WAV: ${pcmFile.absolutePath} ($cachedLength bytes)")
            sampleCount = ((cachedLength - 44) / 2).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
            val durStr = com.emilio.whispert.ui.formatDurationSeconds((sampleCount / 16000f).toDouble())
            _lastDecodeInfo.value = "⚡ Loaded from cache ($durStr audio, decode: 0.0s)"
            // Keep it hot for LRU eviction.
            withContext(Dispatchers.IO) { pcmFile.setLastModified(System.currentTimeMillis()) }
        } else {
            _transcribeState.value = TranscribeUiState.Busy("Decoding audio…", progress = 0f)
            TranscriptionService.update(getApplication(), "Whisper Transcriber", "Decoding audio…", 0)
            val tmpPcm = File(audioCacheDir, "${pcmFile.name}.part")
            try {
                var lastNotify = 0L
                val decodeResult = withContext(Dispatchers.IO) {
                    AudioDecoder.decodeToWavFile(
                        context = getApplication(),
                        uri = uri,
                        outFile = tmpPcm,
                        engine = config.decoderEngine,
                        isCancelled = { !coroutineContext.isActive },
                        onProgress = { p ->
                            val pct = (p * 100).toInt()
                            _transcribeState.value = TranscribeUiState.Busy("Decoding audio… $pct%", progress = p)
                            val now = System.currentTimeMillis()
                            if (now - lastNotify >= NOTIFY_INTERVAL_MS) {
                                lastNotify = now
                                TranscriptionService.update(
                                    getApplication(), "Whisper Transcriber", "Decoding audio… $pct%", pct
                                )
                            }
                        }
                    )
                }
                if (pcmFile.exists()) pcmFile.delete()
                check(tmpPcm.renameTo(pcmFile)) { "Could not move the decoded audio into the cache" }
                sampleCount = decodeResult.sampleCount
                decodeMs = decodeResult.decodeMs
                withContext(Dispatchers.IO) { evictAudioCache(keep = pcmFile) }
                refreshAudioCacheSize()
                val statsStr = "⚡ Decoded in ${com.emilio.whispert.ui.formatDurationMs(decodeMs)} " +
                    "via ${decodeResult.engine} (${"%.0f".format(decodeResult.speedX)}x realtime)"
                _lastDecodeInfo.value = statsStr
                Log.i(LOG_TAG, statsStr)
            } catch (t: Throwable) {
                tmpPcm.delete()
                throw t
            }
        }

        val throttle = ProgressThrottle()
        _transcribeState.value = TranscribeUiState.Busy("Whisper running… ($sampleCount samples)", progress = 0f)
        TranscriptionService.update(getApplication(), "Whisper Transcriber", "Whisper transcribing… 0%", 0)

        val whisperStart = System.currentTimeMillis()
        val result = ctx.transcribeFile(
            wavPath = pcmFile.absolutePath,
            language = language,
            config = config,
            printTimestamps = false,
            onProgress = throttle::setPercent,
            onSegment = throttle::addSegment,
        )
        val whisperMs = System.currentTimeMillis() - whisperStart
        throttle.emit(force = true)

        val diarization = diarize(result.segments, config) { mappedPcmSource(pcmFile) }

        publishDone(
            result = result,
            diarization = diarization,
            decodeMs = decodeMs,
            whisperMs = whisperMs,
            language = language,
            desc = activeDesc,
            sampleCount = sampleCount,
            ctx = ctx,
            pcmFile = pcmFile,
            openPcm = { mappedPcmSource(pcmFile) },
        )
    }

    private suspend fun runSmokeTest() {
        _transcribeState.value = TranscribeUiState.Busy("Checking model…")
        val ready = ensureModelLoaded()
        if (ready.err != null) {
            _transcribeState.value = TranscribeUiState.Error(ready.err)
            return
        }
        val ctx = ready.ctx!!
        val activeDesc = ready.desc
        val language = effectiveLanguage(activeDesc)
        val config = _settings.value

        _transcribeState.value = TranscribeUiState.Busy("Reading test audio from assets…", progress = 0f)
        val decodeStart = System.currentTimeMillis()
        val samples = withContext(Dispatchers.IO) {
            getApplication<Application>().assets.open(TEST_SAMPLE).use {
                WavHelper.decodePcm16ToMonoFloat(it)
            }
        }
        val decodeMs = System.currentTimeMillis() - decodeStart

        val throttle = ProgressThrottle()
        _transcribeState.value = TranscribeUiState.Busy("Whisper running… (${samples.size} samples)", progress = 0f)

        val whisperStart = System.currentTimeMillis()
        val result = ctx.transcribe(
            audioData = samples,
            language = language,
            config = config,
            printTimestamps = false,
            onProgress = throttle::setPercent,
            onSegment = throttle::addSegment,
        )
        val whisperMs = System.currentTimeMillis() - whisperStart
        throttle.emit(force = true)

        val diarization = diarize(result.segments, config) { InMemoryPcmSource(samples) }

        publishDone(
            result = result,
            diarization = diarization,
            decodeMs = decodeMs,
            whisperMs = whisperMs,
            language = language,
            desc = activeDesc,
            sampleCount = samples.size,
            ctx = ctx,
            pcmFile = null,
            openPcm = { InMemoryPcmSource(samples) },
        )
    }

    /**
     * Post-processing speaker diarization, shared by both entry points.
     * [openPcm] is called lazily so nothing is mapped when the toggle is off.
     */
    private suspend fun diarize(
        segments: List<Segment>,
        config: WhisperConfig,
        openPcm: () -> PcmSource,
    ): SpeakerDiarization.Outcome {
        if (!_identifySpeakers.value || segments.isEmpty()) {
            return SpeakerDiarization.Outcome(segments, null, 0)
        }
        _transcribeState.value = TranscribeUiState.Busy("Identifying speakers… 0%", progress = 0f)
        return try {
            withContext(Dispatchers.Default) {
                val useNeural = config.diarizationEngine == DiarizationEngine.NEURAL_ONNX &&
                    speakerModelManager.isDownloaded()
                val modelFile = if (useNeural) speakerModelManager.getModelFile() else null
                val threads = if (config.customThreads > 0) config.customThreads else autoThreadCount
                var lastNotify = 0L
                openPcm().use { pcm ->
                    SpeakerDiarization.run(
                        pcm = pcm,
                        segments = segments,
                        speakerModelFile = modelFile,
                        threads = threads,
                        mergeThresholdOverride = config.speakerMergeThreshold,
                        expectedSpeakers = config.expectedSpeakers,
                        onProgress = { p ->
                            val pct = (p * 100).toInt()
                            _transcribeState.value = TranscribeUiState.Busy(
                                message = "Identifying speakers… $pct%",
                                progress = p,
                            )
                            val now = System.currentTimeMillis()
                            if (now - lastNotify >= NOTIFY_INTERVAL_MS) {
                                lastNotify = now
                                TranscriptionService.update(
                                    getApplication(), "Whisper Transcriber", "Identifying speakers… $pct%", pct
                                )
                            }
                        },
                    )
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (t: Throwable) {
            // Degrading to an undiarized result is the right call - the
            // transcript itself is fine - but it used to happen silently, so a
            // crashed engine was indistinguishable from the feature being off.
            Log.e(LOG_TAG, "Post-processing diarization failed", t)
            SpeakerDiarization.Outcome(segments, null, 0, failure = describe(t))
        }
    }

    /**
     * Everything needed to replay the diarization pass on its own. The decoded
     * audio is already in the cache, so a replay costs one embedding sweep
     * instead of a decode plus a full whisper run.
     */
    private class LastRun(
        val result: com.emilio.whispert.whisper.TranscriptionResult,
        val decodeMs: Long,
        val whisperMs: Long,
        val language: String,
        val desc: ModelDescriptor,
        val sampleCount: Int,
        val modelInfo: String?,
        val timings: com.emilio.whispert.whisper.WhisperTimings,
        /** null for the bundled smoke test, whose PCM lives in memory. */
        val pcmFile: File?,
        val openPcm: () -> PcmSource,
    )

    private var lastRun: LastRun? = null

    private val _canRediarize = MutableStateFlow(false)
    val canRediarize: StateFlow<Boolean> = _canRediarize.asStateFlow()

    /**
     * Re-runs only the speaker pass over the last transcription, picking up the
     * current engine and *Identify speakers* setting. Nothing is decoded and
     * whisper does not run again.
     */
    fun rediarize() {
        val last = lastRun ?: return
        if (last.pcmFile != null && !last.pcmFile.exists()) {
            _transcribeState.value = TranscribeUiState.Error(
                "The decoded audio is no longer cached. Run the transcription again."
            )
            return
        }
        startTranscription {
            val diarization = diarize(last.result.segments, _settings.value, last.openPcm)
            emitDone(last, diarization)
        }
    }

    /**
     * Drops the speaker labels and republishes the plain transcript.
     *
     * Purely a re-render of the run we already have: [LastRun.result] keeps the
     * segments exactly as whisper produced them, before any speaker pass, so
     * this costs nothing and is reversible with [rediarize].
     */
    fun clearDiarization() {
        val last = lastRun ?: return
        if (transcribeJob?.isActive == true) return
        emitDone(last, SpeakerDiarization.Outcome(last.result.segments, null, 0))
    }

    private suspend fun publishDone(
        result: com.emilio.whispert.whisper.TranscriptionResult,
        diarization: SpeakerDiarization.Outcome,
        decodeMs: Long,
        whisperMs: Long,
        language: String,
        desc: ModelDescriptor,
        sampleCount: Int,
        ctx: WhisperContext,
        pcmFile: File?,
        openPcm: () -> PcmSource,
    ) {
        val modelInfo = runCatching { ctx.modelInfo() }.getOrNull()
        Log.d(LOG_TAG, "modelInfo: $modelInfo")
        val last = LastRun(
            result = result,
            decodeMs = decodeMs,
            whisperMs = whisperMs,
            language = language,
            desc = desc,
            sampleCount = sampleCount,
            modelInfo = modelInfo,
            timings = result.timings,
            pcmFile = pcmFile,
            openPcm = openPcm,
        )
        lastRun = last
        _canRediarize.value = true
        emitDone(last, diarization)
    }

    /** Renders a [LastRun] plus a diarization outcome into the Done state. */
    private fun emitDone(last: LastRun, diarization: SpeakerDiarization.Outcome) {
        val result = last.result
        val finalSegments = diarization.segments
        val finalText = if (finalSegments.any { it.speaker != null }) {
            formatTranscriptWithSpeakers(finalSegments)
        } else {
            result.text
        }

        _transcribeState.value = TranscribeUiState.Done(
            text = finalText,
            segments = finalSegments,
            elapsedMs = last.decodeMs + last.whisperMs + diarization.totalMs,
            decodeMs = last.decodeMs,
            whisperMs = last.whisperMs,
            diarizeMs = diarization.totalMs,
            diarizeExtractMs = diarization.extractMs,
            diarizeClusterMs = diarization.clusterMs,
            diarizeEngine = diarization.engineLabel,
            speakerCount = diarization.speakerCount,
            diarizeThreshold = diarization.mergeThreshold,
            diarizeStats = diarization.stats,
            diarizeError = diarization.failure,
            timings = last.timings,
            language = last.language,
            modelName = last.desc.displayName,
            modelId = last.desc.id,
            sampleCount = last.sampleCount,
            modelInfo = last.modelInfo,
        )
    }

    private fun effectiveLanguage(desc: ModelDescriptor): String =
        if (desc.isEnglishOnly) "en" else _language.value

    private data class ModelReady(val ctx: WhisperContext?, val desc: ModelDescriptor, val err: String?)

    private suspend fun ensureModelLoaded(): ModelReady {
        val active = modelManager.activeModel() ?: return ModelReady(
            null, ModelRegistry.ALL.first(),
            "No model yet. Go to the Models tab and download one."
        )

        if (loadedModelId != active.id || whisperCtx == null) {
            _transcribeState.value = TranscribeUiState.Busy("Loading model '${active.displayName}'…")
            whisperCtx?.release()
            whisperCtx = null
            loadedModelId = null
            val file = modelManager.cachedFile(active)
                ?: return ModelReady(null, active, "Model '${active.displayName}' file is missing. Re-download it.")
            val ctx = try {
                withContext(Dispatchers.IO) { WhisperContext.createContextFromFile(file.absolutePath) }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                return ModelReady(null, active, describe(t))
            }
            whisperCtx = ctx
            loadedModelId = active.id
        }
        return ModelReady(whisperCtx, active, null)
    }

    /** Unwraps the class-initializer wrapper so CPU/library errors read sensibly. */
    private fun describe(t: Throwable): String {
        val root = generateSequence(t) { it.cause }.last()
        return root.message ?: root.toString()
    }

    /**
     * Writes [content] to a SAF-selected destination [uri] (typically Downloads
     * or Documents). Returns true on success.
     */
    suspend fun saveTranscriptTo(content: String, uri: Uri): Boolean = withContext(Dispatchers.IO) {
        try {
            getApplication<Application>().contentResolver.openOutputStream(uri, "w")?.use { out ->
                out.write(content.toByteArray())
                true
            } ?: false
        } catch (t: Throwable) {
            Log.w(LOG_TAG, "Export failed", t)
            false
        }
    }

    override fun onCleared() {
        super.onCleared()
        WhisperLib.cancelTranscription()
        transcribeJob?.cancel()
        transcribeJob = null
        TranscriptionService.stop(getApplication())
        // Fire-and-forget: the free runs on the inference thread, which the
        // executor keeps alive until the task completes. Blocking the main
        // thread here risked an ANR on large models.
        whisperCtx?.releaseAsync()
        whisperCtx = null
        loadedModelId = null
    }

    companion object {
        private const val LOG_TAG = "MainViewModel"
        private const val PREFS = "whispert"
        private const val KEY_LANGUAGE = "language"
        private const val TEST_SAMPLE = "samples/jfk.wav"

        /** Roughly 9 hours of decoded 16 kHz mono audio. */
        private const val MAX_AUDIO_CACHE_BYTES = 1_000_000_000L

        private const val UI_EMIT_INTERVAL_MS = 150L
        private const val NOTIFY_INTERVAL_MS = 1000L
    }
}

// ---------- UI state contracts ----------

sealed interface TranscribeUiState {
    data object Idle : TranscribeUiState
    data class Busy(
        val message: String,
        val progress: Float? = null,
        val partialSegments: List<Segment> = emptyList(),
        val partialText: String = "",
    ) : TranscribeUiState
    data class Done(
        val text: String,
        val segments: List<Segment>,
        val elapsedMs: Long,
        val decodeMs: Long,
        val whisperMs: Long,
        val diarizeMs: Long = 0L,
        val diarizeExtractMs: Long = 0L,
        val diarizeClusterMs: Long = 0L,
        val diarizeEngine: String? = null,
        val speakerCount: Int = 0,
        val diarizeThreshold: Float = 0f,
        val diarizeStats: com.emilio.whispert.whisper.DiarizationStats =
            com.emilio.whispert.whisper.DiarizationStats(),
        /** Non-null when speaker identification was requested but failed. */
        val diarizeError: String? = null,
        val timings: com.emilio.whispert.whisper.WhisperTimings,
        val language: String,
        val modelName: String,
        val modelId: String,
        val sampleCount: Int,
        val modelInfo: String?,
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
    val totalRamGb: Int,
    val modelStates: Map<String, ModelItemState>,
    val activeModelId: String?,
)
