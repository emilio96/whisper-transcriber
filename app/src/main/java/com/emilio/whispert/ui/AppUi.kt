package com.emilio.whispert.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.res.painterResource
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.NavigationDrawerItemDefaults
import androidx.compose.foundation.selection.selectable
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberDrawerState
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.height
import androidx.compose.material.icons.filled.Delete
import com.emilio.whispert.whisper.DecoderEngine
import com.emilio.whispert.whisper.DiarizationEngine
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.emilio.whispert.MainViewModel
import com.emilio.whispert.ThemeMode
import com.emilio.whispert.models.ModelFamily
import com.emilio.whispert.whisper.toTimestamp
import com.emilio.whispert.ModelItemState
import com.emilio.whispert.ModelsUiState
import com.emilio.whispert.TranscribeUiState
import com.emilio.whispert.whisper.WhisperConfig
import kotlinx.coroutines.launch
import java.util.Locale

enum class AppScreen(val title: String) {
    TRANSCRIBE("Transcribe"),
    MODELS("Models"),
    SETTINGS("Model settings")
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(vm: MainViewModel = viewModel()) {
    val audioUri by vm.audioUri.collectAsState()
    val language by vm.language.collectAsState()
    val transcribeState by vm.transcribeState.collectAsState()
    val modelsState by vm.modelsState.collectAsState()
    val settings by vm.settings.collectAsState()
    val audioCacheBytes by vm.audioCacheSizeBytes.collectAsState()
    val themeMode by vm.themeMode.collectAsState()
    val keepScreenAwake by vm.keepScreenAwake.collectAsState()
    val identifySpeakers by vm.identifySpeakers.collectAsState()
    val lastDecodeInfo by vm.lastDecodeInfo.collectAsState()
    val speakerModelDownloaded by vm.speakerModelDownloaded.collectAsState()
    val speakerModelDownloading by vm.speakerModelDownloading.collectAsState()
    val speakerModelProgress by vm.speakerModelProgress.collectAsState()
    val speakerModelError by vm.speakerModelError.collectAsState()
    val canRediarize by vm.canRediarize.collectAsState()

    var currentScreen by rememberSaveable { mutableStateOf(AppScreen.TRANSCRIBE) }
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 20.dp, vertical = 20.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    Image(
                        painter = painterResource(id = com.emilio.whispert.R.drawable.ic_app_logo),
                        contentDescription = "Whisper Transcriber Logo",
                        modifier = Modifier
                            .size(48.dp)
                            .clip(RoundedCornerShape(12.dp))
                    )
                    Column {
                        Text(
                            "Whisper Transcriber",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                        Text(
                            "On-Device AI Engine",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
                HorizontalDivider(modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp))

                NavigationDrawerItem(
                    label = { Text("Transcribe") },
                    selected = currentScreen == AppScreen.TRANSCRIBE,
                    onClick = {
                        currentScreen = AppScreen.TRANSCRIBE
                        scope.launch { drawerState.close() }
                    },
                    modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding)
                )
                NavigationDrawerItem(
                    label = { Text("Models") },
                    selected = currentScreen == AppScreen.MODELS,
                    onClick = {
                        currentScreen = AppScreen.MODELS
                        scope.launch { drawerState.close() }
                    },
                    modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding)
                )
                NavigationDrawerItem(
                    label = { Text("Model settings") },
                    selected = currentScreen == AppScreen.SETTINGS,
                    onClick = {
                        currentScreen = AppScreen.SETTINGS
                        scope.launch { drawerState.close() }
                    },
                    modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding)
                )

                // Drawer footer: Keep Screen Awake + Theme Mode Selector
                Spacer(Modifier.weight(1f))
                HorizontalDivider()
                Column(modifier = Modifier.padding(16.dp)) {
                    // Keep Screen Awake Toggle
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                "Keep screen awake",
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Medium,
                            )
                            Spacer(Modifier.size(4.dp))
                            InfoHelpButton(
                                title = "Keep Screen Awake",
                                explanation = "Prevents the screen from turning off while transcription is running.\n\nSmart Dimming: To save battery and protect the display, brightness will automatically dim after your system screen timeout is reached. Tapping anywhere on the screen instantly restores normal brightness."
                            )
                        }
                        Switch(
                            checked = keepScreenAwake,
                            onCheckedChange = vm::setKeepScreenAwake
                        )
                    }

                    Text(
                        "Theme",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(Modifier.size(8.dp))
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(MaterialTheme.colorScheme.surfaceVariant, shape = RoundedCornerShape(24.dp))
                            .padding(4.dp),
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        ThemeMode.entries.forEach { mode ->
                            val isSelected = themeMode == mode
                            Surface(
                                modifier = Modifier.weight(1f),
                                shape = RoundedCornerShape(20.dp),
                                color = if (isSelected) MaterialTheme.colorScheme.primary else Color.Transparent,
                                onClick = { vm.setThemeMode(mode) }
                            ) {
                                Box(
                                    contentAlignment = Alignment.Center,
                                    modifier = Modifier.padding(vertical = 8.dp)
                                ) {
                                    Text(
                                        text = when (mode) {
                                            ThemeMode.LIGHT -> "☀️ Light"
                                            ThemeMode.DARK -> "🌙 Dark"
                                            ThemeMode.SYSTEM -> "⚙️ Auto"
                                        },
                                        style = MaterialTheme.typography.labelSmall,
                                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                                        color = if (isSelected) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text(currentScreen.title, fontWeight = FontWeight.SemiBold) },
                    navigationIcon = {
                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                            Icon(
                                imageVector = Icons.Default.Menu,
                                contentDescription = "Menu"
                            )
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainer
                    )
                )
            }
        ) { inner ->
            Box(modifier = Modifier.padding(inner).fillMaxSize()) {
                when (currentScreen) {
                    AppScreen.TRANSCRIBE -> TranscribeTab(
                        audioUri = audioUri,
                        language = language,
                        state = transcribeState,
                        identifySpeakers = identifySpeakers,
                        onIdentifySpeakersChange = vm::setIdentifySpeakers,
                        expectedSpeakers = settings.expectedSpeakers,
                        onExpectedSpeakersChange = { vm.updateSettings(settings.copy(expectedSpeakers = it)) },
                        onPickAudioPicked = vm::onAudioPicked,
                        onClearAudio = { vm.clearAudio() },
                        onLanguageChange = { vm.setLanguage(it) },
                        onTranscribe = { vm.transcribe() },
                        onTranscribeJfk = { vm.transcribeBundledJfk() },
                        onRediarize = { vm.rediarize() },
                        canRediarize = canRediarize,
                        onCancel = { vm.cancelTranscription() },
                        onSaveTranscript = { text, uri ->
                            scope.launch { vm.saveTranscriptTo(text, uri) }
                        },
                        onSaveReport = { text, uri ->
                            scope.launch { vm.saveTranscriptTo(text, uri) }
                        },
                        activeModelId = modelsState.activeModelId,
                    )
                    AppScreen.MODELS -> ModelsTab(
                        state = modelsState,
                        onDownload = { vm.downloadModel(it) },
                        onDelete = { vm.deleteModel(it) },
                        onActivate = { vm.activateModel(it) },
                        onImportModelUri = { vm.importModel(it) }
                    )
                    AppScreen.SETTINGS -> ModelSettingsTab(
                        settings = settings,
                        onUpdateSettings = { vm.updateSettings(it) },
                        onResetSettings = { vm.resetSettings() },
                        audioCacheBytes = audioCacheBytes,
                        onClearAudioCache = { vm.clearAudioCache() },
                        cpuCoreCount = vm.deviceCoreCount,
                        autoThreadCount = vm.autoThreadCount,
                        hardwareInfoSummary = vm.hardwareInfoSummary,
                        speakerModelDownloaded = speakerModelDownloaded,
                        speakerModelDownloading = speakerModelDownloading,
                        speakerModelProgress = speakerModelProgress,
                        speakerModelError = speakerModelError,
                        onDownloadSpeakerModel = { vm.downloadSpeakerModel() },
                        onDeleteSpeakerModel = { vm.deleteSpeakerModel() },
                    )
                }
            }
        }
    }
}

/** Formats milliseconds into human-readable scaling units: e.g. 12.4s, 1m 32s, 1h 14m 20s. */
fun formatDurationMs(ms: Long): String {
    if (ms < 0) return "0.0s"
    return formatDurationSeconds(ms / 1000.0)
}

/** Formats fractional seconds into human-readable scaling units: e.g. 12.4s, 1m 32s, 1h 14m 20s. */
fun formatDurationSeconds(sec: Double): String {
    if (sec < 0) return "0.0s"
    if (sec < 60.0) {
        return String.format(Locale.US, "%.1fs", sec)
    }
    val totalSeconds = sec.toLong()
    val hours = totalSeconds / 3600
    val remainderSeconds = totalSeconds % 3600
    val minutes = remainderSeconds / 60
    val seconds = remainderSeconds % 60
    val tenths = ((sec - totalSeconds) * 10).toInt().coerceIn(0, 9)

    return if (hours > 0) {
        if (tenths > 0) {
            String.format(Locale.US, "%dh %dm %d.%ds", hours, minutes, seconds, tenths)
        } else {
            String.format(Locale.US, "%dh %dm %ds", hours, minutes, seconds)
        }
    } else {
        if (tenths > 0) {
            String.format(Locale.US, "%dm %d.%ds", minutes, seconds, tenths)
        } else {
            String.format(Locale.US, "%dm %ds", minutes, seconds)
        }
    }
}

@Composable
private fun TranscribeTab(
    audioUri: Uri?,
    language: String,
    state: TranscribeUiState,
    identifySpeakers: Boolean,
    onIdentifySpeakersChange: (Boolean) -> Unit,
    expectedSpeakers: Int,
    onExpectedSpeakersChange: (Int) -> Unit,
    onPickAudioPicked: (Uri) -> Unit,
    onClearAudio: () -> Unit,
    onLanguageChange: (String) -> Unit,
    onTranscribe: () -> Unit,
    onTranscribeJfk: () -> Unit,
    onRediarize: () -> Unit,
    canRediarize: Boolean,
    onCancel: () -> Unit,
    onSaveTranscript: (text: String, uri: Uri) -> Unit,
    onSaveReport: (text: String, uri: Uri) -> Unit,
    activeModelId: String?,
) {
    // SAF picker: any audio/* file from Documents/Downloads etc.
    val pickAudio = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri -> if (uri != null) onPickAudioPicked(uri) }

    // Saveable: the CreateDocument dialog can outlive the Activity (rotation,
    // process death). With a plain remember the pending payload came back null
    // and the file the user had just chosen was written empty - silently.
    var pendingSaveText by rememberSaveable { mutableStateOf<String?>(null) }
    val saveTxt = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("text/plain"),
    ) { uri ->
        if (uri != null) {
            val text = pendingSaveText
            if (text != null) onSaveTranscript(text, uri)
        }
        pendingSaveText = null
    }

    var pendingSaveReport by rememberSaveable { mutableStateOf<String?>(null) }
    val saveReport = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("text/markdown"),
    ) { uri ->
        if (uri != null) {
            val report = pendingSaveReport
            if (report != null) onSaveReport(report, uri)
        }
        pendingSaveReport = null
    }

    var showDiagnostics by rememberSaveable { mutableStateOf(false) }

    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        // Active model summary
        Text("Active model: ${activeModelId ?: "none"}",
             style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)

        // Audio picker row
        Row(verticalAlignment = Alignment.CenterVertically) {
            Button(
                onClick = { pickAudio.launch(arrayOf("audio/*")) },
                enabled = state !is TranscribeUiState.Busy,
            ) {
                Text("Pick audio file")
            }
            Spacer(Modifier.size(8.dp))
            if (audioUri != null) {
                OutlinedButton(
                    onClick = onClearAudio,
                    enabled = state !is TranscribeUiState.Busy,
                ) {
                    Text("Clear")
                }
            }
        }
        Text(audioUri?.let { "File: ${shortened(it)}" } ?: "Pick a mp3/m4a/wav/opus audio file.",
             style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)

        // Language selector (only relevant for multilingual models; EN-only pins 'en').
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Language: ", style = MaterialTheme.typography.bodyMedium)
            InfoHelpButton(
                title = "Language Selection",
                explanation = "Selecting the exact language (e.g. IT or EN) instead of AUTO significantly speeds up transcription (up to 2x faster).\n\nWhen set to AUTO, Whisper must execute an extra full neural network pass to detect the language across 99 candidates. Selecting the language explicitly skips this pass and begins transcription immediately."
            )
            Spacer(Modifier.size(2.dp))
            listOf("auto", "en", "it").forEach { lang ->
                FilterChip(
                    selected = language == lang,
                    onClick = { onLanguageChange(lang) },
                    label = { Text(lang.uppercase(Locale.ROOT)) },
                    enabled = state !is TranscribeUiState.Busy,
                )
            }
        }

        // Speaker Diarization / Identification toggle
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "Identify speakers",
                    style = MaterialTheme.typography.bodyMedium,
                )
                Spacer(Modifier.size(4.dp))
                InfoHelpButton(
                    title = "Speaker Diarization",
                    explanation = "Distinguishes between different speakers in the conversation (e.g. Speaker 1, Speaker 2) using acoustic neural embeddings and voice clustering."
                )
            }
            Switch(
                checked = identifySpeakers,
                onCheckedChange = onIdentifySpeakersChange,
                enabled = state !is TranscribeUiState.Busy
            )
        }

        // Speaker count, only while the toggle above is on: it is meaningless
        // otherwise, and this row is the first thing on an otherwise plain screen.
        AnimatedVisibility(visible = identifySpeakers) {
            Column(modifier = Modifier.fillMaxWidth().padding(bottom = 4.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Expected speakers", style = MaterialTheme.typography.bodySmall)
                    Spacer(Modifier.size(4.dp))
                    InfoHelpButton(
                        title = "Expected Speakers",
                        explanation = "When you know how many people are on the recording, set it here. The speakers are then found by cutting the voice dendrogram at that count instead of at a fixed distance.\n\nThis asks much less of the audio: the voices only have to be the last groups to join, rather than sitting on opposite sides of one absolute threshold. On noisy recordings — phone microphones, rooms, overlapping speech — no single threshold works, and this is the more reliable way to get the right count.\n\nWith Auto, the merge threshold in Model settings decides both the grouping and how many speakers there are."
                    )
                }
                Spacer(Modifier.size(6.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(0, 1, 2, 3, 4, 5, 6).forEach { k ->
                        FilterChip(
                            selected = expectedSpeakers == k,
                            onClick = { onExpectedSpeakersChange(k) },
                            enabled = state !is TranscribeUiState.Busy,
                            label = { Text(if (k == 0) "Auto" else "$k") },
                        )
                    }
                }
            }
        }

        // Action row
        Row(verticalAlignment = Alignment.CenterVertically) {
            if (state is TranscribeUiState.Busy) {
                Button(
                    onClick = onCancel,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error,
                        contentColor = MaterialTheme.colorScheme.onError,
                    ),
                ) {
                    Text("Cancel transcription")
                }
            } else {
                Button(onClick = onTranscribe, enabled = audioUri != null) {
                    Text("Transcribe")
                }
                Spacer(Modifier.size(12.dp))
                OutlinedButton(onClick = onTranscribeJfk) {
                    Text("Smoke test (jfk.wav / EN)")
                }
            }
        }

        HorizontalDivider()
        when (val s = state) {
            TranscribeUiState.Idle -> Text("Idle. Pick an audio file and press Transcribe.")
            is TranscribeUiState.Busy -> Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.5.dp)
                    Spacer(Modifier.size(10.dp))
                    Text("⏳ ${s.message}", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                }
                if (s.progress != null) {
                    LinearProgressIndicator(
                        progress = { s.progress },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                    )
                }
                if (s.partialSegments.isNotEmpty() || s.partialText.isNotBlank()) {
                    OutlinedCard(
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
                    ) {
                        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween,
                            ) {
                                Text(
                                    "Streaming Transcript (${s.partialSegments.size} segments)",
                                    style = MaterialTheme.typography.labelMedium,
                                    fontWeight = FontWeight.Bold,
                                    color = MaterialTheme.colorScheme.primary,
                                )
                                Text(
                                    "● LIVE",
                                    style = MaterialTheme.typography.labelSmall,
                                    fontWeight = FontWeight.Bold,
                                    color = MaterialTheme.colorScheme.primary,
                                )
                            }
                            HorizontalDivider()
                            LazyColumn(
                                modifier = Modifier.fillMaxWidth().heightIn(max = 260.dp),
                            ) {
                                items(s.partialSegments) { seg ->
                                    val speakerPrefix = if (seg.speaker != null) "[${seg.speaker}] " else ""
                                    Text(
                                        "$speakerPrefix${seg.text.trim()}",
                                        modifier = Modifier.padding(vertical = 2.dp),
                                        style = MaterialTheme.typography.bodySmall,
                                        fontFamily = FontFamily.Monospace,
                                    )
                                }
                            }
                        }
                    }
                }
            }
            is TranscribeUiState.Done -> {
                val audioSec = s.sampleCount / 16_000.0
                val rtf = if (audioSec > 0) (s.whisperMs / 1000.0) / audioSec else 0.0
                val t = s.timings
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    // 1. Full transcript surface (front and center)
                    Surface(
                        tonalElevation = 2.dp,
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        if (s.segments.isEmpty()) {
                            Text(
                                s.text.ifBlank { "(empty transcript)" },
                                modifier = Modifier.padding(14.dp),
                                style = MaterialTheme.typography.bodySmall,
                                fontFamily = FontFamily.Monospace,
                            )
                        } else {
                            LazyColumn(
                                modifier = Modifier.fillMaxWidth().heightIn(max = 340.dp),
                            ) {
                                items(s.segments) { seg ->
                                    val speakerPrefix = if (seg.speaker != null) "[${seg.speaker}] " else ""
                                    Text(
                                        "$speakerPrefix${seg.text.trim()}",
                                        modifier = Modifier.padding(horizontal = 14.dp, vertical = 3.dp),
                                        style = MaterialTheme.typography.bodySmall,
                                        fontFamily = FontFamily.Monospace,
                                    )
                                }
                            }
                        }
                    }
                    Text(
                        "${s.segments.size} segments recognized",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )

                    // 2. Action buttons
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Button(onClick = {
                            pendingSaveText = s.text
                            saveTxt.launch("transcript_${System.currentTimeMillis()}.txt")
                        }) { Text("Save .txt") }
                        Spacer(Modifier.size(10.dp))
                        OutlinedButton(onClick = {
                            pendingSaveReport = buildReport(s)
                            saveReport.launch("whisper_report_${System.currentTimeMillis()}.md")
                        }) { Text("Save report (.md)") }
                        Spacer(Modifier.size(10.dp))
                        OutlinedButton(onClick = { onTranscribe() }) { Text("Re-run") }
                    }
                    // Replays the speaker pass alone, on the audio already in
                    // cache: no decode, no whisper. This is the loop for trying
                    // a different diarization engine or setting.
                    if (canRediarize) {
                        OutlinedButton(onClick = onRediarize) { Text("Re-run diarization only") }
                    }

                    // 3. Collapsible dropdown / accordion for statistics & diagnostics
                    OutlinedCard(
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.fillMaxWidth(),
                        onClick = { showDiagnostics = !showDiagnostics },
                    ) {
                        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween,
                            ) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                ) {
                                    Text(
                                        "Statistics & Diagnostics",
                                        style = MaterialTheme.typography.titleSmall,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Surface(
                                        shape = RoundedCornerShape(12.dp),
                                        color = MaterialTheme.colorScheme.secondaryContainer,
                                    ) {
                                        Text(
                                            "${formatDurationMs(s.elapsedMs)} · ${String.format(Locale.US, "%.1fx", rtf)} RTF",
                                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
                                            style = MaterialTheme.typography.labelSmall,
                                            fontWeight = FontWeight.Bold,
                                            color = MaterialTheme.colorScheme.onSecondaryContainer,
                                        )
                                    }
                                }
                                val rotation by animateFloatAsState(targetValue = if (showDiagnostics) 180f else 0f, label = "chevron")
                                Icon(
                                    imageVector = Icons.Default.KeyboardArrowDown,
                                    contentDescription = if (showDiagnostics) "Collapse diagnostics" else "Expand diagnostics",
                                    modifier = Modifier.rotate(rotation),
                                )
                            }

                            AnimatedVisibility(visible = showDiagnostics) {
                                Column(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 4.dp)) {
                                    HorizontalDivider()
                                    val diarizeSummary = if (s.diarizeEngine != null) {
                                        val speakersStr = if (s.speakerCount > 0) "${s.speakerCount} detected" else "None"
                                        "• Speaker diarization: ${formatDurationMs(s.diarizeMs)} (${s.diarizeEngine})\n" +
                                        "• Embedding extraction: ${formatDurationMs(s.diarizeExtractMs)}, clustering: ${formatDurationMs(s.diarizeClusterMs)}\n" +
                                        "• Speakers identified: $speakersStr\n" +
                                        clusteringDetail(s)
                                    } else if (s.diarizeError != null) {
                                        "• Speaker diarization: FAILED - ${s.diarizeError}\n" +
                                        "• The transcript below is complete but has no speaker labels.\n"
                                    } else {
                                        "• Speaker diarization: Disabled\n"
                                    }
                                    val breakdown = if (s.diarizeMs > 0) {
                                        " (decode: ${formatDurationMs(s.decodeMs)} + whisper: ${formatDurationMs(s.whisperMs)} + diarization: ${formatDurationMs(s.diarizeMs)})"
                                    } else {
                                        " (decode: ${formatDurationMs(s.decodeMs)} + whisper: ${formatDurationMs(s.whisperMs)})"
                                    }
                                    Text(
                                        "• Audio duration: ${formatDurationSeconds(audioSec)} (${s.sampleCount} samples)\n" +
                                        "• Total elapsed time: ${formatDurationMs(s.elapsedMs)}$breakdown\n" +
                                        "• Audio decoding: ${formatDurationMs(s.decodeMs)}\n" +
                                        "• Whisper inference: ${formatDurationMs(s.whisperMs)}\n" +
                                        diarizeSummary +
                                        "• Realtime speed factor: ${String.format(Locale.US, "%.2fx", rtf)} (faster than 1.0x is realtime)\n" +
                                        "• Language: ${s.language.uppercase(Locale.ROOT)}\n" +
                                        "• Active model: ${s.modelName} (${s.modelId})",
                                        style = MaterialTheme.typography.bodySmall,
                                        fontFamily = FontFamily.Monospace,
                                        color = MaterialTheme.colorScheme.onSurface,
                                    )
                                    HorizontalDivider()
                                    Text(
                                        "Whisper internal phases:\n" +
                                        "• encode: ${formatDurationMs(t.encodeMs.toLong())} (${t.encodeMs} ms)\n" +
                                        "• decode: ${formatDurationMs(t.decodeMs.toLong())} (${t.decodeMs} ms)\n" +
                                        "• sample: ${formatDurationMs(t.sampleMs.toLong())} (${t.sampleMs} ms)\n" +
                                        "• batchd: ${formatDurationMs(t.batchdMs.toLong())} (${t.batchdMs} ms)\n" +
                                        "• prompt: ${formatDurationMs(t.promptMs.toLong())} (${t.promptMs} ms)",
                                        style = MaterialTheme.typography.labelSmall,
                                        fontFamily = FontFamily.Monospace,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    )
                                    s.modelInfo?.let { mi ->
                                        HorizontalDivider()
                                        Text(
                                            "Model & CPU backend capabilities:\n$mi",
                                            style = MaterialTheme.typography.labelSmall,
                                            fontFamily = FontFamily.Monospace,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
            is TranscribeUiState.Error -> Text("✗ ${s.message}", color = MaterialTheme.colorScheme.error)
        }
    }
}

/**
 * Cluster-level view of the speaker pass: how many clusters the dendrogram cut
 * produced, how many survived the minority fold, and the two distances that
 * decide both. This is what the merge threshold has to be calibrated against.
 */
private fun clusteringDetail(s: TranscribeUiState.Done): String {
    val st = s.diarizeStats
    if (st.clustersAtCut <= 0) return ""
    fun dist(v: Float) = if (v < 0f) "n/a" else String.format(Locale.US, "%.3f", v)
    val cut = if (st.expectedSpeakers > 0) {
        "• Cut: pinned to ${st.expectedSpeakers} speakers (merge threshold unused)\n"
    } else {
        "• Clusters: ${st.clustersAtCut} at the cut -> ${st.clustersFinal} after the minority fold\n" +
            "• Merge threshold: ${String.format(Locale.US, "%.2f", s.diarizeThreshold)}, " +
            "closest surviving pair: ${dist(st.closestClusterDistance)}, " +
            "widest fold: ${dist(st.widestFoldDistance)}\n"
    }
    return cut +
        "• Embedding quality: median nearest-neighbour distance ${dist(st.medianNearestDistance)} " +
        "(under ~0.30 means the voices are separable; near 0.60 means they are not)\n" +
        "• Segment audio: median ${formatDurationMs(st.medianSegmentMs)}, " +
        "${String.format(Locale.US, "%.0f%%", st.shortSegmentFraction * 100)} under 1 s\n"
}

/**
 * Builds a self-contained Markdown report of a finished transcription run.
 */
private fun buildReport(s: TranscribeUiState.Done): String {
    val audioSec = s.sampleCount / 16_000.0
    val rtf = if (audioSec > 0) (s.whisperMs / 1000.0) / audioSec else 0.0
    val t = s.timings
    val ts = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ISO_LOCAL_DATE_TIME)
    return buildString {
        appendLine("# Whisper Transcriber — Report")
        appendLine()
        appendLine("- **Date**: $ts")
        appendLine("- **Model**: ${s.modelName} (${s.modelId})")
        appendLine("- **Language**: ${s.language.uppercase(Locale.ROOT)}")
        appendLine("- **Audio Duration**: ${formatDurationSeconds(audioSec)} (${s.sampleCount} samples)")
        val distinctSpeakers = s.segments.mapNotNull { it.speaker }.distinct()
        if (s.diarizeEngine != null) {
            val spkInfo = if (distinctSpeakers.isNotEmpty()) "${distinctSpeakers.size} (${distinctSpeakers.joinToString(", ")})" else "None detected"
            appendLine("- **Speaker Diarization**: ${s.diarizeEngine} (${formatDurationMs(s.diarizeMs)})")
            appendLine("- **Diarization Breakdown**: embedding extraction ${formatDurationMs(s.diarizeExtractMs)} + clustering ${formatDurationMs(s.diarizeClusterMs)}")
            appendLine("- **Speakers Identified**: $spkInfo")
            clusteringDetail(s).lines().filter { it.isNotBlank() }.forEach {
                appendLine("  - ${it.removePrefix("• ")}")
            }
        } else if (s.diarizeError != null) {
            appendLine("- **Speaker Diarization**: FAILED - ${s.diarizeError}")
        } else if (distinctSpeakers.isNotEmpty()) {
            appendLine("- **Speakers Identified**: ${distinctSpeakers.size} (${distinctSpeakers.joinToString(", ")})")
        }
        val timeBreakdown = if (s.diarizeMs > 0) {
            "decode: ${formatDurationMs(s.decodeMs)} + whisper: ${formatDurationMs(s.whisperMs)} + diarize: ${formatDurationMs(s.diarizeMs)}"
        } else {
            "decode: ${formatDurationMs(s.decodeMs)} + whisper: ${formatDurationMs(s.whisperMs)}"
        }
        appendLine("- **Total Time**: ${formatDurationMs(s.elapsedMs)} ($timeBreakdown)")
        appendLine("- **Realtime Speed Factor (RTF)**: ${String.format(Locale.US, "%.2fx", rtf)}")
        appendLine()
        appendLine("## Whisper Phases")
        appendLine("```")
        appendLine("encode=${t.encodeMs} ms  decode=${t.decodeMs} ms  sample=${t.sampleMs} ms  batchd=${t.batchdMs} ms  prompt=${t.promptMs} ms")
        appendLine("```")
        appendLine()
        appendLine("## Model & Hardware Capabilities")
        appendLine("```")
        appendLine(s.modelInfo ?: "(n/a)")
        appendLine("```")
        appendLine()
        appendLine("## Transcript")
        appendLine("```")
        appendLine(s.text.ifBlank { "(empty transcript)" })
        appendLine("```")
    }
}

@Composable
private fun ModelsTab(
    state: ModelsUiState,
    onDownload: (com.emilio.whispert.models.ModelDescriptor) -> Unit,
    onDelete: (com.emilio.whispert.models.ModelDescriptor) -> Unit,
    onActivate: (com.emilio.whispert.models.ModelDescriptor) -> Unit,
    onImportModelUri: (Uri) -> Unit,
) {
    val pickModel = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri -> if (uri != null) onImportModelUri(uri) }

    val standardFamilies = listOf(
        ModelFamily.TINY,
        ModelFamily.BASE,
        ModelFamily.SMALL,
        ModelFamily.MEDIUM,
        ModelFamily.LARGE_V3_TURBO,
        ModelFamily.LARGE_V3
    )

    val customModels = state.registry.filter { it.family == ModelFamily.CUSTOM }

    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        // Device Profile Banner
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
            )
        ) {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    "📱 Device Capabilities: ${state.deviceDescription}",
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    "Physical RAM: ${state.totalRamGb} GB · Model availability is automatically filtered based on your device specifications.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        Text(
            "Whisper Model Families",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )

        // Standard Model Family Macro-Tiles
        standardFamilies.forEach { family ->
            val familyModels = state.registry.filter { it.family == family }
            if (familyModels.isNotEmpty()) {
                ModelFamilyMacroCard(
                    family = family,
                    models = familyModels,
                    modelStates = state.modelStates,
                    deviceClass = state.deviceClass,
                    totalRamGb = state.totalRamGb,
                    onDownload = onDownload,
                    onDelete = onDelete,
                    onActivate = onActivate
                )
            }
        }

        HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))

        // Custom & SAF Imported Models Card
        CustomModelsCard(
            customModels = customModels,
            modelStates = state.modelStates,
            onPickModel = { pickModel.launch(arrayOf("*/*")) },
            onDelete = onDelete,
            onActivate = onActivate
        )
    }
}

@Composable
private fun ModelFamilyMacroCard(
    family: ModelFamily,
    models: List<com.emilio.whispert.models.ModelDescriptor>,
    modelStates: Map<String, ModelItemState>,
    deviceClass: com.emilio.whispert.models.DeviceClass,
    totalRamGb: Int,
    onDownload: (com.emilio.whispert.models.ModelDescriptor) -> Unit,
    onDelete: (com.emilio.whispert.models.ModelDescriptor) -> Unit,
    onActivate: (com.emilio.whispert.models.ModelDescriptor) -> Unit,
) {
    if (models.isEmpty()) return

    val isCompatible = totalRamGb >= family.minRamGb
    val isRecommended = models.any { deviceClass in it.recommendedFor }

    // Check if any variant is currently active or downloaded
    val activeDesc = models.firstOrNull { modelStates[it.id]?.active == true }
    val downloadedDesc = models.firstOrNull { modelStates[it.id]?.downloaded == true }

    var isEnglishOnly by remember(activeDesc, downloadedDesc) {
        mutableStateOf(
            when {
                activeDesc != null -> activeDesc.isEnglishOnly
                downloadedDesc != null -> downloadedDesc.isEnglishOnly
                else -> false
            }
        )
    }

    val selectedDesc = if (family.supportsEnglishOnly) {
        models.firstOrNull { it.isEnglishOnly == isEnglishOnly } ?: models.first()
    } else {
        models.first()
    }

    val itemState = modelStates[selectedDesc.id] ?: return
    val isVariantActive = itemState.active
    val isFamilyActive = models.any { modelStates[it.id]?.active == true }

    OutlinedCard(
        modifier = Modifier
            .fillMaxWidth()
            .then(if (!isCompatible) Modifier.alpha(0.48f) else Modifier),
        border = if (isVariantActive) BorderStroke(2.dp, MaterialTheme.colorScheme.primary)
                 else if (isFamilyActive) BorderStroke(1.5.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.5f))
                 else BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
        colors = CardDefaults.outlinedCardColors(
            containerColor = if (isVariantActive) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.15f)
                             else MaterialTheme.colorScheme.surface
        )
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            // Header Row: Family name + badges
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            family.displayName,
                            fontWeight = FontWeight.Bold,
                            style = MaterialTheme.typography.titleMedium,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                        Spacer(Modifier.size(8.dp))
                        if (isRecommended && isCompatible) {
                            Surface(
                                color = MaterialTheme.colorScheme.secondaryContainer,
                                shape = RoundedCornerShape(6.dp)
                            ) {
                                Text(
                                    "★ Recommended",
                                    style = MaterialTheme.typography.labelSmall,
                                    fontWeight = FontWeight.SemiBold,
                                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                )
                            }
                        }
                    }
                    Spacer(Modifier.size(2.dp))
                    Text(
                        family.description,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                if (isVariantActive) {
                    Spacer(Modifier.size(8.dp))
                    Surface(
                        color = MaterialTheme.colorScheme.primary,
                        shape = RoundedCornerShape(6.dp)
                    ) {
                        Text(
                            "ACTIVE",
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onPrimary,
                            style = MaterialTheme.typography.labelSmall,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                        )
                    }
                }
            }

            // Specs / Tags Row: Speed + Precision + Size + RAM Requirement
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Surface(
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    shape = RoundedCornerShape(6.dp)
                ) {
                    Text(
                        family.speedBadge,
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Medium,
                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                    )
                }
                Surface(
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    shape = RoundedCornerShape(6.dp)
                ) {
                    Text(
                        "Quality: ${family.qualityBadge}",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Medium,
                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                    )
                }
                Surface(
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    shape = RoundedCornerShape(6.dp)
                ) {
                    Text(
                        "RAM: ≥${family.minRamGb} GB",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Medium,
                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                    )
                }
            }

            // Compatibility Warning if low RAM
            if (!isCompatible) {
                Surface(
                    color = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.5f),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            "⚠️ Incompatible with this device: requires at least ${family.minRamGb} GB RAM (Device has $totalRamGb GB RAM).",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onErrorContainer,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
            }

            // Language Selector Chips (Multilingual vs English Only)
            if (family.supportsEnglishOnly && isCompatible) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    FilterChip(
                        selected = !isEnglishOnly,
                        onClick = { isEnglishOnly = false },
                        label = { Text("🌐 Multilingual (All Languages)") }
                    )
                    FilterChip(
                        selected = isEnglishOnly,
                        onClick = { isEnglishOnly = true },
                        label = { Text("🇬🇧 English Only (.en)") }
                    )
                }
            } else if (!family.supportsEnglishOnly && isCompatible) {
                Text(
                    "🌐 Universal Multilingual (All languages supported)",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Medium
                )
            }

            // Downloading Progress
            if (itemState.downloading) {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text("Downloading model…", style = MaterialTheme.typography.bodySmall)
                        Text(
                            "${(itemState.progress * 100).toInt()}%",
                            style = MaterialTheme.typography.bodySmall,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    LinearProgressIndicator(
                        progress = { itemState.progress },
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }

            if (itemState.error != null) {
                Text("✗ Download failed: ${itemState.error}", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            }

            // Action Buttons
            if (isCompatible) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        "Download size: ${humanSize(selectedDesc.sizeBytes)}",
                        style = MaterialTheme.typography.labelSmall,
                        fontFamily = FontFamily.Monospace,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        if (itemState.downloaded) {
                            if (!itemState.active) {
                                Button(
                                    onClick = { onActivate(selectedDesc) },
                                    enabled = !itemState.downloading
                                ) {
                                    Text("Activate")
                                }
                            }
                            OutlinedButton(
                                onClick = { onDelete(selectedDesc) },
                                enabled = !itemState.downloading
                            ) {
                                Text("Delete")
                            }
                        } else {
                            Button(
                                onClick = { onDownload(selectedDesc) },
                                enabled = !itemState.downloading
                            ) {
                                Text("Download (${humanSize(selectedDesc.sizeBytes)})")
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun CustomModelsCard(
    customModels: List<com.emilio.whispert.models.ModelDescriptor>,
    modelStates: Map<String, ModelItemState>,
    onPickModel: () -> Unit,
    onDelete: (com.emilio.whispert.models.ModelDescriptor) -> Unit,
    onActivate: (com.emilio.whispert.models.ModelDescriptor) -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f))
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(
                "Custom & Imported Models",
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.titleMedium
            )
            Text(
                "Pick any .bin Whisper model file from phone storage or USB. It will be copied to private storage and available for offline use.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Button(onClick = onPickModel) {
                Text("Load model from file (SAF)")
            }

            if (customModels.isNotEmpty()) {
                HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
                Text("Imported Models:", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold)
                customModels.forEach { desc ->
                    val item = modelStates[desc.id]
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(desc.displayName, fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.bodyMedium)
                            Text(humanSize(desc.sizeBytes), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            if (item?.active == true) {
                                Surface(
                                    color = MaterialTheme.colorScheme.primary,
                                    shape = RoundedCornerShape(6.dp)
                                ) {
                                    Text(
                                        "ACTIVE",
                                        fontWeight = FontWeight.Bold,
                                        color = MaterialTheme.colorScheme.onPrimary,
                                        style = MaterialTheme.typography.labelSmall,
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                    )
                                }
                            } else {
                                Button(onClick = { onActivate(desc) }) { Text("Use") }
                            }
                            OutlinedButton(onClick = { onDelete(desc) }) { Text("Delete") }
                        }
                    }
                }
            }
        }
    }
}

/**
 * Clean circular help button that opens a popup dialog with concise explanation.
 */
@Composable
private fun InfoHelpButton(
    title: String,
    explanation: String,
    modifier: Modifier = Modifier,
) {
    var showDialog by remember { mutableStateOf(false) }

    IconButton(
        onClick = { showDialog = true },
        modifier = modifier.size(24.dp)
    ) {
        Surface(
            shape = CircleShape,
            color = MaterialTheme.colorScheme.surfaceVariant,
            modifier = Modifier.size(18.dp)
        ) {
            Box(contentAlignment = Alignment.Center) {
                Text(
                    text = "?",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }

    if (showDialog) {
        AlertDialog(
            onDismissRequest = { showDialog = false },
            title = { Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold) },
            text = { Text(explanation, style = MaterialTheme.typography.bodyMedium) },
            confirmButton = {
                TextButton(onClick = { showDialog = false }) {
                    Text("OK")
                }
            }
        )
    }
}

@Composable
private fun ModelSettingsTab(
    settings: WhisperConfig,
    onUpdateSettings: (WhisperConfig) -> Unit,
    onResetSettings: () -> Unit,
    audioCacheBytes: Long,
    onClearAudioCache: () -> Unit,
    cpuCoreCount: Int,
    autoThreadCount: Int,
    hardwareInfoSummary: String,
    speakerModelDownloaded: Boolean,
    speakerModelDownloading: Boolean,
    speakerModelProgress: Float,
    speakerModelError: String?,
    onDownloadSpeakerModel: () -> Unit,
    onDeleteSpeakerModel: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(modifier = Modifier.weight(1f)) {
                Text("Model Settings & Cache", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                Text("Whisper.cpp inference hyperparameters and storage options.", style = MaterialTheme.typography.bodySmall)
            }
            OutlinedButton(onClick = onResetSettings) {
                Text("Reset defaults")
            }
        }

        HorizontalDivider()

        // 1. Audio Cache
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Decoded Audio Cache", fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.size(4.dp))
                    InfoHelpButton(
                        title = "Decoded Audio Cache",
                        explanation = "Audio files (MP3, M4A, Opus, etc.) are converted to 16 kHz PCM WAV only once. On subsequent runs or re-tries with the same file, decoding is skipped instantly, saving processing time and battery."
                    )
                }
                Text(
                    "Cached storage: ${humanSize(audioCacheBytes)}",
                    style = MaterialTheme.typography.bodyMedium,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Medium
                )
                Button(
                    onClick = onClearAudioCache,
                    enabled = audioCacheBytes > 0,
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary)
                ) {
                    Text("Clear audio cache / Force re-decode")
                }
            }
        }

        // 2. Audio Decoder Engine Selector
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Audio Decoder Engine", fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.size(4.dp))
                    InfoHelpButton(
                        title = "Audio Decoder Engine",
                        explanation = "Select the engine used to decode audio files into 16 kHz WAV:\n\n• Native C++ (NDK/NEON): Ultra-fast C demuxer with ARM NEON SIMD stereo downmix.\n• Kotlin MediaCodec: Standard Android Java/Kotlin framework hardware decoder."
                    )
                }
                DecoderEngine.entries.forEach { engine ->
                    val selected = settings.decoderEngine == engine
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .selectable(
                                selected = selected,
                                onClick = { onUpdateSettings(settings.copy(decoderEngine = engine)) }
                            )
                            .padding(vertical = 4.dp)
                    ) {
                        RadioButton(
                            selected = selected,
                            onClick = { onUpdateSettings(settings.copy(decoderEngine = engine)) }
                        )
                        Spacer(Modifier.size(8.dp))
                        Column {
                            Text(
                                text = engine.label,
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal
                            )
                            Text(
                                text = engine.description,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
        }

        // 3. Speaker Diarization Engine Selector
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Speaker Diarization Engine", fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.size(4.dp))
                    InfoHelpButton(
                        title = "Speaker Diarization Engine",
                        explanation = "Select the engine used to identify speakers during transcription:\n\n• Acoustic DSP (Built-in, 0 MB): Zero-memory classical DSP pipeline using MFCCs and pitch autocorrelation. Ultra-fast with 0 MB download.\n\n• Neural Voiceprint (CAM++, 14 MB): Deep neural embedding model offering superior separation precision even with overlapping speech or similar voice registers. Half-precision build of the code-switched Mandarin+English CAM++ model; slower than the DSP engine but noticeably more accurate."
                    )
                }
                DiarizationEngine.entries.forEach { engine ->
                    val selected = settings.diarizationEngine == engine
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .selectable(
                                selected = selected,
                                onClick = {
                                    if (engine == DiarizationEngine.NEURAL_ONNX && !speakerModelDownloaded && !speakerModelDownloading) {
                                        onDownloadSpeakerModel()
                                    }
                                    onUpdateSettings(settings.copy(diarizationEngine = engine))
                                }
                            )
                            .padding(vertical = 4.dp)
                    ) {
                        RadioButton(
                            selected = selected,
                            onClick = {
                                if (engine == DiarizationEngine.NEURAL_ONNX && !speakerModelDownloaded && !speakerModelDownloading) {
                                    onDownloadSpeakerModel()
                                }
                                onUpdateSettings(settings.copy(diarizationEngine = engine))
                            }
                        )
                        Spacer(Modifier.size(8.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = engine.label,
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal
                            )
                            Text(
                                text = engine.description,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )

                            if (engine == DiarizationEngine.NEURAL_ONNX) {
                                Spacer(Modifier.size(6.dp))
                                if (speakerModelDownloading) {
                                    Column(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                                        LinearProgressIndicator(
                                            progress = { speakerModelProgress },
                                            modifier = Modifier.fillMaxWidth().height(6.dp),
                                        )
                                        Spacer(Modifier.size(4.dp))
                                        Text(
                                            "Downloading model… ${(speakerModelProgress * 100).toInt()}%",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.primary,
                                        )
                                    }
                                } else if (speakerModelDownloaded) {
                                    OutlinedButton(
                                        onClick = onDeleteSpeakerModel,
                                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 2.dp),
                                        colors = ButtonDefaults.outlinedButtonColors(
                                            contentColor = MaterialTheme.colorScheme.error,
                                        ),
                                    ) {
                                        Icon(
                                            Icons.Default.Delete,
                                            contentDescription = "Delete model",
                                            modifier = Modifier.size(16.dp),
                                        )
                                        Spacer(Modifier.size(6.dp))
                                        Text("Delete model", style = MaterialTheme.typography.labelMedium)
                                    }
                                } else {
                                    Button(
                                        onClick = onDownloadSpeakerModel,
                                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 2.dp),
                                    ) {
                                        Text(
                                            "Download (${com.emilio.whispert.models.SPEAKER_MODEL_SIZE_LABEL})",
                                            style = MaterialTheme.typography.labelMedium
                                        )
                                    }
                                }

                                if (speakerModelError != null) {
                                    Spacer(Modifier.size(4.dp))
                                    Text(
                                        "Error: $speakerModelError",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.error,
                                    )
                                }
                            }
                        }
                    }
                }

                HorizontalDivider()

                // Speaker merge threshold
                Column {
                    val engineDefault = if (settings.diarizationEngine == DiarizationEngine.NEURAL_ONNX) 0.40f else 0.64f
                    val effective = if (settings.speakerMergeThreshold > 0f) settings.speakerMergeThreshold else engineDefault
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Speaker merge threshold: ", style = MaterialTheme.typography.bodyMedium)
                        Text(
                            String.format(Locale.ROOT, "%.2f", effective) +
                                if (settings.speakerMergeThreshold <= 0f) " (auto)" else "",
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace,
                        )
                        Spacer(Modifier.size(4.dp))
                        InfoHelpButton(
                            title = "Speaker Merge Threshold",
                            explanation = "Cosine distance at which two voice clusters are considered the same speaker. Lower values split more readily (more speakers), higher values merge more readily (fewer speakers).\n\nAfter a transcription you can change this and press 'Re-run diarization only' to see the effect without transcribing again. The 'Statistics & Diagnostics' panel reports the closest surviving pair of clusters, which is the distance this threshold has to sit below.\n\nDefaults: 0.40 for the neural engine, 0.64 for the DSP engine. Set to the far left for auto."
                        )
                    }
                    Slider(
                        value = if (settings.speakerMergeThreshold > 0f) settings.speakerMergeThreshold else 0f,
                        onValueChange = {
                            val v = if (it < 0.05f) 0f else Math.round(it * 100f) / 100f
                            onUpdateSettings(settings.copy(speakerMergeThreshold = v))
                        },
                        valueRange = 0f..1.0f,
                        steps = 19,
                    )
                }
            }
        }

        // 3. Anti-Hallucination & Fallback Tuning
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Hallucination & Fallback Tuning", fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.titleMedium)

                // Temperature Inc
                Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Temperature increment: ", style = MaterialTheme.typography.bodyMedium)
                        Text(String.format(Locale.ROOT, "%.2f", settings.temperatureInc), fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
                        Spacer(Modifier.size(4.dp))
                        InfoHelpButton(
                            title = "Temperature Increment (temperature_inc)",
                            explanation = "Controls temperature fallback when Whisper encounters ambiguous audio or low confidence. Setting this to 0.00 disables multi-pass retries, preventing extreme slowdowns and freezes on longer recordings.\n\nRecommended: 0.00."
                        )
                    }
                    Slider(
                        value = settings.temperatureInc,
                        onValueChange = { onUpdateSettings(settings.copy(temperatureInc = Math.round(it * 20f) / 20f)) },
                        valueRange = 0.0f..1.0f,
                        steps = 19
                    )
                }

                // Base Temperature
                Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Base temperature: ", style = MaterialTheme.typography.bodyMedium)
                        Text(String.format(Locale.ROOT, "%.2f", settings.temperature), fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
                        Spacer(Modifier.size(4.dp))
                        InfoHelpButton(
                            title = "Base Temperature & Fallback (-nf)",
                            explanation = "Sampling randomness during decoding. 0.00 uses greedy decoding (deterministic and most accurate for transcription).\n\nTemperature increment fallback is disabled by default (0.00, equivalent to '-nf' / '--no-fallback'), preventing Whisper from getting stuck in slow multi-temperature retry loops on difficult or quiet audio.\n\nRecommended: 0.00."
                        )
                    }
                    Slider(
                        value = settings.temperature,
                        onValueChange = { onUpdateSettings(settings.copy(temperature = Math.round(it * 20f) / 20f)) },
                        valueRange = 0.0f..1.0f,
                        steps = 19
                    )
                }

                // No-context switch
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Row(modifier = Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically) {
                        Text("No context (prevents loops)", fontWeight = FontWeight.Medium, style = MaterialTheme.typography.bodyMedium)
                        Spacer(Modifier.size(4.dp))
                        InfoHelpButton(
                            title = "No Context / Max Context 0 (-mc 0)",
                            explanation = "Equivalent to '-mc 0' / '--no-context' in whisper.cpp CLI. When enabled, Whisper sets max context tokens to 0 and does not feed the previous segment text as a prompt into the next 30s window.\n\nThis completely eliminates autoregressive repetition loops (where the model repeats the same sentence dozens of times on long recordings).\n\nRecommended: Enabled."
                        )
                    }
                    Switch(checked = settings.noContext, onCheckedChange = { onUpdateSettings(settings.copy(noContext = it)) })
                }

                // Entropy Threshold
                Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Entropy threshold: ", style = MaterialTheme.typography.bodyMedium)
                        Text(String.format(Locale.ROOT, "%.2f", settings.entropyThold), fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
                        Spacer(Modifier.size(4.dp))
                        InfoHelpButton(
                            title = "Entropy Threshold (entropy_thold)",
                            explanation = "Threshold for gzip compression ratio / entropy test used to detect repetitive or hallucinated text. Lower values trigger fallback earlier. 0.0 disables this check.\n\nDefault: 2.40."
                        )
                    }
                    Slider(
                        value = settings.entropyThold,
                        onValueChange = { onUpdateSettings(settings.copy(entropyThold = Math.round(it * 10f) / 10f)) },
                        valueRange = 0.0f..5.0f,
                        steps = 50
                    )
                }

                // Logprob Threshold
                Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Logprob threshold: ", style = MaterialTheme.typography.bodyMedium)
                        Text(String.format(Locale.ROOT, "%.2f", settings.logprobThold), fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
                        Spacer(Modifier.size(4.dp))
                        InfoHelpButton(
                            title = "Logprob Threshold (logprob_thold)",
                            explanation = "Threshold for average token log-probability. If the average probability falls below this value, Whisper considers the segment low-confidence. 0.0 disables this check.\n\nDefault: -1.00."
                        )
                    }
                    Slider(
                        value = settings.logprobThold,
                        onValueChange = { onUpdateSettings(settings.copy(logprobThold = Math.round(it * 10f) / 10f)) },
                        valueRange = -5.0f..0.0f,
                        steps = 50
                    )
                }

                // No speech threshold (VAD)
                Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Silence threshold (VAD): ", style = MaterialTheme.typography.bodyMedium)
                        Text(String.format(Locale.ROOT, "%.2f", settings.noSpeechThold), fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
                        Spacer(Modifier.size(4.dp))
                        InfoHelpButton(
                            title = "No Speech Threshold (no_speech_thold)",
                            explanation = "Voice Activity Detection (VAD) threshold to classify segments as silence. Segments with silence probability higher than this are skipped.\n\nDefault: 0.60."
                        )
                    }
                    Slider(
                        value = settings.noSpeechThold,
                        onValueChange = { onUpdateSettings(settings.copy(noSpeechThold = Math.round(it * 20f) / 20f)) },
                        valueRange = 0.0f..1.0f,
                        steps = 19
                    )
                }
            }
        }

        // 3. Domain Prompt / Technical Vocabulary
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Domain Vocabulary / Initial Prompt", fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.size(4.dp))
                    InfoHelpButton(
                        title = "Initial Prompt & Technical Vocabulary",
                        explanation = "Optional text or comma-separated list of domain-specific jargon, acronyms, and proper nouns (e.g. 'BDocs, Combi stopper, SOP, RMM, Context 64'). Whisper uses this context to bias recognition towards correct spelling."
                    )
                }
                OutlinedTextField(
                    value = settings.initialPrompt,
                    onValueChange = { onUpdateSettings(settings.copy(initialPrompt = it)) },
                    modifier = Modifier.fillMaxWidth(),
                    placeholder = { Text("e.g. BDocs, Combi stopper, SOP, RMM, Context 64") },
                    singleLine = false,
                    maxLines = 3
                )
            }
        }

        // 4. Performance & Dynamic CPU Threads
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("CPU & Performance", fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.titleMedium)

                // Dynamic CPU Threads
                val maxCores = if (cpuCoreCount > 0) cpuCoreCount else 0
                Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("CPU Threads: ", style = MaterialTheme.typography.bodyMedium)
                        val threadLabel = when {
                            maxCores <= 1 || settings.customThreads == 0 -> "Auto ($autoThreadCount performance cores)"
                            else -> "${settings.customThreads} threads"
                        }
                        Text(threadLabel, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
                        Spacer(Modifier.size(4.dp))
                        InfoHelpButton(
                            title = "CPU Worker Threads",
                            explanation = "Number of parallel CPU worker threads for inference. 'Auto' automatically detects the high-performance cores (excluding slow efficiency cores) and leaves 1 core for smooth UI rendering.\n\nDetected Hardware Specifications:\n$hardwareInfoSummary"
                        )
                    }

                    if (maxCores > 1) {
                        // Slider adapts dynamically to the physical core count of the phone
                        val effectiveValue = settings.customThreads.coerceIn(0, maxCores)
                        Slider(
                            value = effectiveValue.toFloat(),
                            onValueChange = { onUpdateSettings(settings.copy(customThreads = it.toInt())) },
                            valueRange = 0f..maxCores.toFloat(),
                            steps = maxCores - 1
                        )
                    } else {
                        Text(
                            "Hardware core detection fixed to Auto configuration.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }

                // Translate switch
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Row(modifier = Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically) {
                        Text("Translate to English", fontWeight = FontWeight.Medium, style = MaterialTheme.typography.bodyMedium)
                        Spacer(Modifier.size(4.dp))
                        InfoHelpButton(
                            title = "Translate to English",
                            explanation = "When enabled, Whisper automatically translates non-English speech directly into English text during transcription."
                        )
                    }
                    Switch(checked = settings.translate, onCheckedChange = { onUpdateSettings(settings.copy(translate = it)) })
                }

                // Suppress Blank
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Row(modifier = Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically) {
                        Text("Suppress blank tokens", fontWeight = FontWeight.Medium, style = MaterialTheme.typography.bodyMedium)
                        Spacer(Modifier.size(4.dp))
                        InfoHelpButton(
                            title = "Suppress Blank Tokens",
                            explanation = "Suppresses empty or whitespace-only tokens to avoid producing blank segments.\n\nRecommended: Enabled."
                        )
                    }
                    Switch(checked = settings.suppressBlank, onCheckedChange = { onUpdateSettings(settings.copy(suppressBlank = it)) })
                }
            }
        }
    }
}

private fun humanSize(bytes: Long): String =
    when {
        bytes >= 1L shl 30 -> "%.1f GB".format(Locale.ROOT, bytes.toDouble() / (1L shl 30))
        bytes >= 1L shl 20 -> "%.0f MB".format(Locale.ROOT, bytes.toDouble() / (1L shl 20))
        bytes >= 1L shl 10 -> "%.0f KB".format(Locale.ROOT, bytes.toDouble() / (1L shl 10))
        else -> "$bytes B"
    }

private fun shortened(uri: Uri): String =
    Uri.decode(uri.lastPathSegment ?: uri.toString())