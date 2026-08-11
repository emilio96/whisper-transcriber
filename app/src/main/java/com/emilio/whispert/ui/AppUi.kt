package com.emilio.whispert.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.emilio.whispert.MainViewModel
import com.emilio.whispert.ModelItemState
import com.emilio.whispert.ModelsUiState
import com.emilio.whispert.TranscribeUiState
import kotlinx.coroutines.launch
import java.util.Locale

@Composable
fun MainScreen(vm: MainViewModel = viewModel()) {
    val audioUri by vm.audioUri.collectAsState()
    val language by vm.language.collectAsState()
    val transcribeState by vm.transcribeState.collectAsState()
    val modelsState by vm.modelsState.collectAsState()

    var selectedTab by remember { mutableIntStateOf(0) }
    val scope = androidx.compose.runtime.rememberCoroutineScope()

    Scaffold { inner ->
        Column(modifier = Modifier.padding(inner).fillMaxSize()) {
            PrimaryTabRow(selectedTabIndex = selectedTab) {
                Tab(selected = selectedTab == 0, onClick = { selectedTab = 0 }, text = { Text("Transcribe") })
                Tab(selected = selectedTab == 1, onClick = { selectedTab = 1 }, text = { Text("Models") })
            }
            when (selectedTab) {
            0 -> TranscribeTab(
                    audioUri = audioUri,
                    language = language,
                    state = transcribeState,
                    onPickAudioPicked = vm::onAudioPicked,
                    onClearAudio = { vm.clearAudio() },
                    onLanguageChange = { vm.setLanguage(it) },
                    onTranscribe = { vm.transcribe() },
                    onTranscribeJfk = { vm.transcribeBundledJfk() },
                    onSaveTranscript = { text, uri ->
                        scope.launch { vm.saveTranscriptTo(text, uri) }
                    },
                    activeModelId = modelsState.activeModelId,
                )
                1 -> ModelsTab(state = modelsState,
                    onDownload = { vm.downloadModel(it) },
                    onDelete = { vm.deleteModel(it) },
                    onActivate = { vm.activateModel(it) },
                    onImportModelUri = { vm.importModel(it) })
            }
        }
    }
}

@Composable
private fun TranscribeTab(
    audioUri: Uri?,
    language: String,
    state: TranscribeUiState,
    onPickAudioPicked: (Uri) -> Unit,
    onClearAudio: () -> Unit,
    onLanguageChange: (String) -> Unit,
    onTranscribe: () -> Unit,
    onTranscribeJfk: () -> Unit,
    onSaveTranscript: (text: String, uri: Uri) -> Unit,
    activeModelId: String?,
) {
    // SAF picker: any audio/* file from Documents/Downloads etc.
    val pickAudio = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri -> if (uri != null) onPickAudioPicked(uri) }

    // SAF saver: ActivityResultContracts.CreateDocument opens the system file
    // picker in 'create new file' mode; we hand the chosen Uri back to the VM
    // with the transcript text. We hold the in-flight text in this state so the
    // callback (called later when the user confirms the destination filename)
    // still knows what to write.
    var pendingSaveText by remember { androidx.compose.runtime.mutableStateOf<String?>(null) }
    val saveTxt = rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.CreateDocument("text/plain"),
    ) { uri ->
        if (uri != null) {
            val text = pendingSaveText
            if (text != null) onSaveTranscript(text, uri)
        }
        pendingSaveText = null
    }

    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Whisper Transcriber", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Text("On-device speech to text (whisper.cpp + JNI). Your audio never leaves the phone.",
             style = MaterialTheme.typography.bodySmall)

        HorizontalDivider()

        // Active model summary
        Text("Active model: ${activeModelId ?: "none"}",
             style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)

        // Audio picker row
        Row(verticalAlignment = Alignment.CenterVertically) {
            Button(onClick = { pickAudio.launch(arrayOf("audio/*")) }) { Text("Pick audio file") }
            Spacer(Modifier.size(8.dp))
            if (audioUri != null) OutlinedButton(onClick = onClearAudio) { Text("Clear") }
        }
        Text(audioUri?.let { "File: ${shortened(it)}" } ?: "Pick a mp3/m4a/wav/opus from anywhere reachable.",
             style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)

        // Language selector (only relevant for multilingual models; EN-only pins 'en')
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Language: ", style = MaterialTheme.typography.bodyMedium)
            listOf("auto", "en", "it").forEach { lang ->
                TextButton(
                    onClick = { onLanguageChange(lang) },
                    colors = androidx.compose.material3.ButtonDefaults.textButtonColors(
                        contentColor = if (language == lang) MaterialTheme.colorScheme.primary
                        else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                ) { Text(lang.uppercase(Locale.ROOT)) }
            }
        }

        // Action row
        Row(verticalAlignment = Alignment.CenterVertically) {
            Button(onClick = onTranscribe, enabled = audioUri != null && state !is TranscribeUiState.Busy) {
                Text("Transcribe")
            }
            Spacer(Modifier.size(12.dp))
            OutlinedButton(onClick = onTranscribeJfk, enabled = state !is TranscribeUiState.Busy) {
                Text("Smoke test (jfk.wav / EN)")
            }
        }

        HorizontalDivider()
        when (val s = state) {
            TranscribeUiState.Idle -> Text("Idle. Pick an audio file and press Transcribe.")
            is TranscribeUiState.Busy -> Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.size(8.dp))
                    Text("⏳ ${s.message}")
                }
            }
            is TranscribeUiState.Done -> Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Done in ${s.elapsedMs} ms · ${s.sampleCount} samples · lang=${s.language} · model=${s.modelId}",
                     fontWeight = FontWeight.SemiBold)
                Surface(tonalElevation = 2.dp, shape = RoundedCornerShape(8.dp), modifier = Modifier.fillMaxWidth()) {
                    Text(s.text.ifBlank { "(empty transcript)" },
                         modifier = Modifier.padding(12.dp),
                         style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Button(onClick = {
                        pendingSaveText = s.text
                        saveTxt.launch("transcript_${System.currentTimeMillis()}.txt")
                    }) { Text("Save .txt") }
                    Spacer(Modifier.size(12.dp))
                    OutlinedButton(onClick = { onTranscribe() }) { Text("Re-run") }
                }
            }
            is TranscribeUiState.Error -> Text("✗ ${s.message}", color = MaterialTheme.colorScheme.error)
        }
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

    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Models", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Text("Device: ${state.deviceDescription}", style = MaterialTheme.typography.bodySmall)
        HorizontalDivider()

        state.registry.forEach { desc ->
            val item: ModelItemState = state.modelStates[desc.id]!!
            ModelCard(item, state.deviceClass, onDownload, onDelete, onActivate)
        }
        Text("All downloads go to private app storage. The first download needs internet; "
             + "afterwards transcriptions run fully offline.", style = MaterialTheme.typography.bodySmall)

        HorizontalDivider()
        Text("Already have a .bin model file on this phone?", fontWeight = FontWeight.SemiBold)
        Button(onClick = { pickModel.launch(arrayOf("*/*")) }) {
            Text("Load model from file (SAF)")
        }
        Text(
            "Picks any model .bin you already downloaded (e.g. via browser) or copied via USB. " +
            "We copy it into private app storage and add it to the list above. Disable english-only " +
            "models with the 'en' language in the Transcribe tab.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

@Composable
private fun ModelCard(
    item: ModelItemState,
    deviceClass: com.emilio.whispert.models.DeviceClass,
    onDownload: (com.emilio.whispert.models.ModelDescriptor) -> Unit,
    onDelete: (com.emilio.whispert.models.ModelDescriptor) -> Unit,
    onActivate: (com.emilio.whispert.models.ModelDescriptor) -> Unit,
) {
    val isRecommended = deviceClass in item.descriptor.recommendedFor
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = if (item.active) CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        ) else CardDefaults.cardColors()
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(item.descriptor.displayName, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.size(8.dp))
                if (isRecommended) {
                    Text(
                        "★ recommended",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.tertiary,
                    )
                }
                Spacer(Modifier.weight(1f))
                if (item.active) Text("active", fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
            }
            Text(item.descriptor.description, style = MaterialTheme.typography.bodySmall)
            Text("${humanSize(item.descriptor.sizeBytes)} · ${if (item.descriptor.multilingual) "multilingual" else "EN only"}",
                 style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)

            // Progress bar / download state / error
            AnimatedVisibility(visible = item.downloading) {
                Column {
                    LinearProgressIndicator(progress = { item.progress }, modifier = Modifier.fillMaxWidth())
                    Text("Downloading… ${"%.0f".format(item.progress * 100)}%",
                         style = MaterialTheme.typography.labelSmall)
                }
            }
            item.error?.let { Text("Error: $it", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }

            // Action buttons
            Row(verticalAlignment = Alignment.CenterVertically) {
                when {
                    item.downloading -> Text("Please wait…", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    !item.downloaded -> Button(onClick = { onDownload(item.descriptor) }) { Text("Download") }
                    item.active -> OutlinedButton(onClick = { onDelete(item.descriptor) }) { Text("Delete") }
                    else -> Row {
                        Button(onClick = { onActivate(item.descriptor) }) { Text("Use") }
                        Spacer(Modifier.size(8.dp))
                        OutlinedButton(onClick = { onDelete(item.descriptor) }) { Text("Delete") }
                    }
                }
            }
        }
    }
}

private fun humanSize(bytes: Long): String =
    when {
        bytes >= 1L shl 30 -> "%.1f GB".format(Locale.ROOT, bytes.toDouble() / (1L shl 30))
        bytes >= 1L shl 20 -> "%.0f MB".format(Locale.ROOT, bytes.toDouble() / (1L shl 20))
        else -> "$bytes B"
    }

private fun shortened(uri: Uri): String =
    Uri.decode(uri.lastPathSegment ?: uri.toString())