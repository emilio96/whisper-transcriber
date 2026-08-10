package com.emilio.whispert

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.emilio.whispert.ui.theme.WhisperTranscriberTheme
import com.emilio.whispert.whisper.WhisperContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            WhisperTranscriberTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    MainScreen()
                }
            }
        }
    }
}

@Composable
private fun MainScreen(vm: MainViewModel = viewModel()) {
    val state by vm.state.collectAsState()

    // SAF picker: lets us open any .bin / .gguf file the user downloaded. No
    // storage permission is required because the user actively selects the file.
    val pickModel = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri ->
        if (uri != null) vm.onModelPicked(uri)
    }

    Scaffold { inner ->
        Column(
            modifier = Modifier
                .padding(inner)
                .padding(16.dp)
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                "Whisper Transcriber",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
            )
            Text(
                "On-device smoke test (whisper.cpp v1.9.2 + JNI).",
                style = MaterialTheme.typography.bodySmall,
            )

            HorizontalDivider()

            // System info: lets us confirm fp16 / threads at a glance in logs.
            val sysInfo = remember { runCatching { WhisperContext.getSystemInfo() }.getOrDefault("(n/a)") }
            Text(
                "System: $sysInfo",
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
            )

            Spacer(Modifier.height(4.dp))

            // Model row
            Row(verticalAlignment = Alignment.CenterVertically) {
                Button(onClick = {
                    // Allow any file; we filter on common model extensions in action.
                    pickModel.launch(arrayOf("*/*"))
                }) { Text("Pick model file") }
                Spacer(Modifier.fillMaxWidth(0.0f))
            }
            Text(
                "Pick the .bin/.gguf you downloaded (e.g. ggml-tiny.en.bin) — it's read once and cached by Uri.",
                style = MaterialTheme.typography.bodySmall,
            )

            // Action row
            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedButton(
                    onClick = { vm.transcribeTestSample(language = "en") },
                    enabled = state is UiState.Ready || state is UiState.Done || state is UiState.Error,
                ) { Text("Transcribe bundled jfk.wav (EN test)") }
            }

            Spacer(Modifier.height(8.dp))
            HorizontalDivider()

            // State render
            when (val s = state) {
                UiState.Idle -> Text("Idle. Pick a model to begin.")
                is UiState.Busy -> Text("⏳ ${s.message}")
                is UiState.Ready -> Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("✓ Model ready")
                    Text("URI: ${shortened(s.modelUri)}", style = MaterialTheme.typography.bodySmall)
                    Text(s.note, style = MaterialTheme.typography.bodySmall)
                }
                is UiState.Done -> Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(
                        "Done in ${s.elapsedMs} ms · ${s.sampleCount} samples · lang=${s.language}",
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        text = s.text.ifBlank { "(empty transcript)" },
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                    )
                }
                is UiState.Error -> Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("✗ ${s.message}", color = MaterialTheme.colorScheme.error)
                }
            }
        }
    }
}

private fun shortened(uri: String): String {
    runCatching { return Uri.decode(Uri.parse(uri).lastPathSegment ?: uri) }
    return uri
}