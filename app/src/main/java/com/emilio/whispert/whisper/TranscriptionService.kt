package com.emilio.whispert.whisper

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import com.emilio.whispert.MainActivity

/**
 * Foreground service that maintains high CPU priority and an active WakeLock
 * during long-running transcription or audio decoding sessions.
 *
 * This prevents modern Android battery managers (e.g. Xiaomi HyperOS / Samsung OneUI / Doze)
 * from freezing the process or throttling CPU threads when the display locks.
 *
 * The WakeLock lives here and nowhere else: its lifetime is exactly the
 * lifetime of the service, so it cannot outlive the work it protects.
 */
class TranscriptionService : Service() {

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel(this)
        acquireWakeLock()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // A null intent means the system restarted us after killing the process.
        // The transcription died with that process, so there is nothing left to
        // protect: staying up would mean a three-hour WakeLock and a notification
        // claiming work that no longer exists. Shut down instead.
        if (intent == null) {
            android.util.Log.i(TAG, "Restarted with a null intent (process death); no work to resume, stopping")
            stopForegroundService()
            return START_NOT_STICKY
        }

        if (intent.action == ACTION_STOP) {
            stopForegroundService()
            return START_NOT_STICKY
        }

        val title = intent.getStringExtra(EXTRA_TITLE) ?: "Whisper Transcriber"
        val message = intent.getStringExtra(EXTRA_MESSAGE) ?: "Processing audio…"
        val progress = intent.getIntExtra(EXTRA_PROGRESS, -1)

        val notification = buildNotification(this, title, message, progress)
        // The typed overload only exists from API 29; minSdk here is 28.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        // Not sticky: the transcription lives in the app process and cannot be
        // resumed from a restarted service, so a restart would only ever produce
        // a WakeLock with nothing behind it.
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        releaseWakeLock()
        super.onDestroy()
    }

    private fun acquireWakeLock() {
        runCatching {
            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "whisper:transcription_wakelock").apply {
                setReferenceCounted(false)
                acquire(3 * 60 * 60 * 1000L) // 3 hours timeout safety
            }
        }
    }

    private fun releaseWakeLock() {
        runCatching {
            if (wakeLock?.isHeld == true) {
                wakeLock?.release()
            }
            wakeLock = null
        }
    }

    private fun stopForegroundService() {
        releaseWakeLock()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    companion object {
        private const val TAG = "TranscriptionService"

        const val CHANNEL_ID = "whisper_transcription_channel"
        const val NOTIFICATION_ID = 1001

        /**
         * Creating the channel is idempotent but it is still a binder call into
         * system_server, and update() runs on the inference thread once a second.
         * Once per process is enough.
         */
        private val channelCreated = java.util.concurrent.atomic.AtomicBoolean(false)

        const val ACTION_START = "com.emilio.whispert.START_FOREGROUND"
        const val ACTION_STOP = "com.emilio.whispert.STOP_FOREGROUND"

        const val EXTRA_TITLE = "extra_title"
        const val EXTRA_MESSAGE = "extra_message"
        const val EXTRA_PROGRESS = "extra_progress"

        private fun createNotificationChannel(context: Context) {
            if (!channelCreated.compareAndSet(false, true)) return
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Transcription Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Keeps transcription active in background"
                setShowBadge(false)
            }
            context.getSystemService(NotificationManager::class.java)?.createNotificationChannel(channel)
        }

        private fun buildNotification(
            context: Context,
            title: String,
            message: String,
            progress: Int
        ): Notification {
            val pendingIntent = PendingIntent.getActivity(
                context,
                0,
                Intent(context, MainActivity::class.java),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
            )

            val builder = NotificationCompat.Builder(context, CHANNEL_ID)
                .setContentTitle(title)
                .setContentText(message)
                .setSmallIcon(android.R.drawable.stat_sys_download)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setContentIntent(pendingIntent)

            if (progress in 0..100) {
                builder.setProgress(100, progress, false)
            } else {
                builder.setProgress(0, 0, true)
            }

            return builder.build()
        }

        fun start(context: Context, title: String, message: String, progress: Int = -1) {
            val intent = Intent(context, TranscriptionService::class.java).apply {
                action = ACTION_START
                putExtra(EXTRA_TITLE, title)
                putExtra(EXTRA_MESSAGE, message)
                putExtra(EXTRA_PROGRESS, progress)
            }
            try {
                context.startForegroundService(intent)
            } catch (t: Throwable) {
                android.util.Log.e(TAG, "Failed to start foreground service", t)
            }
        }

        /**
         * Refreshes the ongoing notification in place.
         *
         * This used to go through startForegroundService() on every progress
         * tick, i.e. a full Intent + service restart round trip to system_server
         * a hundred times per run, all of it on the whisper inference thread.
         * Posting to the same notification id updates the existing foreground
         * notification directly.
         */
        fun update(context: Context, title: String, message: String, progress: Int = -1) {
            try {
                createNotificationChannel(context)
                NotificationManagerCompat.from(context)
                    .notify(NOTIFICATION_ID, buildNotification(context, title, message, progress))
            } catch (_: SecurityException) {
                // POST_NOTIFICATIONS not granted: the work continues regardless.
            } catch (t: Throwable) {
                android.util.Log.w(TAG, "Notification update failed", t)
            }
        }

        fun stop(context: Context) {
            val intent = Intent(context, TranscriptionService::class.java)
            try {
                context.stopService(intent)
            } catch (t: Throwable) {
                android.util.Log.w(TAG, "Failed to stop service", t)
            }
        }
    }
}
