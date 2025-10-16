package com.example.myapplication
// MainActivity.kt  — 姿勢計測 + NDJSON/gzip 送信（API Gateway→Lambda→S3→SageMaker）

import android.app.Activity
import android.content.Context
import android.hardware.*
import android.os.Bundle
import android.os.SystemClock
import android.view.Surface
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlin.math.PI
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.math.exp
import kotlin.math.abs
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import okio.Buffer
import okio.GzipSink
import okio.buffer
import java.util.UUID
import java.util.concurrent.TimeUnit

class MainActivity : ComponentActivity(), SensorEventListener {

    // ====== ★ 設定（差し替え） ======
    private val API_URL = "秘密"
    private val DEVICE_ID = "android-" + UUID.randomUUID().toString().take(8)
    private val BATCH_SIZE = 50               // これ以上貯まったら送信
    private val FLUSH_INTERVAL_MS = 3000L     // 3秒ごとに送信トリガ

    // ===== Sensors =====
    private lateinit var sensorManager: SensorManager
    private var rotationVector: Sensor? = null
    private var linearAcc: Sensor? = null

    // ===== AtFtitude states (deg) =====
    private var yawDeg by mutableStateOf(0f)
    private var pitchDeg by mutableStateOf(0f)
    private var rollDeg by mutableStateOf(0f)

    // ===== Integration states =====
    private var px by mutableStateOf(0.0)
    private var py by mutableStateOf(0.0)
    private var pz by mutableStateOf(0.0)
    private var speed by mutableStateOf(0.0)
    private var distance by mutableStateOf(0.0)
    private var vx = 0.0
    private var vy = 0.0
    private var vz = 0.0
    private var lastAccTimestampNs: Long? = null

    // ===== Buffers =====
    private val rotVec = FloatArray(5)
    private val quat = FloatArray(4) // [w,x,y,z]
    private val R = FloatArray(9)
    private val Rremap = FloatArray(9)
    private val ori = FloatArray(3)
    private val aDev = FloatArray(3)
    private val aWorld = FloatArray(3)

    // ===== Sender =====
    private val ioScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val ok = OkHttpClient.Builder()
        .callTimeout(15, TimeUnit.SECONDS)
        .build()
    private val buffer = ArrayList<String>(BATCH_SIZE * 2) // NDJSON lines
    private var lastFlushAt = SystemClock.elapsedRealtime()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        sensorManager = getSystemService(Activity.SENSOR_SERVICE) as SensorManager
        rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
        linearAcc = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

        setContent {
            MaterialTheme {
                Surface(Modifier.fillMaxSize()) {
                    PostureScreen(
                        yaw = yawDeg, pitch = pitchDeg, roll = rollDeg,
                        px = px, py = py, pz = pz,
                        speed = speed, distance = distance,
                        onReset = { resetIntegration() }
                    )
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        rotationVector?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        linearAcc?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        // 定期フラッシュ（保険）
        ioScope.launch {
            while (isActive) {
                delay(FLUSH_INTERVAL_MS)
                maybeFlush(forceTime = true)
            }
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
        // 可能なら最後にフラッシュ
        runBlocking { maybeFlush(forceTime = true, forceCount = true) }
        ioScope.coroutineContext.cancelChildren()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ROTATION_VECTOR,
            Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                // ---- 姿勢更新 ----
                for (i in event.values.indices) rotVec[i] = event.values[i]
                SensorManager.getRotationMatrixFromVector(R, rotVec)

                // 画面回転に合わせてリマップ
                val rotation = currentRotation(this)
                when (rotation) {
                    Surface.ROTATION_0 -> {
                        SensorManager.remapCoordinateSystem(
                            R, SensorManager.AXIS_X, SensorManager.AXIS_Z, Rremap
                        )
                    }
                    Surface.ROTATION_90 -> {
                        SensorManager.remapCoordinateSystem(
                            R, SensorManager.AXIS_Z, SensorManager.AXIS_MINUS_X, Rremap
                        )
                    }
                    Surface.ROTATION_180 -> {
                        SensorManager.remapCoordinateSystem(
                            R, SensorManager.AXIS_MINUS_X, SensorManager.AXIS_MINUS_Z, Rremap
                        )
                    }
                    Surface.ROTATION_270 -> {
                        SensorManager.remapCoordinateSystem(
                            R, SensorManager.AXIS_MINUS_Z, SensorManager.AXIS_X, Rremap
                        )
                    }
                }

                SensorManager.getOrientation(Rremap, ori)
                val rad2deg = (180f / PI).toFloat()
                yawDeg = wrapDeg(ori[0] * rad2deg)
                pitchDeg = (ori[1] * rad2deg)
                rollDeg = (ori[2] * rad2deg)

                // 四元数（RotationVector→Quat）
                // getQuaternionFromVector(q, rv): q[0]=w, q[1]=x, q[2]=y, q[3]=z
                try { SensorManager.getQuaternionFromVector(quat, rotVec) } catch (_: Throwable) {}

                // ログ1行を作成してバッファ
                appendSample(
                    tsNs = event.timestamp,
                    displayRotation = rotation,
                    source = if (event.sensor.type == Sensor.TYPE_ROTATION_VECTOR) "ROTATION_VECTOR" else "GAME_ROTATION_VECTOR"
                )
            }

            Sensor.TYPE_LINEAR_ACCELERATION -> {
                // ---- 加速度→速度→変位 の積分 ----
                aDev[0] = event.values[0]
                aDev[1] = event.values[1]
                aDev[2] = event.values[2]

                // world = Rremap * device
                aWorld[0] = Rremap[0]*aDev[0] + Rremap[1]*aDev[1] + Rremap[2]*aDev[2]
                aWorld[1] = Rremap[3]*aDev[0] + Rremap[4]*aDev[1] + Rremap[5]*aDev[2]
                aWorld[2] = Rremap[6]*aDev[0] + Rremap[7]*aDev[1] + Rremap[8]*aDev[2]

                val tNs = event.timestamp
                val last = lastAccTimestampNs
                if (last != null) {
                    val dt = (tNs - last) / 1_000_000_000.0

                    val ax = if (abs(aWorld[0]) < 0.05f) 0.0 else aWorld[0].toDouble()
                    val ay = if (abs(aWorld[1]) < 0.05f) 0.0 else aWorld[1].toDouble()
                    val az = if (abs(aWorld[2]) < 0.05f) 0.0 else aWorld[2].toDouble()

                    vx += ax * dt
                    vy += ay * dt
                    vz += az * dt

                    val lambda = 0.4
                    val decay = exp(-lambda * dt)
                    vx *= decay; vy *= decay; vz *= decay

                    px += vx * dt
                    py += vy * dt
                    pz += vz * dt

                    speed = sqrt(vx*vx + vy*vy + vz*vz)
                    distance = sqrt(px*px + py*py + pz*pz)
                }
                lastAccTimestampNs = tNs
            }
        }
    }

    // ====== ログ行をNDJSONに追加 ======
    private fun appendSample(tsNs: Long, displayRotation: Int, source: String) {
        // 注意：高速化のため簡易JSON生成（必要に応じてMoshi/Kotlinx-serializationへ切替）
        val sb = StringBuilder(256)
        sb.append('{')
        // 端末時刻は単調増加Ns。SageMaker側で壁時計時刻に合わせたいならクライアント側で System.currentTimeMillis も送る
        sb.append("\"ts_ns\":").append(tsNs).append(',')
        sb.append("\"device_id\":\"").append(DEVICE_ID).append("\",")

        // 四元数（w,x,y,z）
        sb.append("\"qw\":").append(quat.getOrElse(0){1f}).append(',')
        sb.append("\"qx\":").append(quat.getOrElse(1){0f}).append(',')
        sb.append("\"qy\":").append(quat.getOrElse(2){0f}).append(',')
        sb.append("\"qz\":").append(quat.getOrElse(3){0f}).append(',')

        // Euler (deg)
        sb.append("\"yaw\":").append(yawDeg).append(',')
        sb.append("\"pitch\":").append(pitchDeg).append(',')
        sb.append("\"roll\":").append(rollDeg).append(',')

        // Acc / Vel / Pos
        sb.append("\"ax\":").append(aWorld[0]).append(',')
        sb.append("\"ay\":").append(aWorld[1]).append(',')
        sb.append("\"az\":").append(aWorld[2]).append(',')
        sb.append("\"vx\":").append(vx).append(',')
        sb.append("\"vy\":").append(vy).append(',')
        sb.append("\"vz\":").append(vz).append(',')
        sb.append("\"px\":").append(px).append(',')
        sb.append("\"py\":").append(py).append(',')
        sb.append("\"pz\":").append(pz).append(',')

        // Metadata
        sb.append("\"display_rotation\":").append(displayRotation).append(',')
        sb.append("\"source\":\"").append(source).append('"')
        sb.append('}')

        synchronized(buffer) {
            buffer.add(sb.toString())
        }
        maybeFlush()
    }

    // ====== フラッシュ判定 & 実送信 ======
    private fun maybeFlush(forceTime: Boolean = false, forceCount: Boolean = false) {
        val now = SystemClock.elapsedRealtime()
        val shouldByTime = forceTime || (now - lastFlushAt >= FLUSH_INTERVAL_MS)
        val shouldByCount = forceCount || synchronized(buffer) { buffer.size >= BATCH_SIZE }
        if (!shouldByTime && !shouldByCount) return

        val lines: List<String> = synchronized(buffer) {
            if (buffer.isEmpty()) return
            val copy = buffer.toList()
            buffer.clear()
            copy
        }

        lastFlushAt = now
        ioScope.launch {
            try {
                val ndjson = lines.joinToString("\n").toByteArray(Charsets.UTF_8)
                val gzBody = gzip(ndjson)
                val reqBody = gzBody.toRequestBody("application/x-ndjson".toMediaType())

                val req = Request.Builder()
                    .url(API_URL)
                    .addHeader("Content-Type", "application/x-ndjson")
                    .addHeader("Content-Encoding", "gzip")
                    .addHeader("x-device-id", DEVICE_ID)
                    .post(reqBody)
                    .build()

                ok.newCall(req).execute().use { resp ->
                    if (!resp.isSuccessful) {
                        android.util.Log.w("POSE_SEND", "HTTP ${resp.code}: ${resp.message}")
                    } else {
                        android.util.Log.d("POSE_SEND", "sent ${lines.size} recs")
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("POSE_SEND", "send error", e)
            }
        }
    }

    // ====== Helpers ======
    private fun gzip(bytes: ByteArray): ByteArray {
        val buf = Buffer()
        GzipSink(buf).buffer().use { it.write(bytes) }
        return buf.readByteArray()
    }

    private fun resetIntegration() {
        vx = 0.0; vy = 0.0; vz = 0.0
        px = 0.0; py = 0.0; pz = 0.0
        speed = 0.0; distance = 0.0
        lastAccTimestampNs = null
    }

    private fun wrapDeg(v: Float): Float {
        var x = v
        while (x <= -180f) x += 360f
        while (x > 180f) x -= 360f
        return x
    }

    private fun currentRotation(ctx: Context): Int {
        return try {
            if (android.os.Build.VERSION.SDK_INT >= 30) {
                display?.rotation ?: Surface.ROTATION_0
            } else {
                @Suppress("DEPRECATION")
                (ctx.getSystemService(Context.WINDOW_SERVICE) as WindowManager)
                    .defaultDisplay?.rotation ?: Surface.ROTATION_0
            }
        } catch (_: Throwable) {
            Surface.ROTATION_0
        }
    }
}

// ===== Compose UI（そのまま） =====
@Composable
fun PostureScreen(
    yaw: Float, pitch: Float, roll: Float,
    px: Double, py: Double, pz: Double,
    speed: Double, distance: Double,
    onReset: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("姿勢モニター", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)

        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            AngleChip(label = "Yaw", value = yaw)
            AngleChip(label = "Pitch", value = pitch)
            AngleChip(label = "Roll", value = roll)
        }

        LevelGauge(roll = roll)

        Spacer(Modifier.height(8.dp))
        Text("位置 (m):  x=${px.format(3)} , y=${py.format(3)} , z=${pz.format(3)}")
        Text("速度 |v| (m/s): ${speed.format(3)}")
        Text("原点からの距離 |p| (m): ${distance.format(3)}")

        OutlinedButton(onClick = onReset) { Text("原点リセット") }
    }
}

private fun Double.format(n: Int): String = "%.${n}f".format(this)

@Composable
fun AngleChip(label: String, value: Float) {
    AssistChip(onClick = {}, label = { Text("$label ${value.roundToInt()}°") })
}

@Composable
fun LevelGauge(roll: Float) {
    Box(modifier = Modifier.fillMaxWidth().height(240.dp)) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val center = this.center
            val length = size.minDimension * 0.8f
            val theta = (-roll / 180f) * Math.PI.toFloat()
            val cosT = kotlin.math.cos(theta.toDouble()).toFloat()
            val sinT = kotlin.math.sin(theta.toDouble()).toFloat()
            val half = length / 2f
            val dx = half * cosT
            val dy = half * sinT
            val p1 = Offset(center.x - dx, center.y - dy)
            val p2 = Offset(center.x + dx, center.y + dy)
            drawLine(color = Color.Black, start = p1, end = p2, strokeWidth = 8f)
            drawCircle(color = Color.Red, center = center, radius = 6f)
        }
    }
}
