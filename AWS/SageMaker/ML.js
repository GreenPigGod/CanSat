/*dependencies
npm init -y
npm i @aws-sdk/client-s3 @aws-sdk/lib-storage @tensorflow/tfjs-node
*/

// Node.js 18+
const { S3Client, HeadObjectCommand, GetObjectCommand, PutObjectCommand, SelectObjectContentCommand } = require('@aws-sdk/client-s3');
const { Upload } = require('@aws-sdk/lib-storage');
const fs = require('fs');
const os = require('os');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

// ===== 0) 設定 =====
const BUCKET = 'pose-logs-2025';                   // ←変更可
const BASE   = 'curated/android/2025/10/10';       // 例）"curated/android/YYYY/MM/DD"
const TRAIN  = `${BASE}/train.parquet`;
const VAL    = `${BASE}/val.parquet`;
const BATCH  = `${BASE}/batch.parquet`;

const s3 = new S3Client({});

// ===== ユーティリティ =====
async function s3KeyExists(bucket, key) {
  try {
    await s3.send(new HeadObjectCommand({ Bucket: bucket, Key: key }));
    return true;
  } catch {
    return false;
  }
}

// S3 SelectでParquet→JSON（NDJSON）として取得して解析
async function readParquetViaSelectToObjects(bucket, key) {
  const resp = await s3.send(new SelectObjectContentCommand({
    Bucket: bucket,
    Key: key,
    ExpressionType: 'SQL',
    Expression: 'SELECT * FROM s3object s',
    InputSerialization: { Parquet: {} },
    OutputSerialization: { JSON: { RecordDelimiter: '\n' } },
  }));

  const records = [];
  await new Promise((resolve, reject) => {
    try {
      resp.Payload.on('data', (event) => {
        if (event.Records && event.Records.Payload) {
          const chunk = event.Records.Payload.toString('utf8');
          const lines = chunk.split('\n').filter(Boolean);
          for (const line of lines) {
            try { records.push(JSON.parse(line)); } catch {}
          }
        }
      });
      resp.Payload.on('end', resolve);
      resp.Payload.on('error', reject);
    } catch (e) { reject(e); }
  });

  return records; // Array<Object>
}

// DataFrameライク：Array<Object>で操作
function sortByTsThenDevice(rows) {
  const hasDevice = rows.length && Object.prototype.hasOwnProperty.call(rows[0], 'device_id');
  if (hasDevice) {
    rows.sort((a, b) => {
      if (a.device_id < b.device_id) return -1;
      if (a.device_id > b.device_id) return 1;
      return (a.ts_ns ?? 0) - (b.ts_ns ?? 0);
    });
  } else {
    rows.sort((a, b) => (a.ts_ns ?? 0) - (b.ts_ns ?? 0));
  }
  return hasDevice;
}

function toNumber(x) {
  const v = (x === null || x === undefined) ? NaN : Number(x);
  return Number.isFinite(v) ? v : NaN;
}

// Δt と next quaternion を付与
function ensureDtAndNext(rows) {
  const hasDevice = sortByTsThenDevice(rows);
  // dt_s
  if (!rows.every(r => 'dt_s' in r)) {
    if (hasDevice) {
      let prevByDev = new Map();
      for (const r of rows) {
        const dev = r.device_id;
        const ts = toNumber(r.ts_ns);
        if (!prevByDev.has(dev)) {
          r.dt_s = 0;
          prevByDev.set(dev, ts);
        } else {
          const prevTs = prevByDev.get(dev);
          r.dt_s = Number.isFinite(prevTs) && Number.isFinite(ts) ? (ts - prevTs) / 1e9 : 0;
          prevByDev.set(dev, ts);
        }
      }
    } else {
      let prevTs = null;
      for (const r of rows) {
        const ts = toNumber(r.ts_ns);
        r.dt_s = (prevTs == null || !Number.isFinite(prevTs) || !Number.isFinite(ts)) ? 0 : (ts - prevTs) / 1e9;
        prevTs = ts;
      }
    }
  }

  // next quaternion
  const needNext = ['qw_next','qx_next','qy_next','qz_next'].some(c => !(c in rows[0]));
  if (needNext) {
    if (hasDevice) {
      let prevDev = null;
      for (let i = 0; i < rows.length; i++) {
        const curr = rows[i];
        const next = (i + 1 < rows.length && rows[i+1].device_id === curr.device_id) ? rows[i+1] : null;
        curr.qw_next = next?.qw;
        curr.qx_next = next?.qx;
        curr.qy_next = next?.qy;
        curr.qz_next = next?.qz;
      }
    } else {
      for (let i = 0; i < rows.length; i++) {
        const next = rows[i+1];
        rows[i].qw_next = next?.qw;
        rows[i].qx_next = next?.qx;
        rows[i].qy_next = next?.qy;
        rows[i].qz_next = next?.qz;
      }
    }
  }

  // dropna 相当（next欠損除去）
  const cleaned = rows.filter(r =>
    [r.qw_next, r.qx_next, r.qy_next, r.qz_next].every(v => Number.isFinite(toNumber(v)))
  );
  return cleaned;
}

function normalizeQuat(q) {
  // q: [qw,qx,qy,qz]
  const n = Math.hypot(q[0], q[1], q[2], q[3]) + 1e-8;
  return [q[0]/n, q[1]/n, q[2]/n, q[3]/n];
}

function selectFeatures(rows, cand = ["qw","qx","qy","qz","ax","ay","az","vx","vy","vz","dt_s"]) {
  const present = cand.filter(c => rows.some(r => r[c] !== undefined));
  return present;
}

function rowsToTensors(rows, feats) {
  const X = [];
  const Qnow = [];
  const Y = [];
  for (const r of rows) {
    const x = feats.map(f => toNumber(r[f]));
    if (x.some(v => !Number.isFinite(v))) continue;

    const q0 = [toNumber(r.qw), toNumber(r.qx), toNumber(r.qy), toNumber(r.qz)];
    const qt = [toNumber(r.qw_next), toNumber(r.qx_next), toNumber(r.qy_next), toNumber(r.qz_next)];
    if (![...q0, ...qt].every(Number.isFinite)) continue;

    X.push(x);
    Qnow.push(normalizeQuat(q0));
    Y.push(normalizeQuat(qt));
  }
  return {
    x: tf.tensor2d(X, [X.length, feats.length], 'float32'),
    qnow: tf.tensor2d(Qnow, [Qnow.length, 4], 'float32'),
    y: tf.tensor2d(Y, [Y.length, 4], 'float32'),
  };
}

// ===== 1) データ読み込み =====
async function loadTrainVal() {
  const hasTrain = await s3KeyExists(BUCKET, TRAIN);
  const hasVal   = await s3KeyExists(BUCKET, VAL);
  if (hasTrain) {
    const tr = await readParquetViaSelectToObjects(BUCKET, TRAIN);
    let va = [];
    if (hasVal) {
      va = await readParquetViaSelectToObjects(BUCKET, VAL);
    } else {
      // valが無ければ同日batchを使う分割fallback
      const all = await readParquetViaSelectToObjects(BUCKET, BATCH);
      const rows = ensureDtAndNext(all);
      const hasDevice = sortByTsThenDevice(rows);
      const n = Math.floor(rows.length * 0.8);
      return [rows.slice(0, n), rows.slice(n)];
    }
    return [ensureDtAndNext(tr), ensureDtAndNext(va)];
  } else {
    // trainが無ければbatchを8:2で分割
    const all = await readParquetViaSelectToObjects(BUCKET, BATCH);
    const rows = ensureDtAndNext(all);
    const n = Math.floor(rows.length * 0.8);
    return [rows.slice(0, n), rows.slice(n)];
  }
}

// ===== 4) モデル & 損失 =====
function buildModel(dIn, dH = 256) {
  // 入力: [feats] と [q_now(4)] を結合してDense→Dense→4、最後にL2正規化
  const xIn = tf.input({ shape: [dIn], name: 'x' });
  const qIn = tf.input({ shape: [4],   name: 'q_now' });
  const h   = tf.layers.concatenate().apply([xIn, qIn]);
  const h1  = tf.layers.dense({ units: dH, activation: 'relu' }).apply(h);
  const h2  = tf.layers.dense({ units: dH, activation: 'relu' }).apply(h1);
  const out = tf.layers.dense({ units: 4, activation: null }).apply(h2);
  // L2正規化をcustom layer風に
  const norm = tf.layers.lambda({
    func: (q) => tf.div(q, tf.add(tf.norm(q, 'euclidean', -1, true), 1e-8)),
  }).apply(out);

  const model = tf.model({ inputs: [xIn, qIn], outputs: norm, name: 'FwdModel' });
  return model;
}

// 測地損失 θ=2 arccos(|sum(qp*qt)|)
function geodesicLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const dot = tf.sum(tf.mul(yTrue, yPred), -1).abs().clipByValue(0.0, 1.0 - 1e-7);
    const theta = tf.mul(2.0, tf.acos(dot));
    return tf.mean(tf.mul(theta, theta));
  });
}

function degMetric(yTrue, yPred) {
  return tf.tidy(() => {
    const dot = tf.sum(tf.mul(yTrue, yPred), -1).abs().clipByValue(0.0, 1.0 - 1e-7);
    const theta = tf.mul(2.0, tf.acos(dot));
    const deg = tf.mul(theta, 180 / Math.PI);
    return tf.mean(deg);
  });
}

// ===== 5) 学習 =====
async function train(model, tr, va, batchSize = 512, epochs = 20) {
  model.compile({
    optimizer: tf.train.adamw(3e-4, 1e-4),
    loss: geodesicLoss,
    metrics: [degMetric],
  });

  const trainDs = tf.data
    .tensorSlices({ x: tr.x, q: tr.qnow, y: tr.y })
    .shuffle(tr.x.shape[0])
    .batch(batchSize);

  const valDs = tf.data
    .tensorSlices({ x: va.x, q: va.qnow, y: va.y })
    .batch(Math.min(1024, va.x.shape[0]));

  let best = { loss: Infinity, weights: null };
  await model.fitDataset(trainDs, {
    epochs,
    validationData: valDs,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const tl = logs?.loss?.toFixed(4);
        const tm = logs?.degMetric?.toFixed(2);
        const vl = logs?.val_loss?.toFixed(4);
        const vm = logs?.val_degMetric?.toFixed(2);
        console.log(`[${String(epoch+1).padStart(2,'0')}] train: loss=${tl}, mean°=${tm} | val: loss=${vl}, mean°=${vm}`);

        if (logs?.val_loss != null && logs.val_loss < best.loss) {
          best.loss = logs.val_loss;
          best.weights = model.getWeights().map(w => w.clone());
        }
      },
    },
  });

  if (best.weights) {
    model.setWeights(best.weights);
    best.weights.forEach(w => w.dispose());
  }
  console.log('best val loss:', best.loss);
}

// ===== 6) 推論関数（1件確認用） =====
function predictNext(model, q_now_arr, x_arr) {
  return tf.tidy(() => {
    const q = tf.tensor2d(Array.isArray(q_now_arr[0]) ? q_now_arr : [q_now_arr], [1, 4], 'float32');
    const x = tf.tensor2d(Array.isArray(x_arr[0]) ? x_arr : [x_arr], [1, x_arr.length], 'float32');
    const out = model.predict({ x, q });
    const qPred = out.dataSync();
    return qPred;
  });
}

function angleErrDeg(q1, q2) {
  const dot = Math.min(Math.abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]), 1 - 1e-7);
  return 2 * Math.acos(dot) * 180 / Math.PI;
}

// ===== 7) モデル保存（tfjs形式をS3へ） =====
async function saveModelToS3(model) {
  // tfjsは model.json + shard(s).bin 形式。tmpに保存してからアップロード。
  const now = new Date();
  const baseKey = `models/pose_fwd/${now.getUTCFullYear()}/${String(now.getUTCMonth()+1).padStart(2,'0')}/${String(now.getUTCDate()).padStart(2,'0')}`;
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pose-model-'));

  // 保存
  await model.save(tf.io.fileSystem(tmpDir));
  // 生成ファイル探索
  const files = fs.readdirSync(tmpDir).map(n => path.join(tmpDir, n));
  for (const f of files) {
    const body = fs.createReadStream(f);
    const key = `${baseKey}/${path.basename(f)}`;
    const contentType = f.endsWith('.json') ? 'application/json' : 'application/octet-stream';
    await new Upload({
      client: s3,
      params: { Bucket: BUCKET, Key: key, Body: body, ContentType: contentType },
    }).done();
    console.log('uploaded:', `s3://${BUCKET}/${key}`);
  }
}

// ===== メイン =====
(async () => {
  // 1) 読み込み
  const [trainRowsRaw, valRowsRaw] = await loadTrainVal();

  // 2) Δt＆next作成・特徴選択・NaN除去
  const trainRows = ensureDtAndNext(trainRowsRaw);
  const valRows   = ensureDtAndNext(valRowsRaw);
  const feats = selectFeatures(trainRows);

  // 行→テンソル
  const tr = rowsToTensors(trainRows, feats);
  const va = rowsToTensors(valRows, feats);

  console.log('features:', feats);
  console.log('train/val sizes:', tr.x.shape[0], va.x.shape[0]);

  // 4) モデル
  const model = buildModel(feats.length);

  // 5) 学習
  await train(model, tr, va, 512, 20);

  // 6) サンプル1件で確認
  if (valRows.length > 0) {
    const sample = valRows[0];
    const x0 = feats.map(f => toNumber(sample[f]));
    const q0 = normalizeQuat([toNumber(sample.qw), toNumber(sample.qx), toNumber(sample.qy), toNumber(sample.qz)]);
    const pred = predictNext(model, q0, x0);
    const gt = normalizeQuat([sample.qw_next, sample.qx_next, sample.qy_next, sample.qz_next].map(toNumber));
    console.log('pred:', pred);
    console.log('gt  :', gt);
    console.log('angle error [deg]:', angleErrDeg(pred, gt));
  }

  // 7) 保存（S3, tfjs形式）
  await saveModelToS3(model);

  // メモリ掃除
  tr.x.dispose(); tr.qnow.dispose(); tr.y.dispose();
  va.x.dispose(); va.qnow.dispose(); va.y.dispose();
})();
