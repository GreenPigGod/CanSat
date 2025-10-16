%pip install --quiet pyarrow fastparquet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


#1) 設定 & データ読み込み（S3→DataFrame）
import boto3, json, gzip
from io import BytesIO
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from datetime import datetime

BUCKET = "pose-logs-2025"               # ←変更可
# さっき作った curated の日付に合わせて
BASE   = "curated/android/2025/10/10"   # 例）"curated/android/YYYY/MM/DD"

s3 = boto3.client("s3")

def read_parquet_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))

# 存在する方のみ読む（train/val を作っていなければ batch.parquet を train として使う）
keys = [f"{BASE}/train.parquet", f"{BASE}/val.parquet"]
available = []
for k in keys:
    try:
        s3.head_object(Bucket=BUCKET, Key=k)
        available.append(k)
    except s3.exceptions.ClientError:
        pass
if not available:
    available = [f"{BASE}/batch.parquet"]

dfs = [read_parquet_s3(BUCKET, k) for k in available]
if len(dfs) == 1:
    # 8:2 に時系列順で分割
    df = dfs[0].sort_values(["device_id","ts_ns"]) if "device_id" in dfs[0].columns else dfs[0].sort_values("ts_ns")
    n = int(len(df)*0.8)
    df_train, df_val = df.iloc[:n].copy(), df.iloc[n:].copy()
else:
    df_train, df_val = dfs[0], dfs[1]

print(df_train.shape, df_val.shape)
df_train.head()



#2) 特徴量/目的変数の準備（Δt と next 姿勢が無い場合は作成）


import numpy as np

def ensure_dt_and_next(df):
    # Δt[s]
    if "dt_s" not in df.columns:
        if "device_id" in df.columns:
            df = df.sort_values(["device_id","ts_ns"])
            df["dt_s"] = df.groupby("device_id")["ts_ns"].diff().fillna(0)/1e9
        else:
            df = df.sort_values("ts_ns")
            df["dt_s"] = df["ts_ns"].diff().fillna(0)/1e9

    # next quaternion（無ければ作る）
    need_next = any(c not in df.columns for c in ["qw_next","qx_next","qy_next","qz_next"])
    if need_next:
        if "device_id" in df.columns:
            df["qw_next"] = df.groupby("device_id")["qw"].shift(-1)
            df["qx_next"] = df.groupby("device_id")["qx"].shift(-1)
            df["qy_next"] = df.groupby("device_id")["qy"].shift(-1)
            df["qz_next"] = df.groupby("device_id")["qz"].shift(-1)
        else:
            for c in ["w","x","y","z"]:
                df[f"q{c}_next"] = df[f"q{c}"].shift(-1)
    df = df.dropna(subset=["qw_next","qx_next","qy_next","qz_next"]).reset_index(drop=True)
    return df

df_train = ensure_dt_and_next(df_train)
df_val   = ensure_dt_and_next(df_val)

# 使用する入力特徴（存在する列のみ使う）
CAND_IN = ["qw","qx","qy","qz","ax","ay","az","vx","vy","vz","dt_s"]
feats = [c for c in CAND_IN if c in df_train.columns]
print("features:", feats)

# 型を数値に統一 & NaN処理（簡易）
for c in feats + ["qw_next","qx_next","qy_next","qz_next"]:
    df_train[c] = pd.to_numeric(df_train[c], errors="coerce")
    df_val[c]   = pd.to_numeric(df_val[c],   errors="coerce")
df_train = df_train.dropna(subset=feats + ["qw_next","qx_next","qy_next","qz_next"])
df_val   = df_val.dropna(subset=feats + ["qw_next","qx_next","qy_next","qz_next"])

len(df_train), len(df_val)



#3) PyTorch データセット & ローダー

import torch
from torch.utils.data import Dataset, DataLoader

def normalize_quat_np(q):
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8
    return q / n

class PoseDataset(Dataset):
    def __init__(self, df, feats):
        self.X = df[feats].to_numpy(np.float32)
        q_now  = df[["qw","qx","qy","qz"]].to_numpy(np.float32)
        q_tgt  = df[["qw_next","qx_next","qy_next","qz_next"]].to_numpy(np.float32)
        self.q_now = normalize_quat_np(q_now)
        self.y     = normalize_quat_np(q_tgt)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.from_numpy(self.q_now[i]),
            torch.from_numpy(self.y[i]),
        )

train_ds = PoseDataset(df_train, feats)
val_ds   = PoseDataset(df_val,   feats)

train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=1024, shuffle=False, num_workers=0)

len(train_ds), len(val_ds)













#4) モデル & 測地損失（クォータニオン角度誤差）
出力は 次のクォータニオン。最後に L2 正規化。
ロスは θ = 2 arccos(|dot(q_pred, q_true)|) の二乗平均。

import torch.nn as nn
import torch.nn.functional as F
import math

class FwdModel(nn.Module):
    def __init__(self, d_in, d_h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in + 4, d_h), nn.ReLU(),
            nn.Linear(d_h, d_h), nn.ReLU(),
            nn.Linear(d_h, 4)
        )
    def forward(self, x, q_now):
        # 入力に現在姿勢も与える（学習を助ける）
        h = torch.cat([x, q_now], dim=-1)
        q = self.net(h)
        q = F.normalize(q, dim=-1)  # 単位化
        return q

def geodesic_loss(q_pred, q_true, eps=1e-7):
    # |dot| を使い符号の二重性を吸収
    dot = torch.clamp(torch.sum(q_pred * q_true, dim=-1).abs(), 0.0, 1.0 - 1e-7)
    theta = 2.0 * torch.acos(dot)
    return (theta * theta).mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = FwdModel(d_in=len(feats)).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)














#5) 学習ループ


from tqdm.auto import tqdm
import numpy as np

def evaluate(model, loader):
    model.eval()
    losses, degs = [], []
    with torch.no_grad():
        for x, q_now, y in loader:
            x, q_now, y = x.to(device), q_now.to(device), y.to(device)
            q = model(x, q_now)
            # rad → deg も記録
            dot = torch.clamp(torch.sum(q * y, dim=-1).abs(), 0.0, 1.0 - 1e-7)
            theta = 2.0 * torch.acos(dot)
            losses.append((theta*theta).mean().item())
            degs.extend((theta * 180/math.pi).cpu().numpy().tolist())
    return float(np.mean(losses)), float(np.mean(degs)), float(np.percentile(degs,95))

best = {"val_loss": 1e9, "state": None}
EPOCHS = 20

for ep in range(1, EPOCHS+1):
    model.train()
    pbar = tqdm(train_dl, leave=False)
    for x, q_now, y in pbar:
        x, q_now, y = x.to(device), q_now.to(device), y.to(device)
        q = model(x, q_now)
        loss = geodesic_loss(q, y)
        opt.zero_grad(); loss.backward(); opt.step()
        pbar.set_description(f"ep{ep} loss={loss.item():.4f}")

    tr_loss, tr_deg, tr_p95 = evaluate(model, train_dl)
    va_loss, va_deg, va_p95 = evaluate(model, val_dl)
    print(f"[{ep:02d}] train: loss={tr_loss:.4f}, mean°={tr_deg:.2f}, p95°={tr_p95:.2f} | "
          f"val: loss={va_loss:.4f}, mean°={va_deg:.2f}, p95°={va_p95:.2f}")

    if va_loss < best["val_loss"]:
        best = {"val_loss": va_loss, "state": model.state_dict()}

# ベストで上書き
model.load_state_dict(best["state"])
print("best val loss:", best["val_loss"])




#6) 簡易推論関数 & 1件テスト

def predict_next(q_now_np, x_np):
    model.eval()
    with torch.no_grad():
        q_now_t = torch.from_numpy(np.asarray(q_now_np, np.float32)).to(device)
        x_t     = torch.from_numpy(np.asarray(x_np,     np.float32)).to(device)
        if q_now_t.ndim == 1: q_now_t = q_now_t[None, :]
        if x_t.ndim == 1:     x_t     = x_t[None, :]
        q = model(x_t, q_now_t)
        return F.normalize(q, dim=-1).cpu().numpy()

# 先頭のサンプルで確認
sample = df_val.iloc[0]
x0 = sample[feats].to_numpy(np.float32)
q0 = sample[["qw","qx","qy","qz"]].to_numpy(np.float32)
pred = predict_next(q0, x0)[0]
gt   = sample[["qw_next","qx_next","qy_next","qz_next"]].to_numpy(np.float32)

def ang_err_deg(q1,q2):
    dot = np.clip(np.abs(np.dot(q1,q2)), 0.0, 1.0 - 1e-7)
    return float(2*np.arccos(dot) * 180/np.pi)

print("pred:", pred)
print("gt  :", gt)
print("angle error [deg]:", ang_err_deg(pred, gt))






















7) モデルを S3 へ保存（.pt）

import torch, os

now = datetime.utcnow()
model_key = f"models/pose_fwd/{now:%Y/%m/%d}/model.pt"

buf = BytesIO()
torch.save({"state_dict": model.state_dict(), "feats": feats}, buf)
s3.put_object(Bucket=BUCKET, Key=model_key, Body=buf.getvalue(),
              ContentType="application/octet-stream")
print("uploaded:", f"s3://{BUCKET}/{model_key}")
