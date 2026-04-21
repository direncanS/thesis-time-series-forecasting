"""Export aligned per-test-window predictions for all 4 core models.

Producer for the common data layer consumed by:
    - src/evaluation/post_training_analysis.py  (S-04 paired bootstrap intervals,
                                                 formerly src/uncertainty_bootstrap.py;
                                                 merged S-09c)
    - src/explainability/faithfulness_test.py   (S-08 faithfulness evaluation;
                                                 S-10b rewrite — checkpoint-based)

Outputs 10 files with identical shape (3365, 24, 7), all in original scale
(post inverse_transform_3d). Window alignment is preserved because every model
operates on the same test_scaled split with the same create_windows logic
(INPUT_LEN=96, OUTPUT_LEN=24, stride=1).

    results/preds_lr.npy
    results/preds_mlp_seed{42,123,456}.npy
    results/preds_lstm_seed{42,123,456}.npy
    results/preds_tft_seed{42,123,456}.npy

Deterministic re-fit for LR (no checkpoint exists); MLP / LSTM / TFT load from
their saved best-val-loss checkpoints produced by src/training/multi_seed.py and
src/training/tft_fair_3seed.py. Per CLAUDE.md § 18 the student runs this script
manually.
"""

import os

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import MultiHorizonMetric
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# MSE class — must be defined at module level for pickle to resolve
# `__main__.MSE` when loading TFT checkpoints saved by tft_fair_3seed.py
# (where MSE is also defined at module level). Byte-compatible definition.
# ============================================================
class MSE(MultiHorizonMetric):
    """Mean squared error — symmetric with nn.MSELoss; duplicates the class
    defined in src/tft_fair_3seed.py so checkpoints saved there can be
    unpickled here without a shared module dependency."""

    def __init__(self, reduction="mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def loss(self, y_pred, target):
        return torch.pow(self.to_prediction(y_pred) - target, 2)


# ============================================================
# Constants — mirror CLAUDE.md § 10 / src/multi_seed.py / src/tft_fair_3seed.py
# ============================================================
SEEDS = [42, 123, 456]
INPUT_LEN = 96
OUTPUT_LEN = 24
BATCH_SIZE = 64


# ============================================================
# Helpers — inline copy of src/multi_seed.py functions (avoids running
# multi_seed.py at module load; we only need the helpers, not the training).
# ============================================================
def create_windows(data, input_len=INPUT_LEN, output_len=OUTPUT_LEN):
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len])
    return np.array(X), np.array(y)


def inverse_transform_3d(arr_3d, scaler):
    n_w, t, n_f = arr_3d.shape
    flat = arr_3d.reshape(n_w * t, n_f)
    return scaler.inverse_transform(flat).reshape(n_w, t, n_f)


# ============================================================
# Model classes — inline copy from src/multi_seed.py (same architecture so
# load_state_dict matches the saved .pt weights exactly).
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


# ============================================================
# 0. Load + split + scale — identical pipeline to multi_seed.py and
#    tft_fair_3seed.py (same scaler fit on train only).
# ============================================================
print("Loading ETTh1 + split + scale...")
df = pd.read_csv("data/ETTh1.csv")
features = df.drop(columns=["date"])
var_names = features.columns.tolist()
n_features = len(var_names)

n = len(features)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

train = features.iloc[:train_end]
val = features.iloc[train_end:val_end]
test = features.iloc[val_end:]

scaler = StandardScaler()
scaler.fit(train)

train_scaled = scaler.transform(train)
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)

# Split-then-window — matches multi_seed.py lines 203-205 and
# tft_fair_3seed.py per-split DataFrames. No cross-split encoder leakage.
X_train, y_train = create_windows(train_scaled)
X_test, y_test = create_windows(test_scaled)

n_train = X_train.shape[0]
n_test = X_test.shape[0]
print(f"n_test windows: {n_test} (expected 3365)")

X_train_flat = X_train.reshape(n_train, -1)
X_test_flat = X_test.reshape(n_test, -1)
y_train_flat = y_train.reshape(n_train, -1)

os.makedirs("results", exist_ok=True)


# ============================================================
# 1. LR — deterministic re-fit, no checkpoint
# ============================================================
print("\n[LR] deterministic re-fit + predict...")
lr_model = LinearRegression()
lr_model.fit(X_train_flat, y_train_flat)

preds_lr_scaled = lr_model.predict(X_test_flat).reshape(
    n_test, OUTPUT_LEN, n_features
)
preds_lr_orig = inverse_transform_3d(preds_lr_scaled, scaler)
np.save("results/preds_lr.npy", preds_lr_orig)
print(f"  saved results/preds_lr.npy  shape={preds_lr_orig.shape}")


# ============================================================
# 2. MLP — load checkpoint, forward pass on test flat
# ============================================================
mlp_input_size = X_train_flat.shape[1]   # 672
mlp_output_size = y_train_flat.shape[1]  # 168

X_test_flat_tensor = torch.FloatTensor(X_test_flat)

for seed in SEEDS:
    print(f"\n[MLP seed {seed}] load + predict...")
    model = MLP(mlp_input_size, mlp_output_size)
    state = torch.load(
        f"checkpoints/mlp_seed{seed}.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        preds_flat = model(X_test_flat_tensor).numpy()
    preds_scaled = preds_flat.reshape(n_test, OUTPUT_LEN, n_features)
    preds_orig = inverse_transform_3d(preds_scaled, scaler)
    out_path = f"results/preds_mlp_seed{seed}.npy"
    np.save(out_path, preds_orig)
    print(f"  saved {out_path}  shape={preds_orig.shape}")


# ============================================================
# 3. LSTM — load checkpoint, forward pass on test 3D tensor
# ============================================================
X_test_tensor = torch.FloatTensor(X_test)  # (n_test, 96, 7)

for seed in SEEDS:
    print(f"\n[LSTM seed {seed}] load + predict...")
    model = LSTMModel(
        input_size=n_features, hidden_size=64, output_size=mlp_output_size
    )
    state = torch.load(
        f"checkpoints/lstm_seed{seed}.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        preds_flat = model(X_test_tensor).numpy()
    preds_scaled = preds_flat.reshape(n_test, OUTPUT_LEN, n_features)
    preds_orig = inverse_transform_3d(preds_scaled, scaler)
    out_path = f"results/preds_lstm_seed{seed}.npy"
    np.save(out_path, preds_orig)
    print(f"  saved {out_path}  shape={preds_orig.shape}")


# ============================================================
# 4. TFT — rebuild per-split TimeSeriesDataSet, load checkpoint, predict
# ============================================================
def make_split_df(scaled_arr):
    d = pd.DataFrame(scaled_arr, columns=var_names)
    d["time_idx"] = np.arange(len(d), dtype=np.int64)
    d["group_id"] = "ETTh1"
    return d


def make_identity_normalizer():
    return MultiNormalizer(
        [TorchNormalizer(method="identity") for _ in range(n_features)]
    )


train_df = make_split_df(train_scaled)
test_df = make_split_df(test_scaled)

training_ds = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target=var_names,
    group_ids=["group_id"],
    min_encoder_length=INPUT_LEN,
    max_encoder_length=INPUT_LEN,
    min_prediction_length=OUTPUT_LEN,
    max_prediction_length=OUTPUT_LEN,
    time_varying_unknown_reals=var_names,
    time_varying_known_reals=[],
    time_varying_known_categoricals=[],
    target_normalizer=make_identity_normalizer(),
    add_relative_time_idx=False,
    add_target_scales=False,
    add_encoder_length=False,
    allow_missing_timesteps=False,
)
test_ds = TimeSeriesDataSet.from_dataset(
    training_ds, test_df, stop_randomization=True,
)
test_loader = test_ds.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0
)

for seed in SEEDS:
    print(f"\n[TFT seed {seed}] load + predict...")
    ckpt_path = f"checkpoints/tft_seed{seed}.ckpt"
    tft = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    tft.eval()

    raw = tft.predict(
        test_loader,
        mode="prediction",
        return_y=False,
        trainer_kwargs={"accelerator": "auto", "devices": 1, "logger": False},
    )
    # raw is either a tensor (n_test, 24) [single target] or a list of 7
    # tensors (n_test, 24) [multi-target]. Multi-target path is what we use.
    if isinstance(raw, (list, tuple)):
        pred_stack = torch.stack(list(raw), dim=-1).cpu().numpy()
    else:
        pred_stack = raw.cpu().numpy()
    if pred_stack.ndim == 2:
        pred_stack = pred_stack[..., None]

    # Scaled space (identity normalizer → nothing to undo on TFT side);
    # apply our shared StandardScaler inverse to reach original scale.
    preds_orig = inverse_transform_3d(pred_stack, scaler)
    out_path = f"results/preds_tft_seed{seed}.npy"
    np.save(out_path, preds_orig)
    print(f"  saved {out_path}  shape={preds_orig.shape}")


# ============================================================
# 5. Final shape report
# ============================================================
print("\n" + "=" * 60)
print("DONE — prediction export")
print("=" * 60)

expected_shape = (n_test, OUTPUT_LEN, n_features)
all_files = (
    ["results/preds_lr.npy"]
    + [f"results/preds_mlp_seed{s}.npy" for s in SEEDS]
    + [f"results/preds_lstm_seed{s}.npy" for s in SEEDS]
    + [f"results/preds_tft_seed{s}.npy" for s in SEEDS]
)
print(f"\nExpected shape for all files: {expected_shape}")
for p in all_files:
    arr = np.load(p)
    ok = arr.shape == expected_shape
    print(f"  {p:45s}  shape={arr.shape}  {'OK' if ok else 'SHAPE MISMATCH'}")

# Smoke test: OT (column 6) first window first horizon — all 10 files
# should produce finite values in the original-scale OT range (~5-45 typical).
print("\nSmoke test — OT[:, 0, 6] across all 10 files (should be finite, similar order):")
for p in all_files:
    arr = np.load(p)
    sample = float(arr[0, 0, 6])
    print(f"  {p:45s}  OT[0,0,6] = {sample:.3f}")
