"""Per-horizon disaggregated metrics export (S-10c).

Purpose
-------
Compute MSE / MAE / RMSE at each forecast step h ∈ [1, 24] for each of the
four core models, on the locked test partition. Used for descriptive
horizon-wise comparison with the literature (Zhou et al., 2021;
Zeng et al., 2023) and for the per-horizon visualisation in Discussion.

Explicit scope constraint (CLAUDE.md § 11C / § 16 / S-10c plan T2.1)
-------------------------------------------------------------------
    No new per-variable or per-horizon significance testing is introduced
    here; outputs are used for descriptive disaggregation only. The paired
    bootstrap layer (`src/evaluation/post_training_analysis.py` Section 3)
    remains the only sanctioned uncertainty layer in this thesis, and it
    operates on the overall (windows × horizon × variable)-collapsed metric.
    Per-horizon ranking-style claims (e.g., "model A is significantly better
    than model B at horizon 12") are not supported by the present
    aggregation and are explicitly out of scope.

§ 11C lifting
-------------
This script lifts the disaggregation-doctrine restriction *for the
horizon axis only* and *for descriptive comparison only*. The per-horizon
view enables prose statements such as:
    ✓ "the horizon-wise error profile suggests..."
    ✓ "a descriptive comparison with prior reports indicates..."
    ✓ "under the present configuration, the per-horizon view shows..."
and forbids:
    ✗ "transformers degrade with horizon, therefore..."
    ✗ "this proves..."
    ✗ "literature confirmed..."

Inputs (frozen artefacts produced upstream)
-------------------------------------------
    results/preds_lr.npy                — (3365, 24, 7) original scale
    results/preds_{mlp,lstm,tft}_seed{42,123,456}.npy

Output
------
    results/per_horizon_metrics.csv
        columns: model, horizon, mse, mae, rmse, seed_count, seed_std_mse

Run
---
    python src/evaluation/per_horizon_metrics.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# Constants (CLAUDE.md § 10 locked)
# ============================================================
SEEDS = [42, 123, 456]
INPUT_LEN = 96
OUTPUT_LEN = 24
N_FEATURES = 7
ETTH1_VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


# ============================================================
# Helpers (inline standalone)
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


def per_horizon_mse_mae_rmse(y_true_orig, y_pred_orig):
    """For each horizon h, mean squared / absolute error over (windows × variables).

    y_true_orig, y_pred_orig: (n_test, OUTPUT_LEN, n_features)
    Returns: arrays of shape (OUTPUT_LEN,) for mse, mae, rmse.
    """
    diff = y_true_orig - y_pred_orig
    mse_per_h = np.mean(diff ** 2, axis=(0, 2))
    mae_per_h = np.mean(np.abs(diff), axis=(0, 2))
    rmse_per_h = np.sqrt(mse_per_h)
    return mse_per_h, mae_per_h, rmse_per_h


# ============================================================
# 0. Reconstruct y_test in original scale (aligned with preds_*.npy)
# ============================================================
print("=" * 70)
print("Per-horizon disaggregated metrics (S-10c, descriptive only)")
print("=" * 70)
print("\n[0/3] Loading ETTh1 + rebuilding y_test (original scale)...")

df = pd.read_csv("data/ETTh1.csv").drop(columns=["date"])
n = len(df)
train_end = int(n * 0.6)
val_end = int(n * 0.8)
train = df.iloc[:train_end]
test = df.iloc[val_end:]

scaler = StandardScaler().fit(train)
test_scaled = scaler.transform(test)
_, y_test = create_windows(test_scaled)
y_test_orig = inverse_transform_3d(y_test, scaler)
n_test = y_test_orig.shape[0]
assert y_test_orig.shape == (3365, OUTPUT_LEN, N_FEATURES), \
    f"unexpected y_test shape: {y_test_orig.shape}"
print(f"  y_test_orig shape: {y_test_orig.shape}")


# ============================================================
# 1. Per-model per-horizon MSE/MAE/RMSE
# ============================================================
print("\n[1/3] Computing per-horizon metrics for 4 models...")

rows = []

# LR — deterministic single trajectory
lr_preds = np.load("results/preds_lr.npy")
mse_h, mae_h, rmse_h = per_horizon_mse_mae_rmse(y_test_orig, lr_preds)
for h in range(OUTPUT_LEN):
    rows.append({
        "model": "LR",
        "horizon": h + 1,  # 1-indexed for prose readability
        "mse": float(mse_h[h]),
        "mae": float(mae_h[h]),
        "rmse": float(rmse_h[h]),
        "seed_count": 1,
        "seed_std_mse": 0.0,
    })

# MLP / LSTM / TFT — per-seed MSE then seed-mean ± std
for model_name in ("mlp", "lstm", "tft"):
    per_seed_mse = []  # list of (24,) arrays
    per_seed_mae = []
    per_seed_rmse = []
    for seed in SEEDS:
        preds = np.load(f"results/preds_{model_name}_seed{seed}.npy")
        mse_h, mae_h, rmse_h = per_horizon_mse_mae_rmse(y_test_orig, preds)
        per_seed_mse.append(mse_h)
        per_seed_mae.append(mae_h)
        per_seed_rmse.append(rmse_h)
    mse_stack = np.stack(per_seed_mse, axis=0)  # (3, 24)
    mae_stack = np.stack(per_seed_mae, axis=0)
    rmse_stack = np.stack(per_seed_rmse, axis=0)
    mse_mean = mse_stack.mean(axis=0)
    mse_std = mse_stack.std(axis=0, ddof=1)
    mae_mean = mae_stack.mean(axis=0)
    rmse_mean = rmse_stack.mean(axis=0)
    for h in range(OUTPUT_LEN):
        rows.append({
            "model": model_name.upper(),
            "horizon": h + 1,
            "mse": float(mse_mean[h]),
            "mae": float(mae_mean[h]),
            "rmse": float(rmse_mean[h]),
            "seed_count": len(SEEDS),
            "seed_std_mse": float(mse_std[h]),
        })


# ============================================================
# 2. Save CSV
# ============================================================
print("\n[2/3] Saving CSV...")
out_df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
out_path = "results/per_horizon_metrics.csv"
out_df.to_csv(out_path, index=False)
print(f"  saved {out_path}  ({len(out_df)} rows = 4 models × 24 horizons)")


# ============================================================
# 3. Compact summary print (descriptive)
# ============================================================
print("\n[3/3] Compact summary (descriptive only):")
print("       MSE at h=1, h=12, h=24 per model:")
print(f"  {'Model':<6}  {'h=1':>10}  {'h=12':>10}  {'h=24':>10}")
print("  " + "-" * 42)
for model in ["LR", "MLP", "LSTM", "TFT"]:
    sub = out_df[out_df["model"] == model].set_index("horizon")["mse"]
    if not sub.empty:
        print(f"  {model:<6}  {sub.loc[1]:>10.4f}  {sub.loc[12]:>10.4f}  {sub.loc[24]:>10.4f}")

print("\n" + "-" * 70)
print("Reading guide:")
print("  Each row is one (model, horizon) cell. Stochastic models report seed-mean")
print("  ± seed_std_mse over seeds 42/123/456 (ddof=1). LR is deterministic.")
print("  This is a DESCRIPTIVE disaggregation; no significance testing applies on")
print("  the per-horizon axis. § 11C overall-only ranking discipline preserved.")
print("-" * 70)
