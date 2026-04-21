"""Faithfulness test (S-10b refactor — 4-model, 3-seed, checkpoint-based)

Purpose
-------
Verify that each model's variable-importance ranking (SHAP for LR/MLP/LSTM,
VSN for TFT) is *faithful* to the model's actual behaviour under perturbation.

Logic
-----
For each (model, k ∈ {1, 2, 3}):
    1. Identify top-k most important variables from that model's ranking.
    2. Mask them in scaled-space test input (set to 0 = training mean).
    3. Measure MSE increase on the locked test partition.
    4. Compare to bottom-k masking (least important variables).
    5. Ranking is "faithful" if top-k masking produces a *larger* MSE
       increase than bottom-k masking.

Explicit scope sentence (CLAUDE.md § 13 + S-10b refinement)
----------------------------------------------------------
    The TFT faithfulness test evaluates ranking-consistency under VSN-guided
    perturbation and is not treated as method-equivalent to SHAP-based
    attribution faithfulness. SHAP (LR/MLP/LSTM) and VSN (TFT) are different
    explanation families (post-hoc vs architecture-native, § 13); the
    faithfulness score here measures **behavioural consistency** of each
    model's own ranking under masking, not cross-method explanation
    equivalence.

Alignment with locked defaults (CLAUDE.md § 10)
-----------------------------------------------
This refactor (S-10b, 2026-04-20) loads the **validated best-val-loss
checkpoints** rather than re-training with different hyperparameters. All
evaluations use:
    - scaled-space MSE (matches y_test scaled from train-only StandardScaler)
    - same test window set as multi_seed.py / tft_fair_3seed.py (n_test=3365)
    - 3 seeds for stochastic models (MLP/LSTM/TFT); LR deterministic single
Per-seed MSE values are averaged before comparison so that the reported
faithfulness is robust to seed noise.

Prerequisites
-------------
    - SHAP CSVs exist: `results/shap_{lr,mlp,lstm}.csv`
    - VSN CSV exists: `results/tft_importance.csv`
    - Checkpoints exist:
        checkpoints/mlp_seed{42,123,456}.pt
        checkpoints/lstm_seed{42,123,456}.pt
        checkpoints/tft_seed{42,123,456}.ckpt

Output
------
    results/faithfulness.csv
        columns: model, k, mask_type, masked_vars, mse_scaled_mean,
                 mse_increase, mse_increase_pct, seed_count

Run
---
    python src/explainability/faithfulness_test.py
"""

import os
import warnings

import lightning as L  # noqa: F401  (pytorch-forecasting dependency)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import MultiHorizonMetric
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# MSE class — module-level for TFT checkpoint pickle resolution
# (byte-compatible with src/training/tft_fair_3seed.py MSE class;
# also mirrored in src/evaluation/export_predictions.py).
# ============================================================
class MSE(MultiHorizonMetric):
    def __init__(self, reduction="mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def loss(self, y_pred, target):
        return torch.pow(self.to_prediction(y_pred) - target, 2)


# ============================================================
# Constants (CLAUDE.md § 10 locked)
# ============================================================
SEEDS = [42, 123, 456]
INPUT_LEN = 96
OUTPUT_LEN = 24
BATCH_SIZE = 64
K_VALUES = [1, 2, 3]
ETTH1_VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


# ============================================================
# Helpers (inline — standalone script, no internal imports)
# ============================================================
def create_windows(data, input_len=INPUT_LEN, output_len=OUTPUT_LEN):
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len])
    return np.array(X), np.array(y)


def mask_variables_in_scaled_window(X_3d, var_indices):
    """Set specified variable columns to 0 (training mean in scaled space)."""
    X_masked = X_3d.copy()
    for idx in var_indices:
        X_masked[:, :, idx] = 0.0
    return X_masked


# ============================================================
# Model classes (byte-identical to training/multi_seed.py)
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
# 0. Load + split + scale (identical to multi_seed.py)
# ============================================================
print("=" * 72)
print("Faithfulness test (4-model, 3-seed, checkpoint-based)")
print("=" * 72)
print("\n[0/6] Load + split + scale ETTh1...")

df = pd.read_csv("data/ETTh1.csv").drop(columns=["date"])
assert list(df.columns) == ETTH1_VARS, f"column order mismatch: {list(df.columns)}"
n = len(df)
train_end = int(n * 0.6)
val_end = int(n * 0.8)
train = df.iloc[:train_end]
test = df.iloc[val_end:]

scaler = StandardScaler().fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

X_train, y_train = create_windows(train_scaled)
X_test, y_test = create_windows(test_scaled)
n_train, n_test = X_train.shape[0], X_test.shape[0]
X_train_flat = X_train.reshape(n_train, -1)
X_test_flat = X_test.reshape(n_test, -1)
y_train_flat = y_train.reshape(n_train, -1)

n_features = len(ETTH1_VARS)
var_to_idx = {v: i for i, v in enumerate(ETTH1_VARS)}
input_flat = X_train_flat.shape[1]
output_flat = y_train_flat.shape[1]

print(f"  n_test windows: {n_test} (expected 3365)")
assert n_test == 3365, "n_test mismatch with locked baseline"


# ============================================================
# 1. Load rankings (SHAP for LR/MLP/LSTM, VSN for TFT)
# ============================================================
print("\n[1/6] Loading variable-importance rankings...")


def load_shap_ranking(csv_path, model_name):
    d = pd.read_csv(csv_path)
    if "shap_rank" in d.columns:
        ranked = d.sort_values("shap_rank")["variable"].tolist()
    else:
        agg = d.groupby("variable")["shap_importance"].mean().sort_values(ascending=False)
        ranked = agg.index.tolist()
    assert set(ranked) == set(ETTH1_VARS), f"{model_name} ranking missing vars"
    return ranked


def load_vsn_ranking():
    d = pd.read_csv("results/tft_importance.csv")
    agg = d[d["variable"].isin(ETTH1_VARS)].groupby("variable")["importance"].mean()
    return agg.sort_values(ascending=False).index.tolist()


rankings = {}
for name, path in [
    ("LR", "results/shap_lr.csv"),
    ("MLP", "results/shap_mlp.csv"),
    ("LSTM", "results/shap_lstm.csv"),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing — run shap_{name.lower()}.py first")
    rankings[name] = load_shap_ranking(path, name)
    print(f"  {name} SHAP ranking: {rankings[name]}")

if not os.path.exists("results/tft_importance.csv"):
    raise FileNotFoundError(
        "results/tft_importance.csv missing — run src/training/tft_fair_3seed.py first"
    )
rankings["TFT"] = load_vsn_ranking()
print(f"  TFT VSN ranking: {rankings['TFT']}")


# ============================================================
# 2. LR — deterministic re-fit (single evaluation)
# ============================================================
print("\n[2/6] LR deterministic fit...")
lr_model = LinearRegression()
lr_model.fit(X_train_flat, y_train_flat)


def mse_scaled_lr(X_3d_scaled):
    flat = X_3d_scaled.reshape(X_3d_scaled.shape[0], -1)
    pred = lr_model.predict(flat).reshape(n_test, OUTPUT_LEN, n_features)
    return float(np.mean((y_test - pred) ** 2))


# ============================================================
# 3. MLP / LSTM — load checkpoints per seed
# ============================================================
print("\n[3/6] Load MLP + LSTM checkpoints (3 seeds each)...")


def load_mlp(seed):
    m = MLP(input_flat, output_flat)
    state = torch.load(f"checkpoints/mlp_seed{seed}.pt",
                       map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


def load_lstm(seed):
    m = LSTMModel(input_size=n_features, hidden_size=64, output_size=output_flat)
    state = torch.load(f"checkpoints/lstm_seed{seed}.pt",
                       map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


mlp_models = {s: load_mlp(s) for s in SEEDS}
lstm_models = {s: load_lstm(s) for s in SEEDS}
print(f"  loaded MLP seeds {SEEDS}, LSTM seeds {SEEDS}")


def mse_scaled_mlp(X_3d_scaled, seed):
    flat = X_3d_scaled.reshape(X_3d_scaled.shape[0], -1)
    with torch.no_grad():
        pred_flat = mlp_models[seed](torch.FloatTensor(flat)).numpy()
    pred = pred_flat.reshape(n_test, OUTPUT_LEN, n_features)
    return float(np.mean((y_test - pred) ** 2))


def mse_scaled_lstm(X_3d_scaled, seed):
    with torch.no_grad():
        pred_flat = lstm_models[seed](torch.FloatTensor(X_3d_scaled)).numpy()
    pred = pred_flat.reshape(n_test, OUTPUT_LEN, n_features)
    return float(np.mean((y_test - pred) ** 2))


# ============================================================
# 4. TFT — build training dataset once + per-seed checkpoint predict
# ============================================================
print("\n[4/6] TFT dataset + per-seed load + predict...")


def make_split_df(scaled_arr):
    d = pd.DataFrame(scaled_arr, columns=ETTH1_VARS)
    d["time_idx"] = np.arange(len(d), dtype=np.int64)
    d["group_id"] = "ETTh1"
    return d


def make_identity_normalizer():
    return MultiNormalizer(
        [TorchNormalizer(method="identity") for _ in range(n_features)]
    )


train_df = make_split_df(train_scaled)
training_ds = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target=ETTH1_VARS,
    group_ids=["group_id"],
    min_encoder_length=INPUT_LEN,
    max_encoder_length=INPUT_LEN,
    min_prediction_length=OUTPUT_LEN,
    max_prediction_length=OUTPUT_LEN,
    time_varying_unknown_reals=ETTH1_VARS,
    time_varying_known_reals=[],
    time_varying_known_categoricals=[],
    target_normalizer=make_identity_normalizer(),
    add_relative_time_idx=False,
    add_target_scales=False,
    add_encoder_length=False,
    allow_missing_timesteps=False,
)


def mse_scaled_tft(test_scaled_masked, tft_model):
    """Reconstruct masked test dataset + predict + compute scaled MSE."""
    masked_df = make_split_df(test_scaled_masked)
    masked_ds = TimeSeriesDataSet.from_dataset(
        training_ds, masked_df, stop_randomization=True,
    )
    masked_loader = masked_ds.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0,
    )
    raw = tft_model.predict(
        masked_loader,
        mode="prediction",
        return_y=False,
        trainer_kwargs={"accelerator": "auto", "devices": 1, "logger": False},
    )
    if isinstance(raw, (list, tuple)):
        pred_stack = torch.stack(list(raw), dim=-1).cpu().numpy()
    else:
        pred_stack = raw.cpu().numpy()
    if pred_stack.ndim == 2:
        pred_stack = pred_stack[..., None]
    # pred_stack already in scaled space (identity normalizer). Reshape to
    # match y_test: TimeSeriesDataSet output is (n_windows, OUTPUT_LEN, n_features).
    if pred_stack.shape != (n_test, OUTPUT_LEN, n_features):
        raise RuntimeError(
            f"TFT prediction shape {pred_stack.shape} != expected "
            f"({n_test}, {OUTPUT_LEN}, {n_features})"
        )
    return float(np.mean((y_test - pred_stack) ** 2))


# ============================================================
# 5. Run faithfulness grid — 4 models × (baseline + 2 × 3 k) = 28 rows
# ============================================================
print("\n[5/6] Running faithfulness grid (4 models × 7 configs each)...")


def mask_configs(ranking):
    """Yield (k, mask_type, masked_vars, masked_indices) for each config."""
    cfgs = [(0, "none", [], [])]
    for k in K_VALUES:
        top = ranking[:k]
        bot = ranking[-k:]
        cfgs.append((k, "top", top, [var_to_idx[v] for v in top]))
        cfgs.append((k, "bottom", bot, [var_to_idx[v] for v in bot]))
    return cfgs


def aggregate_seed_mse(mse_list):
    """Return (mean, std_ddof1) across seeds; single value passes through."""
    if len(mse_list) == 1:
        return float(mse_list[0]), 0.0
    arr = np.array(mse_list, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


results_rows = []

# --- LR ---
print("\n  LR faithfulness...")
for k, mask_type, mvars, midx in mask_configs(rankings["LR"]):
    X_masked = mask_variables_in_scaled_window(X_test, midx) if midx else X_test
    mse = mse_scaled_lr(X_masked)
    results_rows.append({
        "model": "LR", "k": k, "mask_type": mask_type,
        "masked_vars": ", ".join(mvars),
        "mse_scaled_mean": mse, "mse_scaled_std": 0.0, "seed_count": 1,
    })

# --- MLP ---
print("  MLP faithfulness (3 seeds)...")
for k, mask_type, mvars, midx in mask_configs(rankings["MLP"]):
    X_masked = mask_variables_in_scaled_window(X_test, midx) if midx else X_test
    per_seed = [mse_scaled_mlp(X_masked, s) for s in SEEDS]
    mse_mean, mse_std = aggregate_seed_mse(per_seed)
    results_rows.append({
        "model": "MLP", "k": k, "mask_type": mask_type,
        "masked_vars": ", ".join(mvars),
        "mse_scaled_mean": mse_mean, "mse_scaled_std": mse_std, "seed_count": len(SEEDS),
    })

# --- LSTM ---
print("  LSTM faithfulness (3 seeds)...")
for k, mask_type, mvars, midx in mask_configs(rankings["LSTM"]):
    X_masked = mask_variables_in_scaled_window(X_test, midx) if midx else X_test
    per_seed = [mse_scaled_lstm(X_masked, s) for s in SEEDS]
    mse_mean, mse_std = aggregate_seed_mse(per_seed)
    results_rows.append({
        "model": "LSTM", "k": k, "mask_type": mask_type,
        "masked_vars": ", ".join(mvars),
        "mse_scaled_mean": mse_mean, "mse_scaled_std": mse_std, "seed_count": len(SEEDS),
    })

# --- TFT --- (outer loop over seeds so checkpoint loaded once per seed)
print("  TFT faithfulness (3 seeds × 7 configs; slowest model)...")
tft_cfgs = mask_configs(rankings["TFT"])
tft_per_config_per_seed = {i: [] for i in range(len(tft_cfgs))}
for seed in SEEDS:
    print(f"    seed {seed}: load ckpt + 7 predicts...")
    tft = TemporalFusionTransformer.load_from_checkpoint(
        f"checkpoints/tft_seed{seed}.ckpt"
    )
    tft.eval()
    for i, (k, mask_type, mvars, midx) in enumerate(tft_cfgs):
        test_masked = test_scaled.copy()
        for idx in midx:
            test_masked[:, idx] = 0.0
        mse_i = mse_scaled_tft(test_masked, tft)
        tft_per_config_per_seed[i].append(mse_i)
        print(f"      k={k} {mask_type:<6} vars={mvars}  mse={mse_i:.4f}")
    del tft  # free memory

for i, (k, mask_type, mvars, midx) in enumerate(tft_cfgs):
    mse_mean, mse_std = aggregate_seed_mse(tft_per_config_per_seed[i])
    results_rows.append({
        "model": "TFT", "k": k, "mask_type": mask_type,
        "masked_vars": ", ".join(mvars),
        "mse_scaled_mean": mse_mean, "mse_scaled_std": mse_std, "seed_count": len(SEEDS),
    })


# ============================================================
# 6. Compute increase metrics + faithfulness verdicts + save CSV
# ============================================================
print("\n[6/6] Compute increases + faithfulness verdicts + save CSV...")

out_df = pd.DataFrame(results_rows)

# For each row, compute mse_increase vs its model's baseline
baseline_by_model = (
    out_df[out_df["k"] == 0]
    .set_index("model")["mse_scaled_mean"]
    .to_dict()
)
out_df["mse_increase"] = out_df.apply(
    lambda r: r["mse_scaled_mean"] - baseline_by_model[r["model"]], axis=1
)
out_df["mse_increase_pct"] = out_df.apply(
    lambda r: 100.0 * r["mse_increase"] / baseline_by_model[r["model"]]
    if baseline_by_model[r["model"]] > 0 else 0.0,
    axis=1,
)

os.makedirs("results", exist_ok=True)
csv_path = "results/faithfulness.csv"
out_df.to_csv(csv_path, index=False)
print(f"  saved {csv_path}  ({len(out_df)} rows)")


# ============================================================
# Summary + verification
# ============================================================
print("\n" + "=" * 72)
print("FAITHFULNESS SUMMARY (seed-averaged)")
print("=" * 72)
print(f"  {'Model':<6} {'k':>3}  {'Top-k MSE':>10}  {'Bot-k MSE':>10}  "
      f"{'Top-k%':>8}  {'Bot-k%':>8}  {'Faithful':>10}")
print("  " + "-" * 72)

faithful_count = 0
total_count = 0
for model in ["LR", "MLP", "LSTM", "TFT"]:
    for k in K_VALUES:
        t_rows = out_df[(out_df["model"] == model) & (out_df["k"] == k) &
                        (out_df["mask_type"] == "top")]
        b_rows = out_df[(out_df["model"] == model) & (out_df["k"] == k) &
                        (out_df["mask_type"] == "bottom")]
        if t_rows.empty or b_rows.empty:
            continue
        t, b = t_rows.iloc[0], b_rows.iloc[0]
        faithful = t["mse_increase"] > b["mse_increase"]
        faithful_count += int(faithful)
        total_count += 1
        print(f"  {model:<6} {k:>3}  {t['mse_scaled_mean']:>10.4f}  "
              f"{b['mse_scaled_mean']:>10.4f}  {t['mse_increase_pct']:>7.1f}%  "
              f"{b['mse_increase_pct']:>7.1f}%  {'YES' if faithful else 'NO':>10}")

print("  " + "-" * 72)
print(f"  Faithfulness score: {faithful_count}/{total_count} "
      f"({'faithful majority' if faithful_count >= total_count // 2 + 1 else 'not majority'})")

print("\nVERIFICATION CHECKLIST:")
print(f"  [{'x' if total_count == 12 else ' '}] 4 models × 3 k-values = 12 faithfulness tests")
print(f"  [{'x' if len(out_df) == 28 else ' '}] 4 models × (1 baseline + 6 masking) = 28 CSV rows")
print(f"  [{'x' if faithful_count > 0 else ' '}] At least one faithful result")
print(f"  [{'x' if os.path.exists(csv_path) else ' '}] CSV file created")
print("=" * 72)
print("\nNote: Faithfulness reported in scaled-space MSE. Scale does not affect")
print("ranking (top-k vs bottom-k comparison) but kept consistent across models.")
print("TFT uses VSN ranking (architecture-native); LR/MLP/LSTM use SHAP ranking")
print("(post-hoc). Method-family distinction is NOT collapsed: each model's")
print("ranking is tested against its own masked predictions (S-10b scope sentence).")
