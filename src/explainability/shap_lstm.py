"""
LSTM SHAP Analysis — thesis-valid, loads multi_seed.py checkpoints.
===================================================================

Purpose: Compute SHAP variable-importance for LSTM using the saved best-val-loss
         checkpoints from src/multi_seed.py. NOT a fresh-trained model —
         § 11A item 9 + SUSPECT-09 guard.

Fix 2026-04-19 (S-06 pre-run audit):
    Previous version fresh-trained an LSTM with lr=1e-3 for 50 fixed epochs,
    diverging from multi_seed.py's § 10 locked configuration (lr=1e-4,
    patience=10, val_loss monitored, restore-best). That characterized a
    different model from the one reported in multi_seed_fair_baseline.csv.
    This version loads `checkpoints/lstm_seed{42,123,456}.pt`.

LSTM specifics:
    - Input is 3D (batch, 96, 7) — NOT flattened
    - SHAP output on 3D input: (n_outputs, n_eval, 96, 7) after normalization
    - DeepExplainer may fail with LSTM (known issue) → GradientExplainer →
      manual gradient-attribution fallback

SHAP Protocol (CLAUDE.md § 10 locked; methodology § 2.9.0 baseline spec):
    - N_EVAL = 100, N_BG = 100, EVAL_SEED = 42, BG_SEED = 42
    - N_BG is held identical to shap_mlp.py per CLAUDE.md § 11A item 5
      (preprocessing symmetry). Runtime cost on RTX 5080: ~5–7 min per seed,
      ~20 min total for SEEDS = {42, 123, 456}.
    - Baseline: training-distribution mean in the scaled input space
      (equivalently the zero-vector by StandardScaler construction, § 4)
    - SEEDS = {42, 123, 456} — per-seed SHAP for S-09 stability analysis
    - Aggregation (§ 15): mean |SHAP| over output dims × eval samples × input
      timesteps → per-seed (7,); mean across seeds → final (7,)

Output:
    results/shap_lstm.csv       — long format, 28 rows (3 seeds × 7 + 7 mean)
    results/shap_lstm_cross.csv — seed-averaged cross-variable matrix (7, 7)
"""

import gc
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import shap

    print(f"SHAP version: {shap.__version__}")
except ImportError:
    print("ERROR: shap is not installed. Run: pip install shap")
    exit(1)


# ============================================================
# Constants (CLAUDE.md § 10 locked; methodology § 2.9.0 baseline)
# ============================================================
N_EVAL = 100
N_BG = 100                               # symmetric with shap_mlp.py (§ 11A item 5)
EVAL_SEED = 42
BG_SEED = 42                             # = EVAL_SEED; background uses the second
                                         # consecutive stream of the same RandomState
SHAP_BASELINE = "train_mean_scaled"      # training-distribution mean in scaled space
SEEDS = [42, 123, 456]

# LSTM-specific note: SHAP DeepExplainer does not support nn.LSTM (assertion
# failure — explanations do not sum to model output). The script therefore
# uses GradientExplainer, whose runtime scales linearly with N_BG. At
# N_BG = 100 the per-seed runtime is ~5–7 min on an RTX 5080 Laptop GPU,
# ~20 min total for three seeds. The previous N_BACKGROUND = 30 default
# was removed (2026-04-20) because the resulting MLP/LSTM asymmetry
# violated CLAUDE.md § 11A item 5 (preprocessing symmetry); runtime is
# accepted as the explicit cost of symmetry.


# ============================================================
# 0. Load + split + scale (identical pipeline to multi_seed.py)
# ============================================================
df = pd.read_csv("data/ETTh1.csv")
features = df.drop(columns=["date"])
var_names = features.columns.tolist()

n = len(features)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

train = features.iloc[:train_end]
test = features.iloc[val_end:]

scaler = StandardScaler()
scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)


def create_windows(data, input_len=96, output_len=24):
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len])
    return np.array(X), np.array(y)


X_train, y_train = create_windows(train_scaled)
X_test, y_test = create_windows(test_scaled)

n_train = X_train.shape[0]
n_test = X_test.shape[0]
y_train_flat = y_train.reshape(n_train, -1)
output_size = y_train_flat.shape[1]  # 168

print(f"Data: {n} rows, {len(var_names)} variables: {var_names}")
print(f"Train windows: {n_train}, Test windows: {n_test}")
print(f"LSTM input: (batch, 96, 7); output: {output_size} flat targets (24×7)")


# ============================================================
# 1. LSTM model definition — identical to multi_seed.py
# ============================================================
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
# 2. Shared eval + background subsets (3D tensors)
# ============================================================
rng = np.random.RandomState(EVAL_SEED)
eval_idx = rng.choice(n_test, size=N_EVAL, replace=False)
bg_idx = rng.choice(n_train, size=N_BG, replace=False)

X_eval = torch.FloatTensor(X_test[eval_idx])         # (100, 96, 7)
X_background = torch.FloatTensor(X_train[bg_idx])    # (100, 96, 7)

print(f"\nEvaluation subset: {N_EVAL} random test samples (EVAL_SEED={EVAL_SEED})")
print(f"Background subset: {N_BG} random training samples (BG_SEED={BG_SEED})")
print(f"SHAP baseline: {SHAP_BASELINE}")


# ============================================================
# SHAP helpers for LSTM (3D input)
# ============================================================
def run_shap_for_lstm(lstm_model, X_eval, X_background):
    """GradientExplainer (primary) → manual gradient-attribution (fallback).

    DeepExplainer is intentionally skipped: SHAP's DeepExplainer does not
    fully support nn.LSTM and consistently raises an assertion that the
    SHAP values do not sum to the model output (max diff ~0.9 vs tolerance
    0.01). The previous fast-fail attempt cost ~30s per seed without
    producing usable values; it is now removed (S-10d patch 2026-04-20).
    """
    try:
        t0 = time.time()
        print(f"    [SHAP] building GradientExplainer (bg={len(X_background)})...", flush=True)
        explainer = shap.GradientExplainer(lstm_model, X_background)
        print(f"    [SHAP] explainer built in {time.time()-t0:.1f}s; running shap_values on {len(X_eval)} eval samples...", flush=True)
        t1 = time.time()
        vals = explainer.shap_values(X_eval)
        print(f"    [SHAP] shap_values done in {time.time()-t1:.1f}s", flush=True)
        del explainer
        gc.collect()
        return vals, "GradientExplainer"
    except Exception as e:
        print(f"    GradientExplainer failed: {type(e).__name__}: {e}", flush=True)
    # Manual gradient-attribution fallback
    print("    Falling back to manual gradient attribution (not true SHAP).")
    lstm_model.eval()
    grads = []
    for i in range(N_EVAL):
        x_single = X_eval[i : i + 1].clone().requires_grad_(True)
        out = lstm_model(x_single).sum()
        out.backward()
        grads.append(x_single.grad.detach().numpy()[0])  # (96, 7)
        lstm_model.zero_grad()
    grad_array = np.stack(grads, axis=0)  # (N_EVAL, 96, 7)
    # Wrap as single-output SHAP-like shape (1, N_EVAL, 96, 7)
    return grad_array[np.newaxis, :, :, :], "GradientAttribution (fallback)"


def parse_shap_output_3d(shap_values, n_eval, n_outputs):
    """Normalize SHAP output to (n_outputs, n_eval, 96, 7) for LSTM 3D input."""
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=0)
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 4:
            if shap_values.shape[0] == n_eval and shap_values.shape[3] == n_outputs:
                return shap_values.transpose(3, 0, 1, 2)
            if shap_values.shape[0] == n_outputs:
                return shap_values
            return shap_values.transpose(3, 0, 1, 2)
        if shap_values.ndim == 3:
            # single output wrapped
            return shap_values[np.newaxis, :, :, :]
    vals = getattr(shap_values, "values", np.array(shap_values))
    if vals.ndim == 4:
        if vals.shape[0] == n_eval:
            return vals.transpose(3, 0, 1, 2)
        return vals
    if vals.ndim == 3:
        return vals[np.newaxis, :, :, :]
    raise ValueError(f"Unexpected SHAP output shape: {vals.shape}")


def cross_variable_from_gradient_fallback(lstm_model, X_eval, n_samples_cross=50):
    """For manual gradient fallback: compute per-output-variable gradient magnitude
    to populate a (7_out, 7_in) cross-variable matrix."""
    cross = np.zeros((7, 7))
    for out_var_idx in range(7):
        out_grads = []
        for i in range(min(N_EVAL, n_samples_cross)):
            x_single = X_eval[i : i + 1].clone().requires_grad_(True)
            output = lstm_model(x_single)  # (1, 168)
            # Select outputs for this variable: columns out_var_idx, +7, +14, ..., across 24 horizons
            out_for_var = output[0, out_var_idx::7].sum()
            out_for_var.backward()
            grad = x_single.grad.detach().numpy()[0]  # (96, 7)
            out_grads.append(np.mean(np.abs(grad), axis=0))  # (7,)
            lstm_model.zero_grad()
        cross[out_var_idx] = np.mean(out_grads, axis=0)
    return cross


# ============================================================
# 3. Per-seed SHAP (load best-val-loss checkpoint, compute importance)
# ============================================================
per_seed_importance = {}
per_seed_cross = {}
per_seed_mse_check = {}
explainer_used = None

for seed in SEEDS:
    print(f"\n--- LSTM seed {seed}: load checkpoint + SHAP ---")
    ckpt_path = f"checkpoints/lstm_seed{seed}.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} missing — run src/multi_seed.py first."
        )
    lstm_model = LSTMModel(input_size=7, hidden_size=64, output_size=output_size)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    lstm_model.load_state_dict(state)
    lstm_model.eval()

    # Sanity: test MSE check
    with torch.no_grad():
        y_pred_flat = lstm_model(torch.FloatTensor(X_test)).numpy()
    y_pred_3d = y_pred_flat.reshape(n_test, 24, 7)
    mse_scaled = float(np.mean((y_test - y_pred_3d) ** 2))
    per_seed_mse_check[seed] = mse_scaled
    print(f"  seed {seed} scaled MSE: {mse_scaled:.4f}")

    # SHAP
    shap_values, explainer_name = run_shap_for_lstm(lstm_model, X_eval, X_background)
    if explainer_used is None:
        explainer_used = explainer_name
    print(f"  explainer: {explainer_name}")

    # Parse to (n_outputs, n_eval, 96, 7)
    is_fallback = explainer_name == "GradientAttribution (fallback)"
    if is_fallback:
        # shap_values already in (1, N_EVAL, 96, 7) form
        shap_array = shap_values
    else:
        shap_array = parse_shap_output_3d(shap_values, N_EVAL, output_size)

    # (7,) per-variable importance: mean |SHAP| over outputs × samples × in-timesteps
    var_importance = np.mean(np.abs(shap_array), axis=(0, 1, 2))

    # (7, 7) cross-variable
    if is_fallback:
        # Manual gradient attribution per output-variable
        print("    computing per-output-variable gradients for cross matrix...")
        cross_importance = cross_variable_from_gradient_fallback(lstm_model, X_eval)
    else:
        # Reshape outputs 168 → (24, 7); average over out-t × samples × in-t
        shap_by_outvar = shap_array.reshape(24, 7, N_EVAL, 96, 7)
        cross_importance = np.mean(np.abs(shap_by_outvar), axis=(0, 2, 3))

    per_seed_importance[seed] = var_importance
    per_seed_cross[seed] = cross_importance
    print(
        "  per-seed importance: "
        + ", ".join(f"{v}={imp:.4f}" for v, imp in zip(var_names, var_importance))
    )

    del lstm_model, shap_values, shap_array, y_pred_flat, y_pred_3d
    gc.collect()


# ============================================================
# 4. Aggregate across seeds
# ============================================================
imp_stack = np.stack([per_seed_importance[s] for s in SEEDS], axis=0)  # (3, 7)
mean_importance = imp_stack.mean(axis=0)
std_importance = imp_stack.std(axis=0, ddof=1)

cross_stack = np.stack([per_seed_cross[s] for s in SEEDS], axis=0)  # (3, 7, 7)
mean_cross = cross_stack.mean(axis=0)


# ============================================================
# 5. 3-way SHAP comparison (LR / MLP / LSTM)
# ============================================================
lr_ranking, mlp_ranking = None, None

lr_path = "results/shap_lr.csv"
if os.path.exists(lr_path):
    lr_df = pd.read_csv(lr_path).sort_values("shap_rank")
    lr_ranking = lr_df["variable"].tolist()[:7]
    print(f"\nLR SHAP ranking: {lr_ranking}")

mlp_path = "results/shap_mlp.csv"
if os.path.exists(mlp_path):
    mlp_df = pd.read_csv(mlp_path)
    # Filter to mean-across-seeds rows (new format)
    if "aggregation" in mlp_df.columns:
        mlp_mean_rows = mlp_df[mlp_df["aggregation"] == "mean-across-seeds"]
        if len(mlp_mean_rows) == 7:
            mlp_mean_rows_sorted = mlp_mean_rows.sort_values(
                "shap_importance", ascending=False
            )
            mlp_ranking = mlp_mean_rows_sorted["variable"].tolist()
    else:
        # Legacy format fallback
        mlp_ranking = mlp_df.sort_values("shap_rank")["variable"].tolist()[:7]
    print(f"MLP SHAP ranking (mean across seeds): {mlp_ranking}")


# ============================================================
# 6. Report
# ============================================================
print("\n" + "=" * 72)
print("LSTM SHAP RESULTS — seed-averaged")
print("=" * 72)

mean_ranking = np.argsort(-mean_importance)
lstm_order = [var_names[i] for i in mean_ranking]

print(f"\nMean |SHAP| across seeds (explainer: {explainer_used}):")
print(f"  {'Rank':<5} {'Variable':<8} {'Mean':>10} {'Std':>10}")
print("  " + "-" * 38)
for rank, idx in enumerate(mean_ranking):
    print(
        f"  {rank + 1:<5} {var_names[idx]:<8} "
        f"{mean_importance[idx]:>10.4f} {std_importance[idx]:>10.4f}"
    )

if lr_ranking and mlp_ranking:
    print(f"\n3-way ranking comparison (position):")
    print(f"  {'Rank':<5} {'LR':<10} {'MLP':<10} {'LSTM':<10}")
    print("  " + "-" * 35)
    for rank in range(7):
        lr_v = lr_ranking[rank] if rank < len(lr_ranking) else "?"
        mlp_v = mlp_ranking[rank] if rank < len(mlp_ranking) else "?"
        lstm_v = lstm_order[rank]
        print(f"  {rank + 1:<5} {lr_v:<10} {mlp_v:<10} {lstm_v:<10}")
    lr_lstm = sum(1 for a, b in zip(lr_ranking, lstm_order) if a == b)
    mlp_lstm = sum(1 for a, b in zip(mlp_ranking, lstm_order) if a == b)
    print(f"\n  Position matches: LR vs LSTM = {lr_lstm}/7; MLP vs LSTM = {mlp_lstm}/7")

print("\nPer-seed importance matrix (variable × seed):")
print(f"  {'Variable':<8}", end="")
for s in SEEDS:
    print(f" {('seed' + str(s)):>10}", end="")
print()
print("  " + "-" * (8 + 11 * len(SEEDS)))
for i, v in enumerate(var_names):
    print(f"  {v:<8}", end="")
    for s in SEEDS:
        print(f" {per_seed_importance[s][i]:>10.4f}", end="")
    print()


# ============================================================
# 7. Save — long format
# ============================================================
os.makedirs("results", exist_ok=True)

rows = []
for seed in SEEDS:
    imp_vec = per_seed_importance[seed]
    for idx in range(len(var_names)):
        rows.append(
            {
                "model": "LSTM",
                "method": f"SHAP ({explainer_used})",
                "aggregation": "per-seed",
                "seed": seed,
                "variable": var_names[idx],
                "shap_importance": float(imp_vec[idx]),
                "n_eval": N_EVAL,
                "n_bg": N_BG,
                "eval_seed": EVAL_SEED,
                "bg_seed": BG_SEED,
                "shap_baseline": SHAP_BASELINE,
                "explainer": explainer_used,
                "mse_scaled_check": per_seed_mse_check[seed],
            }
        )
for idx in range(len(var_names)):
    rows.append(
        {
            "model": "LSTM",
            "method": f"SHAP ({explainer_used})",
            "aggregation": "mean-across-seeds",
            "seed": "mean",
            "variable": var_names[idx],
            "shap_importance": float(mean_importance[idx]),
            "n_eval": N_EVAL,
            "n_bg": N_BG,
            "eval_seed": EVAL_SEED,
            "bg_seed": BG_SEED,
            "shap_baseline": SHAP_BASELINE,
            "explainer": explainer_used,
            "mse_scaled_check": None,
        }
    )

csv_df = pd.DataFrame(rows)
csv_path = "results/shap_lstm.csv"
csv_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path} ({len(rows)} rows)")

cross_df = pd.DataFrame(
    mean_cross,
    index=[f"output_{v}" for v in var_names],
    columns=[f"input_{v}" for v in var_names],
)
cross_path = "results/shap_lstm_cross.csv"
cross_df.to_csv(cross_path)
print(f"Saved: {cross_path} (seed-averaged 7×7 cross-variable matrix)")

print("\nVERIFICATION CHECKLIST:")
print(f"  [x] 3 seeds loaded from checkpoints/lstm_seed{{42,123,456}}.pt")
print(f"  [x] SHAP run on best-val-loss checkpoints (SUSPECT-09 guard)")
print(f"  [x] § 11A item 9 validation/checkpoint selection symmetric with multi_seed.py")
print(f"  [x] Explainer: {explainer_used} (SUSPECT-08 documented in CSV)")
print(f"  [x] Per-seed + mean-across-seeds rows in CSV (S-09 stability-ready)")
