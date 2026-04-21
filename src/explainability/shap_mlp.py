"""
MLP SHAP Analysis — thesis-valid, loads multi_seed.py checkpoints.
================================================================

Purpose: Compute SHAP variable-importance for MLP using the saved best-val-loss
         checkpoints from src/multi_seed.py (fair-core thesis-valid run). NOT
         a fresh-trained model — § 11A item 9 (validation/checkpoint selection
         symmetry) + SUSPECT-09 guard (SHAP on best checkpoint, not re-train).

Fix 2026-04-19 (S-06 pre-run audit):
    Previous version fresh-trained an MLP with lr=1e-3 for 50 fixed epochs,
    which did NOT match multi_seed.py's § 10 locked configuration (lr=1e-4,
    patience=10, val_loss monitored, restore-best). That characterized a
    different model from the one reported in multi_seed_fair_baseline.csv.
    This version loads `checkpoints/mlp_seed{42,123,456}.pt` directly.

SHAP Protocol (CLAUDE.md § 10 locked; methodology § 2.9.0 baseline spec):
    - N_EVAL = 100 random test windows (shared across all seeds, paired)
    - N_BG = 100 random training samples (shared across all seeds; symmetric
      with shap_lstm.py per CLAUDE.md § 11A item 5 preprocessing symmetry)
    - EVAL_SEED = 42, BG_SEED = 42 (numerically identical; the background
      subset is drawn as the second consecutive stream of the EVAL_SEED-seeded
      NumPy RandomState, preserving the locked baseline run)
    - Baseline: training-distribution mean in the scaled input space
      (equivalently the zero-vector by StandardScaler construction, § 4)
    - Per-seed SHAP for seeds {42, 123, 456} — enables S-09 stability analysis
    - Aggregation (§ 15): mean |SHAP| over output dims × eval samples × input
      timesteps → per-seed (7,); mean across seeds → final (7,)

Output:
    results/shap_mlp.csv       — long format, 4 × 7 = 28 rows (3 seeds + mean)
    results/shap_mlp_cross.csv — seed-averaged cross-variable matrix (7, 7)
"""

import os
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
N_BG = 100                               # shared with shap_lstm.py (§ 11A item 5)
EVAL_SEED = 42
BG_SEED = 42                             # = EVAL_SEED; background uses the second
                                         # consecutive stream of the same RandomState
SHAP_BASELINE = "train_mean_scaled"      # training-distribution mean in scaled space
SEEDS = [42, 123, 456]


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
X_train_flat = X_train.reshape(n_train, -1)
X_test_flat = X_test.reshape(n_test, -1)
y_train_flat = y_train.reshape(n_train, -1)

input_size = X_train_flat.shape[1]   # 672
output_size = y_train_flat.shape[1]  # 168

print(f"Data: {n} rows, {len(var_names)} variables: {var_names}")
print(f"Train windows: {n_train}, Test windows: {n_test}")


# ============================================================
# 1. MLP model definition — identical to multi_seed.py
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


# ============================================================
# 2. Shared eval + background subsets (same across all seeds for paired analysis)
# ============================================================
rng = np.random.RandomState(EVAL_SEED)
eval_idx = rng.choice(n_test, size=N_EVAL, replace=False)
bg_idx = rng.choice(n_train, size=N_BG, replace=False)

X_eval = torch.FloatTensor(X_test_flat[eval_idx])
X_background = torch.FloatTensor(X_train_flat[bg_idx])

print(f"\nEvaluation subset: {N_EVAL} random test samples (EVAL_SEED={EVAL_SEED})")
print(f"Background subset: {N_BG} random training samples (BG_SEED={BG_SEED})")
print(f"SHAP baseline: {SHAP_BASELINE}")


# ============================================================
# SHAP output parsing helper
# ============================================================
def parse_shap_output(shap_values, n_eval, n_outputs, n_features):
    """Normalize SHAP output to (n_outputs, n_eval, n_features)."""
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=0)
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            if shap_values.shape[0] == n_eval and shap_values.shape[2] == n_outputs:
                return shap_values.transpose(2, 0, 1)
            if shap_values.shape[0] == n_outputs:
                return shap_values
            return shap_values.transpose(2, 0, 1)
        if shap_values.ndim == 2:
            return shap_values[np.newaxis, :, :]
    # shap.Explanation
    vals = getattr(shap_values, "values", np.array(shap_values))
    if vals.ndim == 3:
        if vals.shape[0] == n_eval:
            return vals.transpose(2, 0, 1)
        return vals
    if vals.ndim == 2:
        return vals[np.newaxis, :, :]
    raise ValueError(f"Unexpected SHAP output shape: {vals.shape}")


def run_shap_for_mlp(mlp_model, X_eval, X_background, bg_idx_np):
    """Try DeepExplainer → GradientExplainer → KernelExplainer fallback chain."""
    try:
        explainer = shap.DeepExplainer(mlp_model, X_background)
        return explainer.shap_values(X_eval), "DeepExplainer"
    except Exception as e:
        print(f"    DeepExplainer failed: {e}")
    try:
        explainer = shap.GradientExplainer(mlp_model, X_background)
        return explainer.shap_values(X_eval), "GradientExplainer"
    except Exception as e:
        print(f"    GradientExplainer failed: {e}")

    def mlp_predict(x):
        with torch.no_grad():
            return mlp_model(torch.FloatTensor(x)).numpy()

    explainer = shap.KernelExplainer(mlp_predict, X_train_flat[bg_idx_np])
    return explainer.shap_values(X_eval.numpy()), "KernelExplainer"


# ============================================================
# 3. Per-seed SHAP (load best-val-loss checkpoint, compute importance)
# ============================================================
per_seed_importance = {}    # seed → (7,) array
per_seed_cross = {}         # seed → (7, 7) array
per_seed_mse_check = {}     # seed → scalar (sanity check vs multi_seed.py)
explainer_used = None

for seed in SEEDS:
    print(f"\n--- MLP seed {seed}: load checkpoint + SHAP ---")
    ckpt_path = f"checkpoints/mlp_seed{seed}.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} missing — run src/multi_seed.py first to produce the "
            f"thesis-valid MLP checkpoints."
        )
    mlp_model = MLP(input_size, output_size)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mlp_model.load_state_dict(state)
    mlp_model.eval()

    # Sanity: test MSE should match multi_seed.py seed={seed} record
    with torch.no_grad():
        y_pred_flat = mlp_model(torch.FloatTensor(X_test_flat)).numpy()
    y_pred_3d = y_pred_flat.reshape(n_test, 24, 7)
    mse_scaled = float(np.mean((y_test - y_pred_3d) ** 2))
    per_seed_mse_check[seed] = mse_scaled
    print(f"  seed {seed} scaled MSE: {mse_scaled:.4f}")

    # SHAP
    shap_values, explainer_name = run_shap_for_mlp(
        mlp_model, X_eval, X_background, bg_idx
    )
    if explainer_used is None:
        explainer_used = explainer_name
    print(f"  explainer: {explainer_name}")

    # Parse + reshape
    shap_array = parse_shap_output(shap_values, N_EVAL, output_size, input_size)
    shap_reshaped = shap_array.reshape(output_size, N_EVAL, 96, 7)

    # (7,) per-variable importance: mean |SHAP| over outputs × samples × in-timesteps
    var_importance = np.mean(np.abs(shap_reshaped), axis=(0, 1, 2))
    # (7, 7) cross-variable: which input var → which output var (mean over out-t × samples × in-t)
    shap_by_outvar = shap_reshaped.reshape(24, 7, N_EVAL, 96, 7)
    cross_importance = np.mean(np.abs(shap_by_outvar), axis=(0, 2, 3))

    per_seed_importance[seed] = var_importance
    per_seed_cross[seed] = cross_importance
    print(
        "  per-seed importance: "
        + ", ".join(f"{v}={imp:.4f}" for v, imp in zip(var_names, var_importance))
    )


# ============================================================
# 4. Aggregate across seeds
# ============================================================
imp_stack = np.stack([per_seed_importance[s] for s in SEEDS], axis=0)  # (3, 7)
mean_importance = imp_stack.mean(axis=0)
std_importance = imp_stack.std(axis=0, ddof=1)

cross_stack = np.stack([per_seed_cross[s] for s in SEEDS], axis=0)  # (3, 7, 7)
mean_cross = cross_stack.mean(axis=0)


# ============================================================
# 5. LR comparison (supplementary)
# ============================================================
lr_shap_path = "results/shap_lr.csv"
if os.path.exists(lr_shap_path):
    lr_df = pd.read_csv(lr_shap_path)
    lr_ranking = lr_df.sort_values("shap_rank")["variable"].tolist()
    lr_importance_map = dict(zip(lr_df["variable"], lr_df["shap_importance"]))
    print(f"\nLR SHAP ranking (for comparison): {lr_ranking}")
else:
    lr_ranking = None
    lr_importance_map = {}
    print("\nWARNING: results/shap_lr.csv not found. LR comparison skipped.")


# ============================================================
# 6. Report
# ============================================================
print("\n" + "=" * 72)
print("MLP SHAP RESULTS — seed-averaged")
print("=" * 72)

mean_ranking = np.argsort(-mean_importance)
print(f"\nMean |SHAP| across seeds (explainer: {explainer_used}):")
print(f"  {'Rank':<5} {'Variable':<8} {'Mean':>10} {'Std':>10}")
print("  " + "-" * 38)
for rank, idx in enumerate(mean_ranking):
    print(
        f"  {rank + 1:<5} {var_names[idx]:<8} "
        f"{mean_importance[idx]:>10.4f} {std_importance[idx]:>10.4f}"
    )

if lr_ranking is not None:
    mlp_order = [var_names[i] for i in mean_ranking]
    matches = sum(1 for a, b in zip(mlp_order, lr_ranking) if a == b)
    print(f"\nMLP vs LR ranking position-matches: {matches}/7")

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
# 7. Save — long format (seed-level + mean row)
# ============================================================
os.makedirs("results", exist_ok=True)

rows = []
# Per-seed rows
for seed in SEEDS:
    imp_vec = per_seed_importance[seed]
    for idx in range(len(var_names)):
        rows.append(
            {
                "model": "MLP",
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
# Mean-across-seeds rows
for idx in range(len(var_names)):
    rows.append(
        {
            "model": "MLP",
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
csv_path = "results/shap_mlp.csv"
csv_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path} ({len(rows)} rows — 3 seeds × 7 + 7 mean)")

# Cross-variable matrix (seed-averaged)
cross_df = pd.DataFrame(
    mean_cross,
    index=[f"output_{v}" for v in var_names],
    columns=[f"input_{v}" for v in var_names],
)
cross_path = "results/shap_mlp_cross.csv"
cross_df.to_csv(cross_path)
print(f"Saved: {cross_path} (seed-averaged 7×7 cross-variable matrix)")

print("\nVERIFICATION CHECKLIST:")
print("  [x] 3 seeds loaded from checkpoints/mlp_seed{42,123,456}.pt")
print("  [x] SHAP run on best-val-loss checkpoints (SUSPECT-09 guard)")
print("  [x] § 11A item 9 validation/checkpoint selection symmetric with multi_seed.py")
print(f"  [x] Explainer: {explainer_used} (SUSPECT-08 documented in CSV `explainer` column)")
print("  [x] Per-seed + mean-across-seeds rows in CSV (S-09 stability analysis ready)")
