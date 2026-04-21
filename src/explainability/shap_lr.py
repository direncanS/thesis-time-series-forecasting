"""
LR SHAP Analysis (S3a)
======================
Purpose: Compute SHAP variable-importance attributions for Linear Regression.
         First SHAP result in the thesis pipeline.

Prerequisites:
    pip install shap

SHAP Protocol (CLAUDE.md § 10 locked; see methodology § 2.9.0 for baseline spec):
    - N_EVAL = 100 random test windows, EVAL_SEED = 42
    - Baseline: training-distribution mean in the scaled input space
      (equivalently the zero-vector by StandardScaler construction, § 4)
    - Background: full training set via LinearExplainer (closed-form; N_BG
      and BG_SEED are not applicable for this explainer class)
    - Aggregation: mean |SHAP| across eval samples, input timesteps, output horizons -> (7,)
    - Supplementary check: compare SHAP ranking vs LR coefficient ranking

Expected output:
    - Per-variable SHAP importance ranking (7 variables)
    - Per-variable LR coefficient importance ranking (supplementary)
    - Cross-variable importance matrix (which input var -> which output var)
    - CSV saved to results/shap_lr.csv

Verification:
    - SHAP ranking should be plausible (not random)
    - For linear models, SHAP and coefficient rankings should be similar
      (exact match not required due to correlation effects)
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Check SHAP availability
try:
    import shap
    print(f"SHAP version: {shap.__version__}")
except ImportError:
    print("ERROR: shap is not installed.")
    print("Run: pip install shap")
    exit(1)

# ============================================================
# SHAP Protocol Parameters (CLAUDE.md § 10 locked;
# methodology § 2.9.0 for baseline specification)
# ============================================================
N_EVAL = 100                              # number of test samples for SHAP evaluation
EVAL_SEED = 42                            # seed for reproducible subset selection
N_BG = "full_X_train_flat"                # LinearExplainer uses full training set (closed-form)
BG_SEED = None                            # not applicable — deterministic background
SHAP_BASELINE = "train_mean_scaled"       # training-distribution mean in scaled input space

# ============================================================
# 0. Load and preprocess (same pipeline as multi_seed.py)
# ============================================================
df = pd.read_csv("data/ETTh1.csv")
features = df.drop(columns=["date"])
var_names = features.columns.tolist()

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

print(f"Data: {n} rows, {len(var_names)} variables: {var_names}")
print(f"Train windows: {n_train}, Test windows: {n_test}")
print(f"Flattened input: {X_train_flat.shape[1]} features (96 timesteps x 7 variables)")
print(f"Flattened output: {y_train_flat.shape[1]} targets (24 timesteps x 7 variables)")

# ============================================================
# 1. Train Linear Regression (same as multi_seed.py)
# ============================================================
print("\n--- Training Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train_flat, y_train_flat)
print(f"LR coefficients shape: {lr_model.coef_.shape}")  # (168, 672)
print(f"LR intercept shape: {lr_model.intercept_.shape}")  # (168,)

# ============================================================
# 2. Select evaluation subset
# ============================================================
rng = np.random.RandomState(EVAL_SEED)
eval_idx = rng.choice(n_test, size=N_EVAL, replace=False)
X_eval = X_test_flat[eval_idx]
print(f"\nEvaluation subset: {N_EVAL} random test samples (seed={EVAL_SEED})")
print(f"X_eval shape: {X_eval.shape}")

# ============================================================
# 3. Compute SHAP values using LinearExplainer
# ============================================================
print("\n--- Computing SHAP values (LinearExplainer) ---")
print("This may take a minute...")

explainer = shap.LinearExplainer(lr_model, X_train_flat)
shap_values = explainer.shap_values(X_eval)

# LinearExplainer with multi-output returns:
# - list of n_outputs arrays, each (n_eval, n_features)
# - OR array of shape (n_eval, n_features) for single output
if isinstance(shap_values, list):
    print(f"SHAP returned list of {len(shap_values)} arrays")
    print(f"Each array shape: {shap_values[0].shape}")
    # Stack to (n_outputs, n_eval, n_features) = (168, 100, 672)
    shap_array = np.stack(shap_values, axis=0)
elif isinstance(shap_values, np.ndarray):
    if shap_values.ndim == 3:
        # (n_eval, n_features, n_outputs) -> (n_outputs, n_eval, n_features)
        print(f"SHAP returned array shape: {shap_values.shape}")
        shap_array = shap_values.transpose(2, 0, 1)
    elif shap_values.ndim == 2:
        # Single output case (shouldn't happen for multi-output LR)
        print(f"SHAP returned 2D array shape: {shap_values.shape}")
        shap_array = shap_values[np.newaxis, :, :]
    else:
        print(f"Unexpected SHAP shape: {shap_values.shape}")
        exit(1)
else:
    # shap.Explanation object (newer API)
    print("SHAP returned Explanation object")
    vals = shap_values.values
    print(f"Values shape: {vals.shape}")
    if vals.ndim == 3:
        shap_array = vals.transpose(2, 0, 1)
    else:
        shap_array = vals[np.newaxis, :, :]

print(f"SHAP array shape: {shap_array.shape}")
# Expected: (168, 100, 672) = (outputs, eval_samples, features)

n_outputs = shap_array.shape[0]
n_features = shap_array.shape[2]

# ============================================================
# 4. Reshape and aggregate SHAP values
# ============================================================
print("\n--- Aggregating SHAP values ---")

# Reshape features: 672 -> (96 timesteps, 7 variables)
# shap_array: (168, 100, 672) -> (168, 100, 96, 7)
shap_reshaped = shap_array.reshape(n_outputs, N_EVAL, 96, 7)

# Primary aggregation: mean |SHAP| across outputs, samples, timesteps -> (7,)
var_importance = np.mean(np.abs(shap_reshaped), axis=(0, 1, 2))

# Per-output-variable view: which input variable matters for which output variable?
# Reshape outputs: 168 -> (24 timesteps, 7 variables)
# (168, 100, 96, 7) -> (24, 7, 100, 96, 7) = (out_t, out_var, samples, in_t, in_var)
shap_by_outvar = shap_reshaped.reshape(24, 7, N_EVAL, 96, 7)
# Mean |SHAP| across output timesteps, samples, input timesteps -> (7_out, 7_in)
cross_var_importance = np.mean(np.abs(shap_by_outvar), axis=(0, 2, 3))

# ============================================================
# 5. LR Coefficient comparison (supplementary)
# ============================================================
print("\n--- LR Coefficient analysis (supplementary) ---")

coef = lr_model.coef_  # (168, 672)
coef_reshaped = coef.reshape(168, 96, 7)
# Per-variable importance from coefficients: mean |coef| across outputs, timesteps -> (7,)
coef_var_importance = np.mean(np.abs(coef_reshaped), axis=(0, 1))

# ============================================================
# 6. Results
# ============================================================
print("\n" + "=" * 70)
print("LR SHAP RESULTS")
print("=" * 70)

# SHAP ranking
shap_ranking = np.argsort(-var_importance)
print("\nSHAP Variable Importance (mean |SHAP|, aggregated across all dimensions):")
print(f"  {'Rank':<6} {'Variable':<10} {'Importance':>12}")
print("  " + "-" * 28)
for rank, idx in enumerate(shap_ranking):
    print(f"  {rank+1:<6} {var_names[idx]:<10} {var_importance[idx]:>12.6f}")

# Coefficient ranking
coef_ranking = np.argsort(-coef_var_importance)
print("\nLR Coefficient Importance (mean |coef|, supplementary):")
print(f"  {'Rank':<6} {'Variable':<10} {'Importance':>12}")
print("  " + "-" * 28)
for rank, idx in enumerate(coef_ranking):
    print(f"  {rank+1:<6} {var_names[idx]:<10} {coef_var_importance[idx]:>12.6f}")

# Ranking comparison
shap_order = [var_names[i] for i in shap_ranking]
coef_order = [var_names[i] for i in coef_ranking]
print(f"\nSHAP ranking:  {shap_order}")
print(f"Coef ranking:  {coef_order}")
matches = sum(1 for a, b in zip(shap_order, coef_order) if a == b)
print(f"Position matches: {matches}/7")

# Cross-variable importance matrix
print("\nCross-Variable Importance (input -> output):")
print(f"  {'':>10}", end="")
for v in var_names:
    print(f" {v:>8}", end="")
print()
print("  " + "-" * (10 + 9 * 7))
for i, out_var in enumerate(var_names):
    print(f"  {out_var + ' <-':>10}", end="")
    for j in range(7):
        print(f" {cross_var_importance[i, j]:>8.4f}", end="")
    print()

# ============================================================
# 7. Save results
# ============================================================
os.makedirs("results", exist_ok=True)

# Main importance table
rows = []
for idx in shap_ranking:
    rows.append({
        "model": "Linear Regression",
        "method": "SHAP (LinearExplainer)",
        "variable": var_names[idx],
        "shap_importance": var_importance[idx],
        "coef_importance": coef_var_importance[idx],
        "shap_rank": list(shap_ranking).index(idx) + 1,
        "coef_rank": list(coef_ranking).index(idx) + 1,
        "n_eval": N_EVAL,
        "eval_seed": EVAL_SEED,
        "n_bg": N_BG,
        "bg_seed": BG_SEED,
        "shap_baseline": SHAP_BASELINE,
    })

csv_df = pd.DataFrame(rows)
csv_path = "results/shap_lr.csv"
csv_df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")

# Cross-variable importance matrix
cross_df = pd.DataFrame(
    cross_var_importance,
    index=[f"output_{v}" for v in var_names],
    columns=[f"input_{v}" for v in var_names]
)
cross_path = "results/shap_lr_cross.csv"
cross_df.to_csv(cross_path)
print(f"Cross-variable matrix saved to: {cross_path}")

print(f"\n{'=' * 70}")
print("VERIFICATION CHECKLIST:")
print(f"  [{'x' if len(shap_ranking) == 7 else ' '}] 7 variables ranked")
print(f"  [{'x' if var_importance.sum() > 0 else ' '}] SHAP values are non-zero")
print(f"  [{'x' if matches >= 3 else ' '}] SHAP and coef rankings partially agree ({matches}/7)")
print(f"  [{'x' if os.path.exists(csv_path) else ' '}] CSV file created")
print(f"{'=' * 70}")
