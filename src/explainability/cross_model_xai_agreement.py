"""Cross-model XAI agreement (S-10 strong-package addition).

Purpose
-------
Quantifies how much the four core models agree on which variables are most
important, by computing Spearman ρ and Kendall τ rank correlations between
their per-variable importance vectors. Six pairwise comparisons total.

Why this metric
---------------
RQ2 ("how stable and faithful are model explanations across the compared
models?") asks not only about per-model faithfulness (covered by
faithfulness_test.py) but about the *agreement* of explanations across
models. A high cross-model rank correlation means the models concur on
which variables matter; a low correlation means the models disagree, which
itself is a finding worth reporting.

Construct-validity caveat (CLAUDE.md § 13)
------------------------------------------
SHAP (LR / MLP / LSTM) is post-hoc; VSN (TFT) is architecture-native. They
are not equivalent measurement instruments. Rank correlations between
SHAP-based and VSN-based vectors are still computable (rank is a coarse
operation), but the absolute correlation values must be read with this
distinction in mind. The output CSV flags TFT pairs explicitly.

Inputs (frozen artefacts produced upstream)
-------------------------------------------
    results/shap_lr.csv             — per-variable SHAP importance for LR
    results/shap_mlp.csv            — per-variable SHAP importance for MLP (seed-averaged)
    results/shap_lstm.csv           — per-variable SHAP importance for LSTM (seed-averaged)
    results/tft_importance.csv      — per-variable VSN importance for TFT (seed-averaged)

Output
------
    results/xai_agreement.csv with columns:
        model_pair, spearman_corr, kendall_tau, n_features, includes_tft

Run
---
    python src/explainability/cross_model_xai_agreement.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


# ============================================================
# Constants
# ============================================================
ETTH1_VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
MODELS = ["LR", "MLP", "LSTM", "TFT"]


# ============================================================
# Per-model importance loaders (each returns a dict {variable: importance})
# ============================================================
def load_lr_importance():
    df = pd.read_csv("results/shap_lr.csv")
    return dict(zip(df["variable"], df["shap_importance"]))


def load_mlp_importance():
    df = pd.read_csv("results/shap_mlp.csv")
    if "seed" in df.columns:
        df = df.groupby("variable")["shap_importance"].mean().reset_index()
    return dict(zip(df["variable"], df["shap_importance"]))


def load_lstm_importance():
    df = pd.read_csv("results/shap_lstm.csv")
    if "seed" in df.columns:
        df = df.groupby("variable")["shap_importance"].mean().reset_index()
    return dict(zip(df["variable"], df["shap_importance"]))


def load_tft_importance():
    df = pd.read_csv("results/tft_importance.csv")
    if "seed" in df.columns:
        df = df.groupby("variable")["importance"].mean().reset_index()
    df = df[df["variable"].isin(ETTH1_VARS)]
    return dict(zip(df["variable"], df["importance"]))


LOADERS = {
    "LR": load_lr_importance,
    "MLP": load_mlp_importance,
    "LSTM": load_lstm_importance,
    "TFT": load_tft_importance,
}


# ============================================================
# Build per-model importance vectors aligned to ETTH1_VARS order
# ============================================================
def build_aligned_vectors():
    vectors = {}
    for model, loader in LOADERS.items():
        try:
            imp_dict = loader()
        except FileNotFoundError as e:
            print(f"  [skip] {model}: {e}")
            continue
        missing = [v for v in ETTH1_VARS if v not in imp_dict]
        if missing:
            print(f"  [warn] {model} missing variables: {missing} — skipping model")
            continue
        vectors[model] = np.array([imp_dict[v] for v in ETTH1_VARS], dtype=float)
        print(f"  [ok]   {model}: {vectors[model]}")
    return vectors


# ============================================================
# Pairwise rank-correlation matrix
# ============================================================
def compute_pair_correlations(vectors):
    rows = []
    models_present = list(vectors.keys())
    for i in range(len(models_present)):
        for j in range(i + 1, len(models_present)):
            a, b = models_present[i], models_present[j]
            sp_corr, _ = spearmanr(vectors[a], vectors[b])
            kt_corr, _ = kendalltau(vectors[a], vectors[b])
            rows.append({
                "model_pair": f"{a}_vs_{b}",
                "spearman_corr": round(float(sp_corr), 6),
                "kendall_tau": round(float(kt_corr), 6),
                "n_features": len(ETTH1_VARS),
                "includes_tft": (a == "TFT" or b == "TFT"),
            })
    return pd.DataFrame(rows)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Cross-model XAI agreement (Spearman + Kendall on importance vectors)")
    print("=" * 70)

    print("\n[1/3] Loading per-model importance vectors...")
    vectors = build_aligned_vectors()

    if len(vectors) < 2:
        print("\n[error] fewer than 2 models loaded — cannot compute pairwise agreement")
        raise SystemExit(1)

    print(f"\n[2/3] Computing {len(vectors) * (len(vectors) - 1) // 2} pairwise correlations...")
    out_df = compute_pair_correlations(vectors)
    print(out_df.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    out_path = "results/xai_agreement.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[3/3] Saved {out_path} ({len(out_df)} pairs)")

    print("\n" + "-" * 70)
    print("Reading guide:")
    print("  Spearman / Kendall in [-1, +1]; +1 = identical ranking, 0 = no association.")
    print("  Pairs with includes_tft=True compare SHAP (post-hoc) vs VSN (architecture-")
    print("  native) — interpret rank agreement, not absolute value (CLAUDE.md § 13).")
    print("-" * 70)
