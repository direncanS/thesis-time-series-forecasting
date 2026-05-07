"""Cross-model XAI agreement.

Purpose (RQ2 closure target)
----------------------------
Quantifies how much the four core models agree on which variables are most
important. Primary cross-model agreement is computed on the common
*occlusion-importance* instrument produced by
``src/explainability/common_importance.py`` (B7a), because occlusion is the
only importance measure that is the same measurement object across all four
models.

Auxiliary outputs retain the SHAP-family and VSN rank correlations so the
thesis can discuss the method-family distinction (SHAP post-hoc for
LR/MLP/LSTM vs VSN architecture-native for TFT) — but the
headline cross-model Spearman / Kendall results read off the occlusion
primary.

Method
------
For each model, aggregate per-(seed, variable) occlusion importance into a
single per-variable score by taking the mean across seeds (LR is
deterministic, so its single value passes through). Then for each of the
six model pairs, compute Spearman ρ and Kendall τ on the 7 ETTh1 variables.

Inputs
------
    results/bachelor_safe_v2/occlusion_importance.csv          — B7a primary
    results/bachelor_safe_v2/shap_{lr,mlp,lstm}.csv (optional) — auxiliary
    results/bachelor_safe_v2/tft_importance.csv     (optional) — auxiliary

Outputs
-------
    results/bachelor_safe_v2/xai_agreement_occlusion.csv   — PRIMARY
        columns: model_pair, spearman_corr, kendall_tau, n_features, includes_tft

    results/bachelor_safe_v2/xai_agreement_shap_vsn.csv    — AUXILIARY (optional)
        same schema; present only if all four auxiliary sources exist.

Run
---
    python src/explainability/cross_model_xai_agreement.py \\
        --config configs/experiments/fair_core_v2.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from src.common import apply_runtime_overrides, ensure_dir, load_config

ETTH1_VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
MODELS_ORDERED = ("LR", "MLP", "LSTM", "TFT")


# ---------------------------------------------------------------------------
# Primary: occlusion
# ---------------------------------------------------------------------------


def build_occlusion_vectors(occlusion_csv: Path) -> dict[str, np.ndarray]:
    """Return {model: importance_vector aligned to ETTH1_VARS} from occlusion CSV."""
    df = pd.read_csv(occlusion_csv)
    vectors: dict[str, np.ndarray] = {}
    for model in MODELS_ORDERED:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        per_var = sub.groupby("variable")["importance"].mean()
        missing = [v for v in ETTH1_VARS if v not in per_var.index]
        if missing:
            print(f"  [warn] occlusion CSV missing variables for {model}: {missing}")
            continue
        vectors[model] = np.array([per_var[v] for v in ETTH1_VARS], dtype=float)
    return vectors


# ---------------------------------------------------------------------------
# Auxiliary: SHAP + VSN (legacy path)
# ---------------------------------------------------------------------------


def build_shap_vsn_vectors(results_path: Path) -> dict[str, np.ndarray]:
    """Return {model: per-variable auxiliary importance} from SHAP and VSN CSVs.

    All four CSVs must be present for the auxiliary table to be emitted.
    """
    paths = {
        "LR": results_path / "shap_lr.csv",
        "MLP": results_path / "shap_mlp.csv",
        "LSTM": results_path / "shap_lstm.csv",
        "TFT": results_path / "tft_importance.csv",
    }
    if not all(p.exists() for p in paths.values()):
        return {}

    def _load(path: Path, value_col: str) -> dict[str, float]:
        d = pd.read_csv(path)
        col = value_col if value_col in d.columns else d.columns[-1]
        if "seed" in d.columns:
            d = d.groupby("variable")[col].mean().reset_index()
        d = d[d["variable"].isin(ETTH1_VARS)]
        return dict(zip(d["variable"], d[col], strict=False))

    vectors: dict[str, np.ndarray] = {}
    for model, path in paths.items():
        value_col = "importance" if model == "TFT" else "shap_importance"
        imp = _load(path, value_col)
        missing = [v for v in ETTH1_VARS if v not in imp]
        if missing:
            print(f"  [warn] auxiliary {model} missing vars: {missing}")
            continue
        vectors[model] = np.array([imp[v] for v in ETTH1_VARS], dtype=float)
    return vectors


# ---------------------------------------------------------------------------
# Pairwise rank-correlation matrix
# ---------------------------------------------------------------------------


def pair_correlations(vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    models_present = [m for m in MODELS_ORDERED if m in vectors]
    for i in range(len(models_present)):
        for j in range(i + 1, len(models_present)):
            a, b = models_present[i], models_present[j]
            sp, _ = spearmanr(vectors[a], vectors[b])
            kt, _ = kendalltau(vectors[a], vectors[b])
            rows.append(
                {
                    "model_pair": f"{a}_vs_{b}",
                    "spearman_corr": round(float(sp), 6),
                    "kendall_tau": round(float(kt), 6),
                    "n_features": len(ETTH1_VARS),
                    "includes_tft": (a == "TFT" or b == "TFT"),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_agreement(
    config_path: str | None = None,
    *,
    results_dir: str | None = None,
) -> pd.DataFrame:
    cfg = apply_runtime_overrides(load_config(config_path), smoke=False, results_dir=results_dir)
    results_path = ensure_dir(cfg["results_dir"])

    occlusion_csv = results_path / "occlusion_importance.csv"
    if not occlusion_csv.exists():
        raise FileNotFoundError(
            f"missing {occlusion_csv} — run src/explainability/common_importance.py first"
        )

    print(f"[B7] results_dir={results_path}")
    print("\n[PRIMARY] Occlusion-based cross-model agreement")
    occlusion_vectors = build_occlusion_vectors(occlusion_csv)
    for m, v in occlusion_vectors.items():
        print(f"  {m:<5}: {np.round(v, 5)}")
    primary_df = pair_correlations(occlusion_vectors)
    if not primary_df.empty:
        print(primary_df.to_string(index=False))
        primary_csv = results_path / "xai_agreement_occlusion.csv"
        primary_df.to_csv(primary_csv, index=False)
        print(f"  saved {primary_csv} ({len(primary_df)} pairs)")

    print("\n[AUXILIARY] SHAP-family + SHAP↔VSN legacy agreement")
    aux_vectors = build_shap_vsn_vectors(results_path)
    if aux_vectors:
        for m, v in aux_vectors.items():
            print(f"  {m:<5}: {np.round(v, 5)}")
        aux_df = pair_correlations(aux_vectors)
        aux_csv = results_path / "xai_agreement_shap_vsn.csv"
        aux_df.to_csv(aux_csv, index=False)
        print(f"  saved {aux_csv} ({len(aux_df)} pairs)")
    else:
        print("  [note] auxiliary SHAP / VSN CSVs incomplete or absent — legacy output skipped.")

    print("\n[read guide]")
    print("  Primary (occlusion) cross-model agreement is the RQ2 headline — same")
    print("  instrument for all 4 models. Auxiliary SHAP-family / SHAP↔VSN")
    print("  correlations remain available for the construct-validity discussion")
    print("  (SHAP post-hoc vs VSN architecture-native).")

    return primary_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="B7 — cross-model XAI agreement (occlusion primary + SHAP/VSN auxiliary)."
    )
    parser.add_argument("--config", default="configs/experiments/fair_core_v2.yaml")
    parser.add_argument("--results-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_agreement(args.config, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
