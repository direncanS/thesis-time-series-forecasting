"""AOPC continuous faithfulness over occlusion ranks.

Purpose (RQ2/RQ3 closure target)
--------------------------------
Closes C6 (continuous faithfulness metric) for the bachelor-safe v6.1 plan.
Consumes the common occlusion ranks produced by
``src/explainability/common_importance.py`` (B7a) and computes the
Area Over Perturbation Curve (AOPC; Samek et al., 2017 — time-series
adaptation).

Role separation
---------------------------------------
- **Occlusion importance** = the common explanation *instrument*
  (produced by ``common_importance.py``; per-(model, seed, variable) rank
  vectors uniformly for all 4 models).
- **AOPC** = the faithfulness *metric* computed **over the occlusion
  ranks** (this file). AOPC replaces the v1 binary
  ``top > bottom`` decision rule (non-discriminative 12/12 pass → flat
  y-axis) with a continuous strength measure.

Formula
-------
For each (model, seed):

    baseline_mse(model, seed)
        = MSE on the scaled test partition using the unmodified input.

    For k ∈ {1, …, n_features = 7}:
        top_k = first k variables of the per-(model, seed) occlusion
                ranking (descending importance).
        bot_k = last k variables of the same ranking.

        top_k_mse_increase = MSE(top_k occluded) − baseline_mse
        bot_k_mse_increase = MSE(bot_k occluded) − baseline_mse
        gap_k              = top_k_mse_increase − bot_k_mse_increase

    AOPC(model, seed) = mean over k of gap_k.

Higher AOPC = stronger behavioural separation between top-ranked and
bottom-ranked variables under perturbation, i.e. a more *faithful*
ranking.

Baseline convention
-------------------
Variable occlusion sets the scaled-space column to 0.0, which equals the
training-distribution mean in the original space by construction
(``StandardScaler`` fit on the training partition only).

No model-specific branching
---------------------------
The top-level AOPC loop uses the same per-model ``predict_fn`` factories
used by ``common_importance.py``, imported rather than duplicated.

Outputs
-------
``results/bachelor_safe_v2/faithfulness_aopc.csv``        — summary, one row
    per (model, seed): ``model, seed, baseline_mse, aopc, gap_std_across_k``.

``results/bachelor_safe_v2/faithfulness_aopc_per_k.csv``  — long-format
    diagnostic, one row per (model, seed, k): ``model, seed, k,
    top_mse_increase, bot_mse_increase, gap``.

Transparency note
-----------------
AOPC was introduced in the v2 rerun following the observation that the
v1 binary ``top > bottom`` rule produced a 12/12 pass on three seeds and
a flat RQ3 y-axis. AOPC replaces binary as the primary RQ2 faithfulness
measure; the thesis does not claim formal pre-registration.

Run
---
    python src/explainability/faithfulness_test.py \\
        --config configs/experiments/fair_core_v2.yaml
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import lightning as L  # noqa: F401  (pytorch-forecasting dependency)
import numpy as np
import pandas as pd

from src.common import (
    apply_runtime_overrides,
    ensure_dir,
    load_config,
    prepare_supervised_data,
)
from src.explainability.common_importance import (
    MSE,  # noqa: F401 — re-export for TFT checkpoint pickle resolution
    make_lr_predict,
    make_lstm_predict,
    make_mlp_predict,
    make_tft_predict,
    mse_scaled,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Masking + ranking helpers
# ---------------------------------------------------------------------------


def mask_variables(X_3d_scaled: np.ndarray, var_indices: list[int]) -> np.ndarray:
    """Return a copy of ``X_3d_scaled`` with the specified variable columns set to 0."""
    out = X_3d_scaled.copy()
    for idx in var_indices:
        out[:, :, idx] = 0.0
    return out


def rank_from_occlusion(
    occlusion_df: pd.DataFrame,
    model: str,
    seed,
    var_names: list[str],
) -> list[str]:
    """Return the per-(model, seed) variable ranking, descending by importance."""
    subset = occlusion_df[
        (occlusion_df["model"] == model) & (occlusion_df["seed"].astype(str) == str(seed))
    ]
    if subset.empty:
        raise RuntimeError(
            f"occlusion_importance.csv has no rows for model={model} seed={seed}"
        )
    missing = set(var_names) - set(subset["variable"])
    if missing:
        raise RuntimeError(f"occlusion ranking missing variables: {missing}")
    sorted_rows = subset.sort_values("importance", ascending=False)
    return sorted_rows["variable"].tolist()


def aopc_for_model_seed(
    predict_fn,
    X_test_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
    ranking: list[str],
    var_names: list[str],
) -> tuple[float, float, list[dict]]:
    """Return (aopc, gap_std_across_k, per_k_rows) for one (model, seed)."""
    baseline = mse_scaled(y_test_scaled, predict_fn(X_test_scaled))
    var_to_idx = {v: i for i, v in enumerate(var_names)}
    per_k_rows: list[dict] = []
    gaps: list[float] = []
    n = len(var_names)
    for k in range(1, n + 1):
        top_indices = [var_to_idx[v] for v in ranking[:k]]
        bot_indices = [var_to_idx[v] for v in ranking[-k:]]
        top_mse = mse_scaled(y_test_scaled, predict_fn(mask_variables(X_test_scaled, top_indices)))
        bot_mse = mse_scaled(y_test_scaled, predict_fn(mask_variables(X_test_scaled, bot_indices)))
        top_inc = top_mse - baseline
        bot_inc = bot_mse - baseline
        gap = top_inc - bot_inc
        gaps.append(gap)
        per_k_rows.append(
            {
                "k": k,
                "baseline_mse": round(baseline, 8),
                "top_mse_increase": round(top_inc, 8),
                "bot_mse_increase": round(bot_inc, 8),
                "gap": round(gap, 8),
            }
        )
    gaps_arr = np.array(gaps, dtype=float)
    aopc = float(gaps_arr.mean())
    gap_std = float(gaps_arr.std(ddof=1)) if gaps_arr.size > 1 else 0.0
    return aopc, gap_std, per_k_rows


# ---------------------------------------------------------------------------
# Run orchestrator
# ---------------------------------------------------------------------------


def _per_seed_predict_factories(data, cfg, checkpoints_path: Path, seeds):
    factories = {"LR": [("deterministic", make_lr_predict(data, cfg))]}
    factories["MLP"] = [
        (seed, make_mlp_predict(data, cfg, checkpoints_path / f"mlp_seed{seed}.pt"))
        for seed in seeds
    ]
    factories["LSTM"] = [
        (seed, make_lstm_predict(data, cfg, checkpoints_path / f"lstm_seed{seed}.pt"))
        for seed in seeds
    ]
    factories["TFT"] = [
        (seed, make_tft_predict(data, cfg, checkpoints_path / f"tft_seed{seed}.ckpt"))
        for seed in seeds
    ]
    return factories


def run_aopc(
    config_path: str | None = None,
    *,
    results_dir: str | None = None,
    checkpoints_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = apply_runtime_overrides(
        load_config(config_path),
        smoke=False,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
    )
    results_path = ensure_dir(cfg["results_dir"])
    checkpoints_path = Path(cfg["checkpoints_dir"])
    data = prepare_supervised_data(cfg["data_path"], cfg["input_len"], cfg["output_len"])
    var_names = data["var_names"]
    seeds = cfg["seeds"]

    occlusion_csv = results_path / "occlusion_importance.csv"
    if not occlusion_csv.exists():
        raise FileNotFoundError(
            f"missing {occlusion_csv} — run src/explainability/common_importance.py first"
        )
    occlusion_df = pd.read_csv(occlusion_csv)

    print(f"[B7b] results_dir={results_path}")
    print(f"[B7b] occlusion ranks: {len(occlusion_df)} rows loaded")
    print(f"[B7b] seeds={seeds}")

    factories = _per_seed_predict_factories(data, cfg, checkpoints_path, seeds)
    summary_rows: list[dict] = []
    per_k_rows: list[dict] = []

    for model in ("LR", "MLP", "LSTM", "TFT"):
        print(f"\n[{model}] AOPC over occlusion ranks...")
        for seed, predict_fn in factories[model]:
            ranking = rank_from_occlusion(occlusion_df, model, seed, var_names)
            aopc, gap_std, k_rows = aopc_for_model_seed(
                predict_fn, data["X_test"], data["y_test"], ranking, var_names
            )
            summary_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "baseline_mse": round(k_rows[0]["baseline_mse"], 8),
                    "aopc": round(aopc, 8),
                    "gap_std_across_k": round(gap_std, 8),
                }
            )
            for row in k_rows:
                per_k_rows.append({"model": model, "seed": seed, **row})
            print(f"  seed {seed}: aopc={aopc:+.4f}  gap_std_across_k={gap_std:.4f}  ranking={ranking}")

    summary_df = pd.DataFrame(summary_rows)
    per_k_df = pd.DataFrame(per_k_rows)
    summary_csv = results_path / "faithfulness_aopc.csv"
    per_k_csv = results_path / "faithfulness_aopc_per_k.csv"
    summary_df.to_csv(summary_csv, index=False)
    per_k_df.to_csv(per_k_csv, index=False)
    print(f"\n[B7b] saved {summary_csv} ({len(summary_df)} rows)")
    print(f"[B7b] saved {per_k_csv} ({len(per_k_df)} rows)")

    print("\n[B7b] Per-(model, seed) AOPC summary:")
    print(summary_df.to_string(index=False))

    return summary_df, per_k_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="B7b — AOPC continuous faithfulness over occlusion ranks."
    )
    parser.add_argument("--config", default="configs/experiments/fair_core_v2.yaml")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_aopc(args.config, results_dir=args.results_dir, checkpoints_dir=args.checkpoints_dir)


if __name__ == "__main__":
    main()
