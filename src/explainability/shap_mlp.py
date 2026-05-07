"""Auxiliary MLP SHAP artifacts for the active v2 fair-core experiment.

This script keeps SHAP as a model-specific auxiliary explanation artifact.
The primary active XAI workflow remains common occlusion importance + AOPC +
cross-model agreement.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch

from src.common import (
    MLP,
    apply_runtime_overrides,
    ensure_dir,
    load_config,
    prepare_supervised_data,
)

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import shap
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("ERROR: shap is not installed. Install requirements first.") from exc


DEFAULT_CONFIG_PATH = "configs/experiments/fair_core_v2.yaml"
N_EVAL = 100
N_BG = 100
EVAL_SEED = 42
BG_SEED = 42
SHAP_BASELINE = "train_mean_scaled"
INPUT_SPACE = "scaled"
EXPLANATION_TYPE = "SHAP"


def parse_shap_output(shap_values, n_eval: int, n_outputs: int) -> np.ndarray:
    """Return SHAP values as (n_outputs, n_eval, n_features)."""
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

    vals = getattr(shap_values, "values", np.array(shap_values))
    if vals.ndim == 3:
        if vals.shape[0] == n_eval:
            return vals.transpose(2, 0, 1)
        return vals
    if vals.ndim == 2:
        return vals[np.newaxis, :, :]
    raise ValueError(f"Unexpected SHAP output shape: {vals.shape}")


def run_shap_for_mlp(
    mlp_model: MLP,
    x_eval: torch.Tensor,
    x_background: torch.Tensor,
    bg_idx_np: np.ndarray,
    x_train_flat: np.ndarray,
):
    """Try DeepExplainer -> GradientExplainer -> KernelExplainer."""
    try:
        explainer = shap.DeepExplainer(mlp_model, x_background)
        return explainer.shap_values(x_eval), "DeepExplainer"
    except Exception as exc:
        print(f"    DeepExplainer failed: {exc}")

    try:
        explainer = shap.GradientExplainer(mlp_model, x_background)
        return explainer.shap_values(x_eval), "GradientExplainer"
    except Exception as exc:
        print(f"    GradientExplainer failed: {exc}")

    def mlp_predict(x):
        with torch.no_grad():
            return mlp_model(torch.FloatTensor(x)).numpy()

    explainer = shap.KernelExplainer(mlp_predict, x_train_flat[bg_idx_np])
    return explainer.shap_values(x_eval.numpy()), "KernelExplainer"


def run_shap_mlp(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    results_dir: str | None = None,
    checkpoints_dir: str | None = None,
) -> pd.DataFrame:
    cfg = apply_runtime_overrides(
        load_config(config_path),
        smoke=False,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
    )
    results_path = ensure_dir(cfg["results_dir"])
    checkpoints_path = Path(cfg["checkpoints_dir"])
    data = prepare_supervised_data(cfg["data_path"], cfg["input_len"], cfg["output_len"])
    seeds = cfg["seeds"]

    rng = np.random.RandomState(EVAL_SEED)
    eval_idx = rng.choice(data["n_test"], size=N_EVAL, replace=False)
    bg_idx = rng.choice(data["n_train"], size=N_BG, replace=False)

    x_eval = torch.FloatTensor(data["X_test_flat"][eval_idx])
    x_background = torch.FloatTensor(data["X_train_flat"][bg_idx])

    input_len = cfg["input_len"]
    output_len = cfg["output_len"]
    n_features = len(data["var_names"])
    hidden_size = cfg["models"]["mlp"]["hidden_size"]

    per_seed_importance: dict[int, np.ndarray] = {}
    per_seed_cross: dict[int, np.ndarray] = {}
    per_seed_mse_check: dict[int, float] = {}
    per_seed_explainer: dict[int, str] = {}

    for seed in seeds:
        print(f"[shap-mlp] seed={seed}")
        ckpt_path = checkpoints_path / f"mlp_seed{seed}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing MLP checkpoint: {ckpt_path}")

        mlp_model = MLP(data["input_size"], data["output_size"], hidden_size=hidden_size)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        mlp_model.load_state_dict(state)
        mlp_model.eval()

        with torch.no_grad():
            y_pred_flat = mlp_model(torch.FloatTensor(data["X_test_flat"])).numpy()
        y_pred_3d = y_pred_flat.reshape(data["n_test"], output_len, n_features)
        per_seed_mse_check[seed] = float(np.mean((data["y_test"] - y_pred_3d) ** 2))

        shap_values, explainer_name = run_shap_for_mlp(
            mlp_model, x_eval, x_background, bg_idx, data["X_train_flat"]
        )
        per_seed_explainer[seed] = explainer_name

        shap_array = parse_shap_output(shap_values, N_EVAL, data["output_size"])
        shap_reshaped = shap_array.reshape(data["output_size"], N_EVAL, input_len, n_features)
        var_importance = np.mean(np.abs(shap_reshaped), axis=(0, 1, 2))

        shap_by_outvar = shap_reshaped.reshape(output_len, n_features, N_EVAL, input_len, n_features)
        cross_importance = np.mean(np.abs(shap_by_outvar), axis=(0, 2, 3))

        per_seed_importance[seed] = var_importance
        per_seed_cross[seed] = cross_importance

    imp_stack = np.stack([per_seed_importance[s] for s in seeds], axis=0)
    mean_importance = imp_stack.mean(axis=0)
    std_importance = imp_stack.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(n_features)

    cross_stack = np.stack([per_seed_cross[s] for s in seeds], axis=0)
    mean_cross = cross_stack.mean(axis=0)

    explainers = sorted(set(per_seed_explainer.values()))
    mean_explainer = explainers[0] if len(explainers) == 1 else "mixed"

    rows = []
    for seed in seeds:
        ranking = np.argsort(-per_seed_importance[seed])
        for idx in range(n_features):
            rows.append(
                {
                    "model": "MLP",
                    "seed": seed,
                    "variable": data["var_names"][idx],
                    "shap_importance": float(per_seed_importance[seed][idx]),
                    "input_space": INPUT_SPACE,
                    "explanation_type": EXPLANATION_TYPE,
                    "active_config": config_path,
                    "method": f"SHAP ({per_seed_explainer[seed]})",
                    "aggregation": "per-seed",
                    "shap_rank": list(ranking).index(idx) + 1,
                    "n_eval": N_EVAL,
                    "n_bg": N_BG,
                    "eval_seed": EVAL_SEED,
                    "bg_seed": BG_SEED,
                    "shap_baseline": SHAP_BASELINE,
                    "explainer": per_seed_explainer[seed],
                    "mse_scaled_check": per_seed_mse_check[seed],
                    "checkpoint_path": str(checkpoints_path / f"mlp_seed{seed}.pt"),
                }
            )

    mean_ranking = np.argsort(-mean_importance)
    for idx in range(n_features):
        rows.append(
            {
                "model": "MLP",
                "seed": "mean",
                "variable": data["var_names"][idx],
                "shap_importance": float(mean_importance[idx]),
                "input_space": INPUT_SPACE,
                "explanation_type": EXPLANATION_TYPE,
                "active_config": config_path,
                "method": f"SHAP ({mean_explainer})",
                "aggregation": "mean-across-seeds",
                "shap_rank": list(mean_ranking).index(idx) + 1,
                "n_eval": N_EVAL,
                "n_bg": N_BG,
                "eval_seed": EVAL_SEED,
                "bg_seed": BG_SEED,
                "shap_baseline": SHAP_BASELINE,
                "explainer": mean_explainer,
                "mse_scaled_check": None,
                "shap_importance_std_across_seeds": float(std_importance[idx]),
                "checkpoint_path": "not_applicable_mean_across_seeds",
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = results_path / "shap_mlp.csv"
    out_df.to_csv(out_csv, index=False)

    cross_df = pd.DataFrame(
        mean_cross,
        index=[f"output_{v}" for v in data["var_names"]],
        columns=[f"input_{v}" for v in data["var_names"]],
    )
    cross_df.to_csv(results_path / "shap_mlp_cross.csv")

    print(f"[shap-mlp] saved {out_csv} ({len(out_df)} rows)")
    return out_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auxiliary MLP SHAP export for active v2.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_shap_mlp(args.config, results_dir=args.results_dir, checkpoints_dir=args.checkpoints_dir)


if __name__ == "__main__":
    main()
