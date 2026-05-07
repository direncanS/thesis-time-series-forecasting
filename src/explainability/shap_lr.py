"""Auxiliary LR SHAP artifacts for the active v2 fair-core experiment.

This script does not replace the primary XAI workflow
(`common_importance.py` -> `faithfulness_test.py` ->
`cross_model_xai_agreement.py`). It only exports model-specific SHAP
attributions for the deterministic Linear Regression baseline.

LR has no seed-specific checkpoint: it is a deterministic closed-form fit on
the active training windows prepared from the configured split/scaler/window
pipeline.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.common import apply_runtime_overrides, ensure_dir, load_config, prepare_supervised_data

try:
    import shap
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("ERROR: shap is not installed. Install requirements first.") from exc


DEFAULT_CONFIG_PATH = "configs/experiments/fair_core_v2.yaml"
N_EVAL = 100
EVAL_SEED = 42
N_BG = "full_X_train_flat"
BG_SEED = None
SHAP_BASELINE = "train_mean_scaled"
INPUT_SPACE = "scaled"
EXPLANATION_TYPE = "SHAP"


def _normalise_shap_output(shap_values, n_eval: int, n_outputs: int) -> np.ndarray:
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


def run_shap_lr(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    results_dir: str | None = None,
) -> pd.DataFrame:
    cfg = apply_runtime_overrides(
        load_config(config_path),
        smoke=False,
        results_dir=results_dir,
    )
    results_path = ensure_dir(cfg["results_dir"])
    data = prepare_supervised_data(cfg["data_path"], cfg["input_len"], cfg["output_len"])

    lr_model = LinearRegression()
    lr_model.fit(data["X_train_flat"], data["y_train_flat"])

    rng = np.random.RandomState(EVAL_SEED)
    eval_idx = rng.choice(data["n_test"], size=N_EVAL, replace=False)
    x_eval = data["X_test_flat"][eval_idx]

    explainer = shap.LinearExplainer(lr_model, data["X_train_flat"])
    shap_values = explainer.shap_values(x_eval)
    shap_array = _normalise_shap_output(shap_values, N_EVAL, data["output_size"])

    input_len = cfg["input_len"]
    output_len = cfg["output_len"]
    n_features = len(data["var_names"])

    shap_reshaped = shap_array.reshape(data["output_size"], N_EVAL, input_len, n_features)
    var_importance = np.mean(np.abs(shap_reshaped), axis=(0, 1, 2))

    shap_by_outvar = shap_reshaped.reshape(output_len, n_features, N_EVAL, input_len, n_features)
    cross_var_importance = np.mean(np.abs(shap_by_outvar), axis=(0, 2, 3))

    coef_reshaped = lr_model.coef_.reshape(data["output_size"], input_len, n_features)
    coef_var_importance = np.mean(np.abs(coef_reshaped), axis=(0, 1))

    shap_ranking = np.argsort(-var_importance)
    coef_ranking = np.argsort(-coef_var_importance)

    rows = []
    for idx in shap_ranking:
        rows.append(
            {
                "model": "LR",
                "seed": "deterministic",
                "variable": data["var_names"][idx],
                "shap_importance": float(var_importance[idx]),
                "input_space": INPUT_SPACE,
                "explanation_type": EXPLANATION_TYPE,
                "active_config": config_path,
                "method": "SHAP (LinearExplainer)",
                "aggregation": "deterministic",
                "shap_rank": list(shap_ranking).index(idx) + 1,
                "coef_importance": float(coef_var_importance[idx]),
                "coef_rank": list(coef_ranking).index(idx) + 1,
                "n_eval": N_EVAL,
                "eval_seed": EVAL_SEED,
                "n_bg": N_BG,
                "bg_seed": BG_SEED,
                "shap_baseline": SHAP_BASELINE,
                "checkpoint_path": "not_applicable_deterministic_closed_form_fit",
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = results_path / "shap_lr.csv"
    out_df.to_csv(out_csv, index=False)

    cross_df = pd.DataFrame(
        cross_var_importance,
        index=[f"output_{v}" for v in data["var_names"]],
        columns=[f"input_{v}" for v in data["var_names"]],
    )
    cross_df.to_csv(results_path / "shap_lr_cross.csv")

    print(f"[shap-lr] saved {out_csv} ({len(out_df)} rows)")
    print("[shap-lr] LR has no seed checkpoint; fitted once from active prepared data.")
    return out_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auxiliary LR SHAP export for active v2.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--results-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_shap_lr(args.config, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
