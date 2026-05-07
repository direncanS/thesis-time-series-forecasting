"""Bounded auxiliary LSTM explanation artifacts for the active v2 experiment.

This script keeps LSTM SHAP as a model-specific auxiliary explanation artifact.
The primary active XAI workflow remains common occlusion importance + AOPC +
cross-model agreement.

Important LSTM limitation: if SHAP GradientExplainer is too slow or fails, the
script can use manual gradient attribution. That fallback is explicitly not true
SHAP and is marked in the CSV metadata.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import torch

from src.common import (
    LSTMModel,
    apply_runtime_overrides,
    ensure_dir,
    load_config,
    prepare_supervised_data,
)

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import shap
except ImportError:  # pragma: no cover - allows --force-gradient-fallback without SHAP
    shap = None


DEFAULT_CONFIG_PATH = "configs/experiments/fair_core_v2.yaml"
DEFAULT_MAX_EVAL_WINDOWS = 32
DEFAULT_MAX_BACKGROUND_WINDOWS = 32
EVAL_SEED = 42
BG_SEED = 42
SHAP_BASELINE = "train_mean_scaled"
INPUT_SPACE = "scaled"
TRUE_EXPLANATION_TYPE = "SHAP"
FALLBACK_EXPLANATION_TYPE = "ManualGradientAttribution"
FALLBACK_EXPLAINER = "GradientAttribution (fallback; not true SHAP)"


def log(message: str) -> None:
    print(message, flush=True)


def _bounded_choice(
    rng: np.random.RandomState,
    n_available: int,
    requested: int,
    label: str,
) -> np.ndarray:
    if requested <= 0:
        raise ValueError(f"{label} must be positive, got {requested}")
    n_used = min(requested, n_available)
    if n_used < requested:
        log(f"[shap-lstm] {label}: requested {requested}, using available {n_used}")
    return rng.choice(n_available, size=n_used, replace=False)


def manual_gradient_attribution(lstm_model: LSTMModel, x_eval: torch.Tensor) -> np.ndarray:
    """Bounded fallback attribution; this is not true SHAP."""
    log(f"[shap-lstm] fallback start: manual gradients for {len(x_eval)} eval windows")
    lstm_model.eval()
    grads = []
    for i in range(len(x_eval)):
        if i == 0 or (i + 1) % 8 == 0 or i + 1 == len(x_eval):
            log(f"[shap-lstm] fallback progress: {i + 1}/{len(x_eval)}")
        x_single = x_eval[i : i + 1].clone().requires_grad_(True)
        out = lstm_model(x_single).sum()
        out.backward()
        grads.append(x_single.grad.detach().numpy()[0])
        lstm_model.zero_grad()
    grad_array = np.stack(grads, axis=0)
    out = grad_array[np.newaxis, :, :, :]
    log(f"[shap-lstm] fallback finished: attribution shape={out.shape}")
    return out


def run_shap_for_lstm(
    lstm_model: LSTMModel,
    x_eval: torch.Tensor,
    x_background: torch.Tensor,
    *,
    force_gradient_fallback: bool,
) -> tuple[np.ndarray, str, bool]:
    """Run GradientExplainer; fallback to bounded manual gradients if needed."""
    if force_gradient_fallback:
        log("[shap-lstm] force-gradient-fallback enabled; skipping GradientExplainer")
        vals = manual_gradient_attribution(lstm_model, x_eval)
        return vals, FALLBACK_EXPLAINER, False

    if shap is None:
        raise RuntimeError("shap is not installed; use --force-gradient-fallback or install SHAP")

    try:
        log(f"[shap-lstm] GradientExplainer start: background shape={tuple(x_background.shape)}")
        t0 = time.time()
        explainer = shap.GradientExplainer(lstm_model, x_background)
        log(f"[shap-lstm] GradientExplainer built in {time.time() - t0:.1f}s")
        log(f"[shap-lstm] shap_values start: eval shape={tuple(x_eval.shape)}")
        t1 = time.time()
        vals = explainer.shap_values(x_eval)
        log(f"[shap-lstm] shap_values finished in {time.time() - t1:.1f}s")
        del explainer
        gc.collect()
        return vals, "GradientExplainer", True
    except Exception as exc:
        log(f"[shap-lstm] GradientExplainer failed: {type(exc).__name__}: {exc}")

    vals = manual_gradient_attribution(lstm_model, x_eval)
    return vals, FALLBACK_EXPLAINER, False


def parse_shap_output_3d(shap_values, n_eval: int, n_outputs: int) -> np.ndarray:
    """Return SHAP values as (n_outputs, n_eval, input_len, n_features)."""
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
            return shap_values[np.newaxis, :, :, :]

    vals = getattr(shap_values, "values", np.array(shap_values))
    if vals.ndim == 4:
        if vals.shape[0] == n_eval:
            return vals.transpose(3, 0, 1, 2)
        return vals
    if vals.ndim == 3:
        return vals[np.newaxis, :, :, :]
    raise ValueError(f"Unexpected SHAP output shape: {vals.shape}")


def cross_variable_from_gradient_fallback(
    lstm_model: LSTMModel,
    x_eval: torch.Tensor,
    n_features: int,
) -> np.ndarray:
    """Populate a bounded cross-variable matrix for the non-SHAP fallback path."""
    log("[shap-lstm] fallback cross-variable start")
    cross = np.zeros((n_features, n_features))
    for out_var_idx in range(n_features):
        out_grads = []
        for i in range(len(x_eval)):
            x_single = x_eval[i : i + 1].clone().requires_grad_(True)
            output = lstm_model(x_single)
            out_for_var = output[0, out_var_idx::n_features].sum()
            out_for_var.backward()
            grad = x_single.grad.detach().numpy()[0]
            out_grads.append(np.mean(np.abs(grad), axis=0))
            lstm_model.zero_grad()
        cross[out_var_idx] = np.mean(out_grads, axis=0)
    log("[shap-lstm] fallback cross-variable finished")
    return cross


def run_shap_lstm(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    results_dir: str | None = None,
    checkpoints_dir: str | None = None,
    max_background_windows: int = DEFAULT_MAX_BACKGROUND_WINDOWS,
    max_eval_windows: int = DEFAULT_MAX_EVAL_WINDOWS,
    max_seeds: int | None = None,
    force_gradient_fallback: bool = False,
) -> pd.DataFrame:
    log("[shap-lstm] script start")
    cfg = apply_runtime_overrides(
        load_config(config_path),
        smoke=False,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
    )
    results_path = ensure_dir(cfg["results_dir"])
    checkpoints_path = Path(cfg["checkpoints_dir"])
    seeds = list(cfg["seeds"])
    if max_seeds is not None:
        if max_seeds <= 0:
            raise ValueError(f"max_seeds must be positive, got {max_seeds}")
        seeds = seeds[:max_seeds]

    log(
        "[shap-lstm] loaded config: "
        f"config={config_path}, results_dir={results_path}, "
        f"checkpoints_dir={checkpoints_path}, seeds={seeds}"
    )

    data = prepare_supervised_data(cfg["data_path"], cfg["input_len"], cfg["output_len"])
    log(
        "[shap-lstm] prepared data shapes: "
        f"X_train={data['X_train'].shape}, X_test={data['X_test'].shape}, "
        f"y_test={data['y_test'].shape}"
    )

    rng = np.random.RandomState(EVAL_SEED)
    eval_idx = _bounded_choice(rng, data["n_test"], max_eval_windows, "max_eval_windows")
    bg_idx = _bounded_choice(rng, data["n_train"], max_background_windows, "max_background_windows")

    x_eval = torch.FloatTensor(data["X_test"][eval_idx])
    x_background = torch.FloatTensor(data["X_train"][bg_idx])
    num_eval_used = int(x_eval.shape[0])
    num_background_used = int(x_background.shape[0])
    log(
        "[shap-lstm] background/eval tensor shapes: "
        f"background={tuple(x_background.shape)}, eval={tuple(x_eval.shape)}"
    )

    input_len = cfg["input_len"]
    output_len = cfg["output_len"]
    n_features = len(data["var_names"])
    lstm_cfg = cfg["models"]["lstm"]

    per_seed_importance: dict[int, np.ndarray] = {}
    per_seed_cross: dict[int, np.ndarray] = {}
    per_seed_mse_check: dict[int, float] = {}
    per_seed_explainer: dict[int, str] = {}
    per_seed_true_shap: dict[int, bool] = {}

    for seed in seeds:
        log(f"[shap-lstm] seed start: {seed}")
        ckpt_path = checkpoints_path / f"lstm_seed{seed}.pt"
        log(f"[shap-lstm] checkpoint path: {ckpt_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing LSTM checkpoint: {ckpt_path}")

        lstm_model = LSTMModel(
            input_size=n_features,
            hidden_size=lstm_cfg["hidden_size"],
            output_size=data["output_size"],
            num_layers=lstm_cfg["num_layers"],
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        lstm_model.load_state_dict(state)
        lstm_model.eval()
        log(f"[shap-lstm] checkpoint loaded: {ckpt_path}")
        log(
            "[shap-lstm] background/eval tensor shapes: "
            f"background={tuple(x_background.shape)}, eval={tuple(x_eval.shape)}"
        )

        with torch.no_grad():
            y_pred_flat = lstm_model(torch.FloatTensor(data["X_test"])).numpy()
        y_pred_3d = y_pred_flat.reshape(data["n_test"], output_len, n_features)
        per_seed_mse_check[seed] = float(np.mean((data["y_test"] - y_pred_3d) ** 2))

        shap_values, explainer_name, is_true_shap = run_shap_for_lstm(
            lstm_model,
            x_eval,
            x_background,
            force_gradient_fallback=force_gradient_fallback,
        )
        per_seed_explainer[seed] = explainer_name
        per_seed_true_shap[seed] = is_true_shap

        if is_true_shap:
            shap_array = parse_shap_output_3d(shap_values, num_eval_used, data["output_size"])
        else:
            shap_array = shap_values

        var_importance = np.mean(np.abs(shap_array), axis=(0, 1, 2))

        if is_true_shap:
            shap_by_outvar = shap_array.reshape(
                output_len,
                n_features,
                num_eval_used,
                input_len,
                n_features,
            )
            cross_importance = np.mean(np.abs(shap_by_outvar), axis=(0, 2, 3))
        else:
            cross_importance = cross_variable_from_gradient_fallback(
                lstm_model,
                x_eval,
                n_features,
            )

        per_seed_importance[seed] = var_importance
        per_seed_cross[seed] = cross_importance

        del lstm_model, shap_values, shap_array, y_pred_flat, y_pred_3d
        gc.collect()

    imp_stack = np.stack([per_seed_importance[s] for s in seeds], axis=0)
    mean_importance = imp_stack.mean(axis=0)
    std_importance = imp_stack.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(n_features)

    cross_stack = np.stack([per_seed_cross[s] for s in seeds], axis=0)
    mean_cross = cross_stack.mean(axis=0)

    explainers = sorted(set(per_seed_explainer.values()))
    mean_explainer = explainers[0] if len(explainers) == 1 else "mixed"
    mean_true_shap = all(per_seed_true_shap.values())

    rows = []
    for seed in seeds:
        ranking = np.argsort(-per_seed_importance[seed])
        warning = "" if per_seed_true_shap[seed] else "manual_gradient_fallback_not_true_shap"
        explanation_type = TRUE_EXPLANATION_TYPE if per_seed_true_shap[seed] else FALLBACK_EXPLANATION_TYPE
        for idx in range(n_features):
            rows.append(
                {
                    "model": "LSTM",
                    "seed": seed,
                    "variable": data["var_names"][idx],
                    "shap_importance": float(per_seed_importance[seed][idx]),
                    "input_space": INPUT_SPACE,
                    "explanation_type": explanation_type,
                    "active_config": config_path,
                    "is_true_shap": per_seed_true_shap[seed],
                    "warning": warning,
                    "num_background_windows_used": num_background_used,
                    "num_eval_windows_used": num_eval_used,
                    "method": f"SHAP ({per_seed_explainer[seed]})",
                    "aggregation": "per-seed",
                    "shap_rank": list(ranking).index(idx) + 1,
                    "eval_seed": EVAL_SEED,
                    "bg_seed": BG_SEED,
                    "shap_baseline": SHAP_BASELINE,
                    "explainer": per_seed_explainer[seed],
                    "mse_scaled_check": per_seed_mse_check[seed],
                    "checkpoint_path": str(checkpoints_path / f"lstm_seed{seed}.pt"),
                }
            )

    mean_ranking = np.argsort(-mean_importance)
    mean_warning = "" if mean_true_shap else "one_or_more_seeds_used_manual_gradient_fallback_not_true_shap"
    mean_explanation_type = TRUE_EXPLANATION_TYPE if mean_true_shap else "mixed_or_fallback"
    for idx in range(n_features):
        rows.append(
            {
                "model": "LSTM",
                "seed": "mean",
                "variable": data["var_names"][idx],
                "shap_importance": float(mean_importance[idx]),
                "input_space": INPUT_SPACE,
                "explanation_type": mean_explanation_type,
                "active_config": config_path,
                "is_true_shap": mean_true_shap,
                "warning": mean_warning,
                "num_background_windows_used": num_background_used,
                "num_eval_windows_used": num_eval_used,
                "method": f"SHAP ({mean_explainer})",
                "aggregation": "mean-across-seeds",
                "shap_rank": list(mean_ranking).index(idx) + 1,
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
    out_csv = results_path / "shap_lstm.csv"
    out_df.to_csv(out_csv, index=False)
    log(f"[shap-lstm] CSV saved: {out_csv} ({len(out_df)} rows)")

    cross_df = pd.DataFrame(
        mean_cross,
        index=[f"output_{v}" for v in data["var_names"]],
        columns=[f"input_{v}" for v in data["var_names"]],
    )
    cross_csv = results_path / "shap_lstm_cross.csv"
    cross_df.to_csv(cross_csv)
    log(f"[shap-lstm] CSV saved: {cross_csv}")
    return out_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded auxiliary LSTM explanation export for active v2.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--max-background-windows", type=int, default=DEFAULT_MAX_BACKGROUND_WINDOWS)
    parser.add_argument("--max-eval-windows", type=int, default=DEFAULT_MAX_EVAL_WINDOWS)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--force-gradient-fallback", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_shap_lstm(
        args.config,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
        max_background_windows=args.max_background_windows,
        max_eval_windows=args.max_eval_windows,
        max_seeds=args.max_seeds,
        force_gradient_fallback=args.force_gradient_fallback,
    )


if __name__ == "__main__":
    main()
