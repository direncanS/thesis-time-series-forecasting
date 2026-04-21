import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import (
    MLP,
    LSTMModel,
    apply_runtime_overrides,
    compute_original_metrics,
    load_config,
    prepare_supervised_data,
)

try:
    from arch.bootstrap import MovingBlockBootstrap
except ImportError:  # pragma: no cover
    MovingBlockBootstrap = None


BOOTSTRAP_SEED = 2026
N_BOOTSTRAP = 10_000
MODELS_ORDERED = ["LR", "MLP", "LSTM", "TFT"]
HARDWARE_NOTE = "single local machine run; bachelor-safe local-only workflow"


def export_per_seed_metrics(y_test_orig, results_dir, seeds):
    rows = []
    preds_lr = np.load(results_dir / "preds_lr.npy")
    mse, mae, rmse = compute_original_metrics(y_test_orig, preds_lr)
    rows.append(
        {
            "model": "LR",
            "seed": "deterministic",
            "mse_original": mse,
            "mae_original": mae,
            "rmse_original": rmse,
        }
    )
    for model_name in ("mlp", "lstm", "tft"):
        for seed in seeds:
            preds = np.load(results_dir / f"preds_{model_name}_seed{seed}.npy")
            mse, mae, rmse = compute_original_metrics(y_test_orig, preds)
            rows.append(
                {
                    "model": model_name.upper(),
                    "seed": seed,
                    "mse_original": mse,
                    "mae_original": mae,
                    "rmse_original": rmse,
                }
            )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(results_dir / "per_seed_metrics.csv", index=False)
    return out_df


def export_complexity_metrics(results_dir):
    runtime_df = pd.read_csv(results_dir / "runtime_seconds.csv")
    runtime_summary = runtime_df.groupby("model")["wall_clock_seconds"].sum().to_dict()
    tft_metrics = pd.read_csv(results_dir / "tft_metrics.csv")
    tft_n_params = int(tft_metrics["n_params"].iloc[0])

    input_flat = 96 * 7
    output_flat = 24 * 7
    lr_n_params = output_flat * input_flat + output_flat
    mlp_n_params = sum(p.numel() for p in MLP(input_size=input_flat, output_size=output_flat).parameters())
    lstm_n_params = sum(p.numel() for p in LSTMModel(input_size=7, hidden_size=64, output_size=output_flat).parameters())

    rows = [
        {
            "model": "LR",
            "n_params": lr_n_params,
            "architectural_category": "linear",
            "wall_clock_seconds": float(runtime_summary.get("LR", 0.0)),
            "hardware_note": HARDWARE_NOTE,
        },
        {
            "model": "MLP",
            "n_params": mlp_n_params,
            "architectural_category": "shallow-MLP",
            "wall_clock_seconds": float(runtime_summary.get("MLP", 0.0)),
            "hardware_note": HARDWARE_NOTE,
        },
        {
            "model": "LSTM",
            "n_params": lstm_n_params,
            "architectural_category": "recurrent",
            "wall_clock_seconds": float(runtime_summary.get("LSTM", 0.0)),
            "hardware_note": HARDWARE_NOTE,
        },
        {
            "model": "TFT",
            "n_params": tft_n_params,
            "architectural_category": "transformer-family",
            "wall_clock_seconds": float(runtime_summary.get("TFT", 0.0)),
            "hardware_note": HARDWARE_NOTE,
        },
    ]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(results_dir / "complexity_metrics.csv", index=False)
    return out_df


def _manual_moving_block_ci(diff, *, block_length, reps, seed):
    rng = np.random.default_rng(seed)
    n = len(diff)
    dist = np.empty(reps, dtype=float)
    max_start = max(n - block_length + 1, 1)
    for idx in range(reps):
        chunks = []
        current = 0
        while current < n:
            start = int(rng.integers(0, max_start))
            block = diff[start : start + block_length]
            chunks.append(block)
            current += len(block)
        sample = np.concatenate(chunks)[:n]
        dist[idx] = sample.mean()
    return dist


def _bootstrap_distribution(diff, *, block_length, reps, seed):
    if MovingBlockBootstrap is None:
        return _manual_moving_block_ci(diff, block_length=block_length, reps=reps, seed=seed)

    bs = MovingBlockBootstrap(block_length, diff, seed=seed)
    dist = np.empty(reps, dtype=float)
    for idx, data in enumerate(bs.bootstrap(reps)):
        sample = data[0][0]
        dist[idx] = np.mean(sample)
    return dist


def export_bootstrap_intervals(y_test_orig, results_dir, input_len):
    preds = {"LR": np.load(results_dir / "preds_lr.npy")}
    seeds = sorted(
        {
            int(path.stem.split("seed")[-1])
            for path in results_dir.glob("preds_mlp_seed*.npy")
        }
    )
    for model_name in ("mlp", "lstm", "tft"):
        stack = np.stack(
            [np.load(results_dir / f"preds_{model_name}_seed{seed}.npy") for seed in seeds],
            axis=0,
        )
        preds[model_name.upper()] = stack.mean(axis=0)

    def per_window_errors(y_true, y_pred):
        diff = y_true - y_pred
        flat = diff.reshape(diff.shape[0], -1)
        return (flat ** 2).mean(axis=1), np.abs(flat).mean(axis=1)

    errors = {
        name: dict(zip(("mse", "mae"), per_window_errors(y_test_orig, pred)))
        for name, pred in preds.items()
    }
    for name, payload in errors.items():
        payload["rmse"] = np.sqrt(payload["mse"])

    pairs = [
        (MODELS_ORDERED[i], MODELS_ORDERED[j])
        for i in range(len(MODELS_ORDERED))
        for j in range(i + 1, len(MODELS_ORDERED))
    ]

    rows = []
    sensitivity_rows = []
    for a, b in pairs:
        for metric in ("mse", "mae", "rmse"):
            diff = errors[a][metric] - errors[b][metric]
            dist = _bootstrap_distribution(
                diff,
                block_length=input_len,
                reps=N_BOOTSTRAP,
                seed=BOOTSTRAP_SEED,
            )
            rows.append(
                {
                    "model_pair": f"{a}_vs_{b}",
                    "metric": metric.upper(),
                    "mean_diff": round(float(diff.mean()), 6),
                    "ci_low": round(float(np.percentile(dist, 2.5)), 6),
                    "ci_high": round(float(np.percentile(dist, 97.5)), 6),
                    "bootstrap_n": N_BOOTSTRAP,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "block_length": input_len,
                    "n_test_windows": len(diff),
                }
            )

            for block_length in (24, 48, 96, 192):
                sensitivity_dist = _bootstrap_distribution(
                    diff,
                    block_length=block_length,
                    reps=1000,
                    seed=BOOTSTRAP_SEED,
                )
                sensitivity_rows.append(
                    {
                        "model_pair": f"{a}_vs_{b}",
                        "metric": metric.upper(),
                        "block_length": block_length,
                        "ci_low": round(float(np.percentile(sensitivity_dist, 2.5)), 6),
                        "ci_high": round(float(np.percentile(sensitivity_dist, 97.5)), 6),
                        "ci_width": round(
                            float(np.percentile(sensitivity_dist, 97.5) - np.percentile(sensitivity_dist, 2.5)),
                            6,
                        ),
                    }
                )

    intervals_df = pd.DataFrame(rows)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    intervals_df.to_csv(results_dir / "bootstrap_intervals.csv", index=False)
    sensitivity_df.to_csv(results_dir / "bootstrap_block_sensitivity.csv", index=False)
    return intervals_df, sensitivity_df


def run_analysis(config_path=None, *, results_dir=None):
    cfg = apply_runtime_overrides(load_config(config_path), smoke=False, results_dir=results_dir)
    results_path = cfg["results_dir"]
    data = prepare_supervised_data(cfg["data_path"], cfg["input_len"], cfg["output_len"])
    y_test_orig = data["y_test_orig"]

    per_seed = export_per_seed_metrics(y_test_orig, Path(results_path), cfg["seeds"])
    complexity = export_complexity_metrics(Path(results_path))
    intervals, sensitivity = export_bootstrap_intervals(y_test_orig, Path(results_path), cfg["input_len"])
    return {
        "per_seed": per_seed,
        "complexity": complexity,
        "intervals": intervals,
        "sensitivity": sensitivity,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Post-training metrics and moving-block bootstrap.")
    parser.add_argument("--config", default="configs/experiments/fair_core_v2.yaml")
    parser.add_argument("--results-dir", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    run_analysis(args.config, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
