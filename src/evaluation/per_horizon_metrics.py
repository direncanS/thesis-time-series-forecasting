import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import (
    apply_runtime_overrides,
    load_config,
    prepare_supervised_data,
)


def per_horizon_mse_mae_rmse(y_true_orig, y_pred_orig):
    diff = y_true_orig - y_pred_orig
    mse_per_h = np.mean(diff ** 2, axis=(0, 2))
    mae_per_h = np.mean(np.abs(diff), axis=(0, 2))
    rmse_per_h = np.sqrt(mse_per_h)
    return mse_per_h, mae_per_h, rmse_per_h


def run_per_horizon(config_path=None, *, results_dir=None):
    cfg = apply_runtime_overrides(load_config(config_path), smoke=False, results_dir=results_dir)
    results_path = Path(cfg["results_dir"])
    data = prepare_supervised_data(cfg["data_path"], cfg["input_len"], cfg["output_len"])
    y_test_orig = data["y_test_orig"]
    rows = []

    lr_preds = np.load(results_path / "preds_lr.npy")
    mse_h, mae_h, rmse_h = per_horizon_mse_mae_rmse(y_test_orig, lr_preds)
    for h in range(cfg["output_len"]):
        rows.append(
            {
                "model": "LR",
                "horizon": h + 1,
                "mse": float(mse_h[h]),
                "mae": float(mae_h[h]),
                "rmse": float(rmse_h[h]),
                "seed_count": 1,
                "seed_std_mse": 0.0,
            }
        )

    for model_name in ("mlp", "lstm", "tft"):
        per_seed_mse = []
        per_seed_mae = []
        per_seed_rmse = []
        for seed in cfg["seeds"]:
            preds = np.load(results_path / f"preds_{model_name}_seed{seed}.npy")
            mse_h, mae_h, rmse_h = per_horizon_mse_mae_rmse(y_test_orig, preds)
            per_seed_mse.append(mse_h)
            per_seed_mae.append(mae_h)
            per_seed_rmse.append(rmse_h)

        mse_stack = np.stack(per_seed_mse, axis=0)
        mae_stack = np.stack(per_seed_mae, axis=0)
        rmse_stack = np.stack(per_seed_rmse, axis=0)
        for h in range(cfg["output_len"]):
            rows.append(
                {
                    "model": model_name.upper(),
                    "horizon": h + 1,
                    "mse": float(mse_stack.mean(axis=0)[h]),
                    "mae": float(mae_stack.mean(axis=0)[h]),
                    "rmse": float(rmse_stack.mean(axis=0)[h]),
                    "seed_count": len(cfg["seeds"]),
                    "seed_std_mse": float(mse_stack.std(axis=0, ddof=1)[h]),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(results_path / "per_horizon_metrics.csv", index=False)
    return out_df


def build_parser():
    parser = argparse.ArgumentParser(description="Per-horizon descriptive metrics export.")
    parser.add_argument("--config", default="configs/experiments/fair_core_v2.yaml")
    parser.add_argument("--results-dir", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    run_per_horizon(args.config, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
