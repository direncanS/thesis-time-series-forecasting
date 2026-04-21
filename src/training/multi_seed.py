import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, TensorDataset

from src.common import (
    MLP,
    LSTMModel,
    apply_runtime_overrides,
    compute_original_metrics,
    compute_scaled_metrics,
    configure_stochastic_runtime,
    ensure_dir,
    finalize_history_rows,
    inverse_transform_3d,
    load_config,
    prepare_supervised_data,
    runtime_flag_block,
    summarize_results,
    upsert_runtime_rows,
)


def train_with_early_stopping(
    *,
    model_name,
    model,
    X_train_tensor,
    y_train_tensor,
    X_val_tensor,
    y_val_tensor,
    X_test_tensor,
    seed,
    batch_size,
    learning_rate,
    max_epochs,
    patience,
    weight_decay,
    device,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    best_val_loss = float("inf")
    best_epoch = 1
    best_model_state = None
    patience_counter = 0
    history_rows = []

    X_val_tensor = X_val_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(loader)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = loss_fn(val_pred, y_val_tensor).item()

        history_rows.append(
            {
                "model": model_name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state is None:
        raise RuntimeError(
            f"{model_name} seed={seed}: no finite validation loss was recorded; aborting restore-best."
        )

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        y_pred_flat = model(X_test_tensor).detach().cpu().numpy()

    history_rows = finalize_history_rows(history_rows, best_epoch)
    return y_pred_flat, best_epoch, history_rows


def run_pipeline(config_path=None, *, smoke=False, results_dir=None, checkpoints_dir=None):
    cfg = apply_runtime_overrides(
        load_config(config_path),
        smoke=smoke,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
    )
    results_path = ensure_dir(cfg["results_dir"])
    checkpoints_path = ensure_dir(cfg["checkpoints_dir"])
    data = prepare_supervised_data(cfg["data_path"], cfg["input_len"], cfg["output_len"])

    batch_size = cfg["batch_size"]
    max_epochs = cfg["max_epochs"]
    patience = cfg["patience"]
    seeds = cfg["seeds"]
    mlp_cfg = cfg["models"]["mlp"]
    lstm_cfg = cfg["models"]["lstm"]

    print(f"[v2] results_dir={results_path}")
    print(f"[v2] checkpoints_dir={checkpoints_path}")
    print(
        f"[v2] data={cfg['data_path']} n_train={data['n_train']} n_val={data['n_val']} "
        f"n_test={data['n_test']} seeds={seeds}"
    )

    lr_start = time.perf_counter()
    lr_model = LinearRegression()
    lr_model.fit(data["X_train_flat"], data["y_train_flat"])
    lr_seconds = time.perf_counter() - lr_start

    y_pred_lr_scaled = lr_model.predict(data["X_test_flat"]).reshape(
        data["n_test"],
        cfg["output_len"],
        len(data["var_names"]),
    )
    y_pred_lr_orig = inverse_transform_3d(y_pred_lr_scaled, data["scaler"])
    lr_mse_scaled, lr_mae_scaled = compute_scaled_metrics(data["y_test"], y_pred_lr_scaled)
    lr_mse_orig, lr_mae_orig, lr_rmse_orig = compute_original_metrics(data["y_test_orig"], y_pred_lr_orig)

    print("[v2] model=LR solver=closed-form device=cpu deterministic=True")

    device = configure_stochastic_runtime(cfg)
    print(f"[v2] {runtime_flag_block(device)}")

    training_curve_rows = []
    mlp_results = []
    lstm_results = []
    runtime_rows = [
        {
            "model": "LR",
            "seed": "deterministic",
            "wall_clock_seconds": round(lr_seconds, 6),
        }
    ]

    for model_name in ("MLP", "LSTM"):
        print(f"\n[{model_name}] seeds={seeds}")
        for seed in seeds:
            model_start = time.perf_counter()
            if model_name == "MLP":
                model = MLP(
                    data["input_size"],
                    data["output_size"],
                    hidden_size=mlp_cfg["hidden_size"],
                ).to(device)
                X_train_tensor = torch.FloatTensor(data["X_train_flat"])
                X_val_tensor = torch.FloatTensor(data["X_val_flat"])
                X_test_tensor = torch.FloatTensor(data["X_test_flat"])
                learning_rate = mlp_cfg["learning_rate"]
                weight_decay = mlp_cfg["weight_decay"]
            else:
                model = LSTMModel(
                    input_size=len(data["var_names"]),
                    hidden_size=lstm_cfg["hidden_size"],
                    output_size=data["output_size"],
                    num_layers=lstm_cfg["num_layers"],
                ).to(device)
                X_train_tensor = torch.FloatTensor(data["X_train"])
                X_val_tensor = torch.FloatTensor(data["X_val"])
                X_test_tensor = torch.FloatTensor(data["X_test"])
                learning_rate = lstm_cfg["learning_rate"]
                weight_decay = lstm_cfg["weight_decay"]

            y_pred_flat, best_epoch, history_rows = train_with_early_stopping(
                model_name=model_name,
                model=model,
                X_train_tensor=X_train_tensor,
                y_train_tensor=torch.FloatTensor(data["y_train_flat"]),
                X_val_tensor=X_val_tensor,
                y_val_tensor=torch.FloatTensor(data["y_val_flat"]),
                X_test_tensor=X_test_tensor,
                seed=seed,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                patience=patience,
                weight_decay=weight_decay,
                device=device,
            )
            runtime_seconds = time.perf_counter() - model_start
            training_curve_rows.extend(history_rows)

            ckpt_suffix = "mlp" if model_name == "MLP" else "lstm"
            torch.save(model.state_dict(), checkpoints_path / f"{ckpt_suffix}_seed{seed}.pt")

            y_pred_scaled = y_pred_flat.reshape(data["n_test"], cfg["output_len"], len(data["var_names"]))
            y_pred_orig = inverse_transform_3d(y_pred_scaled, data["scaler"])
            mse_scaled, mae_scaled = compute_scaled_metrics(data["y_test"], y_pred_scaled)
            mse_orig, mae_orig, rmse_orig = compute_original_metrics(data["y_test_orig"], y_pred_orig)

            runtime_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "wall_clock_seconds": round(runtime_seconds, 6),
                }
            )

            result_row = {
                "seed": seed,
                "best_epoch": best_epoch,
                "mse_scaled": mse_scaled,
                "mae_scaled": mae_scaled,
                "mse_original": mse_orig,
                "mae_original": mae_orig,
                "rmse_original": rmse_orig,
            }
            if model_name == "MLP":
                mlp_results.append(result_row)
            else:
                lstm_results.append(result_row)

    training_curves_df = pd.DataFrame(training_curve_rows)
    training_curves_df.to_csv(results_path / "training_curves.csv", index=False)

    mlp_best_epochs = [row["best_epoch"] for row in mlp_results]
    lstm_best_epochs = [row["best_epoch"] for row in lstm_results]

    mlp_mse_scaled_mean, mlp_mse_scaled_std = summarize_results(mlp_results, "mse_scaled")
    mlp_mae_scaled_mean, mlp_mae_scaled_std = summarize_results(mlp_results, "mae_scaled")
    mlp_mse_orig_mean, mlp_mse_orig_std = summarize_results(mlp_results, "mse_original")
    mlp_mae_orig_mean, mlp_mae_orig_std = summarize_results(mlp_results, "mae_original")
    mlp_rmse_orig_mean, mlp_rmse_orig_std = summarize_results(mlp_results, "rmse_original")

    lstm_mse_scaled_mean, lstm_mse_scaled_std = summarize_results(lstm_results, "mse_scaled")
    lstm_mae_scaled_mean, lstm_mae_scaled_std = summarize_results(lstm_results, "mae_scaled")
    lstm_mse_orig_mean, lstm_mse_orig_std = summarize_results(lstm_results, "mse_original")
    lstm_mae_orig_mean, lstm_mae_orig_std = summarize_results(lstm_results, "mae_original")
    lstm_rmse_orig_mean, lstm_rmse_orig_std = summarize_results(lstm_results, "rmse_original")

    summary_rows = [
        {
            "model": "LR",
            "seeds": "deterministic",
            "deterministic": True,
            "best_epochs": "",
            "batch_size": batch_size,
            "learning_rate": 0.0,
            "patience": 0,
            "max_epochs": 0,
            "weight_decay": 0.0,
            "mse_scaled_mean": round(lr_mse_scaled, 6),
            "mse_scaled_std": 0.0,
            "mae_scaled_mean": round(lr_mae_scaled, 6),
            "mae_scaled_std": 0.0,
            "mse_original_mean": round(lr_mse_orig, 6),
            "mse_original_std": 0.0,
            "mae_original_mean": round(lr_mae_orig, 6),
            "mae_original_std": 0.0,
            "rmse_original_mean": round(lr_rmse_orig, 6),
            "rmse_original_std": 0.0,
        },
        {
            "model": "MLP",
            "seeds": ",".join(str(seed) for seed in seeds),
            "deterministic": False,
            "best_epochs": ",".join(str(epoch) for epoch in mlp_best_epochs),
            "batch_size": batch_size,
            "learning_rate": mlp_cfg["learning_rate"],
            "patience": patience,
            "max_epochs": max_epochs,
            "weight_decay": mlp_cfg["weight_decay"],
            "mse_scaled_mean": round(mlp_mse_scaled_mean, 6),
            "mse_scaled_std": round(mlp_mse_scaled_std, 6),
            "mae_scaled_mean": round(mlp_mae_scaled_mean, 6),
            "mae_scaled_std": round(mlp_mae_scaled_std, 6),
            "mse_original_mean": round(mlp_mse_orig_mean, 6),
            "mse_original_std": round(mlp_mse_orig_std, 6),
            "mae_original_mean": round(mlp_mae_orig_mean, 6),
            "mae_original_std": round(mlp_mae_orig_std, 6),
            "rmse_original_mean": round(mlp_rmse_orig_mean, 6),
            "rmse_original_std": round(mlp_rmse_orig_std, 6),
        },
        {
            "model": "LSTM",
            "seeds": ",".join(str(seed) for seed in seeds),
            "deterministic": False,
            "best_epochs": ",".join(str(epoch) for epoch in lstm_best_epochs),
            "batch_size": batch_size,
            "learning_rate": lstm_cfg["learning_rate"],
            "patience": patience,
            "max_epochs": max_epochs,
            "weight_decay": lstm_cfg["weight_decay"],
            "mse_scaled_mean": round(lstm_mse_scaled_mean, 6),
            "mse_scaled_std": round(lstm_mse_scaled_std, 6),
            "mae_scaled_mean": round(lstm_mae_scaled_mean, 6),
            "mae_scaled_std": round(lstm_mae_scaled_std, 6),
            "mse_original_mean": round(lstm_mse_orig_mean, 6),
            "mse_original_std": round(lstm_mse_orig_std, 6),
            "mae_original_mean": round(lstm_mae_orig_mean, 6),
            "mae_original_std": round(lstm_mae_orig_std, 6),
            "rmse_original_mean": round(lstm_rmse_orig_mean, 6),
            "rmse_original_std": round(lstm_rmse_orig_std, 6),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_path / "multi_seed_fair_baseline.csv", index=False)

    upsert_runtime_rows(results_path / "runtime_seconds.csv", runtime_rows)
    return {
        "config": cfg,
        "results_dir": results_path,
        "checkpoints_dir": checkpoints_path,
        "summary": summary_df,
        "training_curves": training_curves_df,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Bachelor-safe LR/MLP/LSTM training pipeline.")
    parser.add_argument("--config", default="configs/experiments/fair_core_v2.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    run_pipeline(
        args.config,
        smoke=args.smoke,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
    )


if __name__ == "__main__":
    main()
