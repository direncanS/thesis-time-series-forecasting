import argparse
import time

import lightning as L
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer

from src.common import (
    UnifiedMSE,
    apply_runtime_overrides,
    compute_original_metrics,
    compute_scaled_metrics,
    configure_stochastic_runtime,
    ensure_dir,
    load_config,
    make_split_frame,
    prepare_supervised_data,
    runtime_flag_block,
    summarize_results,
    upsert_runtime_rows,
)


class MSE(UnifiedMSE):
    """Module-level alias for checkpoint pickle compatibility."""


class EpochMetricsCollector(L.pytorch.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.records = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss_epoch", metrics.get("train_loss"))
        val_loss = metrics.get("val_loss")
        self.records.append(
            {
                "epoch": int(trainer.current_epoch),
                "train_loss": float(train_loss) if train_loss is not None else None,
                "val_loss": float(val_loss) if val_loss is not None else None,
            }
        )


def make_identity_normalizer(n_features):
    return MultiNormalizer([TorchNormalizer(method="identity") for _ in range(n_features)])


def build_datasets(data, cfg):
    train_df = make_split_frame(data["train_scaled"], data["var_names"])
    val_df = make_split_frame(data["val_scaled"], data["var_names"])
    test_df = make_split_frame(data["test_scaled"], data["var_names"])
    n_features = len(data["var_names"])

    training_ds = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=data["var_names"],
        group_ids=["group_id"],
        min_encoder_length=cfg["input_len"],
        max_encoder_length=cfg["input_len"],
        min_prediction_length=cfg["output_len"],
        max_prediction_length=cfg["output_len"],
        time_varying_unknown_reals=data["var_names"],
        time_varying_known_reals=[],
        time_varying_known_categoricals=[],
        target_normalizer=make_identity_normalizer(n_features),
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
        allow_missing_timesteps=False,
    )
    val_ds = TimeSeriesDataSet.from_dataset(training_ds, val_df, stop_randomization=True)
    test_ds = TimeSeriesDataSet.from_dataset(training_ds, test_df, stop_randomization=True)
    return training_ds, val_ds, test_ds


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
    tft_cfg = cfg["models"]["tft"]
    device = configure_stochastic_runtime(cfg)
    print(f"[v2] {runtime_flag_block(device)}")

    all_results = []
    all_history_rows = []
    all_importance_rows = []
    runtime_rows = []
    n_features = len(data["var_names"])

    for seed in cfg["seeds"]:
        print(f"\n[TFT] seed={seed}")
        seed_start = time.perf_counter()
        L.seed_everything(seed, workers=True)

        training_ds, val_ds, test_ds = build_datasets(data, cfg)
        train_loader = training_ds.to_dataloader(train=True, batch_size=cfg["batch_size"], num_workers=0)
        val_loader = val_ds.to_dataloader(train=False, batch_size=cfg["batch_size"], num_workers=0)
        test_loader = test_ds.to_dataloader(train=False, batch_size=cfg["batch_size"], num_workers=0)

        tft = TemporalFusionTransformer.from_dataset(
            training_ds,
            learning_rate=tft_cfg["learning_rate"],
            hidden_size=tft_cfg["hidden_size"],
            attention_head_size=tft_cfg["attention_head_size"],
            dropout=tft_cfg["dropout"],
            hidden_continuous_size=tft_cfg["hidden_continuous_size"],
            loss=MSE(),
            log_interval=0,
            reduce_on_plateau_patience=0,
            weight_decay=tft_cfg["weight_decay"],
            optimizer="adam",
        )
        n_params = sum(p.numel() for p in tft.parameters())

        ckpt_dir = ensure_dir(checkpoints_path)
        ckpt_path = ckpt_dir / f"tft_seed{seed}.ckpt"
        if ckpt_path.exists():
            ckpt_path.unlink()

        early_stop = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"], mode="min")
        checkpoint = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"tft_seed{seed}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=False,
        )
        epoch_collector = EpochMetricsCollector()

        trainer = L.Trainer(
            max_epochs=cfg["max_epochs"],
            accelerator="gpu" if device.type == "cuda" else "cpu",
            devices=1,
            gradient_clip_val=0.1,
            callbacks=[early_stop, checkpoint, epoch_collector],
            enable_progress_bar=not smoke,
            enable_model_summary=False,
            logger=False,
            deterministic=cfg["precision"]["cudnn_deterministic"],
        )
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_path = checkpoint.best_model_path
        if not best_path:
            raise RuntimeError(f"TFT seed={seed}: no checkpoint path was produced.")

        ckpt_data = torch.load(best_path, map_location="cpu", weights_only=False)
        best_epoch = int(ckpt_data.get("epoch", -1))
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_path)
        best_tft.eval()

        raw_predictions = best_tft.predict(
            test_loader,
            mode="prediction",
            return_y=True,
            trainer_kwargs={"accelerator": "gpu" if device.type == "cuda" else "cpu", "devices": 1, "logger": False},
        )
        if isinstance(raw_predictions.output, (list, tuple)):
            pred_stack = torch.stack(list(raw_predictions.output), dim=-1).cpu().numpy()
        else:
            pred_stack = raw_predictions.output.cpu().numpy()
        if pred_stack.ndim == 2:
            pred_stack = pred_stack[..., None]

        if isinstance(raw_predictions.y[0], (list, tuple)):
            y_true_stack = torch.stack(list(raw_predictions.y[0]), dim=-1).cpu().numpy()
        else:
            y_true_stack = raw_predictions.y[0].cpu().numpy()
        if y_true_stack.ndim == 2:
            y_true_stack = y_true_stack[..., None]

        y_pred_orig = data["scaler"].inverse_transform(pred_stack.reshape(-1, n_features)).reshape(pred_stack.shape)
        y_true_orig = data["scaler"].inverse_transform(y_true_stack.reshape(-1, n_features)).reshape(y_true_stack.shape)

        mse_scaled, mae_scaled = compute_scaled_metrics(y_true_stack, pred_stack)
        rmse_scaled = float(np.sqrt(mse_scaled))
        mse_orig, mae_orig, rmse_orig = compute_original_metrics(y_true_orig, y_pred_orig)

        all_results.append(
            {
                "seed": seed,
                "n_test_samples": pred_stack.shape[0],
                "best_epoch": best_epoch,
                "checkpoint_path": best_path,
                "n_params": int(n_params),
                "mse_scaled": mse_scaled,
                "mae_scaled": mae_scaled,
                "rmse_scaled": rmse_scaled,
                "mse_original": mse_orig,
                "mae_original": mae_orig,
                "rmse_original": rmse_orig,
            }
        )

        interpretation = best_tft.interpret_output(
            best_tft.predict(
                test_loader,
                mode="raw",
                return_x=True,
                trainer_kwargs={"accelerator": "gpu" if device.type == "cuda" else "cpu", "devices": 1, "logger": False},
            ).output,
            reduction="mean",
        )
        encoder_var_imp = interpretation["encoder_variables"].cpu().numpy()
        encoder_var_names = best_tft.hparams.get("x_reals", data["var_names"])
        for vname, imp in zip(encoder_var_names, encoder_var_imp):
            all_importance_rows.append({"seed": seed, "variable": vname, "importance": float(imp)})

        for rec in epoch_collector.records:
            all_history_rows.append(
                {
                    "model": "TFT",
                    "seed": seed,
                    "epoch": rec["epoch"],
                    "train_loss": rec["train_loss"],
                    "val_loss": rec["val_loss"],
                    "is_best_epoch": rec["epoch"] == best_epoch,
                }
            )

        runtime_rows.append(
            {
                "model": "TFT",
                "seed": seed,
                "wall_clock_seconds": round(time.perf_counter() - seed_start, 6),
            }
        )

    metrics_df = pd.DataFrame(all_results)
    metrics_df.to_csv(results_path / "tft_metrics.csv", index=False)

    best_epochs = [row["best_epoch"] for row in all_results]
    mse_scaled_mean, mse_scaled_std = summarize_results(all_results, "mse_scaled")
    mae_scaled_mean, mae_scaled_std = summarize_results(all_results, "mae_scaled")
    rmse_scaled_mean, rmse_scaled_std = summarize_results(all_results, "rmse_scaled")
    mse_orig_mean, mse_orig_std = summarize_results(all_results, "mse_original")
    mae_orig_mean, mae_orig_std = summarize_results(all_results, "mae_original")
    rmse_orig_mean, rmse_orig_std = summarize_results(all_results, "rmse_original")

    pd.DataFrame(
        [
            {
                "model": "TFT",
                "seeds": ",".join(str(s) for s in cfg["seeds"]),
                "deterministic": False,
                "best_epochs": ",".join(str(e) for e in best_epochs),
                "batch_size": cfg["batch_size"],
                "learning_rate": tft_cfg["learning_rate"],
                "patience": cfg["patience"],
                "max_epochs": cfg["max_epochs"],
                "weight_decay": tft_cfg["weight_decay"],
                "n_params": all_results[0]["n_params"],
                "mse_scaled_mean": round(mse_scaled_mean, 6),
                "mse_scaled_std": round(mse_scaled_std, 6),
                "mae_scaled_mean": round(mae_scaled_mean, 6),
                "mae_scaled_std": round(mae_scaled_std, 6),
                "rmse_scaled_mean": round(rmse_scaled_mean, 6),
                "rmse_scaled_std": round(rmse_scaled_std, 6),
                "mse_original_mean": round(mse_orig_mean, 6),
                "mse_original_std": round(mse_orig_std, 6),
                "mae_original_mean": round(mae_orig_mean, 6),
                "mae_original_std": round(mae_orig_std, 6),
                "rmse_original_mean": round(rmse_orig_mean, 6),
                "rmse_original_std": round(rmse_orig_std, 6),
            }
        ]
    ).to_csv(results_path / "tft_summary.csv", index=False)

    pd.DataFrame(all_history_rows).to_csv(results_path / "tft_training_curves.csv", index=False)
    pd.DataFrame(all_importance_rows).to_csv(results_path / "tft_importance.csv", index=False)
    upsert_runtime_rows(results_path / "runtime_seconds.csv", runtime_rows)

    return {
        "config": cfg,
        "results_dir": results_path,
        "checkpoints_dir": checkpoints_path,
        "metrics": metrics_df,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Bachelor-safe TFT training pipeline.")
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
