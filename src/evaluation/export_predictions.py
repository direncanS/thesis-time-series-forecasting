import argparse

import lightning as L
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer
from sklearn.linear_model import LinearRegression

from src.common import (
    MLP,
    LSTMModel,
    UnifiedMSE,
    apply_runtime_overrides,
    ensure_dir,
    inverse_transform_3d,
    load_config,
    make_split_frame,
    prepare_supervised_data,
)


class MSE(UnifiedMSE):
    """Module-level alias for legacy/new TFT checkpoint pickle compatibility."""


def make_identity_normalizer(n_features):
    return MultiNormalizer([TorchNormalizer(method="identity") for _ in range(n_features)])


def build_test_loader(data, cfg):
    train_df = make_split_frame(data["train_scaled"], data["var_names"])
    test_df = make_split_frame(data["test_scaled"], data["var_names"])
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
        target_normalizer=make_identity_normalizer(len(data["var_names"])),
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
        allow_missing_timesteps=False,
    )
    test_ds = TimeSeriesDataSet.from_dataset(training_ds, test_df, stop_randomization=True)
    return test_ds.to_dataloader(train=False, batch_size=cfg["batch_size"], num_workers=0)


def run_export(config_path=None, *, results_dir=None, checkpoints_dir=None):
    cfg = apply_runtime_overrides(
        load_config(config_path),
        smoke=False,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
    )
    data = prepare_supervised_data(cfg["data_path"], cfg["input_len"], cfg["output_len"])
    results_path = ensure_dir(cfg["results_dir"])
    checkpoints_path = ensure_dir(cfg["checkpoints_dir"])
    n_features = len(data["var_names"])

    lr_model = LinearRegression()
    lr_model.fit(data["X_train_flat"], data["y_train_flat"])
    preds_lr_scaled = lr_model.predict(data["X_test_flat"]).reshape(data["n_test"], cfg["output_len"], n_features)
    np.save(results_path / "preds_lr.npy", inverse_transform_3d(preds_lr_scaled, data["scaler"]))

    X_test_flat_tensor = torch.FloatTensor(data["X_test_flat"])
    X_test_tensor = torch.FloatTensor(data["X_test"])

    for seed in cfg["seeds"]:
        mlp = MLP(data["input_size"], data["output_size"], hidden_size=cfg["models"]["mlp"]["hidden_size"])
        mlp.load_state_dict(
            torch.load(checkpoints_path / f"mlp_seed{seed}.pt", map_location="cpu", weights_only=True)
        )
        mlp.eval()
        with torch.no_grad():
            preds_flat = mlp(X_test_flat_tensor).numpy()
        preds_scaled = preds_flat.reshape(data["n_test"], cfg["output_len"], n_features)
        np.save(results_path / f"preds_mlp_seed{seed}.npy", inverse_transform_3d(preds_scaled, data["scaler"]))

        lstm = LSTMModel(
            input_size=n_features,
            hidden_size=cfg["models"]["lstm"]["hidden_size"],
            output_size=data["output_size"],
            num_layers=cfg["models"]["lstm"]["num_layers"],
        )
        lstm.load_state_dict(
            torch.load(checkpoints_path / f"lstm_seed{seed}.pt", map_location="cpu", weights_only=True)
        )
        lstm.eval()
        with torch.no_grad():
            preds_flat = lstm(X_test_tensor).numpy()
        preds_scaled = preds_flat.reshape(data["n_test"], cfg["output_len"], n_features)
        np.save(results_path / f"preds_lstm_seed{seed}.npy", inverse_transform_3d(preds_scaled, data["scaler"]))

    test_loader = build_test_loader(data, cfg)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    for seed in cfg["seeds"]:
        L.seed_everything(seed, workers=True)
        raw = TemporalFusionTransformer.load_from_checkpoint(checkpoints_path / f"tft_seed{seed}.ckpt").predict(
            test_loader,
            mode="prediction",
            return_y=False,
            trainer_kwargs={"accelerator": accelerator, "devices": 1, "logger": False},
        )
        if isinstance(raw, (list, tuple)):
            pred_stack = torch.stack(list(raw), dim=-1).cpu().numpy()
        else:
            pred_stack = raw.cpu().numpy()
        if pred_stack.ndim == 2:
            pred_stack = pred_stack[..., None]
        np.save(results_path / f"preds_tft_seed{seed}.npy", inverse_transform_3d(pred_stack, data["scaler"]))


def build_parser():
    parser = argparse.ArgumentParser(description="Export aligned prediction tensors.")
    parser.add_argument("--config", default="configs/experiments/fair_core_v2.yaml")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    run_export(args.config, results_dir=args.results_dir, checkpoints_dir=args.checkpoints_dir)


if __name__ == "__main__":
    main()
