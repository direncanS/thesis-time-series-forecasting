"""Fair-core TFT — single TFT configuration for the thesis core comparison.

Per CLAUDE.md § 4 / § 10 / § 11 / § 11A (Comparability Gate) / § 12 and the
approved plan § B contract:
    - multivariate target: all 7 ETTh1 variables.
    - past-only information regime: time_varying_known_reals = [] (per CLAUDE.md § 11 Fairness Protocol).
    - train-only StandardScaler shared with src/multi_seed.py (admissibility constraint).
    - deterministic point forecast only (no quantile / probabilistic head).
    - same prediction horizon, same patience/max_epochs, same seed handling.
    - same metric semantics as multi_seed.py: scalar np.mean over flattened
      (windows x 24 horizon x 7 variables) after manual inverse-transform.
    - training loss: MultiLoss([MSE()]) — symmetric with multi_seed.py's
      nn.MSELoss (§ 11A item 8 training-loss symmetry). LOSS-SYMMETRY-01
      hard-fail is satisfied only under this symmetric configuration; the
      pre-S-02b MAE-loss variant is superseded and not thesis-valid.

Final thesis sentence (CLAUDE.md):
    "There is only one TFT configuration in this thesis, and it is the
     fair-core multivariate past-only configuration used in the final
     core comparison."
"""

import copy
import os

import lightning as L
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import MultiHorizonMetric, MultiLoss


class MSE(MultiHorizonMetric):
    """Mean squared error — training-loss symmetric with src/multi_seed.py's
    nn.MSELoss (§ 11A item 8 training-loss symmetry). pytorch-forecasting does
    not expose an MSE class, only RMSE (reduction="sqrt-mean"); this subclass
    uses reduction="mean" over the same squared-error element definition."""

    def __init__(self, reduction="mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def loss(self, y_pred, target):
        return torch.pow(self.to_prediction(y_pred) - target, 2)
from sklearn.preprocessing import StandardScaler

# ============================================================
# Locked implementation defaults (CLAUDE.md § 10)
# ============================================================
SEEDS = [42, 123, 456]
INPUT_LEN = 96
OUTPUT_LEN = 24
BATCH_SIZE = 64
LEARNING_RATE = 1e-3  # TFT-specific (CLAUDE.md § 10 + § 11)
MAX_EPOCHS = 200
PATIENCE = 10
WEIGHT_DECAY = 0.0
HIDDEN_SIZE = 16  # pytorch-forecasting TFT default; tunable within the comparable budget rule (§ 11)
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 8

torch.set_float32_matmul_precision("high")  # RTX 5080 Tensor Cores


# ============================================================
# Shared metric helpers — identical semantics to src/multi_seed.py
# ============================================================
def inverse_transform_3d(arr_3d, scaler):
    n_w, t, n_f = arr_3d.shape
    flat = arr_3d.reshape(n_w * t, n_f)
    return scaler.inverse_transform(flat).reshape(n_w, t, n_f)


def compute_scaled_metrics(y_true_scaled, y_pred_scaled):
    mse = float(np.mean((y_true_scaled - y_pred_scaled) ** 2))
    mae = float(np.mean(np.abs(y_true_scaled - y_pred_scaled)))
    return mse, mae


def compute_original_metrics(y_true_orig, y_pred_orig):
    mse = float(np.mean((y_true_orig - y_pred_orig) ** 2))
    mae = float(np.mean(np.abs(y_true_orig - y_pred_orig)))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse


def summarize_results(results, metric_key):
    values = [row[metric_key] for row in results]
    return float(np.mean(values)), float(np.std(values, ddof=1))


# ============================================================
# Per-epoch metrics collector (mirrors multi_seed.py training_curves.csv schema)
# Lightning calls on_train_epoch_end AFTER the validation epoch, so both
# train_loss and val_loss are present in callback_metrics at that point.
# ============================================================
class EpochMetricsCollector(L.pytorch.callbacks.Callback):
    """Collect per-epoch (train_loss, val_loss) rows for tft_training_curves.csv."""

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


# ============================================================
# 0. Load + split + scale — IDENTICAL pipeline to src/multi_seed.py
#    Admissibility constraint (plan § B.2): same StandardScaler semantics.
# ============================================================
df = pd.read_csv("data/ETTh1.csv")
features = df.drop(columns=["date"])
var_names = features.columns.tolist()
n_features = len(var_names)

n = len(features)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

train = features.iloc[:train_end]
val = features.iloc[train_end:val_end]
test = features.iloc[val_end:]

scaler = StandardScaler()
scaler.fit(train)

train_scaled = scaler.transform(train)
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)

# Split-then-window: each split becomes an independent series starting at
# time_idx=0, so no encoder window can cross a split boundary. This mirrors
# src/multi_seed.py's create_windows(train_scaled) / create_windows(val_scaled)
# / create_windows(test_scaled) layout and preserves § 11 Fairness Protocol
# items 4 (same information at prediction time) + 5 (same preprocessing).
# With a concatenated single series, TFT's test-time encoder would otherwise
# read from the val split — giving TFT context the other core models do not
# have. That is explicitly disallowed.
def make_split_df(scaled_arr):
    d = pd.DataFrame(scaled_arr, columns=var_names)
    d["time_idx"] = np.arange(len(d), dtype=np.int64)
    d["group_id"] = "ETTh1"
    return d


train_df = make_split_df(train_scaled)
val_df = make_split_df(val_scaled)
test_df = make_split_df(test_scaled)

print(f"Data: {n} rows, {n_features} variables: {var_names}")
print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
print(
    f"Settings: batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}, "
    f"patience={PATIENCE}, max_epochs={MAX_EPOCHS}, weight_decay={WEIGHT_DECAY}"
)


# ============================================================
# Build TimeSeriesDataSet with identity normalizer
#   - data is pre-scaled with multi_seed.py's StandardScaler
#   - MultiNormalizer of TorchNormalizer(center=0, scale=1) acts as
#     pass-through; pytorch-forecasting does no further normalization.
#   - admissibility check at audit time: see plan § B.2.
# ============================================================
def make_identity_normalizer():
    return MultiNormalizer(
        [TorchNormalizer(method="identity") for _ in range(n_features)]
    )


def build_datasets():
    """Return (training_ds, val_ds, test_ds) with shared identity normalizer.

    Split-then-window: each dataset is built over an independent per-split
    DataFrame (train_df / val_df / test_df), so encoder windows cannot span
    split boundaries — matching src/multi_seed.py and preserving § 11
    Fairness Protocol items 4 + 5 (no cross-split information leakage).
    """
    training_ds = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=var_names,
        group_ids=["group_id"],
        min_encoder_length=INPUT_LEN,
        max_encoder_length=INPUT_LEN,
        min_prediction_length=OUTPUT_LEN,
        max_prediction_length=OUTPUT_LEN,
        time_varying_unknown_reals=var_names,
        time_varying_known_reals=[],
        time_varying_known_categoricals=[],
        target_normalizer=make_identity_normalizer(),
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
        allow_missing_timesteps=False,
    )
    val_ds = TimeSeriesDataSet.from_dataset(
        training_ds, val_df, stop_randomization=True,
    )
    test_ds = TimeSeriesDataSet.from_dataset(
        training_ds, test_df, stop_randomization=True,
    )
    return training_ds, val_ds, test_ds


# ============================================================
# Per-seed training loop
# ============================================================
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

all_results = []
all_history_rows = []
all_importance_rows = []
n_test_windows = None

for seed in SEEDS:
    print("\n" + "=" * 60)
    print(f"SEED {seed}")
    print("=" * 60)

    L.seed_everything(seed, workers=True)

    training_ds, val_ds, test_ds = build_datasets()

    if n_test_windows is None:
        n_test_windows = len(test_ds)

    train_loader = training_ds.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=0
    )
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0
    )
    test_loader = test_ds.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0
    )

    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        loss=MultiLoss([MSE() for _ in range(n_features)]),
        log_interval=0,
        reduce_on_plateau_patience=0,
        weight_decay=WEIGHT_DECAY,
        optimizer="adam",
    )
    n_params = sum(p.numel() for p in tft.parameters())
    print(f"TFT parameters: {n_params / 1000:.1f}k")

    early_stop = L.pytorch.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, mode="min"
    )
    ckpt_path = f"checkpoints/tft_seed{seed}.ckpt"
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"tft_seed{seed}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=False,
    )
    epoch_collector = EpochMetricsCollector()

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint, epoch_collector],
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        deterministic=False,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Parse best epoch from the saved checkpoint metadata. ModelCheckpoint
    # writes the 'epoch' key at save time, so this reports the VAL-OPTIMAL
    # epoch — not trainer.current_epoch (which is the last training epoch).
    best_path = checkpoint.best_model_path
    best_epoch = -1
    if best_path:
        ckpt_data = torch.load(best_path, map_location="cpu", weights_only=False)
        best_epoch = int(ckpt_data.get("epoch", -1))
    print(f"Training complete (seed {seed}).")
    print(f"  Checkpoint saved: {best_path}")

    # Reload best
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_path)
    best_tft.eval()

    # Best epoch from checkpoint metadata if available; otherwise from training_curves
    history = trainer.callback_metrics

    # ---- Predict on test set ----
    print("  Generating predictions...")
    raw_predictions = best_tft.predict(
        test_loader,
        mode="prediction",
        return_y=True,
        trainer_kwargs={"accelerator": "auto", "devices": 1, "logger": False},
    )

    # raw_predictions.output: list of 7 tensors, each (n_test_windows, OUTPUT_LEN)
    # Convert to ndarray (n_test_windows, OUTPUT_LEN, n_features)
    if isinstance(raw_predictions.output, (list, tuple)):
        pred_stack = torch.stack(list(raw_predictions.output), dim=-1).cpu().numpy()
    else:
        pred_stack = raw_predictions.output.cpu().numpy()
    if pred_stack.ndim == 2:
        pred_stack = pred_stack[..., None]
    n_test = pred_stack.shape[0]

    # Ground truth in scaled space (since we passed pre-scaled data + identity normalizer)
    if isinstance(raw_predictions.y[0], (list, tuple)):
        y_true_stack = (
            torch.stack(list(raw_predictions.y[0]), dim=-1).cpu().numpy()
        )
    else:
        y_true_stack = raw_predictions.y[0].cpu().numpy()
    if y_true_stack.ndim == 2:
        y_true_stack = y_true_stack[..., None]

    print(f"  Test samples: {n_test}")
    print(f"  Best epoch: {best_epoch}")

    # ---- Metrics: scaled space (compare to multi_seed.py semantics) ----
    mse_scaled, mae_scaled = compute_scaled_metrics(y_true_stack, pred_stack)
    rmse_scaled = float(np.sqrt(mse_scaled))

    # ---- Metrics: original scale ----
    y_pred_orig = inverse_transform_3d(pred_stack, scaler)
    y_true_orig = inverse_transform_3d(y_true_stack, scaler)
    mse_orig, mae_orig, rmse_orig = compute_original_metrics(y_true_orig, y_pred_orig)

    print(
        f"  Scaled   | MSE: {mse_scaled:.4f}, MAE: {mae_scaled:.4f}, RMSE: {rmse_scaled:.4f}"
    )
    print(
        f"  Original | MSE: {mse_orig:.4f}, MAE: {mae_orig:.4f}, RMSE: {rmse_orig:.4f}"
    )

    all_results.append(
        {
            "seed": seed,
            "n_test_samples": n_test,
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

    # ---- VSN variable importance ----
    print("  Extracting VSN variable importance...")
    interpretation = best_tft.interpret_output(
        best_tft.predict(
            test_loader,
            mode="raw",
            return_x=True,
            trainer_kwargs={"accelerator": "auto", "devices": 1, "logger": False},
        ).output,
        reduction="mean",
    )
    encoder_var_imp = interpretation["encoder_variables"].cpu().numpy()
    encoder_var_names = best_tft.hparams.get("x_reals", var_names)

    # Filter to original var_names (drop relative_time_idx etc. if present)
    for vname, imp in zip(encoder_var_names, encoder_var_imp):
        all_importance_rows.append(
            {"seed": seed, "variable": vname, "importance": float(imp)}
        )

    # ---- Training curves: per-epoch (train_loss, val_loss) records ----
    # Schema matches src/multi_seed.py results/training_curves.csv for
    # cross-model comparability (SUSPECT-06 verification + overfitting plots).
    for rec in epoch_collector.records:
        all_history_rows.append(
            {
                "model": "TFT",
                "seed": seed,
                "epoch": rec["epoch"],
                "train_loss": rec["train_loss"],
                "val_loss": rec["val_loss"],
                "is_best_epoch": (rec["epoch"] == best_epoch),
            }
        )


# ============================================================
# Aggregate + report
# ============================================================
print("\n" + "=" * 72)
print("TFT 3-SEED RESULTS (fair-core: multivariate, past-only, identity-normalized)")
print("=" * 72)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

mse_scaled_mean, mse_scaled_std = summarize_results(all_results, "mse_scaled")
mae_scaled_mean, mae_scaled_std = summarize_results(all_results, "mae_scaled")
rmse_scaled_mean, rmse_scaled_std = summarize_results(all_results, "rmse_scaled")

mse_orig_mean, mse_orig_std = summarize_results(all_results, "mse_original")
mae_orig_mean, mae_orig_std = summarize_results(all_results, "mae_original")
rmse_orig_mean, rmse_orig_std = summarize_results(all_results, "rmse_original")

best_epochs = [row["best_epoch"] for row in all_results]
n_params = all_results[0]["n_params"]

print("\nScaled-space test metrics")
print(f"{'Model':<12} {'MSE':>22} {'MAE':>22} {'RMSE':>22}")
print("-" * 80)
print(
    f"{'TFT':<12} {mse_scaled_mean:>10.4f} +/- {mse_scaled_std:.4f}  "
    f"{mae_scaled_mean:>10.4f} +/- {mae_scaled_std:.4f}  "
    f"{rmse_scaled_mean:>10.4f} +/- {rmse_scaled_std:.4f}"
)

print("\nOriginal-scale test metrics")
print(f"{'Model':<12} {'MSE':>22} {'MAE':>22} {'RMSE':>22}")
print("-" * 80)
print(
    f"{'TFT':<12} {mse_orig_mean:>10.4f} +/- {mse_orig_std:.4f}  "
    f"{mae_orig_mean:>10.4f} +/- {mae_orig_std:.4f}  "
    f"{rmse_orig_mean:>10.4f} +/- {rmse_orig_std:.4f}"
)

print("\nRun metadata")
print(f"Seeds used: {SEEDS}")
print(f"Best epochs: {best_epochs}")
print(f"TFT parameters: {n_params}")
print(
    f"Settings: batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}, "
    f"patience={PATIENCE}, max_epochs={MAX_EPOCHS}, weight_decay={WEIGHT_DECAY}"
)
print("Target scope: all 7 ETTh1 variables (multivariate).")
print("Information regime: past-only at prediction time — § 11 Fairness Protocol.")
print("Preprocessing: train-only StandardScaler shared with src/multi_seed.py.")
print("Forecast type: deterministic point forecast.")
print("Aggregation (§ 15): scalar np.mean over flattened (windows x 24 horizon x 7 variables);")
print("    seed axis: mean ± std with ddof=1.")

# ============================================================
# Save artifacts
# ============================================================
metrics_df = pd.DataFrame(all_results)
metrics_df.to_csv("results/tft_metrics.csv", index=False)
print("\nMetrics saved to results/tft_metrics.csv")

summary_row = {
    "model": "TFT",
    "seeds": ",".join(str(s) for s in SEEDS),
    "deterministic": False,
    "best_epochs": ",".join(str(e) for e in best_epochs),
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "patience": PATIENCE,
    "max_epochs": MAX_EPOCHS,
    "weight_decay": WEIGHT_DECAY,
    "n_params": n_params,
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
pd.DataFrame([summary_row]).to_csv("results/tft_summary.csv", index=False)
print("Summary saved to results/tft_summary.csv")

pd.DataFrame(all_history_rows).to_csv(
    "results/tft_training_curves.csv", index=False
)
print("Training curves saved to results/tft_training_curves.csv")

importance_df = pd.DataFrame(all_importance_rows)
importance_df.to_csv("results/tft_importance.csv", index=False)

# Mean importance across seeds for the original 7 variables
mean_imp = (
    importance_df[importance_df["variable"].isin(var_names)]
    .groupby("variable")["importance"]
    .mean()
    .sort_values(ascending=False)
)
print("\nFull variable importance saved to results/tft_importance.csv")
print("\nTFT VSN Variable Importance (mean across seeds):")
for i, (vname, imp) in enumerate(mean_imp.items(), 1):
    print(f"  #{i} {vname}: {imp:.4f}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print("\nExpected outputs:")
print("  results/tft_metrics.csv         — 3-seed TFT metrics (multivariate, original-scale)")
print("  results/tft_summary.csv         — TFT summary + run metadata")
print("  results/tft_training_curves.csv — TFT per-seed best epoch + best val_loss")
print("  results/tft_importance.csv      — TFT VSN variable importance")
