"""Post-training analysis on frozen prediction tensors (S-09c merge).

Consolidates three previously-separate scripts into one entry point:
    1. per-seed metric export        → results/per_seed_metrics.csv
    2. § 14 complexity metrics       → results/complexity_metrics.csv
    3. § 16 paired bootstrap CIs     → results/bootstrap_intervals.csv

Output CSV paths and contents are preserved byte-equivalent to the legacy
scripts (per_seed_metrics_export.py, complexity_report.py,
uncertainty_bootstrap.py) so existing prose / VALIDATION_LOG references
remain valid.

§ 16 mandatory safety sentence (CLAUDE.md):
    "Interval estimates are used as comparative uncertainty evidence, not as a
     claim of strict independent-sample inference."

Test-window independence warning (CLAUDE.md § 16): ETTh1 windows are stride-1
sliding; the standard percentile bootstrap underestimates true sampling
variance. Documented as Internal-validity threat in methodology + discussion.

Run:
    python src/evaluation/post_training_analysis.py
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ============================================================
# Constants — same as the locked § 10 defaults
# ============================================================
SEEDS = [42, 123, 456]
INPUT_LEN = 96
OUTPUT_LEN = 24
N_FEATURES = 7
BOOTSTRAP_SEED = 2026
N_BOOTSTRAP = 10_000
MODELS_ORDERED = ["LR", "MLP", "LSTM", "TFT"]
HARDWARE_NOTE = "single NVIDIA RTX 5080 Laptop GPU (CUDA), batch_size = 64"


# ============================================================
# Shared helpers — identical semantics to training/multi_seed.py
# ============================================================
def create_windows(data, input_len=INPUT_LEN, output_len=OUTPUT_LEN):
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len])
    return np.array(X), np.array(y)


def inverse_transform_3d(arr_3d, scaler):
    n_w, t, n_f = arr_3d.shape
    flat = arr_3d.reshape(n_w * t, n_f)
    return scaler.inverse_transform(flat).reshape(n_w, t, n_f)


def compute_original_metrics(y_true_orig, y_pred_orig):
    mse = float(np.mean((y_true_orig - y_pred_orig) ** 2))
    mae = float(np.mean(np.abs(y_true_orig - y_pred_orig)))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse


# ============================================================
# Model class definitions — architecture-only (for parameter count);
# byte-identical to training/multi_seed.py
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


# ============================================================
# Section 0 — shared data reconstruction
# ============================================================
def rebuild_y_test_orig():
    df = pd.read_csv("data/ETTh1.csv").drop(columns=["date"])
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train = df.iloc[:train_end]
    test = df.iloc[val_end:]
    scaler = StandardScaler().fit(train)
    test_scaled = scaler.transform(test)
    _, y_test = create_windows(test_scaled)
    y_test_orig = inverse_transform_3d(y_test, scaler)
    assert y_test_orig.shape == (3365, OUTPUT_LEN, N_FEATURES), \
        f"unexpected y_test shape: {y_test_orig.shape}"
    return y_test_orig


# ============================================================
# Section 1 — per-seed metrics export
# ============================================================
def export_per_seed_metrics(y_test_orig):
    rows = []
    preds_lr = np.load("results/preds_lr.npy")
    mse, mae, rmse = compute_original_metrics(y_test_orig, preds_lr)
    rows.append({"model": "LR", "seed": "deterministic",
                 "mse_original": mse, "mae_original": mae, "rmse_original": rmse})
    for model_name in ("mlp", "lstm", "tft"):
        for seed in SEEDS:
            preds = np.load(f"results/preds_{model_name}_seed{seed}.npy")
            mse, mae, rmse = compute_original_metrics(y_test_orig, preds)
            rows.append({"model": model_name.upper(), "seed": seed,
                         "mse_original": mse, "mae_original": mae,
                         "rmse_original": rmse})
    out_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out_df.to_csv("results/per_seed_metrics.csv", index=False)
    return out_df


# ============================================================
# Section 2 — § 14 complexity metrics
# ============================================================
def export_complexity_metrics():
    input_flat = INPUT_LEN * N_FEATURES
    output_flat = OUTPUT_LEN * N_FEATURES
    lr_n_params = output_flat * input_flat + output_flat

    mlp_model = MLP(input_size=input_flat, output_size=output_flat)
    mlp_n_params = sum(p.numel() for p in mlp_model.parameters())

    lstm_model = LSTMModel(input_size=N_FEATURES, hidden_size=64, output_size=output_flat)
    lstm_n_params = sum(p.numel() for p in lstm_model.parameters())

    tft_metrics = pd.read_csv("results/tft_metrics.csv")
    tft_n_params = int(tft_metrics["n_params"].iloc[0])

    categories = {
        "LR": "linear",
        "MLP": "shallow-MLP",
        "LSTM": "recurrent",
        "TFT": "transformer-family",
    }

    tft_curves = pd.read_csv("results/tft_training_curves.csv")
    tft_last_epoch_per_seed = tft_curves.groupby("seed")["epoch"].max().to_dict()
    tft_sec_per_epoch = 15
    tft_per_seed_sec = {
        int(seed): (int(last_ep) + 1) * tft_sec_per_epoch
        for seed, last_ep in tft_last_epoch_per_seed.items()
    }
    tft_total_sec = sum(tft_per_seed_sec.values())
    tft_wallclock_note = (
        f"best-effort estimate ~{tft_total_sec} sec total across 3 seeds "
        f"(per-seed {tft_per_seed_sec}; ~{tft_sec_per_epoch} sec/epoch × epoch counts from training_curves.csv)"
    )

    wallclock = {
        "LR": "< 1 sec (deterministic closed-form OLS; no iterative training)",
        "MLP": (
            "not instrumented (§ 14 secondary — src/training/multi_seed.py does not wrap the "
            "training loop with time.perf_counter(); training_curves.csv captures "
            "epoch progression but not wall-clock seconds)"
        ),
        "LSTM": (
            "not instrumented (§ 14 secondary — same instrumentation gap as MLP)"
        ),
        "TFT": tft_wallclock_note,
    }

    n_params_by_model = {
        "LR": lr_n_params, "MLP": mlp_n_params,
        "LSTM": lstm_n_params, "TFT": tft_n_params,
    }
    rows = [
        {"model": m,
         "n_params": n_params_by_model[m],
         "architectural_category": categories[m],
         "wall_clock_note": wallclock[m],
         "hardware_note": HARDWARE_NOTE}
        for m in MODELS_ORDERED
    ]
    out_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out_df.to_csv("results/complexity_metrics.csv", index=False)
    return out_df


# ============================================================
# Section 3 — § 16 paired bootstrap CIs
# ============================================================
def export_bootstrap_intervals(y_test_orig):
    preds = {"LR": np.load("results/preds_lr.npy")}
    for m in ["mlp", "lstm", "tft"]:
        stack = np.stack(
            [np.load(f"results/preds_{m}_seed{s}.npy") for s in SEEDS], axis=0
        )
        preds[m.upper()] = stack.mean(axis=0)
    for name, p in preds.items():
        assert p.shape == y_test_orig.shape, f"{name} shape mismatch"

    def per_window_errors(y_true, y_pred):
        diff = y_true - y_pred
        n_w = diff.shape[0]
        flat = diff.reshape(n_w, -1)
        return (flat ** 2).mean(axis=1), np.abs(flat).mean(axis=1)

    errors = {name: dict(zip(("sq", "abs"), per_window_errors(y_test_orig, p)))
              for name, p in preds.items()}

    n_test = y_test_orig.shape[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_indices = rng.integers(0, n_test, size=(N_BOOTSTRAP, n_test), dtype=np.int32)

    boot_stats = {}
    for name, err in errors.items():
        sq_resampled = err["sq"][boot_indices]
        mse_dist = sq_resampled.mean(axis=1)
        del sq_resampled
        abs_resampled = err["abs"][boot_indices]
        mae_dist = abs_resampled.mean(axis=1)
        del abs_resampled
        boot_stats[name] = {"mse": mse_dist, "mae": mae_dist, "rmse": np.sqrt(mse_dist)}

    pairs = [(MODELS_ORDERED[i], MODELS_ORDERED[j])
             for i in range(len(MODELS_ORDERED))
             for j in range(i + 1, len(MODELS_ORDERED))]

    rows = []
    for a, b in pairs:
        for metric in ["mse", "mae", "rmse"]:
            diff = boot_stats[a][metric] - boot_stats[b][metric]
            rows.append({
                "model_pair": f"{a}_vs_{b}",
                "metric": metric.upper(),
                "mean_diff": round(float(diff.mean()), 6),
                "ci_low": round(float(np.percentile(diff, 2.5)), 6),
                "ci_high": round(float(np.percentile(diff, 97.5)), 6),
                "bootstrap_n": N_BOOTSTRAP,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "n_test_windows": n_test,
            })
    out_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out_df.to_csv("results/bootstrap_intervals.csv", index=False)
    return out_df


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Post-training analysis (S-09c merged)")
    print("=" * 70)

    print("\n[1/3] Per-seed metrics → results/per_seed_metrics.csv")
    y_test_orig = rebuild_y_test_orig()
    per_seed = export_per_seed_metrics(y_test_orig)
    print(per_seed.to_string(index=False))

    print("\n[2/3] § 14 complexity metrics → results/complexity_metrics.csv")
    complexity = export_complexity_metrics()
    print(complexity[["model", "n_params", "architectural_category"]].to_string(index=False))

    print("\n[3/3] § 16 paired-bootstrap intervals → results/bootstrap_intervals.csv")
    boot = export_bootstrap_intervals(y_test_orig)
    significant = boot[(boot["ci_low"] > 0) | (boot["ci_high"] < 0)]
    print(f"  {len(boot)} intervals total, {len(significant)} with CI excluding zero")

    print("\nDone. § 16 reminder: intervals are comparative uncertainty evidence, not strict independent-sample inference.")
