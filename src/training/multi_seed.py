import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Fair-Baseline Multi-Seed Experiment Pipeline
# ============================================================

SEEDS = [42, 123, 456]
INPUT_LEN = 96
OUTPUT_LEN = 24
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MAX_EPOCHS = 200
PATIENCE = 10
WEIGHT_DECAY = 0.0


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


def compute_scaled_metrics(y_true_scaled, y_pred_scaled):
    mse = float(np.mean((y_true_scaled - y_pred_scaled) ** 2))
    mae = float(np.mean(np.abs(y_true_scaled - y_pred_scaled)))
    return mse, mae


def compute_original_metrics(y_true_orig, y_pred_orig):
    mse = float(np.mean((y_true_orig - y_pred_orig) ** 2))
    mae = float(np.mean(np.abs(y_true_orig - y_pred_orig)))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse


def finalize_history_rows(history_rows, best_epoch):
    finalized = []
    for row in history_rows:
        new_row = row.copy()
        new_row["is_best_epoch"] = row["epoch"] == best_epoch
        finalized.append(new_row)
    return finalized


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


def train_with_early_stopping(
    model_name,
    model,
    X_train_tensor,
    y_train_tensor,
    X_val_tensor,
    y_val_tensor,
    X_test_tensor,
    seed,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    best_val_loss = float("inf")
    best_epoch = 1
    best_model_state = None
    patience_counter = 0
    history_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
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

        if epoch % 5 == 0:
            print(
                f"    Epoch {epoch:>3} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}")
                break

    if best_model_state is None:
        raise RuntimeError(
            f"{model_name} seed={seed}: training produced no finite val_loss "
            f"in {epoch} epochs; no best checkpoint recorded, cannot restore."
        )

    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        y_pred_flat = model(X_test_tensor).numpy()

    print(f"    Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    history_rows = finalize_history_rows(history_rows, best_epoch)
    return y_pred_flat, best_epoch, history_rows


# ============================================================
# 0. Load and preprocess data
# ============================================================
df = pd.read_csv("data/ETTh1.csv")
features = df.drop(columns=["date"])
var_names = features.columns.tolist()

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

X_train, y_train = create_windows(train_scaled)
X_val, y_val = create_windows(val_scaled)
X_test, y_test = create_windows(test_scaled)

n_train = X_train.shape[0]
n_val = X_val.shape[0]
n_test = X_test.shape[0]

X_train_flat = X_train.reshape(n_train, -1)
X_val_flat = X_val.reshape(n_val, -1)
X_test_flat = X_test.reshape(n_test, -1)
y_train_flat = y_train.reshape(n_train, -1)
y_val_flat = y_val.reshape(n_val, -1)
y_test_flat = y_test.reshape(n_test, -1)

input_size = X_train_flat.shape[1]  # 672
output_size = y_train_flat.shape[1]  # 168

y_test_orig = inverse_transform_3d(y_test, scaler)

print(f"Veri: {n} satir, {len(var_names)} degisken")
print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
print(f"Pencereler: train={n_train}, val={n_val}, test={n_test}")
print(
    f"Ayarlar: batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, "
    f"patience={PATIENCE}, max_epochs={MAX_EPOCHS}, weight_decay={WEIGHT_DECAY}"
)

# ============================================================
# 1. Deterministic Linear Regression
# ============================================================
print("\n" + "=" * 72)
print("LINEAR REGRESSION (deterministic, single run)")
print("=" * 72)

lr_model = LinearRegression()
lr_model.fit(X_train_flat, y_train_flat)

y_pred_lr_scaled = lr_model.predict(X_test_flat).reshape(n_test, OUTPUT_LEN, len(var_names))
y_pred_lr_orig = inverse_transform_3d(y_pred_lr_scaled, scaler)

lr_mse_scaled, lr_mae_scaled = compute_scaled_metrics(y_test, y_pred_lr_scaled)
lr_mse_orig, lr_mae_orig, lr_rmse_orig = compute_original_metrics(y_test_orig, y_pred_lr_orig)

print(
    f"  Scaled   | MSE: {lr_mse_scaled:.4f}, MAE: {lr_mae_scaled:.4f}\n"
    f"  Original | MSE: {lr_mse_orig:.4f}, MAE: {lr_mae_orig:.4f}, RMSE: {lr_rmse_orig:.4f}"
)

# ============================================================
# 2. Multi-seed MLP
# ============================================================
print("\n" + "=" * 72)
print(f"MLP - {len(SEEDS)} Seeds: {SEEDS}")
print("=" * 72)

os.makedirs("checkpoints", exist_ok=True)

training_curve_rows = []
mlp_results = []

for seed in SEEDS:
    print(f"  Seed {seed} egitiliyor...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(input_size, output_size)
    y_pred_flat, best_epoch, history_rows = train_with_early_stopping(
        model_name="MLP",
        model=model,
        X_train_tensor=torch.FloatTensor(X_train_flat),
        y_train_tensor=torch.FloatTensor(y_train_flat),
        X_val_tensor=torch.FloatTensor(X_val_flat),
        y_val_tensor=torch.FloatTensor(y_val_flat),
        X_test_tensor=torch.FloatTensor(X_test_flat),
        seed=seed,
    )
    training_curve_rows.extend(history_rows)

    # Save best model checkpoint
    torch.save(model.state_dict(), f"checkpoints/mlp_seed{seed}.pt")

    y_pred_scaled = y_pred_flat.reshape(n_test, OUTPUT_LEN, len(var_names))
    y_pred_orig = inverse_transform_3d(y_pred_scaled, scaler)

    mse_scaled, mae_scaled = compute_scaled_metrics(y_test, y_pred_scaled)
    mse_orig, mae_orig, rmse_orig = compute_original_metrics(y_test_orig, y_pred_orig)

    mlp_results.append(
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "mse_scaled": mse_scaled,
            "mae_scaled": mae_scaled,
            "mse_original": mse_orig,
            "mae_original": mae_orig,
            "rmse_original": rmse_orig,
        }
    )
    print(
        f"    Scaled   | MSE: {mse_scaled:.4f}, MAE: {mae_scaled:.4f}\n"
        f"    Original | MSE: {mse_orig:.4f}, MAE: {mae_orig:.4f}, RMSE: {rmse_orig:.4f}"
    )

# ============================================================
# 3. Multi-seed LSTM
# ============================================================
print("\n" + "=" * 72)
print(f"LSTM - {len(SEEDS)} Seeds: {SEEDS}")
print("=" * 72)

lstm_results = []

for seed in SEEDS:
    print(f"  Seed {seed} egitiliyor...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LSTMModel(input_size=7, hidden_size=64, output_size=output_size)
    y_pred_flat, best_epoch, history_rows = train_with_early_stopping(
        model_name="LSTM",
        model=model,
        X_train_tensor=torch.FloatTensor(X_train),
        y_train_tensor=torch.FloatTensor(y_train_flat),
        X_val_tensor=torch.FloatTensor(X_val),
        y_val_tensor=torch.FloatTensor(y_val_flat),
        X_test_tensor=torch.FloatTensor(X_test),
        seed=seed,
    )
    training_curve_rows.extend(history_rows)

    # Save best model checkpoint
    torch.save(model.state_dict(), f"checkpoints/lstm_seed{seed}.pt")

    y_pred_scaled = y_pred_flat.reshape(n_test, OUTPUT_LEN, len(var_names))
    y_pred_orig = inverse_transform_3d(y_pred_scaled, scaler)

    mse_scaled, mae_scaled = compute_scaled_metrics(y_test, y_pred_scaled)
    mse_orig, mae_orig, rmse_orig = compute_original_metrics(y_test_orig, y_pred_orig)

    lstm_results.append(
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "mse_scaled": mse_scaled,
            "mae_scaled": mae_scaled,
            "mse_original": mse_orig,
            "mae_original": mae_orig,
            "rmse_original": rmse_orig,
        }
    )
    print(
        f"    Scaled   | MSE: {mse_scaled:.4f}, MAE: {mae_scaled:.4f}\n"
        f"    Original | MSE: {mse_orig:.4f}, MAE: {mae_orig:.4f}, RMSE: {rmse_orig:.4f}"
    )


def summarize_results(results, metric_key):
    values = [row[metric_key] for row in results]
    return float(np.mean(values)), float(np.std(values, ddof=1))


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

print("\n" + "=" * 84)
print("MULTI-SEED SUMMARY (fair baseline, validation-based early stopping)")
print("=" * 84)

print("\nScaled-space test metrics")
print(f"{'Model':<20} {'MSE':>20} {'MAE':>20}")
print("-" * 64)
print(f"{'Linear Regression':<20} {lr_mse_scaled:>8.4f} +/- {'0.0000':>6}   {lr_mae_scaled:>8.4f} +/- {'0.0000':>6}")
print(f"{'MLP':<20} {mlp_mse_scaled_mean:>8.4f} +/- {mlp_mse_scaled_std:.4f}   {mlp_mae_scaled_mean:>8.4f} +/- {mlp_mae_scaled_std:.4f}")
print(f"{'LSTM':<20} {lstm_mse_scaled_mean:>8.4f} +/- {lstm_mse_scaled_std:.4f}   {lstm_mae_scaled_mean:>8.4f} +/- {lstm_mae_scaled_std:.4f}")

print("\nOriginal-scale test metrics")
print(f"{'Model':<20} {'MSE':>20} {'MAE':>20} {'RMSE':>20}")
print("-" * 84)
print(f"{'Linear Regression':<20} {lr_mse_orig:>8.4f} +/- {'0.0000':>6}   {lr_mae_orig:>8.4f} +/- {'0.0000':>6}   {lr_rmse_orig:>8.4f} +/- {'0.0000':>6}")
print(f"{'MLP':<20} {mlp_mse_orig_mean:>8.4f} +/- {mlp_mse_orig_std:.4f}   {mlp_mae_orig_mean:>8.4f} +/- {mlp_mae_orig_std:.4f}   {mlp_rmse_orig_mean:>8.4f} +/- {mlp_rmse_orig_std:.4f}")
print(f"{'LSTM':<20} {lstm_mse_orig_mean:>8.4f} +/- {lstm_mse_orig_std:.4f}   {lstm_mae_orig_mean:>8.4f} +/- {lstm_mae_orig_std:.4f}   {lstm_rmse_orig_mean:>8.4f} +/- {lstm_rmse_orig_std:.4f}")

print("\nRun metadata")
print(f"Kullanilan seedler: {SEEDS}")
print(f"MLP best epochs: {mlp_best_epochs}")
print(f"LSTM best epochs: {lstm_best_epochs}")
print(
    f"Early stopping: patience={PATIENCE}, max_epochs={MAX_EPOCHS}, "
    f"batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}"
)
print("Linear Regression deterministik (seed varyasyonu yok).")
print("std: ddof=1 (Bessel's correction).")

os.makedirs("results", exist_ok=True)

training_curves_df = pd.DataFrame(training_curve_rows)
training_curves_path = "results/training_curves.csv"
training_curves_df.to_csv(training_curves_path, index=False)
print(f"\nTraining curves saved to: {training_curves_path}")

summary_rows = [
    {
        "model": "Linear Regression",
        "seeds": "deterministic",
        "deterministic": True,
        "best_epochs": "",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "max_epochs": MAX_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
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
        "seeds": ",".join(str(seed) for seed in SEEDS),
        "deterministic": False,
        "best_epochs": ",".join(str(epoch) for epoch in mlp_best_epochs),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "max_epochs": MAX_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
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
        "seeds": ",".join(str(seed) for seed in SEEDS),
        "deterministic": False,
        "best_epochs": ",".join(str(epoch) for epoch in lstm_best_epochs),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "max_epochs": MAX_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
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
summary_path = "results/multi_seed_fair_baseline.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Run summary saved to: {summary_path}")
