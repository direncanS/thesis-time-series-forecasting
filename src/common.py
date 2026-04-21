from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler

try:
    from pytorch_forecasting.metrics import MultiHorizonMetric
except ImportError:  # pragma: no cover - exercised only in envs without TFT deps
    MultiHorizonMetric = object


DEFAULT_CONFIG: dict[str, Any] = {
    "data_path": "data/ETTh1.csv",
    "input_len": 96,
    "output_len": 24,
    "batch_size": 64,
    "max_epochs": 200,
    "patience": 10,
    "seeds": [42, 123, 456, 789, 1024],
    "results_dir": "results/bachelor_safe_v2",
    "checkpoints_dir": "checkpoints/bachelor_safe_v2",
    "precision": {
        "matmul": "highest",
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    },
    "models": {
        "mlp": {"hidden_size": 128, "learning_rate": 1e-4, "weight_decay": 0.0},
        "lstm": {
            "hidden_size": 64,
            "num_layers": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
        },
        "tft": {
            "hidden_size": 16,
            "attention_head_size": 4,
            "dropout": 0.1,
            "hidden_continuous_size": 8,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(copy.deepcopy(base[key]), value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if not config_path:
        return cfg

    config_file = Path(config_path)
    loaded = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    extends = loaded.pop("extends", None)
    if extends:
        base_path = (config_file.parent / extends).resolve()
        base_loaded = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        cfg = _deep_update(cfg, base_loaded)
    return _deep_update(cfg, loaded)


def apply_runtime_overrides(
    cfg: dict[str, Any],
    *,
    smoke: bool = False,
    results_dir: str | None = None,
    checkpoints_dir: str | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    if results_dir:
        out["results_dir"] = results_dir
    if checkpoints_dir:
        out["checkpoints_dir"] = checkpoints_dir
    if smoke:
        out["seeds"] = out["seeds"][:1]
        out["max_epochs"] = 1
        out["patience"] = 1
        out["results_dir"] = str(Path(out["results_dir"]) / "smoke")
        out["checkpoints_dir"] = str(Path(out["checkpoints_dir"]) / "smoke")
    return out


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def create_windows(
    data: np.ndarray,
    input_len: int = DEFAULT_CONFIG["input_len"],
    output_len: int = DEFAULT_CONFIG["output_len"],
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len])
    return np.array(X), np.array(y)


def inverse_transform_3d(arr_3d: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n_w, t, n_f = arr_3d.shape
    flat = arr_3d.reshape(n_w * t, n_f)
    return scaler.inverse_transform(flat).reshape(n_w, t, n_f)


def chronological_split(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(features)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train = features.iloc[:train_end]
    val = features.iloc[train_end:val_end]
    test = features.iloc[val_end:]
    return train, val, test


def fit_scaler_on_train(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(train)
    return scaler, scaler.transform(train), scaler.transform(val), scaler.transform(test)


def load_feature_frame(data_path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(data_path)
    features = df.drop(columns=["date"])
    return features, features.columns.tolist()


def prepare_supervised_data(
    data_path: str,
    input_len: int,
    output_len: int,
) -> dict[str, Any]:
    features, var_names = load_feature_frame(data_path)
    train, val, test = chronological_split(features)
    scaler, train_scaled, val_scaled, test_scaled = fit_scaler_on_train(train, val, test)

    X_train, y_train = create_windows(train_scaled, input_len, output_len)
    X_val, y_val = create_windows(val_scaled, input_len, output_len)
    X_test, y_test = create_windows(test_scaled, input_len, output_len)

    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]

    return {
        "features": features,
        "var_names": var_names,
        "train": train,
        "val": val,
        "test": test,
        "scaler": scaler,
        "train_scaled": train_scaled,
        "val_scaled": val_scaled,
        "test_scaled": test_scaled,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_flat": X_train.reshape(n_train, -1),
        "X_val_flat": X_val.reshape(n_val, -1),
        "X_test_flat": X_test.reshape(n_test, -1),
        "y_train_flat": y_train.reshape(n_train, -1),
        "y_val_flat": y_val.reshape(n_val, -1),
        "y_test_flat": y_test.reshape(n_test, -1),
        "y_test_orig": inverse_transform_3d(y_test, scaler),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "input_size": X_train.reshape(n_train, -1).shape[1],
        "output_size": y_train.reshape(n_train, -1).shape[1],
    }


def make_split_frame(scaled_arr: np.ndarray, var_names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(scaled_arr, columns=var_names)
    frame["time_idx"] = np.arange(len(frame), dtype=np.int64)
    frame["group_id"] = "ETTh1"
    return frame


def compute_scaled_metrics(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((y_true_scaled - y_pred_scaled) ** 2))
    mae = float(np.mean(np.abs(y_true_scaled - y_pred_scaled)))
    return mse, mae


def compute_original_metrics(y_true_orig: np.ndarray, y_pred_orig: np.ndarray) -> tuple[float, float, float]:
    mse = float(np.mean((y_true_orig - y_pred_orig) ** 2))
    mae = float(np.mean(np.abs(y_true_orig - y_pred_orig)))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse


def summarize_results(results: list[dict[str, Any]], metric_key: str) -> tuple[float, float]:
    values = [row[metric_key] for row in results]
    return float(np.mean(values)), float(np.std(values, ddof=1))


def finalize_history_rows(history_rows: list[dict[str, Any]], best_epoch: int) -> list[dict[str, Any]]:
    finalized = []
    for row in history_rows:
        new_row = row.copy()
        new_row["is_best_epoch"] = row["epoch"] == best_epoch
        finalized.append(new_row)
    return finalized


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class UnifiedMSE(MultiHorizonMetric):
    def __init__(self, reduction: str = "mean", **kwargs: Any):
        super().__init__(reduction=reduction, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.pow(self.to_prediction(y_pred) - target, 2)


def configure_stochastic_runtime(cfg: dict[str, Any]) -> torch.device:
    precision = cfg["precision"]
    torch.set_float32_matmul_precision(precision["matmul"])
    torch.backends.cuda.matmul.allow_tf32 = precision["tf32_matmul"]
    torch.backends.cudnn.allow_tf32 = precision["tf32_cudnn"]
    torch.backends.cudnn.deterministic = precision["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = precision["cudnn_benchmark"]
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def runtime_flag_block(device: torch.device) -> str:
    return (
        f"device={device} "
        f"matmul_precision={torch.get_float32_matmul_precision()} "
        f"cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32} "
        f"cudnn.deterministic={torch.backends.cudnn.deterministic} "
        f"cudnn.benchmark={torch.backends.cudnn.benchmark}"
    )


def upsert_runtime_rows(output_csv: str | Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    output_path = Path(output_csv)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        existing = existing[~existing["model"].isin([row["model"] for row in rows])]
        merged = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    else:
        merged = pd.DataFrame(rows)
    merged.to_csv(output_path, index=False)
    return merged
