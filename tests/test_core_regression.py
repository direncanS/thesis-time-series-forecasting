"""B2 regression smoke for the src/common.py surgical refactor.

Asserts that the refactored MLP and LSTMModel classes, combined with the
common.py preprocessing pipeline, reproduce the pre-refactor (v1) prediction
tensors within tolerance when loaded against the v1 checkpoints archived under
``archive/pre-v2-2026-04-22/``.

Closure Plan v6.1 B2 acceptance — per-file tolerances:
    - CPU-deterministic paths (LR closed-form): exact match (atol=0, rtol=0)
    - GPU / FP32 numerical paths (MLP, LSTM forward): rtol=1e-6, atol=1e-8

TFT regression is intentionally NOT part of this scaffold. TFT checkpoints
were trained under the legacy MultiLoss-of-7 + TF32 regime; B5 will produce
v2 TFT checkpoints under UnifiedMSE + 5-flag FP32 and the comparison between
v1 and v2 TFT predictions is a thesis-level observation, not a refactor
regression target.

Running:
    pytest tests/test_core_regression.py

Skips when the archived v1 artefacts or the MLP/LSTM checkpoints are missing
so the test suite can run on fresh clones without the v1 data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.linear_model import LinearRegression
from src.common import (
    DEFAULT_CONFIG,
    MLP,
    LSTMModel,
    inverse_transform_3d,
    prepare_supervised_data,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = REPO_ROOT / "archive" / "pre-v2-2026-04-22"
V1_RESULTS = ARCHIVE_DIR / "results"
V1_CHECKPOINTS = ARCHIVE_DIR / "checkpoints"

V1_SEEDS = (42, 123, 456)

RTOL = 1e-6
ATOL = 1e-8


@pytest.fixture(scope="module")
def v1_available():
    missing = []
    if not V1_RESULTS.exists():
        missing.append(str(V1_RESULTS))
    if not V1_CHECKPOINTS.exists():
        missing.append(str(V1_CHECKPOINTS))
    if missing:
        pytest.skip(f"Archived v1 artefacts not available: {missing}")
    return True


@pytest.fixture(scope="module")
def supervised_data():
    cfg = DEFAULT_CONFIG
    data_path = REPO_ROOT / cfg["data_path"]
    if not data_path.exists():
        pytest.skip(f"Dataset missing at {data_path}")
    return prepare_supervised_data(str(data_path), cfg["input_len"], cfg["output_len"])


def test_lr_regression_exact(v1_available, supervised_data):
    """LR is closed-form OLS; refactored pipeline must reproduce v1 preds exactly."""
    expected = np.load(V1_RESULTS / "preds_lr.npy")

    lr = LinearRegression()
    lr.fit(supervised_data["X_train_flat"], supervised_data["y_train_flat"])
    n_features = len(supervised_data["var_names"])
    output_len = DEFAULT_CONFIG["output_len"]
    preds_scaled = lr.predict(supervised_data["X_test_flat"]).reshape(
        supervised_data["n_test"], output_len, n_features
    )
    actual = inverse_transform_3d(preds_scaled, supervised_data["scaler"])

    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} vs {expected.shape}"
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("seed", V1_SEEDS)
def test_mlp_forward_tolerance(v1_available, supervised_data, seed):
    """MLP class forward pass over v1 checkpoint must match v1 preds within tolerance."""
    ckpt_path = V1_CHECKPOINTS / f"mlp_seed{seed}.pt"
    expected_path = V1_RESULTS / f"preds_mlp_seed{seed}.npy"
    if not ckpt_path.exists() or not expected_path.exists():
        pytest.skip(f"missing v1 artefact: {ckpt_path} or {expected_path}")

    expected = np.load(expected_path)
    mlp = MLP(
        supervised_data["input_size"],
        supervised_data["output_size"],
        hidden_size=DEFAULT_CONFIG["models"]["mlp"]["hidden_size"],
    )
    mlp.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    mlp.eval()

    x = torch.FloatTensor(supervised_data["X_test_flat"])
    with torch.no_grad():
        preds_flat = mlp(x).numpy()
    n_features = len(supervised_data["var_names"])
    output_len = DEFAULT_CONFIG["output_len"]
    preds_scaled = preds_flat.reshape(supervised_data["n_test"], output_len, n_features)
    actual = inverse_transform_3d(preds_scaled, supervised_data["scaler"])

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", V1_SEEDS)
def test_lstm_forward_tolerance(v1_available, supervised_data, seed):
    """LSTMModel class forward pass over v1 checkpoint must match v1 preds within tolerance."""
    ckpt_path = V1_CHECKPOINTS / f"lstm_seed{seed}.pt"
    expected_path = V1_RESULTS / f"preds_lstm_seed{seed}.npy"
    if not ckpt_path.exists() or not expected_path.exists():
        pytest.skip(f"missing v1 artefact: {ckpt_path} or {expected_path}")

    expected = np.load(expected_path)
    n_features = len(supervised_data["var_names"])
    lstm = LSTMModel(
        input_size=n_features,
        hidden_size=DEFAULT_CONFIG["models"]["lstm"]["hidden_size"],
        output_size=supervised_data["output_size"],
        num_layers=DEFAULT_CONFIG["models"]["lstm"]["num_layers"],
    )
    lstm.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    lstm.eval()

    x = torch.FloatTensor(supervised_data["X_test"])
    with torch.no_grad():
        preds_flat = lstm(x).numpy()
    output_len = DEFAULT_CONFIG["output_len"]
    preds_scaled = preds_flat.reshape(supervised_data["n_test"], output_len, n_features)
    actual = inverse_transform_3d(preds_scaled, supervised_data["scaler"])

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
