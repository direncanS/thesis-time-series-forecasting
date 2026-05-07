"""Unit tests for src/common.py preprocessing helpers.

Closes the behavioural-equivalence audit items earlier audit pass:
tensor shape / scaler fit noktası / split sırası.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.common import (
    DEFAULT_CONFIG,
    chronological_split,
    create_windows,
    fit_scaler_on_train,
    inverse_transform_3d,
    make_split_frame,
)


def test_create_windows_shapes():
    data = np.random.randn(200, 7)
    X, y = create_windows(data, input_len=10, output_len=4)
    expected_n = 200 - 10 - 4 + 1
    assert X.shape == (expected_n, 10, 7)
    assert y.shape == (expected_n, 4, 7)


def test_create_windows_stride_one_alignment():
    """Window i covers [i, i+input_len); target covers [i+input_len, i+input_len+output_len)."""
    data = np.arange(30).reshape(30, 1).astype(float)
    X, y = create_windows(data, input_len=3, output_len=2)
    np.testing.assert_array_equal(X[0].ravel(), [0, 1, 2])
    np.testing.assert_array_equal(y[0].ravel(), [3, 4])
    np.testing.assert_array_equal(X[1].ravel(), [1, 2, 3])
    np.testing.assert_array_equal(y[1].ravel(), [4, 5])


def test_create_windows_uses_defaults():
    data = np.random.randn(300, 7)
    X, y = create_windows(data)
    assert X.shape == (300 - DEFAULT_CONFIG["input_len"] - DEFAULT_CONFIG["output_len"] + 1,
                       DEFAULT_CONFIG["input_len"], 7)
    assert y.shape == (300 - DEFAULT_CONFIG["input_len"] - DEFAULT_CONFIG["output_len"] + 1,
                       DEFAULT_CONFIG["output_len"], 7)


def test_chronological_split_60_20_20():
    df = pd.DataFrame({"a": np.arange(100)})
    train, val, test = chronological_split(df)
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
    assert train["a"].iloc[-1] < val["a"].iloc[0]
    assert val["a"].iloc[-1] < test["a"].iloc[0]


def test_chronological_split_non_overlapping_and_ordered():
    df = pd.DataFrame({"t": np.arange(1000)})
    train, val, test = chronological_split(df)
    # Contiguous, non-overlapping
    assert train["t"].iloc[-1] + 1 == val["t"].iloc[0]
    assert val["t"].iloc[-1] + 1 == test["t"].iloc[0]
    # Total coverage
    assert len(train) + len(val) + len(test) == 1000


def test_scaler_fit_train_only_not_contaminated_by_val_test():
    """Scaler fit uses train only; val/test outliers must NOT shift the learned scale."""
    train = pd.DataFrame({"a": np.array([1.0, 2.0, 3.0, 4.0, 5.0])})
    val = pd.DataFrame({"a": np.array([1000.0, 2000.0])})
    test = pd.DataFrame({"a": np.array([5000.0])})
    scaler, train_s, val_s, test_s = fit_scaler_on_train(train, val, test)

    # Scaler fit on train only — mean reflects [1..5]
    assert scaler.mean_[0] == pytest.approx(3.0)
    # Train transformed is standardised (mean ~0)
    assert train_s.mean() == pytest.approx(0.0, abs=1e-10)
    # Val transformed is NOT mean-0 because it used train's stats
    assert abs(val_s.mean()) > 100.0  # dramatically off zero


def test_inverse_transform_3d_round_trip():
    """inverse_transform_3d(scaler.transform_3d(x)) should ~= x."""
    train = pd.DataFrame({"a": np.linspace(0, 10, 50), "b": np.linspace(-5, 5, 50)})
    scaler, train_s, _, _ = fit_scaler_on_train(train, train.iloc[:5], train.iloc[:5])
    # Make a 3D scaled window
    arr_3d = train_s.reshape(10, 5, 2)
    back = inverse_transform_3d(arr_3d, scaler)
    expected = train.to_numpy().reshape(10, 5, 2)
    np.testing.assert_allclose(back, expected, rtol=1e-9, atol=1e-9)


def test_make_split_frame_has_time_idx_and_group_id():
    train_scaled = np.random.randn(20, 3)
    var_names = ["f0", "f1", "f2"]
    frame = make_split_frame(train_scaled, var_names)
    assert "time_idx" in frame.columns
    assert "group_id" in frame.columns
    assert (frame["time_idx"] == np.arange(20)).all()
    assert (frame["group_id"] == "ETTh1").all()
    for name in var_names:
        assert name in frame.columns
