"""Unit tests for src/common.py metric helpers."""

from __future__ import annotations

import numpy as np
import pytest
from src.common import (
    compute_original_metrics,
    compute_scaled_metrics,
    summarize_results,
)


def test_compute_scaled_metrics_constant_offset():
    y_true = np.zeros((10, 5, 3))
    y_pred = np.ones((10, 5, 3))
    mse, mae = compute_scaled_metrics(y_true, y_pred)
    assert mse == pytest.approx(1.0)
    assert mae == pytest.approx(1.0)


def test_compute_original_metrics_all_three():
    """MSE=4, MAE=2, RMSE=sqrt(4)=2 when residuals are constant 2.0."""
    y_true = np.zeros((8, 4, 2))
    y_pred = np.full((8, 4, 2), 2.0)
    mse, mae, rmse = compute_original_metrics(y_true, y_pred)
    assert mse == pytest.approx(4.0)
    assert mae == pytest.approx(2.0)
    assert rmse == pytest.approx(2.0)


def test_compute_original_metrics_zero_residual():
    y = np.random.randn(5, 3, 2)
    mse, mae, rmse = compute_original_metrics(y, y)
    assert mse == pytest.approx(0.0)
    assert mae == pytest.approx(0.0)
    assert rmse == pytest.approx(0.0)


def test_summarize_results_ddof1_std():
    """Per the active configuration.: across-seed aggregation uses mean ± std with ddof=1."""
    rows = [{"mse": 1.0}, {"mse": 2.0}, {"mse": 3.0}]
    mean, std = summarize_results(rows, "mse")
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)  # np.std([1,2,3], ddof=1) == 1.0


def test_summarize_results_single_element():
    """Single-element std with ddof=1 is NaN; verify API does not crash."""
    rows = [{"mse": 42.0}]
    mean, std = summarize_results(rows, "mse")
    assert mean == pytest.approx(42.0)
    assert np.isnan(std)
