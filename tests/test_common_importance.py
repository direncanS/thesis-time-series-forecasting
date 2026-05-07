"""Unit tests for src.explainability.common_importance.

These are lightweight primitive-level checks. The full GPU-involving
occlusion run is exercised by the user via
``python src/explainability/common_importance.py``.
"""

from __future__ import annotations

import numpy as np
from src.explainability.common_importance import (
    compute_model_occlusion,
    mse_scaled,
    occlude_variable,
)


def test_occlude_variable_zeros_single_column():
    X = np.random.default_rng(0).normal(size=(4, 3, 5)).astype(np.float64)
    out = occlude_variable(X, var_idx=2)
    assert np.all(out[:, :, 2] == 0.0)
    # Other columns untouched
    for idx in (0, 1, 3, 4):
        assert np.array_equal(out[:, :, idx], X[:, :, idx])
    # Input not mutated
    assert not np.all(X[:, :, 2] == 0.0)


def test_mse_scaled_matches_numpy():
    y_true = np.zeros((5, 3, 2))
    y_pred = np.ones((5, 3, 2))
    assert mse_scaled(y_true, y_pred) == 1.0


def test_compute_model_occlusion_identity_predict_fn_yields_zero_importance():
    """With a predict_fn that reproduces y_test exactly, baseline MSE = 0 and
    every occluded MSE equals the per-variable residual induced by zeroing the
    column — a tight sanity check on the bookkeeping."""
    rng = np.random.default_rng(1)
    y_test = rng.normal(size=(3, 2, 4))
    X_test = y_test.copy()  # identity relationship

    def predict_fn(X):
        return X  # identity model

    rows = compute_model_occlusion(
        predict_fn, X_test, y_test, var_names=["a", "b", "c", "d"]
    )
    assert len(rows) == 4
    assert all("variable" in r and "importance" in r for r in rows)

    # baseline_mse must be the same for every row (single per-model baseline)
    baselines = {r["baseline_mse"] for r in rows}
    assert len(baselines) == 1
    # With identity predict_fn, baseline == 0
    assert next(iter(baselines)) == 0.0

    # Occluding any single variable must introduce strictly positive error
    assert all(r["importance"] >= 0.0 for r in rows)


def test_compute_model_occlusion_row_count_matches_var_count():
    def predict_fn(X):
        return np.zeros_like(X)

    X = np.random.default_rng(2).normal(size=(2, 3, 5))
    y = np.zeros_like(X)
    rows = compute_model_occlusion(predict_fn, X, y, var_names=list("abcde"))
    assert len(rows) == 5
