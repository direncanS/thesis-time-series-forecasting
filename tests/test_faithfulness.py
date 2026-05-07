"""Unit tests for src.explainability.faithfulness_test.

AOPC primitive-level checks. The full GPU-involving AOPC run is exercised
by the user via ``python src/explainability/faithfulness_test.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.explainability.faithfulness_test import (
    aopc_for_model_seed,
    mask_variables,
    rank_from_occlusion,
)


def test_mask_variables_zeros_multiple_columns():
    X = np.ones((3, 4, 5))
    out = mask_variables(X, var_indices=[1, 3])
    assert np.all(out[:, :, 1] == 0.0)
    assert np.all(out[:, :, 3] == 0.0)
    # Other columns untouched
    for idx in (0, 2, 4):
        assert np.all(out[:, :, idx] == 1.0)
    # Input untouched
    assert np.all(X == 1.0)


def test_rank_from_occlusion_respects_descending_importance():
    df = pd.DataFrame(
        [
            {"model": "MLP", "seed": 42, "variable": "a", "importance": 0.1},
            {"model": "MLP", "seed": 42, "variable": "b", "importance": 0.5},
            {"model": "MLP", "seed": 42, "variable": "c", "importance": 0.2},
        ]
    )
    ranking = rank_from_occlusion(df, "MLP", 42, ["a", "b", "c"])
    assert ranking == ["b", "c", "a"]


def test_aopc_for_model_seed_identity_predict_produces_symmetric_mask_effect():
    """With a predict_fn that simply returns the input, the top-k and bot-k
    occlusions create symmetric damage across k because the ranking is
    deterministic — AOPC is then driven by how importance differs between
    the most- and least-important variables, exactly as designed."""
    rng = np.random.default_rng(7)
    n_windows, horizon, n_features = 8, 3, 4
    var_names = ["v0", "v1", "v2", "v3"]
    # Construct y_test so that variable 0 dominates the signal and variable 3 is near-zero;
    # ranking from occlusion will put v0 first, v3 last.
    y = rng.normal(size=(n_windows, horizon, n_features))
    y[:, :, 0] *= 10.0
    y[:, :, 3] *= 0.01
    X = y.copy()

    def predict_fn(Z):
        return Z  # identity

    ranking = ["v0", "v1", "v2", "v3"]
    aopc, gap_std, per_k = aopc_for_model_seed(
        predict_fn, X, y, ranking, var_names
    )
    assert len(per_k) == n_features
    # Top-1 (mask v0) should hurt far more than Bot-1 (mask v3) → gap_1 > 0
    assert per_k[0]["gap"] > 0
    # At k = n_features all vars are masked in both top and bot → identical → gap_n = 0
    assert per_k[-1]["gap"] == 0.0
    # AOPC is the mean of per-k gaps (per_k gaps are rounded to 8 decimals
    # before storage, so allow a tiny rounding tolerance).
    expected = float(np.mean([r["gap"] for r in per_k]))
    assert abs(aopc - expected) < 1e-6
    # gap_std is population std ddof=1 across k, non-negative
    assert gap_std >= 0.0


def test_aopc_records_baseline_consistently_across_k():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(4, 2, 3))
    y = X.copy()

    def predict_fn(Z):
        return Z

    _, _, per_k = aopc_for_model_seed(predict_fn, X, y, ["a", "b", "c"], ["a", "b", "c"])
    baselines = {r["baseline_mse"] for r in per_k}
    assert len(baselines) == 1  # one baseline, reused across k
