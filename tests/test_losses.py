"""Unit tests for src/common.py::UnifiedMSE.

Closure Plan v6.1 C1 closure target: UnifiedMSE (single instance,
reduction="mean" over flattened tensor) replaces v1's
MultiLoss([MSE() for _ in range(7)]) 7×-asymmetric TFT construction.

UnifiedMSE subclasses pytorch_forecasting.MultiHorizonMetric; its .loss()
method routes y_pred through MultiHorizonMetric.to_prediction, which for
3D inputs requires `y_pred.size(-1) == 1` (a single point-prediction
dimension). These tests therefore exercise the contract TFT actually uses
at training time: y_pred shape (batch, horizon, 1), target shape
(batch, horizon).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from src.common import UnifiedMSE


def test_unified_mse_loss_returns_elementwise_squared_error():
    torch.manual_seed(0)
    pred = torch.randn(4, 6, 1)
    target = torch.randn(4, 6)
    unified = UnifiedMSE()
    raw = unified.loss(pred, target)
    expected = (pred.squeeze(-1) - target) ** 2
    assert raw.shape == expected.shape
    assert torch.allclose(raw, expected)


def test_unified_mse_mean_matches_nn_mseloss_mean():
    """When reduced to mean, UnifiedMSE must equal nn.MSELoss(reduction='mean').

    Both operate on the same underlying per-element squared residuals. The
    rough equivalence path is:
        UnifiedMSE.loss(pred3d, target2d).mean()  ==  nn.MSELoss()(pred2d, target2d)
    where pred2d = pred3d.squeeze(-1).
    """
    torch.manual_seed(1)
    pred3d = torch.randn(8, 12, 1)
    target2d = torch.randn(8, 12)
    pred2d = pred3d.squeeze(-1)

    unified_mean = UnifiedMSE().loss(pred3d, target2d).mean()
    nn_mean = nn.MSELoss(reduction="mean")(pred2d, target2d)
    assert torch.allclose(unified_mean, nn_mean, atol=1e-7)


def test_unified_mse_reduction_argument_accepted():
    """UnifiedMSE must accept reduction kwarg without error."""
    UnifiedMSE(reduction="mean")
    UnifiedMSE(reduction="sum")
    UnifiedMSE(reduction="none")


def test_unified_mse_zero_residual():
    torch.manual_seed(2)
    x = torch.randn(3, 4, 1)
    raw = UnifiedMSE().loss(x, x.squeeze(-1))
    assert torch.all(raw == 0.0)
