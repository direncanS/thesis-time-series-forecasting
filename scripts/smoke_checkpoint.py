"""
Import-safe checkpoint-load + forward smoke test.

Purpose: Verify that the newly-installed env can load and forward-pass the
existing MLP, LSTM, and TFT checkpoints without triggering any src/* top-level
code (which would overwrite live artefacts snapshot).

Design:
  - MLP and LSTMModel classes redefined INLINE (bit-identical to
    src/training/multi_seed.py:62-86) — never `from src.training... import`
    because multi_seed.py:230+ runs training at import time.
  - MSE(MultiHorizonMetric) class defined at MODULE LEVEL for pickle's
    `__main__.MSE` resolution during TFT checkpoint unpickle (mirrors
    src/evaluation/export_predictions.py:45-54).

Usage:
  python scripts/smoke_checkpoint.py

Expected output (3 PASS lines):
  [OK] MLP checkpoint load + forward: params=124328 shape=(1, 168)
  [OK] LSTM checkpoint load + forward: params=29608 shape=(1, 168)
  [OK] TFT checkpoint load: params=18261

Failure modes:
  - torch version incompatible with saved state_dict → load_state_dict error
  - pytorch-forecasting version incompatible with TFT ckpt → load_from_checkpoint error
  - CPU-only env without map_location patch → TFT .ckpt device resolution error
    (see plan v11 Gate A conditional path)
"""

import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MultiHorizonMetric


# ============================================================
# Module-level MSE — required for pickle `__main__.MSE` resolution
# when loading TFT checkpoints saved by tft_fair_5seed.py.
# Byte-equivalent to src/evaluation/export_predictions.py:45-54.
# ============================================================
class MSE(MultiHorizonMetric):
    """Mean squared error — symmetric with nn.MSELoss. Must be at module level."""

    def __init__(self, reduction="mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def loss(self, y_pred, target):
        return torch.pow(self.to_prediction(y_pred) - target, 2)


# ============================================================
# Inline model defs — mirror src/training/multi_seed.py:62-86 exactly.
# Do NOT import from src.training.multi_seed (would trigger training at import).
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
# 1) MLP smoke
# ============================================================
mlp = MLP(input_size=672, output_size=168, hidden_size=128)
mlp.load_state_dict(torch.load("checkpoints/mlp_seed42.pt", map_location="cpu"))
mlp.eval()
x_mlp = torch.randn(1, 672)
with torch.no_grad():
    y_mlp = mlp(x_mlp)
assert y_mlp.shape == (1, 168), f"MLP forward shape hatalı: {y_mlp.shape}"
assert torch.isfinite(y_mlp).all(), "MLP forward NaN/Inf"
mlp_params = sum(p.numel() for p in mlp.parameters())
assert mlp_params == 124328, (
    f"MLP param count beklenmedik: {mlp_params} "
    f"(beklenen 124328 per EXPERIMENT_MANIFEST)"
)
print(f"[OK] MLP checkpoint load + forward: params={mlp_params} shape={tuple(y_mlp.shape)}")


# ============================================================
# 2) LSTM smoke
# ============================================================
lstm = LSTMModel(input_size=7, hidden_size=64, output_size=168, num_layers=1)
lstm.load_state_dict(torch.load("checkpoints/lstm_seed42.pt", map_location="cpu"))
lstm.eval()
x_lstm = torch.randn(1, 96, 7)  # (batch, seq_len, features)
with torch.no_grad():
    y_lstm = lstm(x_lstm)
assert y_lstm.shape == (1, 168), f"LSTM forward shape hatalı: {y_lstm.shape}"
assert torch.isfinite(y_lstm).all(), "LSTM forward NaN/Inf"
lstm_params = sum(p.numel() for p in lstm.parameters())
assert lstm_params == 29608, (
    f"LSTM param count beklenmedik: {lstm_params} "
    f"(beklenen 29608 per EXPERIMENT_MANIFEST)"
)
print(f"[OK] LSTM checkpoint load + forward: params={lstm_params} shape={tuple(y_lstm.shape)}")


# ============================================================
# 3) TFT smoke — load + param count only (full forward needs dataloader)
# ============================================================
tft = TemporalFusionTransformer.load_from_checkpoint("checkpoints/tft_seed42.ckpt")
tft.eval()
tft_params = sum(p.numel() for p in tft.parameters())
assert tft_params == 18261, (
    f"TFT param count beklenmedik: {tft_params} "
    f"(beklenen 18261 per the active configuration § 10)"
)
print(f"[OK] TFT checkpoint load: params={tft_params}")

print("\n=== SMOKE PASS ===")
