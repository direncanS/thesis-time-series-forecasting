"""Common, model-agnostic occlusion importance.

Produces a uniform cross-model variable-importance instrument for all four
core models (LR, MLP, LSTM, TFT). The interpretability contract of CLAUDE.md
§ 13 forbids treating SHAP (post-hoc, fitted to three of the four models)
and VSN (architecture-native, TFT-only) as the same measurement object; this
script introduces a common instrument that *is* the same measurement object
across all four.

Role:
    - Occlusion importance **is** the common explanation *instrument*.
    - AOPC (computed in `faithfulness_test.py`) is the faithfulness *metric*
      that consumes the rankings produced here.
    - Instrument and metric are not interchangeable; this file produces only
      the instrument.

Method
------
For each (model, seed, variable v):
    1. Establish a per-model baseline MSE on the scaled test partition.
    2. Occlude variable v by overwriting its column in the scaled test
       tensor with the baseline value 0.0. Because the StandardScaler is
       fit on the training partition only, the zero-vector in the scaled
       input space equals the training-distribution mean in the original
       space by construction (CLAUDE.md § 10).
    3. Re-predict with the unchanged checkpoint and compute the occluded
       MSE.
    4. Importance(model, seed, v) = occluded_MSE − baseline_MSE. Higher
       values indicate the model relied more on variable v.

Per-seed handling
-----------------
LR is deterministic (single closed-form OLS solution); it produces one row
per variable with ``seed = "deterministic"``. MLP, LSTM, and TFT are
stochastic and are averaged at the downstream cross-model-agreement /
AOPC layers rather than here, so this script keeps the per-(model, seed)
resolution intact.

Output
------
``results/bachelor_safe_v2/occlusion_importance.csv`` — long format:
    model, seed, variable, baseline_mse, occluded_mse, importance
Row count: 7 (LR) + 5 × 7 × 3 (MLP, LSTM, TFT) = 112 rows.

No model-specific branching in the top-level occlusion loop; each model's
forward pass is wrapped in a per-model ``predict_fn`` closure built at
setup time. The occlusion inner loop is identical across all four models.

Run
---
    python src/explainability/common_importance.py \\
        --config configs/experiments/fair_core_v2.yaml
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import lightning as L  # noqa: F401  (pytorch-forecasting dependency)
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer
from sklearn.linear_model import LinearRegression

from src.common import (
    MLP,
    LSTMModel,
    UnifiedMSE,
    apply_runtime_overrides,
    ensure_dir,
    load_config,
    make_split_frame,
    prepare_supervised_data,
)

warnings.filterwarnings("ignore")


# Module-level alias preserves pickle compatibility for legacy TFT checkpoints
# that encoded the training-time loss class as ``__main__.MSE``.
class MSE(UnifiedMSE):
    """Legacy/new TFT checkpoint pickle-resolution alias."""


MODELS_ORDERED = ("LR", "MLP", "LSTM", "TFT")


# ---------------------------------------------------------------------------
# Core occlusion primitive
# ---------------------------------------------------------------------------


def occlude_variable(X_3d_scaled: np.ndarray, var_idx: int) -> np.ndarray:
    """Return a copy of ``X_3d_scaled`` with variable column ``var_idx`` set to 0.

    Zero in scaled space equals the training-distribution mean in the
    original space (StandardScaler construction); see CLAUDE.md § 10.
    """
    out = X_3d_scaled.copy()
    out[:, :, var_idx] = 0.0
    return out


def mse_scaled(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray) -> float:
    return float(np.mean((y_true_scaled - y_pred_scaled) ** 2))


def compute_model_occlusion(
    predict_fn,
    X_test_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
    var_names: list[str],
) -> list[dict]:
    """Run baseline + per-variable occlusion through a generic predict_fn.

    ``predict_fn`` is the only point at which any model-specific detail
    enters this function; everything else is uniform across all four models.
    """
    rows = []
    baseline = mse_scaled(y_test_scaled, predict_fn(X_test_scaled))
    for var_idx, var in enumerate(var_names):
        occluded = predict_fn(occlude_variable(X_test_scaled, var_idx))
        occluded_mse = mse_scaled(y_test_scaled, occluded)
        rows.append(
            {
                "variable": var,
                "baseline_mse": round(baseline, 8),
                "occluded_mse": round(occluded_mse, 8),
                "importance": round(occluded_mse - baseline, 8),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Per-model predict_fn factories
# ---------------------------------------------------------------------------


def make_lr_predict(data: dict, cfg: dict):
    """Fit the deterministic LR once and return a closed-over predict_fn."""
    lr = LinearRegression()
    lr.fit(data["X_train_flat"], data["y_train_flat"])
    n_features = len(data["var_names"])
    output_len = cfg["output_len"]

    def predict(X_scaled: np.ndarray) -> np.ndarray:
        flat = X_scaled.reshape(X_scaled.shape[0], -1)
        return lr.predict(flat).reshape(X_scaled.shape[0], output_len, n_features)

    return predict


def make_mlp_predict(data: dict, cfg: dict, ckpt_path: Path):
    model = MLP(
        data["input_size"],
        data["output_size"],
        hidden_size=cfg["models"]["mlp"]["hidden_size"],
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    n_features = len(data["var_names"])
    output_len = cfg["output_len"]

    def predict(X_scaled: np.ndarray) -> np.ndarray:
        flat = torch.FloatTensor(X_scaled.reshape(X_scaled.shape[0], -1))
        with torch.no_grad():
            out = model(flat).numpy()
        return out.reshape(X_scaled.shape[0], output_len, n_features)

    return predict


def make_lstm_predict(data: dict, cfg: dict, ckpt_path: Path):
    n_features = len(data["var_names"])
    model = LSTMModel(
        input_size=n_features,
        hidden_size=cfg["models"]["lstm"]["hidden_size"],
        output_size=data["output_size"],
        num_layers=cfg["models"]["lstm"]["num_layers"],
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    output_len = cfg["output_len"]

    def predict(X_scaled: np.ndarray) -> np.ndarray:
        tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            out = model(tensor).numpy()
        return out.reshape(X_scaled.shape[0], output_len, n_features)

    return predict


def make_tft_predict(data: dict, cfg: dict, ckpt_path: Path):
    """TFT requires rebuilding a TimeSeriesDataSet for each masked input.

    The heavy lift (checkpoint load) happens once per seed; the per-variable
    inner loop pays the dataloader-construction + predict cost.
    """
    tft = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))
    tft.eval()
    n_features = len(data["var_names"])
    output_len = cfg["output_len"]
    input_len = cfg["input_len"]
    batch_size = cfg["batch_size"]
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    train_df = make_split_frame(data["train_scaled"], data["var_names"])
    template_ds = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=data["var_names"],
        group_ids=["group_id"],
        min_encoder_length=input_len,
        max_encoder_length=input_len,
        min_prediction_length=output_len,
        max_prediction_length=output_len,
        time_varying_unknown_reals=data["var_names"],
        time_varying_known_reals=[],
        time_varying_known_categoricals=[],
        target_normalizer=MultiNormalizer(
            [TorchNormalizer(method="identity") for _ in range(n_features)]
        ),
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
        allow_missing_timesteps=False,
    )

    def predict(X_scaled: np.ndarray) -> np.ndarray:
        # The test partition is a single contiguous block; reconstruct a
        # length-n_test DataFrame whose sliding windows reproduce the
        # masked inputs the caller wants evaluated.
        test_scaled = data["test_scaled"].copy()
        # The inner loop always passes a single-variable-occluded (or
        # pristine) X_3d; reverse-engineer the test_scaled block by
        # replaying the windowing contract: test_scaled[i:i+input_len]
        # equals X_scaled[i]. Setting entire test_scaled column to 0 for
        # the occluded variable matches the intended occlusion because
        # the sliding-window stride is 1 and every test window shares
        # the overlapping rows.
        for var_idx in range(n_features):
            if np.allclose(X_scaled[:, :, var_idx], 0.0):
                test_scaled[:, var_idx] = 0.0
        test_df = make_split_frame(test_scaled, data["var_names"])
        test_ds = TimeSeriesDataSet.from_dataset(
            template_ds, test_df, stop_randomization=True
        )
        loader = test_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        raw = tft.predict(
            loader,
            mode="prediction",
            return_y=False,
            trainer_kwargs={"accelerator": accelerator, "devices": 1, "logger": False},
        )
        if isinstance(raw, (list, tuple)):
            stack = torch.stack(list(raw), dim=-1).cpu().numpy()
        else:
            stack = raw.cpu().numpy()
        if stack.ndim == 2:
            stack = stack[..., None]
        return stack

    return predict


# ---------------------------------------------------------------------------
# Run orchestrator
# ---------------------------------------------------------------------------


def _collect_rows(model: str, seed, rows: list[dict]) -> list[dict]:
    return [
        {"model": model, "seed": seed, **row}
        for row in rows
    ]


def run_common_importance(
    config_path: str | None = None,
    *,
    results_dir: str | None = None,
    checkpoints_dir: str | None = None,
) -> pd.DataFrame:
    cfg = apply_runtime_overrides(
        load_config(config_path),
        smoke=False,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
    )
    results_path = ensure_dir(cfg["results_dir"])
    checkpoints_path = Path(cfg["checkpoints_dir"])
    data = prepare_supervised_data(cfg["data_path"], cfg["input_len"], cfg["output_len"])
    var_names = data["var_names"]
    seeds = cfg["seeds"]

    print(f"[B7a] results_dir={results_path}")
    print(f"[B7a] n_test={data['n_test']}, n_features={len(var_names)}")
    print(f"[B7a] seeds={seeds}")

    all_rows: list[dict] = []

    # LR — deterministic single fit
    print("\n[LR] deterministic fit + occlusion...")
    lr_predict = make_lr_predict(data, cfg)
    lr_rows = compute_model_occlusion(lr_predict, data["X_test"], data["y_test"], var_names)
    all_rows.extend(_collect_rows("LR", "deterministic", lr_rows))

    # MLP — per seed
    print("\n[MLP] occlusion across seeds...")
    for seed in seeds:
        ckpt = checkpoints_path / f"mlp_seed{seed}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing MLP checkpoint: {ckpt}")
        predict_fn = make_mlp_predict(data, cfg, ckpt)
        rows = compute_model_occlusion(predict_fn, data["X_test"], data["y_test"], var_names)
        all_rows.extend(_collect_rows("MLP", seed, rows))
        print(f"  seed {seed}: baseline={rows[0]['baseline_mse']:.4f}")

    # LSTM — per seed
    print("\n[LSTM] occlusion across seeds...")
    for seed in seeds:
        ckpt = checkpoints_path / f"lstm_seed{seed}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing LSTM checkpoint: {ckpt}")
        predict_fn = make_lstm_predict(data, cfg, ckpt)
        rows = compute_model_occlusion(predict_fn, data["X_test"], data["y_test"], var_names)
        all_rows.extend(_collect_rows("LSTM", seed, rows))
        print(f"  seed {seed}: baseline={rows[0]['baseline_mse']:.4f}")

    # TFT — per seed (heavier: rebuild dataloader per occlusion)
    print("\n[TFT] occlusion across seeds (dataloader rebuilt per variable; slowest model)...")
    for seed in seeds:
        ckpt = checkpoints_path / f"tft_seed{seed}.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing TFT checkpoint: {ckpt}")
        predict_fn = make_tft_predict(data, cfg, ckpt)
        rows = compute_model_occlusion(predict_fn, data["X_test"], data["y_test"], var_names)
        all_rows.extend(_collect_rows("TFT", seed, rows))
        print(f"  seed {seed}: baseline={rows[0]['baseline_mse']:.4f}")

    out_df = pd.DataFrame(all_rows)
    out_csv = results_path / "occlusion_importance.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n[B7a] saved {out_csv} ({len(out_df)} rows)")
    return out_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="B7a — common, model-agnostic occlusion importance for all 4 core models."
    )
    parser.add_argument("--config", default="configs/experiments/fair_core_v2.yaml")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_common_importance(
        args.config,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
    )


if __name__ == "__main__":
    main()
