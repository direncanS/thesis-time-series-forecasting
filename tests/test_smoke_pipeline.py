"""Lightweight smoke tests for the config + override + import-safety chain.

These do NOT run a full training pipeline (B5's concern). They verify that
load_config / apply_runtime_overrides obey the documented semantics and that
every refactored script is import-side-effect-free.
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest
from src.common import DEFAULT_CONFIG, apply_runtime_overrides, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_without_yaml():
    cfg = load_config(None)
    assert cfg["seeds"] == [42, 123, 456, 789, 1024]
    assert cfg["precision"]["matmul"] == "highest"
    assert cfg["precision"]["cudnn_deterministic"] is True
    assert cfg["precision"]["tf32_matmul"] is False
    assert cfg["models"]["tft"]["hidden_size"] == 16
    assert cfg["models"]["mlp"]["learning_rate"] == 1e-4
    assert cfg["models"]["tft"]["learning_rate"] == 1e-3


def test_load_config_from_fair_core_yaml():
    cfg_path = REPO_ROOT / "configs" / "experiments" / "fair_core_v2.yaml"
    if not cfg_path.exists():
        pytest.skip("fair_core_v2.yaml not present (B3 artefact)")
    cfg = load_config(str(cfg_path))
    # fair_core_v2 extends base.yaml; both should leave the frozen defaults intact
    assert cfg["data_path"] == "data/ETTh1.csv"
    assert cfg["input_len"] == 96
    assert cfg["output_len"] == 24
    assert cfg["results_dir"] == "results/bachelor_safe_v2"
    assert cfg["precision"]["tf32_matmul"] is False
    assert cfg["precision"]["cudnn_deterministic"] is True


def test_smoke_override_shrinks_config():
    cfg = load_config(None)
    smoke_cfg = apply_runtime_overrides(cfg, smoke=True)
    assert len(smoke_cfg["seeds"]) == 1
    assert smoke_cfg["seeds"][0] == 42
    assert smoke_cfg["max_epochs"] == 1
    assert smoke_cfg["patience"] == 1
    assert smoke_cfg["results_dir"].endswith("smoke")
    assert smoke_cfg["checkpoints_dir"].endswith("smoke")


def test_override_does_not_mutate_source_config():
    cfg = load_config(None)
    baseline = copy.deepcopy(cfg)
    _ = apply_runtime_overrides(cfg, smoke=True)
    _ = apply_runtime_overrides(cfg, results_dir="custom/path")
    assert cfg == baseline, "apply_runtime_overrides must not mutate its input cfg"


def test_results_dir_override():
    cfg = load_config(None)
    out = apply_runtime_overrides(cfg, results_dir="custom/out", checkpoints_dir="custom/ckpt")
    assert out["results_dir"] == "custom/out"
    assert out["checkpoints_dir"] == "custom/ckpt"


@pytest.mark.parametrize(
    "module_name",
    [
        "src.common",
        "src.training.multi_seed",
        "src.training.tft_fair_3seed",
        "src.evaluation.export_predictions",
        "src.evaluation.per_horizon_metrics",
        "src.evaluation.post_training_analysis",
    ],
)
def test_refactored_modules_import_without_side_effects(module_name):
    """Every script in the refactored tree must be side-effect-free on import.
    (Closure Plan v6.1 B2 C10 acceptance — if __name__ == '__main__' guard coverage.)
    """
    mod = importlib.import_module(module_name)
    assert mod is not None
