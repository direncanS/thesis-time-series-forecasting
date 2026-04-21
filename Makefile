# Bachelor-safe Makefile. Closure Plan v6.1 B3 deliverable.
#
# Assumes the `thesis` conda env is active. If PATH drift on Windows
# PowerShell leaves `python` pointing at the base env, override:
#
#   make test PYTHON="C:/Users/diren/miniconda3/envs/thesis/python.exe"
#
# Default config points at configs/experiments/fair_core_v2.yaml; override:
#
#   make train CONFIG=configs/experiments/other.yaml

PYTHON ?= python
CONFIG ?= configs/experiments/fair_core_v2.yaml

.PHONY: help test lint smoke \
        train train-core train-tft \
        analyze export-preds per-horizon bootstrap \
        tables tables-verify \
        all clean

help:
	@echo "Bachelor-safe Makefile (Closure Plan v6.1)"
	@echo "  test          pytest tests/ (unit + regression)"
	@echo "  lint          ruff check src tests scripts"
	@echo "  smoke         lightweight unit + smoke suite"
	@echo "  train-core    LR/MLP/LSTM training → results/bachelor_safe_v2/"
	@echo "  train-tft     TFT training"
	@echo "  train         train-core + train-tft"
	@echo "  export-preds  frozen prediction tensors (.npy) for downstream analysis"
	@echo "  per-horizon   per-horizon metrics CSV"
	@echo "  bootstrap     moving-block paired bootstrap + sensitivity table"
	@echo "  analyze       export-preds + per-horizon + bootstrap"
	@echo "  all           train + analyze + test"
	@echo "  clean         remove __pycache__ / .pytest_cache / .ruff_cache"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src tests scripts

smoke:
	$(PYTHON) -m pytest tests/test_preprocess.py tests/test_metrics.py tests/test_losses.py tests/test_smoke_pipeline.py -v

train-core:
	$(PYTHON) src/training/multi_seed.py --config $(CONFIG)

train-tft:
	$(PYTHON) src/training/tft_fair_3seed.py --config $(CONFIG)

train: train-core train-tft

export-preds:
	$(PYTHON) src/evaluation/export_predictions.py --config $(CONFIG)

per-horizon:
	$(PYTHON) src/evaluation/per_horizon_metrics.py --config $(CONFIG)

bootstrap:
	$(PYTHON) src/evaluation/post_training_analysis.py --config $(CONFIG)

analyze: export-preds per-horizon bootstrap

tables:
	$(PYTHON) scripts/regen_tables.py

tables-verify:
	$(PYTHON) scripts/regen_tables.py --verify

all: train analyze tables test

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.ruff_cache')]"
