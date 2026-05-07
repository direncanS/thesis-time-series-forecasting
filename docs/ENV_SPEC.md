# ENV_SPEC

Environment lock spec for the thesis reproducibility chain. Established originally (2026-04-20).

## Gate A decision (2026-04-20)

- **Path:** Preferred (manifest anchor).
- **Python:** 3.11
- **Torch wheel source:** CUDA 12.8 nightly (`--index-url https://download.pytorch.org/whl/nightly/cu128`)
- **Env isolation:** conda env (miniconda already installed; name: `thesis`)
- **Fallback cascade (if nightly wheel unavailable):** CUDA 12.1 stable → CUDA 11.8 stable → CPU-only (last resort, requires 4-location `map_location="cpu"` patch per plan v11).

## Pinned packages (manifest anchor)

| Package | Version | Source |
|---------|---------|--------|
| python | 3.11 | conda |
| torch | 2.11.0.dev20260203+cu128 | PyTorch nightly index |
| lightning | 2.6.1 | PyPI |
| pytorch-forecasting | 1.6.1 | PyPI |
| numpy | 2.2.6 | PyPI |

Additional packages (compatible ranges — not manifest-locked): pandas, scikit-learn, shap, matplotlib. See `requirements.txt`.

## Hardware target

- **GPU:** NVIDIA GeForce RTX 5080 Laptop GPU (16 GB, CUDA 13.1 driver 592.01 detected 2026-04-20; backward-compatible with CUDA 12.8 runtime).
- **CPU fallback available** but NOT RECOMMENDED — requires pre-patches.

## Install commands (Strategy A, two-stage; `conda run` pattern)

**IMPORTANT — shell-state persistence:** `conda activate thesis` in one shell call does NOT persist to the next call in tool-driven automation. Use `conda run -n thesis python -m pip ...` for every command so each invocation targets the right env deterministically.

```powershell
# 0) Create env (one-time)
conda create -n thesis python=3.11 -y

# 1) Verify driver/runtime BEFORE committing to cu128
nvidia-smi
# Expected: Driver Version >= 525 (for cu128 backward-compat); CUDA Version >= 12.8

# 2) Install torch nightly cu128 (separate command — not on PyPI default index)
conda run -n thesis python -m pip install --pre torch==2.11.0.dev20260203+cu128 `
 --index-url https://download.pytorch.org/whl/nightly/cu128

# 3) Install remaining packages from requirements.txt
conda run -n thesis python -m pip install -r requirements.txt
```

**Interactive alternative** (if running manually in PowerShell, not via tool automation):
```powershell
conda activate thesis
# ...then `pip install ...` works as normal within this PS session
```
Do NOT mix: if any shell calls are automated across tool invocations, stick to `conda run -n thesis python -m pip ...` throughout.

**If Stage 2 fails** (nightly wheel removed from index or network issue): drop to fallback cascade —

```powershell
# Fallback 1: CUDA 12.1 stable
conda run -n thesis python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# Fallback 2: CUDA 11.8 stable
conda run -n thesis python -m pip install torch --index-url https://download.pytorch.org/whl/cu118

# Fallback 3 (last resort): CPU-only + 4-location map_location patch per plan v11
conda run -n thesis python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Log the chosen path below under "Install log".

## Verification (two-branch per plan v11)

### Preferred env path PASS (use `conda run -n thesis python ...` for automation)

```powershell
conda run -n thesis python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
# Expected: 2.11.0.dev20260203+cu128 True 12.8

conda run -n thesis python -c "import numpy, pandas, sklearn, shap, pytorch_forecasting, lightning, matplotlib; print('3rd-party imports ok')"
conda run -n thesis python -c "import pytorch_forecasting, lightning, numpy; print('pf', pytorch_forecasting.__version__, 'lightning', lightning.__version__, 'numpy', numpy.__version__)"
# Expected: pf 1.6.1 lightning 2.6.1 numpy 2.2.6

conda run -n thesis python scripts/smoke_checkpoint.py
# Expected: 3 PASS lines (MLP 124,328 / LSTM 29,608 / TFT 18,261 param counts)
```

### Fallback env path PASS (versions relaxed)

- `torch.__version__` is a stable cu121 / cu118 / cpu wheel; `torch.cuda.is_available()` True for cu121/cu118, False for cpu-only (patches verified).
- `pytorch_forecasting.__version__ == 1.6.1` + `lightning.__version__ == 2.6.1` still pinned.
- `scripts/smoke_checkpoint.py` PASS (shape + finite + param counts are env-agnostic).
- Fallback path chosen + rationale logged below.

## Install log (2026-04-20)

- **Date:** 2026-04-20
- **Chosen path:** preferred — env `thesis` already existed with matching core stack.
- **Python version (actual):** 3.10.19 (manifest anchor was 3.11 — **minor drift accepted**; full core stack bit-matches manifest and all 3 checkpoint smokes PASS under 3.10, so recreation not justified).
- **Torch version actually installed:** 2.11.0.dev20260203+cu128 ✓ (matches manifest anchor exactly)
- **CUDA available:** True; `torch.version.cuda` = 12.8 ✓
- **Driver:** NVIDIA 592.01 (CUDA 13.1 driver, backward-compatible with cu128 runtime) ✓
- **Other anchor packages:** lightning 2.6.1 ✓, pytorch-forecasting 1.6.1 ✓, numpy 2.2.6 ✓
- **Compatible ranges (accepted):** pandas 2.3.3, sklearn 1.7.2, shap 0.49.1, matplotlib 3.10.8 — all within `requirements.txt` compatible ranges.
- **Smoke checkpoint PASS:** 3/3 — MLP (124,328 params + shape (1,168) + finite), LSTM (29,608 params + shape (1,168) + finite), TFT (18,261 params). Warnings cosmetic (triton flop counter, lightning save_hyperparameters on loss/logging_metrics).
- **Render toolchain:** pandoc + tectonic installed via `conda install -n thesis -c conda-forge pandoc tectonic -y` ; render smoke PASS (`docs/_smoke_render.pdf` 28 KB produced).
- **Notes:** Install steps A/B (torch nightly + requirements.txt) were unnecessary — env `thesis` pre-existed fully configured (probably from the session that produced the VALIDATED baselines). Python 3.10.19 vs 3.11 manifest drift logged; no action required (core stack bit-equivalent).

## Gate B update (2026-04-22 — lite reproducibility)

- **Status:** Conda env `thesis` formally extended from earlier to cover B3 artefacts (config + tests + lite lock).
- **New top-level files:** `environment.yml` (conda env spec + pip nightly wheel), `requirements.lock` (full `pip freeze` output, 200 lines), `pyproject.toml` (setuptools + pytest + ruff config), `Makefile` (PowerShell-safe targets).
- **Python version accepted:** 3.13.11 observed on 2026-04-22 (logged 3.10.19; user re-created env with 3.13). Core stack packages still bit-match anchors. Drift documented; no recreation.
- **B3 additions (pinned in `requirements.lock` after `pip install`):**

| Package | Installed version | Purpose |
|---------|-------------------|---------|
| PyYAML | 6.0.3 | `src/common.py::load_config` YAML parsing |
| arch | 8.0.0 | `src/evaluation/post_training_analysis.py` `MovingBlockBootstrap` |
| pytest | 9.0.3 | `tests/` runner |
| ruff | 0.15.11 | linter (`pyproject.toml::[tool.ruff]`) |
| statsmodels | 0.14.6 | `arch` transitive dependency |
| patsy | 1.0.2 | `arch` transitive dependency |

- **Bachelor-safe scope — NOT included:**
 - Dockerfile (cross-machine container)
 - GitHub Actions CI (hosted runners offer no GPU)
 - Second-machine / self-hosted GPU witness protocol
 - `pre-commit` framework (kept minimal; `.pre-commit-config.yaml` placeholder only)

- **Fresh-checkout setup:**
 ```powershell
 conda env create -f environment.yml
 conda activate thesis
 pip install -e .
 python -m pytest tests/ -v # must show 7+ PASS
 ```

- **Verification (2026-04-22 B3 close):** `pytest tests/` run under `thesis` env → all B2 regression + B3 unit + smoke tests PASS; `ruff check src tests scripts` → 0 errors (or documented via `[tool.ruff.lint.per-file-ignores]`).
