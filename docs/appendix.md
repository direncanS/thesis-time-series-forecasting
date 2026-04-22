# Appendix

<!-- Main text must stand alone without it (§ 7). -->
<!-- S-07b 2026-04-19: lifted from skeleton via thesis-writer revise mode. -->
<!-- File names and implementation snippets are admissible here (CLAUDE.md skeleton note). -->

## A. Full Hyperparameter Tables

The values below are the locked implementation defaults of § 10. They are reproduced from `CLAUDE.md` and `docs/EXPERIMENT_MANIFEST.md` and are not changed across any reported run.

### A.1 Linear Regression (LR)

| Field | Value |
|-------|-------|
| Solver | Closed-form OLS (`sklearn.linear_model.LinearRegression`) |
| Input dimensionality | 672 (96 × 7, flattened) |
| Output dimensionality | 168 (24 × 7, flattened) |
| Optimiser | N/A (deterministic; § 11A item 8 deterministic-baseline exception) |
| Learning rate | N/A |
| Validation monitor | N/A (§ 11A item 9 exception) |
| Checkpoint | N/A |
| Seeds | N/A (deterministic) |

### A.2 MLP

| Field | Value |
|-------|-------|
| Architecture | 672 → 128 → 128 → 168 (ReLU activations) |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| Loss | `nn.MSELoss()` |
| Batch size | 64 |
| Patience | 10 |
| Max epochs | 200 |
| Weight decay | 0.0 |
| Validation monitor | val_loss (MSE on validation windows) |
| Checkpoint rule | Restore-best-weights via `load_state_dict` |
| Seeds | 42, 123, 456, 789, 1024 |

### A.3 LSTM

| Field | Value |
|-------|-------|
| Architecture | Single LSTM layer (hidden = 64, 1 layer); last-hidden → linear → 168 |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| Loss | `nn.MSELoss()` |
| Batch size | 64 |
| Patience | 10 |
| Max epochs | 200 |
| Weight decay | 0.0 |
| Validation monitor | val_loss (MSE) |
| Checkpoint rule | Restore-best-weights via `load_state_dict` |
| Seeds | 42, 123, 456, 789, 1024 |

### A.4 TFT

| Field | Value |
|-------|-------|
| Framework | `pytorch-forecasting` 1.6.1 on Lightning 2.6.1 |
| Hidden size | 16 |
| Attention heads | 4 |
| Hidden continuous size | 8 |
| Dropout | 0.1 |
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Loss | Custom MSE class subclassing `MultiHorizonMetric` with `reduction="mean"` (LOSS-SYMMETRY-01 PASS — symmetric with `nn.MSELoss()` in the core models) |
| Batch size | 64 |
| Patience | 10 |
| Max epochs | 200 |
| Weight decay | 0.0 |
| Information regime | Past-only (`time_varying_known_reals=[]`) |
| Target scope | All 7 ETTh1 variables (multivariate) |
| Output | Deterministic point forecast (no quantile / probabilistic head) |
| Validation monitor | val_loss (sum of per-target MSE) |
| Checkpoint rule | Lightning `ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)` + `load_from_checkpoint(best_path)` |
| Seeds | 42, 123, 456, 789, 1024 |

### A.5 Pipeline-level constants

| Field | Value |
|-------|-------|
| Dataset | ETTh1 (17 420 hourly observations × 7 variables) |
| Split | 60 / 20 / 20 chronological (train / val / test) |
| Scaler | `StandardScaler` fit on training partition only |
| INPUT_LEN | 96 |
| OUTPUT_LEN | 24 |
| Test window count (aligned) | 3 365 |
| Bootstrap N | 10 000 |
| BOOTSTRAP_SEED | 2026 |
| SHAP N_EVAL | 100 |
| SHAP EVAL_SEED | 42 |
| SHAP N_BG | 100 (MLP, LSTM); full training set (LR, `LinearExplainer` closed-form) |
| SHAP BG_SEED | 42 (MLP, LSTM); not applicable (LR) |
| SHAP baseline | training-distribution mean in the scaled input space — zero-vector by `StandardScaler` construction; see methodology § 2.9.0 |

## B. Additional Result Plots

The following plot artefacts are produced by the pipeline and are referenced in the main text but are not embedded inline:

- Per-seed validation-loss training curves for MLP and LSTM (`results/training_curves.csv`).
- Per-seed validation-loss training curves for TFT (`results/tft_training_curves.csv`).

Combined-overlay plots were previously planned via `src/utils/prediction_vis_combined.py`, which was dropped from the pipeline on 2026-04-20 (S-11 archive migration; the script assumed an OT-only TFT target inconsistent with § 10). The thesis therefore does not include rendered combined-overlay plots; source CSVs above remain the per-model reference.

## C. Code Structure Reference

The mapping below relates the conceptual components of § 3.2 to the implementation files in `src/` and the artefacts in `results/` and `checkpoints/`. This is the only place in the thesis where file names appear (§ 5; main text is component-level).

The `src/` tree is organised by concern into four subfolders (S-09c refactor 2026-04-20):

```
src/
├── training/         core-model training paths
├── evaluation/       prediction export + post-training analysis + per-horizon metrics
└── explainability/   SHAP + faithfulness + cross-model agreement + trade-off plot
```

(`src/utils/` now holds only a package-marker `__init__.py` after the S-11 archive migration 2026-04-20: its prior inhabitants — `verify_reproducibility.py`, `eval_original_scale.py`, `prediction_vis_combined.py` — all assumed an OT-only TFT regime inconsistent with § 10 fair-core and were moved to `archive/superseded_src/`.)

<!-- @begin-include _generated/pipeline_status_table.md -->
| Conceptual component | File(s) | Status | Output artefact(s) |
|----------------------|---------|--------|---------------------|
| Data processing | `src/training/multi_seed.py`; `src/training/tft_fair_3seed.py` | **VALIDATED** | scaled tensors in memory |
| LR training path | `src/training/multi_seed.py` (LR pathway) | **VALIDATED** | `results/multi_seed_fair_baseline.csv` (LR row) |
| MLP training path | `src/training/multi_seed.py` (MLP pathway) | **VALIDATED** | `results/multi_seed_fair_baseline.csv` (MLP rows); `checkpoints/mlp_seed*.pt` |
| LSTM training path | `src/training/multi_seed.py` (LSTM pathway) | **VALIDATED** | `results/multi_seed_fair_baseline.csv` (LSTM rows); `checkpoints/lstm_seed*.pt` |
| TFT training path | `src/training/tft_fair_3seed.py` | **VALIDATED** | `results/tft_summary.csv`; `results/tft_metrics.csv`; `results/tft_training_curves.csv`; `results/tft_importance.csv`; `checkpoints/tft_seed*.ckpt` |
| Prediction export | `src/evaluation/export_predictions.py` | **VALIDATED** | `results/preds_*.npy` (shape `(n_test, 24, 7)`) |
| Post-training analysis (per-seed + complexity + bootstrap) | `src/evaluation/post_training_analysis.py` | **VALIDATED** | `results/per_seed_metrics.csv`; `results/complexity_metrics.csv`; `results/bootstrap_intervals.csv` + `bootstrap_block_sensitivity.csv` |
| Explainability — SHAP | `src/explainability/shap_lr.py`; `src/explainability/shap_mlp.py`; `src/explainability/shap_lstm.py` | **VALIDATED** | `results/shap_{lr,mlp,lstm}.csv` |
| Explainability — VSN | `src/training/tft_fair_3seed.py` (TFT VSN extraction) | **VALIDATED** | `results/tft_importance.csv` |
| Faithfulness layer | `src/explainability/faithfulness_test.py` | **VALIDATED** | `results/faithfulness.csv` |
| Cross-model XAI agreement | `src/explainability/cross_model_xai_agreement.py` | **VALIDATED** | `results/xai_agreement.csv` |
| Accuracy ↔ interpretability trade-off plot | `src/explainability/trade_off_plot.py` | **VALIDATED** | `results/trade_off_data.csv`; `results/trade_off_plot.png` |
| Per-horizon disaggregated metrics | `src/evaluation/per_horizon_metrics.py` | **VALIDATED** | `results/per_horizon_metrics.csv` |
<!-- @end-include -->

## D. Configuration Files

The pipeline does not use external configuration files. All run-time constants are encoded as module-level constants inside the relevant `src/` files; the canonical reference for these constants is `CLAUDE.md` § 10 (locked implementation defaults). The Comparison Cards in `docs/EXPERIMENT_MANIFEST.md` reproduce the per-model field set in tabular form.

## E. Reproducibility Verification — Tier 1 vs Tier 2

The checklist below describes the reproducibility evidence the thesis relies on. A dedicated automation script was prototyped (`src/utils/verify_reproducibility.py`) but was dropped on 2026-04-20 during the S-11 archive migration because it reconstructed TFT under the OT-only + future-covariates regime (inconsistent with § 10 fair-core) and re-trained the model inside the check rather than loading the authoritative `checkpoints/tft_seed*.ckpt`. A fair-core replacement (checkpoint-load witness only; no re-train) is future work. Until then, the individual checks listed below are verified *ad hoc* from the current artefacts in `results/` + `checkpoints/`.

| ID | Check | Tier |
|----|-------|------|
| RV-01 | LR coefficient matrix byte-identical across reruns | Tier 1 |
| RV-02 | LR prediction tensor (`results/preds_lr.npy`) byte-identical across reruns | Tier 1 |
| RV-03 | MLP / LSTM prediction tensors byte-identical across reruns from the same checkpoint | Tier 1 |
| RV-04 | TFT prediction tensors byte-identical across reruns from the same checkpoint | Tier 1 |
| RV-05 | Metric CSV row-set byte-identical across reruns (deterministic re-export) | Tier 1 |
| RV-06 | Bootstrap intervals byte-identical for fixed `BOOTSTRAP_SEED` and fixed predictions | Tier 1 |
| RV-07 | Complexity metrics CSV byte-identical across reruns | Tier 1 |
| RV-08 | MLP MSE rerun within ±0.5 of recorded baseline (CONSIST-02 tolerance) | Tier 2 |
| RV-09 | LSTM MSE rerun within ±1.0 of recorded baseline (CONSIST-03 tolerance) | Tier 2 |
| RV-10 | TFT MSE rerun within ±2 × seed-std of recorded baseline | Tier 2 |
| RV-11 | Per-seed best-epoch values within recorded ranges (CONSIST-05 / -06) | Tier 2 |
| RV-12 | Window counts identical (3 365 test windows, fixed split) | Tier 1 |
| RV-13 | Train-only scaler fit verified by inspection of fitted statistics | Tier 1 |

## F. Extended Result Tables

### F.1 Per-seed point metrics (original scale)

The headline § 3.8.1 table reports mean ± std across seeds. The per-seed values used to compute those summaries are reproduced below for transparency. TFT per-seed values are read directly from `results/tft_metrics.csv`; LR is deterministic. MLP and LSTM per-seed values are produced by the post-training analysis helper `src/evaluation/post_training_analysis.py` (Section 1, which loads the frozen prediction tensors `results/preds_<model>_seed<s>.npy` and reapplies `compute_original_metrics`); the cells marked `[run helper]` will be filled in once the helper has been run.

<!-- @begin-include _generated/per_seed_table.md -->
| Model | Seed | MSE       | MAE       | RMSE      | Best epoch |
|-------|------|-----------|-----------|-----------|------------|
| LR    | —    |  7.660473 |  1.465105 |  2.767756 | — (closed-form) |
| MLP   | 42   |  9.607342 |  1.810953 |  3.099571 | 13 |
| MLP   | 123  |  9.378154 |  1.820981 |  3.062377 | 24 |
| MLP   | 456  |  9.570781 |  1.807955 |  3.093668 | 11 |
| MLP   | 789  |  9.444790 |  1.807262 |  3.073238 | 23 |
| MLP   | 1024 |  9.336527 |  1.794262 |  3.055573 | 20 |
| LSTM  | 42   | 12.795810 |  2.092083 |  3.577123 | 25 |
| LSTM  | 123  | 12.807588 |  2.171723 |  3.578769 | 28 |
| LSTM  | 456  | 13.574508 |  2.205135 |  3.684360 | 36 |
| LSTM  | 789  | 13.020063 |  2.184277 |  3.608332 | 24 |
| LSTM  | 1024 | 12.390662 |  2.064890 |  3.520037 | 20 |
| TFT   | 42   | 13.843709 |  2.353622 |  3.720714 | 5 |
| TFT   | 123  | 14.990285 |  2.366630 |  3.871729 | 14 |
| TFT   | 456  | 13.744812 |  2.247099 |  3.707400 | 9 |
| TFT   | 789  | 12.625386 |  2.217307 |  3.553222 | 34 |
| TFT   | 1024 | 26.500271 |  2.938889 |  5.147841 | 1 |
<!-- @end-include -->

### F.2 Pairwise paired moving-block bootstrap intervals — full set

The headline § 3.8.2 table reports the six MSE intervals. The full 18-row table (six pairs × three metrics) from `results/bachelor_safe_v2/bootstrap_intervals.csv` is reproduced below. All intervals are computed with `arch.bootstrap.MovingBlockBootstrap` at block length **L = 96** (= `INPUT_LEN`), `BOOTSTRAP_SEED = 2026`, and `N = 10 000` resamples.

<!-- @begin-include _generated/bootstrap_full.md -->
| Pair (A vs B) | Metric | Mean diff (A − B) | 95 % CI low | 95 % CI high | Bootstrap N | Bootstrap seed | Block length | n_test_windows |
|---------------|--------|--------------------|--------------|---------------|--------------|------------------|---------------|-------------------|
| LR vs MLP | MSE | -1.270611 | -1.583030 | -0.880680 | 10 000 | 2026 | 96 | 3 365 |
| LR vs MLP | MAE | -0.260448 | -0.301583 | -0.206169 | 10 000 | 2026 | 96 | 3 365 |
| LR vs MLP | RMSE | -0.248450 | -0.298103 | -0.177971 | 10 000 | 2026 | 96 | 3 365 |
| LR vs LSTM | MSE | -4.413084 | -5.318296 | -3.410152 | 10 000 | 2026 | 96 | 3 365 |
| LR vs LSTM | MAE | -0.586847 | -0.658561 | -0.497758 | 10 000 | 2026 | 96 | 3 365 |
| LR vs LSTM | RMSE | -0.739074 | -0.871760 | -0.575593 | 10 000 | 2026 | 96 | 3 365 |
| LR vs TFT | MSE | -5.791200 | -6.741966 | -4.612476 | 10 000 | 2026 | 96 | 3 365 |
| LR vs TFT | MAE | -0.799158 | -0.873996 | -0.687095 | 10 000 | 2026 | 96 | 3 365 |
| LR vs TFT | RMSE | -0.944549 | -1.071772 | -0.767360 | 10 000 | 2026 | 96 | 3 365 |
| MLP vs LSTM | MSE | -3.142473 | -4.199145 | -2.094700 | 10 000 | 2026 | 96 | 3 365 |
| MLP vs LSTM | MAE | -0.326398 | -0.413099 | -0.238654 | 10 000 | 2026 | 96 | 3 365 |
| MLP vs LSTM | RMSE | -0.490624 | -0.649368 | -0.324561 | 10 000 | 2026 | 96 | 3 365 |
| MLP vs TFT | MSE | -4.520589 | -5.594069 | -3.324719 | 10 000 | 2026 | 96 | 3 365 |
| MLP vs TFT | MAE | -0.538710 | -0.617334 | -0.442398 | 10 000 | 2026 | 96 | 3 365 |
| MLP vs TFT | RMSE | -0.696099 | -0.841591 | -0.525371 | 10 000 | 2026 | 96 | 3 365 |
| LSTM vs TFT | MSE | -1.378116 | -1.730803 | -0.885637 | 10 000 | 2026 | 96 | 3 365 |
| LSTM vs TFT | MAE | -0.212311 | -0.256155 | -0.153113 | 10 000 | 2026 | 96 | 3 365 |
| LSTM vs TFT | RMSE | -0.205475 | -0.264532 | -0.131254 | 10 000 | 2026 | 96 | 3 365 |
<!-- @end-include -->

All 18 intervals exclude zero, so the corresponding mean per-window error differences remain distinguishable at the moving-block bootstrap-CI level with block length L = 96 under the present configuration. The § 16 safety sentence continues to apply; the overlapping-sliding-window dependence is method-matched by the block variant rather than silently assumed away.

**Block length sensitivity.** Table `results/bachelor_safe_v2/bootstrap_block_sensitivity.csv` reports the 95 % CI widths at L ∈ {24, 48, 96, 192} for each (pair × metric) combination. Widths remain stable across block lengths (variation well within 10 % of the headline widths), confirming that L = 96 is not a knife-edge choice; the headline moving-block intervals are robust to reasonable L perturbations.

### F.3 Complexity metrics — full table

<!-- @begin-include _generated/complexity_table.md -->
| Model | n_params | Architectural category | Wall-clock seconds | Hardware note |
|-------|----------|------------------------|---------------------|----------------|
| LR    | 113 064 | linear                 | 0.7 s               | single local machine run; bachelor-safe local-only workflow |
| MLP   | 124 328 | shallow-MLP            | 28.2 s              | single local machine run; bachelor-safe local-only workflow |
| LSTM  |  29 608 | recurrent              | 44.9 s              | single local machine run; bachelor-safe local-only workflow |
| TFT   |  18 261 | transformer-family     | 2373.4 s            | single local machine run; bachelor-safe local-only workflow |
<!-- @end-include -->

## G. Extra SHAP / VSN Plots

The SHAP per-feature attribution profiles for LR, MLP, and LSTM and the VSN importance vector for TFT are written by the explainability scripts (§ 3.6 / appendix C). Under the v6.1 plan the primary cross-model explanation instrument is model-agnostic occlusion importance (see § 4.2); SHAP and VSN are reported as auxiliary architecture-native explanations per CLAUDE.md § 13. The dedicated input-perturbation stability layer was dropped in the S-11 archive migration (2026-04-20); stability evidence in the thesis comes from the five-seed training pipeline's seed-variance signal (§ 2.9 methodology).

<!-- @begin-include _generated/status_summary.md -->
Tables and figures in this section are auto-generated from v2 fair-core rerun artefacts (`results/bachelor_safe_v2/`). The training and post-training analysis layers (`multi_seed.py`, `tft_fair_3seed.py`, `export_predictions.py`, `post_training_analysis.py`) are **VALIDATED** in `docs/VALIDATION_LOG.md`. The explainability layer (SHAP for LR/MLP/LSTM + VSN for TFT + faithfulness test) is **VALIDATED**.
<!-- @end-include -->

## H. Implementation Notes

A small number of implementation decisions are recorded here for reuse without being load-bearing on the main argument.

*Identity normaliser inside `pytorch-forecasting`.* The TFT path uses an identity normaliser inside the framework's `TimeSeriesDataSet` to avoid a double-scaling pathology that would otherwise occur because the inputs are already standardised by the shared `StandardScaler` in the data-processing component. This is documented in the TFT Comparison Card of `docs/EXPERIMENT_MANIFEST.md` and is the mechanism by which TEXT-02 (TFT preprocessing admissibility under § 11A) clears.

*Pickle-resolution fix in `export_predictions.py`.* The custom MSE class introduced in S-02b is defined at module level in `src/evaluation/export_predictions.py` so that the pickled TFT checkpoints can be unpickled successfully when loading via `load_from_checkpoint`. This is a `__main__`-vs-module namespace detail and is documented for future maintainers.

*Fail-fast guard at the restore-best step.* The MLP / LSTM training paths include a guard that raises `RuntimeError` if `best_model_state` is `None` at the point of restoration; this would surface a silent training-collapse failure mode in which no validation step ever produced a finite loss. The guard never fires in the locked baseline runs and therefore does not appear in the headline output, but it is a non-trivial robustness lever and is mentioned in § 4.4.1.

*Train-loss vs val-loss reporting asymmetry (cosmetic).* `train_loss` in `results/training_curves.csv` is the mean of per-batch means over the training epoch, whereas `val_loss` is computed in a single full-set pass. This is a cosmetic inconsistency in the training-curves CSV (CURVE-NOTE-01 in the audit log) that has no effect on selection or test metrics; it is recorded here so that any reuse of the training-curves CSV for subsequent diagnostic work uses the appropriate interpretation.
