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
| Seeds | 42, 123, 456 |

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
| Seeds | 42, 123, 456 |

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
| Seeds | 42, 123, 456 |

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

| Conceptual component | File(s) | Output artefact(s) |
|----------------------|---------|---------------------|
| Data processing | `src/training/multi_seed.py` (lines ~188–227); `src/training/tft_fair_3seed.py` (per-split DataFrame construction) | scaled tensors in memory |
| LR training path | `src/training/multi_seed.py` (LR pathway) | `results/multi_seed_fair_baseline.csv` (LR row) |
| MLP training path | `src/training/multi_seed.py` (MLP pathway) | `results/multi_seed_fair_baseline.csv` (MLP rows); `checkpoints/mlp_seed{42,123,456}.pt` |
| LSTM training path | `src/training/multi_seed.py` (LSTM pathway) | `results/multi_seed_fair_baseline.csv` (LSTM rows); `checkpoints/lstm_seed{42,123,456}.pt` |
| TFT training path | `src/training/tft_fair_3seed.py` | `results/tft_summary.csv`; `results/tft_metrics.csv`; `results/tft_training_curves.csv`; `results/tft_importance.csv`; `checkpoints/tft_seed{42,123,456}.ckpt` |
| Prediction export | `src/evaluation/export_predictions.py` | `results/preds_lr.npy`; `results/preds_{mlp,lstm,tft}_seed{42,123,456}.npy` (10 files, shape `(3 365, 24, 7)`) |
| Post-training analysis (per-seed metrics + § 14 complexity + § 16 bootstrap, S-09c merge) | `src/evaluation/post_training_analysis.py` | `results/per_seed_metrics.csv`; `results/complexity_metrics.csv`; `results/bootstrap_intervals.csv` |
| Original-scale evaluation | `src/training/multi_seed.py` (`compute_original_metrics`); `src/training/tft_fair_3seed.py` (same function) | `results/multi_seed_fair_baseline.csv`; `results/tft_summary.csv` |
| Explainability — SHAP | `src/explainability/shap_lr.py`; `src/explainability/shap_mlp.py`; `src/explainability/shap_lstm.py` (PENDING) | `results/shap_{lr,mlp,lstm}.csv` |
| Explainability — VSN | `src/training/tft_fair_3seed.py` (TFT VSN extraction) | `results/tft_importance.csv` |
| Faithfulness layer | `src/explainability/faithfulness_test.py` (PENDING; S-10b rewrite: checkpoint-based, 4 model × 3 seed × 7 config) | `results/faithfulness.csv` |
| Cross-model XAI agreement (S-10 strong-package addition) | `src/explainability/cross_model_xai_agreement.py` (PENDING; Spearman + Kendall on per-variable importance vectors; 6 pairs) | `results/xai_agreement.csv` |
| Accuracy ↔ interpretability trade-off plot (S-10 strong-package addition; RQ3 artefact) | `src/explainability/trade_off_plot.py` (PENDING; 2D scatter of overall MSE vs faithfulness score) | `results/trade_off_data.csv`; `results/trade_off_plot.png` |
| Per-horizon disaggregated metrics (S-10c; descriptive only) | `src/evaluation/per_horizon_metrics.py` (PENDING; 4 models × 24 horizons; § 11C kısmî horizon-axis lifting) | `results/per_horizon_metrics.csv` |

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

| Model | Seed | MSE       | MAE       | RMSE      | Best epoch |
|-------|------|-----------|-----------|-----------|------------|
| LR    | —    | 7.660473  | 1.465105  | 2.767756  | — (closed-form) |
| MLP   | 42   | 9.479172  | 1.829385  | 3.078826  | 13 |
| MLP   | 123  | 9.253977  | 1.774533  | 3.042035  | 19 |
| MLP   | 456  | 9.338430  | 1.788149  | 3.055884  | 21 |
| LSTM  | 42   | 12.927132 | 2.178683  | 3.595432  | 19 |
| LSTM  | 123  | 12.939920 | 2.152492  | 3.597210  | 34 |
| LSTM  | 456  | 12.133417 | 2.085681  | 3.483305  | 42 |
| TFT   | 42   | 13.844168 | 2.353615  | 3.720775  | 5 |
| TFT   | 123  | 14.973468 | 2.365260  | 3.869557  | 14 |
| TFT   | 456  | 13.741943 | 2.246574  | 3.707013  | 9 |

All per-seed values pass internal consistency against the headline summary in § 3.8.1:

- MLP: mean = 9.357193, std (`ddof=1`) = 0.113764 → **9.36 ± 0.11** ✓
- LSTM: mean = 12.666823, std (`ddof=1`) = 0.461988 → **12.67 ± 0.46** ✓
- TFT: mean = 14.186526, std (`ddof=1`) = 0.682947 → **14.19 ± 0.68** ✓

Any cell-level discrepancy versus the headline summary would indicate a recomputation drift and would be auditable by `ml-experiment-auditor`.

### F.2 Pairwise paired-bootstrap intervals — full set

The headline § 3.8.2 table reports the six MSE intervals. The full 18-row table (six pairs × three metrics) from `results/bootstrap_intervals.csv` is reproduced below.

| Pair (A vs B) | Metric | Mean diff (A − B) | 95 % CI low | 95 % CI high | Bootstrap N | Bootstrap seed | n_test_windows |
|---------------|--------|--------------------|--------------|---------------|--------------|------------------|-------------------|
| LR vs MLP     | MSE    | −1.302982          | −1.39163     | −1.216635     | 10 000       | 2026             | 3 365             |
| LR vs MLP     | MAE    | −0.271331          | −0.280503    | −0.261997     | 10 000       | 2026             | 3 365             |
| LR vs MLP     | RMSE   | −0.226134          | −0.241444    | −0.211103     | 10 000       | 2026             | 3 365             |
| LR vs LSTM    | MSE    | −4.256782          | −4.43794     | −4.075003     | 10 000       | 2026             | 3 365             |
| LR vs LSTM    | MAE    | −0.595211          | −0.60995     | −0.580781     | 10 000       | 2026             | 3 365             |
| LR vs LSTM    | RMSE   | −0.684343          | −0.712757    | −0.655848     | 10 000       | 2026             | 3 365             |
| LR vs TFT     | MSE    | −5.141411          | −5.314982    | −4.963153     | 10 000       | 2026             | 3 365             |
| LR vs TFT     | MAE    | −0.765015          | −0.779915    | −0.74971      | 10 000       | 2026             | 3 365             |
| LR vs TFT     | RMSE   | −0.810168          | −0.83642     | −0.783357     | 10 000       | 2026             | 3 365             |
| MLP vs LSTM   | MSE    | −2.9538            | −3.155955    | −2.758215     | 10 000       | 2026             | 3 365             |
| MLP vs LSTM   | MAE    | −0.323879          | −0.339046    | −0.309267     | 10 000       | 2026             | 3 365             |
| MLP vs LSTM   | RMSE   | −0.458209          | −0.489132    | −0.428008     | 10 000       | 2026             | 3 365             |
| MLP vs TFT    | MSE    | −3.838429          | −4.013034    | −3.663545     | 10 000       | 2026             | 3 365             |
| MLP vs TFT    | MAE    | −0.493683          | −0.506076    | −0.481195     | 10 000       | 2026             | 3 365             |
| MLP vs TFT    | RMSE   | −0.584034          | −0.609719    | −0.558543     | 10 000       | 2026             | 3 365             |
| LSTM vs TFT   | MSE    | −0.884629          | −1.025084    | −0.74296      | 10 000       | 2026             | 3 365             |
| LSTM vs TFT   | MAE    | −0.169804          | −0.182277    | −0.157697     | 10 000       | 2026             | 3 365             |
| LSTM vs TFT   | RMSE   | −0.125825          | −0.145574    | −0.105736     | 10 000       | 2026             | 3 365             |

All 18 intervals exclude zero, so the corresponding mean per-window error differences are statistically distinguishable at the bootstrap-CI level under the present configuration. The independence caveat of § 4.4.1 applies.

### F.3 Complexity metrics — full table

| Model | n_params | Architectural category | Wall-clock note | Hardware note |
|-------|----------|------------------------|------------------|----------------|
| LR    | 113 064  | linear                 | < 1 s (closed-form OLS; no iterative training) | single NVIDIA RTX 5080 Laptop GPU (CUDA), batch_size = 64 |
| MLP   | 124 328  | shallow-MLP            | not instrumented (§ 14 secondary; the training loop is not wrapped with `time.perf_counter()`; training-curves CSV captures epoch progression but not wall-clock seconds) | single NVIDIA RTX 5080 Laptop GPU (CUDA), batch_size = 64 |
| LSTM  | 29 608   | recurrent              | not instrumented (§ 14 secondary; same instrumentation gap as MLP) | single NVIDIA RTX 5080 Laptop GPU (CUDA), batch_size = 64 |
| TFT   | 18 261   | transformer-family     | best-effort estimate ≈ 915 s total across 3 seeds (per-seed approximately 240 / 375 / 300 s; ≈ 15 s/epoch × epoch counts from `results/tft_training_curves.csv`) | single NVIDIA RTX 5080 Laptop GPU (CUDA), batch_size = 64 |

## G. Extra SHAP / VSN Plots

The SHAP per-feature attribution profiles for LR, MLP, and LSTM and the VSN importance vector for TFT are written by the explainability scripts (§ 3.6 / appendix C). Because the SHAP and faithfulness scripts are PENDING in `docs/VALIDATION_LOG.md`, no plots are inserted in this appendix at the present revision. The dedicated input-perturbation stability layer was dropped in the S-11 archive migration (2026-04-20); stability evidence in the thesis comes from the 3-seed training pipeline's seed-variance signal (§ 2.9 methodology). Once the remaining rows in the validation log read VALIDATED, this section will be filled with the per-feature heatmaps and rank-order comparisons described in § 4.2.

## H. Implementation Notes

A small number of implementation decisions are recorded here for reuse without being load-bearing on the main argument.

*Identity normaliser inside `pytorch-forecasting`.* The TFT path uses an identity normaliser inside the framework's `TimeSeriesDataSet` to avoid a double-scaling pathology that would otherwise occur because the inputs are already standardised by the shared `StandardScaler` in the data-processing component. This is documented in the TFT Comparison Card of `docs/EXPERIMENT_MANIFEST.md` and is the mechanism by which TEXT-02 (TFT preprocessing admissibility under § 11A) clears.

*Pickle-resolution fix in `export_predictions.py`.* The custom MSE class introduced in S-02b is defined at module level in `src/evaluation/export_predictions.py` so that the pickled TFT checkpoints can be unpickled successfully when loading via `load_from_checkpoint`. This is a `__main__`-vs-module namespace detail and is documented for future maintainers.

*Fail-fast guard at the restore-best step.* The MLP / LSTM training paths include a guard that raises `RuntimeError` if `best_model_state` is `None` at the point of restoration; this would surface a silent training-collapse failure mode in which no validation step ever produced a finite loss. The guard never fires in the locked baseline runs and therefore does not appear in the headline output, but it is a non-trivial robustness lever and is mentioned in § 4.4.1.

*Train-loss vs val-loss reporting asymmetry (cosmetic).* `train_loss` in `results/training_curves.csv` is the mean of per-batch means over the training epoch, whereas `val_loss` is computed in a single full-set pass. This is a cosmetic inconsistency in the training-curves CSV (CURVE-NOTE-01 in the audit log) that has no effect on selection or test metrics; it is recorded here so that any reuse of the training-curves CSV for subsequent diagnostic work uses the appropriate interpretation.
