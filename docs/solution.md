# 3. Solution

<!-- Target share: ~50 % — heaviest chapter. -->
<!-- S-07b 2026-04-19: lifted from skeleton via thesis-writer revise mode. -->
<!-- Claims about cross-model performance respect § 11C: overall observations only; per-variable / per-horizon claims forbidden until disaggregated path implemented. -->

## 3.1 Overview of the Developed Experimental Framework

The artifact developed in this thesis is a **reproducible forecasting evaluation framework** that runs four structurally different model families — Linear Regression (LR), a multilayer perceptron (MLP), an LSTM recurrent network, and a Temporal Fusion Transformer (TFT) — through a single, controlled pipeline that enforces identical task and information constraints across all of them. The framework is the contribution; the metric table it produces is its output, not its purpose. This framing follows directly from the artifact definition fixed in the project guidelines (§ 5) and from the FH thesis structure (§ 7), which assigns the largest weight of the main body to the Solution chapter.

The framework is purpose-built rather than assembled from off-the-shelf forecasting libraries. Standard libraries provide model implementations but do not enforce a single environment in which the same chronological split, the same input/output horizon, the same train-only scaling rule, the same metric-computation logic, the same seed-handling discipline, the same checkpoint-traceability rule, and a comparable interpretability protocol can be applied across model families. The Fairness and Comparability Protocol of the Methodology (§ 2.5) describes the eight-item rule each model must satisfy; the present chapter describes the engineering that makes the rule operational.

The pipeline ingests the ETTh1 hourly multivariate electricity-load dataset, runs all four core models under their locked hyperparameter defaults, exports an aligned set of per-test-window prediction tensors, computes original-scale point metrics with seed-level aggregation, and feeds two downstream layers — a paired moving-block bootstrap uncertainty layer (block length L = 96) and a complexity-reporting layer — that together produce the evidence base for the research-question discussion. A separate explainability workflow produces a model-agnostic occlusion-importance ranking for every (model, seed, variable) triple and evaluates those rankings against model behaviour through the AOPC faithfulness metric; auxiliary SHAP / VSN outputs from the v1 pipeline are retained for the construct-validity discussion.

## 3.2 Pipeline Architecture

The pipeline is composed of seven conceptual components arranged in a linear data flow with two parallel downstream branches:

1. **Data processing** — load, chronological split, train-only standardisation, sliding-window construction.
2. **Model-specific training** — three core training paths (LR closed-form solver; MLP / LSTM gradient-based with Adam; TFT under the `pytorch-forecasting` framework) each producing per-seed checkpoints.
3. **Best-checkpoint restoration** — early-stopping on validation loss with patience 10 and restore-best-weights logic identical across stochastic models.
4. **Prediction export** — frozen, original-scale per-test-window prediction tensors of shape `(3 365, 24, 7)` per model and per seed, written to disk as the canonical handoff to all downstream layers.
5. **Original-scale evaluation** — MSE / MAE / RMSE computed on the inverse-transformed prediction tensors with seed-level aggregation.
6. **Uncertainty layer** — paired **moving-block** bootstrap on aligned test-window prediction-error differences for the six pairwise comparisons.
7. **Complexity layer** — primary metric (parameter count) plus secondary contextual metrics (architectural category, wall-clock training-time best-effort, hardware note).

The explainability workflow (§ 3.6) operates as a parallel branch off the prediction-export component and consumes the same checkpoints used by the evaluation branch.

The architectural choice is deliberate: by routing every downstream consumer (evaluation, uncertainty, complexity, explainability) through a single frozen prediction artefact rather than through repeated model re-loading, the pipeline guarantees that bootstrap intervals and faithfulness tests all see exactly the same numbers as the headline metric table. This is the engineering mechanism that converts the Fairness and Comparability Protocol from a methodological intent into an operational property of the system.

## 3.3 Data Processing Component

The data-processing component is the first point at which the fairness rule must be enforced. It performs four operations in fixed order: load, split, scale, window.

**Load.** The ETTh1 hourly file (Zhou et al., 2021) is read once into memory; the seven covariate columns become both the input feature set and the target set, in line with the multivariate fair-core configuration (§ 2.4).

**Chronological split.** A 60 / 20 / 20 partition is applied by row index — the first 60 % of timestamps form the training partition, the next 20 % the validation partition, the last 20 % the test partition. No shuffling, no random sampling, no stratification: the split is purely temporal so that every model evaluates on a held-out *future* segment of the series.

**Train-only standardisation.** A `StandardScaler` is fit on the training partition only. The fitted scaler is then applied to validation and test partitions and to the inverse-transform step that returns predictions to original scale before metric computation. This isolates the test partition from any leakage of distributional information about the future — a precondition for the comparability of the four models, all of which consume the same scaled tensors.

**Sliding-window construction.** Windows of length 96 + 24 are constructed *separately within each split* (§ 2.3.1). The first 96 rows of the validation and test partitions are consumed as encoder context within their own split, not as forecast targets, so the effective forecast spans are `val[96:]` and `test[96:]`, yielding 3 365 aligned test windows for all four models. This split-then-window protocol is leak-free and exactly reproducible across model families. It is acknowledged, in the Methodology and again in § 4.4 below, as not equivalent to the alternative ETTh1 convention that constructs windows over the full scaled series and only afterwards assigns each window to a split.

The component's output is three pairs of `(X, y)` tensors (training, validation, test), all in scaled space, all of identical shape across models. From this point on, the four model-specific training paths consume identical input.

## 3.4 Training and Checkpointing Component

The training component runs each model under its locked hyperparameter defaults (§ 2.4 / § 2.7) and returns a per-seed best checkpoint. Three training paths exist, but they are wrapped in a single contract: same monitored quantity (validation loss), same patience (10), same restore-best-weights logic, same per-seed bookkeeping (training curves CSV, best-epoch index, best-val-loss value).

**LR path.** A closed-form ordinary-least-squares solver fits the flattened 96 × 7 → 168 input/output mapping in a single pass. The deterministic-baseline exception declared in the Comparison Card (§ 2.5) covers the absence of a validation monitor and a checkpoint file: the closed-form solver has a unique optimum and produces no training trajectory.

**MLP / LSTM path.** Both models are trained with the Adam optimiser at learning rate 1e-4, batch size 64, `max_epochs = 200`, and an early-stopping mechanism with `patience = 10` on validation loss. After each epoch the validation loss is computed in a single full-set pass; if it improves, the current weights are stored as the best-state-so-far. When training terminates (either by patience exhaustion or by hitting `max_epochs`), the best state is restored via `load_state_dict` before any test-time inference. A defensive guard at the restore step raises a `RuntimeError` if no finite validation loss was ever recorded — a fail-fast safety net that would surface a silent training collapse before it could leak into the headline numbers.

**TFT path.** The TFT training path is structurally analogous but uses Lightning's `ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)` together with `EarlyStopping(monitor="val_loss", patience=10)`; the best checkpoint is restored via `load_from_checkpoint(best_path)` before test-time prediction. The training loss is a custom MSE class that subclasses `MultiHorizonMetric` with `reduction="mean"`; this is symmetric with `nn.MSELoss()` in element-wise gradient behaviour. The earlier MAE-loss TFT variant was superseded by the present configuration after a loss-symmetry-fix cycle (LOSS-SYMMETRY-01) that resolved the prior cross-model training-objective asymmetry.

For each stochastic model, the training loop is repeated for five seeds (42, 123, 456, 789, 1024) using `torch.manual_seed`; the resulting per-seed checkpoints are persisted to `checkpoints/bachelor_safe_v2/`.

## 3.5 Evaluation Component

The evaluation component closes the loop on the original scale. For every (model, seed) combination, the best checkpoint is restored, predictions for the 3 365 test windows are produced in a single forward pass (or, for LR, a single matrix multiplication), and the inverse-transform step reverses the train-fit `StandardScaler` to return values to the original units of each variable.

Three point metrics are computed per (model, seed): **MSE**, **MAE**, and **RMSE**. Each is the `np.mean` of the per-element error tensor of shape (3 365, 24, 7). Per the four-axis aggregation contract (§ 2.10.1), this scalar mean averages simultaneously across the forecast horizon, across the seven variables, and across the test windows. The seed axis is summarised separately: for each metric, the per-seed scalars are reported as **mean ± std with `ddof = 1`** over seeds {42, 123, 456, 789, 1024}; for the deterministic LR path the seed axis is not applicable.

The component writes a single output table (`results/multi_seed_fair_baseline.csv` for LR / MLP / LSTM, `results/tft_summary.csv` for TFT) and a per-epoch training-curves CSV that retains the validation-loss trajectory for each (model, seed) pair. These artefacts are then consumed by the uncertainty and complexity layers below.

The disaggregation doctrine of § 11C is enforced at this layer: claims at the per-variable or per-horizon granularity are not supported by the present scalar aggregation, and a disaggregated evaluation path is left as a dedicated future-work item (§ 4.6).

## 3.6 Explainability Component

The explainability component implements the interpretability contract of § 13: interpretability in this thesis is operationalised as **aggregated, model-level feature-importance comparison**, not as a claim of full causal understanding, and faithfulness is **tested**, not assumed.

The v6.1 plan separates the *explanation instrument* from the *faithfulness metric*. The primary cross-model instrument is **model-agnostic occlusion importance** (`src/explainability/common_importance.py`, § 2.9.1–2.9.2): for each (model, seed, variable) it establishes a per-model baseline MSE, occludes the variable by setting its column to 0.0 in the scaled input (which equals the training-distribution mean in the original space by `StandardScaler` construction), re-predicts with the unchanged checkpoint, and reports the MSE delta. The same procedure is applied verbatim to all four models, so the resulting rankings are measurement-equivalent across architectures.

The primary faithfulness metric is **AOPC** (Samek et al., 2017; `src/explainability/faithfulness_test.py`, § 2.9.3): for each (model, seed) it compares the MSE increase from occluding the top-k variables against the MSE increase from occluding the bottom-k variables, across k ∈ {1, …, 7}, and averages. Higher AOPC indicates a more faithful ranking — the top-ranked variables damage the model more than the bottom-ranked variables under perturbation.

The v1 SHAP attribution profiles for LR / MLP / LSTM (`results/shap_<model>.csv`) and the TFT Variable Selection Network (VSN) importance (`results/tft_importance.csv`) are retained as **auxiliary architecture-specific explanations** and are not used for the primary cross-model claim. They appear only in the construct-validity discussion of § 4.4.2 as the original motivation for introducing the common instrument (SHAP and VSN are not methodologically equivalent objects — post-hoc vs architecture-native). A dedicated input-perturbation stability layer was prototyped but was dropped on 2026-04-20 (S-11 archive migration; `stability_test.py` retrained models under lr=0.001 + 50-epoch + single-seed, inconsistent with § 10 locked defaults); explanation-stability in the present thesis is therefore restricted to the seed-CV evidence already produced by the training pipeline (§ 3.4).

<!-- @begin-include _generated/status_summary.md -->
Tables and figures in this section are auto-generated from v2 fair-core rerun artefacts (`results/bachelor_safe_v2/`). The training and post-training analysis layers (`multi_seed.py`, `tft_fair_3seed.py`, `export_predictions.py`, `post_training_analysis.py`) are **VALIDATED** in `docs/VALIDATION_LOG.md`. The explainability layer (SHAP for LR/MLP/LSTM + VSN for TFT + faithfulness test) is **VALIDATED**.
<!-- @end-include -->

## 3.7 Reproducibility Verification Component

Reproducibility operates on two tiers.

**Tier 1 (byte-level).** Deterministic code paths — closed-form OLS for LR and the prediction-export layer that consumes already-trained checkpoints — are required to produce byte-identical outputs across reruns. An automation script (`src/utils/verify_reproducibility.py`) was prototyped for this purpose but was dropped on 2026-04-20 (S-11 archive migration) because it reconstructed TFT under the OT-only + future-covariates regime (inconsistent with § 10 fair-core) and re-trained the model inside the check rather than loading the authoritative checkpoints. The Tier 1 checks listed in Appendix E are therefore verified *ad hoc* from the current `results/` + `checkpoints/` artefacts until a fair-core replacement is written.

**Tier 2 (metric-tolerance).** Iteratively-trained GPU code paths — the TFT training path under non-deterministic CuDNN kernels — are not required to be byte-identical, but their summary metrics must reproduce within seed-level tolerance. The present TFT thesis-valid baseline (MSE 14.19 ± 0.68) was established by a single audited run; rerun reproducibility is therefore expressed as a tolerance band around this baseline rather than as bit-exact identity.

Per-seed checkpoints are stored under `checkpoints/`; the test-time prediction tensors of all four models are exported to `results/preds_*.npy` (shape `(3 365, 24, 7)`, original scale) so that the downstream uncertainty and faithfulness layers operate from a single frozen prediction artefact rather than from re-loaded model weights. This frozen-prediction-artefact discipline is the primary reproducibility lever: whatever the reproducibility properties of the upstream training paths, the downstream comparison numbers are computed from a fixed input.

The component also encodes a checkpoint-traceability rule: every reported metric and every bootstrap interval is traceable, by file path, to the checkpoint that produced the underlying predictions. Restoration of any reported number from cold storage requires only the corresponding checkpoint file and the deterministic export step.

## 3.8 Experimental Results

This section reports the **observed overall performance** of the four core models under the locked configuration described above. All claims in this section are bounded by the Ranking-Interpretation Gate of § 11C: the Comparability Gate has cleared 10 / 10 (`docs/VALIDATION_LOG.md` row `COMPAR-GATE`), so cross-model interpretation is permitted, but the disaggregation doctrine restricts claims to the **overall** aggregation level only. Per-variable and per-horizon claims are not supported by the present aggregation and are flagged as future work in § 4.6. Interpretation of these numbers is deferred to the Discussion (§ 4.1, § 4.3).

### 3.8.1 Overall point metrics (original scale)

The per-seed mean and standard deviation of each metric over seeds {42, 123, 456, 789, 1024} are reported below; LR is deterministic.

<!-- @begin-include _generated/overall_metrics_table.md -->
| Model | Parameter count | MSE (mean ± std) | MAE (mean ± std) | RMSE (mean ± std) | Best epochs (per seed) |
|-------|-----------------|-------------------|-------------------|---------------------|--------------------------|
| LR    | 113 064      | 7.66 | 1.47 | 2.77 | — (closed-form) |
| MLP   | 124 328      | 9.47 ± 0.12 | 1.81 ± 0.01 | 3.08 ± 0.02 | 13 / 24 / 11 / 23 / 20 |
| LSTM  |  29 608      | 12.92 ± 0.43 | 2.14 ± 0.06 | 3.59 ± 0.06 | 25 / 28 / 36 / 24 / 20 |
| TFT   |  18 261      | 16.34 ± 5.74 | 2.42 ± 0.29 | 4.00 ± 0.65 | 5 / 14 / 9 / 34 / 1 |
<!-- @end-include -->

### 3.8.2 Pairwise uncertainty intervals

Paired **moving-block** bootstrap 95 % confidence intervals on the mean per-window error difference for each of the six model pairs were computed with `BOOTSTRAP_SEED = 2026`, `N = 10 000` resamples, and block length **L = 96** (`results/bachelor_safe_v2/bootstrap_intervals.csv`). Block length L = 96 coincides with `INPUT_LEN` and follows the optimal-block-length heuristic of Politis & White (2004); Appendix § F.2 reports CI widths at L ∈ {24, 48, 96, 192}. The moving-block variant preserves within-block serial dependence, method-matching the uncertainty layer to the overlapping-sliding-window structure of ETTh1 rather than silently assuming away the independence violation. The safety sentence of § 16 applies — *Interval estimates are used as comparative uncertainty evidence, not as a claim of strict independent-sample inference.*

For the MSE metric, the six pairwise intervals are:

<!-- @begin-include _generated/bootstrap_headline.md -->
| Pair (A vs B) | Mean diff (A − B) | 95 % CI         |
|---------------|--------------------|-----------------|
| LR vs MLP     |   -1.27            | [ -1.58,  -0.88] |
| LR vs LSTM    |   -4.41            | [ -5.32,  -3.41] |
| LR vs TFT     |   -5.79            | [ -6.74,  -4.61] |
| MLP vs LSTM   |   -3.14            | [ -4.20,  -2.09] |
| MLP vs TFT    |   -4.52            | [ -5.59,  -3.32] |
| LSTM vs TFT   |   -1.38            | [ -1.73,  -0.89] |
<!-- @end-include -->

### 3.8.3 Complexity context

Three operationalisations of "complexity" are reported (§ 14, `results/bachelor_safe_v2/complexity_metrics.csv`): parameter count (primary), wall-clock training time (secondary, with hardware note), and architectural category. Parameter counts appear in the table of § 3.8.1; the architectural-category ordering is **linear (LR) < shallow-MLP (MLP) < recurrent (LSTM) < transformer-family (TFT)**. Wall-clock training time (fair-core v2 rerun, 2026-04-22) totals across the five seeds on a single NVIDIA RTX 5080 Laptop GPU with batch size 64: LR **0.7 s** (closed-form), MLP **28 s**, LSTM **45 s**, TFT **2 373 s (≈ 39.5 min)** — TFT dominates by two orders of magnitude.

The three orderings are mutually orthogonal: parameter-count ordering is **MLP > LR > LSTM > TFT**, architectural-category ordering is **LR < MLP < LSTM < TFT**, and accuracy ordering (lower error first) is **LR > MLP > LSTM > TFT**. None of the three coincide. This three-way orthogonality is taken up in § 4.1 and § 4.3 as a material framing point for RQ1 and RQ3.

### 3.8.4 Aggregated explainability outputs

The RQ2 and RQ3 headline summaries, under the occlusion + AOPC instrumentation of § 3.6, are reported below; full interpretation is deferred to § 4.2 (RQ2) and § 4.3 (RQ3).

**Per-model AOPC summary (seed-averaged, higher = more faithful).** From `results/bachelor_safe_v2/faithfulness_aopc.csv`:

| Model | MSE (orig, seed-mean) | AOPC (seed-mean) | AOPC std across seeds |
|-------|------------------------|-------------------|------------------------|
| LR    | 7.66                   | 0.202             | — (deterministic)      |
| MLP   | 9.47                   | 0.126             | 0.049                  |
| LSTM  | 12.92                  | 0.231             | 0.067                  |
| TFT   | 16.34                  | 0.180             | 0.147                  |

The two orderings are non-monotonic under the present configuration: the accuracy ordering (lower-first) is LR < MLP < LSTM < TFT, while the AOPC ordering (higher-first) is LSTM > LR > TFT > MLP. No single model dominates on both axes; § 4.3 treats this as the RQ3 trade-off signal. TFT's across-seed AOPC standard deviation (0.147) is roughly twice LSTM's and three times MLP's, mirroring the five-seed MSE variance spike documented in § 4.1 (Internal validity).

**Cross-model occlusion rank agreement (primary RQ2 anchor).** From `results/bachelor_safe_v2/xai_agreement_occlusion.csv`, the six pairwise Spearman ρ on the seed-averaged per-variable occlusion importance are: LR↔MLP 0.000, LR↔LSTM −0.179, LR↔TFT 0.500, MLP↔LSTM 0.750, MLP↔TFT 0.357, LSTM↔TFT 0.500. The highest agreement is MLP↔LSTM (ρ = 0.75); LR disagrees with both other post-hoc models on the rank-trend, while its agreement with TFT is moderate. These rank correlations are computed on the *same* measurement object (occlusion) for all four models and are therefore directly comparable — § 4.4.2 records the construct-validity context in which the auxiliary SHAP↔VSN comparison was superseded.

<!-- @begin-include _generated/status_summary.md -->
Tables and figures in this section are auto-generated from v2 fair-core rerun artefacts (`results/bachelor_safe_v2/`). The training and post-training analysis layers (`multi_seed.py`, `tft_fair_3seed.py`, `export_predictions.py`, `post_training_analysis.py`) are **VALIDATED** in `docs/VALIDATION_LOG.md`. The explainability layer (SHAP for LR/MLP/LSTM + VSN for TFT + faithfulness test) is **VALIDATED**.
<!-- @end-include -->
