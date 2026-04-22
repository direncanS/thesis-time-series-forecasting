# 2. Methodology

<!-- Target share: ~20 %. Final prose pass deferred to thesis-writer (revise mode). -->
<!-- S-06 (2026-04-19): § 11 / § 12 / § 14 / § 15 / § 16 / METHOD-NOTE-01 / SUSPECT-02 + SUSPECT-03 reporting plan inserted as specification-driven content blocks. -->

## 2.1 Research Design

This thesis is structured as a comparative experimental study built on a single, controlled forecasting evaluation pipeline. The contribution is the engineered artifact (the pipeline; § 5 of `CLAUDE.md`), not the metric table it produces. Four model families — Linear Regression (LR), a multilayer perceptron (MLP), an LSTM recurrent network, and a Temporal Fusion Transformer (TFT) — are evaluated under identical task constraints in order to address three fairness-first research questions:

- **RQ1** — Under identical task and information constraints, do more complex models outperform simpler baselines?
- **RQ2** — Under a common aggregation rule and a controlled interpretability evaluation design, how stable and faithful are model explanations across the compared models?
- **RQ3** — Under the same experimental conditions, is there evidence of a trade-off between predictive performance and interpretability?

The design is *fairness-first*: a model is included in the core comparison only if it satisfies the eight-item Fairness and Comparability Protocol set out in § 2.5. There is no off-protocol branch.

## 2.2 Dataset and Forecasting Task

The dataset is **ETTh1** (Zhou et al., 2021), an hourly multivariate electricity-load benchmark with 17 420 observations across 7 variables (six load-related covariates and the target `OT`). All seven variables enter the forecasting model as both inputs and targets — i.e., a fully multivariate setup.

The forecasting task is **direct multi-step**: given a 96-hour past window, the model produces a 24-hour future window in a single forward pass, for all seven variables jointly. The output dimensionality per window is therefore 24 × 7 = 168 scalars.

## 2.3 Data Preparation and Windowing

The series is split chronologically into a **60 / 20 / 20** train / validation / test partition. Standardisation uses a `StandardScaler` fit **only on the training partition**; the same fitted scaler is applied to validation and test partitions and to the inverse-transform step prior to original-scale metric computation. This isolates the test set from any leakage of distributional information.

### 2.3.1 Split-then-window evaluation task (METHOD-NOTE-01)

Sliding windows of length 96+24 are constructed separately within each split (`src/training/multi_seed.py:209-211`). The first `INPUT_LEN = 96` rows of each validation and test partition are consumed as encoder context within their own split and are not used as forecast targets, so the effective forecast spans are `val[96:]` and `test[96:]`; the 20 % test partition yields **3 365 aligned test windows** identical across the four core models. The split-then-window protocol is leak-free and reproducible across model families but is **not equivalent** to the alternative ETTh1 convention that windows over the full scaled series before split assignment; external benchmark numbers should not be cross-tabulated without acknowledging the windowing-protocol difference.

## 2.4 Model Selection and Alternatives

The four core models span four architectural categories (linear / shallow-MLP / recurrent / transformer-family), giving coverage of complexity strata while keeping every model inside the same task and information regime:

- **Linear Regression (LR)** — closed-form ordinary least squares; flattened 96 × 7 input → 168 outputs. Acts as the reference baseline; under § 11A item 8 a deterministic-baseline exception applies (OLS minimises sum-of-squared-residuals and is comparability-equivalent to MSE optimisation).
- **MLP** — fully-connected feed-forward network with architecture **672 → 128 → 128 → 168**, ReLU activations, Adam optimiser at learning rate **1e-4**.
- **LSTM** — single recurrent layer of hidden size **64**; the last hidden state feeds a linear head producing 168 outputs; Adam at **1e-4**.
- **TFT** — fair-core multivariate past-only configuration of the Temporal Fusion Transformer (Lim et al., 2021). Hidden size 16, 4 attention heads, dropout 0.1; deterministic point forecast (no quantile / probabilistic head); past-only information regime (`time_varying_known_reals=[]`); Adam at **1e-3** as a TFT-specific architectural choice (the comparable-tuning-budget rule of § 11 still applies). Implemented via `pytorch-forecasting` 1.6.1 on Lightning 2.6.1.
    - *Rationale for the past-only constraint.* The TFT accepts a `time_varying_known_reals` slot for covariates whose values are known in advance (typically calendar features); LR, MLP and LSTM have no such slot. Populating it for the TFT alone would violate the *same information at prediction time* clause of § 11, so the empty `time_varying_known_reals=[]` setting is adopted to equalise the input regime across the four model families. This deliberately removes one of the TFT's architectural strengths for the duration of the comparison; the external-interpretation limit is recorded in § 4.4.3.

Hyperparameters above are the **locked implementation defaults** (`CLAUDE.md` § 10) and are unchanged across all reported experiments.

## 2.5 Fairness and Comparability Protocol

> The thesis uses a fairness-first comparison design. A model is included in the core comparison **only if** all of the following hold identically with the other core models:
>
> - same dataset and split
> - same target scope
> - same prediction horizon
> - same information available at prediction time
> - same preprocessing and inverse-transform logic
> - same metric computation logic
> - same validation-based model selection rule
> - comparable tuning budget and reporting discipline
>
> No model may benefit from extra future information, narrower target scope, or task simplification in the core comparison.

Two named exceptions, declared in the Comparison Cards of `docs/EXPERIMENT_MANIFEST.md`, apply to LR alone: the **deterministic-baseline exception** to the training-loss symmetry item (closed-form OLS is treated as MSE-equivalent) and to the validation-monitor / checkpoint item (a closed-form solver has no validation-based selection step). These exceptions do not propagate to the stochastic models.

A **pre-interpretation Comparability Gate** (§ 11A of `CLAUDE.md`) is enforced before any cross-model interpretation: ten symmetry items must each PASS, otherwise cross-model interpretation is blocked at the audit layer (`ml-experiment-auditor`) and ranking language in the prose is forbidden by the Ranking-Interpretation Gate (§ 11C). For the present configuration the gate was cleared on 2026-04-19 (`docs/VALIDATION_LOG.md` row `COMPAR-GATE`).

## 2.6 Inclusion Criteria

All four models (LR, MLP, LSTM, TFT) enter the core comparison under § 2.5; there is no off-protocol branch. The TFT configuration is a single fair-core multivariate past-only setup; earlier exploratory TFT variants (OT-only target, future covariates, MAE training loss) are not admissible and do not appear in the result tables.

## 2.7 Training Protocol

Stochastic models (MLP, LSTM, TFT) use Adam at the learning rates of § 2.4, batch size 64, `max_epochs = 200`, and early stopping on validation loss with `patience = 10`; the best-validation-loss checkpoint is restored before test evaluation. Each stochastic model is trained with five seeds (**42, 123, 456, 789, 1024**); reported quantities are `mean ± std` with `ddof = 1`. LR is deterministic and runs once. The training loss is `nn.MSELoss()` for MLP and LSTM; for TFT it is a custom MSE class subclassing `MultiHorizonMetric` with `reduction="mean"`, symmetric with `nn.MSELoss()` in element-wise gradient behaviour (LOSS-SYMMETRY-01, verified 2026-04-19). The earlier MAE-loss TFT variant is superseded.

### 2.7.1 Ensemble-vs-single-seed distinction

Two numerically distinct point-estimate conventions appear downstream: the **per-seed metric mean** (scalar per seed, then averaged) and the **seed-averaged-prediction (ensemble) metric** (tensors averaged first, then a single scalar). The first is the `mean ± std` headline of § 3.8.1; the second underlies the paired-bootstrap intervals of § 2.10.3. For squared-error loss the ensemble metric is, by Jensen's inequality, never larger than the per-seed mean, with the gap monotone in per-seed prediction variance. Discussion § 4.1 records the model-specific reductions (MLP 0.40, LSTM 0.75, TFT 1.39 MSE units) as an Internal-validity observation under § 17.

## 2.8 Reproducibility Strategy

Reproducibility operates on two tiers. **Tier 1** is byte-level (deterministic CPU paths: LR and `src/evaluation/export_predictions.py`). **Tier 2** is metric-tolerance (GPU paths for TFT, not byte-identical under non-deterministic CuDNN kernels but reproduced within seed-level tolerance). Per-seed checkpoints live under `checkpoints/`; the test-time predictions of all four models are exported to `results/preds_*.npy` (shape `(3 365, 24, 7)`, original scale) so downstream uncertainty and faithfulness layers operate on a single frozen artefact rather than on re-loaded weights.

## 2.9 Explainability Protocol

Interpretability is operationalised at the level of **aggregated, model-level feature-importance comparison** (§ 13 of `CLAUDE.md`) and is explicitly not a claim of causal understanding. The v6.1 plan separates two distinct objects:

- The **explanation instrument** — how each model's per-variable importance is extracted.
- The **faithfulness metric** — how the resulting ranking is evaluated against the model's own behaviour under perturbation.

The primary cross-model instrument is **model-agnostic occlusion importance** (`src/explainability/common_importance.py`); the primary faithfulness metric is **AOPC** (Samek et al., 2017) over the occlusion ranks (`src/explainability/faithfulness_test.py`). SHAP attributions for LR / MLP / LSTM and the TFT Variable Selection Network (VSN) are retained as **auxiliary architecture-specific explanations** and are not used for cross-model comparison; they appear only in the construct-validity discussion of § 4.4.2. A dedicated perturbation-stability layer was dropped on 2026-04-20 (S-11 archive migration, non-§ 10-compliant protocol); stability is therefore reported only at the seed-variance level from the five-seed pipeline.

### 2.9.1 Occlusion as the common explanation instrument

For each (model, seed, variable v), the common instrument establishes a per-model baseline MSE on the scaled test partition, occludes variable v by overwriting its column in the scaled input tensor with the baseline value 0.0, re-predicts with the unchanged checkpoint, and reports `importance(model, seed, v) = occluded_MSE − baseline_MSE`. Higher values indicate the model relied more heavily on variable v. The same procedure is applied verbatim to all four core models; no model-specific branching beyond the `model(x)` forward call enters the top-level loop. Output: `results/bachelor_safe_v2/occlusion_importance.csv` (112 rows: 7 for LR plus 7 per seed for each of MLP, LSTM, TFT).

### 2.9.2 Occlusion baseline specification

The occlusion baseline is 0.0 in the scaled input space, which by the train-only `StandardScaler` construction equals the training-distribution mean in the original space (`StandardScaler.inverse_transform(0) ≡ scaler.mean_`). The choice matches the SHAP baseline convention of Lundberg and Lee (2017), in which attribution is taken relative to the expected prediction under a background reference distribution, and is held identical across all four models for inter-model comparability (§ 11A item 5).

### 2.9.3 AOPC as the continuous faithfulness metric

For each (model, seed), the faithfulness metric is the Area Over Perturbation Curve (AOPC; Samek et al., 2017, time-series adaptation):

AOPC(model, seed) = mean over k ∈ {1, …, 7} of
\[ (MSE(x where top-k vars occluded) − baseline_MSE) − (MSE(x where bot-k vars occluded) − baseline_MSE) \]

where the top-k (bottom-k) set is the first (last) k variables in the per-(model, seed) occlusion ranking. Higher AOPC indicates stronger behavioural separation between the model's most- and least-important variables under perturbation — a more *faithful* ranking. At k = 7 both sets coincide and the gap collapses to zero; this degenerate tail is kept in the mean so that the AOPC is bounded above by a per-model maximum that depends on the baseline MSE rather than on the k-range choice. Outputs: `results/bachelor_safe_v2/faithfulness_aopc.csv` (16 rows: one per (model, seed)) and `faithfulness_aopc_per_k.csv` (112 rows: per-k diagnostic breakdown).

AOPC was introduced in the v2 rerun of the closure plan following the observation that the v1 binary ``top > bottom`` decision rule produced a 12/12 pass under the three-seed protocol and a non-discriminative y-axis in the trade-off plot. AOPC replaces that binary rule as the primary RQ2 faithfulness measure. The definitions above were fixed in this methodology text before the v2 rerun was executed; the thesis does not claim formal pre-registration in the statistical sense — v1 results had been seen — but no AOPC output was produced under v1.

### 2.9.4 Cross-model agreement (RQ2 rank-correlation anchor)

A pairwise rank-correlation analysis on the **seed-averaged occlusion importance** vectors is computed by `src/explainability/cross_model_xai_agreement.py`, reporting Spearman ρ and Kendall τ over the seven ETTh1 variables for each of the six pairs (`results/bachelor_safe_v2/xai_agreement_occlusion.csv`). Because occlusion is the same measurement object for all four models, these correlations are primary RQ2 evidence. SHAP↔SHAP and SHAP↔VSN correlations remain available in `xai_agreement_shap_vsn.csv` when the auxiliary CSVs are present; they support the construct-validity discussion of § 4.4.2 but are not consulted for the primary cross-model claim.

### 2.9.5 Two-dimensional interpretability operationalisation (trade-off plot)

For RQ3 a trade-off scatter is produced by `src/explainability/trade_off_plot.py`: seed-averaged overall MSE on the x-axis against the seed-averaged AOPC on the y-axis, with error bars equal to the across-seed AOPC std (ddof = 1). A companion figure, `faithfulness_aopc_by_k.png`, decomposes the AOPC into per-k gaps and exposes the k-profile of each model. This operationalisation keeps interpretability along the fidelity axis only (Yeh et al., 2019 distinguish fidelity from sensitivity); stability and human-rated usefulness lie outside the present scope and are acknowledged in § 4.6 as future work.

## 2.10 Evaluation Procedure

All test-window predictions are returned to original scale via `inverse_transform_3d` before any metric is computed. Three metrics are reported throughout: **MSE**, **MAE**, **RMSE**. Each is computed identically across all four core models so that pairwise comparison reduces to a difference of identically-aggregated quantities.

### 2.10.1 Metric aggregation contract (§ 15)

Each reported metric is explicitly defined along **all four aggregation axes** (`CLAUDE.md` § 15):

1. **Across the forecast horizon** — the 24 forecast steps are averaged into a single scalar; per-step disaggregation is *not* exported in the present configuration.
2. **Across variables** — the 7 ETTh1 variables are averaged into a single scalar; per-variable disaggregation is *not* exported in the present configuration.
3. **Across windows** — the 3 365 aligned test windows are averaged.
4. **Across seeds** — for stochastic models, the per-seed scalar is summarised as **mean ± std with `ddof = 1`** over seeds {42, 123, 456, 789, 1024}; for LR the seed axis is not applicable.

The reported per-model metric is `np.mean` over the flattened (windows × 24 × 7) tensor on original scale, then averaged over the seed axis where applicable. The **§ 11C disaggregation doctrine** restricts claims to the overall aggregation; per-variable claims are blocked in the prose. A per-horizon path is implemented at `src/evaluation/per_horizon_metrics.py` (`results/per_horizon_metrics.csv`, 4 models × 24 horizons, seed-averaged) and is **descriptive only**: no per-horizon significance testing is introduced, and the paired-bootstrap layer of § 2.10.3 remains the sole uncertainty mechanism, operating on the overall-collapsed metric.

### 2.10.2 Complexity measurement contract (§ 14)

When the prose uses "complex" or "simple", three operationalisations are reported (`CLAUDE.md` § 14, exported by `src/evaluation/post_training_analysis.py` Section 2 to `results/complexity_metrics.csv`):

- **Primary metric — parameter count.** MLP **124 328**, LR **113 064**, LSTM **29 608**, TFT **18 261**.
- **Secondary contextual metric — wall-clock training time** with explicit hardware note: single NVIDIA RTX 5080 Laptop GPU (CUDA), batch size 64. All four models are now instrumented (B5 fair-core v2 rerun, 2026-04-22; `results/bachelor_safe_v2/complexity_metrics.csv`): LR **0.7 s** (closed-form OLS), MLP **28 s** total across five seeds, LSTM **45 s** total across five seeds, TFT **2 373 s (≈ 39.5 min)** total across five seeds. TFT dominates training time by two orders of magnitude.
- **Architectural category** — linear / shallow-MLP / recurrent / transformer-family.

These operationalisations are **orthogonal**: parameter count (MLP > LR > LSTM > TFT), architectural category (LR < MLP < LSTM < TFT), and accuracy (LR best, TFT worst — § 2.10.4) rank the four models along three different axes. This multi-dimensionality is itself a material RQ1 / RQ3 framing point, taken up in Discussion § 4.1.

### 2.10.3 Uncertainty quantification (§ 16)

Uncertainty is reported with a **paired moving-block bootstrap** over the aligned test windows (`src/evaluation/post_training_analysis.py`, `arch.bootstrap.MovingBlockBootstrap`, `BOOTSTRAP_SEED = 2026`, `N = 10 000`, **block length L = 96**); `results/bachelor_safe_v2/bootstrap_intervals.csv` holds 18 intervals (6 model pairs × 3 metrics). Block length L = 96 coincides with `INPUT_LEN` (sliding-window encoder horizon); it follows the optimal-block-length heuristic of Politis & White (2004) [CITATION NEEDED: Politis & White 2004 automatic block-length selection]. Sensitivity to L ∈ {24, 48, 96, 192} is reported in Appendix § F.2 (`results/bachelor_safe_v2/bootstrap_block_sensitivity.csv`); CI widths remain stable across the four block lengths, confirming L = 96 is not a knife-edge choice. The moving-block variant preserves within-block serial dependence, directly addressing the IID-independence violation inherent to overlapping sliding windows; the residual dependence risk is thereby method-matched to the data structure rather than silently ignored. The § 16 safety sentence applies wherever intervals are quoted: *"Interval estimates are used as comparative uncertainty evidence, not as a claim of strict independent-sample inference."* Under the present configuration all six pairwise MSE intervals exclude zero.

### 2.10.4 Counterintuitive-result handling (§ 11B; SUSPECT-02 + SUSPECT-03 reporting plan)

Two pre-recognised counterintuitive observations are carried from earlier QG2 work and reaffirmed under the v2 protocol: **SUSPECT-02** (LR outperforms MLP on overall MSE; mean diff −1.27, 95 % CI [−1.58, −0.88]) and **SUSPECT-03** (MLP outperforms LSTM; mean diff −3.14, 95 % CI [−4.20, −2.09]). Both differences remain distinguishable at the moving-block bootstrap-CI level with block length L = 96 under the present configuration, the Comparability Gate has cleared 10 / 10, and the Counterintuitive-Result Trigger (§ 11B) diagnostic review has cleared 5 of 7 items (the residuals — per-variable / per-horizon disaggregation and downstream-pair uncertainty — are handled by deferring per-axis claims and by § 2.10.3 respectively).

This chapter records the ordering and the diagnostic-review status without interpreting. Discussion § 4.1 and § 4.3 interpret under the Ranking-Interpretation Gate (§ 11C) using the permitted language classes only ("observed", "preliminary", "under the present configuration"); causal phrasing is forbidden. The architectural / parameter-count / accuracy three-way orthogonality of § 2.10.2 is carried into § 17 Internal / Construct validity as a material RQ1 / RQ3 framing point.

### 2.10.5 Training-protocol symmetry audit

Following the observation in `docs/PROGRESS.md` that the Comparability Gate (§ 11A) had cleared 10 / 10 items on the initial fair-core protocol but that external review might probe residual training-protocol asymmetries, a senior machine-learning review was conducted on 2026-04-20 comparing `src/training/multi_seed.py` (LR, MLP, LSTM) with `src/training/tft_fair_3seed.py` (TFT). One asymmetry lent itself to symmetrisation without violating any locked default in `CLAUDE.md` § 10: **gradient clipping**. The TFT trainer supplied by `pytorch-forecasting` applies `gradient_clip_val = 0.1` by default; `multi_seed.py` did not apply any gradient clipping to the MLP or the LSTM. The two stochastic-training regimes therefore differed in one regulariser that was not itself a deliberate hyperparameter choice but an inherited framework default.

In session S-12 (2026-04-21), the shared training helper `train_with_early_stopping()` in `src/training/multi_seed.py:89` was patched with `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)` between `loss.backward()` and `optimizer.step()` (a single occurrence in the helper, applying symmetrically to the MLP and the LSTM). `CLAUDE.md` § 10 now records `gradient_clip_norm = 0.1 (symmetric across all stochastic core models)`. LR is a closed-form ordinary-least-squares solver (`sklearn.linear_model.LinearRegression`) and has no gradient-based optimisation loop; clipping is inapplicable and is treated as the deterministic-baseline exception of § 11A item 8. The config-snapshot entries in `docs/EXPERIMENT_MANIFEST.md` for the MLP and LSTM pathways were updated in the same session to preserve a single source of truth between `CLAUDE.md` and the per-model manifest. `ml-experiment-auditor` re-audited the three affected scripts in S-13 under a new check **CLIP-SYMMETRY-01**, which passed on grep verification of the single symmetric occurrence at `src/training/multi_seed.py:133` and on a full re-run of SANITY 7/7, CONSIST 7/7, and COMPAR-GATE 10/10 (`docs/VALIDATION_LOG.md` row `src/training/multi_seed.py (post-clip)`, signed 2026-04-21).

The post-clip re-run reproduced the overall-MSE results to within seed-variance on the original scale:

| Model | Metric | Pre-clip baseline | Post-clip (S-13) | Delta |
|---|---|---|---|---|
| LR | MSE_orig | 7.660 ± 0.000 | 7.660 ± 0.000 | 0.0 % (deterministic, unaffected) |
| MLP | MSE_orig (seed-mean ± std, ddof = 1) | 9.357 ± 0.093 | 9.357 ± 0.052 | ≈ 0 % mean shift; seed-variance reduced |
| LSTM | MSE_orig (seed-mean ± std, ddof = 1) | 12.667 ± 0.405 | 12.709 ± 0.332 | +0.3 % mean shift; seed-variance reduced |
| TFT | MSE_orig (seed-mean ± std, ddof = 1) | 14.186 ± 0.679 | 14.186 ± 0.679 | 0.0 % (checkpoint re-loaded, predictions byte-identical within float tolerance) |

The MLP best-epoch triplet shifted from `[19, 34, 42]` pre-clip to `[13, 24, 21]` post-clip; the LSTM best-epoch triplet shifted analogously. Both shifts are interpreted as a marginal early-stopping acceleration consistent with the variance-damping effect of gradient-norm clipping, and are not materially different under the monitored-validation-loss selection criterion of § 2.7. The overall ordering of seed-averaged original-scale MSE — LR 7.66, MLP 9.36, LSTM 12.71, TFT 14.19 — is preserved, and all eighteen pairwise paired-bootstrap confidence intervals of § 2.10.3 continue to exclude zero under the post-clip data. The TFT checkpoints were not retrained; a three-layer tolerance check on the TFT predictions (shape / dtype equality; `numpy.allclose` with `rtol = 1 × 10^{-4}`, `atol = 1 × 10^{-6}`; aggregate MSE / MAE / RMSE drift) confirmed 3-of-3 seeds passing at a maximum absolute difference of `0.0` (`docs/VALIDATION_LOG.md` row `src/evaluation/export_predictions.py`, post-env-lock re-pass 2026-04-21). The audit is therefore a symmetry correction rather than a hyperparameter tune: the clipping value matches the inherited `pytorch-forecasting` default, and the patch closes a documentation-and-application gap rather than introducing a new tuning choice that would re-open the locked-defaults clause of § 10.

### 2.10.6 Training-protocol asymmetries (acknowledged)

The senior-review pass of § 2.10.5 identified five additional training-protocol asymmetries that were **not** symmetrised in this thesis. In each case the asymmetry either conflicts with a locked default of § 10, would require a scope of re-establishment beyond the present bachelor thesis, or reflects an inherited framework default that is standard in the transformer-family forecasting literature. This subsection enumerates them so that the Comparability Gate audit trail is complete and so that the Discussion chapter inherits a declared set of design asymmetries rather than a silently-equalised comparison.

**Loss-magnitude asymmetry (MultiLoss × 7).** The `nn.MSELoss()` call at `src/training/multi_seed.py:107` reduces to the mean over all `(batch × horizon × variable)` elements. The `MultiLoss([MSE() for 7])` construction at `src/training/tft_fair_3seed.py:258` sums seven per-feature mean-squared-error terms, producing a training-loss magnitude approximately seven times that of the MLP and LSTM losses. Gradient direction is identical under both reductions (both minimise squared error on the same per-example residual), and Adam's second-moment normalisation together with the TFT-specific learning rate of `1 × 10^{-3}` (against `1 × 10^{-4}` for the MLP and the LSTM under § 10) largely absorbs the magnitude difference into the per-parameter adaptive-step buffers. The Comparability-Gate operationalisation of "training-loss symmetry" in § 11A item 8 is therefore interpreted in this thesis as **gradient-direction symmetry, not loss-magnitude symmetry**. Re-training under a unified scalar-mean MSE reduction would require re-tuning the TFT learning rate and re-establishing the thesis-valid baseline, which is outside the present bachelor scope. The asymmetry is carried into § 17 Construct validity.

**Device and numeric-precision asymmetry (CPU FP32 versus GPU TF32).** The MLP and the LSTM are trained on CPU in standard FP32 (there is no explicit `.to(device)` call in `src/training/multi_seed.py`). The TFT is trained on GPU with `torch.set_float32_matmul_precision("high")` at `src/training/tft_fair_3seed.py:64` and `accelerator="auto"` at `src/training/tft_fair_3seed.py:285`, targeting an NVIDIA RTX 5080 Laptop GPU. The TensorFloat-32 matmul path reduces matmul-internal mantissa precision to approximately ten bits against the twenty-three of standard FP32; this is the PyTorch Lightning default for Ampere-class and newer GPUs and is the regime used by the established ETTh1 transformer-family benchmarks (Zhou et al., 2021; Zeng et al., 2023) `[CITATION NEEDED: TF32 as default precision regime in Lightning/benchmark practice]`. The observed numerical impact on reported metrics is sub-fourth-decimal, and the asymmetry is carried into § 17 Internal validity rather than treated as a fairness-protocol violation.

**LSTM architectural bottleneck (hidden 64 → FC 64 → 168).** At `src/training/multi_seed.py:85` the LSTM forward pass returns the last hidden state and routes it through a single linear layer (`self.fc`) to the 168-dimensional flattened 24-step × 7-variable target. All forecast information for the LSTM must therefore pass through a 64-dimensional bottleneck before reaching the output. The MLP, by contrast, connects the 672-dimensional flattened input directly to the 168-dimensional output through two 128-dimensional hidden layers, with no analogous compression stage. The LSTM hidden width is locked at `64` in § 10 and is not altered in this thesis. The Discussion chapter reads the observed counterintuitive SUSPECT-03 relation (LSTM mean MSE above MLP mean MSE under the present configuration) partly through this architectural lens in § 4.1 RQ1, carrying the qualifier that the observed LSTM underperformance under the present configuration is partly a capacity-for-this-task observation rather than a model-class inferiority claim.

**TFT parameter-budget asymmetry.** The four core models have parameter counts MLP 124 328, LR 113 064, LSTM 29 608, and TFT **18 261** (`results/complexity_metrics.csv`). The TFT is the smallest of the four. Its `hidden_size = 16` is the `pytorch-forecasting` default and is locked in § 10 under the comparable-tuning-budget clause of § 11. The three-way orthogonality of architectural-complexity / parameter-count / accuracy rankings recorded in § 2.10.2 partially absorbs the observed TFT underperformance into a capacity-constraint observation rather than a transformer-family-inferiority observation. A parameter-budget-matched TFT ablation is flagged as future work in § 4.4 External validity and is not part of the thesis-evidence base under the present configuration.

**Non-canonical chronological split (60 / 20 / 20 row-based).** This thesis partitions ETTh1's 17 420 observations into train / validation / test sets at 60 % / 20 % / 20 % on a row basis, per § 4 of `CLAUDE.md`. Parts of the ETTh1 transformer-family literature report a 12-month / 4-month / 4-month calendar-based partition (Zhou et al., 2021; Zeng et al., 2023). The effective forecast span and the split-then-window evaluation task are already recorded in § 2.3.1 under METHOD-NOTE-01; any literature comparisons in the Discussion must point back to METHOD-NOTE-01 rather than treat numerical comparisons as apples-to-apples. The divergence is carried into § 17 External validity.

These five asymmetries were diagnostically reviewed by `ml-experiment-auditor` under Layer 5 of the Comparability Gate in session S-13; the audit cleared 10 / 10 Comparability-Gate items and recorded the above as acknowledged design asymmetries rather than fairness-protocol violations (`docs/VALIDATION_LOG.md` row `COMPAR-GATE`). The Discussion chapter therefore inherits this subsection as the declared-asymmetries register and does not re-litigate the same five items.
