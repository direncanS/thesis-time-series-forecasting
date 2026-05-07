# Thesis figures — captions, inputs, interpretation, limitations

One entry per must-have figure. Captions are in English; German translation
to be added by the student during final thesis rendering. Interpretation and
limitation notes follow the thesis's ranking-language discipline,
interpretability contract, aggregation-axis visibility, and
uncertainty-layer discipline.

---

## Figure 1 — Dataset overview

- **File:** `reports/figures/main/fig_01_dataset_overview.{pdf,png}`
- **Caption:** ETTh1 hourly multivariate time series across the seven variables
  (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) over the full 17,420-hour record.
  Dashed vertical lines indicate the 60 / 20 / 20 chronological split boundaries
  used throughout the thesis.
- **Input data:** `data/ETTh1.csv`.
- **Interpretation note:** Visualises the scale differences and non-stationary
  dynamics that motivate multivariate forecasting in the fair-core design.
- **Limitation note:** A single dataset cannot support general-domain
  generalisation claims (§ 17 external validity).

## Figure 2 — Train / validation / test split and sliding window

- **File:** `reports/figures/main/fig_02_train_test_split.{pdf,png}`
- **Caption:** Chronological 60 / 20 / 20 split of ETTh1 with a representative
  96-hour input window and 24-hour forecast horizon drawn above the timeline.
  The StandardScaler is fit on the training segment only; validation and test
  segments inherit the fitted transform.
- **Input data:** `data/ETTh1.csv` (row count for boundaries); schematic drawn
  programmatically.
- **Interpretation note:** Documents the experimental regime that the
  Fairness and Comparability Protocol (§ 11) enforces identically across the
  four core models.
- **Limitation note:** Sliding windows overlap, so the independence assumption
  underlying classical inference is violated; this is acknowledged in the
  Methods chapter (§ 16).

## Figure 3 — Pipeline architecture

- **File:** `reports/figures/main/fig_03_pipeline_architecture.{pdf,png}`
- **Caption:** End-to-end data flow of the fair-core evaluation pipeline.
  Solid arrows denote primary data flow; dashed arrows denote artifact reuse
  into reporting. The pipeline applies a shared task, preprocessing,
  evaluation, and artifact-recording protocol across the selected models.
- **Input data:** None (structural diagram derived from the repository layout:
  `src/training/`, `src/evaluation/`, `src/explainability/`, `results/`).
- **Interpretation note:** Supports the Solution chapter's standard-software
  defence — an off-the-shelf library cannot enforce these symmetries across
  structurally different model families without this integration layer.
- **Limitation note:** A diagram is an abstraction of the executable pipeline;
  the authoritative specification remains the locked configuration in
  `configs/` and the scripts in `src/`.

## Figure 4 — Cross-model predictive performance

- **File:** `reports/figures/main/fig_04_model_performance_comparison.{pdf,png}`
- **Caption:** Mean test-set RMSE (primary, left) and MAE (secondary, right) per
  core model. Error bars show one standard deviation across the five seeds
  and are **not** confidence intervals. LR is deterministic and therefore has
  no seed dispersion. The same values computed for MSE are reported in the
  per-seed CSV.
- **Input data:** `results/bachelor_safe_v2/per_seed_metrics.csv`.
- **Interpretation note:** Values are observed under the present configuration.
  Pairwise statistical differences with 95 % moving-block bootstrap confidence
  intervals are reported separately in `bootstrap_intervals.csv` and are the
  evidentiary basis required by § 11C for any ranking-language claim.
- **Limitation note:** Error bars capture seed-to-seed variability only; they
  are not paired-difference confidence intervals. The comparison is made on
  the ETTh1 test set alone.

## Figure 5 — Per-horizon test error

- **File:** `reports/figures/main/fig_05_per_horizon_error.{pdf,png}`
- **Caption:** Test-set MSE as a function of the forecast horizon step
  h ∈ {1, …, 24} for each core model, averaged across seeds (five seeds for
  stochastic models, one deterministic run for LR). Shaded bands denote
  ±1 standard deviation across seeds on MSE.
- **Input data:** `results/bachelor_safe_v2/per_horizon_metrics.csv`.
- **Interpretation note:** The horizon axis is entirely within the configured
  96 h → 24 h forecasting task. The figure makes the aggregation along the
  horizon axis visible as required by § 15.
- **Limitation note:** Findings apply strictly to this forecast-horizon
  configuration; they do not imply model superiority at other horizons or on
  other forecasting tasks.

## Figure 6 — AOPC faithfulness by k

- **File:** `reports/figures/main/fig_06_faithfulness_aopc_by_k.{pdf,png}`
- **Caption:** Decomposition of the Area Over Perturbation Curve (AOPC) by the
  number of masked variables k ∈ {1, …, 7}. For each model and seed, the
  plotted "gap" is the difference between the MSE increase when top-ranked
  variables are occluded and the MSE increase when bottom-ranked variables are
  occluded. Lines are means across seeds; bands are ±1 standard deviation.
- **Input data:** `results/bachelor_safe_v2/faithfulness_aopc_per_k.csv`.
- **Interpretation note:** Under this project's AOPC definition, a larger gap
  indicates that the model's own variable ranking is more faithful, because
  removing its top-ranked variables hurts more than removing its bottom-ranked
  ones. The sign and magnitude must be interpreted strictly against this
  definition; no claim of general "explanation quality" is made.
- **Limitation note:** Faithfulness here is quantified by a single occlusion
  instrument on scaled inputs with the training-mean baseline; other
  faithfulness definitions may yield different orderings (§ 13, § 17
  construct validity).

## Figure 7 — Cross-model XAI rank agreement

- **File:** `reports/figures/main/fig_07_xai_agreement_heatmap.{pdf,png}`
- **Caption:** Pairwise Spearman rank correlation between seed-averaged
  variable importances produced by the shared common-XAI occlusion
  instrument. The diagonal represents self-agreement and is rendered in a
  neutral tone; off-diagonal cells quantify how similarly a pair of models
  orders the seven ETTh1 variables.
- **Input data:** `results/bachelor_safe_v2/xai_agreement_occlusion.csv`
  (Spearman ρ column; Kendall τ is available in the CSV but not plotted here).
- **Interpretation note:** Figure 7 reports pairwise Spearman rank agreement
  on seed-averaged occlusion importance. The heatmap indicates whether models
  assign higher occlusion importance to similar variables; it does not provide
  causal evidence.
- **Limitation note:** The XAI instrument is fixed; agreement reported here
  does not generalise to other XAI families such as SHAP or integrated
  gradients (§ 13).

## Figure 8 — Performance / interpretability positioning

- **File:** `reports/figures/main/fig_08_performance_interpretability_tradeoff.{pdf,png}`
- **Caption:** Position of each core model in the mean-test-MSE (x) versus
  mean-AOPC (y) plane. Error bars are ±1 seed standard deviation on each axis
  and are not confidence intervals. Models with a black marker edge are on
  the Pareto-efficient frontier under these two axes; the frontier is
  recomputed from `trade_off_data.csv` at render time and printed to stdout
  during generation.
- **Input data:** `results/bachelor_safe_v2/trade_off_data.csv`,
  `results/bachelor_safe_v2/per_seed_metrics.csv`.
- **Interpretation note:** The trade-off shown applies strictly to the
  present experimental configuration and to the selected interpretability
  metric (occlusion-based AOPC). It is not evidence for or against a
  universal accuracy / interpretability trade-off.
- **Limitation note:** Pareto-dominance is defined here along only two
  aggregated axes. Changing the faithfulness metric (for example to a
  continuous infidelity or ordered-masking curve) may produce a different
  positioning.

## Figure 9 — Model complexity overview

- **File:** `reports/figures/main/fig_09_model_complexity.{pdf,png}`
- **Caption:** Parameter count on a logarithmic x-axis (left panel) and
  training wall-clock in seconds (right panel) per core model. Architectural
  categories — linear, shallow MLP, recurrent, transformer-family — are
  annotated on the left panel.
- **Input data:** `results/bachelor_safe_v2/complexity_metrics.csv`.
- **Interpretation note:** Parameter count is the § 14-defined primary
  complexity measure. Wall-clock is a contextual secondary measure only.
- **Limitation note:** Training wall-clock is hardware-dependent and was
  measured on a single local machine (see README and the `hardware_note`
  column of the source CSV). It should not be treated as a universal model
  property.
