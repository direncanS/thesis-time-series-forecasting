# 3. Solution

<!-- Target share: ~50 % — heaviest chapter. -->
<!-- S-07b 2026-04-19: lifted from skeleton via thesis-writer revise mode. -->
<!-- Claims about cross-model performance respect § 11C: overall observations only; per-variable / per-horizon claims forbidden until disaggregated path implemented. -->

## 3.1 Overview of the Developed Experimental Framework

The artifact developed in this thesis is a **reproducible forecasting evaluation framework** that runs four structurally different model families — Linear Regression (LR), a multilayer perceptron (MLP), an LSTM recurrent network, and a Temporal Fusion Transformer (TFT) — through a single, controlled pipeline that enforces identical task and information constraints across all of them. The framework is the contribution; the metric table it produces is its output, not its purpose. This framing follows directly from the artifact definition fixed in the project guidelines (§ 5) and from the FH thesis structure (§ 7), which assigns the largest weight of the main body to the Solution chapter.

The framework is purpose-built rather than assembled from off-the-shelf forecasting libraries. Standard libraries provide model implementations but do not enforce a single environment in which the same chronological split, the same input/output horizon, the same train-only scaling rule, the same metric-computation logic, the same seed-handling discipline, the same checkpoint-traceability rule, and a comparable interpretability protocol can be applied across model families. The Fairness and Comparability Protocol of the Methodology (§ 2.5) describes the eight-item rule each model must satisfy; the present chapter describes the engineering that makes the rule operational.

The pipeline ingests the ETTh1 hourly multivariate electricity-load dataset, runs all four core models under their locked hyperparameter defaults, exports an aligned set of per-test-window prediction tensors, computes original-scale point metrics with seed-level aggregation, and feeds two downstream layers — a paired-bootstrap uncertainty layer and a complexity-reporting layer — that together produce the evidence base for the research-question discussion. A separate explainability workflow attaches post-hoc SHAP attributions to the LR / MLP / LSTM models and the architecture-native Variable Selection Network (VSN) importance to TFT.

## 3.2 Pipeline Architecture

The pipeline is composed of seven conceptual components arranged in a linear data flow with two parallel downstream branches:

1. **Data processing** — load, chronological split, train-only standardisation, sliding-window construction.
2. **Model-specific training** — three core training paths (LR closed-form solver; MLP / LSTM gradient-based with Adam; TFT under the `pytorch-forecasting` framework) each producing per-seed checkpoints.
3. **Best-checkpoint restoration** — early-stopping on validation loss with patience 10 and restore-best-weights logic identical across stochastic models.
4. **Prediction export** — frozen, original-scale per-test-window prediction tensors of shape `(3 365, 24, 7)` per model and per seed, written to disk as the canonical handoff to all downstream layers.
5. **Original-scale evaluation** — MSE / MAE / RMSE computed on the inverse-transformed prediction tensors with seed-level aggregation.
6. **Uncertainty layer** — paired bootstrap on aligned test-window prediction-error differences for the six pairwise comparisons.
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

For each stochastic model, the training loop is repeated for three seeds (42, 123, 456) using `torch.manual_seed`; the resulting per-seed checkpoints are persisted to `checkpoints/`.

## 3.5 Evaluation Component

The evaluation component closes the loop on the original scale. For every (model, seed) combination, the best checkpoint is restored, predictions for the 3 365 test windows are produced in a single forward pass (or, for LR, a single matrix multiplication), and the inverse-transform step reverses the train-fit `StandardScaler` to return values to the original units of each variable.

Three point metrics are computed per (model, seed): **MSE**, **MAE**, and **RMSE**. Each is the `np.mean` of the per-element error tensor of shape (3 365, 24, 7). Per the four-axis aggregation contract (§ 2.10.1), this scalar mean averages simultaneously across the forecast horizon, across the seven variables, and across the test windows. The seed axis is summarised separately: for each metric, the per-seed scalars are reported as **mean ± std with `ddof = 1`** over seeds {42, 123, 456}; for the deterministic LR path the seed axis is not applicable.

The component writes a single output table (`results/multi_seed_fair_baseline.csv` for LR / MLP / LSTM, `results/tft_summary.csv` for TFT) and a per-epoch training-curves CSV that retains the validation-loss trajectory for each (model, seed) pair. These artefacts are then consumed by the uncertainty and complexity layers below.

The disaggregation doctrine of § 11C is enforced at this layer: claims at the per-variable or per-horizon granularity are not supported by the present scalar aggregation, and a disaggregated evaluation path is left as a dedicated future-work item (§ 4.6).

## 3.6 Explainability Component

The explainability component implements the interpretability contract of § 13: interpretability in this thesis is operationalised as **aggregated, model-level feature-importance comparison**, not as a claim of full causal understanding, and faithfulness is **tested**, not assumed.

For LR, MLP, and LSTM, the component runs the **SHAP** algorithm (Lundberg and Lee, 2017) against `N_EVAL = 100` evaluation windows at `EVAL_SEED = 42` (locked in § 10), using each model's best-checkpoint weights, and writes the per-feature attribution profile to `results/shap_<model>.csv`. The attribution layer uses a fixed baseline — the training-distribution mean in the scaled input space (methodology § 2.9.0), which coincides with the zero-vector of the scaled input space by `StandardScaler` construction. This baseline is operationalised as `shap.LinearExplainer` with the full training set as background for LR, and as `shap.DeepExplainer` (MLP) / `shap.GradientExplainer` (LSTM) with a fixed training subset of `N_BG = 100` windows drawn at `BG_SEED = 42` for the two neural models, holding the reference distribution identical across MLP and LSTM per the preprocessing-symmetry requirement of § 11A item 5.

For TFT, the component reads the architecture-native **Variable Selection Network (VSN)** importance produced by `pytorch-forecasting` and writes it to `results/tft_importance.csv`. SHAP and VSN are *not equivalent* explanation methods: SHAP is a post-hoc attribution layer applied to a fixed prediction function, whereas VSN is an intrinsic component of the TFT architecture whose weights are co-optimised with the rest of the network. Section § 4.2 below treats this distinction in detail and constrains the cross-method comparison to qualitative observations.

A faithfulness layer (`src/explainability/faithfulness_test.py`) is wired into this component but is currently **PENDING** validation (`docs/VALIDATION_LOG.md`). A dedicated stability layer was previously prototyped but was dropped on 2026-04-20 (S-11 archive migration; `stability_test.py` retrained models under lr=0.001 + 50-epoch + single-seed, inconsistent with § 10 locked defaults); explanation-stability in the present thesis is therefore restricted to the seed-CV evidence already produced by the 3-seed training pipeline (§ 3.4). Any explanation result discussed in this thesis is reported under the preliminary-results clause of § 20: observations may be reported, but no definitive comparison or causal explanation may be drawn until the faithfulness script moves from PENDING to VALIDATED.

## 3.7 Reproducibility Verification Component

Reproducibility operates on two tiers.

**Tier 1 (byte-level).** Deterministic code paths — closed-form OLS for LR and the prediction-export layer that consumes already-trained checkpoints — are required to produce byte-identical outputs across reruns. An automation script (`src/utils/verify_reproducibility.py`) was prototyped for this purpose but was dropped on 2026-04-20 (S-11 archive migration) because it reconstructed TFT under the OT-only + future-covariates regime (inconsistent with § 10 fair-core) and re-trained the model inside the check rather than loading the authoritative checkpoints. The Tier 1 checks listed in Appendix E are therefore verified *ad hoc* from the current `results/` + `checkpoints/` artefacts until a fair-core replacement is written.

**Tier 2 (metric-tolerance).** Iteratively-trained GPU code paths — the TFT training path under non-deterministic CuDNN kernels — are not required to be byte-identical, but their summary metrics must reproduce within seed-level tolerance. The present TFT thesis-valid baseline (MSE 14.19 ± 0.68) was established by a single audited run; rerun reproducibility is therefore expressed as a tolerance band around this baseline rather than as bit-exact identity.

Per-seed checkpoints are stored under `checkpoints/`; the test-time prediction tensors of all four models are exported to `results/preds_*.npy` (shape `(3 365, 24, 7)`, original scale) so that the downstream uncertainty and faithfulness layers operate from a single frozen prediction artefact rather than from re-loaded model weights. This frozen-prediction-artefact discipline is the primary reproducibility lever: whatever the reproducibility properties of the upstream training paths, the downstream comparison numbers are computed from a fixed input.

The component also encodes a checkpoint-traceability rule: every reported metric and every bootstrap interval is traceable, by file path, to the checkpoint that produced the underlying predictions. Restoration of any reported number from cold storage requires only the corresponding checkpoint file and the deterministic export step.

## 3.8 Experimental Results

This section reports the **observed overall performance** of the four core models under the locked configuration described above. All claims in this section are bounded by the Ranking-Interpretation Gate of § 11C: the Comparability Gate has cleared 10 / 10 (`docs/VALIDATION_LOG.md` row `COMPAR-GATE`), so cross-model interpretation is permitted, but the disaggregation doctrine restricts claims to the **overall** aggregation level only. Per-variable and per-horizon claims are not supported by the present aggregation and are flagged as future work in § 4.6. Interpretation of these numbers is deferred to the Discussion (§ 4.1, § 4.3).

### 3.8.1 Overall point metrics (original scale)

The per-seed mean and standard deviation of each metric over seeds {42, 123, 456} are reported below; LR is deterministic.

| Model | Parameter count | MSE (mean ± std) | MAE (mean ± std) | RMSE (mean ± std) | Best epochs (per seed) |
|-------|-----------------|-------------------|-------------------|---------------------|--------------------------|
| LR    |  113 064        | 7.66              | 1.47              | 2.77                | — (closed-form)          |
| MLP   |  124 328        | 9.36 ± 0.11       | 1.80 ± 0.03       | 3.06 ± 0.02         | 13 / 19 / 21             |
| LSTM  |   29 608        | 12.67 ± 0.46      | 2.14 ± 0.05       | 3.56 ± 0.07         | 19 / 34 / 42             |
| TFT   |   18 261        | 14.19 ± 0.68      | 2.32 ± 0.07       | 3.77 ± 0.09         | 5 / 14 / 9               |

The observed overall ordering on every reported metric, under the present configuration, is **LR < MLP < LSTM < TFT** (lower error first). The same ordering holds when point estimates are recomputed from seed-averaged-prediction ensembles rather than per-seed metric means: LR 7.66, MLP 8.96, LSTM 11.92, TFT 12.80. The gap between the two computation modes (per-seed mean vs ensemble) varies systematically across models (§ 4.1), which is itself a § 17 Internal-validity observation.

### 3.8.2 Pairwise uncertainty intervals

Paired-bootstrap 95 % confidence intervals on the mean per-window error difference for each of the six model pairs were computed with `BOOTSTRAP_SEED = 2026` and `N = 10 000` resamples (`results/bootstrap_intervals.csv`); the safety sentence of § 16 applies — *Interval estimates are used as comparative uncertainty evidence, not as a claim of strict independent-sample inference.* Sliding-window overlap on ETTh1 violates the strict independence assumption underlying the standard bootstrap, and the intervals carry that residual dependence risk.

For the MSE metric, the six pairwise intervals are:

| Pair (A vs B) | Mean diff (A − B) | 95 % CI         |
|---------------|--------------------|-----------------|
| LR vs MLP     | −1.30              | [−1.39, −1.22]  |
| LR vs LSTM    | −4.26              | [−4.44, −4.08]  |
| LR vs TFT     | −5.14              | [−5.31, −4.96]  |
| MLP vs LSTM   | −2.95              | [−3.16, −2.76]  |
| MLP vs TFT    | −3.84              | [−4.01, −3.66]  |
| LSTM vs TFT   | −0.88              | [−1.03, −0.74]  |

All six intervals exclude zero, so all six pairwise mean MSE differences are statistically distinguishable at the bootstrap-CI level under the present configuration. The MAE and RMSE intervals tell the same story (`results/bootstrap_intervals.csv`).

### 3.8.3 Complexity context

Three operationalisations of "complexity" are reported (§ 14, `results/complexity_metrics.csv`): parameter count (primary), wall-clock training time (secondary, with hardware note), and architectural category. Parameter counts appear in the table of § 3.8.1; the architectural-category ordering is **linear (LR) < shallow-MLP (MLP) < recurrent (LSTM) < transformer-family (TFT)**; wall-clock training time is dominated by TFT (best-effort estimate ≈ 915 s across the three seeds on a single NVIDIA RTX 5080 Laptop GPU with batch size 64), with LR completing in under one second by closed-form solver; MLP and LSTM wall-clock seconds were not instrumented in the training script and are acknowledged as a § 14 secondary-metric reporting gap.

The three orderings are mutually orthogonal: parameter-count ordering is **MLP > LR > LSTM > TFT**, architectural-category ordering is **LR < MLP < LSTM < TFT**, and accuracy ordering (lower error first) is **LR > MLP > LSTM > TFT**. None of the three coincide. This three-way orthogonality is taken up in § 4.1 and § 4.3 as a material framing point for RQ1 and RQ3.

### 3.8.4 Aggregated explainability outputs (preliminary)

The per-feature SHAP attribution profiles for LR / MLP / LSTM and the VSN importance vector for TFT are written to `results/shap_<model>.csv` and `results/tft_importance.csv` respectively. These outputs are reported here as **preliminary** under § 20 because the SHAP scripts (`src/explainability/shap_lr.py`, `src/explainability/shap_mlp.py`, `src/explainability/shap_lstm.py`) and the faithfulness script (`src/explainability/faithfulness_test.py`) are PENDING in `docs/VALIDATION_LOG.md`. A dedicated stability layer was dropped in the S-11 archive migration (see § 3.6 above); explanation-stability is therefore covered only by the seed-CV evidence from the 3-seed training pipeline, and the present section makes no standalone stability claim. Final claims about explanation quality, agreement, and faithfulness await the corresponding `ml-experiment-auditor` sign-off; the chapter therefore confines itself to the observation that the explainability artefacts have been generated and are aligned with the same checkpoints used in §§ 3.8.1–3.8.2.
