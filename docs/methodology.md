# 2. Methodology

## 2.1 Research Design

This thesis is designed as a reproducibility-supported evaluation pipeline for a fairness-oriented comparison of selected time-series forecasting models. The artifact is the pipeline that prepares the data, trains the selected models, exports predictions, evaluates results, produces explanation outputs, and verifies expected artifacts. The final metric tables are outputs of this artifact, not the artifact itself. LR, MLP, LSTM, and TFT keep model-specific training implementations, but they must use the same dataset, chronological split, preprocessing rule, input/output horizon, target structure, metric definitions, and reporting discipline.

## 2.1.1 Requirements Engineering

The functional requirements were derived from the research questions: identical task configuration, reproducible preprocessing, shared evaluation metrics, common explainability outputs, and preserved generated artifacts. All core models must consume the same prepared ETTh1 tensors, use the same 96-hour input and 24-hour output task, and produce outputs for one downstream evaluation layer.

The non-functional requirements are auditability, deterministic configuration where feasible, artifact-level verification, and separation between primary and auxiliary explainability outputs. Result CSVs, prediction arrays, checkpoints where applicable, figures, and verification records must be traceable to the active configuration. Primary XAI outputs are kept separate from auxiliary SHAP and TFT-native explanation artifacts.

## 2.2 Dataset and Forecasting Task

The dataset is ETTh1 (Zhou et al., 2021), an hourly multivariate electricity-load benchmark with 17,420 observations and seven numerical variables. All seven numerical variables are used both as inputs and as multivariate prediction targets; the task is not restricted to `OT` as a single target. The data are split chronologically into 60 percent training, 20 percent validation, and 20 percent test data. The task is direct multi-step forecasting: each model receives a 96-hour input window and predicts the next 24 hours for all seven variables in one forward pass.

## 2.3 Data Preparation and Windowing

Data preparation follows a split-first design. The chronological train, validation, and test partitions are created before supervised windows are generated. A `StandardScaler` is fitted only on the training partition and then applied to validation and test data. This prevents validation or test distributional information from entering the scaling parameters. Final reported metrics are computed after inverse transformation to the original data scale, while diagnostic scaled quantities may still be retained.

Sliding windows are constructed separately inside each split through `src/common.py:create_windows` and `src/common.py:prepare_supervised_data`. This split-local windowing prevents windows from crossing train/validation/test boundaries. METHOD-NOTE-01 is the split-then-window rule used throughout the thesis; external ETTh1 benchmark values that use another windowing convention are not treated as directly comparable.

## 2.4 Model Selection and Tools

The four selected models represent different forms of modelling complexity. LR is included as a deterministic linear baseline and is fitted once. MLP represents a feed-forward neural model, LSTM a recurrent sequence model, and TFT a transformer-family forecasting architecture with architecture-native explanation components. This selection supports comparison across simple, neural, recurrent, and transformer-family approaches under one controlled task.

MLP, LSTM, and TFT are stochastic models and are evaluated over the five active v2 seeds: 42, 123, 456, 789, and 1024. Checkpoints are relevant only for these neural models. LR has no seed-specific checkpoint because it is fitted deterministically as a closed-form baseline. This exception is part of the methodology rather than a missing training feature.

The implementation uses Python 3.10 as specified in the environment file. PyTorch is used for neural modelling, Lightning and `pytorch-forecasting` for TFT, and scikit-learn for LR and `StandardScaler`. The `arch` package provides `MovingBlockBootstrap`, and SHAP is used only for auxiliary LR/MLP/LSTM explanation artifacts. These tools were chosen because they support explicit control over tensors, checkpoints, metrics, and artifacts. A fully off-the-shelf forecasting workflow was not selected because the thesis requires one shared comparison contract rather than separate defaults with hidden preprocessing or evaluation assumptions.

## 2.5 Fairness and Comparability Protocol

The fairness protocol defines which comparisons are admissible. All core models must use the same dataset, target scope, chronological split, scaling rule, input length, forecast horizon, metric aggregation, and evaluation outputs. No model may receive additional future information, a narrower target task, or a different metric definition. The deterministic-baseline exception applies to LR: it is fitted once and does not use seed repetition, epoch-based early stopping, or checkpoint selection.

The protocol also controls reporting language. Claims are restricted to the selected dataset, selected models, implemented metrics, and defined input/output horizon. This prevents the thesis from turning an observed result into a universal statement about model classes.

## 2.6 Training Protocol

The stochastic models use the active five-seed configuration: 42, 123, 456, 789, and 1024. They are trained under the same batch size, maximum epoch limit, validation-loss monitoring idea, and reporting discipline defined by the active configuration. Neural model selection follows the same validation-loss-based restore-best principle with framework-specific implementation: a custom PyTorch loop for MLP/LSTM and Lightning callbacks for TFT.

LR is fitted once and evaluated through the same downstream prediction and metric pipeline. Detailed notes on counterintuitive-result handling, CLIP-SYMMETRY-01, residual training-protocol asymmetries, and the ensemble-versus-single-seed distinction are assigned to Appendix B.

## 2.7 Evaluation Procedure

Evaluation keeps the final reported metrics comparable across models. Predictions and targets used for final reporting are transformed back to the original ETTh1 scale before computing MSE, MAE, and RMSE. Diagnostic scaled metrics may also be retained for internal checks or explainability. Metrics are aggregated over test windows, forecast horizon steps, and the seven target variables. For stochastic models, seed-level results are summarized across the five active seeds; LR reports a single deterministic value.

Uncertainty is evaluated with a paired moving-block bootstrap over aligned test windows, using exported prediction artifacts rather than retraining models. This reflects the overlapping-window structure better than an independent-sample bootstrap. Complexity is operationalized through parameter count, runtime trace, and model class, which are reported separately because they need not lead to the same ordering.

## 2.8 Explainability Protocol

The primary explainability workflow uses one common post-hoc measurement object: occlusion-based feature importance. For each variable, the model is evaluated again after that variable is perturbed in the scaled input space, and the change in error is recorded as model reliance under this perturbation design. AOPC is then used as a criterion-bound faithfulness measure by comparing highly ranked and lower-ranked variables.

Rank agreement compares model-level feature-importance orderings across the selected models. It is evidence about agreement under the common occlusion protocol, not a formal statistical stability proof. SHAP outputs for LR, MLP, and LSTM are auxiliary. TFT uses architecture-native VSN / `interpret_output` importance. These auxiliary outputs are not treated as method-identical with the primary occlusion workflow.

## 2.9 Reproducibility and Verification

The final artifact is tested through automated artifact-level verification for the active repository state. The verification checks the active configuration, expected result files, prediction arrays, checkpoints where applicable, primary explainability outputs, generated figures, and checksum manifest coverage. This supports traceability and completeness of produced pipeline outputs, but it is not a general claim about machine-independent numerical execution.

The development and evaluation process translated requirements into configuration files, training and evaluation scripts, generated artifacts, and verification checks. The active Makefile exposes commands for training, analysis, table generation, and verification, while detailed verification information is assigned to Appendix C. This keeps the main methodology focused on research design while preserving technical traceability.
