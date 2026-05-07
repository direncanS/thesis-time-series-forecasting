=== PASTE INTO: Titel der Arbeit ===
Eine vergleichende Analyse von Vorhersageleistung und Interpretierbarkeit ausgewählter Zeitreihenvorhersagemodelle
=== END PASTE ===

=== PASTE INTO: Kurzfassung ===
Diese Bachelorarbeit untersucht Vorhersageleistung und Interpretierbarkeit ausgewählter Zeitreihenvorhersagemodelle unter kontrollierten experimentellen Bedingungen. Ausgangspunkt ist die Beobachtung, dass Modellvergleiche im Bereich der Zeitreihenvorhersage durch unterschiedliche Datenaufbereitung, Zieldefinitionen, Horizontlängen oder Erklärbarkeitsverfahren beeinflusst werden können. Ziel der Arbeit ist daher nicht ein isolierter Benchmark eines einzelnen Modells, sondern die Entwicklung und Auswertung einer reproduzierbarkeitsunterstützten Evaluationspipeline. Der Eigenanteil dieser Arbeit besteht im Entwurf, in der Implementierung, Validierung und Analyse einer reproduzierbarkeitsunterstützten Evaluationspipeline zum Vergleich ausgewählter Zeitreihenvorhersagemodelle unter identischen experimentellen Bedingungen.

Die Pipeline wird auf dem ETTh1-Datensatz angewandt, einem stündlichen multivariaten Benchmark mit sieben numerischen Variablen. Verglichen werden Linear Regression (LR), ein multilayer perceptron (MLP), ein LSTM-Modell und ein Temporal Fusion Transformer (TFT). Alle Modelle verwenden denselben chronologischen 60/20/20-Split, eine nur auf den Trainingsdaten angepasste Standardisierung, ein 96-Stunden-Eingabefenster und einen direkten 24-Stunden-Mehrschritt-Vorhersagehorizont. Die final berichteten Metriken werden im Originalmaßstab ausgewertet; diagnostische skalierte Metriken bleiben als Artefakte erhalten. Interpretierbarkeit wird primär über okklusionsbasierte Wichtigkeit, AOPC und Rangübereinstimmung operationalisiert. SHAP-Ausgaben für LR, MLP und LSTM sowie VSN- bzw. interpret_output-Wichtigkeiten für TFT werden nur als ergänzende, methodenspezifische Erklärungsartefakte behandelt.

Unter dem definierten ETTh1-Setup erzielte die deterministische LR-Baseline die niedrigsten Fehlerwerte unter den ausgewählten Modellen, während die neuronalen Modelle modell- und seedabhängige Unterschiede zeigten. Die Erklärbarkeitsauswertung beschreibt keine physikalischen Wirkzusammenhänge, sondern modellbezogene Unterschiede in okklusionsbasierter Wichtigkeit, AOPC und Rangübereinstimmung. Die Ergebnisse stützen eine vorsichtige Diskussion über Leistung, Interpretierbarkeit und Komplexität innerhalb dieses konkreten Versuchsaufbaus. Die Reproduzierbarkeit wird als automatisierte Artefaktprüfung beschrieben, nicht als allgemeine Garantie maschinenübergreifender Rechenergebnisse.
=== END PASTE ===

=== PASTE INTO: Schlagwörter ===
Zeitreihenvorhersage, fairer Modellvergleich, okklusionsbasierte Erklärbarkeit, AOPC, Reproduzierbarkeit
=== END PASTE ===

=== PASTE INTO: Abstract ===
This bachelor thesis investigates predictive performance and interpretability of selected time-series forecasting models under controlled experimental conditions. The motivation is that forecasting comparisons can be distorted when models are evaluated with different preprocessing choices, target definitions, horizons, tuning assumptions, or explanation methods. The thesis develops and analyses a reproducibility-supported evaluation pipeline that compares selected time-series forecasting models under identical experimental conditions.

The pipeline is applied to the ETTh1 dataset, an hourly multivariate benchmark with seven numerical variables. Four model families are compared: Linear Regression (LR), a multilayer perceptron (MLP), an LSTM network, and a Temporal Fusion Transformer (TFT). The models share a chronological 60/20/20 split, train-only standardization, split-local sliding-window generation, a 96-hour input window, and a direct 24-hour multi-step forecasting task. Final reported metrics are evaluated in the original data scale, while diagnostic scaled metrics may be retained as supporting artifacts. Interpretability is operationalized primarily through common occlusion importance, AOPC-based faithfulness evaluation, and rank agreement. SHAP outputs for LR, MLP, and LSTM and architecture-native VSN / interpret_output importance for TFT are treated as auxiliary, method-specific explanation artifacts.

Under the defined ETTh1 setup, the deterministic LR baseline achieved the lowest forecasting error among the selected models, while the neural models showed model- and seed-dependent differences. The interpretability analysis reports model-level evidence from occlusion behaviour, AOPC summaries, and rank agreement rather than statements about physical variable relevance. The results support a cautious discussion of predictive performance, interpretability, and operational complexity within this specific experimental setup. Reproducibility is framed as automated artifact-level verification, not as a general guarantee of machine-independent numerical execution.
=== END PASTE ===

=== PASTE INTO: Keywords ===
Time-series forecasting, fair model comparison, occlusion-based explainability, AOPC, reproducibility
=== END PASTE ===

=== PASTE INTO: 1. Introduction ===
# 1. Introduction

## 1.1 Motivation

Time-series forecasting is used when future values must be estimated from ordered historical observations. Technical and operational systems use forecasts for load, demand, temperature, traffic, and related signals. Predictive performance is important, but it is not the only relevant criterion. A model with a low error score may still be difficult to inspect, expensive to train, or unfairly compared if the experimental setup is not controlled.

This thesis approaches the problem from a fairness-oriented perspective. The comparison is not designed around one preferred architecture. Linear Regression, a multilayer perceptron, an LSTM network, and a Temporal Fusion Transformer are treated as selected representatives of different model families. The aim is to compare them under the same task definition and information constraints, rather than to ask whether a single advanced model is generally superior. This framing matters because higher operational complexity does not automatically imply better results for every dataset, horizon, metric, or explanation criterion.

Reproducibility is also part of the motivation. If a model comparison is reduced to a final metric table, it is difficult to verify how the values were produced. The thesis therefore treats the experimental pipeline itself as the artifact. The pipeline connects data preparation, model training, prediction export, evaluation, uncertainty analysis, explainability outputs, and artifact-level verification into one traceable workflow.

## 1.2 Problem Statement

The problem addressed in this thesis is the lack of a controlled, reproducibility-supported comparison of selected forecasting models under identical task, preprocessing, evaluation, and explanation conditions within the scope of this work. Forecasting results can become difficult to interpret when models use different data splits, scaling assumptions, target definitions, or explanation procedures. Under such conditions, observed differences may reflect implementation choices rather than model behaviour.

This thesis therefore defines the task as an experimental-system problem, not only as a model-performance problem. The goal is to build and evaluate a pipeline that enforces a shared forecasting contract and makes clear which data, partitions, predictions, metrics, and explanation outputs are used.

## 1.3 State of the Art and Research Gap

Time-series forecasting research includes classical statistical models, linear baselines, recurrent neural networks, and transformer-based architectures. Benchmark datasets such as ETTh1 are commonly used to compare forecasting methods on multivariate hourly data. Informer made transformer-style long-sequence forecasting visible in this field (Zhou et al., 2021), while TFT was introduced as a multi-horizon architecture with built-in interpretability components (Lim et al., 2021). Transformer-based forecasting also appears in energy, probabilistic energy, PV, and meteorological settings (Rathnayaka et al., 2022; Ye et al., 2023; Islam et al., 2023; Xue et al., 2025). These studies show relevant application contexts, but they do not remove the need for a controlled comparison inside this thesis.

At the same time, recent work has questioned whether more complex architectures are necessary by default for forecasting benchmarks. Zeng et al. (2023) show that simple linear baselines can be highly competitive in long-sequence forecasting settings. Ouyang, Ravier and Jabloun (2022) also discuss multistep forecasting strategies for deep learning models. These points motivate a design that includes both a deterministic linear baseline and neural models instead of assuming that complexity is beneficial.

Explainability research provides post-hoc methods and evaluation concepts. Lundberg and Lee (2017) define SHAP as an additive feature-attribution framework, Samek et al. (2017) use perturbation-based evaluation for explanation heatmaps, and Yeh et al. (2019) discuss infidelity and sensitivity as objective explanation measures. Recent application papers also use SHAP or interpretation-oriented methods in time-series contexts, including distributed fuzzy cognitive maps, cryptocurrency forecasting, and meteorological forecasting (Haritha and Judy, 2024; Shetty et al., 2025; Xue et al., 2025). In this thesis, these works support the explainability context, while the primary comparison remains the common occlusion protocol.

Existing studies often focus either on forecasting performance, specific model families, or individual explainability methods. Fewer studies combine a controlled multi-model forecasting comparison with a common post-hoc explanation protocol and reproducibility-oriented artifact verification under one setup. This thesis addresses that narrower gap by combining forecasting evaluation, explanation behaviour, uncertainty summaries, and artifact verification in one applied pipeline. It does not claim to introduce a new forecasting architecture or explainability theory.

## 1.4 Research Questions

The thesis is guided by three research questions:

RQ1: Under identical task, dataset, preprocessing, and evaluation conditions, how do selected forecasting models differ in predictive performance?

RQ2: Under a common post-hoc explanation protocol, how do the selected models differ in explanation faithfulness and rank agreement?

RQ3: Under the same experimental conditions, is there evidence of a performance-interpretability trade-off for the selected models and criteria?

These questions intentionally avoid universal model claims. They are limited to the selected models, selected dataset, defined input/output horizon, implemented metrics, and explanation criteria used in the pipeline.

## 1.5 Contributions / Eigenanteil

The main contribution of this thesis is the design, implementation, validation, and analysis of a reproducibility-supported evaluation pipeline for comparing selected time-series forecasting models under identical experimental conditions.

The contribution consists of connected parts: pipeline implementation, a fair-core comparison protocol, a common post-hoc explanation protocol, AOPC and rank-agreement evaluation, moving-block bootstrap uncertainty summaries, and automated artifact-level verification. Together, these parts define how the selected models are prepared, trained, evaluated, explained, and checked as one experimental artifact.

The contribution is therefore not the claim that one model family is generally best. It is the construction and analysis of a reproducibility-supported experimental framework in which performance, explanation behaviour, complexity, and uncertainty can be discussed together for the selected models and metrics.

## 1.6 Thesis Structure

The thesis follows a four-chapter structure. Chapter 1 introduces the motivation, problem statement, state of the art, research gap, research questions, and contribution. Chapter 2 explains the methodology, including research design, requirements, dataset, model-selection rationale, evaluation, explainability, and reproducibility concept. Chapter 3 presents the Solution: the experimental pipeline, generated artifacts, figures, tables, and core results. Chapter 4 discusses the research questions, limitations, generalisability, conclusion, and future work.
=== END PASTE ===

=== PASTE INTO: 2. Methodology ===
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

The four selected models represent different forms of modelling complexity. LR is included as a deterministic linear baseline and is fitted once, which is consistent with the linear-baseline debate in long-sequence forecasting (Zeng et al., 2023). MLP represents a feed-forward neural model, LSTM a recurrent sequence model, and TFT a transformer-family forecasting architecture with architecture-native explanation components (Lim et al., 2021). Ouyang, Ravier and Jabloun (2022) motivate treating multistep forecasting strategy as part of the comparison design rather than as a secondary implementation detail.

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

Uncertainty is evaluated with a paired moving-block bootstrap over aligned test windows, using exported prediction artifacts rather than retraining models. This follows the dependent-bootstrap motivation of Politis and White (2004) and reflects the overlapping-window structure better than an independent-sample bootstrap. Complexity is operationalized through parameter count, runtime trace, and model class, which are reported separately because they need not lead to the same ordering.

## 2.8 Explainability Protocol

The primary explainability workflow uses one common post-hoc measurement object: occlusion-based feature importance. For each variable, the model is evaluated again after that variable is perturbed in the scaled input space, and the change in error is recorded as model reliance under this perturbation design. AOPC is then used as a criterion-bound faithfulness measure by comparing highly ranked and lower-ranked variables, while SHAP remains an auxiliary attribution family following Lundberg and Lee (2017).

Rank agreement compares model-level feature-importance orderings across the selected models. It is evidence about agreement under the common occlusion protocol, not a formal statistical stability proof. SHAP outputs for LR, MLP, and LSTM are auxiliary; related time-series applications use SHAP in settings such as distributed fuzzy cognitive maps, cryptocurrency forecasting, and meteorological forecasting (Haritha and Judy, 2024; Shetty et al., 2025; Xue et al., 2025). TFT uses architecture-native VSN / `interpret_output` importance. These auxiliary outputs are not treated as method-identical with the primary occlusion workflow, and objective explanation-evaluation ideas such as infidelity and sensitivity remain outside the primary metric set (Yeh et al., 2019).

## 2.9 Reproducibility and Verification

The final artifact is tested through automated artifact-level verification for the active repository state. The verification checks the active configuration, expected result files, prediction arrays, checkpoints where applicable, primary explainability outputs, generated figures, and checksum manifest coverage. This supports traceability and completeness of produced pipeline outputs, but it is not a general claim about machine-independent numerical execution.

The development and evaluation process translated requirements into configuration files, training and evaluation scripts, generated artifacts, and verification checks. The active Makefile exposes commands for training, analysis, table generation, and verification, while detailed verification information is assigned to Appendix C. This keeps the main methodology focused on research design while preserving technical traceability.
=== END PASTE ===

=== PASTE INTO: 3. Solution ===
# 3. Solution

## 3.1 Pipeline Overview

The solution developed in this thesis is a reproducibility-supported forecasting evaluation pipeline. Its purpose is to compare the selected forecasting models under one shared experimental contract, not to produce an isolated result table. The pipeline starts with the ETTh1 dataset, removes the timestamp column from the numerical modelling tensor, applies a chronological train/validation/test split, fits the scaler only on the training partition, and creates supervised sliding windows inside each split. These prepared tensors form the common input for LR, MLP, LSTM, and TFT.

After data preparation, the pipeline separates model-specific training from downstream analysis. LR is fitted once as a deterministic baseline. MLP, LSTM, and TFT are trained as stochastic neural models across the active five-seed configuration: 42, 123, 456, 789, and 1024. For these neural models, per-seed checkpoints are part of the artifact. After training or fitting, predictions are exported as fixed arrays. Evaluation, uncertainty analysis, complexity reporting, and explainability analysis then consume those exported artifacts rather than repeatedly rerunning training code.

The pipeline has two downstream analysis branches. The evaluation branch produces final original-scale point metrics, paired moving-block bootstrap intervals, per-horizon descriptive metrics, complexity indicators, training-curve traces, runtime traces, figures, and generated tables. The explainability branch produces the primary occlusion-based importance outputs, AOPC summaries, and cross-model rank agreement. SHAP outputs for LR, MLP, and LSTM and TFT-native VSN / `interpret_output` importance are retained as auxiliary explanation artifacts. The final verification layer checks the active configuration, expected artifacts, checkpoints where applicable, prediction arrays, figures, and checksum manifest coverage.

This organization is important because the research questions depend on several linked outputs. RQ1 needs prediction tensors and original-scale metrics. RQ2 needs a shared explanation instrument and agreement measures. RQ3 needs performance, explainability, and complexity information in the same configuration. The pipeline therefore acts as the connecting artifact between methodology and results. It turns abstract requirements such as "same task" and "same preprocessing" into concrete files, arrays, tables, and figures that can be inspected.

## 3.2 Data Processing Component

The data-processing component is where the comparability contract first becomes concrete. The pipeline loads `data/ETTh1.csv`, whose columns are `date`, HUFL, HULL, MUFL, MULL, LUFL, LULL, and OT. The date column is used as a timestamp column and is removed from the modelling tensor. The remaining seven numerical variables are used both as inputs and as multivariate prediction targets. This choice avoids an OT-only task and keeps all four models aligned on the same multivariate forecasting problem.

![Figure 1: ETTh1 dataset overview](../reports/figures/main/fig_01_dataset_overview.png)

*Figure 1: Hourly multivariate ETTh1 over the 17,420-hour record with 60/20/20 split markers.*

Figure 1 shows the full ETTh1 record across the seven numerical variables. The figure is useful because it makes two properties visible before modelling begins: the variables do not share the same scale, and the time series changes over the chronological record. These two properties motivate train-only standardization and a chronological split. The split markers in the figure also show that the thesis evaluates later time segments as validation and test data rather than shuffling rows randomly.

The pipeline then performs a chronological 60/20/20 split. The `StandardScaler` is fitted only on the training partition and then applied to the validation and test partitions. This avoids using future validation or test distributional information in the scaling parameters. Sliding windows are created separately inside each split with a 96-hour input window and a 24-hour forecast horizon. Because windows are created after splitting, no supervised window crosses from training into validation or from validation into test.

![Figure 2: Chronological split and forecasting window](../reports/figures/main/fig_02_train_test_split.png)

*Figure 2: Chronological split with a representative 96-hour input + 24-hour forecast window.*

Figure 2 illustrates the split-then-window regime. The figure is not a result plot; it documents the experimental task. It shows that each forecast uses a fixed past context and predicts a fixed future horizon. This matters for fairness because every model receives the same amount of past information and is evaluated on the same direct multi-step forecasting task.

The output of the data-processing component is not a single table but a set of aligned tensors. The unflattened windows are needed by sequence models such as LSTM and TFT, while flattened versions are needed by LR and MLP. Both forms represent the same information: 96 time steps, seven variables, and a 24-step multivariate target. This shared representation is the reason the later model comparison can focus on model behaviour rather than on different input definitions.

## 3.3 Model Implementation Summary

The model layer implements four structurally different forecasting approaches while preserving the shared task. LR is the deterministic baseline. It receives the flattened 96 x 7 input window and predicts the flattened 24 x 7 output window once through a fitted linear model. Because LR is deterministic in this setup, it has no seed axis, no epoch-based early stopping, and no checkpoint.

MLP is implemented as a feed-forward neural model. LSTM is implemented as a recurrent model that compresses temporal information through a hidden state before the output layer. TFT is implemented through the `pytorch-forecasting` and Lightning pathway as a transformer-family forecasting model. MLP, LSTM, and TFT are trained across the active seeds 42, 123, 456, 789, and 1024. For these models, checkpoints are saved per seed and later restored for prediction export and evaluation.

The explanation interfaces differ by model family, but the primary comparison is kept common. LR, MLP, LSTM, and TFT all enter the primary occlusion workflow. SHAP outputs are produced only as auxiliary artifacts for LR, MLP, and LSTM. TFT is not treated as a SHAP model; its architecture-native explanation output comes from VSN / `interpret_output` importance.

![Figure 3: Pipeline architecture](../reports/figures/main/fig_03_pipeline_architecture.png)

*Figure 3: End-to-end fair-core evaluation pipeline (training-curve and runtime traces written to result CSVs).*

Figure 3 visualizes the full artifact structure. It shows that the data-processing layer feeds several model-specific training or fitting paths, and that downstream evaluation and explainability reuse the resulting artifacts. The diagram is important for the thesis argument because it shows why the artifact is a pipeline: performance metrics, uncertainty intervals, explanation outputs, figures, and verification records are generated by connected components rather than by disconnected model runs.

The implementation preserves model-specific differences where they are necessary. LR is not forced into an epoch-based training loop, because that would misrepresent the deterministic baseline. TFT is not rewritten as a plain PyTorch module, because its implementation depends on the `pytorch-forecasting` and Lightning ecosystem. MLP and LSTM use a custom PyTorch training loop because their architecture is simpler and the restore-best behaviour can be controlled directly. The comparison is therefore not "same code for every model", but "same experimental contract around model-specific code".

## 3.4 Evaluation and Artifact Generation

The evaluation component converts model predictions and targets back to the original ETTh1 scale before computing the final reported point metrics. MSE, MAE, and RMSE are computed over aligned prediction tensors. Each scalar metric aggregates across test windows, forecast horizon steps, and the seven target variables. For stochastic models, seed-level metrics are summarized across the five active seeds; LR reports one deterministic value. Diagnostic scaled metrics may still appear in supporting artifacts, especially where scaled inputs are required for explainability, but they are not the headline performance metrics.

Prediction export is the handoff between model training and downstream analysis. The active result directory contains `preds_lr.npy` for LR and per-seed prediction arrays for MLP, LSTM, and TFT. This frozen-prediction design matters because bootstrap intervals, point metric summaries, and several downstream checks can operate on fixed artifacts. It reduces the risk that later tables are silently generated from a different model state.

The generated CSV artifacts include per-seed metrics, per-horizon metrics, bootstrap intervals, runtime traces, training-curve traces, complexity indicators, occlusion importance, AOPC summaries, cross-model rank agreement, and trade-off data. The verification report in `results/bachelor_safe_v2/reproducibility_verification_report.json` records core mode with status PASS for the primary pipeline. It also records the active config, result directory, checkpoint directory, active seeds, checksum manifest status, and 74 verified artifacts. This is an artifact-level verification statement; it does not claim machine-independent numerical execution.

This artifact design also makes the result chapter easier to check. The same active directory contains the CSVs used for tables, the arrays used for downstream evaluation, the prediction files used for bootstrap input, and the figure files used for the thesis. For neural models, the checkpoint files provide the traceable model state behind exported predictions. For LR, traceability comes from the deterministic fitting procedure and the exported prediction array. This distinction keeps the LR exception explicit without weakening the shared evaluation path.

## 3.5 Predictive Performance Results

Within this experimental setup, the evaluated models show model-dependent differences across the final original-scale metrics. Tabelle 1 summarizes the overall MSE, MAE, and RMSE values. Lower values indicate lower forecasting error. The table reports a single fitted value for LR and mean ± standard deviation across stochastic seeds for MLP, LSTM, and TFT.

**Tabelle 1: Overall point metrics on ETTh1 in original scale, reported as mean and standard deviation across stochastic seeds where applicable; LR is deterministic and reports a single fitted value.**

Source: `results/bachelor_safe_v2/per_seed_metrics.csv`.

| Model | MSE | MAE | RMSE |
|---|---:|---:|---:|
| LR | 7.66 | 1.47 | 2.77 |
| MLP | 9.49 ± 0.15 | 1.81 ± 0.01 | 3.08 ± 0.02 |
| LSTM | 12.92 ± 0.43 | 2.14 ± 0.06 | 3.59 ± 0.06 |
| TFT | 16.34 ± 5.74 | 2.42 ± 0.29 | 4.00 ± 0.65 |

Tabelle 1 shows that the deterministic LR baseline has the lowest overall error values in the active ETTh1 configuration. MLP has the next lowest values, followed by LSTM and TFT. This ordering is descriptive for the selected dataset, split, horizon, and metrics. It should not be read as a general statement that simple models are better in general or that neural models cannot be useful in other forecasting settings.

The table also shows why seed handling must be visible in the thesis. MLP and LSTM have relatively small standard deviations in this run, while TFT has a larger seed-level spread in all three final metrics. This does not automatically make TFT unreliable in general, but it matters for the evaluated configuration because the same model family can produce noticeably different outcomes under different stochastic seeds. LR does not have this axis because it is fitted once as a deterministic baseline.

![Figure 4: Predictive performance comparison](../reports/figures/main/fig_04_model_performance_comparison.png)

*Figure 4: Mean test-set RMSE and MAE per model. Error bars indicate ±1 standard deviation across stochastic seeds where applicable; LR is deterministic and therefore has no seed-based variation.*

Figure 4 presents the same performance pattern visually for RMSE and MAE. The error bars represent seed-to-seed variation for the stochastic models, not confidence intervals. LR has no error bar because it is deterministic. The figure helps separate two ideas that are easy to mix: the mean error level and the dispersion across stochastic seeds.

The uncertainty layer uses paired moving-block bootstrap intervals over aligned test-window errors. Tabelle 2 reports MSE differences for all six model pairs. A negative value means that the first model has lower MSE than the second model under the exported prediction artifacts.

**Tabelle 2: Paired moving-block bootstrap 95 % confidence intervals on per-window MSE differences (block length L = 96, N = 10,000 resamples, BOOTSTRAP_SEED = 2026).**

Source: `results/bachelor_safe_v2/bootstrap_intervals.csv`.

| Pair (A vs B) | Mean MSE diff (A - B) | 95 % CI |
|---|---:|---:|
| LR vs MLP | -1.30 | [-1.63, -0.89] |
| LR vs LSTM | -4.41 | [-5.32, -3.41] |
| LR vs TFT | -5.79 | [-6.74, -4.61] |
| MLP vs LSTM | -3.11 | [-4.18, -2.05] |
| MLP vs TFT | -4.49 | [-5.57, -3.29] |
| LSTM vs TFT | -1.38 | [-1.73, -0.89] |

All six intervals are below zero in the reported pair direction. This supports the observed ordering under the active evaluation contract. The intervals should still be interpreted as comparative uncertainty evidence under overlapping time-series windows, not as strict independent-sample inference.

![Figure 5: Per-horizon test error](../reports/figures/main/fig_05_per_horizon_error.png)

*Figure 5: Per-horizon mean error per model, shown descriptively for the configured 24-step forecast horizon.*

Figure 5 makes the horizon axis visible. It plots test-set error across forecast steps 1 to 24, averaged across seeds for stochastic models and shown once for LR. The figure is descriptive only. It helps the reader see whether error patterns change across the forecast horizon, but the thesis does not introduce separate per-horizon significance tests or per-horizon model claims from this plot.

The predictive-performance layer therefore has three levels. Tabelle 1 gives the overall point metrics, Figure 4 makes RMSE and MAE visually comparable, and Tabelle 2 adds paired uncertainty intervals for MSE differences. Figure 5 then opens the horizon dimension without changing the main aggregation rule. This separation prevents the chapter from mixing headline comparison, uncertainty evidence, and descriptive diagnostics into one overloaded statement.

## 3.6 Interpretability Results

The primary interpretability comparison uses occlusion importance, AOPC, and rank agreement. The occlusion procedure perturbs one input variable at a time in the scaled input space and measures the change in model error. This gives a common post-hoc measurement object for all four selected models. It is a model-behaviour test under one perturbation rule, not a statement about physical variable relevance in the ETTh1 system.

![Figure 6: AOPC faithfulness by k](../reports/figures/main/fig_06_faithfulness_aopc_by_k.png)

*Figure 6: AOPC contribution per k in {1, ..., 7} per model.*

Figure 6 decomposes the AOPC calculation by the number of occluded variables k. This is important because the AOPC mean can hide whether a model's separation is driven by one highly ranked variable or by a broader ranking pattern. The figure supports the methodological caution that AOPC is criterion-bound: it evaluates faithfulness under the adopted occlusion rule and k range, not explanation quality in general.

The aggregated trade-off data report the following AOPC means: LR 0.202, MLP 0.113, LSTM 0.231, and TFT 0.180. LSTM has the highest AOPC mean, while LR has the lowest forecasting error in Tabelle 1. This means that the performance and AOPC axes do not produce the same ordering.

The AOPC values must be read together with Figure 6. A model can have a high mean AOPC because one or a few ranked variables produce a large error gap when occluded. Another model may show a flatter pattern across k. For this reason, the thesis treats AOPC as a controlled faithfulness criterion, not as a complete measure of interpretability. The plotted per-k behaviour is part of the evidence because it shows how the aggregate is formed.

![Figure 7: Cross-model XAI rank agreement](../reports/figures/main/fig_07_xai_agreement_heatmap.png)

*Figure 7: Pairwise Spearman rho on seed-averaged occlusion-importance vectors (rank agreement; not evidence about physical variable relevance).*

Figure 7 reports pairwise Spearman rank agreement over seed-averaged occlusion-importance vectors. The Spearman correlations are LR-MLP 0.000, LR-LSTM -0.179, LR-TFT 0.500, MLP-LSTM 0.750, MLP-TFT 0.357, and LSTM-TFT 0.500. The highest agreement is between MLP and LSTM. These values are primary RQ2 evidence because every model is measured through the same occlusion instrument. They are not a formal statistical stability proof.

![Figure 8: Performance and interpretability positioning](../reports/figures/main/fig_08_performance_interpretability_tradeoff.png)

*Figure 8: Performance-interpretability scatter for overall MSE and occlusion-based AOPC; AOPC error bars are across stochastic seeds where applicable.*

Figure 8 places each model on the two axes used for RQ3: overall MSE and AOPC. The plotted values are LR (MSE 7.66, AOPC 0.202), MLP (9.49, 0.113), LSTM (12.92, 0.231), and TFT (16.34, 0.180). No universal performance-interpretability rule follows from this plot. Within this configuration, the figure shows that LR and LSTM form the clearest comparison: LR has the lowest error, while LSTM has the highest AOPC under the selected occlusion criterion.

SHAP and VSN outputs remain outside the primary comparison. SHAP is used as auxiliary output for LR, MLP, and LSTM. TFT uses VSN / `interpret_output` importance as an architecture-native explanation. These outputs can support a construct-validity discussion, but they are not treated as method-identical to each other or to the primary occlusion workflow.

The distinction between primary and auxiliary explainability outputs is central for thesis safety. Occlusion importance is the common instrument because every model can be perturbed in the same prepared input space. SHAP and VSN do not share that property: SHAP is a post-hoc attribution method applied to fixed prediction functions, while VSN / `interpret_output` comes from the TFT architecture. The Solution chapter therefore reports SHAP/VSN as available artifacts but does not use them as the primary cross-model evidence.

## 3.7 Complexity and Reproducibility Results

Operational complexity is reported through three indicators: parameter count, runtime trace, and model class. The thesis does not reduce complexity to one number because the three indicators can lead to different orderings. Parameter count reflects the number of trainable or fitted parameters. Runtime trace reflects the observed wall-clock time in the local thesis setup. Model class describes the architectural family.

**Tabelle 3: Operational complexity indicators for the selected core models (parameter count, total wall-clock training time on a single NVIDIA RTX 5080 Laptop GPU at batch size 64 — runtime trace; LR is closed-form and reports a single fitted runtime).**

Source: `results/bachelor_safe_v2/complexity_metrics.csv`.

| Model | Parameter count | Model class | Runtime trace |
|---|---:|---|---:|
| LR | 113,064 | linear | 0.67 s |
| MLP | 124,328 | shallow-MLP | 28.60 s |
| LSTM | 29,608 | recurrent | 51.65 s |
| TFT | 18,261 | transformer-family | 2122.80 s |

Tabelle 3 shows why complexity must be treated carefully. MLP has the highest parameter count, TFT has the smallest parameter count, and TFT has the highest runtime trace. The architectural ordering and the parameter-count ordering therefore do not match. This matters for the later discussion of RQ1 and RQ3 because "more complex" can mean different things depending on the chosen complexity indicator.

![Figure 9: Model complexity overview](../reports/figures/main/fig_09_model_complexity.png)

*Figure 9: Operational complexity indicators per model (parameter count, runtime trace, model class).*

Figure 9 visualizes the same complexity information. The logarithmic parameter-count panel and the runtime panel show that a single complexity axis would be misleading. In the evaluated configuration, TFT is not the largest model by parameter count, but it is the most expensive model by runtime trace.

The current saved verification report records `mode: core` and `status: PASS`. It also records the active configuration `configs/experiments/fair_core_v2.yaml`, the result directory `results/bachelor_safe_v2`, the checkpoint directory `checkpoints/bachelor_safe_v2`, five active seeds, checksum manifest creation, and 74 verified artifacts. The report verifies the primary pipeline artifacts, including configuration checks, dataset checks, core CSVs, primary XAI CSVs, prediction arrays, checkpoints where applicable, figures, generated tables, and provenance outputs. Auxiliary SHAP regeneration is not part of the saved core report state.

This verification result supports the artifact definition used in the thesis. It shows that the active repository state contains the expected primary outputs for the configured pipeline. It does not replace scientific interpretation, and it does not make the numerical results universally reproducible across all hardware or software environments. Its role is narrower and practical: it checks that the active artifact tree is complete and internally consistent enough to serve as the evidence base for the written thesis.

## 3.8 Self-Contained Summary

Chapter 3 shows the artifact and the results together. The artifact is a controlled evaluation pipeline: it loads ETTh1, applies chronological splitting, fits the scaler only on training data, creates split-local windows, trains or fits the selected models, exports predictions, computes metrics, produces uncertainty and explainability outputs, and verifies expected artifacts. The core results show model-dependent forecasting differences, criterion-bound explanation behaviour, non-identical rank agreement patterns, and complexity indicators that do not collapse into one simple ordering.

The chapter is self-contained because the reader can see how the data flow, model implementation, evaluation outputs, interpretability outputs, figures, tables, and verification records connect. The appendix remains useful for technical details, but the main Solution chapter already explains the artifact, the central result tables, and the scope limits needed for the Discussion chapter.
=== END PASTE ===

=== PASTE INTO: 4. Discussion ===
# 4. Discussion

## 4.1 Interpretation of the Research Questions

### 4.1.1 RQ1: Predictive Performance

Within this experimental setup, the evaluated models show model-dependent differences in predictive performance. The interpretation is limited to the selected models, the ETTh1 dataset, the chronological 60/20/20 split, the 96-hour input to 24-hour direct multi-step forecasting task, and the final original-scale MSE, MAE, and RMSE metrics reported in Chapter 3. LR is deterministic and not seed-dependent; MLP, LSTM, and TFT were evaluated across the stochastic seeds {42, 123, 456, 789, 1024}. Under this setup, LR produced the lowest final error values, followed by MLP, LSTM, and TFT. This should be read as configuration-bound evidence, not as a general statement about model complexity. The result supports the practical lesson that architectures with higher operational complexity need controlled comparison against simple baselines before their added complexity can be justified.

### 4.1.2 RQ2: Explanation Faithfulness and Rank Agreement

For RQ2, the primary evidence is the common occlusion workflow, AOPC, and rank agreement. The explanation outputs show how the selected post-hoc criteria behave under the common occlusion protocol. AOPC measures whether variables ranked as important by a model produce a stronger error change when occluded than lower-ranked variables, but this is criterion-bound and depends on the perturbation setup. Rank agreement compares model-level occlusion-importance orderings; it is not a formal statistical stability claim. The strongest rank agreement in Chapter 3 was between MLP and LSTM, while other model pairs showed weaker or mixed agreement. SHAP outputs for LR, MLP, and LSTM and TFT-native VSN / interpret_output importance remain auxiliary because they are not method-identical explanation objects. The results should not be read as evidence about physical variable relevance in the ETTh1 system.

### 4.1.3 RQ3: Performance-Interpretability Trade-off

RQ3 is answered only within the evaluated MSE-AOPC representation. Figure 8 combines overall MSE with occlusion-based AOPC, so the trade-off statement is tied to those two axes. Within that representation, no model dominates all other models on both dimensions. LR has the lowest error, while LSTM has the highest AOPC under the selected criterion. This creates a visible tension between prediction error and the adopted faithfulness measure, but it does not establish a universal performance-interpretability trade-off. If the Pareto frontier is discussed, it should be understood only as the frontier for the plotted MSE-AOPC coordinates in this configuration.

## 4.2 Limitations and Validity

The main limitation is scope. All numerical results come from ETTh1 only. The fixed 96-hour input and 24-hour forecast horizon define one forecasting task, not a general time-series benchmark. The 60/20/20 row-based chronological split is internally consistent for this thesis, but it may differ from calendar-based conventions used in parts of the ETTh1 literature. The sliding-window samples also overlap, so they should not be treated as fully independent IID samples; the moving-block bootstrap addresses this partly but does not remove every dependence concern.

Model validity is also bounded by the locked configuration. The thesis uses a comparable tuning discipline rather than a broad hyperparameter search. This is important for TFT, whose default-sized configuration and seed 1024 result produced a high-error outlier in the active v2 results. The result is therefore evidence about the evaluated TFT configuration, not about all possible TFT variants. Construct validity is limited as well: the main metrics aggregate over windows, horizon steps, and variables, so no per-variable performance claim is made. The XAI results are tied to occlusion-based AOPC and rank agreement, and other explanation criteria could lead to different readings.

## 4.3 Generalisability and Lessons Learned

The exact numerical results are dataset- and configuration-bound. They should not be transferred directly to other datasets, horizons, or tuning budgets. The reusable part is the pipeline design. The same structure could be applied to other multivariate forecasting datasets, model sets, or input/output horizons if the data preparation, scaling, windowing, training, evaluation, and explanation contracts are updated consistently.

A key lesson is that fair comparison requires more than using the same dataset. Preprocessing, aggregation axes, checkpoint handling, random seeds, evaluation metrics, uncertainty analysis, and the explanation protocol must also be controlled. The fair-core protocol, common occlusion interface, and artifact-level verification are therefore transferable methodological ideas, even when the numerical ranking of models is not.

## 4.4 Conclusion and Future Work

### 4.4.1 Conclusion

This work designed and analysed a reproducibility-supported evaluation pipeline for selected time-series forecasting models under controlled experimental conditions. The pipeline made it possible to compare LR, MLP, LSTM, and TFT on ETTh1 using the same data preparation, forecasting task, evaluation logic, and primary explanation protocol. For RQ1, the selected models showed configuration-bound differences in predictive performance, with the deterministic LR baseline producing the lowest final errors in the active setup. For RQ2, the occlusion-based explanation outputs showed model-dependent faithfulness and rank-agreement patterns, without supporting claims about physical variable relevance. For RQ3, the MSE-AOPC view showed a criterion-bound tension between error and the selected faithfulness measure, but not a universal trade-off law. The main thesis output is therefore the controlled, traceable evaluation artifact and its careful analysis, rather than a universal ranking of forecasting model families.

### 4.4.2 Future Work

Future work should first add per-variable disaggregation so that performance and explanation outputs can be compared at a finer level than the current aggregate metrics. Second, richer faithfulness metrics, such as ordered-masking curves or continuous infidelity-style measures, could test whether the AOPC-based findings hold under other criteria. Third, a formal stability layer could be added if it follows the same fair-core constraints as the main pipeline. Further extensions include applying the pipeline to multiple datasets, using a budget-matched hyperparameter search for all models, and testing an independence-aware uncertainty method such as the stationary bootstrap. These extensions would broaden the evidence base without changing the artifact-first framing of the thesis.
=== END PASTE ===

=== PASTE INTO: Literaturverzeichnis ===
# References

<!-- Harvard author-date style. Entries below are active references cited in the thesis body. -->

Haritha, K. and Judy, M.V. (2024) 'Interpreting temporal dynamics in distributed fuzzy cognitive maps for time series prediction', in *2024 11th International Conference on Advances in Computing and Communications (ICACC)*. IEEE. doi:10.1109/ICACC63692.2024.10845549.

Islam, M.M., Shuvo, S.S., Shohan, M.J.A. and Faruque, M.O. (2023) 'Forecasting of PV plant output using interpretable temporal fusion transformer model', in *2023 North American Power Symposium (NAPS)*. IEEE. doi:10.1109/NAPS58826.2023.10318698.

Lim, B., Arık, S.Ö., Loeff, N. and Pfister, T. (2021) 'Temporal fusion transformers for interpretable multi-horizon time series forecasting', *International Journal of Forecasting*, 37(4), pp. 1748-1764. doi:10.1016/j.ijforecast.2021.03.012.

Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*. Long Beach, CA: Curran Associates, pp. 4765-4774. Available at: https://dl.acm.org/doi/10.5555/3295222.3295230.

Ouyang, J., Ravier, P. and Jabloun, M. (2022) 'Are deep learning models practically good as promised? A strategic comparison of deep learning models for time series forecasting', in *2022 30th European Signal Processing Conference (EUSIPCO)*. Belgrade, Serbia: IEEE, pp. 1477-1481. doi:10.23919/EUSIPCO55093.2022.9909926.

Politis, D.N. and White, H. (2004) 'Automatic block-length selection for the dependent bootstrap', *Econometric Reviews*, 23(1), pp. 53-70. doi:10.1081/ETC-120028836.

Rathnayaka, P., Moraliyage, H., Mills, N., De Silva, D. and Jennings, A. (2022) 'Specialist vs generalist: a transformer architecture for global forecasting energy time series', in *2022 15th International Conference on Human System Interaction (HSI)*. IEEE. doi:10.1109/HSI55341.2022.9869463.

Samek, W., Binder, A., Montavon, G., Lapuschkin, S. and Müller, K.-R. (2017) 'Evaluating the visualization of what a deep neural network has learned', *IEEE Transactions on Neural Networks and Learning Systems*, 28(11), pp. 2660-2673. doi:10.1109/TNNLS.2016.2599820.

Shetty, S.K., Saxena, T. and Shreyash (2025) 'Integrating ARIMA with SHAP for cryptocurrency price prediction', in *2025 IEEE International Conference on Electronics, Computing and Communication Technologies (CONECCT)*. IEEE. doi:10.1109/CONECCT65861.2025.11306541.

Xue, R., Zhou, J., Wang, J., Wen, Q. and Zhang, H. (2025) 'Long-term forecasting of atmospheric water vapor using transformer-based architectures and SHAP explainability', in *2025 6th International Conference on Machine Learning and Computer Application (ICMLCA)*. IEEE, pp. 1181-1184. doi:10.1109/ICMLCA66850.2025.11336567.

Ye, J., Zhao, B. and Liu, D. (2023) 'Temporal decomposition transformer for probabilistic energy forecasting', in *2022 First International Conference on Cyber-Energy Systems and Intelligent Energy (ICCSIE)*. IEEE. doi:10.1109/ICCSIE55183.2023.10175223.

Yeh, C.-K., Hsieh, C.-Y., Suggala, A.S., Inouye, D.I. and Ravikumar, P. (2019) 'On the (in)fidelity and sensitivity of explanations', in *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*. Vancouver, Canada: Curran Associates, pp. 10967-10978. Available at: https://dl.acm.org/doi/10.5555/3454287.3455271.

Zeng, A., Chen, M., Zhang, L. and Xu, Q. (2023) 'Are transformers effective for time series forecasting?', *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(9), pp. 11121-11128. doi:10.1609/aaai.v37i9.26317.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H. and Zhang, W. (2021) 'Informer: beyond efficient transformer for long sequence time-series forecasting', *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(12), pp. 11106-11115. doi:10.1609/aaai.v35i12.17325.
=== END PASTE ===

=== PASTE INTO: Abkürzungsverzeichnis ===
| Abbreviation | Meaning |
|---|---|
| AOPC | Area Over the Perturbation Curve; perturbation-based faithfulness criterion used for explanation evaluation |
| CI | Confidence interval |
| CSV | Comma-separated values file |
| ETTh1 | Electricity Transformer Temperature hourly benchmark dataset, first transformer station |
| GPU | Graphics processing unit |
| HUFL | ETTh1 variable: high useful load |
| HULL | ETTh1 variable: high useless load |
| IID | Independent and identically distributed |
| LR | Linear Regression |
| LSTM | Long Short-Term Memory network |
| LUFL | ETTh1 variable: low useful load |
| LULL | ETTh1 variable: low useless load |
| MAE | Mean Absolute Error |
| MLP | Multilayer Perceptron |
| MSE | Mean Squared Error |
| MUFL | ETTh1 variable: middle useful load |
| MULL | ETTh1 variable: middle useless load |
| OT | ETTh1 variable: oil temperature |
| RMSE | Root Mean Squared Error |
| RQ | Research question |
| SHAP | SHapley Additive exPlanations |
| TFT | Temporal Fusion Transformer |
| VSN | Variable Selection Network |
| XAI | Explainable Artificial Intelligence |
=== END PASTE ===

=== PASTE INTO: Dokumentationstabelle KI-basierte Hilfsmittel ===
| Feld | Inhalt |
|---|---|
| Tool | ChatGPT / KI-basiertes Sprachassistenzsystem |
| Verwendungszweck | Sprachliche Überarbeitung, insbesondere Prüfung von Grammatik, Verständlichkeit und semantisch unklaren Formulierungen. |
| Umfang der Nutzung | Unterstützung bei der Formulierung und sprachlichen Glättung einzelner Textpassagen. |
| Kontrolle durch den Autor | Alle fachlichen Inhalte, experimentellen Ergebnisse, Interpretationen und Quellenangaben wurden vom Autor eigenständig geprüft und verantwortet. |
=== END PASTE ===

=== PASTE INTO: Anhang ===
The main thesis text is intended to stand on its own. This appendix records technical details that support traceability, but are not required for understanding the core argument in Chapters 1 to 4.

## Appendix A. Hyperparameter Configurations

The values in this section document the active configuration used for the reported ETTh1 experiment. The primary configuration file is `configs/experiments/fair_core_v2.yaml`, which extends `configs/base.yaml`.

### Tabelle A.1: Pipeline-Level Constants

| Field | Value |
|---|---|
| Dataset path | `data/ETTh1.csv` |
| Active result directory | `results/bachelor_safe_v2/` |
| Active checkpoint directory | `checkpoints/bachelor_safe_v2/` |
| Forecasting target | All seven numerical ETTh1 variables |
| Chronological split | 60 percent train, 20 percent validation, 20 percent test |
| Scaling | `StandardScaler` fitted on the training partition only |
| Input length | 96 hours |
| Forecast horizon | 24 hours |
| Forecasting type | Direct multi-step prediction |
| Batch size | 64 |
| Maximum epochs | 200 |
| Patience | 10 |
| Stochastic seeds | 42, 123, 456, 789, 1024 |
| Final metrics | MSE, MAE, and RMSE after inverse transformation to the original data scale |
| Diagnostic metrics | Scaled-space diagnostics may be retained separately |
| Bootstrap method | Paired moving-block bootstrap |
| Bootstrap block length | 96 |
| Bootstrap resamples | 10,000 |
| Bootstrap seed | 2026 |

### Tabelle A.2: Linear Regression Configuration

| Field | Value |
|---|---|
| Implementation | `sklearn.linear_model.LinearRegression` |
| Input dimensionality | 672 values, corresponding to 96 hours times 7 variables |
| Output dimensionality | 168 values, corresponding to 24 hours times 7 variables |
| Optimisation | Closed-form OLS fitting |
| Seed handling | Not seed-dependent |
| Checkpointing | Not applicable |
| Deterministic-baseline role | LR is fitted once and then passed through the shared prediction and evaluation pipeline |

### Tabelle A.3: MLP Configuration

| Field | Value |
|---|---|
| Architecture | 672 -> 128 -> 128 -> 168 with ReLU activations |
| Implementation | PyTorch custom training loop |
| Optimiser | Adam |
| Learning rate | 0.0001 |
| Weight decay | 0.0 |
| Loss | `nn.MSELoss()` |
| Gradient clipping | `max_norm = 0.1` |
| Validation monitor | Validation loss |
| Model selection | Same validation-loss-based restore-best principle with framework-specific implementation |
| Checkpointing | One `.pt` checkpoint per stochastic seed |
| Seeds | 42, 123, 456, 789, 1024 |

### Tabelle A.4: LSTM Configuration

| Field | Value |
|---|---|
| Architecture | One LSTM layer with hidden size 64, followed by a linear output layer |
| Number of recurrent layers | 1 |
| Input dimensionality per time step | 7 variables |
| Output dimensionality | 168 values, corresponding to 24 hours times 7 variables |
| Implementation | PyTorch custom training loop |
| Optimiser | Adam |
| Learning rate | 0.0001 |
| Weight decay | 0.0 |
| Loss | `nn.MSELoss()` |
| Gradient clipping | `max_norm = 0.1` |
| Validation monitor | Validation loss |
| Model selection | Same validation-loss-based restore-best principle with framework-specific implementation |
| Checkpointing | One `.pt` checkpoint per stochastic seed |
| Seeds | 42, 123, 456, 789, 1024 |

### Tabelle A.5: TFT Configuration

| Field | Value |
|---|---|
| Implementation | `pytorch-forecasting` with Lightning |
| Hidden size | 16 |
| Attention heads | 4 |
| Hidden continuous size | 8 |
| Dropout | 0.1 |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0 |
| Loss | Unified MSE implementation for the active fair-core comparison |
| Gradient clipping | `gradient_clip_val = 0.1` |
| Target normalisation inside TFT dataset | Identity normaliser, because the shared train-only scaler has already been applied |
| Future known reals | None |
| Target scope | All seven ETTh1 variables |
| Output type | Deterministic point forecast |
| Validation monitor | Validation loss |
| Model selection | Same validation-loss-based restore-best principle with framework-specific implementation |
| Checkpointing | One `.ckpt` checkpoint per stochastic seed |
| Seeds | 42, 123, 456, 789, 1024 |

### Tabelle A.6: Toolchain Summary

| Tool or library | Role in the artifact |
|---|---|
| Python 3.10 | Main programming language and execution environment |
| PyTorch | Tensor operations and neural model training |
| Lightning | TFT training loop, callbacks, and checkpoint handling |
| `pytorch-forecasting` | TFT implementation |
| scikit-learn | LR baseline and `StandardScaler` |
| `arch` | Moving-block bootstrap implementation |
| SHAP | Auxiliary LR, MLP, and LSTM explanation artifacts |
| Matplotlib | Figure generation |
| PyYAML | Configuration loading |

## Appendix B. Diagnostic and Symmetrisation Notes

### B.1 Counterintuitive-Result Handling

The thesis does not discard a result because it is counterintuitive relative to expectations about model complexity. When LR reports lower final original-scale error than the neural models within this experimental setup, the result is handled as a configuration-bound empirical outcome. The supporting checks are the shared dataset contract, the same train-only scaling rule, split-local windowing, frozen prediction tensors, common metric aggregation, paired moving-block bootstrap intervals, and artifact-level verification. The interpretation remains limited to the selected dataset, models, metrics, and 96-to-24 forecasting task.

### B.2 CLIP-SYMMETRY-01

The stochastic neural paths use aligned gradient-clipping thresholds. The MLP and LSTM custom PyTorch loop applies `torch.nn.utils.clip_grad_norm_(..., max_norm=0.1)`. The TFT Lightning trainer uses `gradient_clip_val=0.1`. This detail reduces an avoidable training-protocol asymmetry among the stochastic neural models. It does not apply to LR because LR is a deterministic closed-form baseline rather than an iterative neural optimisation path.

### B.3 Training-Protocol Asymmetries Register

| Asymmetry | Reason | Handling in the thesis |
|---|---|---|
| LR has no seed axis | LR is fitted once as a deterministic OLS baseline | Reported separately from stochastic seed summaries |
| LR has no checkpoint | No iterative state needs to be restored for LR | Traceability is provided through the fitting and prediction-export procedure |
| MLP/LSTM use custom PyTorch loops while TFT uses Lightning | TFT is implemented through `pytorch-forecasting` and Lightning callbacks | The common rule is validation-loss-based restore-best state selection, not identical implementation code |
| TFT uses an identity normaliser inside its dataset object | The shared `StandardScaler` has already standardised the data | This avoids double scaling while keeping the train-only scaling contract |
| Auxiliary explanation outputs differ by model family | SHAP and TFT-native VSN / `interpret_output` are different explanation objects | The primary comparison uses occlusion importance, AOPC, and rank agreement instead |

### B.4 Ensemble-Versus-Single-Seed Distinction

The stochastic model summaries are not ensemble forecasts. MLP, LSTM, and TFT are trained separately for the five active seeds, and their results are reported as seed-level values plus mean and standard deviation where applicable. The prediction tensors remain seed-specific. LR is not seed-dependent and therefore has one fitted value rather than a seed distribution.

## Appendix C. Reproducibility Verification Details

The current saved verification report is `results/bachelor_safe_v2/reproducibility_verification_report.json`. It records `mode: core` and `status: PASS` for the primary pipeline state. The report also records the active configuration `configs/experiments/fair_core_v2.yaml`, result directory `results/bachelor_safe_v2`, checkpoint directory `checkpoints/bachelor_safe_v2`, five active seeds, checksum manifest creation, and 74 verified artifacts.

### Tabelle C.1: Verification Groups in the Saved Core Report

| Group | Checked content |
|---|---|
| Configuration | Active config path, data path, input length, output length, result directory, checkpoint directory, and seeds |
| Dataset | Dataset file, expected columns, row count, and missing-value check |
| Core CSV artifacts | Training summaries, per-seed metrics, per-horizon metrics, bootstrap intervals, runtime traces, training-curve traces, complexity metrics, and block-sensitivity data |
| Primary XAI CSV artifacts | Occlusion importance, AOPC summaries, occlusion rank agreement, TFT importance, and trade-off data |
| Prediction arrays | LR predictions and one prediction array for each stochastic model seed |
| Checkpoints | No LR checkpoint expected; MLP, LSTM, and TFT checkpoints expected for each stochastic seed |
| Figures | Figure 1 to Figure 9 in both PDF and PNG form |
| Generated tables | Markdown tables generated from the active result files |
| Provenance outputs | Checksum manifest and verification report |

The saved report verifies the primary artifact tree. Auxiliary SHAP regeneration is not part of the saved core report state; the report records `auxiliary_shap_regenerated: false` and `auxiliary_xai_verified: false`. This distinction is kept so that the thesis does not present auxiliary explanation artifacts as part of the current saved core verification state.

The verification statement is artifact-level. It checks that the active repository state contains the expected files and internally consistent outputs for the documented configuration. It is not a general statement about machine-independent numerical execution across arbitrary hardware or software environments.

## Appendix D. Auxiliary SHAP and VSN Outputs

The primary interpretability comparison in the thesis uses occlusion importance, AOPC, and rank agreement. SHAP and TFT-native outputs are retained only as auxiliary explanation artifacts because they are not method-identical.

### Tabelle D.1: Auxiliary Explanation Artifacts

| Artifact | Role |
|---|---|
| `results/bachelor_safe_v2/shap_lr.csv` | Auxiliary SHAP attribution output for LR |
| `results/bachelor_safe_v2/shap_mlp.csv` | Auxiliary SHAP attribution output for MLP |
| `results/bachelor_safe_v2/shap_lstm.csv` | Auxiliary SHAP attribution output for LSTM |
| `results/bachelor_safe_v2/shap_lr_cross.csv` | Auxiliary cross-model SHAP-format output for LR |
| `results/bachelor_safe_v2/shap_mlp_cross.csv` | Auxiliary cross-model SHAP-format output for MLP |
| `results/bachelor_safe_v2/shap_lstm_cross.csv` | Auxiliary cross-model SHAP-format output for LSTM |
| `results/bachelor_safe_v2/tft_importance.csv` | TFT-native VSN / `interpret_output` importance |
| `results/bachelor_safe_v2/xai_agreement_shap_vsn.csv` | Auxiliary SHAP/VSN agreement artifact |

These files may be useful for additional construct-validity discussion, but they are not used as the primary cross-model explanation evidence. The common comparison remains the occlusion-based workflow because the same perturbation protocol can be applied to all four selected model families.

## Appendix E. Extended Result Tables

### Tabelle E.1: Per-Seed Point Metrics in Original Scale

Source: `results/bachelor_safe_v2/per_seed_metrics.csv`.

| Model | Seed | MSE | MAE | RMSE |
|---|---:|---:|---:|---:|
| LR | deterministic | 7.660473 | 1.465105 | 2.767756 |
| MLP | 42 | 9.704269 | 1.824736 | 3.115168 |
| MLP | 123 | 9.378154 | 1.820981 | 3.062377 |
| MLP | 456 | 9.570781 | 1.807955 | 3.093668 |
| MLP | 789 | 9.444790 | 1.807262 | 3.073238 |
| MLP | 1024 | 9.336527 | 1.794262 | 3.055573 |
| LSTM | 42 | 12.795810 | 2.092083 | 3.577123 |
| LSTM | 123 | 12.807588 | 2.171723 | 3.578769 |
| LSTM | 456 | 13.574508 | 2.205135 | 3.684360 |
| LSTM | 789 | 13.020063 | 2.184277 | 3.608332 |
| LSTM | 1024 | 12.390662 | 2.064890 | 3.520037 |
| TFT | 42 | 13.843709 | 2.353622 | 3.720714 |
| TFT | 123 | 14.990285 | 2.366630 | 3.871729 |
| TFT | 456 | 13.744812 | 2.247099 | 3.707400 |
| TFT | 789 | 12.625386 | 2.217307 | 3.553222 |
| TFT | 1024 | 26.500271 | 2.938889 | 5.147841 |

### Tabelle E.2: Paired Moving-Block Bootstrap Intervals

Source: `results/bachelor_safe_v2/bootstrap_intervals.csv`. The bootstrap uses block length 96, 10,000 resamples, and seed 2026.

| Pair | Metric | Mean difference | 95 percent CI low | 95 percent CI high |
|---|---|---:|---:|---:|
| LR vs MLP | MSE | -1.303264 | -1.632323 | -0.892369 |
| LR vs MLP | MAE | -0.266486 | -0.310032 | -0.208887 |
| LR vs MLP | RMSE | -0.252443 | -0.305068 | -0.178764 |
| LR vs LSTM | MSE | -4.413084 | -5.318296 | -3.410152 |
| LR vs LSTM | MAE | -0.586847 | -0.658561 | -0.497758 |
| LR vs LSTM | RMSE | -0.739074 | -0.871760 | -0.575593 |
| LR vs TFT | MSE | -5.791200 | -6.741966 | -4.612476 |
| LR vs TFT | MAE | -0.799158 | -0.873996 | -0.687095 |
| LR vs TFT | RMSE | -0.944549 | -1.071772 | -0.767360 |
| MLP vs LSTM | MSE | -3.109820 | -4.178061 | -2.053014 |
| MLP vs LSTM | MAE | -0.320361 | -0.409885 | -0.230770 |
| MLP vs LSTM | RMSE | -0.486631 | -0.649627 | -0.317876 |
| MLP vs TFT | MSE | -4.487936 | -5.571081 | -3.289024 |
| MLP vs TFT | MAE | -0.532672 | -0.613125 | -0.435136 |
| MLP vs TFT | RMSE | -0.692106 | -0.841328 | -0.518736 |
| LSTM vs TFT | MSE | -1.378116 | -1.730803 | -0.885637 |
| LSTM vs TFT | MAE | -0.212311 | -0.256155 | -0.153113 |
| LSTM vs TFT | RMSE | -0.205475 | -0.264532 | -0.131254 |

### Tabelle E.3: Operational Complexity Indicators

Sources: `results/bachelor_safe_v2/complexity_metrics.csv` and `results/bachelor_safe_v2/runtime_seconds.csv`.

| Model | Parameter count | Architectural category | Wall-clock seconds |
|---|---:|---|---:|
| LR | 113064 | linear | 0.672768 |
| MLP | 124328 | shallow-MLP | 28.600817 |
| LSTM | 29608 | recurrent | 51.653864 |
| TFT | 18261 | transformer-family | 2122.803423 |
=== END PASTE ===
