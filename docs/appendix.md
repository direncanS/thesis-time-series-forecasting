# Abkürzungsverzeichnis

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

# Dokumentationstabelle KI-basierte Hilfsmittel

| Feld | Inhalt |
|---|---|
| Tool | ChatGPT / KI-basiertes Sprachassistenzsystem |
| Verwendungszweck | Sprachliche Überarbeitung, insbesondere Prüfung von Grammatik, Verständlichkeit und semantisch unklaren Formulierungen. |
| Umfang der Nutzung | Unterstützung bei der Formulierung und sprachlichen Glättung einzelner Textpassagen. |
| Kontrolle durch den Autor | Alle fachlichen Inhalte, experimentellen Ergebnisse, Interpretationen und Quellenangaben wurden vom Autor eigenständig geprüft und verantwortet. |

# Appendix

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
