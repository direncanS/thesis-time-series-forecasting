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
