# 1. Introduction

<!-- Student writes this chapter. thesis-coach provides keyword bullets on request. -->
<!-- Rule 10: Claude never writes sentences here. -->
<!-- Target share of main body: ~15 %. -->

## 1.1 Motivation

<!-- S-09e bridge paragraph (2026-04-20): TFT-aware origin, fairness-first maturation. -->

This thesis began with a concrete curiosity about modern transformer-class architectures for time-series forecasting — in particular, whether the Temporal Fusion Transformer (Lim et al., 2021) lives up to its reputation on benchmark datasets such as ETTh1 (Zhou et al., 2021), and whether its architecture-native explainability could compete with post-hoc methods such as SHAP (Lundberg and Lee, 2017) applied to simpler baselines. Pursuing that question rigorously, however, made it clear that asking *"is TFT good?"* in isolation is methodologically unsound: any answer depends on what the TFT is being compared against, what information regime it is given, what tuning budget it receives, and what aggregation rule the metrics use. The work therefore matured into a **fairness-first comparison across four model families** — Linear Regression, an MLP, an LSTM, and the TFT — under a single controlled pipeline that enforces identical task and information constraints, and into a **trade-off study** that quantifies how predictive accuracy and interpretability move together (or apart) along the architectural-complexity spectrum. The original TFT-centric curiosity is preserved as the motivation; the research questions of § 1.4 are the spectrum-level form that this curiosity took once the methodological constraints were taken seriously.

## 1.2 Problem Statement

<!-- The specific problem this thesis addresses. -->

## 1.3 Research Gap

<!-- What is underexplored in the literature and why this thesis fills part of it. -->

## 1.4 Research Questions

<!-- RQ1: Does higher complexity really yield better forecasts? -->
<!-- RQ2: How reliable are the explanations (XAI faithfulness)? -->
<!-- RQ3: Is there a trade-off between accuracy and interpretability? -->

## 1.5 Contributions

<!-- Eigenanteil: pipeline design, implementation, experiment orchestration, reproducibility verification, evaluation logic, explainability integration. -->

## 1.6 Thesis Structure

<!-- One paragraph mapping chapters. -->
