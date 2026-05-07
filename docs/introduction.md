# 1. Introduction

## 1.1 Motivation

Time-series forecasting is used when future values must be estimated from ordered historical observations. In technical and operational systems, this can support decisions about load, demand, temperature, traffic, or other signals that change over time. Predictive performance is important in such settings, but it is not the only relevant criterion. A model with a low error score may still be difficult to inspect, expensive to train, or unfairly compared with simpler alternatives if the experimental setup is not controlled.

This thesis approaches the problem from a fairness-oriented perspective. The comparison is not designed around one preferred architecture. Linear Regression, a multilayer perceptron, an LSTM network, and a Temporal Fusion Transformer are treated as selected representatives of different model families. The aim is to compare them under the same task definition and information constraints, rather than to ask whether a single advanced model is generally superior. This framing matters because higher operational complexity does not automatically imply better results for every dataset, horizon, metric, or explanation criterion.

Reproducibility is also part of the motivation. If a model comparison is reduced to a final metric table, it is difficult to verify how the values were produced. The thesis therefore treats the experimental pipeline itself as the artifact. The pipeline connects data preparation, model training, prediction export, evaluation, uncertainty analysis, explainability outputs, and artifact-level verification into one traceable workflow.

## 1.2 Problem Statement

The problem addressed in this thesis is the lack of a controlled, reproducibility-supported comparison of selected forecasting models under identical task, preprocessing, evaluation, and explanation conditions within the scope of this work. Forecasting results can become difficult to interpret when models use different data splits, scaling assumptions, target definitions, or explanation procedures. Under such conditions, observed differences may reflect implementation choices rather than model behaviour.

This thesis therefore defines the task as an experimental-system problem, not only as a model-performance problem. The goal is to build and evaluate a pipeline that enforces a shared forecasting contract for the selected models and metrics. The pipeline must make clear which data are used, how training, validation, and test partitions are constructed, how predictions are evaluated, and how explanation outputs are compared.

## 1.3 State of the Art and Research Gap

Time-series forecasting research includes classical statistical models, linear baselines, recurrent neural networks, and transformer-based architectures. Benchmark datasets such as ETTh1 are commonly used to compare forecasting methods on multivariate hourly data. The ETT benchmark context is also connected to long-sequence forecasting work such as Informer (Zhou et al., 2021), which helped make transformer-style forecasting models visible in this field.

At the same time, recent work has questioned whether more complex architectures are always necessary for forecasting benchmarks. Zeng et al. (2023) show that simple linear baselines can be highly competitive in long-sequence forecasting settings. This is relevant for the selected models in this thesis because it motivates a comparison that includes both a deterministic linear baseline and neural models. The purpose is not to assume that complexity is beneficial, but to test model-dependent differences under one defined setup.

Neural and transformer-based forecasting models remain important because they can represent nonlinear temporal patterns and, in some cases, include architecture-native explanation mechanisms. The Temporal Fusion Transformer, for example, was introduced for interpretable multi-horizon forecasting (Lim et al., 2021). Explainability research also provides post-hoc methods and evaluation concepts, including SHAP for model explanations (Lundberg and Lee, 2017), AOPC-style perturbation evaluation (Samek et al., 2017), and faithfulness-oriented criteria such as infidelity and sensitivity (Yeh et al., 2019).

Existing studies often focus either on forecasting performance, specific model families, or individual explainability methods. Fewer studies combine a controlled multi-model forecasting comparison with a common post-hoc explanation protocol and reproducibility-oriented artifact verification under one experimental setup. This thesis addresses that narrower gap by combining forecasting evaluation, explanation behaviour, uncertainty summaries, and artifact verification in one applied pipeline for the selected dataset and model set. The gap is stated cautiously: the thesis does not claim to introduce a new forecasting architecture or a new explainability theory. It contributes a bounded experimental artifact for comparing selected models within this experimental setup.

## 1.4 Research Questions

The thesis is guided by three research questions:

RQ1: Under identical task, dataset, preprocessing, and evaluation conditions, how do selected forecasting models differ in predictive performance?

RQ2: Under a common post-hoc explanation protocol, how do the selected models differ in explanation faithfulness and rank agreement?

RQ3: Under the same experimental conditions, is there evidence of a performance-interpretability trade-off for the selected models and criteria?

These questions intentionally avoid universal model claims. They are limited to the selected models, selected dataset, defined input/output horizon, implemented metrics, and explanation criteria used in the pipeline.

## 1.5 Contributions / Eigenanteil

The main contribution of this thesis is the design, implementation, validation, and analysis of a reproducibility-supported evaluation pipeline for comparing selected time-series forecasting models under identical experimental conditions.

The contribution consists of connected parts: pipeline implementation, a fair-core comparison protocol, a common post-hoc explanation protocol, AOPC and rank-agreement evaluation, moving-block bootstrap uncertainty summaries, and automated artifact-level verification. Together, these parts make the work more than a metric-table comparison. They define how the selected models are prepared, trained, evaluated, explained, and checked as one experimental artifact.

The contribution is therefore not the claim that one model family is generally best. It is the construction and analysis of a reproducibility-supported experimental framework in which performance, explanation behaviour, complexity, and uncertainty can be discussed together for the selected models and metrics.

## 1.6 Thesis Structure

The thesis follows a four-chapter structure. Chapter 1 introduces the motivation, problem statement, state of the art, research gap, research questions, and contribution. Chapter 2 explains the methodology, including research design, requirements, dataset, model-selection rationale, evaluation procedure, explainability protocol, and reproducibility concept. Chapter 3 presents the implemented Solution: the experimental pipeline, generated artifacts, figures, tables, and core results. Chapter 4 discusses the research questions, limitations, generalisability, conclusion, and future work.
