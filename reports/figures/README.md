# Thesis figures

This folder contains the must-have figures for the Bachelor thesis
*"A Comparative Analysis of Predictive Performance and Interpretability in
Time Series Forecasting Models"*. The figures are generated deterministically
from already-validated CSV files in `results/bachelor_safe_v2/` and from the
raw `data/ETTh1.csv` dataset. No model training or evaluation is re-run.

## What is generated

| Number | Filename (stem)                                | Thesis section (suggested)  |
|-------:|------------------------------------------------|-----------------------------|
|      1 | `fig_01_dataset_overview`                      | Introduction / Methodology  |
|      2 | `fig_02_train_test_split`                      | Methodology                 |
|      3 | `fig_03_pipeline_architecture`                 | Solution                    |
|      4 | `fig_04_model_performance_comparison`          | Solution / Discussion (RQ1) |
|      5 | `fig_05_per_horizon_error`                     | Solution / Discussion (RQ1) |
|      6 | `fig_06_faithfulness_aopc_by_k`                | Solution / Discussion (RQ2) |
|      7 | `fig_07_xai_agreement_heatmap`                 | Solution / Discussion (RQ2) |
|      8 | `fig_08_performance_interpretability_tradeoff` | Discussion (RQ3)            |
|      9 | `fig_09_model_complexity`                      | Methodology / Solution      |

Each figure is exported as both `.pdf` (vector, for thesis embedding) and
`.png` at 300 DPI (for quick preview). Captions, interpretation notes, and
limitation notes are kept alongside the figures in `captions.md`.

## Required input files

The script fails clearly with an `ERROR:` message and a non-zero exit code if
any of the following inputs is missing.

| Figure | Required input file                                                  |
|-------:|----------------------------------------------------------------------|
|      1 | `data/ETTh1.csv`                                                     |
|      2 | `data/ETTh1.csv`                                                     |
|      3 | none (structural diagram)                                            |
|      4 | `results/bachelor_safe_v2/per_seed_metrics.csv`                      |
|      5 | `results/bachelor_safe_v2/per_horizon_metrics.csv`                   |
|      6 | `results/bachelor_safe_v2/faithfulness_aopc_per_k.csv`               |
|      7 | `results/bachelor_safe_v2/xai_agreement_occlusion.csv`               |
|      8 | `results/bachelor_safe_v2/trade_off_data.csv` and `per_seed_metrics.csv` |
|      9 | `results/bachelor_safe_v2/complexity_metrics.csv`                    |

## How to regenerate

From the repository root:

```bash
python scripts/make_figures.py --all
```

Generate one figure at a time:

```bash
python scripts/make_figures.py --figure 4
```

List the available figure numbers:

```bash
python scripts/make_figures.py --list
```

Output goes to `reports/figures/main/`. The directory is created if it does
not exist.

## Dependencies

`pandas`, `numpy`, `matplotlib`, and the Python standard library. No new
packages are introduced relative to the existing thesis environment.

## Reproducibility notes

- No random sampling is performed in the figure script; all values are read
  from the validated CSVs.
- Figure 8 recomputes the Pareto-efficient set of models from
  `trade_off_data.csv` at render time and prints it to stdout during
  generation; the set is not hard-coded.
- Figure 4 error bars represent ±1 standard deviation across seeds and are
  not confidence intervals; paired-difference 95 % moving-block bootstrap
  CIs remain in `results/bachelor_safe_v2/bootstrap_intervals.csv`.
- Existing result files in `results/bachelor_safe_v2/` are never modified.
- The two legacy PNGs inside `results/bachelor_safe_v2/`
  (`trade_off_plot.png`, `faithfulness_aopc_by_k.png`) are left untouched; the
  harmonised versions are generated as Figures 6 and 8 inside this folder.
- Style is centralised in `reports/figures/_style.py` so every figure shares
  the same colour palette, font sizes, and grid conventions.

## Hardware note for Figure 9

The training wall-clock values in `complexity_metrics.csv` were measured on
the student's local development machine and are reproduced in Figure 9 only
as a contextual secondary measure. They are not a universal
property of the models and should not be compared against third-party
training-time reports without matching hardware, batch size, and precision
settings. The `hardware_note` column in `complexity_metrics.csv` is the
authoritative record of the measurement environment.

## Caveats for thesis embedding

- Figures are produced in English; German captions are added by the student
  during final prose translation.
- Every figure that compares models carries an "observed / under the present
  configuration" framing in line with the comparability and interpretation
  checks of the thesis. Do not reword captions into absolute-ranking claims
  without the corresponding bootstrap-CI evidence.
- Faithfulness (Figures 6 and 8) is defined strictly through the project's
  occlusion-based AOPC instrument. Do not compare the numeric values to
  AOPC results from other XAI instruments or other masking baselines.
