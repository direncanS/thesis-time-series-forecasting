| Conceptual component | File(s) | Status | Output artefact(s) |
|----------------------|---------|--------|---------------------|
| Data processing | `src/training/multi_seed.py`; `src/training/tft_fair_3seed.py` | **PENDING (v2 auditor pass)** | scaled tensors in memory |
| LR training path | `src/training/multi_seed.py` (LR pathway) | **PENDING (v2 auditor pass)** | `results/multi_seed_fair_baseline.csv` (LR row) |
| MLP training path | `src/training/multi_seed.py` (MLP pathway) | **PENDING (v2 auditor pass)** | `results/multi_seed_fair_baseline.csv` (MLP rows); `checkpoints/mlp_seed*.pt` |
| LSTM training path | `src/training/multi_seed.py` (LSTM pathway) | **PENDING (v2 auditor pass)** | `results/multi_seed_fair_baseline.csv` (LSTM rows); `checkpoints/lstm_seed*.pt` |
| TFT training path | `src/training/tft_fair_3seed.py` | **PENDING (v2 auditor pass)** | `results/tft_summary.csv`; `results/tft_metrics.csv`; `results/tft_training_curves.csv`; `results/tft_importance.csv`; `checkpoints/tft_seed*.ckpt` |
| Prediction export | `src/evaluation/export_predictions.py` | **PENDING (v2 auditor pass)** | `results/preds_*.npy` (shape `(n_test, 24, 7)`) |
| Post-training analysis (per-seed + complexity + bootstrap) | `src/evaluation/post_training_analysis.py` | **PENDING (v2 auditor pass)** | `results/per_seed_metrics.csv`; `results/complexity_metrics.csv`; `results/bootstrap_intervals.csv` + `bootstrap_block_sensitivity.csv` |
| Explainability — SHAP | `src/explainability/shap_lr.py`; `src/explainability/shap_mlp.py`; `src/explainability/shap_lstm.py` | **VALIDATED** | `results/shap_{lr,mlp,lstm}.csv` |
| Explainability — VSN | `src/training/tft_fair_3seed.py` (TFT VSN extraction) | **PENDING (v2 auditor pass)** | `results/tft_importance.csv` |
| Faithfulness layer | `src/explainability/faithfulness_test.py` | **PENDING (v2 auditor pass)** | `results/faithfulness.csv` |
| Cross-model XAI agreement | `src/explainability/cross_model_xai_agreement.py` | **PENDING (v2 auditor pass)** | `results/xai_agreement.csv` |
| Accuracy ↔ interpretability trade-off plot | `src/explainability/trade_off_plot.py` | **PENDING (v2 auditor pass)** | `results/trade_off_data.csv`; `results/trade_off_plot.png` |
| Per-horizon disaggregated metrics | `src/evaluation/per_horizon_metrics.py` | **PENDING (v2 auditor pass)** | `results/per_horizon_metrics.csv` |
