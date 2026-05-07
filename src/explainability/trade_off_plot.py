"""Accuracy ↔ interpretability trade-off plot.

Visualises the RQ3 trade-off by plotting per-model overall MSE (predictive
accuracy axis) against per-model AOPC (continuous faithfulness axis). AOPC
replaces the v1 binary top-vs-bottom pass rate (flat y-axis) and the
post-hoc ``gap_mean`` patch it superseded; both are retired by the v6.1
plan (see methodology § 2.9.X for the transparency note).

Role separation
------------------------------------------
- **Occlusion** = the common explanation *instrument*
  (``results/bachelor_safe_v2/occlusion_importance.csv``).
- **AOPC** = the faithfulness *metric* computed over occlusion ranks
  (``results/bachelor_safe_v2/faithfulness_aopc.csv``).

This file consumes the metric only; it does not compute the instrument.

Inputs
------
    results/bachelor_safe_v2/per_seed_metrics.csv       — accuracy axis (original-scale MSE)
    results/bachelor_safe_v2/faithfulness_aopc.csv      — summary AOPC per (model, seed)
    results/bachelor_safe_v2/faithfulness_aopc_per_k.csv — diagnostic per-k gaps

Outputs
-------
    results/bachelor_safe_v2/trade_off_data.csv         — per-model 2D table
    results/bachelor_safe_v2/trade_off_plot.png         — scatter
    results/bachelor_safe_v2/faithfulness_aopc_by_k.png — per-model gap curves over k

Run
---
    python src/explainability/trade_off_plot.py \\
        --config configs/experiments/fair_core_v2.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.common import apply_runtime_overrides, ensure_dir, load_config

MODELS_ORDERED = ("LR", "MLP", "LSTM", "TFT")
MODEL_COLORS = {
    "LR": "tab:blue",
    "MLP": "tab:orange",
    "LSTM": "tab:green",
    "TFT": "tab:red",
}


# ---------------------------------------------------------------------------
# Axis loaders
# ---------------------------------------------------------------------------


def load_accuracy(results_path: Path) -> dict[str, float]:
    per_seed = results_path / "per_seed_metrics.csv"
    if not per_seed.exists():
        raise FileNotFoundError(f"missing {per_seed} — run post_training_analysis.py first")
    df = pd.read_csv(per_seed)
    grouped = df.groupby("model")["mse_original"].mean().to_dict()
    return {m: float(grouped[m]) for m in MODELS_ORDERED if m in grouped}


def load_aopc(results_path: Path) -> tuple[dict[str, tuple[float, float]], pd.DataFrame | None]:
    """Return ({model: (aopc_mean, aopc_std_across_seeds)}, per_k_df-or-None)."""
    aopc_csv = results_path / "faithfulness_aopc.csv"
    if not aopc_csv.exists():
        raise FileNotFoundError(f"missing {aopc_csv} — run faithfulness_test.py first")
    df = pd.read_csv(aopc_csv)
    out: dict[str, tuple[float, float]] = {}
    for model in MODELS_ORDERED:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        mean = float(sub["aopc"].mean())
        std = float(sub["aopc"].std(ddof=1)) if len(sub) > 1 else 0.0
        out[model] = (mean, std)

    per_k_csv = results_path / "faithfulness_aopc_per_k.csv"
    per_k_df = pd.read_csv(per_k_csv) if per_k_csv.exists() else None
    return out, per_k_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_trade_off(data: pd.DataFrame, out_png: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for _, row in data.iterrows():
        m = row["model"]
        ax.errorbar(
            row["mse_mean"],
            row["aopc_mean"],
            yerr=row["aopc_std_across_seeds"],
            fmt="o",
            markersize=10,
            color=MODEL_COLORS.get(m, "gray"),
            ecolor="gray",
            capsize=4,
            elinewidth=1.2,
            zorder=3,
            label=m,
        )
        ax.annotate(
            m,
            xy=(row["mse_mean"], row["aopc_mean"]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=11,
        )
    ax.set_xlabel("Overall MSE (original scale, seed-mean) — lower is more accurate")
    ax.set_ylabel("AOPC — mean over k ∈ {1..7} of (top-k − bottom-k) scaled-MSE increase")
    ax.set_title("Accuracy ↔ interpretability trade-off (RQ3; occlusion instrument)")
    ax.grid(True, alpha=0.3, zorder=0)
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_aopc_by_k(per_k_df: pd.DataFrame, out_png: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for model in MODELS_ORDERED:
        sub = per_k_df[per_k_df["model"] == model]
        if sub.empty:
            continue
        per_k_mean = sub.groupby("k")["gap"].mean().sort_index()
        per_k_std = sub.groupby("k")["gap"].std(ddof=1).sort_index().fillna(0.0)
        ax.errorbar(
            per_k_mean.index.values,
            per_k_mean.values,
            yerr=per_k_std.values,
            marker="o",
            linewidth=1.8,
            color=MODEL_COLORS.get(model, "gray"),
            ecolor="gray",
            capsize=3,
            label=model,
        )
    ax.set_xlabel("k (number of top / bottom variables occluded)")
    ax.set_ylabel("Gap = top-k − bottom-k scaled-MSE increase (seed-mean)")
    ax.set_title("Per-model AOPC decomposition over k")
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_trade_off(
    config_path: str | None = None,
    *,
    results_dir: str | None = None,
) -> pd.DataFrame:
    cfg = apply_runtime_overrides(load_config(config_path), smoke=False, results_dir=results_dir)
    results_path = ensure_dir(cfg["results_dir"])

    print(f"[B7b-trade-off] results_dir={results_path}")
    accuracy = load_accuracy(results_path)
    aopc_by_model, per_k_df = load_aopc(results_path)

    rows = []
    for model in MODELS_ORDERED:
        if model not in accuracy or model not in aopc_by_model:
            continue
        aopc_mean, aopc_std = aopc_by_model[model]
        rows.append(
            {
                "model": model,
                "mse_mean": round(accuracy[model], 6),
                "aopc_mean": round(aopc_mean, 6),
                "aopc_std_across_seeds": round(aopc_std, 6),
            }
        )
    data = pd.DataFrame(rows)
    data_csv = results_path / "trade_off_data.csv"
    data.to_csv(data_csv, index=False)
    print(f"  saved {data_csv} ({len(data)} rows)")
    print(data.to_string(index=False))

    if data.empty:
        print("[warn] no plottable rows — skipping figures.")
        return data

    scatter_png = plot_trade_off(data, results_path / "trade_off_plot.png")
    print(f"  saved {scatter_png}")
    if per_k_df is not None and not per_k_df.empty:
        by_k_png = plot_aopc_by_k(per_k_df, results_path / "faithfulness_aopc_by_k.png")
        print(f"  saved {by_k_png}")

    print("\n[read guide]")
    print("  x-axis: per-model seed-averaged overall MSE (lower = more accurate).")
    print("  y-axis: per-model seed-averaged AOPC (higher = more faithful).")
    print("  Error bars on y = across-seed std (ddof=1). AOPC aggregates over")
    print("  k ∈ {1..7} the gap (top_k − bot_k) scaled-MSE increase; higher =")
    print("  stronger behavioural separation between the model's top-ranked and")
    print("  bottom-ranked variables under occlusion of the common instrument.")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="B7b — accuracy × AOPC trade-off plot (RQ3)."
    )
    parser.add_argument("--config", default="configs/experiments/fair_core_v2.yaml")
    parser.add_argument("--results-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_trade_off(args.config, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
