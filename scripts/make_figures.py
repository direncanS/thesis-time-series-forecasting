#!/usr/bin/env python3
"""Generate the nine must-have thesis figures from validated result files.

Reads only existing CSVs in ``results/bachelor_safe_v2/`` and the raw
dataset in ``data/ETTh1.csv``. Does not modify experiment outputs. Does
not retrain or re-evaluate anything.

Usage:
    python scripts/make_figures.py --all
    python scripts/make_figures.py --figure 4
    python scripts/make_figures.py --list

Outputs go to ``reports/figures/main/`` as both .pdf (thesis embedding)
and .png (quick preview) at 300 DPI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports" / "figures"))

from _style import (  # noqa: E402
    MODEL_COLOURS,
    MODEL_ORDER,
    SPLIT_COLOURS,
    apply_style,
    colour_for,
)

RESULTS = ROOT / "results" / "bachelor_safe_v2"
DATA = ROOT / "data"
FIG_OUT = ROOT / "reports" / "figures" / "main"

VARIABLES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
INPUT_LEN = 96
OUTPUT_LEN = 24
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2


def require(path: Path, hint: str = "") -> Path:
    if not path.exists():
        msg = f"ERROR: required input not found: {path}"
        if hint:
            msg += f"\n       {hint}"
        raise SystemExit(msg)
    return path


def save(fig: plt.Figure, name: str) -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf and {name}.png")


def _per_model_seed_stats(per_seed: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Mean and std across seeds per model for the given metric column."""
    grouped = per_seed.groupby("model")[metric_col].agg(["mean", "std", "count"])
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped.reindex(MODEL_ORDER)


def _pareto_frontier(tradeoff: pd.DataFrame) -> set[str]:
    """Return the set of Pareto-efficient models for (min mse_mean, max aopc_mean).

    A model A is Pareto-efficient iff no other model B satisfies
    ``B.mse_mean <= A.mse_mean`` AND ``B.aopc_mean >= A.aopc_mean`` with at
    least one strict inequality. Computed from ``trade_off_data.csv`` at
    render time so the figure stays consistent if the underlying numbers
    change.
    """
    rows = tradeoff.to_dict("records")
    pareto: set[str] = set()
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            better_or_equal = (
                b["mse_mean"] <= a["mse_mean"]
                and b["aopc_mean"] >= a["aopc_mean"]
            )
            strictly_better = (
                b["mse_mean"] < a["mse_mean"]
                or b["aopc_mean"] > a["aopc_mean"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.add(str(a["model"]))
    return pareto


def fig_01_dataset_overview() -> None:
    print("fig_01_dataset_overview")
    src = require(DATA / "ETTh1.csv", "expected from the thesis data directory")
    df = pd.read_csv(src, parse_dates=["date"])
    n = len(df)
    train_end = int(TRAIN_FRAC * n)
    val_end = int((TRAIN_FRAC + VAL_FRAC) * n)

    fig, axes = plt.subplots(
        len(VARIABLES), 1, figsize=(7.5, 8.5), sharex=True
    )
    for ax, var in zip(axes, VARIABLES):
        ax.plot(df["date"], df[var], linewidth=0.4, color="#222222")
        ax.set_ylabel(var, rotation=0, ha="right", va="center", fontsize=9)
        ax.axvline(df["date"].iloc[train_end], linestyle="--", linewidth=0.6, color="grey")
        ax.axvline(df["date"].iloc[val_end], linestyle="--", linewidth=0.6, color="grey")
        ax.grid(False)
        ax.tick_params(labelsize=7)
    axes[0].set_title(
        "ETTh1 — hourly multivariate series; 60/20/20 chronological split markers overlaid",
        fontsize=10,
    )
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    save(fig, "fig_01_dataset_overview")


def fig_02_train_test_split() -> None:
    print("fig_02_train_test_split")
    src = require(DATA / "ETTh1.csv")
    n = sum(1 for _ in open(src)) - 1
    train_end = int(TRAIN_FRAC * n)
    val_end = int((TRAIN_FRAC + VAL_FRAC) * n)

    fig, ax = plt.subplots(figsize=(7.5, 2.8))

    bar_y = 0.35
    bar_h = 0.15
    ax.add_patch(mpatches.Rectangle(
        (0, bar_y), train_end, bar_h,
        facecolor=SPLIT_COLOURS["train"], edgecolor="black", linewidth=0.6,
    ))
    ax.add_patch(mpatches.Rectangle(
        (train_end, bar_y), val_end - train_end, bar_h,
        facecolor=SPLIT_COLOURS["val"], edgecolor="black", linewidth=0.6,
    ))
    ax.add_patch(mpatches.Rectangle(
        (val_end, bar_y), n - val_end, bar_h,
        facecolor=SPLIT_COLOURS["test"], edgecolor="black", linewidth=0.6,
    ))
    ax.text(train_end / 2, bar_y + bar_h / 2, f"train\n({train_end:,} h)",
            ha="center", va="center", color="black", fontsize=8)
    ax.text((train_end + val_end) / 2, bar_y + bar_h / 2,
            f"validation\n({val_end - train_end:,} h)",
            ha="center", va="center", color="white", fontsize=8)
    ax.text((val_end + n) / 2, bar_y + bar_h / 2,
            f"test\n({n - val_end:,} h)",
            ha="center", va="center", color="white", fontsize=8)

    # Sliding-window schematic above the bar
    win_y = 0.72
    win_h = 0.12
    win_start = int(train_end * 0.35)
    ax.add_patch(mpatches.Rectangle(
        (win_start, win_y), INPUT_LEN, win_h,
        facecolor="#a6cee3", edgecolor="black", linewidth=0.6,
    ))
    ax.add_patch(mpatches.Rectangle(
        (win_start + INPUT_LEN, win_y), OUTPUT_LEN, win_h,
        facecolor="#fb9a99", edgecolor="black", linewidth=0.6,
    ))
    ax.annotate(
        f"input window ({INPUT_LEN} h)",
        xy=(win_start + INPUT_LEN / 2, win_y + win_h),
        xytext=(win_start + INPUT_LEN / 2, win_y + win_h + 0.06),
        ha="center", va="bottom", fontsize=8,
    )
    ax.annotate(
        f"forecast horizon ({OUTPUT_LEN} h)",
        xy=(win_start + INPUT_LEN + OUTPUT_LEN / 2, win_y + win_h),
        xytext=(win_start + INPUT_LEN + OUTPUT_LEN / 2, win_y + win_h + 0.06),
        ha="center", va="bottom", fontsize=8,
    )

    ax.annotate(
        "StandardScaler fit on training segment only",
        xy=(train_end / 2, bar_y),
        xytext=(train_end / 2, bar_y - 0.14),
        ha="center", va="top", fontsize=8, style="italic",
    )

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Hour index")
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("Chronological split and 96 h → 24 h sliding-window construction",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig_02_train_test_split")


def fig_03_pipeline_architecture() -> None:
    print("fig_03_pipeline_architecture")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, colour="#eeeeee"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=colour, edgecolor="black", linewidth=0.8,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8)

    def arrow(x1, y1, x2, y2, style="-"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", linewidth=0.8, color="black", linestyle=style),
        )

    main_row_y = 4.2
    main_h = 0.7
    widths = [1.4, 1.1, 1.3, 1.4, 1.3]
    labels = [
        "ETTh1 ingest &\npre-processing",
        "60/20/20\nchronological split",
        "Train-only\nStandardScaler",
        "Sliding-window\n(96 h → 24 h)",
        "Model trainers\nLR, MLP, LSTM, TFT",
    ]
    x = 0.2
    xs = []
    for w, lab in zip(widths, labels):
        box(x, main_row_y, w, main_h, lab, colour="#e8f1fb")
        xs.append((x, w))
        x += w + 0.25
    for (x0, w0), (x1, _) in zip(xs[:-1], xs[1:]):
        arrow(x0 + w0, main_row_y + main_h / 2, x1, main_row_y + main_h / 2)

    # Checkpoints + evaluation row
    row2_y = 2.7
    box(xs[-1][0], row2_y, widths[-1], main_h, "Checkpoints\n(.pt / .ckpt)", colour="#fdf2d9")
    arrow(xs[-1][0] + widths[-1] / 2, main_row_y, xs[-1][0] + widths[-1] / 2, row2_y + main_h)

    eval_x = xs[-1][0] - 2.0
    box(eval_x, row2_y, 1.8, main_h,
        "Evaluation\nper-seed / per-horizon /\nmoving-block bootstrap",
        colour="#fdf2d9")
    arrow(xs[-1][0], row2_y + main_h / 2, eval_x + 1.8, row2_y + main_h / 2)

    xai_x = eval_x - 2.3
    box(xai_x, row2_y, 2.0, main_h,
        "Interpretability\ncommon occlusion · AOPC\nSHAP · TFT VSN",
        colour="#fdf2d9")
    arrow(eval_x, row2_y + main_h / 2, xai_x + 2.0, row2_y + main_h / 2)

    # Reports row
    row3_y = 1.1
    box(3.2, row3_y, 3.6, main_h,
        "Report artifacts\nCSV · bootstrap tables · figures\ntraining/runtime traces",
        colour="#eef7e8")
    arrow(eval_x + 0.9, row2_y, 5.0, row3_y + main_h)
    arrow(xai_x + 1.0, row2_y, 4.0, row3_y + main_h, style="--")

    ax.text(5.0, 5.6,
            "Fair-core evaluation pipeline (§ 5 Solution chapter)",
            ha="center", va="center", fontsize=10)

    save(fig, "fig_03_pipeline_architecture")


def fig_04_model_performance_comparison() -> None:
    print("fig_04_model_performance_comparison")
    src = require(RESULTS / "per_seed_metrics.csv")
    df = pd.read_csv(src)

    rmse = _per_model_seed_stats(df, "rmse_original")
    mae = _per_model_seed_stats(df, "mae_original")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    x = np.arange(len(MODEL_ORDER))
    colours = [MODEL_COLOURS[m] for m in MODEL_ORDER]

    ax1.bar(x, rmse["mean"].values, yerr=rmse["std"].values,
            color=colours, capsize=3, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODEL_ORDER)
    ax1.set_ylabel("RMSE (original scale)")
    ax1.set_title("Test-set RMSE — primary metric")
    for xi, (m, s) in enumerate(zip(rmse["mean"].values, rmse["std"].values)):
        ax1.text(xi, m + s + 0.05, f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    ax2.bar(x, mae["mean"].values, yerr=mae["std"].values,
            color=colours, capsize=3, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(MODEL_ORDER)
    ax2.set_ylabel("MAE (original scale)")
    ax2.set_title("Test-set MAE — secondary metric")
    for xi, (m, s) in enumerate(zip(mae["mean"].values, mae["std"].values)):
        ax2.text(xi, m + s + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Cross-model predictive performance on ETTh1 test set — mean ±1 seed std",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig_04_model_performance_comparison")


def fig_05_per_horizon_error() -> None:
    print("fig_05_per_horizon_error")
    src = require(RESULTS / "per_horizon_metrics.csv")
    df = pd.read_csv(src)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values("horizon")
        ax.plot(sub["horizon"], sub["mse"], label=model,
                color=MODEL_COLOURS[model], marker="o", markersize=3)
        if (sub["seed_count"] > 1).any():
            lo = sub["mse"] - sub["seed_std_mse"]
            hi = sub["mse"] + sub["seed_std_mse"]
            ax.fill_between(sub["horizon"], lo, hi,
                            color=MODEL_COLOURS[model], alpha=0.15, linewidth=0)

    ax.set_xlabel(f"Forecast horizon step h (configured output length = {OUTPUT_LEN})")
    ax.set_ylabel("MSE (original scale)")
    ax.set_title(
        "Per-horizon test MSE under the configured 96 h → 24 h task",
        fontsize=10,
    )
    ax.legend(loc="best", frameon=False)
    ax.set_xticks(np.arange(1, OUTPUT_LEN + 1, 2))
    fig.tight_layout()
    save(fig, "fig_05_per_horizon_error")


def fig_06_faithfulness_aopc_by_k() -> None:
    print("fig_06_faithfulness_aopc_by_k")
    src = require(RESULTS / "faithfulness_aopc_per_k.csv")
    df = pd.read_csv(src)

    agg = (df.groupby(["model", "k"])["gap"]
             .agg(["mean", "std", "count"])
             .reset_index())
    agg["std"] = agg["std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for model in MODEL_ORDER:
        sub = agg[agg["model"] == model].sort_values("k")
        ax.plot(sub["k"], sub["mean"], label=model,
                color=MODEL_COLOURS[model], marker="o", markersize=4)
        ax.fill_between(sub["k"], sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                        color=MODEL_COLOURS[model], alpha=0.15, linewidth=0)

    ax.set_xlabel("Number of masked variables k")
    ax.set_ylabel("AOPC gap (top-masked MSE − bottom-masked MSE)")
    ax.set_title(
        "Faithfulness decomposition by k — project-specific AOPC definition",
        fontsize=10,
    )
    ax.legend(loc="best", frameon=False)
    ax.set_xticks(sorted(agg["k"].unique()))
    fig.tight_layout()
    save(fig, "fig_06_faithfulness_aopc_by_k")


def fig_07_xai_agreement_heatmap() -> None:
    print("fig_07_xai_agreement_heatmap")
    src = require(RESULTS / "xai_agreement_occlusion.csv")
    df = pd.read_csv(src)

    matrix = np.full((len(MODEL_ORDER), len(MODEL_ORDER)), np.nan)
    idx = {m: i for i, m in enumerate(MODEL_ORDER)}
    for _, row in df.iterrows():
        a, b = row["model_pair"].split("_vs_")
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            matrix[i, j] = row["spearman_corr"]
            matrix[j, i] = row["spearman_corr"]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(matrix, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_yticklabels(MODEL_ORDER)
    for i in range(len(MODEL_ORDER)):
        for j in range(len(MODEL_ORDER)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="black", fontsize=10)
            else:
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center",
                        color="black" if abs(matrix[i, j]) < 0.6 else "white",
                        fontsize=9)
    ax.set_title(
        "Cross-model rank agreement (Spearman ρ) on occlusion importances",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman ρ", fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    save(fig, "fig_07_xai_agreement_heatmap")


def fig_08_performance_interpretability_tradeoff() -> None:
    print("fig_08_performance_interpretability_tradeoff")
    tradeoff_src = require(RESULTS / "trade_off_data.csv")
    per_seed_src = require(RESULTS / "per_seed_metrics.csv")
    tradeoff = pd.read_csv(tradeoff_src)
    per_seed = pd.read_csv(per_seed_src)

    mse_stats = _per_model_seed_stats(per_seed, "mse_original")

    pareto = _pareto_frontier(tradeoff)
    print(f"  Pareto frontier (computed): {sorted(pareto)}")

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    for _, row in tradeoff.iterrows():
        model = row["model"]
        x = row["mse_mean"]
        y = row["aopc_mean"]
        xerr = float(mse_stats.loc[model, "std"])
        yerr = float(row["aopc_std_across_seeds"])
        on_frontier = model in pareto
        ax.errorbar(
            x, y, xerr=xerr, yerr=yerr,
            fmt="o", markersize=10 if on_frontier else 7,
            color=MODEL_COLOURS[model],
            ecolor=MODEL_COLOURS[model], elinewidth=1.0, capsize=3,
            markeredgecolor="black" if on_frontier else "none",
            markeredgewidth=1.2 if on_frontier else 0,
            label=f"{model}{' (Pareto)' if on_frontier else ''}",
        )
        ax.annotate(model, xy=(x, y), xytext=(8, 4),
                    textcoords="offset points", fontsize=9)

    ax.set_xlabel("Mean test MSE (lower → better accuracy)")
    ax.set_ylabel("Mean AOPC (higher → more faithful, per project definition)")
    ax.set_title(
        "Accuracy vs. occlusion-based faithfulness — present configuration",
        fontsize=10,
    )
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    save(fig, "fig_08_performance_interpretability_tradeoff")


def fig_09_model_complexity() -> None:
    print("fig_09_model_complexity")
    src = require(RESULTS / "complexity_metrics.csv")
    df = pd.read_csv(src).set_index("model").reindex(MODEL_ORDER)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    y = np.arange(len(MODEL_ORDER))
    colours = [MODEL_COLOURS[m] for m in MODEL_ORDER]

    ax1.barh(y, df["n_params"].values, color=colours, edgecolor="black", linewidth=0.5)
    ax1.set_xscale("log")
    ax1.set_yticks(y)
    ax1.set_yticklabels([
        f"{m}\n({df.loc[m, 'architectural_category']})" for m in MODEL_ORDER
    ])
    ax1.invert_yaxis()
    ax1.set_xlabel("Parameter count (log scale)")
    ax1.set_title("Model size")
    for yi, m in enumerate(MODEL_ORDER):
        ax1.text(df.loc[m, "n_params"], yi, f"  {int(df.loc[m, 'n_params']):,}",
                 va="center", fontsize=8)

    ax2.barh(y, df["wall_clock_seconds"].values, color=colours, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(MODEL_ORDER)
    ax2.invert_yaxis()
    ax2.set_xlabel("Training wall-clock (s, single-machine run)")
    ax2.set_title("Observed training time")
    for yi, m in enumerate(MODEL_ORDER):
        ax2.text(df.loc[m, "wall_clock_seconds"], yi,
                 f"  {df.loc[m, 'wall_clock_seconds']:.1f}s",
                 va="center", fontsize=8)

    fig.suptitle(
        "Model complexity overview (hardware-dependent wall-clock; see README)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig_09_model_complexity")


FIGURES: dict[int, Callable[[], None]] = {
    1: fig_01_dataset_overview,
    2: fig_02_train_test_split,
    3: fig_03_pipeline_architecture,
    4: fig_04_model_performance_comparison,
    5: fig_05_per_horizon_error,
    6: fig_06_faithfulness_aopc_by_k,
    7: fig_07_xai_agreement_heatmap,
    8: fig_08_performance_interpretability_tradeoff,
    9: fig_09_model_complexity,
}


def list_figures() -> None:
    for num, fn in sorted(FIGURES.items()):
        print(f"  {num}  {fn.__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Regenerate every must-have figure.")
    group.add_argument("--figure", type=int, help="Generate a single figure by number (1..9).")
    group.add_argument("--list", action="store_true", help="List all figures and exit.")
    args = parser.parse_args()

    if args.list:
        list_figures()
        return

    apply_style()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {FIG_OUT}")

    if args.all:
        for num in sorted(FIGURES):
            FIGURES[num]()
    else:
        if args.figure not in FIGURES:
            raise SystemExit(f"ERROR: unknown figure number {args.figure}; see --list")
        FIGURES[args.figure]()
    print("done.")


if __name__ == "__main__":
    main()
