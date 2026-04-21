"""Accuracy ↔ interpretability trade-off plot (S-10 strong-package addition; S-15.5 patch 2026-04-21).

Purpose
-------
Quantifies and visualises the RQ3 trade-off claim by plotting overall MSE
(predictive accuracy axis) against per-model **faithfulness gap** (strength-
based explanation-quality axis) for the four core models.

Why this artefact
-----------------
Under the binary decision rule used pre-S-15.5 (`top > bottom` per (model,k)
without a minimum separation threshold), all four models scored 1.000 →
y-axis collapsed → RQ3 trade-off not discriminable. S-15.5 replaces the
binary pass rate with a continuous strength-based gap metric derived from
the same `results/faithfulness.csv`; no upstream rerun of
`faithfulness_test.py` is required.

Faithfulness-gap metric definition (S-15.5 PRIMARY)
---------------------------------------------------
For each (model, k) where k ∈ {1,2,3}, define
    gap(model, k) = top_mse_increase(model, k) − bottom_mse_increase(model, k)
Then
    faithfulness_gap_mean(model) = mean over k of gap(model, k)
measures how much more the top-k masked variables damage the model than the
bottom-k masked variables on average. Higher = stronger behavioural
separation between the top-ranked and bottom-ranked variables under this
ranking. Values in `results/faithfulness.csv` are in scaled-MSE space;
scale does not affect cross-model ordering.

Column schema (written to `results/trade_off_data.csv`):
    faithfulness_gap_mean   — PRIMARY metric (plot y-axis, prose anchor)
    faithfulness_gap_std    — dispersion across k ∈ {1,2,3}
                              (NOT a statistical uncertainty interval;
                              n=3 by design; label as dispersion, not CI)
    faithfulness_gap_min    — conservative lower bound, min over k of gap
    faithfulness_pass_rate  — SECONDARY diagnostic (binary passes / total k)
    faithfulness_score      — backward-compat alias = faithfulness_pass_rate
                              (retained for existing docs/log references;
                              candidate for removal at S-18)

Construct-validity caveat (CLAUDE.md § 13)
------------------------------------------
SHAP (LR / MLP / LSTM) and VSN (TFT) are not equivalent measurement
instruments. The faithfulness-gap metric is comparable across models
because it measures behavioural consistency under perturbation of the
model's own ranking, not the explanation's internal representation. The
post-hoc-vs-architecture-native distinction is recorded in
`docs/discussion.md` § 4.4.

Measurement-criterion dependence caveat (S-15.5)
------------------------------------------------
The underlying `faithfulness_test.py` masking rule is coarse: it compares
a single-top-k masking vs a single-bottom-k masking at each k. A richer
metric (continuous infidelity, ordered-masking curves) would strengthen
RQ3 evidence; this remains future work per `docs/discussion.md` § 4.4.
The gap-mean values reported here are empirically discriminating but
criterion-bounded.

Inputs (frozen artefacts produced upstream)
-------------------------------------------
    results/per_seed_metrics.csv     — per-(model, seed) MSE on original scale
                                       (or fall back to multi_seed_fair_baseline.csv + tft_summary.csv)
    results/faithfulness.csv         — per-(model, k, mask_type) mse_increase

Outputs
-------
    results/trade_off_data.csv          — per-model 5-column schema above
    results/trade_off_plot.png          — 2D scatter, y-axis = faithfulness_gap_mean
    results/faithfulness_gap_by_k.png   — per-model gap curves over k ∈ {1,2,3}

Run
---
    python src/explainability/trade_off_plot.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS_ORDERED = ["LR", "MLP", "LSTM", "TFT"]
MODEL_COLORS = {"LR": "tab:blue", "MLP": "tab:orange", "LSTM": "tab:green", "TFT": "tab:red"}


# ============================================================
# Accuracy axis: per-model overall MSE (original scale, seed-mean)
# ============================================================
def load_accuracy():
    if os.path.exists("results/per_seed_metrics.csv"):
        df = pd.read_csv("results/per_seed_metrics.csv")
        agg = df.groupby("model")["mse_original"].mean().to_dict()
        return {m: agg[m] for m in MODELS_ORDERED if m in agg}
    out = {}
    if os.path.exists("results/multi_seed_fair_baseline.csv"):
        ms = pd.read_csv("results/multi_seed_fair_baseline.csv")
        ms["model_short"] = ms["model"].replace({"Linear Regression": "LR"})
        for m in ["LR", "MLP", "LSTM"]:
            row = ms[ms["model_short"] == m]
            if not row.empty:
                out[m] = float(row["mse_original_mean"].iloc[0])
    if os.path.exists("results/tft_summary.csv"):
        tft = pd.read_csv("results/tft_summary.csv")
        if "mse_original_mean" in tft.columns:
            out["TFT"] = float(tft["mse_original_mean"].iloc[0])
        elif "mse_original" in tft.columns:
            out["TFT"] = float(tft["mse_original"].mean())
    return out


# ============================================================
# Interpretability axis: per-model faithfulness-gap summary (S-15.5 primary)
# ============================================================
def compute_per_k_gaps():
    """Return {model: {k: gap}} from results/faithfulness.csv.

    gap(model, k) = top_mse_increase(model, k) - bottom_mse_increase(model, k)
    Expected to be > 0 under the current validated data (all 12/12 YES).
    """
    if not os.path.exists("results/faithfulness.csv"):
        return {}
    df = pd.read_csv("results/faithfulness.csv")
    per_k = {}
    for model, group in df.groupby("model"):
        top = group[group["mask_type"] == "top"].set_index("k")["mse_increase"]
        bot = group[group["mask_type"] == "bottom"].set_index("k")["mse_increase"]
        common_k = sorted(set(top.index) & set(bot.index))
        if not common_k:
            continue
        per_k[model] = {int(k): float(top[k] - bot[k]) for k in common_k}
    return per_k


def summarise_gaps(per_k):
    """Return {model: {faithfulness_gap_mean, _std, _min, _pass_rate, _score}}.

    Per S-15.5 plan v16 schema:
        faithfulness_gap_mean  — PRIMARY continuous metric
        faithfulness_gap_std   — dispersion across k ∈ {1,2,3} (NOT CI)
        faithfulness_gap_min   — min over k; conservative lower bound
        faithfulness_pass_rate — SECONDARY diagnostic (fraction with gap > 0)
        faithfulness_score     — backward-compat alias = faithfulness_pass_rate
    """
    out = {}
    for model, k_to_gap in per_k.items():
        gaps = np.array(list(k_to_gap.values()), dtype=float)
        if gaps.size == 0:
            continue
        passes = int(np.sum(gaps > 0))
        pass_rate = passes / gaps.size
        out[model] = {
            "faithfulness_gap_mean": float(gaps.mean()),
            "faithfulness_gap_std": float(gaps.std(ddof=1)) if gaps.size > 1 else 0.0,
            "faithfulness_gap_min": float(gaps.min()),
            "faithfulness_pass_rate": pass_rate,
            "faithfulness_score": pass_rate,  # backward-compat alias
        }
    return out


# ============================================================
# Plot 1 — trade-off scatter (MSE × faithfulness_gap_mean)
# ============================================================
def plot_trade_off(data, out_png="results/trade_off_plot.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in data.iterrows():
        m = row["model"]
        ax.scatter(row["mse"], row["faithfulness_gap_mean"],
                   s=160, color=MODEL_COLORS.get(m, "gray"),
                   edgecolor="black", linewidth=0.8, zorder=3, label=m)
        pass_note = f"  (pass {int(round(row['faithfulness_pass_rate'] * 3))}/3)"
        ax.annotate(m + pass_note, xy=(row["mse"], row["faithfulness_gap_mean"]),
                    xytext=(8, 6), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Overall MSE (original scale, seed-mean) — lower is more accurate")
    ax.set_ylabel("Faithfulness gap — mean over k of (top-k − bottom-k) scaled-MSE increase")
    ax.set_title("Accuracy ↔ interpretability trade-off (RQ3)")
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# ============================================================
# Plot 2 — per-model gap curves over k (S-15.5 decomposition artefact)
# ============================================================
def plot_gap_by_k(per_k, out_png="results/faithfulness_gap_by_k.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in MODELS_ORDERED:
        if model not in per_k:
            continue
        k_values = sorted(per_k[model].keys())
        gaps = [per_k[model][k] for k in k_values]
        ax.plot(k_values, gaps, marker="o", linewidth=2,
                color=MODEL_COLORS.get(model, "gray"), label=model)
    ax.set_xlabel("k (number of top / bottom variables masked)")
    ax.set_ylabel("Faithfulness gap = top-k − bottom-k scaled-MSE increase")
    ax.set_title("Per-model faithfulness gap decomposed over k (S-15.5)")
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xticks([1, 2, 3])
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Trade-off plot — accuracy (MSE) × faithfulness gap (S-15.5 strength-based)")
    print("=" * 70)

    print("\n[1/4] Loading accuracy axis (per-model overall MSE)...")
    acc = load_accuracy()
    for m, v in acc.items():
        print(f"  {m:6s} MSE = {v:.4f}")

    print("\n[2/4] Loading faithfulness axis (per-model strength-based gap)...")
    per_k = compute_per_k_gaps()
    faith = summarise_gaps(per_k)
    for m in MODELS_ORDERED:
        if m not in faith:
            continue
        s = faith[m]
        print(f"  {m:6s} gap_mean = {s['faithfulness_gap_mean']:+.4f}   "
              f"gap_min = {s['faithfulness_gap_min']:+.4f}   "
              f"gap_std(across k) = {s['faithfulness_gap_std']:.4f}   "
              f"pass_rate = {s['faithfulness_pass_rate']:.2f}")
    if not faith:
        print("  [warn] faithfulness.csv not found or empty — plot will be x-only")

    rows = []
    for m in MODELS_ORDERED:
        if m not in acc:
            continue
        row = {
            "model": m,
            "mse": acc[m],
            "faithfulness_available": m in faith,
        }
        if m in faith:
            row.update(faith[m])
        else:
            row.update({
                "faithfulness_gap_mean": np.nan,
                "faithfulness_gap_std": np.nan,
                "faithfulness_gap_min": np.nan,
                "faithfulness_pass_rate": np.nan,
                "faithfulness_score": np.nan,
            })
        rows.append(row)

    data = pd.DataFrame(rows)
    # Column order: identity, accuracy, primary gap metrics, secondary + alias, flag
    col_order = [
        "model", "mse",
        "faithfulness_gap_mean", "faithfulness_gap_std", "faithfulness_gap_min",
        "faithfulness_pass_rate", "faithfulness_score",
        "faithfulness_available",
    ]
    data = data[[c for c in col_order if c in data.columns]]
    os.makedirs("results", exist_ok=True)
    data.to_csv("results/trade_off_data.csv", index=False)

    print("\n[3/4] Per-model trade-off table (PRIMARY = faithfulness_gap_mean):")
    print(data.to_string(index=False))

    plottable = data.dropna(subset=["faithfulness_gap_mean"])
    if plottable.empty:
        print("\n[warn] No model has both axes populated; skipping plots.")
    else:
        out_png = plot_trade_off(plottable)
        print(f"\nSaved scatter plot: {out_png}")
        if per_k:
            out_png2 = plot_gap_by_k(per_k)
            print(f"Saved gap-by-k decomposition plot: {out_png2}")
        if len(plottable) < len(MODELS_ORDERED):
            missing = [m for m in MODELS_ORDERED if m not in plottable["model"].tolist()]
            print(f"[note] Plots omit: {missing} (no faithfulness gap available).")

    print("\n[4/4] Reading guide (S-15.5 strength-based semantics):")
    print("-" * 70)
    print("  x = overall MSE (lower better; predictive accuracy, original scale)")
    print("  y = faithfulness_gap_mean (higher better; mean over k={1,2,3} of")
    print("      top-k − bottom-k scaled-MSE increase; stronger behavioural")
    print("      separation under the model's own variable ranking)")
    print("  Annotation '(pass N/3)' = secondary binary diagnostic (how many of")
    print("      k ∈ {1,2,3} satisfied top > bottom).")
    print("  `faithfulness_score` column is a backward-compat alias of")
    print("      `faithfulness_pass_rate` (candidate for removal at S-18).")
    print("  `faithfulness_gap_std` = dispersion across k, NOT a statistical CI.")
    print("")
    print("  Interpretation discipline (CLAUDE.md § 11C + memory rq2_prose_")
    print("  calibration S-15 addendum): under the adopted gap metric, a")
    print("  downward-sloping front would indicate a strong-form trade-off;")
    print("  a dominant single point (low MSE AND high gap) would refute it;")
    print("  non-monotonic scatter (observed here) indicates no strong-form")
    print("  trade-off under the present setup. Criterion-dependence applies:")
    print("  the coarse top-vs-bottom masking rule in faithfulness_test.py")
    print("  limits RQ3 evidence strength; a richer metric is future work.")
    print("-" * 70)
