"""Closure Plan v6.1 B4 — prose auto-sync.

Reads authoritative CSVs + the VALIDATION_LOG markdown table, writes Markdown
table fragments under ``docs/_generated/``, and inlines them into the thesis
prose between ``<!-- @begin-include ... -->`` / ``<!-- @end-include -->`` markers
in ``docs/solution.md`` and ``docs/appendix.md``.

Source-of-truth precedence (highest first):
  1. ``results/bachelor_safe_v2/`` — populated after B5 fair-core v2 rerun
  2. ``archive/pre-v2-2026-04-22/results/`` — v1 baseline fallback so B4 is
     runnable before B5

Behaviour is idempotent: reruns must produce byte-identical output given the
same inputs. The pre-commit hook calls ``regen_tables.py`` and blocks the
commit if any fragment / marker-inlined block drifts from disk.

Generated fragments:
  overall_metrics_table.md    — solution.md § 3.8.1
  bootstrap_headline.md       — solution.md § 3.8.2 (MSE, 6 rows)
  per_seed_table.md           — appendix.md § F.1
  bootstrap_full.md           — appendix.md § F.2 (18 rows)
  complexity_table.md         — appendix.md § F.3
  pipeline_status_table.md    — appendix.md § C (parsed from VALIDATION_LOG.md)
  status_summary.md           — status-aware prose (used inline in solution.md
                                § 3.6 / § 3.8.4 / appendix.md § G)

Usage:
  python scripts/regen_tables.py                       # regenerate + inline
  python scripts/regen_tables.py --verify              # fail if diff detected
  python scripts/regen_tables.py --results-dir path    # override source
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
GENERATED_DIR = DOCS_DIR / "_generated"
V2_RESULTS_DIR = REPO_ROOT / "results" / "bachelor_safe_v2"
V1_RESULTS_DIR = REPO_ROOT / "archive" / "pre-v2-2026-04-22" / "results"
VALIDATION_LOG = REPO_ROOT / "archive" / "2026-05-06-pre-submission-cleanup" / "docs" / "VALIDATION_LOG.md"
SOLUTION_MD = DOCS_DIR / "solution.md"
APPENDIX_MD = DOCS_DIR / "appendix.md"

MODELS_ORDERED = ("LR", "MLP", "LSTM", "TFT")
PAIRS_ORDERED = (
    ("LR", "MLP"),
    ("LR", "LSTM"),
    ("LR", "TFT"),
    ("MLP", "LSTM"),
    ("MLP", "TFT"),
    ("LSTM", "TFT"),
)


# ---------------------------------------------------------------------------
# source resolution
# ---------------------------------------------------------------------------


def resolve_results_dir(override: str | None) -> tuple[Path, str]:
    """Return (path, provenance_label). Prefer v2 → fall back to v1 archive."""
    if override:
        path = Path(override).resolve()
        if not path.exists():
            sys.exit(f"regen_tables: --results-dir {path} does not exist")
        return path, f"override:{path.relative_to(REPO_ROOT)}"
    if (V2_RESULTS_DIR / "per_seed_metrics.csv").exists():
        return V2_RESULTS_DIR, "v2"
    if (V1_RESULTS_DIR / "per_seed_metrics.csv").exists():
        return V1_RESULTS_DIR, "v1"
    sys.exit(
        "regen_tables: neither results/bachelor_safe_v2/per_seed_metrics.csv "
        "nor archive/pre-v2-2026-04-22/results/per_seed_metrics.csv found"
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fmt_mean_std(mean: float, std: float, decimals: int = 2) -> str:
    if pd.isna(std) or std == 0.0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _param_counts() -> dict[str, int]:
    """Hardcoded from CLAUDE.md § 10 / smoke_checkpoint.py assertions."""
    return {"LR": 113_064, "MLP": 124_328, "LSTM": 29_608, "TFT": 18_261}


def _best_epochs(summary_csv: Path | None, tft_summary_csv: Path | None) -> dict[str, str]:
    out: dict[str, str] = {"LR": "— (closed-form)"}
    if summary_csv and summary_csv.exists():
        df = pd.read_csv(summary_csv)
        for model in ("MLP", "LSTM"):
            row = df[df["model"] == model]
            if not row.empty:
                out[model] = str(row["best_epochs"].iloc[0]).replace(",", " / ")
    if tft_summary_csv and tft_summary_csv.exists():
        tft_df = pd.read_csv(tft_summary_csv)
        if not tft_df.empty:
            out["TFT"] = str(tft_df["best_epochs"].iloc[0]).replace(",", " / ")
    return out


# ---------------------------------------------------------------------------
# fragment builders
# ---------------------------------------------------------------------------


def build_overall_metrics_table(per_seed_csv: Path, summary_csv: Path, tft_summary_csv: Path) -> str:
    per_seed = pd.read_csv(per_seed_csv)
    best = _best_epochs(summary_csv, tft_summary_csv)
    params = _param_counts()
    lines = [
        "| Model | Parameter count | MSE (mean ± std) | MAE (mean ± std) | RMSE (mean ± std) | Best epochs (per seed) |",
        "|-------|-----------------|-------------------|-------------------|---------------------|--------------------------|",
    ]
    for model in MODELS_ORDERED:
        rows = per_seed[per_seed["model"] == model]
        if rows.empty:
            continue
        mse_mean = rows["mse_original"].mean()
        mse_std = rows["mse_original"].std(ddof=1) if len(rows) > 1 else 0.0
        mae_mean = rows["mae_original"].mean()
        mae_std = rows["mae_original"].std(ddof=1) if len(rows) > 1 else 0.0
        rmse_mean = rows["rmse_original"].mean()
        rmse_std = rows["rmse_original"].std(ddof=1) if len(rows) > 1 else 0.0
        lines.append(
            f"| {model}{' ' * (5 - len(model))} | {params[model]:>7,}".replace(",", " ")
            + f"      | {_fmt_mean_std(mse_mean, mse_std)} "
            + f"| {_fmt_mean_std(mae_mean, mae_std)} "
            + f"| {_fmt_mean_std(rmse_mean, rmse_std)} "
            + f"| {best.get(model, '—')} |"
        )
    return "\n".join(lines) + "\n"


def build_per_seed_table(per_seed_csv: Path, summary_csv: Path, tft_metrics_csv: Path) -> str:
    per_seed = pd.read_csv(per_seed_csv)
    best_epoch_map: dict[tuple[str, str], int] = {}
    if summary_csv.exists():
        sdf = pd.read_csv(summary_csv)
        for _, r in sdf.iterrows():
            if r["model"] in ("MLP", "LSTM") and r["seeds"] != "deterministic":
                seeds = [s.strip() for s in str(r["seeds"]).split(",")]
                bests = [e.strip() for e in str(r["best_epochs"]).split(",")]
                for s, e in zip(seeds, bests, strict=False):
                    best_epoch_map[(r["model"], s)] = int(e)
    if tft_metrics_csv.exists():
        tft = pd.read_csv(tft_metrics_csv)
        for _, r in tft.iterrows():
            best_epoch_map[("TFT", str(int(r["seed"])))] = int(r["best_epoch"])

    lines = [
        "| Model | Seed | MSE       | MAE       | RMSE      | Best epoch |",
        "|-------|------|-----------|-----------|-----------|------------|",
    ]
    for model in MODELS_ORDERED:
        sub = per_seed[per_seed["model"] == model]
        for _, row in sub.iterrows():
            seed_label = "—" if str(row["seed"]) == "deterministic" else str(row["seed"])
            best = "— (closed-form)" if model == "LR" else str(best_epoch_map.get((model, str(row["seed"])), "—"))
            lines.append(
                f"| {model:<5} | {seed_label:<4} | {row['mse_original']:9.6f} | "
                f"{row['mae_original']:9.6f} | {row['rmse_original']:9.6f} | {best} |"
            )
    return "\n".join(lines) + "\n"


def build_bootstrap_headline(bootstrap_csv: Path) -> str:
    df = pd.read_csv(bootstrap_csv)
    mse = df[df["metric"] == "MSE"]
    lines = [
        "| Pair (A vs B) | Mean diff (A − B) | 95 % CI         |",
        "|---------------|--------------------|-----------------|",
    ]
    for a, b in PAIRS_ORDERED:
        row = mse[mse["model_pair"] == f"{a}_vs_{b}"]
        if row.empty:
            continue
        mean = row["mean_diff"].iloc[0]
        lo = row["ci_low"].iloc[0]
        hi = row["ci_high"].iloc[0]
        pair_label = f"{a} vs {b}"
        lines.append(f"| {pair_label:<13} | {mean:7.2f}            | [{lo:6.2f}, {hi:6.2f}] |")
    return "\n".join(lines) + "\n"


def build_bootstrap_full(bootstrap_csv: Path) -> str:
    df = pd.read_csv(bootstrap_csv)
    has_block = "block_length" in df.columns
    header = "| Pair (A vs B) | Metric | Mean diff (A − B) | 95 % CI low | 95 % CI high | Bootstrap N | Bootstrap seed |"
    divider = "|---------------|--------|--------------------|--------------|---------------|--------------|------------------|"
    if has_block:
        header += " Block length |"
        divider += "---------------|"
    header += " n_test_windows |"
    divider += "-------------------|"
    lines = [header, divider]
    for a, b in PAIRS_ORDERED:
        for metric in ("MSE", "MAE", "RMSE"):
            row = df[(df["model_pair"] == f"{a}_vs_{b}") & (df["metric"] == metric)]
            if row.empty:
                continue
            r = row.iloc[0]
            cells = [
                f"{a} vs {b}",
                metric,
                f"{r['mean_diff']:.6f}",
                f"{r['ci_low']:.6f}",
                f"{r['ci_high']:.6f}",
                f"{int(r['bootstrap_n']):>6,}".replace(",", " "),
                f"{int(r['bootstrap_seed'])}",
            ]
            if has_block:
                cells.append(f"{int(r['block_length'])}")
            cells.append(f"{int(r['n_test_windows']):>5,}".replace(",", " "))
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_complexity_table(complexity_csv: Path) -> str:
    df = pd.read_csv(complexity_csv)
    lines = [
        "| Model | n_params | Architectural category | Wall-clock seconds | Hardware note |",
        "|-------|----------|------------------------|---------------------|----------------|",
    ]
    for model in MODELS_ORDERED:
        row = df[df["model"] == model]
        if row.empty:
            continue
        r = row.iloc[0]
        wc = r.get("wall_clock_seconds", "")
        wc_str = "< 1 s" if model == "LR" and (wc in ("", 0, 0.0) or pd.isna(wc)) else (
            f"{float(wc):.1f} s" if isinstance(wc, (int, float, np.floating)) and not pd.isna(wc) else str(wc)
        )
        nparams = int(r["n_params"])
        lines.append(
            f"| {model:<5} | {nparams:>7,}".replace(",", " ")
            + f" | {r['architectural_category']:<22} "
            + f"| {wc_str:<19} "
            + f"| {r.get('hardware_note', '')} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# VALIDATION_LOG parsing
# ---------------------------------------------------------------------------


_STATUS_RE = re.compile(
    r"(?:\*\*)?"
    r"\b(?P<word>VALIDATED|PENDING|PARTIAL|BLOCKED|SUPERSEDED|RECORDED)\b"
    r"(?P<mod>\s*\([^)]+\))?"
    r"(?:\*\*)?"
)
_SCRIPT_RE = re.compile(r"`([^`]+\.py)`")
# VALIDATION_LOG uses plain (un-backticked) paths in the Scripts table; also accept those:
_LOG_SCRIPT_RE = re.compile(r"(src/[a-zA-Z0-9_/]+\.py)")


def parse_validation_log(log_path: Path) -> dict[str, str]:
    """Return {script_path: status} mapping from the VALIDATION_LOG Scripts table."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8")
    # Only consider the "## Scripts" section
    m = re.search(r"^##\s+Scripts\s*$", text, re.MULTILINE)
    if not m:
        return {}
    scripts_section = text[m.end():]
    out: dict[str, str] = {}
    for line in scripts_section.splitlines():
        if not line.startswith("|") or line.startswith("|---") or "Script |" in line:
            continue
        cells = line.split("|")[1:-1]
        if len(cells) < 5:
            continue
        script_cell = cells[0]
        status_cell = cells[-1]
        # skip rows rendered as strikethrough for legacy-superseded entries
        if script_cell.strip().startswith("~~"):
            continue
        m_script = _SCRIPT_RE.search(script_cell) or _LOG_SCRIPT_RE.search(script_cell)
        if not m_script:
            continue
        script_path = m_script.group(1)
        m_status = _STATUS_RE.search(status_cell)
        if not m_status:
            continue
        word = m_status.group("word")
        modifier = (m_status.group("mod") or "").strip()
        status = f"{word} {modifier}".strip() if modifier else word
        # Keep the last (most recent) status for a given script path
        out[script_path] = status
    return out


def build_pipeline_status_table(log_path: Path) -> str:
    status_map = parse_validation_log(log_path)
    # Conceptual rows from solution.md § C; status derived from log where available.
    rows: list[tuple[str, str, str]] = [
        ("Data processing",
         "`src/training/multi_seed.py`; `src/training/tft_fair_5seed.py`",
         "scaled tensors in memory"),
        ("LR training path",
         "`src/training/multi_seed.py` (LR pathway)",
         "`results/multi_seed_fair_baseline.csv` (LR row)"),
        ("MLP training path",
         "`src/training/multi_seed.py` (MLP pathway)",
         "`results/multi_seed_fair_baseline.csv` (MLP rows); `checkpoints/mlp_seed*.pt`"),
        ("LSTM training path",
         "`src/training/multi_seed.py` (LSTM pathway)",
         "`results/multi_seed_fair_baseline.csv` (LSTM rows); `checkpoints/lstm_seed*.pt`"),
        ("TFT training path",
         "`src/training/tft_fair_5seed.py`",
         "`results/tft_summary.csv`; `results/tft_metrics.csv`; `results/tft_training_curves.csv`; `results/tft_importance.csv`; `checkpoints/tft_seed*.ckpt`"),
        ("Prediction export",
         "`src/evaluation/export_predictions.py`",
         "`results/preds_*.npy` (shape `(n_test, 24, 7)`)"),
        ("Post-training analysis (per-seed + complexity + bootstrap)",
         "`src/evaluation/post_training_analysis.py`",
         "`results/per_seed_metrics.csv`; `results/complexity_metrics.csv`; `results/bootstrap_intervals.csv` + `bootstrap_block_sensitivity.csv`"),
        ("Explainability — SHAP",
         "`src/explainability/shap_lr.py`; `src/explainability/shap_mlp.py`; `src/explainability/shap_lstm.py`",
         "`results/shap_{lr,mlp,lstm}.csv`"),
        ("Explainability — VSN",
         "`src/training/tft_fair_5seed.py` (TFT VSN extraction)",
         "`results/tft_importance.csv`"),
        ("Faithfulness layer",
         "`src/explainability/faithfulness_test.py`",
         "`results/faithfulness.csv`"),
        ("Cross-model XAI agreement",
         "`src/explainability/cross_model_xai_agreement.py`",
         "`results/xai_agreement.csv`"),
        ("Accuracy ↔ interpretability trade-off plot",
         "`src/explainability/trade_off_plot.py`",
         "`results/trade_off_data.csv`; `results/trade_off_plot.png`"),
        ("Per-horizon disaggregated metrics",
         "`src/evaluation/per_horizon_metrics.py`",
         "`results/per_horizon_metrics.csv`"),
    ]
    lines = [
        "| Conceptual component | File(s) | Status | Output artefact(s) |",
        "|----------------------|---------|--------|---------------------|",
    ]
    for comp, files, artefacts in rows:
        script_statuses = []
        for m_script in _SCRIPT_RE.finditer(files):
            sp = m_script.group(1)
            st = status_map.get(sp, "PENDING")
            script_statuses.append(st)
        if not script_statuses:
            status_label = "—"
        elif len(set(script_statuses)) == 1:
            status_label = f"**{script_statuses[0]}**"
        else:
            status_label = " / ".join(f"**{s}**" for s in script_statuses)
        lines.append(f"| {comp} | {files} | {status_label} | {artefacts} |")
    return "\n".join(lines) + "\n"


def build_status_summary(log_path: Path, provenance: str) -> str:
    status_map = parse_validation_log(log_path)
    core_training = ["src/training/multi_seed.py", "src/training/tft_fair_5seed.py"]
    eval_layer = ["src/evaluation/export_predictions.py", "src/evaluation/post_training_analysis.py"]
    shap_layer = ["src/explainability/shap_lr.py", "src/explainability/shap_mlp.py", "src/explainability/shap_lstm.py"]
    faith = "src/explainability/faithfulness_test.py"

    def all_validated(paths: list[str]) -> bool:
        return all(status_map.get(p) == "VALIDATED" for p in paths)

    training_ok = all_validated(core_training)
    eval_ok = all_validated(eval_layer)
    shap_ok = all_validated(shap_layer)
    faith_status = status_map.get(faith, "PENDING")

    provenance_note = {
        "v1": "v1 baseline artefacts (pre-B5 rerun; source: `archive/pre-v2-2026-04-22/results/`)",
        "v2": "v2 fair-core rerun artefacts (`results/bachelor_safe_v2/`)",
    }.get(provenance, f"source: {provenance}")

    parts = [f"Tables and figures in this section are auto-generated from {provenance_note}."]
    if training_ok and eval_ok:
        parts.append(
            "The training and post-training analysis layers "
            "(`multi_seed.py`, `tft_fair_5seed.py`, `export_predictions.py`, `post_training_analysis.py`) "
            "are **VALIDATED**."
        )
    else:
        parts.append(
            "Training + post-training layers status: PARTIAL."
        )
    if shap_ok and faith_status == "VALIDATED":
        parts.append(
            "The explainability layer (SHAP for LR/MLP/LSTM + VSN for TFT + faithfulness test) is **VALIDATED**."
        )
    elif shap_ok:
        parts.append(
            f"SHAP scripts are **VALIDATED**; `faithfulness_test.py` is currently **{faith_status}**."
        )
    else:
        missing = [p.split("/")[-1] for p in shap_layer if status_map.get(p) != "VALIDATED"]
        parts.append(
            "Explainability layer is partially validated; pending: " + ", ".join(missing) + "."
        )
    return " ".join(parts) + "\n"


# ---------------------------------------------------------------------------
# prose inlining
# ---------------------------------------------------------------------------


_MARKER_RE = re.compile(
    r"(<!--\s*@begin-include\s+(?P<path>\S+)\s*-->)"
    r"(?P<body>.*?)"
    r"(<!--\s*@end-include\s*-->)",
    re.DOTALL,
)


def compute_inlined(md_path: Path) -> tuple[str, str]:
    """Return (original_text, inlined_text). Pure; does not write."""
    if not md_path.exists():
        return "", ""
    original = md_path.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        fragment_rel = match.group("path")
        fragment_path = (DOCS_DIR / fragment_rel).resolve()
        if not fragment_path.exists():
            sys.exit(f"regen_tables: referenced fragment {fragment_rel} missing")
        fragment_body = fragment_path.read_text(encoding="utf-8").rstrip() + "\n"
        begin_tag = f"<!-- @begin-include {fragment_rel} -->"
        end_tag = "<!-- @end-include -->"
        return f"{begin_tag}\n{fragment_body}{end_tag}"

    return original, _MARKER_RE.sub(replace, original)


def inline_markers(md_path: Path) -> bool:
    """Rewrite body between markers with fragment contents. Return True if file changed."""
    original, updated = compute_inlined(md_path)
    if updated != original and md_path.exists():
        md_path.write_text(updated, encoding="utf-8")
        return True
    return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_regen(results_dir: Path, provenance: str, verify: bool = False) -> int:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_csv = results_dir / "per_seed_metrics.csv"
    summary_csv = results_dir / "multi_seed_fair_baseline.csv"
    tft_summary_csv = results_dir / "tft_summary.csv"
    tft_metrics_csv = results_dir / "tft_metrics.csv"
    bootstrap_csv = results_dir / "bootstrap_intervals.csv"
    complexity_csv = results_dir / "complexity_metrics.csv"

    fragments: dict[str, str] = {}
    if per_seed_csv.exists():
        fragments["overall_metrics_table.md"] = build_overall_metrics_table(per_seed_csv, summary_csv, tft_summary_csv)
        fragments["per_seed_table.md"] = build_per_seed_table(per_seed_csv, summary_csv, tft_metrics_csv)
    if bootstrap_csv.exists():
        fragments["bootstrap_headline.md"] = build_bootstrap_headline(bootstrap_csv)
        fragments["bootstrap_full.md"] = build_bootstrap_full(bootstrap_csv)
    if complexity_csv.exists():
        fragments["complexity_table.md"] = build_complexity_table(complexity_csv)
    fragments["pipeline_status_table.md"] = build_pipeline_status_table(VALIDATION_LOG)
    fragments["status_summary.md"] = build_status_summary(VALIDATION_LOG, provenance)

    drift = 0
    for name, body in fragments.items():
        target = GENERATED_DIR / name
        if target.exists() and target.read_text(encoding="utf-8") == body:
            continue
        if verify:
            print(f"DRIFT: docs/_generated/{name} differs from regenerated content", file=sys.stderr)
            drift += 1
        else:
            target.write_text(body, encoding="utf-8")

    for md in (SOLUTION_MD, APPENDIX_MD):
        if verify:
            original, updated = compute_inlined(md)
            if original != updated:
                print(f"DRIFT: {md.relative_to(REPO_ROOT)} marker block out of sync", file=sys.stderr)
                drift += 1
        else:
            inline_markers(md)

    if verify and drift:
        return 1
    if verify:
        print(f"regen_tables: verify OK (source={provenance}, fragments={len(fragments)})")
    else:
        print(
            f"regen_tables: wrote {len(fragments)} fragments to docs/_generated/ "
            f"(source={provenance}) + inlined solution.md + appendix.md markers"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B4 prose auto-sync for Closure Plan v6.1.")
    parser.add_argument("--results-dir", default=None,
                        help="Override source dir (default: prefer results/bachelor_safe_v2/, fall back to v1 archive).")
    parser.add_argument("--verify", action="store_true",
                        help="Do not write; exit 1 if any fragment or marker-inlined block would change.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results_dir, provenance = resolve_results_dir(args.results_dir)
    return run_regen(results_dir, provenance, verify=args.verify)


if __name__ == "__main__":
    sys.exit(main())
