"""Artifact-level reproducibility verification for the active fair-core v2 pipeline.

Active v2 artifact contract.

Modes:
  --mode core    verify the primary active v2 artifact set from existing outputs.
  --mode full    regenerate bounded auxiliary SHAP outputs, rerun XAI agreement,
                 then verify core + auxiliary artifacts.

Reads the active config (configs/experiments/fair_core_v2.yaml) via
src.common.load_config, then verifies that every artifact listed in
CLAUDE.md § 5 (the artifact = the reproducible pipeline) is present and
schema-valid under the active output roots:

  results/bachelor_safe_v2/
  checkpoints/bachelor_safe_v2/
  reports/figures/main/

Core mode does not retrain models. Full mode writes auxiliary SHAP artifacts and
the regenerated auxiliary agreement file. Both modes write:
  - results/bachelor_safe_v2/reproducibility_verification_report.json
  - results/bachelor_safe_v2/artifact_manifest_sha256.csv

Exit codes:
  0  PASS or PASS_WITH_WARNINGS
  1  FAIL

Examples:
  python scripts/verify_end_to_end_reproducibility.py --mode core
  python scripts/verify_end_to_end_reproducibility.py --mode full
  python scripts/verify_end_to_end_reproducibility.py --mode core --run-and-verify
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Active expected values (CLAUDE.md § 4 + § 10 + configs/base.yaml).
# These are not configurable: they encode the locked v2 contract.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "experiments" / "fair_core_v2.yaml"
ACTIVE_CONFIG_PATH = DEFAULT_CONFIG_PATH  # Backward-compatible name used by older code paths.
EXPECTED_DATA_PATH = "data/ETTh1.csv"
EXPECTED_INPUT_LEN = 96
EXPECTED_OUTPUT_LEN = 24
EXPECTED_SEEDS = [42, 123, 456, 789, 1024]
EXPECTED_RESULTS_DIR = "results/bachelor_safe_v2"
EXPECTED_CHECKPOINTS_DIR = "checkpoints/bachelor_safe_v2"
FIGURES_DIR = REPO_ROOT / "reports" / "figures" / "main"

EXPECTED_ETT_COLUMNS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
EXPECTED_ETT_ROWS = 17420
N_VARIABLES = 7

# CSV schema contract: each artifact maps path → required column subset.
CORE_CSV_SCHEMAS: dict[str, list[str]] = {
    "multi_seed_fair_baseline.csv": ["model", "mse_original_mean", "mae_original_mean"],
    "tft_metrics.csv": ["seed", "mse_original", "mae_original", "checkpoint_path"],
    "tft_summary.csv": ["model", "mse_original_mean", "mae_original_mean", "n_params"],
    "per_seed_metrics.csv": ["model", "seed", "mse_original", "mae_original"],
    "per_horizon_metrics.csv": ["model", "horizon", "mse", "mae"],
    "bootstrap_intervals.csv": [
        "model_pair",
        "metric",
        "mean_diff",
        "ci_low",
        "ci_high",
    ],
    "runtime_seconds.csv": ["model", "seed", "wall_clock_seconds"],
    "training_curves.csv": ["model", "seed", "epoch", "train_loss", "val_loss"],
    "tft_training_curves.csv": ["model", "seed", "epoch", "train_loss", "val_loss"],
    "complexity_metrics.csv": ["model", "n_params", "architectural_category"],
    "bootstrap_block_sensitivity.csv": ["model_pair", "metric", "block_length"],
}

XAI_CSV_SCHEMAS: dict[str, list[str]] = {
    "occlusion_importance.csv": ["model", "seed", "variable", "importance"],
    "faithfulness_aopc.csv": ["model", "seed", "aopc"],
    "faithfulness_aopc_per_k.csv": ["model", "seed", "k", "gap"],
    "xai_agreement_occlusion.csv": ["model_pair", "spearman_corr"],
    "tft_importance.csv": ["seed", "variable", "importance"],
    "trade_off_data.csv": ["model", "mse_mean", "aopc_mean"],
}

SHAP_BASE_REQUIRED_COLUMNS = [
    "model",
    "seed",
    "variable",
    "shap_importance",
    "input_space",
    "explanation_type",
    "active_config",
]

AUXILIARY_SHAP_SCHEMAS: dict[str, list[str]] = {
    "shap_lr.csv": SHAP_BASE_REQUIRED_COLUMNS,
    "shap_mlp.csv": SHAP_BASE_REQUIRED_COLUMNS,
    "shap_lstm.csv": SHAP_BASE_REQUIRED_COLUMNS
    + [
        "is_true_shap",
        "warning",
        "num_background_windows_used",
        "num_eval_windows_used",
    ],
}

AUXILIARY_CSV_SCHEMAS: dict[str, list[str]] = {
    "xai_agreement_shap_vsn.csv": ["model_pair", "spearman_corr", "kendall_tau"],
}

SHAP_CROSS_FILES = [
    "shap_lr_cross.csv",
    "shap_mlp_cross.csv",
    "shap_lstm_cross.csv",
]

FORBIDDEN_TFT_SHAP_FILES = [
    "shap_tft.csv",
    "shap_tft_cross.csv",
]

EXPECTED_MLP_LSTM_SHAP_SEEDS = {"42", "123", "456", "789", "1024", "mean"}
LSTM_FULL_BACKGROUND_WINDOWS = 16
LSTM_FULL_EVAL_WINDOWS = 16
REPORT_FILENAME = "reproducibility_verification_report.json"
MANIFEST_FILENAME = "artifact_manifest_sha256.csv"

FIGURE_STEMS = [
    "fig_01_dataset_overview",
    "fig_02_train_test_split",
    "fig_03_pipeline_architecture",
    "fig_04_model_performance_comparison",
    "fig_05_per_horizon_error",
    "fig_06_faithfulness_aopc_by_k",
    "fig_07_xai_agreement_heatmap",
    "fig_08_performance_interpretability_tradeoff",
    "fig_09_model_complexity",
]

GENERATED_TABLE_FILES = [
    "bootstrap_full.md",
    "bootstrap_headline.md",
    "complexity_table.md",
    "overall_metrics_table.md",
    "per_seed_table.md",
    "pipeline_status_table.md",
    "status_summary.md",
]

CORE_RUN_SCRIPT_PATHS = [
    "src/training/multi_seed.py",
    "src/training/tft_fair_5seed.py",
    "src/evaluation/export_predictions.py",
    "src/evaluation/per_horizon_metrics.py",
    "src/evaluation/post_training_analysis.py",
    "src/explainability/common_importance.py",
    "src/explainability/faithfulness_test.py",
    "src/explainability/cross_model_xai_agreement.py",
    "src/explainability/trade_off_plot.py",
]

TABLE_AND_FIGURE_COMMANDS: list[list[str]] = [
    [sys.executable, "scripts/regen_tables.py"],
    [sys.executable, "scripts/make_figures.py", "--all"],
]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _config_arg(config_path: Path) -> str:
    return _display_path(config_path)


def _core_run_commands(config_arg: str) -> list[list[str]]:
    commands = [[sys.executable, script, "--config", config_arg] for script in CORE_RUN_SCRIPT_PATHS]
    return commands + TABLE_AND_FIGURE_COMMANDS


def _full_shap_commands(config_arg: str) -> list[list[str]]:
    return [
        [sys.executable, "src/explainability/shap_lr.py", "--config", config_arg],
        [sys.executable, "src/explainability/shap_mlp.py", "--config", config_arg],
        [
            sys.executable,
            "src/explainability/shap_lstm.py",
            "--config",
            config_arg,
            "--max-background-windows",
            str(LSTM_FULL_BACKGROUND_WINDOWS),
            "--max-eval-windows",
            str(LSTM_FULL_EVAL_WINDOWS),
        ],
        [sys.executable, "src/explainability/cross_model_xai_agreement.py", "--config", config_arg],
    ]


# ---------------------------------------------------------------------------
# Aggregation containers
# ---------------------------------------------------------------------------
class Report:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.timestamp_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        self.checks: dict[str, dict[str, Any]] = {}
        self.missing_files: list[str] = []
        self.schema_errors: list[str] = []
        self.warnings: list[str] = []
        self.failures: list[str] = []
        self.verified_paths: list[Path] = []
        self.auxiliary_shap_regenerated = False
        self.primary_xai_verified = False
        self.auxiliary_xai_verified = False
        self.checksum_manifest_written = False
        self.number_of_artifacts_checksummed = 0
        self.runtime_seconds = 0.0

    def section(self, name: str) -> dict[str, Any]:
        sect = self.checks.setdefault(name, {"passed": [], "failed": [], "warned": []})
        return sect

    def pass_(self, section: str, item: str) -> None:
        self.section(section)["passed"].append(item)

    def fail(self, section: str, item: str, detail: str) -> None:
        msg = f"{item}: {detail}"
        self.section(section)["failed"].append(msg)
        self.failures.append(f"[{section}] {msg}")

    def warn(self, section: str, item: str, detail: str) -> None:
        msg = f"{item}: {detail}"
        self.section(section)["warned"].append(msg)
        self.warnings.append(f"[{section}] {msg}")

    def status(self) -> str:
        if self.failures:
            return "FAIL"
        if self.warnings:
            return "PASS_WITH_WARNINGS"
        return "PASS"

    def section_has_failures(self, name: str) -> bool:
        return bool(self.checks.get(name, {}).get("failed"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _csv_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        return next(reader, [])


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return sum(1 for _ in fh) - 1  # minus header


def _csv_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _verify_file_exists(report: Report, section: str, path: Path, *, label: str | None = None) -> bool:
    label = label or path.name
    if not path.is_file():
        report.fail(section, label, f"missing: {path}")
        report.missing_files.append(str(path))
        return False
    if path.stat().st_size == 0:
        report.fail(section, label, "empty file")
        return False
    report.pass_(section, label)
    report.verified_paths.append(path)
    return True


def _csv_has_missing(path: Path) -> tuple[int, int]:
    """Return (missing_cells, total_cells) for a CSV. Empty string counts as missing."""
    missing = total = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        for row in reader:
            for cell in row:
                total += 1
                if cell == "":
                    missing += 1
    return missing, total


def _verify_csv_schema(
    report: Report,
    section: str,
    csv_path: Path,
    required_cols: list[str],
    *,
    label: str | None = None,
) -> bool:
    label = label or csv_path.name
    if not csv_path.is_file():
        report.fail(section, label, f"missing: {csv_path}")
        report.missing_files.append(str(csv_path))
        return False
    try:
        cols = _csv_columns(csv_path)
    except Exception as exc:  # pragma: no cover - filesystem edge cases
        report.fail(section, label, f"unreadable: {exc}")
        return False
    missing_cols = [c for c in required_cols if c not in cols]
    if missing_cols:
        msg = f"missing columns {missing_cols}; got {cols}"
        report.schema_errors.append(f"{csv_path}: {msg}")
        report.fail(section, label, msg)
        return False
    if _csv_row_count(csv_path) <= 0:
        report.fail(section, label, "empty (no data rows)")
        return False
    report.pass_(section, label)
    report.verified_paths.append(csv_path)
    return True


def _load_npy_shape(path: Path) -> tuple[int, ...]:
    """Read .npy header without numpy. Supports v1.0 and v2.0 magic.

    Numpy is available in the project env, but we keep this stdlib-only so
    --verify-only works in slim environments.
    """
    import struct

    with path.open("rb") as fh:
        magic = fh.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"not a .npy file: {path}")
        major, minor = struct.unpack("<BB", fh.read(2))
        if major == 1:
            (header_len,) = struct.unpack("<H", fh.read(2))
        elif major == 2:
            (header_len,) = struct.unpack("<I", fh.read(4))
        else:
            raise ValueError(f"unsupported .npy version {major}.{minor}: {path}")
        header = fh.read(header_len).decode("latin1")
    # header is a literal Python dict string; locate the shape tuple.
    start = header.index("'shape':") + len("'shape':")
    paren_open = header.index("(", start)
    paren_close = header.index(")", paren_open)
    inside = header[paren_open + 1 : paren_close].strip().rstrip(",")
    if not inside:
        return ()
    return tuple(int(x.strip()) for x in inside.split(","))


def _sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Section A — Config
# ---------------------------------------------------------------------------
def check_config(report: Report, config_path: Path) -> dict[str, Any]:
    section = "A_config"
    cfg: dict[str, Any] = {}
    if not config_path.is_file():
        report.fail(section, "config_exists", f"missing: {config_path}")
        report.missing_files.append(str(config_path))
        return cfg
    report.pass_(section, "config_exists")

    try:
        cfg = load_config(str(config_path))
    except Exception as exc:
        report.fail(section, "config_loads", f"load_config failed: {exc}")
        return {}
    report.pass_(section, "config_loads")

    expectations = {
        "data_path": EXPECTED_DATA_PATH,
        "input_len": EXPECTED_INPUT_LEN,
        "output_len": EXPECTED_OUTPUT_LEN,
        "results_dir": EXPECTED_RESULTS_DIR,
        "checkpoints_dir": EXPECTED_CHECKPOINTS_DIR,
    }
    for key, expected in expectations.items():
        actual = cfg.get(key)
        if actual is None:
            report.fail(section, f"resolves_{key}", "key missing from resolved config")
        elif actual != expected:
            report.fail(section, f"value_{key}", f"expected {expected!r}, got {actual!r}")
        else:
            report.pass_(section, f"value_{key}")

    seeds = cfg.get("seeds")
    if not isinstance(seeds, list):
        report.fail(section, "value_seeds", f"expected list, got {type(seeds).__name__}")
    else:
        missing = [s for s in EXPECTED_SEEDS if s not in seeds]
        if missing:
            report.fail(section, "value_seeds", f"missing seeds {missing}; got {seeds}")
        else:
            report.pass_(section, "value_seeds")
    return cfg


# ---------------------------------------------------------------------------
# Section B — Dataset
# ---------------------------------------------------------------------------
def check_dataset(report: Report) -> None:
    section = "B_dataset"
    data_path = REPO_ROOT / EXPECTED_DATA_PATH
    if not data_path.is_file():
        report.fail(section, "dataset_exists", f"missing: {data_path}")
        report.missing_files.append(str(data_path))
        return
    report.pass_(section, "dataset_exists")

    cols = _csv_columns(data_path)
    if cols != EXPECTED_ETT_COLUMNS:
        report.fail(section, "dataset_columns", f"expected {EXPECTED_ETT_COLUMNS}, got {cols}")
    else:
        report.pass_(section, "dataset_columns")

    rows = _csv_row_count(data_path)
    if rows != EXPECTED_ETT_ROWS:
        report.fail(section, "dataset_row_count", f"expected {EXPECTED_ETT_ROWS}, got {rows}")
    else:
        report.pass_(section, "dataset_row_count")

    missing_cells, total_cells = _csv_has_missing(data_path)
    if missing_cells > 0:
        report.warn(
            section,
            "dataset_missing_values",
            f"{missing_cells} empty cells out of {total_cells}",
        )
    else:
        report.pass_(section, "dataset_missing_values")
    report.verified_paths.append(data_path)


# ---------------------------------------------------------------------------
# Section C/D — CSV artifacts
# ---------------------------------------------------------------------------
def check_csv_artifacts(report: Report, results_dir: Path) -> None:
    section_c = "C_core_csv"
    for fname, schema in CORE_CSV_SCHEMAS.items():
        _verify_csv_schema(report, section_c, results_dir / fname, schema)

    section_d = "D_xai_csv"
    for fname, schema in XAI_CSV_SCHEMAS.items():
        _verify_csv_schema(report, section_d, results_dir / fname, schema)


def _check_seed_coverage(
    report: Report,
    section: str,
    path: Path,
    *,
    expected: set[str],
) -> None:
    rows = _csv_dict_rows(path)
    actual = {str(row.get("seed", "")).strip() for row in rows}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        report.fail(section, f"{path.name}_seed_coverage", f"missing seeds {missing}; got {sorted(actual)}")
        return
    if unexpected:
        report.warn(section, f"{path.name}_seed_coverage", f"unexpected seed labels {unexpected}")
    report.pass_(section, f"{path.name}_seed_coverage")


def _check_constant_column(
    report: Report,
    section: str,
    path: Path,
    *,
    column: str,
    expected: str,
) -> None:
    rows = _csv_dict_rows(path)
    values = {str(row.get(column, "")).strip() for row in rows}
    if values != {expected}:
        report.fail(section, f"{path.name}_{column}", f"expected only {expected!r}, got {sorted(values)}")
        return
    report.pass_(section, f"{path.name}_{column}")


def _check_lstm_shap_metadata(
    report: Report,
    section: str,
    path: Path,
    *,
    allow_fallback: bool,
) -> None:
    rows = _csv_dict_rows(path)
    true_values = {str(row.get("is_true_shap", "")).strip().lower() for row in rows}
    if true_values != {"true"}:
        detail = f"expected is_true_shap=True for all rows, got {sorted(true_values)}"
        if allow_fallback:
            report.warn(section, "shap_lstm_true_shap", detail)
        else:
            report.fail(section, "shap_lstm_true_shap", detail)
    else:
        report.pass_(section, "shap_lstm_true_shap")
        _check_constant_column(report, section, path, column="explanation_type", expected="SHAP")
    _check_constant_column(report, section, path, column="num_background_windows_used", expected=str(LSTM_FULL_BACKGROUND_WINDOWS))
    _check_constant_column(report, section, path, column="num_eval_windows_used", expected=str(LSTM_FULL_EVAL_WINDOWS))


def check_auxiliary_shap(
    report: Report,
    results_dir: Path,
    *,
    allow_lstm_fallback: bool,
) -> None:
    section = "D_auxiliary_xai"

    for fname in FORBIDDEN_TFT_SHAP_FILES:
        path = results_dir / fname
        if path.exists():
            report.fail(section, fname, f"TFT SHAP artifact is forbidden; TFT must remain VSN/native: {path}")
        else:
            report.pass_(section, f"{fname}_absent")

    for fname, schema in AUXILIARY_SHAP_SCHEMAS.items():
        path = results_dir / fname
        if _verify_csv_schema(report, section, path, schema):
            _check_constant_column(report, section, path, column="input_space", expected="scaled")
            if fname != "shap_lstm.csv":
                _check_constant_column(report, section, path, column="explanation_type", expected="SHAP")

    for fname in ("shap_mlp.csv", "shap_lstm.csv"):
        path = results_dir / fname
        if path.is_file():
            _check_seed_coverage(report, section, path, expected=EXPECTED_MLP_LSTM_SHAP_SEEDS)

    lstm_path = results_dir / "shap_lstm.csv"
    if lstm_path.is_file():
        _check_lstm_shap_metadata(
            report,
            section,
            lstm_path,
            allow_fallback=allow_lstm_fallback,
        )

    for fname in SHAP_CROSS_FILES:
        _verify_file_exists(report, section, results_dir / fname)

    for fname, schema in AUXILIARY_CSV_SCHEMAS.items():
        _verify_csv_schema(report, section, results_dir / fname, schema)


# ---------------------------------------------------------------------------
# Section E — Prediction arrays
# ---------------------------------------------------------------------------
def _check_npy(
    report: Report, section: str, path: Path, *, output_len: int
) -> None:
    label = path.name
    if not path.is_file():
        report.fail(section, label, f"missing: {path}")
        report.missing_files.append(str(path))
        return
    if path.stat().st_size == 0:
        report.fail(section, label, "empty file")
        return
    try:
        shape = _load_npy_shape(path)
    except Exception as exc:
        report.fail(section, label, f"npy header parse failed: {exc}")
        return
    if len(shape) != 3:
        report.fail(section, label, f"expected 3 dims, got {len(shape)} (shape={shape})")
        return
    if shape[-1] != N_VARIABLES:
        report.fail(
            section, label, f"last dim expected {N_VARIABLES}, got {shape[-1]} (shape={shape})"
        )
        return
    horizon_dims = [d for d in shape[:-1] if d == output_len]
    if not horizon_dims:
        # Shape convention not unambiguous; warn rather than fail.
        report.warn(
            section,
            label,
            f"horizon dim {output_len} not found in shape {shape}",
        )
    else:
        report.pass_(section, label)
    report.verified_paths.append(path)


def check_predictions(report: Report, results_dir: Path, seeds: list[int]) -> None:
    section = "E_predictions"
    _check_npy(report, section, results_dir / "preds_lr.npy", output_len=EXPECTED_OUTPUT_LEN)
    for seed in seeds:
        for prefix in ("preds_mlp", "preds_lstm", "preds_tft"):
            _check_npy(
                report,
                section,
                results_dir / f"{prefix}_seed{seed}.npy",
                output_len=EXPECTED_OUTPUT_LEN,
            )


# ---------------------------------------------------------------------------
# Section F — Checkpoints
# ---------------------------------------------------------------------------
def check_checkpoints(report: Report, ckpt_dir: Path, seeds: list[int]) -> None:
    section = "F_checkpoints"
    # LR has no checkpoint (closed-form OLS) — declared exempt per CLAUDE.md § 11A.
    report.pass_(section, "lr_checkpoint_status_NA")

    for seed in seeds:
        for prefix, ext in (("mlp", ".pt"), ("lstm", ".pt")):
            path = ckpt_dir / f"{prefix}_seed{seed}{ext}"
            if not path.is_file():
                report.fail(section, path.name, f"missing: {path}")
                report.missing_files.append(str(path))
                continue
            if path.stat().st_size == 0:
                report.fail(section, path.name, "empty file")
                continue
            report.pass_(section, path.name)
            report.verified_paths.append(path)

        tft_matches = list(ckpt_dir.glob(f"tft_seed{seed}*.ckpt"))
        if not tft_matches:
            report.fail(
                section,
                f"tft_seed{seed}.ckpt",
                f"no checkpoint matching tft_seed{seed}*.ckpt in {ckpt_dir}",
            )
            report.missing_files.append(str(ckpt_dir / f"tft_seed{seed}*.ckpt"))
            continue
        primary = tft_matches[0]
        if primary.stat().st_size == 0:
            report.fail(section, primary.name, "empty file")
            continue
        report.pass_(section, primary.name)
        report.verified_paths.append(primary)


# ---------------------------------------------------------------------------
# Section G — Figures
# ---------------------------------------------------------------------------
def check_figures(report: Report) -> None:
    section = "G_figures"
    if not FIGURES_DIR.is_dir():
        report.fail(section, "figures_dir", f"missing directory: {FIGURES_DIR}")
        return
    for stem in FIGURE_STEMS:
        for ext in (".pdf", ".png"):
            path = FIGURES_DIR / f"{stem}{ext}"
            if not path.is_file():
                report.fail(section, path.name, f"missing: {path}")
                report.missing_files.append(str(path))
                continue
            if path.stat().st_size == 0:
                report.fail(section, path.name, "empty file")
                continue
            report.pass_(section, path.name)
            report.verified_paths.append(path)


# ---------------------------------------------------------------------------
# Section H — Hash manifest
# ---------------------------------------------------------------------------
def check_generated_tables(report: Report) -> None:
    section = "G_generated_tables"
    tables_dir = REPO_ROOT / "docs" / "_generated"
    if not tables_dir.is_dir():
        report.fail(section, "generated_tables_dir", f"missing directory: {tables_dir}")
        return
    for fname in GENERATED_TABLE_FILES:
        _verify_file_exists(report, section, tables_dir / fname)


def write_hash_manifest(report: Report, results_dir: Path) -> Path:
    manifest = results_dir / MANIFEST_FILENAME
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path in report.verified_paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        stat = path.stat()
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            rel = path
        modified = _dt.datetime.fromtimestamp(stat.st_mtime, tz=_dt.timezone.utc).isoformat()
        rows.append(
            {
                "path": str(rel).replace("\\", "/"),
                "sha256": _sha256(path),
                "size_bytes": str(stat.st_size),
                "modified_utc": modified,
            }
        )
    rows.sort(key=lambda r: r["path"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["path", "sha256", "size_bytes", "modified_utc"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


# ---------------------------------------------------------------------------
# Optional regeneration modes
# ---------------------------------------------------------------------------
def run_commands(commands: list[list[str]], *, label: str) -> None:
    print(f"[{label}] executing commands sequentially")
    for cmd in commands:
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        choices=("core", "full"),
        default="core",
        help="core verifies the primary artifact set; full also regenerates and verifies auxiliary SHAP artifacts.",
    )
    parser.add_argument(
        "--config",
        default=_display_path(DEFAULT_CONFIG_PATH),
        help="active v2 config path.",
    )
    legacy = parser.add_mutually_exclusive_group()
    legacy.add_argument(
        "--verify-only",
        action="store_true",
        help="legacy no-op alias; verification is the default behavior.",
    )
    legacy.add_argument(
        "--run-and-verify",
        action="store_true",
        help="legacy behavior: execute the core training/evaluation/XAI pipeline before verification.",
    )
    parser.add_argument(
        "--write-hashes",
        action="store_true",
        help="legacy alias; checksum manifest is always written.",
    )
    parser.add_argument(
        "--require-shap",
        action="store_true",
        help="legacy alias for --mode full.",
    )
    parser.add_argument(
        "--allow-lstm-shap-fallback",
        action="store_true",
        help="allow LSTM manual-gradient fallback rows in full mode; otherwise fallback fails verification.",
    )
    return parser


def write_report(
    report: Report,
    results_dir: Path,
    checkpoints_dir: Path,
    *,
    config_path: Path,
    seeds: list[int],
) -> Path:
    report_path = results_dir / REPORT_FILENAME
    payload = {
        "timestamp_utc": report.timestamp_utc,
        "mode": report.mode,
        "active_config": _display_path(config_path),
        "results_dir": _display_path(results_dir),
        "checkpoints_dir": _display_path(checkpoints_dir),
        "active_seeds": seeds,
        "auxiliary_shap_regenerated": report.auxiliary_shap_regenerated,
        "primary_xai_verified": report.primary_xai_verified,
        "auxiliary_xai_verified": report.auxiliary_xai_verified,
        "checksum_manifest_written": report.checksum_manifest_written,
        "number_of_artifacts_checksummed": report.number_of_artifacts_checksummed,
        "runtime_seconds": round(report.runtime_seconds, 3),
        "status": report.status(),
        "failed_checks": report.failures,
        "checks": report.checks,
        "missing_files": sorted(set(report.missing_files)),
        "schema_errors": report.schema_errors,
        "warnings": report.warnings,
        "verified_artifact_count": len({str(p) for p in report.verified_paths}),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def print_summary(report: Report, report_path: Path, manifest: Path | None, *, config_path: Path) -> None:
    print("=" * 72)
    print(f"Reproducibility verification — {report.status()}")
    print(f"  mode:                 {report.mode}")
    print(f"  active config:        {_display_path(config_path)}")
    print(f"  verified artifacts:   {len({str(p) for p in report.verified_paths})}")
    print(f"  checksummed artifacts:{report.number_of_artifacts_checksummed}")
    print(f"  runtime seconds:      {report.runtime_seconds:.3f}")
    print(f"  failures:             {len(report.failures)}")
    print(f"  warnings:             {len(report.warnings)}")
    print(f"  missing files:        {len(set(report.missing_files))}")
    if report.failures:
        print("  --- failures ---")
        for f in report.failures[:25]:
            print(f"    {f}")
        if len(report.failures) > 25:
            print(f"    ... +{len(report.failures) - 25} more")
    if report.warnings:
        print("  --- warnings ---")
        for w in report.warnings[:25]:
            print(f"    {w}")
        if len(report.warnings) > 25:
            print(f"    ... +{len(report.warnings) - 25} more")
    print(f"  report written:       {report_path.relative_to(REPO_ROOT)}")
    if manifest is not None:
        print(f"  hashes written:       {manifest.relative_to(REPO_ROOT)}")
    print("=" * 72)


def main() -> int:
    started = time.time()
    args = build_parser().parse_args()
    mode = "full" if args.require_shap else args.mode
    config_path = _resolve_repo_path(args.config).resolve()
    config_arg = _config_arg(config_path)

    if args.run_and_verify:
        run_commands(_core_run_commands(config_arg), label="core-run-and-verify")

    report = Report(mode=mode)

    if mode == "full":
        run_commands(_full_shap_commands(config_arg), label="full-auxiliary-shap")
        report.auxiliary_shap_regenerated = True

    cfg = check_config(report, config_path)
    check_dataset(report)

    # Resolve runtime dirs from config when available; fall back to expected.
    results_dir = REPO_ROOT / cfg.get("results_dir", EXPECTED_RESULTS_DIR)
    ckpt_dir = REPO_ROOT / cfg.get("checkpoints_dir", EXPECTED_CHECKPOINTS_DIR)
    seeds = cfg.get("seeds") if isinstance(cfg.get("seeds"), list) else EXPECTED_SEEDS

    check_csv_artifacts(report, results_dir)
    check_predictions(report, results_dir, seeds)
    check_checkpoints(report, ckpt_dir, seeds)
    check_figures(report)
    check_generated_tables(report)

    report.primary_xai_verified = not report.section_has_failures("D_xai_csv")

    if mode == "full":
        check_auxiliary_shap(
            report,
            results_dir,
            allow_lstm_fallback=args.allow_lstm_shap_fallback,
        )
        report.auxiliary_xai_verified = not report.section_has_failures("D_auxiliary_xai")

    manifest_path = write_hash_manifest(report, results_dir)
    report.checksum_manifest_written = manifest_path.is_file()
    report.number_of_artifacts_checksummed = max(_csv_row_count(manifest_path), 0)
    if report.checksum_manifest_written:
        report.pass_("H_provenance_outputs", MANIFEST_FILENAME)
    else:
        report.fail("H_provenance_outputs", MANIFEST_FILENAME, f"missing: {manifest_path}")

    report.runtime_seconds = time.time() - started
    report_path = write_report(
        report,
        results_dir,
        ckpt_dir,
        config_path=config_path,
        seeds=seeds,
    )
    if report_path.is_file():
        report.pass_("H_provenance_outputs", REPORT_FILENAME)
    else:
        report.fail("H_provenance_outputs", REPORT_FILENAME, f"missing: {report_path}")
    # Re-write once so the report includes the report-file self-check above.
    report_path = write_report(
        report,
        results_dir,
        ckpt_dir,
        config_path=config_path,
        seeds=seeds,
    )
    print_summary(report, report_path, manifest_path, config_path=config_path)

    return 0 if report.status() in ("PASS", "PASS_WITH_WARNINGS") else 1


if __name__ == "__main__":
    sys.exit(main())
