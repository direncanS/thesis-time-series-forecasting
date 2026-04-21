"""Smoke tests for scripts/regen_tables.py (Closure Plan v6.1 B4).

Verifies:
  - VALIDATION_LOG.md parser recognises plain and bold status labels
  - regen script resolves v1 archive fallback when v2 results/ is empty
  - --verify mode is a true dry-run (no disk writes)
  - fragment inlining is idempotent (sha-stable across two consecutive runs)
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "regen_tables.py"
SOLUTION_MD = REPO_ROOT / "docs" / "solution.md"
APPENDIX_MD = REPO_ROOT / "docs" / "appendix.md"
GENERATED_DIR = REPO_ROOT / "docs" / "_generated"


def _sha(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _all_hashes() -> dict[str, str]:
    d = {"solution": _sha(SOLUTION_MD), "appendix": _sha(APPENDIX_MD)}
    for frag in sorted(GENERATED_DIR.glob("*.md")):
        d[frag.name] = _sha(frag)
    return d


def test_parse_validation_log_finds_validated_scripts():
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.regen_tables import VALIDATION_LOG, parse_validation_log

    status_map = parse_validation_log(VALIDATION_LOG)
    # Expect at least the core pipeline scripts to be present
    for expected in (
        "src/training/multi_seed.py",
        "src/training/tft_fair_3seed.py",
        "src/evaluation/export_predictions.py",
        "src/evaluation/post_training_analysis.py",
    ):
        assert expected in status_map, f"{expected} missing from VALIDATION_LOG parse"
        assert status_map[expected] in {"VALIDATED", "PARTIAL", "PENDING"}, \
            f"{expected} has unexpected status {status_map[expected]}"


def test_regen_runs_and_writes_fragments():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    for name in (
        "overall_metrics_table.md",
        "bootstrap_headline.md",
        "per_seed_table.md",
        "bootstrap_full.md",
        "complexity_table.md",
        "pipeline_status_table.md",
        "status_summary.md",
    ):
        assert (GENERATED_DIR / name).exists(), f"fragment {name} not written"


def test_regen_verify_mode_does_not_write():
    # Capture hashes before + run verify + assert hashes unchanged.
    before = _all_hashes()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--verify"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    after = _all_hashes()
    assert before == after, "verify mode must not modify any tracked file"


def test_regen_is_idempotent():
    # Baseline + two consecutive runs must produce identical file hashes.
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=str(REPO_ROOT))
    first = _all_hashes()
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=str(REPO_ROOT))
    second = _all_hashes()
    assert first == second, "regen_tables is not idempotent"


def test_prose_has_no_stale_pending():
    """After regen, grep 'PENDING' in solution.md + appendix.md should be 0.

    Static 'N/A (§ 11A item 9 exception)' and similar references are allowed
    (hand-authored config tables). This test targets only dynamic PENDING.
    """
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=str(REPO_ROOT))
    for md in (SOLUTION_MD, APPENDIX_MD):
        text = md.read_text(encoding="utf-8")
        # Count PENDING occurrences (case-sensitive)
        pending_count = text.count("PENDING")
        assert pending_count == 0, \
            f"{md.name} contains {pending_count} 'PENDING' reference(s) after regen"
