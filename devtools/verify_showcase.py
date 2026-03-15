"""Verify showcase tier 0 output matches committed baselines.

Usage: python -m devtools.verify_showcase
Exit code: 0 if baselines match, 1 if drift detected.

Tier 0 exercises are structural tests (help screens, version output)
that produce deterministic output. This script runs them and compares
against stored baselines to detect unintentional CLI changes.
"""

from __future__ import annotations

import difflib
import json
import sys
import tempfile
from pathlib import Path

from polylogue.showcase.exercises import EXERCISES, Exercise
from polylogue.showcase.runner import ShowcaseRunner


BASELINE_DIR = Path(__file__).resolve().parent.parent / "tests" / "baselines" / "showcase"

TIER_0_GROUPS = frozenset({"structural", "sources"})


def get_tier_0_exercises() -> list[Exercise]:
    """Return all tier-0 exercises (structural + sources, no data needed)."""
    return [ex for ex in EXERCISES if ex.group in TIER_0_GROUPS and not ex.needs_data]


def run_tier_0() -> dict[str, str]:
    """Run tier 0 exercises and return {exercise_name: output} mapping."""
    runner = ShowcaseRunner(live=False, fail_fast=False, verbose=False)

    # We only need to run the structural exercises — no seeding needed.
    # Use the runner's _run_exercise directly to avoid full workspace setup.
    from polylogue.showcase.exercises import topological_order

    exercises = get_tier_0_exercises()
    exercises = topological_order(exercises)

    results: dict[str, str] = {}
    for ex in exercises:
        er = runner._run_exercise(ex)
        results[ex.name] = er.output or ""

    return results


def load_baselines() -> dict[str, str]:
    """Load committed baseline outputs from disk.

    Returns empty dict if baseline directory doesn't exist yet.
    """
    if not BASELINE_DIR.exists():
        return {}

    baselines: dict[str, str] = {}
    for path in sorted(BASELINE_DIR.glob("*.txt")):
        name = path.stem  # e.g. "help-main.txt" -> "help-main"
        baselines[name] = path.read_text(encoding="utf-8")

    return baselines


def save_baselines(outputs: dict[str, str]) -> None:
    """Save current outputs as new baselines."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for name, output in sorted(outputs.items()):
        (BASELINE_DIR / f"{name}.txt").write_text(output, encoding="utf-8")


def compare_outputs(
    current: dict[str, str],
    baselines: dict[str, str],
) -> list[str]:
    """Compare current outputs against baselines, returning drift descriptions.

    Returns empty list if no drift detected.
    """
    drifts: list[str] = []

    # Check for exercises in baselines but not in current (removed exercises)
    for name in sorted(set(baselines) - set(current)):
        drifts.append(f"REMOVED: {name} (in baseline but not in current run)")

    # Check for exercises in current but not in baselines (new exercises)
    for name in sorted(set(current) - set(baselines)):
        drifts.append(f"NEW: {name} (no baseline yet)")

    # Check for output differences
    for name in sorted(set(current) & set(baselines)):
        if current[name] != baselines[name]:
            diff = difflib.unified_diff(
                baselines[name].splitlines(keepends=True),
                current[name].splitlines(keepends=True),
                fromfile=f"baseline/{name}",
                tofile=f"current/{name}",
                n=3,
            )
            diff_text = "".join(diff)
            drifts.append(f"CHANGED: {name}\n{diff_text}")

    return drifts


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments (for testing). Supports:
            --update: Update baselines instead of comparing.

    Returns:
        0 if baselines match (or updated), 1 if drift detected.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update", action="store_true",
        help="Update baselines to current output instead of comparing",
    )
    args = parser.parse_args(argv)

    print("Running tier 0 showcase exercises...")
    current = run_tier_0()
    print(f"  Ran {len(current)} exercises")

    if args.update:
        save_baselines(current)
        print(f"  Updated baselines in {BASELINE_DIR}")
        return 0

    baselines = load_baselines()
    if not baselines:
        print("ERROR: No baselines found. Run with --update to create them.")
        return 1

    drifts = compare_outputs(current, baselines)
    if not drifts:
        print("  All baselines match.")
        return 0

    print(f"\n  DRIFT DETECTED ({len(drifts)} differences):\n")
    for drift in drifts:
        print(f"  {drift}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
