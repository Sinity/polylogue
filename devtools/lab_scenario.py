"""Verification-lab showcase scenario runner."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from polylogue.showcase.cli_boundary import invoke_showcase_cli
from polylogue.showcase.exercises import EXERCISES, Exercise
from polylogue.showcase.qa_runner_reporting import format_qa_summary
from polylogue.showcase.qa_runner_request import QAStage, build_qa_session_request
from polylogue.showcase.qa_runner_workflow import run_qa_session
from polylogue.showcase.qa_session_payload import generate_qa_session
from polylogue.showcase.showcase_runner_support import run_exercise

_SCENARIO_NAMES = ("archive-smoke",)
BASELINE_DIR = Path(__file__).resolve().parent.parent / "tests" / "baselines" / "showcase"
TIER_0_GROUPS = frozenset({"structural"})
_ENV_DEPENDENT: frozenset[str] = frozenset({"version"})


def get_tier_0_exercises() -> list[Exercise]:
    """Return tier-0 exercises with stable committed baseline output."""
    return [ex for ex in EXERCISES if ex.group in TIER_0_GROUPS and not ex.needs_data and ex.name not in _ENV_DEPENDENT]


def run_tier_0() -> dict[str, str]:
    """Run tier-0 showcase exercises and return output by exercise name."""
    from polylogue.showcase.exercises import topological_order

    exercises = topological_order(get_tier_0_exercises())
    results: dict[str, str] = {}
    for exercise in exercises:
        result = run_exercise(
            exercise,
            env_vars={},
            invoke_showcase_cli_fn=invoke_showcase_cli,
        )
        results[exercise.name] = result.output or ""
    return results


def load_baselines() -> dict[str, str]:
    """Load committed showcase baselines."""
    if not BASELINE_DIR.exists():
        return {}

    baselines: dict[str, str] = {}
    for path in sorted(BASELINE_DIR.glob("*.txt")):
        baselines[path.stem] = path.read_text(encoding="utf-8")
    return baselines


def save_baselines(outputs: dict[str, str]) -> None:
    """Persist showcase baselines."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for name, output in sorted(outputs.items()):
        (BASELINE_DIR / f"{name}.txt").write_text(output, encoding="utf-8")


def compare_outputs(
    current: dict[str, str],
    baselines: dict[str, str],
) -> list[str]:
    """Compare current exercise output against committed baselines."""
    drifts: list[str] = []

    for name in sorted(set(baselines) - set(current)):
        drifts.append(f"REMOVED: {name} (in baseline but not in current run)")

    for name in sorted(set(current) - set(baselines)):
        drifts.append(f"NEW: {name} (no baseline yet)")

    for name in sorted(set(current) & set(baselines)):
        if current[name] != baselines[name]:
            diff = difflib.unified_diff(
                baselines[name].splitlines(keepends=True),
                current[name].splitlines(keepends=True),
                fromfile=f"baseline/{name}",
                tofile=f"current/{name}",
                n=3,
            )
            drifts.append(f"CHANGED: {name}\n{''.join(diff)}")
    return drifts


def verify_showcase_baselines(*, update: bool) -> int:
    """Verify or update committed tier-0 showcase baselines."""
    print("Running tier 0 showcase exercises...")
    current = run_tier_0()
    print(f"  Ran {len(current)} exercises")

    if update:
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run verification-lab showcase scenarios.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="Run a named showcase scenario set.")
    run_parser.add_argument("scenario", choices=_SCENARIO_NAMES, help="Scenario set to run.")
    run_parser.add_argument(
        "--live", action="store_true", help="Run against the active archive instead of a seeded workspace."
    )
    run_parser.add_argument("--tier", type=int, default=None, help="Only run exercises at this tier.")
    run_parser.add_argument("--report-dir", type=Path, default=None, help="Directory for scenario artifacts.")
    run_parser.add_argument("--json", action="store_true", help="Emit a machine-readable QA session payload.")
    run_parser.add_argument("--verbose", action="store_true", help="Print exercise outputs.")
    run_parser.add_argument("--fail-fast", action="store_true", help="Stop on first exercise failure.")

    baseline_parser = subparsers.add_parser("verify-baselines", help="Verify committed showcase baselines.")
    baseline_parser.add_argument(
        "--update",
        action="store_true",
        help="Update baselines to current output instead of comparing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.action == "verify-baselines":
        return verify_showcase_baselines(update=bool(args.update))
    if args.action != "run":
        parser.error(f"unknown action: {args.action}")
    request = build_qa_session_request(
        synthetic=not bool(args.live),
        source_names=None,
        fresh=None,
        ingest=None,
        regenerate_schemas=False,
        only_stage=QAStage.EXERCISES,
        skip_stages=(),
        workspace=None,
        report_dir=args.report_dir,
        verbose=bool(args.verbose),
        fail_fast=bool(args.fail_fast),
        tier_filter=args.tier,
    )
    result = run_qa_session(request)
    if args.json:
        print(json.dumps(generate_qa_session(result), indent=2))
    else:
        print(format_qa_summary(result))
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
