"""Benchmark regression detection from pytest-benchmark JSON output.

Reads pytest-benchmark JSON output and compares against a saved baseline,
reporting regressions that exceed configured warn/fail thresholds.

Usage:
    # Save baseline
    pytest tests/benchmarks/ --benchmark-enable -p no:xdist \\
      --benchmark-json=.local/benchmark-baseline.json

    # Compare against baseline
    python devtools/benchmark_regression_detect.py \\
      --baseline .local/benchmark-baseline.json \\
      --candidate .local/benchmark-candidate.json

Exit code: 0 (no regressions), 1 (warnings only), 2 (failures).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_benchmarks(path: Path | str) -> dict[str, dict[str, float]]:
    """Load pytest-benchmark JSON and return {test_name: {mean, median, p95, ...}}."""
    data = json.loads(Path(path).read_text())
    benchmarks: dict[str, dict[str, float]] = {}
    for bench in data.get("benchmarks", []):
        name = bench["name"]
        stats = bench.get("stats", {})
        benchmarks[name] = {
            "mean": stats.get("mean", 0),
            "median": stats.get("median", 0),
            "p95": stats.get("p95", 0),
            "min": stats.get("min", 0),
            "max": stats.get("max", 0),
            "stddev": stats.get("stddev", 0),
            "rounds": stats.get("rounds", 0),
        }
    return benchmarks


def detect_regressions(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    warn_pct: float = 10.0,
    fail_pct: float = 20.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Detect regressions between baseline and candidate.

    Returns (warnings, failures) — each a list of regression dicts.
    """
    warnings: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    common = set(baseline) & set(candidate)

    for name in sorted(common):
        base_mean = baseline[name]["mean"]
        cand_mean = candidate[name]["mean"]

        if base_mean <= 0:
            continue

        delta_pct = ((cand_mean - base_mean) / base_mean) * 100

        entry = {
            "test": name,
            "baseline_mean_s": round(base_mean, 6),
            "candidate_mean_s": round(cand_mean, 6),
            "delta_pct": round(delta_pct, 1),
        }

        if delta_pct >= fail_pct:
            failures.append(entry)
        elif delta_pct >= warn_pct:
            warnings.append(entry)

    return warnings, failures


def format_report(
    warnings: list[dict[str, object]],
    failures: list[dict[str, object]],
    new_tests: set[str],
    removed_tests: set[str],
) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Benchmark Regression Report")
    lines.append("=" * 72)

    if new_tests:
        lines.append(f"\nNew tests ({len(new_tests)}):")
        for t in sorted(new_tests):
            lines.append(f"  + {t}")

    if removed_tests:
        lines.append(f"\nRemoved tests ({len(removed_tests)}):")
        for t in sorted(removed_tests):
            lines.append(f"  - {t}")

    if warnings:
        lines.append(f"\nWARNINGS ({len(warnings)} — >10% slowdown):")
        for w in warnings:
            lines.append(
                f"  ⚠ {w['test']}: {w['baseline_mean_s']:.4f}s → {w['candidate_mean_s']:.4f}s ({w['delta_pct']:+.1f}%)"
            )

    if failures:
        lines.append(f"\nFAILURES ({len(failures)} — >20% slowdown):")
        for f in failures:
            lines.append(
                f"  ✗ {f['test']}: {f['baseline_mean_s']:.4f}s → {f['candidate_mean_s']:.4f}s ({f['delta_pct']:+.1f}%)"
            )

    if not warnings and not failures:
        lines.append("\n✓ No regressions detected.")
        if new_tests or removed_tests:
            lines.append("  (test inventory changed; no comparable tests regressed)")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect benchmark regressions")
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline benchmark JSON")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate benchmark JSON")
    parser.add_argument("--warn-pct", type=float, default=10.0, help="Warning threshold %% (default: 10)")
    parser.add_argument("--fail-pct", type=float, default=20.0, help="Failure threshold %% (default: 20)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"Baseline not found: {args.baseline}", file=sys.stderr)
        return 1

    if not args.candidate.exists():
        print(f"Candidate not found: {args.candidate}", file=sys.stderr)
        return 1

    baseline = load_benchmarks(args.baseline)
    candidate = load_benchmarks(args.candidate)

    warnings, failures = detect_regressions(
        baseline,
        candidate,
        warn_pct=args.warn_pct,
        fail_pct=args.fail_pct,
    )

    new_tests = set(candidate) - set(baseline)
    removed_tests = set(baseline) - set(candidate)

    if args.json:
        print(
            json.dumps(
                {
                    "warnings": warnings,
                    "failures": failures,
                    "new_tests": sorted(new_tests),
                    "removed_tests": sorted(removed_tests),
                },
                indent=2,
            )
        )
    else:
        print(format_report(warnings, failures, new_tests, removed_tests))

    if failures:
        return 2
    if warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
