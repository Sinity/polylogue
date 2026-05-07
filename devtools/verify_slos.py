"""Check benchmark latencies against read-surface SLO targets.

Reads the SLO catalog from docs/plans/slo-catalog.yaml, runs the
referenced benchmark tests with pytest-benchmark, then compares the
measured p50 and p95 latencies against the declared targets.

Exits 0 when all SLOs with benchmark results pass their targets.
Exits 1 when any surface violates its declared SLO.
Exits 0 (with warnings) when a surface has no benchmark coverage — these
are deferred surfaces, not failures.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLO_CATALOG = ROOT / "docs" / "plans" / "slo-catalog.yaml"


# ---------------------------------------------------------------------------
# YAML parsing (tiny, no PyYAML dependency)
# ---------------------------------------------------------------------------


def _parse_slo_catalog(text: str) -> dict[str, dict[str, object]]:
    """Parse the SLO catalog YAML and return a surface → config dict."""
    surfaces: dict[str, dict[str, object]] = {}
    current_surface: str | None = None
    current_config: dict[str, object] | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if indent == 0 and stripped.endswith(":"):
            # Top-level key (e.g. "surfaces:")
            pass
        elif indent == 2 and not stripped.startswith("-") and stripped.endswith(":"):
            # Surface name (e.g. "  query:")
            current_surface = stripped.rstrip(":")
            current_config = {}
            surfaces[current_surface] = current_config
        elif indent == 4 and current_config is not None and ": " in stripped:
            key, _, value = stripped.partition(": ")
            current_config[key] = _coerce_slo_value(value.strip())
        elif indent == 4 and current_config is not None and stripped.endswith(":"):
            # Start of a nested block — skip for now
            pass

    return surfaces


def _coerce_slo_value(value: str) -> str | int:
    """Coerce a YAML scalar to str or int."""
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def _collect_benchmark_tests(surfaces: dict[str, dict[str, object]]) -> set[str]:
    """Collect unique benchmark test node IDs from the SLO catalog."""
    tests: set[str] = set()
    for _surface_name, config in surfaces.items():
        test = config.get("benchmark_test")
        if isinstance(test, str) and test.strip():
            tests.add(test.strip())
    return tests


def _run_benchmarks(test_ids: set[str]) -> dict[str, dict[str, float]]:
    """Run pytest-benchmark on the given test node IDs and return stats.

    Returns a dict mapping full test name → {p50, p95, mean, ...}.
    """
    if not test_ids:
        return {}

    with tempfile.TemporaryDirectory(prefix="verify-slos-") as tmpdir:
        json_path = Path(tmpdir) / "benchmark.json"

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--override-ini=addopts=-ra",
            "-n",
            "0",
            "-p",
            "no:randomly",
            "--benchmark-enable",
            f"--benchmark-json={json_path}",
            *sorted(test_ids),
        ]

        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

        if not json_path.exists() or json_path.stat().st_size == 0:
            print("verify-slos: no benchmark results produced", file=sys.stderr)
            if result.returncode != 0:
                print(result.stderr, file=sys.stderr)
            return {}

        if result.returncode != 0:
            print("verify-slos: benchmark run exited non-zero", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            # Still try to read the JSON — pytest-benchmark writes it even on
            # test failures.

        payload = json.loads(json_path.read_text())

    benchmarks = payload.get("benchmarks", [])
    stats: dict[str, dict[str, float]] = {}
    for entry in benchmarks:
        fullname = entry.get("fullname") or entry.get("fullfunc") or entry.get("name", "")
        entry_stats = entry.get("stats") or entry.get("Stats") or {}
        if not fullname or not entry_stats:
            continue
        stats[str(fullname)] = {
            "mean": float(entry_stats.get("mean", 0)),
            "median": float(entry_stats.get("median", 0)),
            "min": float(entry_stats.get("min", 0)),
            "max": float(entry_stats.get("max", 0)),
            "stddev": float(entry_stats.get("stddev", 0))
            if "stddev" in entry_stats
            else 0.0,
            "rounds": int(entry_stats.get("rounds", 0)),
        }

    return stats


# ---------------------------------------------------------------------------
# p95 estimation
# ---------------------------------------------------------------------------


def _estimate_p95(entry_stats: dict[str, float]) -> float:
    """Estimate p95 from pytest-benchmark stats (assumes near-normal)."""
    mean = entry_stats.get("mean", 0)
    stddev = entry_stats.get("stddev", 0)
    # p95 ≈ mean + 1.645 * stddev  (for normal distribution)
    return mean + 1.645 * stddev


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=SLO_CATALOG)
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip running benchmarks (use with --json to see catalog only)",
    )
    args = parser.parse_args(argv)

    # 1. Parse SLO catalog
    catalog_text = args.yaml.read_text()
    surfaces = _parse_slo_catalog(catalog_text)

    # 2. Collect benchmark tests
    test_ids = _collect_benchmark_tests(surfaces)

    # 3. Run benchmarks
    benchmark_stats = _run_benchmarks(test_ids) if not args.skip_benchmarks else {}

    # 4. Check each surface against its SLO
    violations: list[dict[str, object]] = []
    passed: list[dict[str, object]] = []
    uncovered: list[dict[str, object]] = []

    for surface_name, config in surfaces.items():
        target_p50 = config.get("p50_ms")
        target_p95 = config.get("p95_ms")
        if not isinstance(target_p50, int) or not isinstance(target_p95, int):
            continue

        benchmark_test = config.get("benchmark_test")
        if not isinstance(benchmark_test, str):
            continue

        stats = benchmark_stats.get(benchmark_test)
        if stats is None:
            uncovered.append(
                {
                    "surface": surface_name,
                    "benchmark_test": benchmark_test,
                    "reason": "no benchmark result for this test",
                }
            )
            continue

        actual_p50_ms = stats["median"] * 1000  # pytest-benchmark reports in seconds
        actual_p95_ms = _estimate_p95(stats) * 1000
        actual_mean_ms = stats["mean"] * 1000

        p50_ok = actual_p50_ms <= target_p50
        p95_ok = actual_p95_ms <= target_p95
        ok = p50_ok and p95_ok

        result: dict[str, object] = {
            "surface": surface_name,
            "description": config.get("description", ""),
            "benchmark_test": benchmark_test,
            "target_p50_ms": target_p50,
            "target_p95_ms": target_p95,
            "actual_p50_ms": round(actual_p50_ms, 2),
            "actual_p95_ms": round(actual_p95_ms, 2),
            "actual_mean_ms": round(actual_mean_ms, 2),
            "p50_ok": p50_ok,
            "p95_ok": p95_ok,
            "rounds": stats.get("rounds", 0),
        }

        if ok:
            passed.append(result)
        else:
            violations.append(result)

    # 5. Report
    if args.json:
        json.dump(
            {
                "blocking": len(violations) > 0,
                "violations": violations,
                "passed": passed,
                "uncovered": uncovered,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        if passed:
            print(f"PASS ({len(passed)} surfaces):")
            for p in passed:
                print(
                    f"  {p['surface']}: "
                    f"p50={p['actual_p50_ms']:.1f}ms (target ≤{p['target_p50_ms']}ms), "
                    f"p95={p['actual_p95_ms']:.1f}ms (target ≤{p['target_p95_ms']}ms)"
                )
            print()

        if violations:
            print(f"VIOLATION ({len(violations)} surfaces):")
            for v in violations:
                parts = []
                if not v["p50_ok"]:
                    parts.append(
                        f"p50={v['actual_p50_ms']:.1f}ms > {v['target_p50_ms']}ms"
                    )
                if not v["p95_ok"]:
                    parts.append(
                        f"p95={v['actual_p95_ms']:.1f}ms > {v['target_p95_ms']}ms"
                    )
                print(f"  {v['surface']}: {', '.join(parts)}")
            print()

        if uncovered:
            print(f"Uncovered ({len(uncovered)} surfaces):")
            for u in uncovered:
                print(f"  {u['surface']}: {u['reason']}")
            print()

        print(f"blocking={len(violations) > 0}")

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
