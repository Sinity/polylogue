"""Check benchmark latencies against read-surface SLO targets.

Reads the SLO catalog from docs/plans/slo-catalog.yaml, runs the
referenced benchmark tests with pytest-benchmark, then compares the
measured p50 and p95 latencies against the declared targets.

Exits 0 when all required SLOs have benchmark results and pass their targets.
Exits 1 when any required surface violates its declared SLO or has no
benchmark result. Informational surfaces without benchmark results are reported
without blocking.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from devtools import repo_root as _get_root
from devtools.benchmark_results import parse_pytest_benchmark_stats

ROOT = _get_root()
SLO_CATALOG = ROOT / "docs" / "plans" / "slo-catalog.yaml"
SLO_GATES = frozenset({"required", "informational"})


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


def _surface_gate(surface_name: str, config: dict[str, object]) -> tuple[str | None, str | None]:
    """Return the surface gate, or a catalog error message when invalid."""
    raw_gate = config.get("gate", "required")
    if isinstance(raw_gate, str) and raw_gate in SLO_GATES:
        return raw_gate, None
    return None, f"{surface_name}: invalid gate {raw_gate!r}; expected one of {sorted(SLO_GATES)!r}"


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

    stats: dict[str, dict[str, float]] = {}
    for entry in parse_pytest_benchmark_stats(payload):
        stats[entry.fullname] = {
            "mean": entry.mean,
            "median": entry.median,
            "min": entry.minimum,
            "max": entry.maximum,
            "stddev": entry.stddev,
            "rounds": entry.rounds,
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
    catalog_errors: list[str] = []
    violations: list[dict[str, object]] = []
    missing_required: list[dict[str, object]] = []
    passed: list[dict[str, object]] = []
    uncovered_informational: list[dict[str, object]] = []

    for surface_name, config in surfaces.items():
        gate, gate_error = _surface_gate(surface_name, config)
        if gate_error is not None:
            catalog_errors.append(gate_error)
            continue
        assert gate is not None

        target_p50 = config.get("p50_ms")
        target_p95 = config.get("p95_ms")
        if not isinstance(target_p50, int) or not isinstance(target_p95, int):
            continue

        benchmark_test = config.get("benchmark_test")
        if not isinstance(benchmark_test, str):
            continue

        stats = benchmark_stats.get(benchmark_test)
        if stats is None:
            missing_result: dict[str, object] = {
                "surface": surface_name,
                "gate": gate,
                "benchmark_test": benchmark_test,
                "reason": "no benchmark result for this test",
            }
            if gate == "required":
                missing_required.append(missing_result)
            else:
                uncovered_informational.append(missing_result)
            continue

        actual_p50_ms = stats["median"] * 1000  # pytest-benchmark reports in seconds
        actual_p95_ms = _estimate_p95(stats) * 1000
        actual_mean_ms = stats["mean"] * 1000

        p50_ok = actual_p50_ms <= target_p50
        p95_ok = actual_p95_ms <= target_p95
        ok = p50_ok and p95_ok

        result: dict[str, object] = {
            "surface": surface_name,
            "gate": gate,
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
    blocking = bool(catalog_errors or violations or missing_required)
    if args.json:
        json.dump(
            {
                "blocking": blocking,
                "catalog_errors": catalog_errors,
                "violations": violations,
                "missing_required": missing_required,
                "passed": passed,
                "uncovered_informational": uncovered_informational,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        if catalog_errors:
            print(f"CATALOG ERROR ({len(catalog_errors)} entries):")
            for error in catalog_errors:
                print(f"  - {error}")
            print()

        if passed:
            print(f"PASS ({len(passed)} surfaces):")
            for p in passed:
                print(
                    f"  {p['surface']}: "
                    f"p50={p['actual_p50_ms']:.1f}ms (target ≤{p['target_p50_ms']}ms), "
                    f"p95={p['actual_p95_ms']:.1f}ms (target ≤{p['target_p95_ms']}ms)"
                )
            print()

        if missing_required:
            print(f"MISSING REQUIRED ({len(missing_required)} surfaces):")
            for m in missing_required:
                print(f"  {m['surface']}: {m['benchmark_test']} ({m['reason']})")
            print()

        if violations:
            print(f"VIOLATION ({len(violations)} surfaces):")
            for v in violations:
                parts = []
                if not v["p50_ok"]:
                    parts.append(f"p50={v['actual_p50_ms']:.1f}ms > {v['target_p50_ms']}ms")
                if not v["p95_ok"]:
                    parts.append(f"p95={v['actual_p95_ms']:.1f}ms > {v['target_p95_ms']}ms")
                print(f"  {v['surface']}: {', '.join(parts)}")
            print()

        if uncovered_informational:
            print(f"Uncovered informational ({len(uncovered_informational)} surfaces):")
            for u in uncovered_informational:
                print(f"  {u['surface']}: {u['reason']}")
            print()

        print(f"blocking={blocking}")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
