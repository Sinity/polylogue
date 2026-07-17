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
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from devtools import repo_root as _get_root
from devtools.benchmark_results import parse_pytest_benchmark_stats
from devtools.verify_runs import git_head
from polylogue.scenarios.workload import (
    BudgetMeasure,
    BudgetSemantics,
    MeasurementScope,
    WorkloadBudget,
    WorkloadEnvelopeSpec,
    WorkloadInputRef,
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
)

ROOT = _get_root()
SLO_CATALOG = ROOT / "docs" / "plans" / "slo-catalog.yaml"
SLO_GATES = frozenset({"required", "informational"})
SLO_TIERS = frozenset({"cheap-local", "lab"})
DEFAULT_TIER = "cheap-local"
SLO_WORKLOAD_RECEIPT = ROOT / ".cache" / "verify" / "current-slo-workload-receipt.json"


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def _parse_slo_catalog(text: str) -> dict[str, dict[str, object]]:
    """Parse the SLO catalog YAML and return a surface → config dict."""
    import yaml

    data = yaml.safe_load(text)
    return {k: dict(v) for k, v in data["surfaces"].items()}


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def _collect_benchmark_tests(
    surfaces: dict[str, dict[str, object]],
    *,
    active_tiers: frozenset[str] | None = None,
) -> set[str]:
    """Collect unique benchmark test node IDs from the SLO catalog.

    When ``active_tiers`` is provided, only collect tests whose surface tier
    appears in the set. Surfaces with an invalid tier are skipped (the caller
    surfaces the catalog error separately).
    """
    tests: set[str] = set()
    for surface_name, config in surfaces.items():
        if active_tiers is not None:
            tier, tier_error = _surface_tier(surface_name, config)
            if tier_error is not None or tier not in active_tiers:
                continue
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


def _surface_tier(surface_name: str, config: dict[str, object]) -> tuple[str | None, str | None]:
    """Return the surface tier, or a catalog error message when invalid."""
    raw_tier = config.get("tier", DEFAULT_TIER)
    if isinstance(raw_tier, str) and raw_tier in SLO_TIERS:
        return raw_tier, None
    return None, f"{surface_name}: invalid tier {raw_tier!r}; expected one of {sorted(SLO_TIERS)!r}"


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


def _slo_workload_receipt(
    *,
    catalog_text: str,
    surfaces: dict[str, dict[str, object]],
    active_tiers: frozenset[str] | None,
    passed: list[dict[str, object]],
    violations: list[dict[str, object]],
    blocking: bool,
) -> dict[str, object]:
    """Adapt percentile SLO rows into named shared-receipt phases."""
    measured = {str(row["surface"]): row for row in (*passed, *violations) if isinstance(row.get("surface"), str)}
    phases: list[str] = []
    budgets: list[WorkloadBudget] = []
    observations: list[WorkloadPhaseObservation] = []
    for surface_name, config in sorted(surfaces.items()):
        tier, tier_error = _surface_tier(surface_name, config)
        gate, gate_error = _surface_gate(surface_name, config)
        if tier_error is not None or gate_error is not None or tier is None or gate is None:
            continue
        if active_tiers is not None and tier not in active_tiers:
            continue
        row = measured.get(surface_name)
        for statistic in ("p50", "p95"):
            target = config.get(f"{statistic}_ms")
            if not isinstance(target, int):
                continue
            phase = f"{surface_name}:{statistic}"
            phases.append(phase)
            budgets.append(
                WorkloadBudget(
                    BudgetMeasure.WALL_MS,
                    target,
                    (BudgetSemantics.REGRESSION_GATE if gate == "required" else BudgetSemantics.MEASURE_ONLY),
                    phase=phase,
                )
            )
            actual = row.get(f"actual_{statistic}_ms") if row is not None else None
            observations.append(
                WorkloadPhaseObservation(
                    name=phase,
                    wall_ms=float(actual) if isinstance(actual, int | float) else None,
                    unavailable=("wall_ms",) if not isinstance(actual, int | float) else (),
                )
            )
    catalog_id = hashlib.sha256(catalog_text.encode("utf-8")).hexdigest()
    spec = WorkloadEnvelopeSpec(
        workload_id="read-surface-slo-catalog",
        family_id="read-surface-slo",
        version=1,
        inputs=(WorkloadInputRef(input_id=f"slo-catalog:sha256:{catalog_id}"),),
        phases=tuple(phases),
        measurement_scope=MeasurementScope.PROCESS_TREE,
        budgets=tuple(budgets),
    )
    head = git_head(ROOT)
    receipt = WorkloadReceipt.from_observations(
        spec=spec,
        status=WorkloadRunStatus.FAILED if blocking else WorkloadRunStatus.SUCCEEDED,
        build_id=f"git:{head}" if head is not None else None,
        runtime_id=f"python:{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        archive_id=None,
        generation_id=None,
        frame_id=None,
        phases=tuple(observations),
        evidence_refs=(str(SLO_WORKLOAD_RECEIPT.relative_to(ROOT)),),
        notes=("Each surface percentile is a named phase so p50 and p95 retain distinct budgets.",),
    )
    return dict(receipt.to_payload())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_active_tiers(
    *, tier: str | None, include_lab: bool, all_tiers: bool
) -> tuple[frozenset[str] | None, str | None]:
    """Resolve which tiers to evaluate from CLI flags.

    Returns ``(tiers, error)``. ``tiers`` is ``None`` when no filter applies
    (legacy/all-tiers mode), otherwise a frozenset of allowed tier names.
    ``error`` is a user-facing message when flags are inconsistent.
    """
    if all_tiers and (tier is not None or include_lab):
        return None, "--all-tiers is incompatible with --tier <name> / --include-lab"
    if all_tiers:
        return frozenset(SLO_TIERS), None
    if tier is not None:
        if tier not in SLO_TIERS:
            return None, f"--tier {tier!r}: expected one of {sorted(SLO_TIERS)!r}"
        if include_lab and tier == "cheap-local":
            return frozenset({"cheap-local", "lab"}), None
        return frozenset({tier}), None
    if include_lab:
        return frozenset({"cheap-local", "lab"}), None
    return frozenset({DEFAULT_TIER}), None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=SLO_CATALOG)
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip running benchmarks (use with --json to see catalog only)",
    )
    parser.add_argument(
        "--tier",
        choices=sorted(SLO_TIERS),
        default=None,
        help="Run only surfaces declared in this tier (default: cheap-local).",
    )
    parser.add_argument(
        "--include-lab",
        action="store_true",
        help="Run cheap-local plus lab tier (used by `devtools verify --lab`).",
    )
    parser.add_argument(
        "--all-tiers",
        action="store_true",
        help="Run every surface regardless of tier.",
    )
    args = parser.parse_args(argv)

    active_tiers, tier_error = _resolve_active_tiers(
        tier=args.tier, include_lab=args.include_lab, all_tiers=args.all_tiers
    )
    if tier_error is not None:
        print(f"verify-slos: {tier_error}", file=sys.stderr)
        return 2

    # 1. Parse SLO catalog
    catalog_text = args.yaml.read_text()
    surfaces = _parse_slo_catalog(catalog_text)

    # 2. Collect benchmark tests filtered by active tier
    test_ids = _collect_benchmark_tests(surfaces, active_tiers=active_tiers)

    # 3. Run benchmarks
    benchmark_stats = _run_benchmarks(test_ids) if not args.skip_benchmarks else {}

    # 4. Check each surface against its SLO
    catalog_errors: list[str] = []
    violations: list[dict[str, object]] = []
    missing_required: list[dict[str, object]] = []
    passed: list[dict[str, object]] = []
    uncovered_informational: list[dict[str, object]] = []
    skipped_tier: list[dict[str, object]] = []

    for surface_name, config in surfaces.items():
        gate, gate_error = _surface_gate(surface_name, config)
        if gate_error is not None:
            catalog_errors.append(gate_error)
            continue
        assert gate is not None

        tier, tier_catalog_error = _surface_tier(surface_name, config)
        if tier_catalog_error is not None:
            catalog_errors.append(tier_catalog_error)
            continue
        assert tier is not None

        if active_tiers is not None and tier not in active_tiers:
            skipped_tier.append(
                {
                    "surface": surface_name,
                    "gate": gate,
                    "tier": tier,
                    "reason": f"tier {tier!r} not in active tiers {sorted(active_tiers)!r}",
                }
            )
            continue

        target_p50 = config.get("p50_ms")
        target_p95 = config.get("p95_ms")
        if not isinstance(target_p50, int) or not isinstance(target_p95, int):
            if gate == "required":
                catalog_errors.append(f"{surface_name}: required surface must declare integer p50_ms and p95_ms")
            continue

        benchmark_test = config.get("benchmark_test")
        if not isinstance(benchmark_test, str):
            if gate == "required":
                catalog_errors.append(f"{surface_name}: required surface must declare benchmark_test (string)")
            continue

        stats = benchmark_stats.get(benchmark_test)
        if stats is None:
            missing_result: dict[str, object] = {
                "surface": surface_name,
                "gate": gate,
                "tier": tier,
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
            "tier": tier,
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
    workload_receipt = _slo_workload_receipt(
        catalog_text=catalog_text,
        surfaces=surfaces,
        active_tiers=active_tiers,
        passed=passed,
        violations=violations,
        blocking=blocking,
    )
    SLO_WORKLOAD_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    SLO_WORKLOAD_RECEIPT.write_text(json.dumps(workload_receipt, indent=2, ensure_ascii=False) + "\n")
    if args.json:
        json.dump(
            {
                "blocking": blocking,
                "active_tiers": sorted(active_tiers) if active_tiers is not None else None,
                "catalog_errors": catalog_errors,
                "violations": violations,
                "missing_required": missing_required,
                "passed": passed,
                "uncovered_informational": uncovered_informational,
                "skipped_tier": skipped_tier,
                "workload_receipt": workload_receipt,
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

        if skipped_tier:
            print(f"Skipped (tier filter) ({len(skipped_tier)} surfaces):")
            for s in skipped_tier:
                print(f"  {s['surface']} [tier={s['tier']}]")
            print()

        if active_tiers is not None:
            print(f"active_tiers={sorted(active_tiers)}")
        print(f"workload_receipt={workload_receipt['receipt_id']} artifact={SLO_WORKLOAD_RECEIPT.relative_to(ROOT)}")
        print(f"blocking={blocking}")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
