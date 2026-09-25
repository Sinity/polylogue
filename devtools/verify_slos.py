"""Check benchmark latencies against read-surface SLO targets.

Reads the SLO catalog from docs/plans/slo-catalog.yaml, runs the
referenced benchmark tests with pytest-benchmark, then compares the
measured p50 and p95 latencies against the declared targets.

The benchmark run goes through the managed pytest harness, so it is reachable
from an agent job, where bare pytest is refused.

Exits 0 when all required SLOs have benchmark results and pass their targets.
Exits 1 when any required surface violates its declared SLO, has no benchmark
result, or when the benchmark run itself never produced measurements — that
last case is reported as a refused run, never as missing surfaces.
Informational surfaces without benchmark results are reported without blocking.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from devtools import repo_root as _get_root
from devtools.benchmark_results import parse_pytest_benchmark_stats
from devtools.pytest_slot import PytestSlotUnavailableError, run_pytest
from polylogue.core.json import JSONDocument
from polylogue.scenarios import (
    MeasurementScope,
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
_UNMEASURED_WORKLOAD_DIMENSIONS = (
    "cpu_ms",
    "current_rss_bytes",
    "peak_rss_bytes",
    "current_pss_bytes",
    "peak_pss_bytes",
    "anon_bytes",
    "file_cache_bytes",
    "swap_bytes",
    "temp_storage_bytes",
    "storage_bytes",
    "read_io_bytes",
    "write_io_bytes",
    "response_bytes",
    "cancellation_latency_ms",
    "progress_completed",
    "progress_total",
    "queue_depth",
    "backpressure_ms",
    "cleanup_reclaimed_bytes",
    "sqlite_vm_steps",
)


def _slo_workload_receipt(
    *, catalog_text: str, active_tiers: frozenset[str] | None, wall_ms: float, blocking: bool
) -> JSONDocument:
    """Adapt the SLO benchmark run into the shared workload receipt contract."""
    catalog_digest = hashlib.sha256(catalog_text.encode("utf-8")).hexdigest()
    tiers = ",".join(sorted(active_tiers)) if active_tiers is not None else "all"
    receipt = WorkloadReceipt.from_observations(
        spec=WorkloadEnvelopeSpec(
            workload_id=f"devtools:verify-slos:{tiers}",
            family_id="verification-slo",
            version=1,
            inputs=(WorkloadInputRef(input_id=f"slo-catalog:sha256:{catalog_digest}"),),
            phases=("benchmark",),
            measurement_scope=MeasurementScope.PROCESS_TREE,
        ),
        status=WorkloadRunStatus.FAILED if blocking else WorkloadRunStatus.SUCCEEDED,
        build_id=None,
        runtime_id=f"python:{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        archive_id=None,
        generation_id=None,
        frame_id=None,
        phases=(
            WorkloadPhaseObservation(name="benchmark", wall_ms=wall_ms, unavailable=_UNMEASURED_WORKLOAD_DIMENSIONS),
        ),
        notes=("SLO adapter records benchmark wall time only; resource dimensions are explicitly unavailable.",),
    )
    return receipt.to_payload()


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


class BenchmarkRunUnavailableError(RuntimeError):
    """The benchmark run never produced measurements, for a reason that is not absence.

    A refused or failed launch is not the same fact as "this surface has no
    benchmark". Reporting it as missing surfaces reads as an uninstrumented
    catalog when the truth is that nothing was measured at all.
    """


#: Where the SLO benchmark run keeps its JSON and pytest trees: inside the
#: checkout's disposable scratch, not the host's small /tmp tmpfs.
BENCHMARK_SCRATCH = Path(".cache/verify/slo-benchmarks")

_LOG_TAIL_LINES = 40


def _log_tail(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-_LOG_TAIL_LINES:])


def _run_benchmarks(test_ids: set[str]) -> dict[str, dict[str, float]]:
    """Run the catalog's benchmark tests and return their stats.

    The run goes through the managed pytest harness
    (:func:`devtools.pytest_slot.run_pytest`), the same route ``devtools test``
    uses. Bare pytest is refused inside agent jobs, so a dispatched worker
    could never reach this lane; the harness queues the run as the declared
    ``pytest_focused`` operation in the host's single-slot pytest pool, which
    both satisfies the guard and gives the measurement an exclusive pytest
    slot instead of an ad-hoc subprocess beside other jobs.

    Raises :class:`BenchmarkRunUnavailableError` when the run could not be launched
    or produced no measurement file.
    """
    if not test_ids:
        return {}

    scratch = ROOT / BENCHMARK_SCRATCH / f"run-{os.getpid()}-{time.time_ns():x}"
    scratch.mkdir(parents=True, exist_ok=True)
    json_path = scratch / "benchmark.json"
    log_path = scratch / "benchmark.log"
    pytest_basetemp = scratch / "pytest"

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--override-ini=addopts=-ra",
        "-n",
        "0",
        "-p",
        "no:randomly",
        # The repository default excludes benchmark-marked tests so the
        # ordinary test loop stays fast. This command explicitly runs the
        # benchmark tier it is responsible for measuring.
        "-m",
        "benchmark",
        f"--basetemp={pytest_basetemp}",
        "--benchmark-enable",
        f"--benchmark-json={json_path}",
        *sorted(test_ids),
    ]

    with open(log_path, "wb") as log:
        try:
            outcome = run_pytest(command, cwd=str(ROOT), env=dict(os.environ), root=ROOT, stdout=log)
        except PytestSlotUnavailableError as exc:
            raise BenchmarkRunUnavailableError(f"the managed pytest harness would not start the run: {exc}") from exc

    if outcome.log_path is not None and outcome.log_path.exists():
        # The queued run captured its own log; the local handle stayed empty.
        log_path = outcome.log_path

    if not json_path.exists() or json_path.stat().st_size == 0:
        # Kept, not cleaned: the evidence for why nothing was measured is here.
        raise BenchmarkRunUnavailableError(
            f"the benchmark run wrote no measurement file (pytest slot {outcome.slot}, "
            f"exit {outcome.returncode}); evidence kept under {scratch}; last output:\n{_log_tail(log_path)}"
        )

    if outcome.returncode != 0:
        # A measurement artifact describes work the benchmark managed to
        # emit; it does not turn a failed pytest run into a valid workload.
        # Keep both artifacts so the failed run remains diagnosable.
        try:
            json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            detail = f"; measurement JSON is invalid: {exc}"
        else:
            detail = "; measurement JSON was retained as diagnostics"
        raise BenchmarkRunUnavailableError(
            f"benchmark run failed with exit {outcome.returncode}{detail}; "
            f"raw log and measurement file kept under {scratch}; last output:\n{_log_tail(log_path)}"
        )

    payload = json.loads(json_path.read_text())
    with contextlib.suppress(OSError):
        shutil.rmtree(scratch, ignore_errors=True)

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
        help="Run cheap-local plus the explicit lab benchmark tier.",
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
    benchmark_started = time.monotonic()
    benchmark_error: str | None = None
    benchmark_stats: dict[str, dict[str, float]] = {}
    if not args.skip_benchmarks:
        try:
            benchmark_stats = _run_benchmarks(test_ids)
        except BenchmarkRunUnavailableError as exc:
            benchmark_error = str(exc)
    benchmark_wall_ms = (time.monotonic() - benchmark_started) * 1_000

    # 4. Check each surface against its SLO
    catalog_errors: list[str] = []
    violations: list[dict[str, object]] = []
    missing_required: list[dict[str, object]] = []
    passed: list[dict[str, object]] = []
    uncovered_informational: list[dict[str, object]] = []
    unmeasured: list[dict[str, object]] = []
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
            # A run that never happened is not the same fact as a surface with
            # no benchmark: only the second is "missing".
            reason = (
                "the benchmark run did not execute"
                if benchmark_error is not None
                else "no benchmark result for this test"
            )
            missing_result: dict[str, object] = {
                "surface": surface_name,
                "gate": gate,
                "tier": tier,
                "benchmark_test": benchmark_test,
                "reason": reason,
            }
            if benchmark_error is not None:
                unmeasured.append(missing_result)
            elif gate == "required":
                missing_required.append(missing_result)
            else:
                uncovered_informational.append(missing_result)
            continue

        actual_p50_ms = stats["median"] * 1000  # pytest-benchmark reports in seconds
        estimated_p95_ms = _estimate_p95(stats) * 1000
        actual_mean_ms = stats["mean"] * 1000

        p50_ok = actual_p50_ms <= target_p50
        p95_ok = estimated_p95_ms <= target_p95
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
            "estimated_p95_ms": round(estimated_p95_ms, 2),
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
    blocking = bool(catalog_errors or violations or missing_required or benchmark_error)
    workload_receipt = _slo_workload_receipt(
        catalog_text=catalog_text,
        active_tiers=active_tiers,
        wall_ms=benchmark_wall_ms,
        blocking=blocking,
    )
    if args.json:
        json.dump(
            {
                "blocking": blocking,
                "benchmark_error": benchmark_error,
                "active_tiers": sorted(active_tiers) if active_tiers is not None else None,
                "catalog_errors": catalog_errors,
                "violations": violations,
                "missing_required": missing_required,
                "passed": passed,
                "uncovered_informational": uncovered_informational,
                "unmeasured": unmeasured,
                "skipped_tier": skipped_tier,
                "workload_receipt": workload_receipt,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        if benchmark_error is not None:
            print("BENCHMARKS DID NOT RUN:")
            for line in benchmark_error.splitlines():
                print(f"  {line}")
            print()

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
                    f"estimated p95={p['estimated_p95_ms']:.1f}ms (target ≤{p['target_p95_ms']}ms)"
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
                    parts.append(f"estimated p95={v['estimated_p95_ms']:.1f}ms > {v['target_p95_ms']}ms")
                print(f"  {v['surface']}: {', '.join(parts)}")
            print()

        if uncovered_informational:
            print(f"Uncovered informational ({len(uncovered_informational)} surfaces):")
            for u in uncovered_informational:
                print(f"  {u['surface']}: {u['reason']}")
            print()

        if unmeasured:
            print(f"Unmeasured — the run never produced results ({len(unmeasured)} surfaces):")
            for entry in unmeasured:
                print(f"  {entry['surface']} [gate={entry['gate']}]: {entry['benchmark_test']}")
            print()

        if skipped_tier:
            print(f"Skipped (tier filter) ({len(skipped_tier)} surfaces):")
            for s in skipped_tier:
                print(f"  {s['surface']} [tier={s['tier']}]")
            print()

        if active_tiers is not None:
            print(f"active_tiers={sorted(active_tiers)}")
        print(f"blocking={blocking}")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
