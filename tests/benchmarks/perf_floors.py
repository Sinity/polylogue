"""Nightly perf floors: curated regression lane over the war-room harnesses (polylogue-196x).

The 2026-07-18/19 campaign produced measured wins (>20x replay throughput via
planner-statistics seeding, 8x/3.3x elsewhere) that nothing protected from
silent regression. This module is the single entry point that turns four of
those harnesses into a floor-checked measurement:

* **census throughput** (raws/s) over the three
  ``tests/infra/revision_backfill_benchmark`` shapes -- SMALL/LARGE payload
  and the REVISION_CHAIN growing-file shape that polylogue-nh44 targeted.
* **replay throughput** (sessions/min) via
  ``backfill_historical_revision_evidence`` end to end (census + replay) on a
  fresh corpus.
* **action_pairs refresh timing** (ms/session) via the real production
  ``refresh_action_pairs`` over a realistically-shaped seeded archive -- the
  exact regression class polylogue-l3tk fixed (missing planner statistics on
  a fresh generation turning a session-scoped refresh into a full-table scan).
* **query latency** (p50/p95 ms) for ``search_summaries``/``list_summaries``,
  computed through the real production
  ``polylogue.operations.route_observation.compute_latency_percentiles``.

Usage::

    python tests/benchmarks/perf_floors.py                       # measure + compare
    python tests/benchmarks/perf_floors.py --quick                # smaller corpora
    python tests/benchmarks/perf_floors.py --json --out run.json  # raw results artifact
    python tests/benchmarks/perf_floors.py --update-floors         # ratchet baseline

Exit code: 0 (within tolerance or ``--update-floors``), 1 (a metric regressed
beyond its tolerance). The nightly workflow runs this fail-soft (a regression
posts a visible annotation, it does not block the job or other jobs).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_VERSION = 1
DEFAULT_FLOORS_PATH = Path(__file__).parent / "floors.json"
DEFAULT_TOLERANCE_PCT = 35.0
_GIT_TIMEOUT_S = 2.0

# Quick mode shrinks corpora for a fast smoke check (local dev / CI dry runs);
# it still exercises every measured surface, just at a smaller scale.
_ACTION_PAIRS_TARGET_MESSAGES = 3000
_ACTION_PAIRS_TARGET_MESSAGES_QUICK = 600
_ACTION_PAIRS_SAMPLE_SESSIONS = 20
_REPLAY_RAW_COUNT = 200
_REPLAY_RAW_COUNT_QUICK = 40
_QUERY_TERMS: tuple[str, ...] = ("analysis", "error", "function", "test")
_QUERY_LIST_REPEATS = 4


@dataclass(frozen=True, slots=True)
class FloorMetric:
    """One measured, floor-checkable quantity."""

    name: str
    value: float
    unit: str
    direction: str  # "higher_is_better" | "lower_is_better"
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Measurement group 1: census throughput per revision-backfill shape
# ---------------------------------------------------------------------------


def measure_census_throughput(workdir: Path, *, quick: bool = False) -> list[FloorMetric]:
    from polylogue.sources.revision_backfill import census_historical_revision_evidence
    from tests.infra.revision_backfill_benchmark import (
        LARGE_PAYLOAD_SHAPE,
        REVISION_CHAIN_SHAPE,
        SMALL_PAYLOAD_SHAPE,
        build_independent_raw_corpus,
        build_revision_chain_corpus,
    )

    metrics: list[FloorMetric] = []

    small_root = workdir / "census-small"
    build_independent_raw_corpus(
        small_root,
        raw_count=SMALL_PAYLOAD_SHAPE["raw_count"],
        avg_payload_bytes=SMALL_PAYLOAD_SHAPE["avg_payload_bytes"],
    )
    start = time.perf_counter()
    result = census_historical_revision_evidence(small_root)
    elapsed = time.perf_counter() - start
    metrics.append(
        FloorMetric(
            "census_small_raws_per_s",
            result.scanned / elapsed if elapsed > 0 else 0.0,
            "raws/s",
            "higher_is_better",
            {"raw_count": result.scanned, "elapsed_s": round(elapsed, 4), "shape": "SMALL_PAYLOAD_SHAPE"},
        )
    )

    if not quick:
        large_root = workdir / "census-large"
        build_independent_raw_corpus(
            large_root,
            raw_count=LARGE_PAYLOAD_SHAPE["raw_count"],
            avg_payload_bytes=LARGE_PAYLOAD_SHAPE["avg_payload_bytes"],
        )
        start = time.perf_counter()
        result = census_historical_revision_evidence(large_root)
        elapsed = time.perf_counter() - start
        metrics.append(
            FloorMetric(
                "census_large_raws_per_s",
                result.scanned / elapsed if elapsed > 0 else 0.0,
                "raws/s",
                "higher_is_better",
                {"raw_count": result.scanned, "elapsed_s": round(elapsed, 4), "shape": "LARGE_PAYLOAD_SHAPE"},
            )
        )

    chain_root = workdir / "census-chain"
    build_revision_chain_corpus(
        chain_root,
        superseded_count=REVISION_CHAIN_SHAPE["superseded_count"],
        final_payload_bytes=REVISION_CHAIN_SHAPE["final_payload_bytes"],
    )
    start = time.perf_counter()
    result = census_historical_revision_evidence(chain_root)
    elapsed = time.perf_counter() - start
    metrics.append(
        FloorMetric(
            "census_chain_revisions_per_s",
            result.scanned / elapsed if elapsed > 0 else 0.0,
            "revisions/s",
            "higher_is_better",
            {"revisions_scanned": result.scanned, "elapsed_s": round(elapsed, 4), "shape": "REVISION_CHAIN_SHAPE"},
        )
    )
    return metrics


# ---------------------------------------------------------------------------
# Measurement group 2: replay throughput (census + replay end to end)
# ---------------------------------------------------------------------------


def measure_replay_throughput(workdir: Path, *, quick: bool = False) -> list[FloorMetric]:
    from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
    from tests.infra.revision_backfill_benchmark import SMALL_PAYLOAD_SHAPE, build_independent_raw_corpus

    raw_count = _REPLAY_RAW_COUNT_QUICK if quick else _REPLAY_RAW_COUNT
    root = workdir / "replay"
    build_independent_raw_corpus(
        root,
        raw_count=raw_count,
        avg_payload_bytes=SMALL_PAYLOAD_SHAPE["avg_payload_bytes"],
    )
    start = time.perf_counter()
    result = backfill_historical_revision_evidence(root)
    elapsed = time.perf_counter() - start
    sessions_per_min = (result.replayed_logical_sources / elapsed) * 60.0 if elapsed > 0 else 0.0
    return [
        FloorMetric(
            "replay_sessions_per_min",
            sessions_per_min,
            "sessions/min",
            "higher_is_better",
            {
                "sessions_replayed": result.replayed_logical_sources,
                "raw_count": raw_count,
                "elapsed_s": round(elapsed, 4),
            },
        )
    ]


# ---------------------------------------------------------------------------
# Shared seeded archive for groups 3 + 4 (action_pairs timing, query latency)
# ---------------------------------------------------------------------------


def _seed_bench_archive(workdir: Path, *, target_messages: int) -> Path:
    """Seed a realistic-distribution archive and return its ``index.db`` path.

    Reuses the same generator ``tests/benchmarks/conftest.py`` uses for its
    session-scoped fixtures, so this measurement exercises the identical
    write path (session/message/block records through the production
    writer) that the rest of the benchmark suite already trusts.
    """
    from tests.benchmarks.conftest import _seed_realistic_db

    db_path = workdir / "seeded" / "index.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _seed_realistic_db(db_path, target_messages=target_messages)
    return db_path


# ---------------------------------------------------------------------------
# Measurement group 3: action_pairs refresh timing (polylogue-l3tk regression class)
# ---------------------------------------------------------------------------


def measure_action_pairs_refresh(
    index_db: Path, *, sample_size: int = _ACTION_PAIRS_SAMPLE_SESSIONS
) -> list[FloorMetric]:
    import sqlite3

    from polylogue.storage.sqlite.action_pairs import refresh_action_pairs

    conn = sqlite3.connect(str(index_db))
    try:
        rows = conn.execute("SELECT session_id FROM sessions ORDER BY session_id LIMIT ?", (sample_size,)).fetchall()
        session_ids = [row[0] for row in rows]
        durations_ms: list[float] = []
        for session_id in session_ids:
            start = time.perf_counter()
            refresh_action_pairs(conn, session_id)
            conn.commit()
            durations_ms.append((time.perf_counter() - start) * 1000.0)
    finally:
        conn.close()

    if not durations_ms:
        return [
            FloorMetric(
                "action_pairs_refresh_mean_ms",
                0.0,
                "ms",
                "lower_is_better",
                {"sample_size": 0, "note": "no sessions found in seeded archive"},
            )
        ]

    durations_ms.sort()
    mean_ms = sum(durations_ms) / len(durations_ms)
    p95_ms = durations_ms[min(len(durations_ms) - 1, round(0.95 * (len(durations_ms) - 1)))]
    return [
        FloorMetric(
            "action_pairs_refresh_mean_ms",
            mean_ms,
            "ms",
            "lower_is_better",
            {"sample_size": len(durations_ms)},
        ),
        FloorMetric(
            "action_pairs_refresh_p95_ms",
            p95_ms,
            "ms",
            "lower_is_better",
            {"sample_size": len(durations_ms)},
        ),
    ]


# ---------------------------------------------------------------------------
# Measurement group 4: query p50/p95 latency via real route-latency percentiles
# ---------------------------------------------------------------------------


def measure_query_latency(index_db: Path) -> list[FloorMetric]:
    from polylogue.operations.route_observation import compute_latency_percentiles
    from polylogue.storage.sqlite.archive_tiers.ops_write import ArchiveRouteObservation
    from tests.benchmarks.helpers import open_bench_store

    durations_by_route: dict[str, list[float]] = defaultdict(list)
    with open_bench_store(index_db) as store:
        for term in _QUERY_TERMS:
            start = time.perf_counter()
            results = store.run(store.repository.search_summaries(term, limit=20))
            list(results)
            durations_by_route["query.search_summaries"].append((time.perf_counter() - start) * 1000.0)
        for _ in range(_QUERY_LIST_REPEATS):
            start = time.perf_counter()
            results = store.run(store.repository.list_summaries(limit=20))
            list(results)
            durations_by_route["query.list_summaries"].append((time.perf_counter() - start) * 1000.0)

    observations: list[ArchiveRouteObservation] = []
    started_at_ms = 1_700_000_000_000
    for route, durations in durations_by_route.items():
        for index, duration_ms in enumerate(durations):
            observations.append(
                ArchiveRouteObservation(
                    observation_id=f"perf-floor-{route}-{index}",
                    trace_id=f"perf-floor-{route}",
                    surface="perf-floor",
                    route=route,
                    verb=None,
                    daemon_path="direct",
                    phase="total",
                    started_at_ms=started_at_ms + index,
                    duration_ms=round(duration_ms),
                    status="ok",
                    git_head=None,
                    archive_epoch=None,
                    attributes={},
                    sampled=True,
                )
            )

    buckets = compute_latency_percentiles(observations)
    metrics: list[FloorMetric] = []
    for bucket in buckets:
        metric_stub = bucket.route.replace(".", "_").replace("-", "_")
        if bucket.p50_ms is not None:
            metrics.append(
                FloorMetric(
                    f"{metric_stub}_p50_ms",
                    bucket.p50_ms,
                    "ms",
                    "lower_is_better",
                    {"sample_count": bucket.sample_count},
                )
            )
        if bucket.p95_ms is not None:
            metrics.append(
                FloorMetric(
                    f"{metric_stub}_p95_ms",
                    bucket.p95_ms,
                    "ms",
                    "lower_is_better",
                    {"sample_count": bucket.sample_count},
                )
            )
    return metrics


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _git_sha(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def machine_fingerprint() -> dict[str, Any]:
    import os

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def run_perf_floor_set(workdir: Path | None = None, *, quick: bool = False) -> dict[str, Any]:
    """Run every curated measurement group and return one JSON-serializable report."""
    owns_workdir = workdir is None
    base = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="plg-perf-floors-"))
    base.mkdir(parents=True, exist_ok=True)

    try:
        metrics: list[FloorMetric] = []
        metrics.extend(measure_census_throughput(base / "census", quick=quick))
        metrics.extend(measure_replay_throughput(base / "replay", quick=quick))

        target_messages = _ACTION_PAIRS_TARGET_MESSAGES_QUICK if quick else _ACTION_PAIRS_TARGET_MESSAGES
        index_db = _seed_bench_archive(base / "seed", target_messages=target_messages)
        metrics.extend(measure_action_pairs_refresh(index_db))
        metrics.extend(measure_query_latency(index_db))

        return {
            "report_version": REPORT_VERSION,
            "tool": "bench perf-floors",
            "generated_at": datetime.now(UTC).isoformat(),
            "git_sha": _git_sha(Path(__file__).resolve().parents[2]),
            "machine": machine_fingerprint(),
            "quick": quick,
            "metrics": {
                metric.name: {
                    "value": metric.value,
                    "unit": metric.unit,
                    "direction": metric.direction,
                    "detail": metric.detail,
                }
                for metric in metrics
            },
        }
    finally:
        if owns_workdir:
            import shutil
            from contextlib import suppress

            with suppress(OSError):
                shutil.rmtree(base)


# ---------------------------------------------------------------------------
# Floors file I/O + comparison
# ---------------------------------------------------------------------------


def load_floors(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"metrics": {}}
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def save_floors(
    path: Path,
    report: dict[str, Any],
    *,
    default_tolerance_pct: float,
    measured_under_load: bool,
    note: str | None,
) -> None:
    floors_metrics: dict[str, Any] = {}
    for name, entry in report["metrics"].items():
        floors_metrics[name] = {
            "baseline": entry["value"],
            "unit": entry["unit"],
            "direction": entry["direction"],
            "tolerance_pct": default_tolerance_pct,
        }
    payload = {
        "generated_at": report["generated_at"],
        "git_sha": report["git_sha"],
        "machine": report["machine"],
        "measured_under_load": measured_under_load,
        "default_tolerance_pct": default_tolerance_pct,
        "note": note,
        "metrics": floors_metrics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True, slots=True)
class FloorComparison:
    name: str
    baseline: float
    candidate: float
    unit: str
    direction: str
    delta_pct: float
    tolerance_pct: float
    regressed: bool
    is_new: bool = False


def _metric_regressed(direction: str, delta_pct: float, tolerance_pct: float) -> bool:
    if direction == "higher_is_better":
        return delta_pct < -tolerance_pct
    return delta_pct > tolerance_pct


def compare_to_floors(
    report: dict[str, Any],
    floors: dict[str, Any],
    *,
    default_tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
) -> list[FloorComparison]:
    """Direction-aware comparison: higher_is_better metrics regress on a drop,
    lower_is_better metrics regress on a rise. Each metric may carry its own
    ``tolerance_pct`` in the floors file; otherwise the file's
    ``default_tolerance_pct`` (or the caller's) applies."""
    floor_metrics: dict[str, Any] = floors.get("metrics", {})
    file_default = floors.get("default_tolerance_pct", default_tolerance_pct)
    results: list[FloorComparison] = []
    for name, entry in sorted(report["metrics"].items()):
        candidate = entry["value"]
        direction = entry["direction"]
        unit = entry["unit"]
        floor = floor_metrics.get(name)
        if floor is None:
            results.append(
                FloorComparison(
                    name=name,
                    baseline=0.0,
                    candidate=candidate,
                    unit=unit,
                    direction=direction,
                    delta_pct=0.0,
                    tolerance_pct=0.0,
                    regressed=False,
                    is_new=True,
                )
            )
            continue
        baseline = float(floor["baseline"])
        tolerance_pct = float(floor.get("tolerance_pct", file_default))
        delta_pct = ((candidate - baseline) / baseline * 100.0) if baseline > 0 else 0.0
        regressed = baseline > 0 and _metric_regressed(direction, delta_pct, tolerance_pct)
        results.append(
            FloorComparison(
                name=name,
                baseline=baseline,
                candidate=candidate,
                unit=unit,
                direction=direction,
                delta_pct=delta_pct,
                tolerance_pct=tolerance_pct,
                regressed=regressed,
                is_new=False,
            )
        )
    return results


def format_delta_table(comparisons: Sequence[FloorComparison]) -> str:
    lines: list[str] = []
    lines.append(f"{'metric':<38} {'baseline':>12} {'candidate':>12} {'delta':>9}  status")
    lines.append("-" * 90)
    for comparison in comparisons:
        if comparison.is_new:
            lines.append(
                f"{comparison.name:<38} {'--':>12} {comparison.candidate:>12.2f} {'--':>9}  NEW (no floor recorded)"
            )
            continue
        status = "REGRESSED" if comparison.regressed else "ok"
        lines.append(
            f"{comparison.name:<38} {comparison.baseline:>12.2f} {comparison.candidate:>12.2f} "
            f"{comparison.delta_pct:>+8.1f}%  {status} (tolerance {comparison.tolerance_pct:.0f}%)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floors", type=Path, default=DEFAULT_FLOORS_PATH, help="Path to the committed floors JSON")
    parser.add_argument("--out", type=Path, default=None, help="Write the raw measurement report JSON here")
    parser.add_argument("--json", action="store_true", help="Print the raw measurement report JSON to stdout")
    parser.add_argument("--quick", action="store_true", help="Smaller corpora for a fast smoke check")
    parser.add_argument(
        "--update-floors",
        action="store_true",
        help="Regenerate the floors file from this run's measurements (ratchet) instead of comparing",
    )
    parser.add_argument(
        "--tolerance-pct",
        type=float,
        default=DEFAULT_TOLERANCE_PCT,
        help="Default per-metric tolerance when writing floors with --update-floors",
    )
    parser.add_argument(
        "--measured-under-load",
        action="store_true",
        help="Record that this run shared the host with other heavy work (widens the recorded tolerance intent)",
    )
    parser.add_argument("--note", default=None, help="Free-text note to record alongside --update-floors")
    parser.add_argument("--workdir", type=Path, default=None, help="Reuse a directory instead of a private temp dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_perf_floor_set(workdir=args.workdir, quick=args.quick)

    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if args.update_floors:
        save_floors(
            args.floors,
            report,
            default_tolerance_pct=args.tolerance_pct,
            measured_under_load=args.measured_under_load,
            note=args.note,
        )
        print(f"Floors updated: {args.floors}")
        return 0

    floors = load_floors(args.floors)
    comparisons = compare_to_floors(report, floors)
    if not args.json:
        print(format_delta_table(comparisons))
    regressions = [c for c in comparisons if c.regressed]
    if regressions:
        if not args.json:
            print(f"\nFAIL: {len(regressions)} metric(s) regressed beyond tolerance.")
        return 1
    if not args.json:
        print("\nOK: no perf-floor regressions detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
