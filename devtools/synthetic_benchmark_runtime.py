"""Synthetic benchmark runner registry and shared result model."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar


@dataclass
class CampaignResult:
    """Result of a single synthetic benchmark campaign run."""

    campaign_name: str
    scale_level: str
    metrics: dict[str, float] = field(default_factory=dict)
    db_stats: dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    origin: str = "authored"
    path_targets: list[str] = field(default_factory=list)
    artifact_targets: list[str] = field(default_factory=list)
    operation_targets: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


SyntheticBenchmarkRunner = Callable[[Path], CampaignResult | Awaitable[CampaignResult]]
_T = TypeVar("_T")


def _row_count(value: object) -> int:
    return int(value) if isinstance(value, int) else 0


def _db_row_counts(db_path: Path) -> dict[str, int]:
    """Collect row counts and file size for a database."""
    from polylogue.storage.sqlite.connection import open_connection

    stats: dict[str, int] = {}
    if not db_path.exists():
        return stats
    stats["db_size_bytes"] = db_path.stat().st_size
    with open_connection(db_path) as conn:
        for table in ("sessions", "messages", "blocks", "raw_sessions"):
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                stats[f"{table}_count"] = _row_count(row[0]) if row else 0
            except Exception:
                stats[f"{table}_count"] = 0
        try:
            row = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()
            stats["fts_rows"] = _row_count(row[0]) if row else 0
        except Exception:
            stats["fts_rows"] = 0
    return stats


def _measure(fn: Callable[..., _T], *args: object, **kwargs: object) -> tuple[float, _T]:
    """Call fn(*args, **kwargs), return (elapsed_seconds, result)."""
    t0 = time.monotonic()
    result = fn(*args, **kwargs)
    return time.monotonic() - t0, result


async def _ameasure(coro: Awaitable[_T]) -> tuple[float, _T]:
    """Await coro, return (elapsed_seconds, result)."""
    t0 = time.monotonic()
    result = await coro
    return time.monotonic() - t0, result


def _session_insight_table_counts(db_path: Path) -> dict[str, int]:
    """Collect row counts for durable session-insight tables and live FTS projections."""
    from polylogue.storage.sqlite.connection import open_connection

    stats: dict[str, int] = {}
    if not db_path.exists():
        return stats
    with open_connection(db_path) as conn:
        for table in (
            "session_profiles",
            "session_work_events",
            "session_work_events_fts",
            "session_phases",
            "threads",
            "threads_fts",
            "session_tag_rollups",
        ):
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                stats[f"{table}_count"] = _row_count(row[0]) if row else 0
            except Exception:
                stats[f"{table}_count"] = 0
    return stats


def run_fts_rebuild_campaign(db_path: Path) -> CampaignResult:
    """Benchmark full FTS index rebuild."""
    from polylogue.storage.index import rebuild_index
    from polylogue.storage.sqlite.connection import open_connection

    stats_before = _db_row_counts(db_path)

    with open_connection(db_path) as conn:
        try:
            conn.execute("DELETE FROM messages_fts")
            conn.commit()
        except Exception:
            pass

        elapsed, _ = _measure(rebuild_index, conn)

    stats_after = _db_row_counts(db_path)

    return CampaignResult(
        campaign_name="fts-rebuild",
        scale_level="",
        metrics={"rebuild_wall_s": round(elapsed, 3)},
        db_stats={
            "messages_count": stats_before.get("messages_count", 0),
            "fts_rows_before": 0,
            "fts_rows_after": stats_after.get("fts_rows", 0),
            "db_size_bytes": stats_after.get("db_size_bytes", 0),
        },
    )


async def run_incremental_index_campaign(db_path: Path, batch_size: int = 100) -> CampaignResult:
    """Benchmark incremental FTS index updates."""
    from polylogue.storage.query_models import SessionRecordQuery
    from polylogue.storage.search_providers.fts5 import FTS5Provider
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

    backend = SQLiteBackend(db_path=db_path)
    try:
        total_convs = await backend.queries.count_sessions(SessionRecordQuery())
        batch_times: list[float] = []
        indexed_total = 0
        fts = FTS5Provider(db_path=db_path)

        offset = 0
        while offset < total_convs:
            convs = await backend.queries.list_sessions(SessionRecordQuery(limit=batch_size, offset=offset))
            if not convs:
                break

            conv_ids = [str(c.session_id) for c in convs]
            messages_by_conv = await backend.get_messages_batch(conv_ids)
            all_messages = [msg for msgs in messages_by_conv.values() for msg in msgs]

            t0 = time.monotonic()
            fts.index(all_messages)
            batch_elapsed = time.monotonic() - t0
            batch_times.append(batch_elapsed)
            indexed_total += len(all_messages)
            offset += batch_size

        total_time = sum(batch_times)
        avg_batch = total_time / len(batch_times) if batch_times else 0.0

        return CampaignResult(
            campaign_name="incremental-index",
            scale_level="",
            metrics={
                "total_wall_s": round(total_time, 3),
                "avg_batch_s": round(avg_batch, 4),
                "batches": len(batch_times),
                "messages_indexed": indexed_total,
            },
            db_stats=_db_row_counts(db_path),
        )
    finally:
        await backend.close()


async def run_filter_scan_campaign(db_path: Path) -> CampaignResult:
    """Benchmark common filter query patterns."""
    from polylogue.storage.query_models import SessionRecordQuery
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

    backend = SQLiteBackend(db_path=db_path)
    metrics: dict[str, float] = {}

    try:
        elapsed, results = await _ameasure(backend.queries.list_sessions(SessionRecordQuery(limit=50)))
        metrics["list_50_wall_s"] = round(elapsed, 4)
        metrics["list_50_count"] = len(results)

        elapsed, results = await _ameasure(
            backend.queries.list_sessions(SessionRecordQuery(origin="chatgpt", limit=50))
        )
        metrics["filter_provider_wall_s"] = round(elapsed, 4)
        metrics["filter_provider_count"] = len(results)

        elapsed, results = await _ameasure(
            backend.queries.list_sessions(SessionRecordQuery(has_tool_use=True, limit=50))
        )
        metrics["filter_tool_use_wall_s"] = round(elapsed, 4)
        metrics["filter_tool_use_count"] = len(results)

        elapsed, results = await _ameasure(
            backend.queries.list_sessions(SessionRecordQuery(has_thinking=True, limit=50))
        )
        metrics["filter_thinking_wall_s"] = round(elapsed, 4)
        metrics["filter_thinking_count"] = len(results)

        elapsed, count = await _ameasure(backend.queries.count_sessions(SessionRecordQuery()))
        metrics["count_all_wall_s"] = round(elapsed, 4)
        metrics["count_all"] = count

        elapsed, _ = await _ameasure(
            backend.queries.list_sessions(SessionRecordQuery(title_contains="synthetic", limit=50))
        )
        metrics["filter_title_wall_s"] = round(elapsed, 4)

        return CampaignResult(
            campaign_name="filter-scan",
            scale_level="",
            metrics=metrics,
            db_stats=_db_row_counts(db_path),
        )
    finally:
        await backend.close()


async def run_startup_readiness_campaign(db_path: Path) -> CampaignResult:
    """Benchmark startup readiness check speed."""
    from polylogue.storage.query_models import SessionRecordQuery
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

    metrics: dict[str, float] = {}

    t0 = time.monotonic()
    backend = SQLiteBackend(db_path=db_path)
    metrics["backend_init_s"] = round(time.monotonic() - t0, 4)

    try:
        elapsed, count = await _ameasure(backend.queries.count_sessions(SessionRecordQuery()))
        metrics["count_convs_s"] = round(elapsed, 4)
        metrics["session_count"] = count

        elapsed, _ = await _ameasure(backend.get_stats_by("provider"))
        metrics["stats_by_provider_s"] = round(elapsed, 4)

        elapsed, _ = await _ameasure(backend.get_provider_metrics_rows())
        metrics["provider_metrics_s"] = round(elapsed, 4)

        metrics["total_readiness_s"] = round(
            metrics["backend_init_s"]
            + metrics["count_convs_s"]
            + metrics["stats_by_provider_s"]
            + metrics["provider_metrics_s"],
            4,
        )

        return CampaignResult(
            campaign_name="startup-readiness",
            scale_level="",
            metrics=metrics,
            db_stats=_db_row_counts(db_path),
        )
    finally:
        await backend.close()


def run_session_insight_materialization_campaign(db_path: Path) -> CampaignResult:
    """Benchmark full durable session-insight rebuild."""
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
    from polylogue.storage.sqlite.connection import open_connection

    stats_before = _session_insight_table_counts(db_path)

    with open_connection(db_path) as conn:
        elapsed, rebuilt = _measure(rebuild_session_insights_sync, conn)
        conn.commit()

    stats_after = _session_insight_table_counts(db_path)

    return CampaignResult(
        campaign_name="session-insight-materialization",
        scale_level="",
        metrics={
            "rebuild_wall_s": round(elapsed, 3),
            "profiles_rebuilt": rebuilt.profiles,
            "work_events_rebuilt": rebuilt.work_events,
            "phases_rebuilt": rebuilt.phases,
            "threads_rebuilt": rebuilt.threads,
            "tag_rollups_rebuilt": rebuilt.tag_rollups,
        },
        db_stats={
            "session_profiles_before": stats_before.get("session_profiles_count", 0),
            "session_profiles_after": stats_after.get("session_profiles_count", 0),
            "session_work_events_after": stats_after.get("session_work_events_count", 0),
            "session_work_events_fts_after": stats_after.get("session_work_events_fts_count", 0),
            "session_phases_after": stats_after.get("session_phases_count", 0),
            "threads_after": stats_after.get("threads_count", 0),
            "threads_fts_after": stats_after.get("threads_fts_count", 0),
            "session_tag_rollups_after": stats_after.get("session_tag_rollups_count", 0),
        },
    )


async def run_daemon_live_convergence_campaign(db_path: Path) -> CampaignResult:
    """Benchmark in-process daemon live ingestion and convergence."""
    from devtools.daemon_live_benchmark import run_daemon_live_convergence_workload
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
    from polylogue.storage.sqlite.connection import open_connection

    metrics, stats = await run_daemon_live_convergence_workload(db_path)
    with open_connection(db_path) as conn:
        elapsed, rebuilt = _measure(rebuild_session_insights_sync, conn)
        conn.commit()
    metrics.update(
        {
            "session_insight_rebuild_wall_s": round(elapsed, 3),
            "profiles_rebuilt": float(rebuilt.profiles),
            "work_events_rebuilt": float(rebuilt.work_events),
            "phases_rebuilt": float(rebuilt.phases),
            "threads_rebuilt": float(rebuilt.threads),
            "tag_rollups_rebuilt": float(rebuilt.tag_rollups),
        }
    )
    stats.update(_db_row_counts(db_path))
    stats.update(_session_insight_table_counts(db_path))
    return CampaignResult(
        campaign_name="daemon-live-convergence",
        scale_level="",
        metrics=metrics,
        db_stats=stats,
    )


SYNTHETIC_BENCHMARK_RUNNERS: dict[str, SyntheticBenchmarkRunner] = {
    "fts-rebuild": run_fts_rebuild_campaign,
    "incremental-index": run_incremental_index_campaign,
    "filter-scan": run_filter_scan_campaign,
    "startup-readiness": run_startup_readiness_campaign,
    "session-insight-materialization": run_session_insight_materialization_campaign,
    "daemon-live-convergence": run_daemon_live_convergence_campaign,
}


def resolve_synthetic_benchmark_runner(name: str) -> SyntheticBenchmarkRunner:
    try:
        return SYNTHETIC_BENCHMARK_RUNNERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown synthetic benchmark runner {name!r}") from exc


__all__ = [
    "CampaignResult",
    "resolve_synthetic_benchmark_runner",
    "run_daemon_live_convergence_campaign",
    "run_filter_scan_campaign",
    "run_fts_rebuild_campaign",
    "run_incremental_index_campaign",
    "run_session_insight_materialization_campaign",
    "run_startup_readiness_campaign",
    "SYNTHETIC_BENCHMARK_RUNNERS",
    "SyntheticBenchmarkRunner",
]
