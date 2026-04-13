"""Long-haul benchmark campaign runner.

Executes reproducible benchmark campaigns against synthetic archives
and produces durable JSON + Markdown reports under .local/benchmark-campaigns/.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from polylogue.scenarios import ScenarioMetadata


@dataclass
class CampaignResult:
    """Result of a single benchmark campaign run."""

    campaign_name: str
    scale_level: str
    metrics: dict[str, float] = field(default_factory=dict)
    db_stats: dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    origin: str = "authored"
    artifact_targets: list[str] = field(default_factory=list)
    operation_targets: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SyntheticBenchmarkScenario(ScenarioMetadata):
    """Authored scenario metadata for synthetic long-haul benchmark campaigns."""

    scenario_id: str
    description: str
    summary_metric: str
    summary_label: str


SYNTHETIC_BENCHMARK_SCENARIOS: tuple[SyntheticBenchmarkScenario, ...] = (
    SyntheticBenchmarkScenario(
        scenario_id="fts-rebuild",
        description="Benchmark full FTS5 index rebuild",
        summary_metric="rebuild_wall_s",
        summary_label="s",
        origin="authored.synthetic-benchmark",
        artifact_targets=("message_fts",),
        operation_targets=("index.message-fts-rebuild",),
        tags=("benchmark", "synthetic", "fts"),
    ),
    SyntheticBenchmarkScenario(
        scenario_id="incremental-index",
        description="Benchmark incremental FTS index updates",
        summary_metric="total_wall_s",
        summary_label="s",
        origin="authored.synthetic-benchmark",
        artifact_targets=("message_fts",),
        operation_targets=("index.message-fts-incremental",),
        tags=("benchmark", "synthetic", "fts"),
    ),
    SyntheticBenchmarkScenario(
        scenario_id="filter-scan",
        description="Benchmark common filter query patterns",
        summary_metric="list_50_wall_s",
        summary_label="s",
        origin="authored.synthetic-benchmark",
        artifact_targets=("conversation_query_results",),
        operation_targets=("query.filters.synthetic-scan",),
        tags=("benchmark", "synthetic", "filters"),
    ),
    SyntheticBenchmarkScenario(
        scenario_id="startup-health",
        description="Benchmark check --runtime startup speed",
        summary_metric="total_health_s",
        summary_label="s",
        origin="authored.synthetic-benchmark",
        artifact_targets=("archive_health",),
        operation_targets=("health.startup.synthetic",),
        tags=("benchmark", "synthetic", "health"),
    ),
    SyntheticBenchmarkScenario(
        scenario_id="action-event-materialization",
        description="Benchmark action-event read-model rebuild over synthetic tool-use transcripts",
        summary_metric="rebuild_wall_s",
        summary_label="s",
        origin="authored.synthetic-benchmark",
        artifact_targets=("tool_use_source_blocks", "action_event_rows", "action_event_fts"),
        operation_targets=("materialize-action-events",),
        tags=("benchmark", "synthetic", "action-events"),
    ),
    SyntheticBenchmarkScenario(
        scenario_id="session-product-materialization",
        description="Benchmark durable session-product rebuild over synthetic archive conversations",
        summary_metric="rebuild_wall_s",
        summary_label="s",
        origin="authored.synthetic-benchmark",
        artifact_targets=("session_product_source_conversations", "session_product_rows", "session_product_fts"),
        operation_targets=("materialize-session-products",),
        tags=("benchmark", "synthetic", "session-products"),
    ),
)

SYNTHETIC_BENCHMARK_REGISTRY: dict[str, SyntheticBenchmarkScenario] = {
    scenario.scenario_id: scenario for scenario in SYNTHETIC_BENCHMARK_SCENARIOS
}


def _db_row_counts(db_path: Path) -> dict[str, int]:
    """Collect row counts and file size for a database."""
    from polylogue.storage.backends.connection import open_connection

    stats: dict[str, int] = {}
    if not db_path.exists():
        return stats
    stats["db_size_bytes"] = db_path.stat().st_size
    with open_connection(db_path) as conn:
        for table in ("conversations", "messages", "content_blocks", "raw_conversations", "action_events"):
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                stats[f"{table}_count"] = row[0] if row else 0
            except Exception:
                stats[f"{table}_count"] = 0
        # FTS row count
        try:
            row = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()
            stats["fts_rows"] = row[0] if row else 0
        except Exception:
            stats["fts_rows"] = 0
        try:
            row = conn.execute("SELECT COUNT(*) FROM action_events_fts").fetchone()
            stats["action_fts_rows"] = row[0] if row else 0
        except Exception:
            stats["action_fts_rows"] = 0
    return stats


def _measure(fn, *args, **kwargs) -> tuple[float, object]:
    """Call fn(*args, **kwargs), return (elapsed_seconds, result)."""
    t0 = time.monotonic()
    result = fn(*args, **kwargs)
    return time.monotonic() - t0, result


async def _ameasure(coro) -> tuple[float, object]:
    """Await coro, return (elapsed_seconds, result)."""
    t0 = time.monotonic()
    result = await coro
    return time.monotonic() - t0, result


def _session_product_table_counts(db_path: Path) -> dict[str, int]:
    """Collect row counts for durable session-product tables and FTS projections."""
    from polylogue.storage.backends.connection import open_connection

    stats: dict[str, int] = {}
    if not db_path.exists():
        return stats
    with open_connection(db_path) as conn:
        for table in (
            "session_profiles",
            "session_profiles_fts",
            "session_work_events",
            "session_work_events_fts",
            "session_phases",
            "work_threads",
            "work_threads_fts",
            "session_tag_rollups",
            "day_session_summaries",
            "week_session_summaries",
        ):
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                stats[f"{table}_count"] = row[0] if row else 0
            except Exception:
                stats[f"{table}_count"] = 0
    return stats


def run_fts_rebuild_campaign(db_path: Path) -> CampaignResult:
    """Benchmark full FTS index rebuild.

    Drops and rebuilds the entire FTS5 index, measuring wall time.
    """
    from polylogue.storage.backends.connection import open_connection
    from polylogue.storage.index import rebuild_index

    stats_before = _db_row_counts(db_path)

    with open_connection(db_path) as conn:
        # Drop existing FTS data
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
        metrics={
            "rebuild_wall_s": round(elapsed, 3),
        },
        db_stats={
            "messages_count": stats_before.get("messages_count", 0),
            "fts_rows_before": 0,
            "fts_rows_after": stats_after.get("fts_rows", 0),
            "db_size_bytes": stats_after.get("db_size_bytes", 0),
        },
    )


async def run_incremental_index_campaign(db_path: Path, batch_size: int = 100) -> CampaignResult:
    """Benchmark incremental FTS index updates.

    Indexes conversations in batches, measuring per-batch and total time.
    """
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.query_models import ConversationRecordQuery

    backend = SQLiteBackend(db_path=db_path)
    try:
        total_convs = await backend.queries.count_conversations(ConversationRecordQuery())
        batch_times: list[float] = []
        indexed_total = 0

        from polylogue.storage.search_providers.fts5 import FTS5Provider

        fts = FTS5Provider(db_path=db_path)

        # Process in batches
        offset = 0
        while offset < total_convs:
            convs = await backend.queries.list_conversations(ConversationRecordQuery(limit=batch_size, offset=offset))
            if not convs:
                break

            conv_ids = [c.conversation_id for c in convs]
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
    """Benchmark common filter query patterns.

    Runs a series of representative filter queries and measures latency.
    """
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.query_models import ConversationRecordQuery

    backend = SQLiteBackend(db_path=db_path)
    metrics: dict[str, float] = {}

    try:
        # 1. List all (no filter)
        elapsed, results = await _ameasure(backend.queries.list_conversations(ConversationRecordQuery(limit=50)))
        metrics["list_50_wall_s"] = round(elapsed, 4)
        metrics["list_50_count"] = len(results)

        # 2. Filter by provider
        elapsed, results = await _ameasure(
            backend.queries.list_conversations(ConversationRecordQuery(provider="chatgpt", limit=50))
        )
        metrics["filter_provider_wall_s"] = round(elapsed, 4)
        metrics["filter_provider_count"] = len(results)

        # 3. Filter by has_tool_use
        elapsed, results = await _ameasure(
            backend.queries.list_conversations(ConversationRecordQuery(has_tool_use=True, limit=50))
        )
        metrics["filter_tool_use_wall_s"] = round(elapsed, 4)
        metrics["filter_tool_use_count"] = len(results)

        # 4. Filter by has_thinking
        elapsed, results = await _ameasure(
            backend.queries.list_conversations(ConversationRecordQuery(has_thinking=True, limit=50))
        )
        metrics["filter_thinking_wall_s"] = round(elapsed, 4)
        metrics["filter_thinking_count"] = len(results)

        # 5. Count conversations
        elapsed, count = await _ameasure(backend.queries.count_conversations(ConversationRecordQuery()))
        metrics["count_all_wall_s"] = round(elapsed, 4)
        metrics["count_all"] = count

        # 6. Title search
        elapsed, results = await _ameasure(
            backend.queries.list_conversations(ConversationRecordQuery(title_contains="synthetic", limit=50))
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


async def run_startup_health_campaign(db_path: Path) -> CampaignResult:
    """Benchmark startup health check speed.

    Measures the time to open a backend, count conversations, get stats,
    and fetch the latest run — the operations that `polylogue check --runtime`
    performs on startup.
    """
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.query_models import ConversationRecordQuery

    metrics: dict[str, float] = {}

    # Backend creation + schema check
    t0 = time.monotonic()
    backend = SQLiteBackend(db_path=db_path)
    metrics["backend_init_s"] = round(time.monotonic() - t0, 4)

    try:
        # Count conversations
        elapsed, count = await _ameasure(backend.queries.count_conversations(ConversationRecordQuery()))
        metrics["count_convs_s"] = round(elapsed, 4)
        metrics["conversation_count"] = count

        # Get stats by provider
        elapsed, stats = await _ameasure(backend.get_stats_by("provider"))
        metrics["stats_by_provider_s"] = round(elapsed, 4)

        # Get provider metrics rows
        elapsed, rows = await _ameasure(backend.get_provider_metrics_rows())
        metrics["provider_metrics_s"] = round(elapsed, 4)

        # Get latest run
        elapsed, run = await _ameasure(backend.queries.get_latest_run())
        metrics["latest_run_s"] = round(elapsed, 4)

        # Total health check time
        metrics["total_health_s"] = round(
            metrics["backend_init_s"]
            + metrics["count_convs_s"]
            + metrics["stats_by_provider_s"]
            + metrics["provider_metrics_s"]
            + metrics["latest_run_s"],
            4,
        )

        return CampaignResult(
            campaign_name="startup-health",
            scale_level="",
            metrics=metrics,
            db_stats=_db_row_counts(db_path),
        )
    finally:
        await backend.close()


def run_action_event_materialization_campaign(db_path: Path) -> CampaignResult:
    """Benchmark full action-event read-model rebuild.

    Clears the durable action-event rows, then rebuilds them from persisted
    tool-use source blocks. The action-event FTS projection is trigger-maintained
    during rebuild and measured as part of the same operation.
    """
    from polylogue.storage.action_event_rebuild_runtime import rebuild_action_event_read_model_sync
    from polylogue.storage.backends.connection import open_connection

    stats_before = _db_row_counts(db_path)

    with open_connection(db_path) as conn:
        try:
            conn.execute("DELETE FROM action_events")
            conn.commit()
        except Exception:
            pass

        elapsed, rebuilt_rows = _measure(rebuild_action_event_read_model_sync, conn)
        conn.commit()

    stats_after = _db_row_counts(db_path)

    return CampaignResult(
        campaign_name="action-event-materialization",
        scale_level="",
        metrics={
            "rebuild_wall_s": round(elapsed, 3),
            "action_event_rows_rebuilt": int(rebuilt_rows),
        },
        db_stats={
            "action_events_before": stats_before.get("action_events_count", 0),
            "action_events_after": stats_after.get("action_events_count", 0),
            "action_fts_rows_after": stats_after.get("action_fts_rows", 0),
            "db_size_bytes": stats_after.get("db_size_bytes", 0),
        },
    )


def run_session_product_materialization_campaign(db_path: Path) -> CampaignResult:
    """Benchmark full durable session-product rebuild.

    Clears the durable session-product family, then rebuilds rows and FTS-backed
    projections from persisted archive conversations.
    """
    from polylogue.storage.backends.connection import open_connection
    from polylogue.storage.session_product_rebuild import rebuild_session_products_sync

    stats_before = _session_product_table_counts(db_path)

    with open_connection(db_path) as conn:
        elapsed, rebuilt = _measure(rebuild_session_products_sync, conn)
        conn.commit()

    stats_after = _session_product_table_counts(db_path)

    return CampaignResult(
        campaign_name="session-product-materialization",
        scale_level="",
        metrics={
            "rebuild_wall_s": round(elapsed, 3),
            "profiles_rebuilt": int(rebuilt["profiles"]),
            "work_events_rebuilt": int(rebuilt["work_events"]),
            "phases_rebuilt": int(rebuilt["phases"]),
            "threads_rebuilt": int(rebuilt["threads"]),
            "tag_rollups_rebuilt": int(rebuilt["tag_rollups"]),
            "day_summaries_rebuilt": int(rebuilt["day_summaries"]),
        },
        db_stats={
            "session_profiles_before": stats_before.get("session_profiles_count", 0),
            "session_profiles_after": stats_after.get("session_profiles_count", 0),
            "session_profiles_fts_after": stats_after.get("session_profiles_fts_count", 0),
            "session_work_events_after": stats_after.get("session_work_events_count", 0),
            "session_work_events_fts_after": stats_after.get("session_work_events_fts_count", 0),
            "session_phases_after": stats_after.get("session_phases_count", 0),
            "work_threads_after": stats_after.get("work_threads_count", 0),
            "work_threads_fts_after": stats_after.get("work_threads_fts_count", 0),
            "session_tag_rollups_after": stats_after.get("session_tag_rollups_count", 0),
            "day_session_summaries_after": stats_after.get("day_session_summaries_count", 0),
            "week_session_summaries_after": stats_after.get("week_session_summaries_count", 0),
        },
    )


async def run_synthetic_benchmark_campaign(name: str, db_path: Path) -> CampaignResult:
    """Dispatch one synthetic benchmark campaign by authored scenario id."""

    scenario = SYNTHETIC_BENCHMARK_REGISTRY[name]
    match name:
        case "fts-rebuild":
            result = run_fts_rebuild_campaign(db_path)
        case "incremental-index":
            result = await run_incremental_index_campaign(db_path)
        case "filter-scan":
            result = await run_filter_scan_campaign(db_path)
        case "startup-health":
            result = await run_startup_health_campaign(db_path)
        case "action-event-materialization":
            result = run_action_event_materialization_campaign(db_path)
        case "session-product-materialization":
            result = run_session_product_materialization_campaign(db_path)
        case _:
            raise KeyError(name)
    result.origin = scenario.origin
    result.artifact_targets = list(scenario.artifact_targets)
    result.operation_targets = list(scenario.operation_targets)
    result.tags = list(scenario.tags)
    return result


async def run_full_campaign(scale_level: str, output_dir: Path) -> list[CampaignResult]:
    """Run all benchmark campaigns at a given scale level.

    Generates a synthetic archive at the specified scale, then runs
    each campaign against the resulting database.

    Args:
        scale_level: One of "small", "medium", "large", "stretch".
        output_dir: Directory for archive and report output.

    Returns:
        List of CampaignResult for all campaigns.
    """
    from devtools.large_archive_generator import (
        ScaleLevel,
        generate_archive,
        get_default_spec,
    )

    level = ScaleLevel(scale_level)
    spec = get_default_spec(level)

    archive_dir = output_dir / f"archive-{scale_level}"
    print(f"Generating {scale_level} archive ({spec.conversations} conversations, ~{spec.message_count} messages)...")
    archive_metrics = await generate_archive(spec, archive_dir)
    print(
        f"Archive generated in {archive_metrics.wall_time_s:.1f}s "
        f"({archive_metrics.conversation_count} convs, "
        f"{archive_metrics.message_count} msgs, "
        f"{archive_metrics.db_size_bytes / 1024 / 1024:.1f} MB)"
    )

    db_path = archive_dir / "benchmark.db"
    results: list[CampaignResult] = []

    for scenario in SYNTHETIC_BENCHMARK_SCENARIOS:
        print(f"Running {scenario.scenario_id} campaign...")
        result = await run_synthetic_benchmark_campaign(scenario.scenario_id, db_path)
        result.scale_level = scale_level
        results.append(result)
        metric_value = result.metrics.get(scenario.summary_metric, 0)
        print(f"  -> {metric_value:.4f}{scenario.summary_label}")

    return results


CAMPAIGN_REGISTRY: dict[str, str] = {
    scenario.scenario_id: scenario.description for scenario in SYNTHETIC_BENCHMARK_SCENARIOS
}


__all__ = [
    "CAMPAIGN_REGISTRY",
    "CampaignResult",
    "SYNTHETIC_BENCHMARK_REGISTRY",
    "SYNTHETIC_BENCHMARK_SCENARIOS",
    "SyntheticBenchmarkScenario",
    "run_filter_scan_campaign",
    "run_fts_rebuild_campaign",
    "run_action_event_materialization_campaign",
    "run_full_campaign",
    "run_incremental_index_campaign",
    "run_session_product_materialization_campaign",
    "run_synthetic_benchmark_campaign",
    "run_startup_health_campaign",
]
