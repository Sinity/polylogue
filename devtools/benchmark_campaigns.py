"""Long-haul benchmark campaign runner.

Executes reproducible benchmark campaigns against synthetic archives
and produces durable JSON + Markdown reports under .local/benchmark-campaigns/.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class CampaignResult:
    """Result of a single benchmark campaign run."""

    campaign_name: str
    scale_level: str
    metrics: dict[str, float] = field(default_factory=dict)
    db_stats: dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def _db_row_counts(db_path: Path) -> dict[str, int]:
    """Collect row counts and file size for a database."""
    from polylogue.storage.backends.connection import open_connection

    stats: dict[str, int] = {}
    if not db_path.exists():
        return stats
    stats["db_size_bytes"] = db_path.stat().st_size
    with open_connection(db_path) as conn:
        for table in ("conversations", "messages", "content_blocks", "raw_conversations"):
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

    # FTS rebuild
    print("Running fts-rebuild campaign...")
    result = run_fts_rebuild_campaign(db_path)
    result.scale_level = scale_level
    results.append(result)
    print(f"  -> {result.metrics.get('rebuild_wall_s', 0):.3f}s")

    # Incremental index
    print("Running incremental-index campaign...")
    result = await run_incremental_index_campaign(db_path)
    result.scale_level = scale_level
    results.append(result)
    print(f"  -> {result.metrics.get('total_wall_s', 0):.3f}s")

    # Filter scan
    print("Running filter-scan campaign...")
    result = await run_filter_scan_campaign(db_path)
    result.scale_level = scale_level
    results.append(result)
    print(f"  -> list_50: {result.metrics.get('list_50_wall_s', 0):.4f}s")

    # Startup health
    print("Running startup-health campaign...")
    result = await run_startup_health_campaign(db_path)
    result.scale_level = scale_level
    results.append(result)
    print(f"  -> total: {result.metrics.get('total_health_s', 0):.4f}s")

    return results


CAMPAIGN_REGISTRY: dict[str, str] = {
    "fts-rebuild": "Benchmark full FTS5 index rebuild",
    "incremental-index": "Benchmark incremental FTS index updates",
    "filter-scan": "Benchmark common filter query patterns",
    "startup-health": "Benchmark check --runtime startup speed",
}


__all__ = [
    "CAMPAIGN_REGISTRY",
    "CampaignResult",
    "run_filter_scan_campaign",
    "run_fts_rebuild_campaign",
    "run_full_campaign",
    "run_incremental_index_campaign",
    "run_startup_health_campaign",
]
