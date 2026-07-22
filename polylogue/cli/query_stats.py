"""Grouped stats output: structured serialization and SQL-backed archive stats.

Origin/date grouping, semantic action/tool grouping, and profile-backed
grouping were removed (polylogue-t46.6) — ``ArchiveStore.stats_by`` (SQL,
exposed via ``archive_query.py``'s ``stats_by`` dispatch / the API's
``get_stats_by``) already owns every grouping dimension these once
re-derived in Python (origin, day/month/year, action, tool, repo,
work-kind), and nothing outside this module's own re-exports/tests called
them — the live ``analyze --by <dimension>`` CLI path always went through
the SQL aggregator.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import click

from polylogue.cli.query_feedback import emit_no_results

if TYPE_CHECKING:
    from polylogue.archive.filter.filters import SessionFilter
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.cli.shared.types import AppEnv
    from polylogue.core.protocols import SessionArchiveStatsStore


# ---------------------------------------------------------------------------
# Structured stats serialization (from query_stats_structured.py)
# ---------------------------------------------------------------------------


def emit_structured_stats(
    *,
    output_format: str,
    dimension: str,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    multi_membership: bool = False,
) -> bool:
    if output_format == "json":
        click.echo(
            json.dumps(
                {
                    "dimension": dimension,
                    "multi_membership": multi_membership,
                    "rows": rows,
                    "summary": summary,
                },
                indent=2,
            )
        )
        return True

    if output_format == "yaml":
        import yaml

        click.echo(
            yaml.dump(
                {
                    "dimension": dimension,
                    "multi_membership": multi_membership,
                    "rows": rows,
                    "summary": summary,
                },
                default_flow_style=False,
                allow_unicode=True,
            )
        )
        return True

    if output_format == "csv":
        import csv
        import io

        buf = io.StringIO()
        fieldnames = list(summary.keys())
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(summary)
        click.echo(buf.getvalue().rstrip("\r\n"))
        return True

    return False


# ---------------------------------------------------------------------------
# SQL-backed archive stats (from query_sql_stats.py)
# ---------------------------------------------------------------------------


async def output_stats_sql(
    env: AppEnv,
    filter_chain: SessionFilter,
    repo: SessionArchiveStatsStore,
    *,
    selection: SessionQuerySpec | None = None,
    output_format: str = "markdown",
) -> None:
    """Output statistics using SQL aggregation without full message loading."""
    described_filters = filter_chain.describe()
    has_filters = bool(described_filters)

    archive_stats = None
    if has_filters:
        summaries = await filter_chain.list_summaries() if filter_chain.can_use_summaries() else None
        if summaries is not None:
            if not summaries:
                emit_no_results(
                    env,
                    selection=selection,
                    output_format=output_format,
                    exit_code=2 if output_format == "json" else None,
                )
                return
            conv_ids = [str(summary.id) for summary in summaries]
            conv_count = len(conv_ids)
        else:
            conv_count = await filter_chain.count()
            if conv_count == 0:
                emit_no_results(
                    env,
                    selection=selection,
                    output_format=output_format,
                    exit_code=2 if output_format == "json" else None,
                )
                return
            conv_ids = None
    else:
        conv_ids = None
        archive_stats = await repo.get_archive_stats()
        conv_count = archive_stats.total_sessions
        if conv_count == 0:
            emit_no_results(
                env,
                output_format=output_format,
                message="No sessions in archive.",
                exit_code=2 if output_format == "json" else None,
            )
            return

    if has_filters:
        stats = await repo.aggregate_message_stats(conv_ids)
    else:
        stats = await repo.aggregate_message_stats()

    date_range = ""
    if stats["min_sort_key"] and stats["max_sort_key"]:
        min_date = datetime.fromtimestamp(stats["min_sort_key"], tz=timezone.utc).strftime("%Y-%m-%d")
        max_date = datetime.fromtimestamp(stats["max_sort_key"], tz=timezone.utc).strftime("%Y-%m-%d")
        date_range = f"{min_date} to {max_date}"

    structured_summary: dict[str, object] = {
        "sessions": conv_count,
        "messages_total": stats["total"],
        "messages_user": stats["user"],
        "messages_assistant": stats["assistant"],
        "words_approx": stats["words_approx"],
        "attachment_refs": stats["attachment_refs"],
        "distinct_attachments": stats["distinct_attachments"],
        "origins": stats["origins"],
        "date_range": None,
        "filtered": has_filters,
    }
    pending_embedding_sessions = 0
    stale_embedding_messages = 0
    if not has_filters:
        assert archive_stats is not None
        pending_embedding_sessions = getattr(archive_stats, "pending_embedding_sessions", 0)
        stale_embedding_messages = getattr(archive_stats, "stale_embedding_messages", 0)
        embeddings_payload = {
            "embedded_sessions": getattr(archive_stats, "embedded_sessions", 0),
            "embedded_messages": getattr(archive_stats, "embedded_messages", 0),
            "pending_embedding_sessions": pending_embedding_sessions,
            "stale_embedding_messages": stale_embedding_messages,
            "messages_missing_embedding_provenance": getattr(
                archive_stats,
                "messages_missing_embedding_provenance",
                0,
            ),
            "embedding_coverage_percent": round(getattr(archive_stats, "embedding_coverage", 0.0), 1),
            "embedding_readiness_status": getattr(archive_stats, "embedding_readiness_status", None),
            "retrieval_ready": getattr(archive_stats, "retrieval_ready", None),
        }
        structured_summary["embeddings"] = embeddings_payload
    if date_range:
        structured_summary["date_range"] = date_range

    if emit_structured_stats(
        output_format=output_format,
        dimension="archive",
        rows=[],
        summary=structured_summary,
    ):
        return

    out = env.ui.console.print
    out(f"\nSessions: {conv_count:,}\n")
    if stats["user"] or stats["assistant"]:
        out(f"Messages: {stats['total']:,} total ({stats['user']:,} user, {stats['assistant']:,} assistant)")
    else:
        out(f"Messages: {stats['total']:,}")

    if stats["words_approx"]:
        out(f"Words: ~{stats['words_approx']:,}")

    raw_origins = stats["origins"]
    origins = raw_origins
    if origins:
        origin_parts = [f"{name} ({count:,})" for name, count in origins.items()]
        out(f"Origins: {', '.join(origin_parts)}")

    out(f"Attachment refs: {stats['attachment_refs']:,}")
    out(f"Unique attachments: {stats['distinct_attachments']:,}")
    if not has_filters:
        assert archive_stats is not None
        embedding_line = (
            f"Embeddings: {archive_stats.embedded_sessions:,}/{archive_stats.total_sessions:,} convs, "
            f"{archive_stats.embedded_messages:,} msgs ({archive_stats.embedding_coverage:.1f}%)"
        )
        if pending_embedding_sessions:
            embedding_line += f", pending {pending_embedding_sessions:,}"
        if stale_embedding_messages:
            embedding_line += f", stale {stale_embedding_messages:,}"
        out(embedding_line)
    if date_range:
        out(f"Date range: {date_range}")


__all__ = [
    "emit_structured_stats",
    "output_stats_sql",
]
