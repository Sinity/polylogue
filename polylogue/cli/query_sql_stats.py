"""SQL-backed archive stats output helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.filters import ConversationFilter
    from polylogue.storage.repository import ConversationRepository


async def output_stats_sql(
    env: AppEnv,
    filter_chain: ConversationFilter,
    repo: ConversationRepository,
) -> None:
    """Output statistics using SQL aggregation without full message loading."""
    has_filters = bool(filter_chain.describe())

    if has_filters:
        summaries = await filter_chain.list_summaries() if filter_chain.can_use_summaries() else None
        if summaries is not None:
            if not summaries:
                env.ui.console.print("No conversations matched.")
                return
            conv_ids = [str(summary.id) for summary in summaries]
            conv_count = len(conv_ids)
        else:
            conv_count = await filter_chain.count()
            if conv_count == 0:
                env.ui.console.print("No conversations matched.")
                return
            conv_ids = None
    else:
        conv_ids = None
        conv_count = await filter_chain.count()
        if conv_count == 0:
            env.ui.console.print("No conversations in archive.")
            return

    stats = await repo.queries.aggregate_message_stats(conv_ids)

    date_range = ""
    if stats["min_sort_key"] and stats["max_sort_key"]:
        min_date = datetime.fromtimestamp(stats["min_sort_key"], tz=timezone.utc).strftime("%Y-%m-%d")
        max_date = datetime.fromtimestamp(stats["max_sort_key"], tz=timezone.utc).strftime("%Y-%m-%d")
        date_range = f"{min_date} to {max_date}"

    out = env.ui.console.print
    out(f"\nConversations: {conv_count:,}\n")
    if stats["user"] or stats["assistant"]:
        out(
            f"Messages: {stats['total']:,} total "
            f"({stats['user']:,} user, {stats['assistant']:,} assistant)"
        )
    else:
        out(f"Messages: {stats['total']:,}")

    if stats["words_approx"]:
        out(f"Words: ~{stats['words_approx']:,}")

    if stats.get("providers"):
        provider_parts = [f"{name} ({count:,})" for name, count in stats["providers"].items()]
        out(f"Providers: {', '.join(provider_parts)}")

    out(f"Attachments: {stats['attachments']:,}")
    if not has_filters:
        archive_stats = await repo.get_archive_stats()
        pending_embedding_conversations = getattr(archive_stats, "pending_embedding_conversations", 0)
        stale_embedding_messages = getattr(archive_stats, "stale_embedding_messages", 0)
        embedding_line = (
            f"Embeddings: {archive_stats.embedded_conversations:,}/{archive_stats.total_conversations:,} convs, "
            f"{archive_stats.embedded_messages:,} msgs ({archive_stats.embedding_coverage:.1f}%)"
        )
        if pending_embedding_conversations:
            embedding_line += f", pending {pending_embedding_conversations:,}"
        if stale_embedding_messages:
            embedding_line += f", stale {stale_embedding_messages:,}"
        out(embedding_line)
    if date_range:
        out(f"Date range: {date_range}")


__all__ = ["output_stats_sql"]
