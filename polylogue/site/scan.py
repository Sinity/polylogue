"""Archive scan helpers for static-site generation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import IO, TYPE_CHECKING

from polylogue.site.models import ArchiveIndexStats, ConversationIndex, ConversationPageBuildStats, SiteConfig
from polylogue.site.search import build_search_document

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository


async def iter_conversation_indexes(
    *,
    repository: ConversationRepository,
    backend: SQLiteBackend,
    page_size: int,
    provider: str | None = None,
) -> AsyncIterator[ConversationIndex]:
    """Yield lightweight conversation indexes in repository sort order."""
    async for summaries in repository.iter_summary_pages(
        page_size=page_size,
        provider=provider,
    ):
        message_counts = await backend.queries.get_message_counts_batch([str(summary.id) for summary in summaries])
        for summary in summaries:
            yield ConversationIndex.from_summary(
                summary,
                message_counts.get(str(summary.id), 0),
            )


async def scan_archive(
    *,
    output_dir: Path,
    config: SiteConfig,
    conversation_iter: Callable[[], AsyncIterator[ConversationIndex]],
    generate_conversation_page: Callable[[ConversationIndex, bool], Awaitable[str]],
    incremental: bool,
    progress_callback: Callable[[int, str | None], None] | None = None,
) -> tuple[ArchiveIndexStats, ConversationPageBuildStats]:
    """Scan archive summaries once and drive streaming site outputs from that pass."""
    stats = ArchiveIndexStats()
    page_stats = ConversationPageBuildStats()
    search_path = output_dir / "search-index.json"
    search_handle: IO[str] | None = None
    wrote_search_entry = False

    if config.enable_search and config.search_provider != "pagefind":
        search_handle = search_path.open("w", encoding="utf-8")
        search_handle.write("[")

    try:
        if progress_callback is not None:
            progress_callback(0, desc="Building site: scanning archive 0")
        async for conversation in conversation_iter():
            stats.record(
                conversation,
                root_page_size=config.conversations_per_page,
            )
            if search_handle is not None:
                if wrote_search_entry:
                    search_handle.write(",")
                json.dump(build_search_document(conversation), search_handle)
                wrote_search_entry = True
            page_stats.record(
                await generate_conversation_page(
                    conversation,
                    incremental,
                )
            )
            if progress_callback is not None:
                progress_callback(1, desc=f"Building site: scanning archive {stats.total_conversations:,}")
    except Exception:
        if search_handle is not None:
            search_handle.close()
            search_path.unlink(missing_ok=True)
            search_handle = None
        raise
    finally:
        if search_handle is not None:
            search_handle.write("]")
            search_handle.close()

    return stats, page_stats
