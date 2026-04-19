"""Archive scan helpers for static-site generation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from polylogue.site.models import ArchiveIndexStats, ConversationIndex, ConversationPageBuildStats, SiteConfig
from polylogue.site.search import SearchIndexWriter, build_search_document

if TYPE_CHECKING:
    from polylogue.protocols import ProgressCallback
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository


@runtime_checkable
class _MessageCountQueries(Protocol):
    async def get_message_counts_batch(self, conversation_ids: list[str]) -> dict[str, int]: ...


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
        summary_page = list(summaries)
        message_counts = await _load_message_counts(
            backend,
            [str(summary.id) for summary in summary_page],
        )
        for summary in summary_page:
            yield ConversationIndex.from_summary(
                summary,
                message_counts.get(str(summary.id), 0),
            )


async def _load_message_counts(
    backend: SQLiteBackend,
    conversation_ids: list[str],
) -> dict[str, int]:
    queries = getattr(backend, "queries", None)
    if isinstance(queries, _MessageCountQueries):
        return await queries.get_message_counts_batch(conversation_ids)
    return await backend.get_message_counts_batch(conversation_ids)


async def scan_archive(
    *,
    output_dir: Path,
    config: SiteConfig,
    conversation_iter: Callable[[], AsyncIterator[ConversationIndex]],
    generate_conversation_page: Callable[[ConversationIndex, bool], Awaitable[str]],
    incremental: bool,
    progress_callback: ProgressCallback | None = None,
) -> tuple[ArchiveIndexStats, ConversationPageBuildStats]:
    """Scan archive summaries once and drive streaming site outputs from that pass."""
    stats = ArchiveIndexStats()
    page_stats = ConversationPageBuildStats()
    search_writer = SearchIndexWriter(output_dir, config)
    search_writer.open()

    try:
        if progress_callback is not None:
            progress_callback(0, desc="Building site: scanning archive 0")
        async for conversation in conversation_iter():
            stats.record(
                conversation,
                root_page_size=config.conversations_per_page,
            )
            search_writer.append(build_search_document(conversation))
            page_stats.record(
                await generate_conversation_page(
                    conversation,
                    incremental,
                )
            )
            if progress_callback is not None:
                progress_callback(1, desc=f"Building site: scanning archive {stats.total_conversations:,}")
    except Exception:
        search_writer.abort()
        raise
    finally:
        search_writer.finish()

    return stats, page_stats
