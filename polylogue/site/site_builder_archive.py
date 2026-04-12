"""Archive scan helpers for the site builder."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from polylogue.site.conversation_pages import generate_conversation_page
from polylogue.site.scan import iter_conversation_indexes, scan_archive
from polylogue.site.search import build_search_document

if TYPE_CHECKING:
    from polylogue.site.builder import SiteBuilder
    from polylogue.site.models import ArchiveIndexStats, ConversationIndex, ConversationPageBuildStats
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository


async def iter_conversation_indexes_for_builder(
    builder: SiteBuilder,
    *,
    provider: str | None = None,
    backend: SQLiteBackend | None = None,
    repository: ConversationRepository | None = None,
) -> AsyncIterator[ConversationIndex]:
    """Yield lightweight conversation indexes in repository sort order."""
    if backend is None or repository is None:
        backend, repository = builder._open_storage()
    async for conversation in iter_conversation_indexes(
        repository=repository,
        backend=backend,
        page_size=builder.SUMMARY_PAGE_SIZE,
        provider=provider,
    ):
        yield conversation


def search_document_for_builder(builder: SiteBuilder, conversation: ConversationIndex) -> dict[str, str]:
    del builder
    return build_search_document(conversation)


async def scan_archive_for_builder(
    builder: SiteBuilder,
    *,
    incremental: bool,
) -> tuple[ArchiveIndexStats, ConversationPageBuildStats]:
    backend, repository = builder._open_storage()
    return await scan_archive(
        output_dir=builder.output_dir,
        config=builder.config,
        conversation_iter=lambda: iter_conversation_indexes_for_builder(
            builder,
            backend=backend,
            repository=repository,
        ),
        generate_conversation_page=lambda conversation, rebuild: generate_conversation_page_for_builder(
            builder,
            repository,
            conversation,
            incremental=rebuild,
        ),
        incremental=incremental,
        progress_callback=builder._progress_callback,
    )


async def generate_conversation_page_for_builder(
    builder: SiteBuilder,
    repository: ConversationRepository,
    conversation: ConversationIndex,
    *,
    incremental: bool = True,
) -> str:
    return await generate_conversation_page(
        output_dir=builder.output_dir,
        page_env=builder.page_env,
        repository=repository,
        conversation=conversation,
        render_html=builder._message_renderer.render,
        incremental=incremental,
    )


__all__ = [
    "generate_conversation_page_for_builder",
    "iter_conversation_indexes_for_builder",
    "scan_archive_for_builder",
    "search_document_for_builder",
]
