"""Static site builder for polylogue archives."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.publication import SitePublicationManifest
from polylogue.rendering.renderers.html import HTMLMessageRenderer, PygmentsHighlighter
from polylogue.site.models import ArchiveIndexStats, ConversationIndex, ConversationPageBuildStats, SiteConfig
from polylogue.site.publication_flow import (
    build_site_publication_manifest,
    load_archive_maintenance_summary_for_backend,
    load_artifact_proof_summary_for_backend,
    load_latest_run_summary,
    record_site_publication_manifest,
    write_site_publication_manifest,
)
from polylogue.site.search import SearchBuildStatus
from polylogue.site.site_builder_archive import (
    generate_conversation_page_for_builder,
    iter_conversation_indexes_for_builder,
    scan_archive_for_builder,
)
from polylogue.site.site_builder_pages import (
    generate_dashboard_for_builder,
    generate_pagefind_config_for_builder,
    generate_provider_indexes_for_builder,
    generate_root_index_for_builder,
    search_markup_for_builder,
    write_template_stream_for_builder,
)
from polylogue.site.template_environment import build_template_environments

if TYPE_CHECKING:
    from jinja2 import Template

    from polylogue.protocols import ProgressCallback
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository


class SiteBuilder:
    """Build a static HTML site from a polylogue archive."""

    SUMMARY_PAGE_SIZE = 500

    def __init__(
        self,
        output_dir: Path,
        config: SiteConfig | None = None,
        *,
        backend: SQLiteBackend | None = None,
        repository: ConversationRepository | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or SiteConfig()
        if repository is not None and backend is None:
            backend = repository.backend
        self._backend = backend
        self._repository = repository
        self._owns_storage = backend is None and repository is None
        self._progress_callback = progress_callback

        self._highlighter = PygmentsHighlighter()
        self._message_renderer = HTMLMessageRenderer(self._highlighter)
        self.env, self.page_env = build_template_environments(self._highlighter)

    def _open_storage(self) -> tuple[SQLiteBackend, ConversationRepository]:
        """Return the canonical storage pair used by site generation."""
        if self._backend is None:
            from polylogue.storage.backends.async_sqlite import SQLiteBackend

            self._backend = SQLiteBackend()
        if self._repository is None:
            from polylogue.storage.repository import ConversationRepository

            self._repository = ConversationRepository(backend=self._backend)
        return self._backend, self._repository

    def build(self, incremental: bool = True) -> SitePublicationManifest:
        """Build the complete static site and return the typed manifest."""
        return run_coroutine_sync(self._build_async(incremental))

    async def _build_async(self, incremental: bool = True) -> SitePublicationManifest:
        """Async implementation of the site build."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        build_started = perf_counter()
        try:
            generated_display = datetime.now().strftime("%Y-%m-%d %H:%M")
            generated_at = datetime.now(timezone.utc).isoformat()
            if self._progress_callback is not None:
                self._progress_callback(0, desc="Building site: preparing output")
            archive_stats, conversation_pages = await self._scan_archive(incremental=incremental)

            if self._progress_callback is not None:
                self._progress_callback(0, desc="Building site: writing indexes")
            await self._generate_root_index(archive_stats, generated_at=generated_display)
            provider_index_pages = await self._generate_provider_indexes(
                archive_stats,
                generated_at=generated_display,
            )

            dashboard_pages = 0
            if self.config.include_dashboard:
                if self._progress_callback is not None:
                    self._progress_callback(0, desc="Building site: writing dashboard")
                await self._generate_dashboard(
                    archive_stats,
                    generated_at=generated_display,
                )
                dashboard_pages = 1

            search_status = SearchBuildStatus.DISABLED
            if self.config.uses_pagefind:
                if self._progress_callback is not None:
                    self._progress_callback(0, desc="Building site: writing pagefind config")
                search_status = await asyncio.to_thread(self._generate_pagefind_config)
            elif self.config.enable_search:
                search_status = SearchBuildStatus.JSON_INDEX_WRITTEN

            duration_ms = int((perf_counter() - build_started) * 1000)
            backend, repository = self._open_storage()
            if self._progress_callback is not None:
                self._progress_callback(0, desc="Building site: writing manifest")
            manifest = await build_site_publication_manifest(
                output_dir=self.output_dir,
                config=self.config,
                archive_stats=archive_stats,
                conversation_pages=conversation_pages,
                generated_at=generated_at,
                duration_ms=duration_ms,
                provider_index_pages=provider_index_pages,
                dashboard_pages=dashboard_pages,
                search_status=str(search_status),
                incremental=incremental,
                latest_run=await load_latest_run_summary(backend),
                artifact_proof=await load_artifact_proof_summary_for_backend(backend),
                maintenance=await load_archive_maintenance_summary_for_backend(backend),
            )
            write_site_publication_manifest(self.output_dir, manifest)
            if self._progress_callback is not None:
                self._progress_callback(0, desc="Building site: recording publication")
            await record_site_publication_manifest(repository, manifest)
            return manifest
        finally:
            if self._owns_storage and self._backend is not None:
                await self._backend.close()
                self._backend = None
                self._repository = None

    async def _iter_conversation_indexes(
        self,
        *,
        provider: str | None = None,
        backend: SQLiteBackend | None = None,
        repository: ConversationRepository | None = None,
    ) -> AsyncIterator[ConversationIndex]:
        async for conversation in iter_conversation_indexes_for_builder(
            self,
            provider=provider,
            backend=backend,
            repository=repository,
        ):
            yield conversation

    async def _scan_archive(self, *, incremental: bool) -> tuple[ArchiveIndexStats, ConversationPageBuildStats]:
        return await scan_archive_for_builder(self, incremental=incremental)

    async def _generate_conversation_page(
        self,
        repository: ConversationRepository,
        conversation: ConversationIndex,
        *,
        incremental: bool = True,
    ) -> str:
        return await generate_conversation_page_for_builder(
            self,
            repository,
            conversation,
            incremental=incremental,
        )

    async def _write_template_stream(self, template: Template, output_path: Path, **context: object) -> None:
        await write_template_stream_for_builder(self, template, output_path, **context)

    async def _generate_root_index(self, archive_stats: ArchiveIndexStats, *, generated_at: str) -> None:
        await generate_root_index_for_builder(self, archive_stats, generated_at=generated_at)

    async def _generate_provider_indexes(self, archive_stats: ArchiveIndexStats, *, generated_at: str) -> int:
        return await generate_provider_indexes_for_builder(self, archive_stats, generated_at=generated_at)

    async def _generate_dashboard(self, archive_stats: ArchiveIndexStats, *, generated_at: str) -> None:
        await generate_dashboard_for_builder(self, archive_stats, generated_at=generated_at)

    def _search_markup(self) -> str:
        return search_markup_for_builder(self)

    def _generate_pagefind_config(self) -> SearchBuildStatus:
        return generate_pagefind_config_for_builder(self)


__all__ = [
    "ArchiveIndexStats",
    "ConversationIndex",
    "ConversationPageBuildStats",
    "SiteBuilder",
    "SiteConfig",
]
