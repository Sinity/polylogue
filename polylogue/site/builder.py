"""Static site builder for polylogue archives."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from polylogue.publication import (
    ArchivePublicationSummary,
    OutputManifest,
    SiteOutputSummary,
    SitePublicationManifest,
)
from polylogue.rendering.renderers.html import MarkdownRenderer, PygmentsHighlighter
from polylogue.site.models import (
    ArchiveIndexStats,
    ConversationIndex,
    ConversationPageBuildStats,
    SiteConfig,
)
from polylogue.site.publication_support import (
    build_latest_run_summary,
    load_artifact_proof_summary,
    load_semantic_proof_summary,
)
from polylogue.site.rendering import (
    build_template_environments,
    generate_conversation_page,
    generate_dashboard,
    generate_provider_indexes,
    generate_root_index,
    write_template_stream,
)
from polylogue.site.scan import iter_conversation_indexes, scan_archive
from polylogue.site.search import build_search_document, generate_pagefind_config, render_search_markup
from polylogue.storage.store import PublicationRecord

if TYPE_CHECKING:
    from jinja2 import Template

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
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or SiteConfig()
        if repository is not None and backend is None:
            backend = repository.backend
        self._backend = backend
        self._repository = repository
        self._owns_storage = backend is None and repository is None

        self._highlighter = PygmentsHighlighter()
        self._md_renderer = MarkdownRenderer(self._highlighter)
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
            archive_stats, conversation_pages = await self._scan_archive(
                incremental=incremental
            )

            await self._generate_root_index(archive_stats, generated_at=generated_display)
            provider_index_pages = await self._generate_provider_indexes(
                archive_stats,
                generated_at=generated_display,
            )

            dashboard_pages = 0
            if self.config.include_dashboard:
                await self._generate_dashboard(
                    archive_stats,
                    generated_at=generated_display,
                )
                dashboard_pages = 1

            search_status = "disabled"
            if self.config.enable_search and self.config.search_provider == "pagefind":
                search_status = await asyncio.to_thread(self._generate_pagefind_config)
            elif self.config.enable_search:
                search_status = "json_index_written"

            proof_summary = await self._artifact_proof_summary()
            semantic_summary = await self._semantic_proof_summary()
            latest_run = await self._latest_run_summary()
            artifact_manifest = await asyncio.to_thread(
                OutputManifest.scan,
                self.output_dir,
                include_hashes=True,
                exclude_paths={"site-manifest.json"},
            )
            duration_ms = int((perf_counter() - build_started) * 1000)
            manifest = SitePublicationManifest(
                publication_id=f"site-{uuid4().hex[:16]}",
                generated_at=generated_at,
                output_dir=str(self.output_dir),
                duration_ms=duration_ms,
                config={
                    "title": self.config.title,
                    "description": self.config.description,
                    "enable_search": self.config.enable_search,
                    "search_provider": str(self.config.search_provider),
                    "conversations_per_page": self.config.conversations_per_page,
                    "include_dashboard": self.config.include_dashboard,
                },
                archive=ArchivePublicationSummary(
                    total_conversations=archive_stats.total_conversations,
                    total_messages=archive_stats.total_messages,
                    provider_count=len(archive_stats.provider_counts),
                    provider_counts=dict(sorted(archive_stats.provider_counts.items())),
                    provider_messages=dict(sorted(archive_stats.provider_messages.items())),
                ),
                outputs=SiteOutputSummary(
                    root_index_pages=1,
                    provider_index_pages=provider_index_pages,
                    dashboard_pages=dashboard_pages,
                    total_index_pages=1 + provider_index_pages + dashboard_pages,
                    total_conversation_pages=conversation_pages.total,
                    rendered_conversation_pages=conversation_pages.rendered,
                    reused_conversation_pages=conversation_pages.reused,
                    failed_conversation_pages=conversation_pages.failed,
                    search_documents=(
                        archive_stats.total_conversations if self.config.enable_search else 0
                    ),
                    search_enabled=self.config.enable_search,
                    search_provider=(
                        str(self.config.search_provider) if self.config.enable_search else None
                    ),
                    search_status=search_status,
                    incremental=incremental,
                ),
                latest_run=latest_run,
                artifact_proof=proof_summary,
                semantic_proof=semantic_summary,
                artifacts=artifact_manifest,
            )
            manifest_path = self.output_dir / "site-manifest.json"
            manifest_path.write_text(
                json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _, repository = self._open_storage()
            await repository.record_publication(
                PublicationRecord(
                    publication_id=manifest.publication_id,
                    publication_kind=manifest.publication_kind,
                    generated_at=manifest.generated_at,
                    output_dir=manifest.output_dir,
                    duration_ms=manifest.duration_ms,
                    manifest=manifest.model_dump(mode="json"),
                )
            )
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
        """Yield lightweight conversation indexes in repository sort order."""
        if backend is None or repository is None:
            backend, repository = self._open_storage()
        async for conversation in iter_conversation_indexes(
            repository=repository,
            backend=backend,
            page_size=self.SUMMARY_PAGE_SIZE,
            provider=provider,
        ):
            yield conversation

    def _search_document(self, conversation: ConversationIndex) -> dict[str, str]:
        return build_search_document(conversation)

    async def _scan_archive(
        self,
        *,
        incremental: bool,
    ) -> tuple[ArchiveIndexStats, ConversationPageBuildStats]:
        backend, repository = self._open_storage()
        return await scan_archive(
            output_dir=self.output_dir,
            config=self.config,
            conversation_iter=lambda: self._iter_conversation_indexes(
                backend=backend,
                repository=repository,
            ),
            generate_conversation_page=lambda conversation, rebuild: self._generate_conversation_page(
                repository,
                conversation,
                incremental=rebuild,
            ),
            incremental=incremental,
        )

    async def _generate_conversation_page(
        self,
        repository: ConversationRepository,
        conversation: ConversationIndex,
        *,
        incremental: bool = True,
    ) -> str:
        return await generate_conversation_page(
            output_dir=self.output_dir,
            page_env=self.page_env,
            repository=repository,
            conversation=conversation,
            render_html=self._md_renderer.render,
            incremental=incremental,
        )

    async def _write_template_stream(
        self,
        template: Template,
        output_path: Path,
        **context: object,
    ) -> None:
        await write_template_stream(template, output_path, **context)

    async def _generate_root_index(
        self,
        archive_stats: ArchiveIndexStats,
        *,
        generated_at: str,
    ) -> None:
        await generate_root_index(
            output_dir=self.output_dir,
            env=self.env,
            config=self.config,
            archive_stats=archive_stats,
            generated_at=generated_at,
            write_stream=self._write_template_stream,
        )

    async def _generate_provider_indexes(
        self,
        archive_stats: ArchiveIndexStats,
        *,
        generated_at: str,
    ) -> int:
        return await generate_provider_indexes(
            output_dir=self.output_dir,
            env=self.env,
            config=self.config,
            archive_stats=archive_stats,
            generated_at=generated_at,
            conversation_iter_factory=lambda provider=None: self._iter_conversation_indexes(provider=provider),
            write_stream=self._write_template_stream,
        )

    async def _generate_dashboard(
        self,
        archive_stats: ArchiveIndexStats,
        *,
        generated_at: str,
    ) -> None:
        await generate_dashboard(
            output_dir=self.output_dir,
            env=self.env,
            config=self.config,
            archive_stats=archive_stats,
            generated_at=generated_at,
            write_stream=self._write_template_stream,
        )

    async def _latest_run_summary(self):
        """Return the latest pipeline run summary for manifest embedding."""
        backend, _repository = self._open_storage()
        return build_latest_run_summary(await backend.get_latest_run())

    async def _artifact_proof_summary(self):
        """Return durable artifact-proof summary for manifest embedding."""
        backend, _repository = self._open_storage()
        if not isinstance(getattr(backend, "db_path", None), Path):
            return None
        return await asyncio.to_thread(load_artifact_proof_summary, db_path=backend.db_path)

    async def _semantic_proof_summary(self):
        """Return semantic-preservation proof summary for manifest embedding."""
        backend, _repository = self._open_storage()
        if not isinstance(getattr(backend, "db_path", None), Path):
            return None
        return await asyncio.to_thread(
            load_semantic_proof_summary,
            db_path=backend.db_path,
        )

    def _search_markup(self) -> str:
        return render_search_markup(self.config)

    def _generate_pagefind_config(self) -> str:
        return generate_pagefind_config(self.output_dir)


__all__ = [
    "ArchiveIndexStats",
    "ConversationIndex",
    "ConversationPageBuildStats",
    "SiteBuilder",
    "SiteConfig",
]
