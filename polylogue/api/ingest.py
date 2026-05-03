"""Ingest and index domain methods for the async Polylogue facade."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import Source

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.pipeline.services.parsing_models import ParseResult
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


class PolylogueIngestMixin:
    if TYPE_CHECKING:

        @property
        def config(self) -> Config: ...

        @property
        def backend(self) -> SQLiteBackend: ...

        @property
        def repository(self) -> ConversationRepository: ...

    async def parse_file(
        self,
        path: str | Path,
        *,
        source_name: str | None = None,
    ) -> ParseResult:
        from polylogue.pipeline.services.parsing import ParsingService

        parsing_service = ParsingService(
            repository=self.repository,
            archive_root=self.config.archive_root,
            config=self.config,
        )

        file_path = Path(path).expanduser().resolve()
        if source_name is None:
            source_name = file_path.stem

        source = Source(name=source_name, path=file_path)
        return await parsing_service.parse_sources(
            sources=[source],
            ui=None,
            download_assets=False,
        )

    async def parse_sources(
        self,
        sources: list[Source] | None = None,
        *,
        download_assets: bool = True,
    ) -> ParseResult:
        from polylogue.pipeline.services.parsing import ParsingService

        parsing_service = ParsingService(
            repository=self.repository,
            archive_root=self.config.archive_root,
            config=self.config,
        )

        if sources is None:
            sources = self.config.sources

        return await parsing_service.parse_sources(
            sources=sources,
            ui=None,
            download_assets=download_assets,
        )

    async def rebuild_index(self) -> bool:
        from polylogue.pipeline.services.indexing import IndexService

        index_service = IndexService(config=self.config, backend=self.backend)
        return await index_service.rebuild_index()
