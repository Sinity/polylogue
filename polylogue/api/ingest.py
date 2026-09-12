"""Ingest and index domain methods for the async Polylogue facade."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import Source

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.pipeline.services.parsing_models import ParseResult
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


class PolylogueIngestMixin:
    if TYPE_CHECKING:

        @property
        def config(self) -> Config: ...

        @property
        def backend(self) -> SQLiteBackend: ...

        @property
        def repository(self) -> SessionRepository: ...

    async def parse_file(
        self,
        path: str | Path,
        *,
        source_name: str | None = None,
    ) -> ParseResult:
        file_path = Path(path).expanduser().resolve()
        if source_name is None:
            source_name = file_path.stem

        source = Source(name=source_name, path=file_path)
        from polylogue.config import active_archive_root as _active_archive_root
        from polylogue.pipeline.services.archive_ingest import parse_sources_archive

        return await parse_sources_archive(_active_archive_root(self.config), [source])

    async def parse_sources(
        self,
        sources: list[Source] | None = None,
        *,
        download_assets: bool = True,
    ) -> ParseResult:
        if sources is None:
            sources = self.config.sources

        del download_assets
        from polylogue.config import active_archive_root as _active_archive_root
        from polylogue.pipeline.services.archive_ingest import parse_sources_archive

        return await parse_sources_archive(_active_archive_root(self.config), sources)
