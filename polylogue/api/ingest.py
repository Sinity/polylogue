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

    async def rebuild_index(self) -> bool:
        """Rebuild the derived block-FTS index through the mutation executor."""
        from polylogue.config import active_archive_root as _active_archive_root
        from polylogue.operations.bindings import runtime_operation_binding
        from polylogue.operations.mutation_actuators import IndexRebuildActuator, IndexRebuildArgs
        from polylogue.operations.mutation_transaction import MutationPrincipal, OperationExecutor
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            actuator = IndexRebuildActuator()
            args = IndexRebuildArgs(archive=archive)
            root = _active_archive_root(self.config)
            executor = OperationExecutor.for_archive_root(root)
            binding = runtime_operation_binding(actuator)
            principal = MutationPrincipal(
                "facade", frozenset({"archive.rebuild_index", "archive.legacy_runtime"}), "api", "write"
            )
            preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=root)
            authorization = executor.authorize_bound(binding, preview, principal)
            receipt = executor.execute_bound(binding, preview, authorization, args)
        return receipt.status in {"applied", "already_satisfied"}
