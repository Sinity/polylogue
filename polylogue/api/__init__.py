"""High-level async library facade for Polylogue."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.api.archive import PolylogueArchiveMixin
from polylogue.api.embeddings import PolylogueEmbeddingsMixin
from polylogue.api.ingest import PolylogueIngestMixin
from polylogue.api.insights import PolylogueInsightsMixin
from polylogue.config import Config, ResolvedRuntimeConfig, resolve_runtime_config
from polylogue.operations import ArchiveStats
from polylogue.services import build_runtime_services
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

if TYPE_CHECKING:
    from polylogue.storage.embeddings.materialization import PendingSession


def select_pending_embedding_session_window(
    conn: sqlite3.Connection,
    *,
    session_ids: list[str] | tuple[str, ...] | None = None,
    rebuild: bool = False,
    max_sessions: int | None = None,
    max_messages: int | None = None,
) -> list[PendingSession]:
    """Return a bounded pending embedding window for public surface adapters."""

    from polylogue.storage.embeddings.materialization import select_pending_session_window

    return select_pending_session_window(
        conn,
        session_ids=session_ids,
        rebuild=rebuild,
        max_sessions=max_sessions,
        max_messages=max_messages,
    )


class Polylogue(PolylogueArchiveMixin, PolylogueEmbeddingsMixin, PolylogueInsightsMixin, PolylogueIngestMixin):
    """High-level async facade for the Polylogue library."""

    def __init__(
        self,
        archive_root: str | Path | None = None,
        db_path: str | Path | None = None,
        *,
        runtime: ResolvedRuntimeConfig | None = None,
    ) -> None:
        explicit_archive = Path(archive_root).expanduser().resolve() if archive_root is not None else None
        explicit_db = Path(db_path).expanduser().resolve() if db_path is not None else None

        if runtime is not None and (explicit_archive is not None or explicit_db is not None):
            if explicit_archive is not None and explicit_archive != runtime.paths.archive_root:
                raise ValueError("explicit archive_root conflicts with resolved runtime")
            if explicit_db is not None and explicit_db != runtime.paths.index_db:
                raise ValueError("explicit db_path conflicts with resolved runtime")

        if runtime is None and explicit_archive is None and explicit_db is None:
            runtime = resolve_runtime_config()

        self._runtime = runtime
        if runtime is not None:
            self._config = runtime.as_config()
            self._services = build_runtime_services(runtime=runtime)
            return

        resolved_archive = explicit_archive or (explicit_db.parent if explicit_db is not None else None)
        if resolved_archive is None:
            raise ValueError("explicit API construction requires archive_root or db_path")
        resolved_db = explicit_db or resolved_archive / "index.db"
        self._config = Config(
            archive_root=resolved_archive,
            render_root=resolved_archive / "render",
            sources=[],
            db_path=resolved_db,
        )
        self._services = build_runtime_services(config=self._config, db_path=resolved_db)

    @classmethod
    def open(
        cls,
        *,
        config: Config | None = None,
        runtime: ResolvedRuntimeConfig | None = None,
        **kwargs: object,
    ) -> Polylogue:
        if runtime is not None:
            return cls(runtime=runtime)
        archive_root: str | Path | None = config.archive_root if config else kwargs.get("archive_root")  # type: ignore[assignment]
        db_path: str | Path | None = config.db_path if config else kwargs.get("db_path")  # type: ignore[assignment]
        return cls(archive_root=archive_root, db_path=db_path)

    async def __aenter__(self) -> Polylogue:
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._services.close()

    @property
    def config(self) -> Config:
        return self._config

    @property
    def runtime(self) -> ResolvedRuntimeConfig | None:
        return self._runtime

    @property
    def archive_root(self) -> Path:
        return self._config.archive_root

    @property
    def backend(self) -> SQLiteBackend:
        return self._services.get_backend()

    @property
    def repository(self) -> SessionRepository:
        return self._services.get_repository()

    def __repr__(self) -> str:
        return f"Polylogue(archive_root={self._config.archive_root!r})"


__all__ = ["ArchiveStats", "Polylogue", "select_pending_embedding_session_window"]
