"""High-level async library facade for Polylogue."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.api.archive import PolylogueArchiveMixin
from polylogue.api.embeddings import PolylogueEmbeddingsMixin
from polylogue.api.ingest import PolylogueIngestMixin
from polylogue.api.insights import PolylogueInsightsMixin
from polylogue.config import Config
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
    ):
        if archive_root is not None:
            archive_root = Path(archive_root).expanduser().resolve()
        if db_path is not None:
            db_path = Path(db_path).expanduser().resolve()

        from polylogue.paths import archive_root as _archive_root

        explicit_archive_root = archive_root is not None
        if archive_root is None:
            archive_root = _archive_root()
        if db_path is None and explicit_archive_root:
            db_path = archive_root / "index.db"

        if db_path is not None:
            self._config = Config(
                archive_root=archive_root,
                render_root=archive_root / "render",
                sources=[],
                db_path=db_path,
            )
        else:
            self._config = Config(
                archive_root=archive_root,
                render_root=archive_root / "render",
                sources=[],
            )
        self._services = build_runtime_services(config=self._config, db_path=db_path)

    @classmethod
    def open(cls, *, config: Config | None = None, **kwargs: object) -> Polylogue:
        archive_root: str | Path | None = config.archive_root if config else kwargs.get("archive_root")  # type: ignore[assignment]
        db_path: str | Path | None = kwargs.get("db_path")  # type: ignore[assignment]
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
