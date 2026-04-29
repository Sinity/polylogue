"""High-level async library facade for Polylogue."""

from __future__ import annotations

from pathlib import Path

from polylogue.api.archive import PolylogueArchiveMixin
from polylogue.api.ingest import PolylogueIngestMixin
from polylogue.api.products import PolylogueProductsMixin
from polylogue.config import Config
from polylogue.operations import ArchiveOperations, ArchiveStats
from polylogue.services import build_runtime_services
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository


class Polylogue(PolylogueArchiveMixin, PolylogueProductsMixin, PolylogueIngestMixin):
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

        if archive_root is None:
            archive_root = _archive_root()

        self._config = Config(
            archive_root=archive_root,
            render_root=archive_root / "render",
            sources=[],
        )
        self._services = build_runtime_services(config=self._config, db_path=db_path)
        self._operations = ArchiveOperations.from_services(self._services)

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
    def repository(self) -> ConversationRepository:
        return self._services.get_repository()

    @property
    def operations(self) -> ArchiveOperations:
        return self._operations

    def __repr__(self) -> str:
        return f"Polylogue(archive_root={self._config.archive_root!r})"


__all__ = ["ArchiveStats", "Polylogue"]
