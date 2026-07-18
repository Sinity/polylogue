"""Explicit runtime service scope for config/backend/repository access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config, ConfigError, ResolvedRuntimeConfig
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite import SQLiteBackend, create_backend


@dataclass
class RuntimeServices:
    """Invocation-scoped runtime dependencies.

    Composition roots normally supply ``runtime``.  Tests and explicit library
    callers may instead supply a fully constructed legacy ``Config``; this
    compatibility path never performs ambient configuration resolution.
    """

    runtime: ResolvedRuntimeConfig | None = None
    config: Config | None = None
    backend: SQLiteBackend | None = None
    repository: SessionRepository | None = None
    db_path: Path | None = None

    def __post_init__(self) -> None:
        if self.runtime is not None:
            projected = self.runtime.as_config()
            if self.config is None:
                self.config = projected
            elif self.config.archive_root != projected.archive_root or self.config.db_path != projected.db_path:
                raise ValueError("RuntimeServices config must project the supplied runtime authority")
            if self.db_path is None:
                self.db_path = self.runtime.paths.index_db
        elif self.config is not None and self.db_path is None:
            self.db_path = self.config.db_path

        if self.repository is not None:
            repo_backend = self.repository.backend
            if self.backend is None:
                self.backend = repo_backend
            elif self.backend is not repo_backend:
                raise ValueError("RuntimeServices backend and repository.backend must match")

    def get_runtime(self) -> ResolvedRuntimeConfig:
        if self.runtime is None:
            raise ConfigError("RuntimeServices has no ResolvedRuntimeConfig; resolve at the composition root")
        return self.runtime

    def get_config(self) -> Config:
        if self.config is None:
            raise ConfigError("RuntimeServices has no config projection; resolve at the composition root")
        return self.config

    def get_backend(self) -> SQLiteBackend:
        if self.backend is None:
            if self.db_path is None:
                raise ConfigError("RuntimeServices has no resolved database path")
            self.backend = create_backend(self.db_path)
        return self.backend

    def get_repository(self) -> SessionRepository:
        if self.repository is None:
            self.repository = SessionRepository(
                backend=self.get_backend(),
                archive_root=self.get_config().archive_root,
            )
        return self.repository

    async def close(self) -> None:
        if self.repository is not None:
            await self.repository.close()
        elif self.backend is not None:
            await self.backend.close()


def build_runtime_services(
    *,
    runtime: ResolvedRuntimeConfig | None = None,
    config: Config | None = None,
    backend: SQLiteBackend | None = None,
    repository: SessionRepository | None = None,
    db_path: Path | None = None,
) -> RuntimeServices:
    """Build an explicit runtime service scope without ambient fallbacks."""
    return RuntimeServices(
        runtime=runtime,
        config=config,
        backend=backend,
        repository=repository,
        db_path=db_path,
    )


__all__ = [
    "RuntimeServices",
    "build_runtime_services",
]
