"""Explicit runtime service scope for config/backend/repository access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config, get_config
from polylogue.storage.backends import SQLiteBackend, create_backend
from polylogue.storage.repository import ConversationRepository


@dataclass
class RuntimeServices:
    """Invocation-scoped runtime dependencies.

    This replaces the old ambient singleton service locator. Each CLI invocation,
    server instance, or test can carry its own runtime services explicitly.
    """

    config: Config | None = None
    backend: SQLiteBackend | None = None
    repository: ConversationRepository | None = None
    db_path: Path | None = None

    def __post_init__(self) -> None:
        if self.repository is not None:
            repo_backend = self.repository.backend
            if self.backend is None:
                self.backend = repo_backend
            elif self.backend is not repo_backend:
                raise ValueError("RuntimeServices backend and repository.backend must match")

    def get_config(self) -> Config:
        if self.config is None:
            self.config = get_config()
        return self.config

    def get_backend(self) -> SQLiteBackend:
        if self.backend is None:
            self.backend = create_backend(self.db_path)
        return self.backend

    def get_repository(self) -> ConversationRepository:
        if self.repository is None:
            self.repository = ConversationRepository(backend=self.get_backend())
        return self.repository

    async def close(self) -> None:
        if self.repository is not None:
            await self.repository.close()
        elif self.backend is not None:
            await self.backend.close()


def build_runtime_services(
    *,
    config: Config | None = None,
    backend: SQLiteBackend | None = None,
    repository: ConversationRepository | None = None,
    db_path: Path | None = None,
) -> RuntimeServices:
    """Build an explicit runtime service scope."""
    return RuntimeServices(
        config=config,
        backend=backend,
        repository=repository,
        db_path=db_path,
    )


__all__ = [
    "RuntimeServices",
    "build_runtime_services",
]
