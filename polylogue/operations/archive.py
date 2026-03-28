"""Archive operations shared across facade, CLI, and MCP call sites."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.archive_products import ArchiveDebtProduct, ProviderAnalyticsProduct
from polylogue.operations.archive_product_support import ArchiveProductMixin
from polylogue.operations.archive_search_support import ArchiveSearchMixin
from polylogue.operations.archive_stats_support import ArchiveStats, ArchiveStatsMixin
from polylogue.services import RuntimeServices, build_runtime_services

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

class ArchiveOperations(ArchiveSearchMixin, ArchiveStatsMixin, ArchiveProductMixin):
    """Canonical archive-level operations over configured runtime dependencies."""

    def __init__(
        self,
        *,
        services: RuntimeServices | None = None,
        config: Config | None = None,
        repository: ConversationRepository | None = None,
        backend: SQLiteBackend | None = None,
    ) -> None:
        self._services = services
        self._config = config
        self._repository = repository
        self._backend = backend

    @classmethod
    def from_services(cls, services: RuntimeServices) -> ArchiveOperations:
        return cls(services=services)

    @property
    def config(self) -> Config:
        if self._config is None:
            if self._services is None:
                raise RuntimeError("ArchiveOperations requires config or runtime services")
            self._config = self._services.get_config()
        return self._config

    @property
    def repository(self) -> ConversationRepository:
        if self._repository is None:
            if self._services is None:
                raise RuntimeError("ArchiveOperations requires repository or runtime services")
            self._repository = self._services.get_repository()
        return self._repository

    @property
    def backend(self) -> SQLiteBackend:
        if self._backend is None:
            if self._services is not None:
                self._backend = self._services.get_backend()
            else:
                self._backend = self.repository.backend
        return self._backend

async def _with_operations(
    action,
    *,
    services: RuntimeServices | None = None,
    db_path: Path | None = None,
):
    owns_services = services is None
    runtime_services = services or build_runtime_services(db_path=db_path)
    operations = ArchiveOperations.from_services(runtime_services)
    try:
        return await action(operations)
    finally:
        if owns_services:
            await runtime_services.close()


async def get_provider_counts(
    *,
    services: RuntimeServices | None = None,
    db_path: Path | None = None,
) -> list[tuple[str, int]]:
    """Return (provider, conversation_count) pairs for archive summaries."""

    async def _action(operations: ArchiveOperations) -> list[tuple[str, int]]:
        return await operations.provider_counts()

    return await _with_operations(_action, services=services, db_path=db_path)


async def list_provider_analytics_products(
    *,
    services: RuntimeServices | None = None,
    db_path: Path | None = None,
) -> list[ProviderAnalyticsProduct]:
    """Return provider-level analytics products for archive summaries."""

    async def _action(operations: ArchiveOperations) -> list[ProviderAnalyticsProduct]:
        return await operations.list_provider_analytics_products()

    return await _with_operations(_action, services=services, db_path=db_path)


__all__ = [
    "ArchiveDebtProduct",
    "ArchiveOperations",
    "ArchiveStats",
    "get_provider_counts",
    "list_provider_analytics_products",
]
