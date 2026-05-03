"""CLI types."""

from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.config import Config
from polylogue.operations import ArchiveOperations
from polylogue.services import RuntimeServices, build_runtime_services
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.ui import UI


@dataclass
class AppEnv:
    """CLI application environment."""

    ui: UI
    services: RuntimeServices = field(default_factory=build_runtime_services)

    @property
    def config(self) -> Config:
        return self.services.get_config()

    @property
    def backend(self) -> SQLiteBackend:
        return self.services.get_backend()

    @property
    def repository(self) -> ConversationRepository:
        return self.services.get_repository()

    @property
    def operations(self) -> ArchiveOperations:
        return ArchiveOperations.from_services(self.services)
