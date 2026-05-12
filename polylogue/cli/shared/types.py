"""CLI types.

Heavy imports (operations, services, storage, ui) are deferred to
``AppEnv._resolve`` so that ``--help`` and tab-completion never pay
the ~2.5 s archive/storage import cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.config import Config
    from polylogue.operations import ArchiveOperations
    from polylogue.services import RuntimeServices
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
    from polylogue.ui import UI


def _lazy_ui() -> UI:
    from polylogue.ui import UI as _UI

    return _UI(plain=True)


def _lazy_services() -> RuntimeServices:
    from polylogue.services import build_runtime_services

    return build_runtime_services()


@dataclass
class AppEnv:
    """CLI application environment.

    Backing services use ``default_factory`` so imports are deferred
    until an instance is created — ``--help`` never pays the cost.
    The real CLI path (``AppEnv(ui=create_ui(...))``) passes explicit
    values, skipping the factories entirely.
    """

    ui: UI = field(default_factory=_lazy_ui)
    services: RuntimeServices = field(default_factory=_lazy_services)

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
        from polylogue.operations import ArchiveOperations

        return ArchiveOperations.from_services(self.services)

    @property
    def polylogue(self) -> Polylogue:
        from polylogue.api import Polylogue

        return Polylogue(archive_root=self.config.archive_root, db_path=self.config.db_path)
