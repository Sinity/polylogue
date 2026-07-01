"""CLI types.

Heavy imports (operations, services, storage, ui) are deferred to
``AppEnv._resolve`` so that ``--help`` and tab-completion never pay
the ~2.5 s archive/storage import cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.config import Config
    from polylogue.services import RuntimeServices
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
    from polylogue.ui import UI


def _lazy_ui() -> UI:
    from polylogue.ui import UI as _UI

    return _UI(plain=True)


def _lazy_services() -> RuntimeServices:
    from polylogue.services import build_runtime_services

    return build_runtime_services()


class AppEnv:
    """CLI application environment.

    Backing UI and services are created lazily on first access so metadata-only
    commands such as ``read --views`` do not pay for UI, storage, or archive
    setup. Tests and command paths can still inject explicit objects with
    ``AppEnv(ui=...)`` and ``AppEnv(services=...)``.
    """

    def __init__(
        self,
        *,
        ui: UI | None = None,
        services: RuntimeServices | None = None,
        plain: bool = True,
    ) -> None:
        self._ui = ui
        self._services = services
        self._plain = plain

    @property
    def ui(self) -> UI:
        if self._ui is None:
            from polylogue.ui import create_ui

            self._ui = create_ui(self._plain)
        return self._ui

    @ui.setter
    def ui(self, value: UI) -> None:
        self._ui = value

    @property
    def services(self) -> RuntimeServices:
        if self._services is None:
            self._services = _lazy_services()
        return self._services

    @services.setter
    def services(self, value: RuntimeServices) -> None:
        self._services = value

    @property
    def config(self) -> Config:
        return self.services.get_config()

    @property
    def backend(self) -> SQLiteBackend:
        return self.services.get_backend()

    @property
    def repository(self) -> SessionRepository:
        return self.services.get_repository()

    @property
    def polylogue(self) -> Polylogue:
        from polylogue.api import Polylogue

        return Polylogue(archive_root=self.config.archive_root, db_path=self.config.db_path)
