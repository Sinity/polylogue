"""CLI types.

Heavy imports (operations, services, storage, ui) are deferred to
``AppEnv._resolve`` so that ``--help`` and tab-completion never pay
the ~2.5 s archive/storage import cost.
"""

from __future__ import annotations

import sys
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.config import Config, ResolvedRuntimeConfig
    from polylogue.services import RuntimeServices
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
    from polylogue.ui import UI


def _lazy_ui() -> UI:
    from polylogue.ui import UI as _UI

    return _UI(plain=True)


def _lazy_services(runtime: ResolvedRuntimeConfig | None) -> RuntimeServices:
    from polylogue.services import build_runtime_services

    return build_runtime_services(runtime=runtime)


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
        runtime: ResolvedRuntimeConfig | None = None,
        plain: bool = True,
        debug_timing: bool = False,
    ) -> None:
        self._ui = ui
        self._services = services
        self._runtime = runtime
        self._plain = plain
        self._debug_timing = debug_timing
        self._active_timings: dict[str, float] = {}
        self._timings: dict[str, float] = {}

    @property
    def debug_timing(self) -> bool:
        """Whether this invocation should emit its phase timing table."""
        return self._debug_timing

    def begin_timing(self, phase: str) -> None:
        """Start a named monotonic timing phase when debugging is enabled."""
        if self._debug_timing:
            self._active_timings[phase] = perf_counter()

    def finish_timing(self, phase: str) -> None:
        """Finish a previously started phase without affecting normal output."""
        started_at = self._active_timings.pop(phase, None)
        if started_at is not None:
            self._timings[phase] = (perf_counter() - started_at) * 1000

    def record_timing(self, phase: str, started_at: float) -> None:
        """Record an externally scoped monotonic timing phase."""
        if self._debug_timing:
            self._timings[phase] = (perf_counter() - started_at) * 1000

    def emit_debug_timings(self) -> None:
        """Write the opt-in phase table to stderr after command output."""
        if not self._debug_timing or not self._timings:
            return
        rows = "\n".join(f"{phase:<12} {elapsed_ms:8.1f} ms" for phase, elapsed_ms in self._timings.items())
        sys.stderr.write(f"polylogue timing\n{rows}\n")

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
            self._services = _lazy_services(self._runtime)
        return self._services

    @services.setter
    def services(self, value: RuntimeServices) -> None:
        self._services = value

    @property
    def runtime(self) -> ResolvedRuntimeConfig:
        return self.services.get_runtime()

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

        try:
            return Polylogue(runtime=self.runtime)
        except Exception:
            return Polylogue(archive_root=self.config.archive_root, db_path=self.config.db_path)
