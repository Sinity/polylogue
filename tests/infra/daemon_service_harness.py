"""Production-backed harness for focused daemon service lifecycle tests.

The harness delegates selection, prerequisite resolution, failure reporting,
and shutdown to the same registry and supervisor used by ``polylogued``.
Tests choose a declared profile; this module owns only test resource cleanup.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Iterable
from pathlib import Path
from typing import Any

from polylogue.daemon.services import (
    PRODUCTION_PROFILE,
    ServiceCapability,
    ServiceProfile,
    ServiceState,
)
from polylogue.daemon.supervisor import DaemonSupervisor, ShutdownReport


class ServiceHarness:
    """Run declared supervisor nodes with bounded, deterministic teardown."""

    def __init__(
        self,
        *,
        profile: ServiceProfile = PRODUCTION_PROFILE,
        capabilities: Iterable[ServiceCapability] = (),
    ) -> None:
        self.supervisor = DaemonSupervisor(profile=profile, capabilities=capabilities)

    @property
    def selected_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.supervisor.selected)

    def require_selected(self, name: str, *, selected: bool = True) -> None:
        """Assert that the production registry made the requested selection."""
        present = name in self.selected_names
        if present is not selected:
            expectation = "selected" if selected else "excluded"
            raise AssertionError(
                f"production profile must have {name!r} {expectation}; selected={self.selected_names!r}"
            )

    def start(self, name: str, factory: Callable[[], Coroutine[Any, Any, None]]) -> asyncio.Task[None] | None:
        return self.supervisor.start(name, factory)

    def prerequisite_missing(self, name: str, reason: str) -> None:
        self.resolve_prerequisite(name, available=False, reason=reason)

    def resolve_prerequisite(self, name: str, *, available: bool, reason: str | None = None) -> None:
        """Resolve one selected service's start prerequisite.

        This follows the production supervisor contract: an unavailable
        prerequisite settles the declared service as unavailable and
        publishes that reason instead of starting a retrying background task.
        """
        self.require_selected(name)
        if available:
            return
        if not reason:
            raise ValueError(f"unavailable prerequisite for {name!r} needs an attributable reason")
        self.supervisor.mark_unavailable(name, reason=reason)

    def api_server(self, archive_root: Path, *, write_bridge: Any = None) -> Any:
        """Construct a production HTTP server for the selected API service.

        Passing a bridge models the borrowed-runtime composition used by
        ``polylogued``; omitting it lets the server construct its standalone
        owned runtime. Both paths use the real server initializer.
        """
        self.require_selected("api_server")
        from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

        kwargs: dict[str, Any] = {"archive_root": archive_root}
        if write_bridge is not None:
            kwargs["write_bridge"] = write_bridge
        return DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler, **kwargs)

    async def close(self) -> ShutdownReport:
        report = await self.supervisor.shutdown()
        if not report.clean:
            raise AssertionError(f"daemon service harness leaked tasks: {report.as_dict()}")
        return report

    def state(self, name: str) -> ServiceState:
        return self.supervisor.state(name)

    def validate_api_bind(self, *, enabled: bool, host: str, allow_remote: bool, auth_token: str | None) -> None:
        """Apply the production API bind policy without starting archive work."""
        from polylogue.daemon.cli import validate_api_bind_policy

        self.require_selected("api_server", selected=enabled)
        validate_api_bind_policy(enabled=enabled, host=host, allow_remote=allow_remote, auth_token=auth_token)
