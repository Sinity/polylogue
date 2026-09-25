"""Production-backed harness for focused daemon service lifecycle tests.

The harness delegates selection, prerequisite resolution, failure reporting,
and shutdown to the same registry and supervisor used by ``polylogued``.
Tests choose a declared profile; this module owns only test resource cleanup.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Iterable
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

    def start(self, name: str, factory: Callable[[], Coroutine[Any, Any, None]]) -> asyncio.Task[None] | None:
        return self.supervisor.start(name, factory)

    def prerequisite_missing(self, name: str, reason: str) -> None:
        self.supervisor.mark_unavailable(name, reason=reason)

    async def close(self) -> ShutdownReport:
        report = await self.supervisor.shutdown()
        if not report.clean:
            raise AssertionError(f"daemon service harness leaked tasks: {report.as_dict()}")
        return report

    def state(self, name: str) -> ServiceState:
        return self.supervisor.state(name)

    @staticmethod
    def validate_api_bind(*, enabled: bool, host: str, allow_remote: bool, auth_token: str | None) -> None:
        """Apply the production API bind policy without starting archive work."""
        from polylogue.daemon.cli import validate_api_bind_policy

        validate_api_bind_policy(enabled=enabled, host=host, allow_remote=allow_remote, auth_token=auth_token)
