"""Production-backed harness for focused daemon service lifecycle tests.

The harness delegates selection, prerequisite resolution, failure reporting,
and shutdown to the same registry and supervisor used by ``polylogued``.
Tests choose a declared profile; this module owns only test resource cleanup.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
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


def record_private_lifecycle_probe(name: str, payload: dict[str, object]) -> None:
    """Optionally retain synthetic lifecycle measures outside tracked content."""
    target = os.environ.get("POLYLOGUE_LIFECYCLE_RECEIPT_DIR")
    if not target:
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name in {".", ".."}:
        raise ValueError("lifecycle receipt name must be a simple filename stem")
    root = Path(target)
    allowed_root = Path("/realm/tmp/work").resolve()
    resolved_root = root.resolve()
    if not root.is_absolute() or not resolved_root.is_relative_to(allowed_root):
        raise ValueError("lifecycle receipt directory must be under /realm/tmp/work")
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / f"{name}.json"
    if not receipt.resolve().is_relative_to(resolved_root):
        raise ValueError("lifecycle receipt path must remain under its receipt directory")
    receipt.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class ServiceHarness:
    """Run declared supervisor nodes with bounded, deterministic teardown."""

    def __init__(
        self,
        *,
        profile: ServiceProfile = PRODUCTION_PROFILE,
        capabilities: Iterable[ServiceCapability] = (),
    ) -> None:
        self.supervisor = DaemonSupervisor(profile=profile, capabilities=capabilities)
        self._servers: list[Any] = []
        self._serving: list[tuple[str, Any, asyncio.Task[None]]] = []

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
        server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler, **kwargs)
        self._servers.append(server)
        return server

    def uds_server(self, archive_root: Path, *, api_server: Any, auth_token: str | None = None) -> Any:
        """Construct the production machine listener beside its API owner.

        The listener borrows the API server's writer bridge, compute kernel,
        and operation runtime exactly as daemon composition does. Reversed
        cleanup closes this borrowed listener before its owning HTTP server.
        """
        self.require_selected("uds_server")
        if api_server not in self._servers:
            raise ValueError("UDS server requires an API server owned by this harness")
        from polylogue.daemon.uds import DaemonAPIUnixHTTPServer, daemon_socket_path

        server = DaemonAPIUnixHTTPServer(
            daemon_socket_path(archive_root),
            archive_root=archive_root,
            auth_token=auth_token,
            write_bridge=api_server.write_bridge,
            execution_kernel=api_server.execution_kernel,
            operation_runtime=api_server.operation_runtime,
        )
        self._servers.append(server)
        return server

    def start_server(self, name: str, server: Any) -> asyncio.Task[None]:
        """Run a constructed server under its production supervisor node."""
        self.require_selected(name)
        if server not in self._servers:
            raise ValueError("server must be constructed by this harness")
        from polylogue.daemon.cli import _serve_until_complete

        task = self.start(name, lambda: _serve_until_complete(server, label=name))
        assert task is not None
        self._serving.append((name, server, task))
        return task

    async def close(self) -> ShutdownReport:
        try:
            from polylogue.daemon.cli import _shutdown_server_if_serving

            try:
                for name, server, task in reversed(self._serving):
                    await _shutdown_server_if_serving(server, task, label=name)
            finally:
                report = await self.supervisor.shutdown()
        finally:
            for server in reversed(self._servers):
                server.server_close()
            self._serving.clear()
            self._servers.clear()
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
