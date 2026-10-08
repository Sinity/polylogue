"""Production-stack fixtures for declared daemon operation tests.

The fixture deliberately starts the independent machine UDS listener rather
than borrowing the browser HTTP handler.  It also owns a real coordinator loop
so mutation tests exercise the same writer retention rules as the daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import socket
import sys
import threading
import traceback
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

from polylogue.core.compute import BoundedComputeAdapter
from polylogue.daemon.http import _recover_startup_with_compute
from polylogue.daemon.operation_runtime import DaemonOperationRuntime
from polylogue.daemon.socket_path import ensure_private_socket_dir
from polylogue.daemon.uds import DaemonAPIUnixHTTPServer
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.daemon_client import DaemonClient
from polylogue.operations.daemon_reads import DaemonReadDependencies
from polylogue.operations.operation_context import prepare_operation_journals
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.archive_templates import run_off_event_loop

if TYPE_CHECKING:
    from polylogue.operations.mutation_transaction import (
        MutationAuthorization,
        MutationPreview,
        MutationPrincipal,
        MutationReceipt,
    )


class _CoordinatorLoop:
    """One dedicated event loop thread for a real write coordinator."""

    def __init__(self, archive_root: Path) -> None:
        self._archive_root = archive_root
        self.loop: asyncio.AbstractEventLoop | None = None
        self.coordinator: DaemonWriteCoordinator | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="test-daemon-coordinator", daemon=True)
        self._thread.start()
        # Wait for the loop itself, not a wall clock: a loaded host can take
        # seconds to schedule this thread, and pytest-timeout bounds the test.
        while not self._ready.wait(timeout=0.05):
            if not self._thread.is_alive():
                raise RuntimeError("test daemon coordinator loop exited before it started")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        self.coordinator = DaemonWriteCoordinator(archive_root=self._archive_root)
        # Signal from inside the running loop, as the production writer loop
        # does: a bridge admitted before ``run_forever`` sees ``is_running()``
        # false and refuses the write as DaemonWriterOwnerLoopStopped.
        loop.call_soon(self._ready.set)
        try:
            loop.run_forever()
        finally:
            loop.close()

    def close(self) -> None:
        assert self.loop is not None
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join()


@dataclass(slots=True)
class DaemonOperationStack:
    """Live machine listener plus its independently-owned production runtime."""

    archive_root: Path
    socket_path: Path
    client: DaemonClient
    server: DaemonAPIUnixHTTPServer
    runtime: DaemonOperationRuntime
    write_coordinator: DaemonWriteCoordinator
    write_bridge: DaemonWriteThreadBridge
    execution_kernel: BoundedComputeAdapter
    _loop: _CoordinatorLoop
    _server_thread: threading.Thread

    def session_exists(self, session_id: str) -> bool:
        """Read the seeded archive through its canonical archive reader."""

        with ArchiveStore.open_existing(self.archive_root) as archive:
            return session_id in archive.resolve_exact_session_ids((session_id,))

    def close(self) -> None:
        """Stop ingress, drain the lifecycle owner, then release resources."""

        self.server.shutdown()
        self.server.server_close()
        self._server_thread.join()

        loop = self._loop.loop
        assert loop is not None
        asyncio.run_coroutine_threadsafe(self.runtime.shutdown(), loop).result()
        drained = asyncio.run_coroutine_threadsafe(self.write_coordinator.shutdown(timeout=30), loop).result()
        if not drained:
            raise RuntimeError("test daemon writer did not drain before loop close")
        self.execution_kernel.shutdown(wait=True)
        self._loop.close()


@contextlib.contextmanager
def running_daemon_operations(
    archive_root: Path,
    *,
    seed_archive: Callable[[Path], None] | None = None,
    server_error_sink: queue.SimpleQueue[str] | None = None,
    compute_workers: int = 2,
    compute_queue_units: int = 4,
    socket_path: Path | None = None,
    session_derivation: bool = False,
    read_dependencies: DaemonReadDependencies | None = None,
) -> Iterator[DaemonOperationStack]:
    """Start one real machine operation stack rooted at ``archive_root``.

    ``seed_archive`` runs after bootstrap and before daemon startup. The shared
    compute adapter is passed to both the runtime and UDS server explicitly;
    no global adapter, browser route, configuration root, or daemon singleton
    is used.  ``compute_workers``/``compute_queue_units`` size that one bounded
    kernel: a test that deliberately saturates admission and one that measures
    service under load need different sizes, and both must state which.
    """

    archive_root = archive_root.resolve()

    def prepare_archive() -> None:
        initialize_active_archive_root(archive_root)
        if seed_archive is not None:
            seed_archive(archive_root)

    # Async tests enter this synchronous stack from their event loop; setup's
    # synchronous write lease must not block that loop.
    run_off_event_loop(prepare_archive)
    socket_path = socket_path or (Path("/tmp") / f"plg-op-{os.getpid()}-{uuid4().hex}.sock")
    if socket_path.parent != Path("/tmp"):
        ensure_private_socket_dir(socket_path.parent)
    probe_path = Path("/tmp") / f"plg-probe-{uuid4().hex}.sock"
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe_bound = False
    try:
        try:
            probe.bind(str(probe_path))
            probe_bound = True
        except PermissionError:
            pytest.skip("sandbox denies AF_UNIX listeners required for the production operation stack")
    finally:
        probe.close()
        if probe_bound:
            probe_path.unlink(missing_ok=True)

    coordinator_loop = _CoordinatorLoop(archive_root)
    assert coordinator_loop.loop is not None
    assert coordinator_loop.coordinator is not None
    coordinator = coordinator_loop.coordinator
    # The production bridge bound (30 s), not a tighter test clock: a loaded
    # host routinely takes longer than 5 s to admit the startup journal write.
    bridge = DaemonWriteThreadBridge(coordinator, coordinator_loop.loop)
    bridge.run_sync("daemon.operation_journals.startup", prepare_operation_journals, archive_root)
    kernel = BoundedComputeAdapter(
        max_workers=compute_workers, queue_units=compute_queue_units, thread_name_prefix="test-daemon-operation"
    )
    # Startup recovery runs on the daemon's own compute creator with its
    # original input admission, exactly as the standalone HTTP server does.
    recovery = asyncio.run_coroutine_threadsafe(
        _recover_startup_with_compute(bridge, kernel, archive_root), coordinator_loop.loop
    )
    try:
        bridge._await_owner_settlement("daemon.operation_recovery.startup", recovery)
    except BaseException:
        kernel.shutdown(wait=True)
        raise
    session_maintenance = None
    if session_derivation:
        from time import time

        from polylogue.daemon.session_profile_composition import compose_session_profile_callback

        session_maintenance = compose_session_profile_callback(
            archive_root, compute_adapter=kernel, write_bridge=bridge, now=time
        ).maintenance
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner

    if read_dependencies is None:
        read_dependencies = DaemonReadDependencies(hermes_root=archive_root.parent / "hermes")
    runtime = DaemonOperationRuntime(
        archive_root,
        write_bridge=bridge,
        execution_kernel=kernel,
        raw_observation_owner=RawObservationConvergenceOwner(
            archive_root, compute_adapter=kernel, write_bridge=bridge, write_coordinator=bridge.coordinator
        ),
        owner_loop=bridge.owner_loop,
        session_maintenance=session_maintenance,
        read_dependencies=read_dependencies,
    )
    server = DaemonAPIUnixHTTPServer(
        socket_path,
        archive_root=archive_root,
        auth_token=None,
        write_bridge=bridge,
        execution_kernel=kernel,
        operation_runtime=runtime,
    )
    if server_error_sink is not None:

        def capture_server_error(_request: object, _client_address: object) -> None:
            """Retain one unhandled handler traceback for a test assertion."""

            exc_type, exc, trace = sys.exc_info()
            assert exc_type is not None and exc is not None
            server_error_sink.put("".join(traceback.format_exception(exc_type, exc, trace)))

        object.__setattr__(server, "handle_error", capture_server_error)
    thread = threading.Thread(target=server.serve_forever, name="test-machine-operation-listener", daemon=True)
    thread.start()
    stack = DaemonOperationStack(
        archive_root=archive_root,
        socket_path=socket_path,
        client=DaemonClient(socket_path, timeout_s=5),
        server=server,
        runtime=runtime,
        write_coordinator=coordinator,
        write_bridge=bridge,
        execution_kernel=kernel,
        _loop=coordinator_loop,
        _server_thread=thread,
    )
    try:
        yield stack
    finally:
        stack.close()


@contextlib.contextmanager
def cli_daemon_archive(
    archive_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed_archive: Callable[[Path], None] | None = None,
    home: Path | None = None,
    session_derivation: bool = False,
) -> Iterator[DaemonOperationStack]:
    """Run a real daemon and point the CLI's mutation route at it.

    CLI mutation commands own syntax, preview and refusal only; the resident
    daemon is the sole writer (``configured_mutation_operation``). A CLI test
    that wants to observe what a mutation actually *does* therefore has to
    supply the daemon the command requires, which is what this does: a real
    ``DaemonOperationRuntime`` behind a real UDS listener, reached over the
    ordinary socket the CLI probes.

    ``POLYLOGUE_ARCHIVE_ROOT`` and the XDG roots are set rather than patching
    ``polylogue.cli.commands.*`` path helpers, because the daemon-side handlers
    resolve their own paths through :mod:`polylogue.paths`; patching a CLI
    module attribute would move only the preview and leave the writer pointed
    at the developer's real home.
    """

    from polylogue.daemon.cli import _acquire_pidfile

    archive_root = archive_root.resolve()
    with (
        contextlib.ExitStack() as residency,
        running_daemon_operations(
            archive_root, seed_archive=seed_archive, session_derivation=session_derivation
        ) as stack,
    ):
        # Claim residency exactly as ``polylogued run`` does, and release it
        # only after the stack drains its writer. The CLI decides whether it is
        # an offline writer from this lock; without it, an in-process CLI arms
        # its process-wide offline-writer probe, which then takes archive
        # custody on the daemon's own operation threads.
        residency.callback(os.close, _acquire_pidfile(archive_root / "daemon.pid"))
        monkeypatch.setattr("polylogue.daemon.socket_path.daemon_socket_path", lambda _root: stack.socket_path)
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        if home is not None:
            home.mkdir(parents=True, exist_ok=True)
            monkeypatch.setenv("HOME", str(home))
            monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
            monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
            monkeypatch.setenv("XDG_STATE_HOME", str(home / ".local" / "state"))
            monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
        yield stack


@pytest.fixture
def daemon_operation_stack(tmp_path: Path) -> Iterator[DaemonOperationStack]:
    """Pytest fixture providing an empty, synthetic, production operation stack."""

    with running_daemon_operations(tmp_path / "archive") as stack:
        yield stack


__all__ = [
    "DaemonOperationStack",
    "cli_daemon_archive",
    "daemon_operation_stack",
    "running_daemon_operations",
]


@contextlib.contextmanager
def daemon_serving_archive(archive_root: Path, *, session_derivation: bool = False) -> Iterator[DaemonOperationStack]:
    """Run the archive's resident writer on its own socket for one test.

    Public archive writes are daemon-owned (#5550): the facade submits a
    declared operation to ``polylogued run`` and refuses with
    ``FacadeDaemonRequiredError`` when none answers. A test that writes
    through the ``Polylogue`` facade wraps the write in this, so it reaches
    the production operation stack. Ingest fixtures enable session derivation,
    which the accepted ingest owner requires before accepting retained work.
    """
    from unittest.mock import patch

    from polylogue.daemon.socket_path import daemon_socket_path

    with (
        patch("polylogue.daemon.api_auth.resolve_api_auth_token", return_value=None),
        running_daemon_operations(
            archive_root,
            socket_path=daemon_socket_path(archive_root.resolve()),
            session_derivation=session_derivation,
        ) as stack,
    ):
        yield stack


def accepted_operation_reference(
    operation: str, *, request_id: str, artifact_kind: str, part_count: int = 1
) -> dict[str, object]:
    """Build a neutral declared wire reference for surface transport doubles."""
    from polylogue.operations.daemon_protocol import AcceptedOperationReference

    return AcceptedOperationReference(
        archive_identity="synthetic-archive",
        request_id=request_id,
        principal_ref="synthetic-actor",
        fingerprint="0" * 64,
        operation_name=operation,
        artifact_kind=artifact_kind,
        artifact_ref=f"synthetic:{artifact_kind}",
        accepted_at_ms=1,
        part_count=part_count,
        accepted_deadline_unix_ms=None,
    ).to_dict()


def prepare_bound_delete(
    stack: DaemonOperationStack, session_ids: tuple[str, ...]
) -> tuple[MutationPreview, MutationAuthorization, MutationPrincipal]:
    """Borrow the original UDS-authenticated plan and authorization for actuator faults."""
    from unittest.mock import patch

    from polylogue.operations.audit import AuditRepository
    from polylogue.operations.daemon_protocol import DaemonOperationRequest

    original_call = stack.runtime.call
    authenticated: list[MutationPrincipal] = []

    def capture(request: DaemonOperationRequest, principal: MutationPrincipal, **kwargs: Any) -> Any:
        if request.operation == "mutation.session.delete.authorize":
            authenticated.append(principal)
        return original_call(request, principal, **kwargs)

    preview_envelope = stack.client.operation_to_completion(
        "mutation.session.delete.preview", {"session_ids": list(session_ids)}, archive_root=str(stack.archive_root)
    )
    assert preview_envelope is not None and preview_envelope["outcome"] == "completed", preview_envelope
    preview = preview_envelope["result"]
    with patch.object(stack.runtime, "call", side_effect=capture):
        authorization_envelope = stack.client.operation_to_completion(
            "mutation.session.delete.authorize",
            {"preview_ref": preview["preview_ref"]},
            archive_root=str(stack.archive_root),
        )
    assert authorization_envelope is not None and authorization_envelope["outcome"] == "completed", (
        authorization_envelope
    )
    result = authorization_envelope["result"]
    assert len(authenticated) == 1
    principal = authenticated[0]

    def load() -> tuple[MutationPreview, MutationAuthorization, MutationPrincipal]:
        audit = AuditRepository(stack.archive_root / "audit.db")
        bound_preview, authorization = audit.authorization_for_principal(result["authorization_ref"], principal)
        return bound_preview, authorization, principal

    return stack.write_bridge.run_sync("test.delete.original-authority", load)


def execute_bound_delete(
    stack: DaemonOperationStack,
    preview: MutationPreview,
    authorization: MutationAuthorization,
    principal: MutationPrincipal,
) -> MutationReceipt:
    """Execute the original bound actuator under the actual daemon writer creator."""
    from polylogue.operations.audit import AuditRepository
    from polylogue.operations.bindings import runtime_operation_binding
    from polylogue.operations.mutation_actuators import SessionDeleteActuator, SessionDeleteArgs
    from polylogue.operations.mutation_transaction import OperationExecutor
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    def execute() -> MutationReceipt:
        audit = AuditRepository(
            stack.archive_root / "audit.db", attempt_owner_id=AuditRepository.current_process_attempt_owner()
        )
        actuator = SessionDeleteActuator()
        session_ids = tuple(target.ref.removeprefix("session:") for target in preview.plan.targets)
        with ArchiveStore.open_existing(stack.archive_root, read_only=False) as archive:
            return OperationExecutor(audit=audit, archive_root=stack.archive_root).execute_bound(
                runtime_operation_binding(actuator),
                preview,
                authorization,
                SessionDeleteArgs(archive=archive, session_ids=session_ids),
            )

    return stack.write_bridge.run_sync("test.delete.bound-actuator", execute)


@contextlib.asynccontextmanager
async def async_daemon_serving_archive(
    archive_root: Path, *, session_derivation: bool = False
) -> AsyncIterator[DaemonOperationStack]:
    """``daemon_serving_archive`` for an async law, started and stopped off its loop.

    Starting the operation stack bootstraps the archive under the synchronous
    write lease, which refuses to block a running event loop.
    """
    serving = daemon_serving_archive(archive_root, session_derivation=session_derivation)
    stack = await asyncio.to_thread(serving.__enter__)
    try:
        yield stack
    except BaseException as error:
        if not await asyncio.to_thread(serving.__exit__, type(error), error, error.__traceback__):
            raise
    else:
        await asyncio.to_thread(serving.__exit__, None, None, None)
