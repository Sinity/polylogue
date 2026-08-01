"""Regression tests for polylogue-kadx3: the daemon's UDS socket path must be
archive-scoped, not just keyed off ``XDG_RUNTIME_DIR``.

Before this fix, :func:`polylogue.daemon.uds.daemon_socket_path` derived the
same ``$XDG_RUNTIME_DIR/polylogue/daemon.sock`` path regardless of which
archive a daemon served. :class:`~polylogue.daemon.uds.DaemonAPIUnixHTTPServer`
unconditionally unlinks whatever sits at that path before binding, so a
second daemon (e.g. a test/dev instance pointed at a scratch archive) would
silently steal the production daemon's socket file out from under it. A CLI
invocation then reached whichever daemon most recently bound the shared
path, independent of ``POLYLOGUE_ARCHIVE_ROOT``/``--archive-root``.

These tests cover two acceptance criteria:

1. Two archive roots always produce distinct socket paths (pure function,
   no I/O).
2. Two real :class:`DaemonAPIUnixHTTPServer` instances bound at
   archive-scoped paths under the *same* runtime dir never collide: the
   second server starting must not disturb the first server's socket file,
   and both stay independently reachable.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.uses_real_clock(
    "starts real UDS server threads and polls socket readiness with a bounded "
    "wall-clock deadline; frozen_clock cannot substitute for waiting on real "
    "socket/thread startup."
)


def test_archive_scope_key_differs_by_archive_root(tmp_path: Path) -> None:
    from polylogue.daemon.socket_path import archive_scope_key

    root_a = tmp_path / "archive-a"
    root_b = tmp_path / "archive-b"

    assert archive_scope_key(root_a) != archive_scope_key(root_b)


def test_archive_scope_key_stable_for_same_root(tmp_path: Path) -> None:
    from polylogue.daemon.socket_path import archive_scope_key

    root = tmp_path / "archive"

    assert archive_scope_key(root) == archive_scope_key(root)
    # Spelling differences (relative vs. resolved, trailing slash) must not
    # change the identity -- otherwise the same archive could still fracture
    # into two distinct sockets depending on how the caller wrote the path.
    assert archive_scope_key(root) == archive_scope_key(str(root) + "/")


def test_daemon_socket_path_differs_by_archive_root(tmp_path: Path) -> None:
    from polylogue.daemon.socket_path import daemon_socket_path

    root_a = tmp_path / "archive-a"
    root_b = tmp_path / "archive-b"
    runtime_dir = str(tmp_path / "runtime")

    path_a = daemon_socket_path(root_a, runtime_dir=runtime_dir)
    path_b = daemon_socket_path(root_b, runtime_dir=runtime_dir)

    assert path_a != path_b
    # Both must live under the same runtime dir (only the archive-scoped
    # component differentiates them).
    assert path_a.parent.parent == path_b.parent.parent == Path(runtime_dir) / "polylogue"


def test_daemon_socket_path_same_for_same_archive_root(tmp_path: Path) -> None:
    from polylogue.daemon.socket_path import daemon_socket_path

    root = tmp_path / "archive"
    runtime_dir = str(tmp_path / "runtime")

    assert daemon_socket_path(root, runtime_dir=runtime_dir) == daemon_socket_path(root, runtime_dir=runtime_dir)


@pytest.fixture
def _short_runtime_dir() -> Iterator[Path]:
    """A short-path runtime dir so the AF_UNIX socket path stays under the OS limit."""

    runtime_dir = Path(tempfile.mkdtemp(prefix="plg-uds-scope-"))
    try:
        yield runtime_dir
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def _wait_for_ready(client: object, deadline_s: float = 2.0) -> None:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if client.request_json("GET", "/api/health") is not None:  # type: ignore[attr-defined]
            return
        time.sleep(0.02)
    pytest.fail("daemon UDS server did not become ready")


def test_two_daemons_for_different_archives_never_collide_on_socket_path(
    tmp_path: Path,
    _short_runtime_dir: Path,
) -> None:
    """Two daemons scoped to different archives must bind distinct sockets and
    neither may knock the other offline.

    Reproduces the exact failure mode from polylogue-kadx3:
    ``DaemonAPIUnixHTTPServer.__init__`` unlinks whatever is at its target
    path before binding, so an unscoped shared path let a second daemon
    silently steal the first daemon's socket file. With archive-scoped
    paths, starting server B must leave server A's socket file and listener
    completely untouched.
    """

    from polylogue.cli.daemon_client import DaemonClient
    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.daemon.socket_path import daemon_socket_path
    from polylogue.daemon.uds import DaemonAPIUnixHTTPServer

    archive_root_a = tmp_path / "archive-a"
    archive_root_b = tmp_path / "archive-b"
    archive_root_a.mkdir()
    archive_root_b.mkdir()

    socket_path_a = daemon_socket_path(archive_root_a, runtime_dir=str(_short_runtime_dir))
    socket_path_b = daemon_socket_path(archive_root_b, runtime_dir=str(_short_runtime_dir))
    assert socket_path_a != socket_path_b

    server_a = DaemonAPIUnixHTTPServer(socket_path_a, DaemonAPIHandler)
    server_a.auth_token = ""
    thread_a = threading.Thread(target=server_a.serve_forever, name="uds-scope-a", daemon=True)
    thread_a.start()

    server_b: DaemonAPIUnixHTTPServer | None = None
    thread_b: threading.Thread | None = None
    try:
        client_a = DaemonClient(socket_path_a, timeout_s=1.0)
        _wait_for_ready(client_a)

        # Server A's socket file must exist and be reachable before B ever
        # starts, establishing the baseline this test protects.
        assert socket_path_a.exists()
        assert client_a.request_json("GET", "/api/health") is not None

        server_b = DaemonAPIUnixHTTPServer(socket_path_b, DaemonAPIHandler)
        server_b.auth_token = ""
        thread_b = threading.Thread(target=server_b.serve_forever, name="uds-scope-b", daemon=True)
        thread_b.start()

        client_b = DaemonClient(socket_path_b, timeout_s=1.0)
        _wait_for_ready(client_b)

        # The critical regression check: starting server B must not have
        # unlinked or otherwise disturbed server A's socket file, and A must
        # still answer requests on its own path.
        assert socket_path_a.exists()
        assert socket_path_a != socket_path_b
        assert client_a.request_json("GET", "/api/health") is not None
        assert client_b.request_json("GET", "/api/health") is not None
    finally:
        server_a.shutdown()
        server_a.server_close()
        thread_a.join(timeout=2)
        if server_b is not None:
            server_b.shutdown()
            server_b.server_close()
        if thread_b is not None:
            thread_b.join(timeout=2)
