"""A derived socket directory is private, and a foreign peer never gets the bearer.

``daemon_socket_path`` derives predictable names under a shared base --
``$XDG_RUNTIME_DIR/polylogue/<key>/`` and, when that path would exceed the
AF_UNIX limit, ``/tmp/polylogue-<uid>/``. Two consequences were unguarded:

- ``mkdir(mode=0o700, exist_ok=True)`` applies its mode only when it creates
  the directory, so another local user who claims the predictable name first
  keeps a directory the daemon then binds its socket inside;
- the CLI's UDS client sends ``Authorization: Bearer <machine token>`` on its
  first request without establishing who is listening, so a socket substituted
  at that predictable path collects the machine bearer.

Anti-vacuity: replace ``ensure_private_socket_dir``'s validation with the
original ``directory.mkdir(mode=0o700, parents=True, exist_ok=True)`` and the
squatted/symlinked directory tests stop raising; delete the
``_reject_foreign_peer`` call in ``_UnixHTTPConnection.connect`` and the client
test stops raising, with the bearer appearing in the recorded request bytes.
"""

from __future__ import annotations

import os
import socket
import tempfile
import threading
from pathlib import Path

import pytest

from polylogue.daemon.socket_path import (
    UnsafeSocketDirectoryError,
    archive_scope_key,
    daemon_socket_path,
    ensure_private_socket_dir,
)
from polylogue.daemon_client import DaemonClient, DaemonSocketOwnershipError

_KEY = archive_scope_key("/")


def test_a_derived_socket_directory_is_created_owner_only(tmp_path: Path) -> None:
    directory = tmp_path / "polylogue" / _KEY
    ensure_private_socket_dir(directory)
    assert directory.is_dir()
    for candidate in (directory, directory.parent):
        mode = candidate.stat().st_mode & 0o777
        assert not mode & 0o077, f"{candidate} is group/world accessible: {oct(mode)}"


def test_a_squatted_world_writable_directory_is_refused(tmp_path: Path) -> None:
    squatted = tmp_path / "polylogue"
    squatted.mkdir()
    os.chmod(squatted, 0o777)
    with pytest.raises(UnsafeSocketDirectoryError) as excinfo:
        ensure_private_socket_dir(squatted / _KEY)
    assert "group/world accessible" in str(excinfo.value)


def test_a_symlinked_socket_directory_is_refused(tmp_path: Path) -> None:
    elsewhere = tmp_path / "attacker"
    elsewhere.mkdir(mode=0o700)
    link = tmp_path / "polylogue"
    link.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(UnsafeSocketDirectoryError) as excinfo:
        ensure_private_socket_dir(link / _KEY)
    assert "not a directory" in str(excinfo.value)


def test_a_caller_supplied_directory_is_created_without_a_private_claim(tmp_path: Path) -> None:
    """A path this module did not derive keeps the previous create-only behaviour."""
    directory = tmp_path / "harness-sockets"
    ensure_private_socket_dir(directory)
    assert directory.is_dir()
    ensure_private_socket_dir(directory)


def test_the_long_runtime_dir_fallback_is_named_for_the_effective_uid(tmp_path: Path) -> None:
    path = daemon_socket_path(tmp_path, runtime_dir="/tmp/" + "x" * 80)
    assert path.parent == Path(f"/tmp/polylogue-{os.geteuid()}")


def test_the_client_withholds_the_bearer_from_a_foreign_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal is driven by the kernel's answer, which the peer cannot forge."""
    # A short directory: AF_UNIX sun_path is 107 bytes, well under a tmp_path.
    directory = Path(tempfile.mkdtemp(prefix="plg-peer-", dir="/tmp"))
    socket_path = directory / "daemon.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    received: list[bytes] = []

    def _serve() -> None:
        try:
            conn, _ = server.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(1.0)
            try:
                received.append(conn.recv(4096))
            except OSError:
                pass

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    try:
        # The peer really is this uid, so stand in for a foreign one by moving
        # the comparison's other side; the kernel's answer is unchanged and
        # everything else on the path is the production route.
        foreign_uid = os.geteuid() + 1
        monkeypatch.setattr("polylogue.daemon_client.os.geteuid", lambda: foreign_uid)

        client = DaemonClient(socket_path, timeout_s=1.0, auth_token="machine-bearer")
        with pytest.raises(DaemonSocketOwnershipError) as excinfo:
            client.operation("status", {})
        assert "refusing to send the machine bearer" in str(excinfo.value)
    finally:
        thread.join(timeout=1.0)
        server.close()
        socket_path.unlink(missing_ok=True)
        directory.rmdir()
    assert not any(b"machine-bearer" in payload for payload in received)
