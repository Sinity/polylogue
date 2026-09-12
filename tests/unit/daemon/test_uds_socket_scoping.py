"""Regression tests for polylogue-kadx3: the daemon's UDS socket path must be
archive-scoped, not just keyed off ``XDG_RUNTIME_DIR``.

Before this fix, :func:`polylogue.daemon.uds.daemon_socket_path` derived the
same ``$XDG_RUNTIME_DIR/polylogue/daemon.sock`` path regardless of which
archive a daemon served. A second daemon (e.g. a test/dev instance pointed at
a scratch archive) would
silently steal the production daemon's socket file out from under it. A CLI
invocation then reached whichever daemon most recently bound the shared
path, independent of ``POLYLOGUE_ARCHIVE_ROOT``/``--archive-root``.

These tests cover two acceptance criteria:

1. Two archive roots always produce distinct socket paths (pure function,
   no I/O).
2. Two real maintained daemon operation stacks for different archives never
   collide: starting the second stack must not disturb the first stack's
   socket file, and both stay independently reachable.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.infra.daemon_operations import running_daemon_operations

pytestmark = pytest.mark.uses_real_clock(
    "starts maintained daemon operation stacks with real UDS listeners; frozen_clock "
    "cannot substitute for their writer/listener lifecycle."
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


def test_daemon_socket_path_differs_by_archive_root(tmp_path: Path, _short_runtime_dir: Path) -> None:
    from polylogue.daemon.socket_path import daemon_socket_path

    root_a = tmp_path / "archive-a"
    root_b = tmp_path / "archive-b"
    # A dedicated short runtime dir, not one derived from ``tmp_path``:
    # pytest's own basetemp naming can exceed the AF_UNIX length budget on
    # its own, which would spuriously exercise the long-path fallback below
    # rather than the plain archive-scoped path this test targets.
    runtime_dir = str(_short_runtime_dir)

    path_a = daemon_socket_path(root_a, runtime_dir=runtime_dir)
    path_b = daemon_socket_path(root_b, runtime_dir=runtime_dir)

    assert path_a != path_b
    # Both must live under the same runtime dir (only the archive-scoped
    # component differentiates them).
    assert path_a.parent.parent == path_b.parent.parent == Path(runtime_dir) / "polylogue"


def test_daemon_socket_path_same_for_same_archive_root(tmp_path: Path, _short_runtime_dir: Path) -> None:
    from polylogue.daemon.socket_path import daemon_socket_path

    root = tmp_path / "archive"
    runtime_dir = str(_short_runtime_dir)

    assert daemon_socket_path(root, runtime_dir=runtime_dir) == daemon_socket_path(root, runtime_dir=runtime_dir)


def test_daemon_socket_path_stays_under_af_unix_limit_for_long_runtime_dir(tmp_path: Path) -> None:
    """P2 finding on PR #3526: a moderately long ``XDG_RUNTIME_DIR`` plus the
    scoped ``polylogue/<key>/daemon.sock`` suffix can exceed the AF_UNIX
    ``sun_path`` limit (107 usable bytes), which previously made
    ``bind()`` raise ``OSError: AF_UNIX path too long`` -- a regression the
    archive-scoping fix itself introduced for long runtime dirs. The path
    must always fit, and must still be archive-scoped (no reintroducing the
    original single-socket collision as the escape hatch).
    """

    from polylogue.daemon.socket_path import daemon_socket_path

    # A 73-character runtime dir is the exact reproduction size the review
    # comment cited as producing a 108-byte (over the limit) scoped path.
    long_runtime_dir = "/tmp/" + ("x" * 68)
    assert len(long_runtime_dir) == 73

    root_a = tmp_path / "archive-a"
    root_b = tmp_path / "archive-b"

    path_a = daemon_socket_path(root_a, runtime_dir=long_runtime_dir)
    path_b = daemon_socket_path(root_b, runtime_dir=long_runtime_dir)

    assert len(str(path_a)) <= 107
    assert len(str(path_b)) <= 107
    assert path_a != path_b


@pytest.fixture
def _short_runtime_dir() -> Iterator[Path]:
    """A short-path runtime dir so the AF_UNIX socket path stays under the OS limit."""

    runtime_dir = Path(tempfile.mkdtemp(prefix="plg-uds-scope-", dir="/tmp"))
    try:
        yield runtime_dir
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def test_two_daemons_for_different_archives_never_collide_on_socket_path(
    tmp_path: Path,
) -> None:
    """Two maintained daemon stacks for different archives stay independent.

    Reproduces the exact failure mode from polylogue-kadx3: an unscoped shared
    path let a second daemon silently steal the first daemon's socket file.
    Both maintained listeners must remain reachable.
    """

    archive_root_a = tmp_path / "archive-a"
    archive_root_b = tmp_path / "archive-b"
    # The maintained fixture supplies each complete machine operation stack;
    # the pure socket-path tests above retain the archive-scope proof itself.
    with running_daemon_operations(archive_root_a) as stack_a:
        with running_daemon_operations(archive_root_b) as stack_b:
            assert stack_a.socket_path != stack_b.socket_path
            assert stack_a.socket_path.exists()
            assert stack_b.socket_path.exists()
            status_a = stack_a.client.operation("status", {}, archive_root=str(archive_root_a))
            status_b = stack_b.client.operation("status", {}, archive_root=str(archive_root_b))
            assert status_a is not None and status_a["archive"]["root"] == str(archive_root_a)
            assert status_b is not None and status_b["archive"]["root"] == str(archive_root_b)
