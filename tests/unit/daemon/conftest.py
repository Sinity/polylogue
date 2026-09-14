"""Daemon-test isolation from the host's ``XDG_RUNTIME_DIR``.

``polylogue.daemon.socket_path`` derives the UDS directory from
``XDG_RUNTIME_DIR`` and refuses a derived component that is not private to
this user (``UnsafeSocketDirectoryError``).  A test that starts daemon
services therefore reads the *machine's* ``/run/user/<uid>/polylogue``, whose
mode is whatever the host happens to have -- a pre-hardening daemon build
created that base with the process umask (0o755), so the same test passes on
one workstation and fails on another with a message about a directory no test
created.  Ambient machine state must not decide a test outcome.

Every daemon test therefore gets its own runtime directory, created
owner-only.  The daemon's own privacy validation still runs against it: this
fixture only chooses *where* the derivation starts, it does not weaken the
check.  A test that wants to exercise the unsafe-directory refusal points
``XDG_RUNTIME_DIR`` at a directory it prepares itself
(``tests/unit/daemon/test_uds_socket_ownership.py``).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _private_runtime_dir(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    runtime_dir = tmp_path_factory.mktemp("xdg-runtime")
    runtime_dir.chmod(0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    yield
