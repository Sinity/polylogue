"""Archive-scoped Unix domain socket path derivation.

Deliberately stdlib-only, with no archive/storage imports, so it stays cheap
to import from the CLI's daemon-probe hot path (mirrors the constraint
documented on :mod:`polylogue.cli.daemon_client`).

Historically the daemon's UDS path was derived from ``XDG_RUNTIME_DIR`` alone
(``$XDG_RUNTIME_DIR/polylogue/daemon.sock``), with no archive-root component.
Every ``polylogued`` instance on a machine shares one ``XDG_RUNTIME_DIR``, so
two daemons pointed at different archives collided on the exact same socket
path: :class:`polylogue.daemon.uds.DaemonAPIUnixHTTPServer` unconditionally
unlinks whatever is at the target path before binding, so the second daemon
to start silently steals the first one's socket file. A CLI invocation then
reaches whichever daemon most recently bound the shared path, regardless of
``POLYLOGUE_ARCHIVE_ROOT``/``--archive-root`` (polylogue-kadx3). Keying the
path off the resolved archive root closes this: two archives never produce
the same socket path, so their daemons never contend for one another's file.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def archive_scope_key(archive_root: Path | str) -> str:
    """Return a short, stable key identifying an archive root.

    Uses the fully resolved (absolute, symlink-followed) path so the same
    archive always maps to the same key regardless of how it was spelled
    (relative path, trailing slash, ``~`` expansion, symlink hop, ...).
    """

    resolved = str(Path(archive_root).expanduser().resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]


# Linux's sockaddr_un.sun_path is a 108-byte buffer that must hold a
# NUL-terminated path, leaving 107 usable bytes. A long-but-legal
# XDG_RUNTIME_DIR (some container/CI setups produce one) plus the scoped
# ``polylogue/<key>/daemon.sock`` suffix can exceed that -- bind() then fails
# with OSError: AF_UNIX path too long.
_AF_UNIX_PATH_MAX = 107


def daemon_socket_path(archive_root: Path | str, *, runtime_dir: str | None = None) -> Path:
    """Return the archive-scoped per-user UDS path without creating it.

    ``archive_root`` should be the same value used to populate the daemon's
    ``/api/health`` ``archive_root`` field (e.g. ``config.archive_root`` or
    the resolved ``archive_root_path`` the daemon runtime starts with), so
    the path a CLI probe computes always matches the path the corresponding
    daemon actually bound.

    Falls back to a deliberately short, still per-user, still archive-scoped
    location when ``runtime_dir`` (or ``XDG_RUNTIME_DIR``) is long enough
    that the scoped path would exceed the AF_UNIX ``sun_path`` limit -- the
    fallback keeps the archive-scoping property (two archives still never
    collide) rather than reintroducing the original single-socket bug.
    """

    key = archive_scope_key(archive_root)
    base = Path(runtime_dir or os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "polylogue"
    candidate = base / key / "daemon.sock"
    if len(str(candidate)) <= _AF_UNIX_PATH_MAX:
        return candidate

    fallback_dir = Path(f"/tmp/polylogue-{os.getuid()}")
    return fallback_dir / f"{key}.sock"


__all__ = ["archive_scope_key", "daemon_socket_path"]
