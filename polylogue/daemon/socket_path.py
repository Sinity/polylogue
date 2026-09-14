"""Archive-scoped Unix domain socket path derivation.

Deliberately stdlib-only, with no archive/storage imports, so it stays cheap
to import from the CLI's daemon-probe hot path (mirrors the constraint
documented on :mod:`polylogue.daemon_client`).

Historically the daemon's UDS path was derived from ``XDG_RUNTIME_DIR`` alone
(``$XDG_RUNTIME_DIR/polylogue/daemon.sock``), with no archive-root component.
Every ``polylogued`` instance on a machine shares one ``XDG_RUNTIME_DIR``, so
two daemons pointed at different archives collided on the exact same socket
path: :class:`polylogue.daemon.uds.DaemonAPIUnixHTTPServer` removes a stale socket
before binding, so the second daemon to start silently stole the first one's
socket file. (It now probes first and unlinks only an unanswered socket, but
that alone would not have separated two archives.) A CLI invocation then
reaches whichever daemon most recently bound the shared path, regardless of
``POLYLOGUE_ARCHIVE_ROOT``/``--archive-root`` (polylogue-kadx3). Keying the
path off the resolved archive root closes this: two archives never produce
the same socket path, so their daemons never contend for one another's file.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


class UnsafeSocketDirectoryError(RuntimeError):
    """A derived socket directory exists but is not private to this user."""


def archive_scope_key(archive_root: Path | str) -> str:
    """Return a short, stable key identifying an archive root.

    Uses the fully resolved (absolute, symlink-followed) path so the same
    archive always maps to the same key regardless of how it was spelled
    (relative path, trailing slash, ``~`` expansion, symlink hop, ...).
    """

    resolved = str(Path(archive_root).expanduser().resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:_SCOPE_KEY_CHARS]


# Linux's sockaddr_un.sun_path is a 108-byte buffer that must hold a
# NUL-terminated path, leaving 107 usable bytes. A long-but-legal
# XDG_RUNTIME_DIR (some container/CI setups produce one) plus the scoped
# ``polylogue/<key>/daemon.sock`` suffix can exceed that -- bind() then fails
# with OSError: AF_UNIX path too long.
_AF_UNIX_PATH_MAX = 107
#: Width of the archive scope key ``archive_scope_key`` produces.
_SCOPE_KEY_CHARS = 12


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

    fallback_dir = Path(f"/tmp/polylogue-{os.geteuid()}")
    return fallback_dir / f"{key}.sock"


def _validate_private_dir(path: Path) -> None:
    """Refuse *path* unless it is a real directory private to this user.

    ``lstat`` rather than ``stat``: a symlink planted at the predictable name
    is exactly the substitution being rejected, and following it would validate
    the target instead.
    """

    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode):
        raise UnsafeSocketDirectoryError(f"{path} exists and is not a directory")
    if info.st_uid != os.geteuid():
        raise UnsafeSocketDirectoryError(f"{path} is owned by uid {info.st_uid}, not {os.geteuid()}")
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o077:
        raise UnsafeSocketDirectoryError(f"{path} is group/world accessible ({oct(mode)})")


def _derived_components(directory: Path) -> tuple[Path, ...]:
    """Return the components of *directory* this module derived, outermost first.

    Matched on the exact shapes ``daemon_socket_path`` produces, never on a
    name alone: a checkout can live at ``/realm/project/polylogue`` and an
    ancestor named for the product is not one this module created.
    """

    if directory.name == f"polylogue-{os.geteuid()}":
        return (directory,)
    if directory.parent.name == "polylogue" and _looks_like_scope_key(directory.name):
        return (directory.parent, directory)
    return ()


def _looks_like_scope_key(name: str) -> bool:
    return len(name) == _SCOPE_KEY_CHARS and all(character in "0123456789abcdef" for character in name)


def ensure_private_socket_dir(directory: Path) -> None:
    """Create *directory* owner-only, refusing an unsafe pre-existing component.

    ``daemon_socket_path`` derives predictable names under a shared base --
    ``$XDG_RUNTIME_DIR/polylogue/<key>/`` and, for a long runtime dir,
    ``/tmp/polylogue-<uid>/``. ``mkdir(mode=0o700, exist_ok=True)`` applies its
    mode only when it creates the directory, so another local user who claims
    the predictable name first -- as a world-writable directory, or as a
    symlink into one they own -- has the daemon bind its socket inside a
    directory they control. They can then unlink the live socket and bind a
    replacement at the same path, and a client that connects there hands over
    the machine bearer.

    Every component this module derived is validated: it must be a real
    directory (not a symlink), owned by this effective uid, with no group or
    world bits. A directory the caller chose instead is created as before and
    not vouched for -- this function makes no claim about a path it did not
    derive.
    """

    derived = _derived_components(directory)
    if not derived:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        return
    for candidate in derived:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(candidate, 0o700)
        except FileExistsError:
            _validate_private_dir(candidate)


__all__ = [
    "UnsafeSocketDirectoryError",
    "archive_scope_key",
    "daemon_socket_path",
    "ensure_private_socket_dir",
]
