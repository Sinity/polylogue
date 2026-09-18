"""The total form of the single-writer boundary at the connection call itself.

:mod:`polylogue.storage.sqlite.write_lease` makes every *declared* write-mode
factory take the lease. That is not the same as totality: roughly seventy
production sites open ``sqlite3.connect(...)`` directly, and a writer that
reaches an archive tier that way contends through the busy timeout exactly as
the rehearsal writer did (polylogue-8qm4k). An AST census of the factories
cannot see those sites, and adding a census rule for each one would make the
boundary a review convention again.

This module closes the remaining door at the only place every writer must pass:
``sqlite3.connect``. While the guard is installed, an open that is *writable*
and resolves to an *archive tier file* asserts the lease before the connection
exists, so an unserialized writer raises :class:`UnleasedWriteError` instead of
locking out the daemon's live chunk.

Scope is deliberately narrow:

* Read-only opens (``mode=ro``, immutable, sealed) are untouched -- refusing a
  reader would be overreach and the single-writer boundary says nothing about
  them.
* Non-archive databases -- spill files, provider caches, the Sinex database,
  scratch and ``:memory:`` -- are untouched. Tier membership is decided by the
  file name the six tiers declare, not by directory.
* The guard defers entirely to :func:`require_write_lease`, which is itself
  inert unless enforcement is armed. Installing it in a one-shot CLI or an
  embedded caller therefore changes nothing; it only bites inside the daemon's
  armed process boundary or a test that arms it.

Archive-root binding stays with the declared factories, which know which
archive they were asked for. The guard passes no root: a generation build and a
backup staging copy legitimately write a file named ``index.db`` outside the
live root, and the question the guard answers is "is any lease held by this
thread", not "is this the live archive".

The one bypass is explicit and named: :func:`declared_unguarded_write` is for
the authorities that own an archive *without* the daemon -- first-time
bootstrap before a lease can exist, offline exclusive rebuild, migration, and
the test seams that construct fixtures. It is thread-local, it takes a reason,
and it is not reachable by accident.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

__all__ = [
    "ARCHIVE_TIER_FILENAMES",
    "archive_write_guard_installed",
    "declared_unguarded_write",
    "guarded_archive_tier_path",
    "install_archive_write_guard",
]

#: The six tier file names. A path is an archive tier open when its file name
#: is one of these; the containing directory is deliberately not consulted.
ARCHIVE_TIER_FILENAMES = frozenset(
    {
        "source.db",
        "index.db",
        "embeddings.db",
        "user.db",
        "audit.db",
        "ops.db",
    }
)

_BYPASS = threading.local()
_INSTALL_GUARD = threading.Lock()
_INSTALL_DEPTH = 0
_ORIGINAL_CONNECT: Callable[..., sqlite3.Connection] | None = None


def _bypassed() -> bool:
    return bool(getattr(_BYPASS, "reason", None))


@contextmanager
def declared_unguarded_write(reason: str) -> Iterator[None]:
    """Run a declared non-daemon archive authority without the guard.

    ``reason`` is required so the seam is legible at its call site: bootstrap
    of an archive that has no lease yet, an offline exclusive rebuild, a
    migration behind its own backup, or a test fixture. It is thread-local and
    does not weaken :func:`require_write_lease` for anything else.
    """
    if not reason:
        raise ValueError("declared_unguarded_write requires a reason")
    previous = getattr(_BYPASS, "reason", None)
    _BYPASS.reason = reason
    try:
        yield
    finally:
        _BYPASS.reason = previous


def guarded_archive_tier_path(database: Any, *, uri: bool = False) -> Path | None:
    """Return the archive tier path this open would *write*, else ``None``.

    Returns ``None`` for read-only opens, in-memory databases, non-path
    arguments and any file whose name is not a declared tier.
    """
    if isinstance(database, int):  # an already-open descriptor: not our boundary
        return None
    if isinstance(database, bytes):
        database = database.decode("utf-8", "surrogateescape")
    try:
        text = str(database if isinstance(database, str) else Path(database))
    except TypeError:
        return None
    if not text or text.startswith(":memory:"):
        return None
    if uri or text.startswith("file:"):
        split = urlsplit(text)
        if split.scheme and split.scheme != "file":
            return None
        query = split.query
        # ``mode`` and ``immutable`` are the only read-only declarations SQLite
        # honours in a URI; anything else is a writable open.
        for part in query.split("&"):
            key, _, value = part.partition("=")
            if key == "mode" and value in {"ro"}:
                return None
            if key == "immutable" and value not in {"", "0", "false"}:
                return None
        text = unquote(split.path if split.path else split.netloc)
        if text.startswith(":memory:") or not text:
            return None
    path = Path(text)
    if path.name not in ARCHIVE_TIER_FILENAMES:
        return None
    return path


def _guarded_connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
    original = _ORIGINAL_CONNECT
    assert original is not None
    if not _bypassed():
        # Imported here, not at module scope: ``write_lease`` re-exports this
        # module's installer so the guard has a static production consumer,
        # and a module-level import back would close that cycle.
        from polylogue.storage.sqlite.write_lease import require_write_lease

        path = guarded_archive_tier_path(database, uri=bool(kwargs.get("uri", False)))
        if path is not None:
            # No ``archive_root``: the guard asserts that *a* lease is held by
            # this thread or task, and leaves root binding to the declared
            # factories, which know which archive they were asked for.
            require_write_lease(f"sqlite3.connect({path})")
    return original(database, *args, **kwargs)


def archive_write_guard_installed() -> bool:
    """Whether writable archive-tier opens are currently intercepted."""
    return _INSTALL_DEPTH > 0


@contextmanager
def install_archive_write_guard() -> Iterator[None]:
    """Intercept writable archive-tier opens for the duration of the block.

    Re-entrant and process-wide by construction, because ``sqlite3.connect`` is
    one module attribute. Enforcement itself stays with
    :func:`require_write_lease`, so an installed guard in an unarmed process is
    a no-op wrapper.
    """
    global _INSTALL_DEPTH, _ORIGINAL_CONNECT
    with _INSTALL_GUARD:
        if _INSTALL_DEPTH == 0:
            _ORIGINAL_CONNECT = sqlite3.connect
            sqlite3.connect = _guarded_connect  # type: ignore[assignment]
        _INSTALL_DEPTH += 1
    try:
        yield
    finally:
        with _INSTALL_GUARD:
            _INSTALL_DEPTH -= 1
            if _INSTALL_DEPTH == 0 and _ORIGINAL_CONNECT is not None:
                sqlite3.connect = _ORIGINAL_CONNECT  # type: ignore[assignment]
                _ORIGINAL_CONNECT = None
