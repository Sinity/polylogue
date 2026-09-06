"""Pin every archive-root resolution in a pytest process to scratch.

``polylogue.paths.archive_root()`` layers the environment over
``polylogue.toml``, so a process that names no archive resolves the one the
operator's config names -- their live archive. The per-test environment
fixture redirects that only for the duration of a test function; collection,
plugin hooks, and session-scoped fixtures run outside it and resolve live.
Pinning the variable for the whole session closes that window: the operator's
archive is unreachable from a test process at any scope.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import MutableMapping
from pathlib import Path

ARCHIVE_ROOT_ENV = "POLYLOGUE_ARCHIVE_ROOT"
_PREFIX = "polylogue-pytest-session-archive-"


def pin_session_archive_root(environ: MutableMapping[str, str], *, base: Path | None = None) -> Path:
    """Point *environ*'s archive root at a fresh scratch directory and return it."""
    root = Path(tempfile.mkdtemp(prefix=_PREFIX, dir=base))
    environ[ARCHIVE_ROOT_ENV] = str(root)
    return root


def discard_session_archive_root(root: Path) -> None:
    """Remove a root minted by :func:`pin_session_archive_root`."""
    if root.name.startswith(_PREFIX):
        shutil.rmtree(root, ignore_errors=True)


__all__ = [
    "ARCHIVE_ROOT_ENV",
    "discard_session_archive_root",
    "pin_session_archive_root",
]
