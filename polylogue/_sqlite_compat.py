"""Swap in bundled SQLite when the system module lacks required capabilities.

index.db's FTS5 tables are declared with ``contentless_delete=1`` (SQLite
3.43+, see docs/internals.md). Most current systems (including this project's
Nix devshell) already ship a new-enough SQLite and this module is a no-op
there. Some Docker-based CI images predate it (e.g. cimg/python:3.13 on
Debian bookworm ships 3.40.1); on those, if the optional ``sqlite-compat``
extra (``pysqlite3-binary``) is installed, prefer it over the stdlib
``sqlite3`` module. Must be imported before anything else imports ``sqlite3``,
so it is the first statement in ``polylogue/__init__.py``.
The compatibility decision covers both required capabilities. Some Python
builds expose a recent SQLite library but compile the stdlib wrapper without
loadable-extension support, which prevents the required sqlite-vec extension
from loading. When the optional ``pysqlite3-binary`` extra is present, its
connection type supplies that capability.
"""

from __future__ import annotations

import sqlite3
import sys

_MINIMUM_FOR_CONTENTLESS_DELETE = (3, 43, 0)


def _needs_sqlite_compat(module: object) -> bool:
    version = getattr(module, "sqlite_version_info", (0, 0, 0))
    connection_type = getattr(module, "Connection", None)
    return version < _MINIMUM_FOR_CONTENTLESS_DELETE or not hasattr(connection_type, "enable_load_extension")


def _ensure_modern_sqlite() -> None:
    if not _needs_sqlite_compat(sqlite3):
        return
    try:
        import pysqlite3
    except ImportError:
        return
    if not _needs_sqlite_compat(pysqlite3):
        sys.modules["sqlite3"] = pysqlite3


_ensure_modern_sqlite()
