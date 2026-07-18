"""Swap in a modern bundled SQLite when the system one lacks required FTS5 features.

index.db's FTS5 tables are declared with ``contentless_delete=1`` (SQLite
3.43+, see docs/internals.md). Most current systems (including this project's
Nix devshell) already ship a new-enough SQLite and this module is a no-op
there. Some Docker-based CI images predate it (e.g. cimg/python:3.13 on
Debian bookworm ships 3.40.1); on those, if the optional ``sqlite-compat``
extra (``pysqlite3-binary``) is installed, prefer it over the stdlib
``sqlite3`` module. Must be imported before anything else imports ``sqlite3``,
so it is the first statement in ``polylogue/__init__.py``.
"""

from __future__ import annotations

import sqlite3
import sys

_MINIMUM_FOR_CONTENTLESS_DELETE = (3, 43, 0)


def _ensure_modern_sqlite() -> None:
    if sqlite3.sqlite_version_info >= _MINIMUM_FOR_CONTENTLESS_DELETE:
        return
    try:
        import pysqlite3
    except ImportError:
        return
    sys.modules["sqlite3"] = pysqlite3


_ensure_modern_sqlite()
