"""Storage backend implementations for Polylogue.

SQLiteBackend is the async-first backend (backed by aiosqlite).
Raw sync utilities (open_connection, connection_context) live in connection.py.

``SQLiteBackend`` is exposed lazily (PEP 562 module ``__getattr__``): the
concrete backend module (``async_sqlite``) transitively imports the whole
mixin surface (archive, raw, query-store) plus everything those mixins touch.
Subpackages of ``polylogue.storage.sqlite`` -- notably ``archive_tiers``,
whose DDL/type modules have no runtime dependency on the backend class --
would otherwise pay that entire weight just to import this parent package
(see polylogue-h1wt). Importing ``SQLiteBackend`` or calling
``create_backend`` still pays the real cost, deferred to first use.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


def create_backend(db_path: Path | None = None) -> SQLiteBackend:
    """Create a SQLite storage backend.

    Args:
        db_path: Optional path to database file. If None, uses default path.

    Returns:
        SQLiteBackend instance
    """
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

    return SQLiteBackend(db_path=db_path)


def __getattr__(name: str) -> object:
    if name == "SQLiteBackend":
        from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

        return SQLiteBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_backend",
    "SQLiteBackend",
]
