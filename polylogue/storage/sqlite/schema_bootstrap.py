"""Canonical index-tier bootstrap for sync and async SQLite backends.

Polylogue has no in-place schema upgrade chain. The runtime knows exactly one
index-tier schema shape: the DDL in
:mod:`polylogue.storage.sqlite.archive_tiers.index` at version
:data:`SCHEMA_VERSION`. Bootstrap accepts only:

* ``current_version == 0`` (a brand-new file) — create fresh.
* ``current_version == SCHEMA_VERSION`` — open as-is.
* anything else — refuse to open the file as an archive. The operator moves it
  aside and rebuilds from source; the runtime never patches an out-of-band
  shape into the canonical one.

The minimal :class:`SchemaSnapshot`/:func:`decide_schema_bootstrap` surface
is retained so the daemon health check and read-only open paths can classify
a database without opening it for writes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

import aiosqlite

from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL, INDEX_SCHEMA_VERSION

SCHEMA_DDL = INDEX_DDL
SCHEMA_VERSION = INDEX_SCHEMA_VERSION


@dataclass(frozen=True)
class SchemaSnapshot:
    """Database state needed to classify the canonical bootstrap branch."""

    current_version: int


@dataclass(frozen=True)
class SchemaBootstrapDecision:
    """Shared schema bootstrap branch chosen from a snapshot."""

    action: Literal["create_fresh", "open_as_is", "version_mismatch"]
    current_version: int | None = None


def schema_version_mismatch_message(current_version: int) -> str:
    if current_version > SCHEMA_VERSION:
        return (
            f"Database schema version {current_version} is newer than this Polylogue runtime expects "
            f"({SCHEMA_VERSION}). Update the installed Polylogue runtime to the build that created the "
            "database before opening it."
        )
    return (
        f"Database schema version {current_version} is not the expected archive version {SCHEMA_VERSION}. "
        "Move the index tier aside and rebuild it from source with `polylogue ops reset --index && polylogued run`."
    )


def decide_schema_bootstrap(snapshot: SchemaSnapshot) -> SchemaBootstrapDecision:
    """Choose the canonical schema bootstrap path for sync and async backends."""
    if snapshot.current_version == 0:
        return SchemaBootstrapDecision(action="create_fresh")
    if snapshot.current_version == SCHEMA_VERSION:
        return SchemaBootstrapDecision(action="open_as_is", current_version=snapshot.current_version)
    return SchemaBootstrapDecision(action="version_mismatch", current_version=snapshot.current_version)


def capture_schema_snapshot(conn: sqlite3.Connection) -> SchemaSnapshot:
    current_version = conn.execute("PRAGMA user_version").fetchone()[0]
    return SchemaSnapshot(current_version=current_version)


async def capture_schema_snapshot_async(conn: aiosqlite.Connection) -> SchemaSnapshot:
    cursor = await conn.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    current_version = row[0] if row else 0
    return SchemaSnapshot(current_version=current_version)


def ensure_vec0_table(conn: sqlite3.Connection) -> None:
    """Index bootstrap does not own vector storage."""
    del conn


async def ensure_vec0_table_async(conn: aiosqlite.Connection) -> None:
    del conn


__all__ = [
    "SCHEMA_DDL",
    "SCHEMA_VERSION",
    "SchemaBootstrapDecision",
    "SchemaSnapshot",
    "capture_schema_snapshot",
    "capture_schema_snapshot_async",
    "decide_schema_bootstrap",
    "ensure_vec0_table",
    "ensure_vec0_table_async",
    "schema_version_mismatch_message",
]
