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


# Representative index selectivities for a mature archive, captured from a live
# ~470K-block index (2026-07-19) and rounded. A freshly bootstrapped database
# has no sqlite_stat1, and without it the planner prefers low-cardinality
# equality indexes (idx_blocks_type_tool: ~100K rows/key) over session-scoped
# ones (idx_blocks_session_position: ~350 rows/key) for writer-hot queries like
# action_pairs_refresh_sql — turning per-session maintenance into full
# tool_use-population scans that grow with archive size (O(N^2) over a bulk
# rebuild; measured >20x replay slowdown before ANALYZE). Seeding stat1 at
# bootstrap makes plans correct from the first write; later ANALYZE / PRAGMA
# optimize passes replace these rows with measured values. Only relative
# magnitudes matter here.
PLANNER_STAT1_SEED_SQL = """
ANALYZE;
INSERT OR REPLACE INTO sqlite_stat1(tbl, idx, stat) VALUES
  ('blocks', 'idx_blocks_session_position', '500000 350 2 1'),
  ('blocks', 'idx_blocks_content_hash', '500000 2'),
  ('blocks', 'idx_blocks_search_text_populated', '500000 1 1'),
  ('blocks', 'idx_blocks_tool_id', '380000 3'),
  ('blocks', 'idx_blocks_tool_result_outcome', '190000 190000 63000 21000 60 1 1'),
  ('blocks', 'idx_blocks_type', '500000 100000'),
  ('blocks', 'idx_blocks_type_tool', '500000 100000 4000'),
  ('blocks', 'sqlite_autoindex_blocks_1', '500000 1'),
  ('blocks', 'sqlite_autoindex_blocks_2', '500000 1 1'),
  ('messages', 'idx_messages_session_position', '500000 350 1 1'),
  ('messages', 'idx_messages_session_sortkey', '500000 350 340 2 1'),
  ('messages', 'idx_messages_parent', '200000 1'),
  ('messages', 'idx_messages_session_role', '500000 350 110'),
  ('messages', 'idx_messages_role', '500000 125000'),
  ('messages', 'idx_messages_session_material_origin', '500000 350 105'),
  ('messages', 'idx_messages_message_type', '500000 83000'),
  ('messages', 'idx_messages_material_origin', '500000 62000'),
  ('messages', 'idx_messages_embedding_prose', '110000 75 1 1 1 1'),
  ('messages', 'idx_messages_active_path', '500000 350 350 1'),
  ('messages', 'idx_messages_active_leaf', '1500 1 1'),
  ('messages', 'sqlite_autoindex_messages_1', '500000 1'),
  ('messages', 'sqlite_autoindex_messages_2', '500000 350 1 1'),
  ('action_pairs', 'idx_action_pairs_session_order', '190000 140 1 1'),
  ('action_pairs', 'idx_action_pairs_message', '190000 1'),
  ('action_pairs', 'idx_action_pairs_tool', '190000 1600 35 1'),
  ('action_pairs', 'idx_action_pairs_semantic', '190000 19000 35 1'),
  ('action_pairs', 'idx_action_pairs_path', '33000 6 3 1'),
  ('action_pairs', 'idx_action_pairs_outcome', '50000 25000 6000 26 1'),
  ('action_pairs', 'sqlite_autoindex_action_pairs_1', '190000 1');
ANALYZE sqlite_master;
"""


__all__ = [
    "PLANNER_STAT1_SEED_SQL",
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
