"""Compose the storage-owned message FTS derivation for daemon convergence.

The operations boundary owns active-generation routing.  The FTS adapter stays
in ``storage`` and speaks the daemon kernel protocol structurally, so neither
the FTS SQL nor its output relation depends on daemon implementation details.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from polylogue.daemon.derivation import DerivationFrame
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.fts.derivation import GLOBAL_PARTITION, FtsDerivationAdapter
from polylogue.storage.introspection import table_exists as _table_exists
from polylogue.storage.sqlite.connection_profile import open_daemon_connection, open_readonly_connection

__all__ = [
    "FTS_ORPHAN_INTERVAL_S",
    "archive_fts_surface",
    "fts_triggers_present",
    "make_fts_derivation",
    "make_fts_frame",
]


# Global residue is defense-in-depth for interrupted/malformed historical
# writes. Session partitions are the normal path, so this process-local cadence
# does not create durable freshness authority or suppress required work.
FTS_ORPHAN_INTERVAL_S = 60.0 * 60.0


def make_fts_derivation(
    index_db_path: Path,
    *,
    archive_root: Path,
    orphan_interval_s: float | None = FTS_ORPHAN_INTERVAL_S,
    monotonic: Callable[[], float] = time.monotonic,
) -> FtsDerivationAdapter:
    """Return one active-generation FTS adapter for the shared daemon owner."""
    del index_db_path

    def active_generation_path() -> Path:
        return resolve_active_index_path(archive_root).resolve()

    def read_connection() -> sqlite3.Connection:
        return open_readonly_connection(active_generation_path(), timeout_class="background-read")

    def write_connection() -> sqlite3.Connection:
        return open_daemon_connection(active_generation_path(), archive_root=archive_root)

    def generation_binding() -> str:
        return str(active_generation_path())

    return FtsDerivationAdapter(
        read_connection,
        write_connection,
        generation_binding=generation_binding,
        orphan_interval_s=orphan_interval_s,
        monotonic=monotonic,
    )


def make_fts_frame(
    index_db_path: Path,
    *,
    archive_root: Path,
    scope: Sequence[str] | None = None,
) -> DerivationFrame:
    """Bind an FTS pass to the active path, generation and identity recipe."""
    del index_db_path
    adapter = FtsDerivationAdapter()
    return DerivationFrame(
        archive_root=str(archive_root),
        source_revision=f"index-generation:{resolve_active_index_path(archive_root).resolve()}",
        recipe_versions={adapter.domain: adapter.recipe_id},
        scope=None if scope is None else tuple(dict.fromkeys(str(session_id) for session_id in scope)),
    )


_ARCHIVE_BLOCKS_FTS_TRIGGERS = ("messages_fts_ai", "messages_fts_ad", "messages_fts_au")


def fts_triggers_present(conn: sqlite3.Connection, trigger_names: tuple[str, ...]) -> bool:
    placeholders = ",".join("?" for _ in trigger_names)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        trigger_names,
    ).fetchall()
    present = {row[0] for row in rows}
    return all(name in present for name in trigger_names)


def archive_fts_surface(conn: sqlite3.Connection) -> dict[str, int | bool | str | None]:
    """Project the FTS domain's global authoritative inspection for status."""
    source_exists = _table_exists(conn, "blocks")
    exists = _table_exists(conn, "messages_fts")
    if not source_exists:
        ready = not exists
        return {
            "source_exists": source_exists,
            "exists": exists,
            "source_rows": 0,
            "indexed_rows": 0,
            "triggers_present": exists and fts_triggers_present(conn, _ARCHIVE_BLOCKS_FTS_TRIGGERS),
            "missing_rows": 0,
            "excess_rows": 0,
            "duplicate_rows": 0,
            "identity_mismatch_rows": 0,
            "ready": ready,
            "exact": True,
        }
    inspection = FtsDerivationAdapter().inspect_partition(conn, GLOBAL_PARTITION)
    return {
        "source_exists": source_exists,
        "exists": exists,
        "source_rows": inspection.required_rows,
        "indexed_rows": inspection.present_rows,
        "triggers_present": inspection.triggers_compatible,
        "missing_rows": inspection.missing_rows,
        "excess_rows": inspection.excess_rows,
        "duplicate_rows": inspection.duplicate_rows,
        "identity_mismatch_rows": inspection.wrong_identity_rows,
        "ready": inspection.valid,
        "exact": True,
    }
