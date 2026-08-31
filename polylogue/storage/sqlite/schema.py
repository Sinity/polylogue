"""Canonical SQLite schema runtime for sync and async backends.

Polylogue has no in-place schema upgrade chain. Databases are either at the
canonical :data:`SCHEMA_VERSION` (open as-is), fresh (bootstrap from
:data:`SCHEMA_DDL`), or rejected. The operator handles rejection by
re-ingesting from source; the runtime never patches an out-of-band shape
into the canonical one.
"""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.core.errors import SchemaVersionMismatchError
from polylogue.storage.sqlite.archive_tiers.index_convergence import (
    apply_index_benign_ddl_convergence,
    apply_index_benign_ddl_convergence_async,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_async, ensure_runtime_indexes_sync
from polylogue.storage.sqlite.schema_bootstrap import (
    PLANNER_STAT1_SEED_SQL,
    SCHEMA_DDL,
    SCHEMA_VERSION,
    assert_derived_schema_identity,
    assert_derived_schema_identity_async,
    capture_schema_snapshot,
    capture_schema_snapshot_async,
    decide_schema_bootstrap,
    ensure_vec0_table,
    ensure_vec0_table_async,
    schema_version_mismatch_message,
    stamp_derived_schema_identity,
)
from polylogue.storage.sqlite.schema_manifest import assert_schema_manifest


def assert_supported_archive_layout(conn: sqlite3.Connection) -> None:
    """Reject archive layouts that cannot be written safely.

    Polylogue has no in-place schema upgrade chain; layout is determined entirely by
    the on-disk ``user_version``. Anything outside ``{0, SCHEMA_VERSION}`` is
    rejected and the operator re-ingests from source.
    """
    snapshot = capture_schema_snapshot(conn)
    if snapshot.current_version not in (0, SCHEMA_VERSION):
        raise SchemaVersionMismatchError(
            schema_version_mismatch_message(snapshot.current_version),
            current_version=snapshot.current_version,
            expected_version=SCHEMA_VERSION,
        )
    if snapshot.current_version == SCHEMA_VERSION:
        assert_schema_manifest(conn, ArchiveTier.INDEX)


def assert_readable_archive_layout(conn: sqlite3.Connection, *, generation_id: str | None = None) -> None:
    """Read-only mode counterpart of :func:`assert_supported_archive_layout`."""
    snapshot = capture_schema_snapshot(conn)
    if snapshot.current_version not in (0, SCHEMA_VERSION):
        lifecycle_action = "upgrade_runtime" if snapshot.current_version > SCHEMA_VERSION else "rebuild_index"
        raise SchemaVersionMismatchError(
            schema_version_mismatch_message(snapshot.current_version, generation_id=generation_id),
            current_version=snapshot.current_version,
            expected_version=SCHEMA_VERSION,
            generation_id=generation_id,
            lifecycle_action=lifecycle_action,
        )
    if snapshot.current_version == SCHEMA_VERSION:
        missing_error: sqlite3.Error | None
        try:
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(sessions)")}
        except sqlite3.Error as exc:
            columns = set()
            missing_error = exc
        else:
            missing_error = None
        if missing_error is not None or "reported_cost_usd" not in columns:
            suffix = f" Generation {generation_id}" if generation_id is not None else ""
            raise SchemaVersionMismatchError(
                f"Archive index schema shape does not match runtime version {SCHEMA_VERSION}.{suffix} "
                "Rebuild the derived index from source with `polylogue ops maintenance rebuild-index`.",
                current_version=snapshot.current_version,
                expected_version=SCHEMA_VERSION,
                generation_id=generation_id,
                lifecycle_action="rebuild_index",
            ) from missing_error


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the database is at the current schema version.

    Polylogue has no versioned in-place upgrade chain. Databases with a mismatched
    schema version are rejected; the operator re-ingests from source.
    """
    snapshot = capture_schema_snapshot(conn)
    decision = decide_schema_bootstrap(snapshot)

    if decision.action == "create_fresh":
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_DDL)
        ensure_vec0_table(conn)
        ensure_runtime_indexes_sync(conn)
        apply_index_benign_ddl_convergence(conn)
        conn.executescript(PLANNER_STAT1_SEED_SQL)
        conn.execute("PRAGMA optimize")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        stamp_derived_schema_identity(conn, "index")
        conn.commit()
        return

    if decision.action == "version_mismatch":
        current = decision.current_version or 0
        raise SchemaVersionMismatchError(
            schema_version_mismatch_message(current),
            current_version=current,
            expected_version=SCHEMA_VERSION,
        )

    # open_as_is — vec0 still needs to be ensured per-connection because the
    # extension may have been newly loaded since fresh init.
    assert_derived_schema_identity(conn, "index")
    ensure_vec0_table(conn)
    ensure_runtime_indexes_sync(conn)
    apply_index_benign_ddl_convergence(conn)


async def ensure_schema_async(conn: aiosqlite.Connection) -> None:
    """Async counterpart of :func:`_ensure_schema`. Same policy."""
    snapshot = await capture_schema_snapshot_async(conn)
    decision = decide_schema_bootstrap(snapshot)

    if decision.action == "create_fresh":
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.executescript(SCHEMA_DDL)
        await ensure_vec0_table_async(conn)
        await ensure_runtime_indexes_async(conn)
        await apply_index_benign_ddl_convergence_async(conn)
        await conn.executescript(PLANNER_STAT1_SEED_SQL)
        await conn.execute("PRAGMA optimize")
        await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        from polylogue.storage.sqlite.archive_tiers.schema_identity import DerivedTier, derived_schema_identity

        await conn.execute(
            "INSERT INTO schema_identity(tier, identity) VALUES (?, ?) "
            "ON CONFLICT(tier) DO UPDATE SET identity=excluded.identity",
            (DerivedTier.INDEX.value, derived_schema_identity(DerivedTier.INDEX)),
        )
        await conn.commit()
        return

    if decision.action == "version_mismatch":
        current = decision.current_version or 0
        raise SchemaVersionMismatchError(
            schema_version_mismatch_message(current),
            current_version=current,
            expected_version=SCHEMA_VERSION,
        )

    await assert_derived_schema_identity_async(conn, "index")
    await ensure_vec0_table_async(conn)
    await ensure_runtime_indexes_async(conn)
    await apply_index_benign_ddl_convergence_async(conn)


__all__ = [
    "SCHEMA_DDL",
    "SCHEMA_VERSION",
    "_ensure_schema",
    "assert_readable_archive_layout",
    "assert_supported_archive_layout",
    "ensure_schema_async",
    "ensure_vec0_table",
]
