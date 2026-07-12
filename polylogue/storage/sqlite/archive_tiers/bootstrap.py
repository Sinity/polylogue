"""Fresh bootstrap helpers for archive databases."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import DURABLE_MIGRATION_TIERS, migrate_archive_tier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

DurabilityClass = Literal["irreplaceable", "rebuildable", "expensive_rebuild", "human", "disposable"]


@dataclass(frozen=True, slots=True)
class ArchiveTierSpec:
    """Runtime metadata for one archive database file."""

    tier: ArchiveTier
    filename: str
    durability: DurabilityClass
    backup_required: bool

    @property
    def version(self) -> int:
        return ARCHIVE_VERSION_BY_TIER[self.tier]

    @property
    def ddl(self) -> str:
        return ARCHIVE_DDL_BY_TIER[self.tier]


ARCHIVE_TIER_SPECS: dict[ArchiveTier, ArchiveTierSpec] = {
    ArchiveTier.SOURCE: ArchiveTierSpec(
        ArchiveTier.SOURCE,
        filename="source.db",
        durability="irreplaceable",
        backup_required=True,
    ),
    ArchiveTier.INDEX: ArchiveTierSpec(
        ArchiveTier.INDEX,
        filename="index.db",
        durability="rebuildable",
        backup_required=False,
    ),
    ArchiveTier.EMBEDDINGS: ArchiveTierSpec(
        ArchiveTier.EMBEDDINGS,
        filename="embeddings.db",
        durability="expensive_rebuild",
        backup_required=True,
    ),
    ArchiveTier.USER: ArchiveTierSpec(
        ArchiveTier.USER,
        filename="user.db",
        durability="human",
        backup_required=True,
    ),
    ArchiveTier.OPS: ArchiveTierSpec(
        ArchiveTier.OPS,
        filename="ops.db",
        durability="disposable",
        backup_required=False,
    ),
}


def archive_tier_spec(tier: ArchiveTier) -> ArchiveTierSpec:
    """Return the database-file spec for one durability tier."""
    return ARCHIVE_TIER_SPECS[tier]


def initialize_archive_tier(conn: sqlite3.Connection, tier: ArchiveTier) -> None:
    """Initialize a fresh archive tier database on an already-open connection."""
    spec = archive_tier_spec(tier)
    conn.execute("PRAGMA foreign_keys = ON")
    if tier is ArchiveTier.EMBEDDINGS:
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            raise RuntimeError("archive embeddings initialization requires sqlite-vec") from error
    conn.executescript(spec.ddl)
    if tier is ArchiveTier.OPS:
        _ensure_ops_runtime_columns(conn)
        _ensure_ops_cursor_lag_sample_columns(conn)
        _ensure_ops_embedding_catchup_run_columns(conn)
    if tier is ArchiveTier.INDEX:
        from polylogue.storage.sqlite.archive_tiers.pricing_seed import seed_price_catalog

        seed_price_catalog(conn)
    conn.execute(f"PRAGMA user_version = {spec.version}")
    conn.commit()


def _ensure_ops_runtime_columns(conn: sqlite3.Connection) -> None:
    """Ensure disposable OPS-tier databases have the current cursor shape."""
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(ingest_cursor)")}
    additions = {
        "record_count": "INTEGER NOT NULL DEFAULT 0 CHECK(record_count >= 0)",
        "last_record_ts_ms": "INTEGER",
        "failure_count": "INTEGER NOT NULL DEFAULT 0 CHECK(failure_count >= 0)",
        "next_retry_at": "TEXT",
        "excluded": "INTEGER NOT NULL DEFAULT 0 CHECK(excluded IN (0, 1))",
    }
    for name, definition in additions.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE ingest_cursor ADD COLUMN {name} {definition}")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ingest_cursor_attention
        ON ingest_cursor(failure_count, excluded, source_path)
        """
    )


def _ensure_ops_cursor_lag_sample_columns(conn: sqlite3.Connection) -> None:
    """Ensure disposable OPS-tier cursor lag samples carry family rollups."""
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(cursor_lag_samples)")}
    additions = {
        "family": "TEXT",
        "stuck_file_count": "INTEGER NOT NULL DEFAULT 1 CHECK(stuck_file_count >= 0)",
        "p50_lag_ms": "INTEGER NOT NULL DEFAULT 0 CHECK(p50_lag_ms >= 0)",
        "p95_lag_ms": "INTEGER NOT NULL DEFAULT 0 CHECK(p95_lag_ms >= 0)",
    }
    for name, definition in additions.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE cursor_lag_samples ADD COLUMN {name} {definition}")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cursor_lag_samples_family_time
        ON cursor_lag_samples(family, sampled_at_ms DESC)
        """
    )


def _ensure_ops_embedding_catchup_run_columns(conn: sqlite3.Connection) -> None:
    """Ensure disposable OPS-tier embedding catch-up rows carry run outcomes."""
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(embedding_catchup_runs)")}
    additions = {
        "embedded_sessions": "INTEGER NOT NULL DEFAULT 0 CHECK(embedded_sessions >= 0)",
        "error_count": "INTEGER NOT NULL DEFAULT 0 CHECK(error_count >= 0)",
    }
    for name, definition in additions.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE embedding_catchup_runs ADD COLUMN {name} {definition}")


def initialize_archive_database(
    path: Path, tier: ArchiveTier, *, migration_backup_manifest: Path | None = None
) -> None:
    """Create or initialize one archive tier database file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        current_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        expected_version = archive_tier_spec(tier).version
        if current_version == expected_version:
            # ops.db is disposable and evolves through idempotent additive DDL
            # without version bumps. Re-apply it so existing same-version
            # archives receive newly introduced tables and indexes.
            if tier is ArchiveTier.OPS:
                initialize_archive_tier(conn, tier)
            return
        if (
            current_version != 0
            and current_version < expected_version
            and tier in DURABLE_MIGRATION_TIERS
            and migration_backup_manifest is not None
        ):
            migrate_archive_tier(conn, tier, backup_manifest=migration_backup_manifest)
            return
        if current_version != 0:
            if current_version < expected_version and tier in DURABLE_MIGRATION_TIERS:
                raise RuntimeError(
                    f"{path.name} schema version {current_version} is older than the current {tier.value} tier "
                    f"version {expected_version}; run an explicit durable-tier migration with a verified backup "
                    "manifest"
                )
            raise RuntimeError(
                f"{path.name} schema version {current_version} is not the current {tier.value} tier version "
                f"{expected_version}; move it aside and rebuild the archive root"
            )
        initialize_archive_tier(conn, tier)
    finally:
        conn.close()


def initialize_active_archive_root(root: Path) -> None:
    """Create or initialize every tier database in an archive root."""
    for spec in ARCHIVE_TIER_SPECS.values():
        initialize_archive_database(root / spec.filename, spec.tier)


__all__ = [
    "ARCHIVE_TIER_SPECS",
    "DurabilityClass",
    "ArchiveTierSpec",
    "initialize_active_archive_root",
    "initialize_archive_database",
    "initialize_archive_tier",
    "archive_tier_spec",
]
