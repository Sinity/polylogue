"""Reconstruct a historical durable source tier from the migration files.

Durable fixtures build an old schema by subtracting from the CURRENT DDL. Every
hand-written removal list went stale the moment a migration added a table,
index or column, so the set is derived here instead.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

__all__ = ["reset_source_fixture_to_version"]

#: The source-migration slot that introduced each retired table.
_RETIRED_SOURCE_INTRODUCED_AT: dict[str, int] = {
    "raw_live_source_reconciliation_receipts": 18,
    "raw_membership_writeback_receipts": 19,
    "raw_append_chain_backfill_receipts": 20,
    "raw_byte_duplicate_supersession_receipts": 23,
    "raw_quarantine_group_dedup_receipts": 25,
    "raw_unknown_export_reclassification_receipts": 26,
    "raw_non_session_duplicate_exclusion_receipts": 27,
    "raw_failure_disposition_receipts": 29,
}

#: The post-v41 shape a migrated tier still carries for each retired table.
_RETIRED_SOURCE_HISTORICAL_DDL: dict[str, str] = {
    "raw_live_source_reconciliation_receipts": """
CREATE TABLE IF NOT EXISTS raw_live_source_reconciliation_receipts (
    raw_id                      TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    verdict                     TEXT NOT NULL CHECK(verdict IN ('exact_match', 'codex_header_strip_match')),
    previous_revision_authority TEXT NOT NULL,
    source_path                 TEXT NOT NULL,
    blob_hash                   BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                   INTEGER NOT NULL CHECK(blob_size >= 0),
    compared_at_ms              INTEGER NOT NULL CHECK(compared_at_ms >= 0),
    tool_version                TEXT NOT NULL,
    backup_manifest_path        TEXT NOT NULL,
    detail                      TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_live_source_reconciliation_receipts_compared_at
ON raw_live_source_reconciliation_receipts(compared_at_ms);
""",
    "raw_membership_writeback_receipts": """
CREATE TABLE IF NOT EXISTS raw_membership_writeback_receipts (
    raw_id                      TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key          TEXT NOT NULL,
    provider_session_id         TEXT NOT NULL,
    membership_decision         TEXT NOT NULL,
    previous_revision_authority TEXT NOT NULL,
    promoted_at_ms              INTEGER NOT NULL CHECK(promoted_at_ms >= 0),
    tool_version                TEXT NOT NULL,
    backup_manifest_path        TEXT NOT NULL,
    detail                      TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_membership_writeback_receipts_promoted_at
ON raw_membership_writeback_receipts(promoted_at_ms);
""",
    "raw_append_chain_backfill_receipts": """
CREATE TABLE IF NOT EXISTS raw_append_chain_backfill_receipts (
    raw_id                           TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key                TEXT,
    source_path                      TEXT NOT NULL,
    blob_hash                        BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                        INTEGER NOT NULL CHECK(blob_size >= 0),
    append_start_offset              INTEGER NOT NULL CHECK(append_start_offset >= 0),
    append_end_offset                INTEGER NOT NULL CHECK(append_end_offset > append_start_offset),
    matched_after_codex_header_strip INTEGER NOT NULL CHECK(matched_after_codex_header_strip IN (0, 1)),
    previous_revision_authority      TEXT NOT NULL,
    compared_at_ms                    INTEGER NOT NULL CHECK(compared_at_ms >= 0),
    tool_version                      TEXT NOT NULL,
    backup_manifest_path              TEXT NOT NULL,
    detail                            TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_append_chain_backfill_receipts_compared_at
ON raw_append_chain_backfill_receipts(compared_at_ms);
""",
    "raw_byte_duplicate_supersession_receipts": """
CREATE TABLE IF NOT EXISTS raw_byte_duplicate_supersession_receipts (
    raw_id                      TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    blob_hash                   BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                   INTEGER NOT NULL CHECK(blob_size >= 0),
    duplicate_of_raw_id         TEXT NOT NULL,
    duplicate_of_session_id     TEXT NOT NULL,
    previous_revision_authority TEXT NOT NULL CHECK(previous_revision_authority IN ('asserted', 'byte_proven', 'quarantined')),
    promoted_at_ms              INTEGER NOT NULL CHECK(promoted_at_ms >= 0),
    tool_version                TEXT NOT NULL,
    backup_manifest_path        TEXT NOT NULL,
    detail                      TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_byte_duplicate_supersession_receipts_promoted_at
ON raw_byte_duplicate_supersession_receipts(promoted_at_ms);

CREATE INDEX IF NOT EXISTS idx_raw_byte_duplicate_supersession_receipts_duplicate_of
ON raw_byte_duplicate_supersession_receipts(duplicate_of_raw_id);
""",
    "raw_failure_disposition_receipts": """
CREATE TABLE IF NOT EXISTS raw_failure_disposition_receipts (
    raw_id                     TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    artifact_id                TEXT NOT NULL UNIQUE REFERENCES raw_artifacts(artifact_id),
    origin                     TEXT NOT NULL,
    source_path                TEXT NOT NULL,
    source_index               INTEGER NOT NULL,
    blob_hash                  BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                  INTEGER NOT NULL CHECK(blob_size >= 0),
    previous_parse_error       TEXT NOT NULL,
    previous_validation_status TEXT,
    previous_artifact_kind     TEXT NOT NULL,
    previous_support_status    TEXT NOT NULL,
    previous_classification_reason TEXT NOT NULL,
    disposition_kind           TEXT NOT NULL CHECK(disposition_kind IN (
        'terminal_corrupt_input',
        'terminal_unsupported_shape'
    )),
    manifest_sha256            TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
    disposed_at_ms             INTEGER NOT NULL CHECK(disposed_at_ms >= 0),
    tool_version               TEXT NOT NULL,
    backup_manifest_path       TEXT NOT NULL,
    detail                     TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_failure_disposition_receipts_disposed_at
ON raw_failure_disposition_receipts(disposed_at_ms);
""",
    "raw_non_session_duplicate_exclusion_receipts": """
CREATE TABLE IF NOT EXISTS raw_non_session_duplicate_exclusion_receipts (
    raw_id                     TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    blob_hash                  BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                  INTEGER NOT NULL CHECK(blob_size >= 0),
    indexed_twin_raw_id        TEXT NOT NULL,
    indexed_twin_session_id    TEXT NOT NULL,
    parser_fingerprint         TEXT NOT NULL,
    excluded_at_ms             INTEGER NOT NULL CHECK(excluded_at_ms >= 0),
    tool_version               TEXT NOT NULL,
    detail                     TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_non_session_duplicate_exclusion_receipts_twin
ON raw_non_session_duplicate_exclusion_receipts(indexed_twin_raw_id);
""",
    "raw_quarantine_group_dedup_receipts": """
CREATE TABLE IF NOT EXISTS raw_quarantine_group_dedup_receipts (
    raw_id                     TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    source_path                TEXT NOT NULL,
    blob_hash                  BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                  INTEGER NOT NULL CHECK(blob_size >= 0),
    representative_raw_id      TEXT NOT NULL,
    representative_session_id  TEXT NOT NULL,
    promoted_at_ms              INTEGER NOT NULL CHECK(promoted_at_ms >= 0),
    tool_version                TEXT NOT NULL,
    backup_manifest_path        TEXT NOT NULL,
    detail                      TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_quarantine_group_dedup_receipts_promoted_at
ON raw_quarantine_group_dedup_receipts(promoted_at_ms);

CREATE INDEX IF NOT EXISTS idx_raw_quarantine_group_dedup_receipts_representative
ON raw_quarantine_group_dedup_receipts(representative_raw_id);
""",
    "raw_unknown_export_reclassification_receipts": """
CREATE TABLE IF NOT EXISTS raw_unknown_export_reclassification_receipts (
    raw_id                  TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    previous_origin         TEXT NOT NULL CHECK(previous_origin = 'unknown-export'),
    new_origin              TEXT NOT NULL CHECK(new_origin = 'chatgpt-export'),
    previous_capture_mode   TEXT,
    new_capture_mode        TEXT NOT NULL CHECK(new_capture_mode = 'chatgpt'),
    embedded_provider       TEXT NOT NULL CHECK(embedded_provider = 'chatgpt'),
    source_path             TEXT NOT NULL,
    blob_hash               BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size               INTEGER NOT NULL CHECK(blob_size >= 0),
    reclassified_at_ms      INTEGER NOT NULL CHECK(reclassified_at_ms >= 0),
    tool_version            TEXT NOT NULL,
    backup_manifest_path    TEXT NOT NULL,
    index_reparse_required  INTEGER NOT NULL CHECK(index_reparse_required = 1),
    detail                  TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_unknown_export_reclassification_receipts_reclassified_at
ON raw_unknown_export_reclassification_receipts(reclassified_at_ms);
""",
}


def reset_source_fixture_to_version(conn: sqlite3.Connection, version: int) -> None:
    """Remove every schema object a source migration above *version* creates.

    The fixtures build a "stale" tier from the CURRENT DDL, so they start out
    carrying tables, indexes and triggers that a later migration will create
    again. Hand-maintained removal lists went stale every time a migration was
    added; this derives the set from the migration files themselves.

    Only objects INTRODUCED above the version are removed: a later migration
    that rebuilds an existing table issues its own CREATE TABLE, and dropping
    that would delete a table the fixture must still have.
    """
    migrations = Path(__file__).parents[2] / "polylogue" / "storage" / "sqlite" / "migrations" / "source"
    table_pattern = re.compile(r"CREATE TABLE (?:IF NOT EXISTS )?([A-Za-z_][A-Za-z0-9_]*)")
    index_pattern = re.compile(r"CREATE (?:UNIQUE )?INDEX (?:IF NOT EXISTS )?([A-Za-z_][A-Za-z0-9_]*)")
    view_pattern = re.compile(r"CREATE VIEW (?:IF NOT EXISTS )?([A-Za-z_][A-Za-z0-9_]*)")
    rebuild_pattern = re.compile(
        r"(?:DROP TABLE (?:IF EXISTS )?([A-Za-z_][A-Za-z0-9_]*)"
        r"|ALTER TABLE ([A-Za-z_][A-Za-z0-9_]*)\s+RENAME)",
        re.I,
    )
    below: set[str] = set()
    above: list[tuple[str, str]] = []
    for path in sorted(migrations.glob("*.sql")):
        slot = int(path.name.split("_", 1)[0])
        text = path.read_text(encoding="utf-8")
        # A create-copy-drop-rename rebuild issues CREATE TABLE for a table it
        # does not introduce. Treat a name the same file also drops or renames
        # as a rebuild, not an introduction, or the reset deletes a table the
        # fixture is required to have (raw_sessions is rebuilt this way).
        rebuilt = {name for m in rebuild_pattern.finditer(text) for name in m.groups() if name}
        created = [("table", m.group(1)) for m in table_pattern.finditer(text) if m.group(1) not in rebuilt]
        created += [("index", m.group(1)) for m in index_pattern.finditer(text)]
        created += [("view", m.group(1)) for m in view_pattern.finditer(text)]
        if slot <= version:
            below.update(name for _kind, name in created)
        else:
            above.extend(created)
    # Columns a later migration adds are present in the current DDL too, so an
    # ALTER TABLE ADD COLUMN above the fixture version fails with "duplicate
    # column name" unless the column is removed first.
    column_pattern = re.compile(r"ALTER TABLE ([A-Za-z_][A-Za-z0-9_]*)\s+ADD COLUMN\s+([A-Za-z_][A-Za-z0-9_]*)", re.I)
    columns_below: set[tuple[str, str]] = set()
    columns_above: list[tuple[str, str]] = []
    for path in sorted(migrations.glob("*.sql")):
        slot = int(path.name.split("_", 1)[0])
        found = [(m.group(1), m.group(2)) for m in column_pattern.finditer(path.read_text(encoding="utf-8"))]
        if slot <= version:
            columns_below.update(found)
        else:
            columns_above.extend(found)

    seen: set[str] = set()
    for kind, name in above:
        if kind == "view" and name not in below and name not in seen:
            seen.add(name)
            conn.execute(f"DROP VIEW IF EXISTS {name}")
    for kind, name in above:
        if name in below or name in seen:
            continue
        seen.add(name)
        if kind == "table":
            for dependent_kind in ("trigger", "index"):
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = ? AND sql LIKE ?", (dependent_kind, f"%{name}%")
                ).fetchall()
                for (object_name,) in rows:
                    conn.execute(f"DROP {dependent_kind.upper()} IF EXISTS {object_name}")
            conn.execute(f"DROP TABLE IF EXISTS {name}")
        elif kind == "index":
            conn.execute(f"DROP INDEX IF EXISTS {name}")
    dropped_tables = {name for kind, name in above if kind == "table" and name not in below}
    for table, column in dict.fromkeys(columns_above):
        if (table, column) in columns_below or table in dropped_tables:
            continue
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column in existing:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    _restore_retired_source_objects(conn, version)


def _restore_retired_source_objects(conn: sqlite3.Connection, version: int) -> None:
    """Re-create the retired objects a tier at *version* still carries.

    Fresh source generations omit them, so a fixture subtracted from current
    DDL starts without them, and the numbered chain above the fixture version
    still rebuilds them.
    """
    for table, introduced_at in _RETIRED_SOURCE_INTRODUCED_AT.items():
        if introduced_at > version:
            continue
        conn.executescript(_RETIRED_SOURCE_HISTORICAL_DDL[table])
