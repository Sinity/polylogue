"""Tests for the ops.db drift-sample bridge (polylogue-da1).

Mirrors ``sample_fts_drift_to_ops_sync``'s contract test
(tests/unit/daemon/test_fts_identity_convergence.py): best-effort, sibling
ops.db, silently returns 0 on a missing tier rather than raising.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.schemas.drift_sentinel import SchemaDriftObservation
from polylogue.schemas.drift_sentinel_sampling import record_schema_drift_observations_to_ops_sync
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.ops_write import list_schema_drift_samples
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_writes_one_row_per_observation_to_sibling_ops_db(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    index_db.touch()
    ops_db = tmp_path / "ops.db"
    conn = sqlite3.connect(str(ops_db))
    try:
        initialize_archive_tier(conn, ArchiveTier.OPS)
    finally:
        conn.close()

    observations = [
        SchemaDriftObservation(
            origin="claude-code-session",
            element_kind="session_record",
            classification="new_field",
            unseen_key_signature="metadata.newThing",
            native_id_example="raw-1",
            raw_id="raw-1",
        ),
        SchemaDriftObservation(
            origin="codex-session",
            element_kind="session_record",
            classification="field_changed",
            unseen_key_signature="",
            native_id_example="raw-2",
            raw_id="raw-2",
        ),
    ]

    written = record_schema_drift_observations_to_ops_sync(index_db, observations)
    assert written == 2

    conn = sqlite3.connect(str(ops_db))
    try:
        samples = list_schema_drift_samples(conn, limit=100)
    finally:
        conn.close()
    raw_ids = {sample.raw_id for sample in samples}
    assert raw_ids == {"raw-1", "raw-2"}


def test_returns_zero_when_ops_tier_missing(tmp_path: Path) -> None:
    """No ops.db sibling (synthetic single-file archive) is swallowed, not raised."""
    index_db = tmp_path / "index.db"
    index_db.touch()
    observations = [
        SchemaDriftObservation(
            origin="claude-code-session",
            element_kind="session_record",
            classification="new_field",
            unseen_key_signature="x",
            native_id_example="raw-1",
            raw_id="raw-1",
        ),
    ]
    written = record_schema_drift_observations_to_ops_sync(index_db, observations)
    assert written == 0


def test_returns_zero_for_empty_observation_list(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    written = record_schema_drift_observations_to_ops_sync(index_db, [])
    assert written == 0
