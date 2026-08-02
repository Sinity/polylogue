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


def test_known_field_unread_observation_is_written(tmp_path: Path) -> None:
    """Regression test for polylogue-sd9s: known_field_unread is a real,
    legitimately risky DriftClassification member (RISKY_CLASSIFICATIONS)
    and must persist, not be silently dropped by the ops-tier CHECK.
    """
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
            classification="known_field_unread",
            unseen_key_signature="",
            native_id_example="raw-1",
            raw_id="raw-1",
        ),
    ]
    written = record_schema_drift_observations_to_ops_sync(index_db, observations)
    assert written == 1

    conn = sqlite3.connect(str(ops_db))
    try:
        samples = list_schema_drift_samples(conn, limit=100)
    finally:
        conn.close()
    assert [s.classification for s in samples] == ["known_field_unread"]


def test_one_rejected_observation_does_not_swallow_the_rest_of_the_batch(tmp_path: Path) -> None:
    """polylogue-sd9s: before this fix, the per-batch write loop caught
    sqlite3.Error around the WHOLE for-loop and returned 0 the instant any
    single observation's INSERT violated a CHECK constraint (e.g. an
    unrecognized ``origin`` token) -- discarding every other observation in
    the same call, not just the bad one. Each ``record_schema_drift_sample``
    call is already its own commit, so the valid rows before/after the bad
    one must both persist and both count toward the returned total.
    """
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
            unseen_key_signature="",
            native_id_example="raw-good-1",
            raw_id="raw-good-1",
        ),
        SchemaDriftObservation(
            origin="not-a-real-origin",
            element_kind="session_record",
            classification="new_field",
            unseen_key_signature="",
            native_id_example="raw-bad",
            raw_id="raw-bad",
        ),
        SchemaDriftObservation(
            origin="codex-session",
            element_kind="session_record",
            classification="field_changed",
            unseen_key_signature="",
            native_id_example="raw-good-2",
            raw_id="raw-good-2",
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
    assert raw_ids == {"raw-good-1", "raw-good-2"}
