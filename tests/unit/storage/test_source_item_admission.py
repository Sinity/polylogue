"""Atomic raw/source-item admission laws."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Origin, Provider
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.raw_admission import (
    PendingPreParseRawAdmissionRequest,
    RawAdmissionPlan,
    SourceItemAdmission,
    execute_source_item_admission,
    plan_raw_admission,
)
from polylogue.storage.sqlite.archive_tiers.source_items import publish_source_generation

_PAYLOAD = b'{"synthetic":"source-item"}\n'
_BLOB_HASH = hashlib.sha256(_PAYLOAD).digest()


def _archive(tmp_path: Path) -> tuple[sqlite3.Connection, str, str, RawAdmissionPlan]:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    BlobStore(root / "blob").write_from_bytes(_PAYLOAD)
    conn = sqlite3.connect(root / "source.db")
    conn.execute("PRAGMA foreign_keys = ON")
    (item_id,) = publish_source_generation(
        conn,
        source_generation_id="synthetic-generation",
        manifest_digest="a" * 64,
        addressing_mode="physical-file-v1",
        coordinates=("capture.json",),
        input_blob_hashes={"capture.json": _BLOB_HASH},
        enumeration_fingerprint="b" * 64,
        observed_at_ms=1,
    )
    request = PendingPreParseRawAdmissionRequest(
        origin=Origin.CLAUDE_CODE_SESSION,
        capture_mode=Provider.CLAUDE_CODE,
        source_path="/synthetic/capture.json",
        source_index=0,
        blob_hash=_BLOB_HASH,
        blob_size=len(_PAYLOAD),
        acquired_at_ms=2,
    )
    return conn, "synthetic-generation", item_id, plan_raw_admission(request)


def _member(item_id: str, *, coordinate: str = "record:0", entry_ordinal: int | None = None) -> SourceItemAdmission:
    return SourceItemAdmission(
        source_generation_id="synthetic-generation",
        source_item_id=item_id,
        record_coordinate=coordinate,
        entry_ordinal=entry_ordinal,
    )


def test_late_membership_error_rolls_back_raw_and_membership_together(tmp_path: Path) -> None:
    conn, _generation, item_id, plan = _archive(tmp_path)
    conn.execute("BEGIN")
    with pytest.raises(ValueError, match="complete container coordinate"):
        execute_source_item_admission(conn, plan, _member(item_id, entry_ordinal=3))

    assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone() == (0,)
    assert conn.in_transaction
    conn.rollback()
    conn.close()


def test_duplicate_admission_is_one_raw_row_and_one_membership_edge(tmp_path: Path) -> None:
    conn, _generation, item_id, plan = _archive(tmp_path)
    conn.execute("BEGIN")
    first = execute_source_item_admission(conn, plan, _member(item_id))
    duplicate = execute_source_item_admission(conn, plan, _member(item_id))
    conn.commit()

    assert first.raw_id == duplicate.raw_id == plan.raw_id
    assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
    assert conn.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone() == (1,)
    assert conn.execute("SELECT raw_id, raw_blob_hash FROM source_item_raw_members").fetchone() == (
        plan.raw_id,
        _BLOB_HASH,
    )
    conn.close()


def test_retired_raw_membership_cannot_be_readmitted(tmp_path: Path) -> None:
    conn, _generation, item_id, plan = _archive(tmp_path)
    conn.execute("BEGIN")
    execute_source_item_admission(conn, plan, _member(item_id))
    conn.commit()
    conn.execute("DELETE FROM raw_sessions WHERE raw_id = ?", (plan.raw_id,))
    assert conn.execute("SELECT raw_id, raw_blob_hash FROM source_item_raw_members").fetchone() == (
        None,
        _BLOB_HASH,
    )
    conn.commit()
    conn.execute("BEGIN")
    with pytest.raises(ValueError, match="retired; readmission is forbidden"):
        execute_source_item_admission(conn, plan, _member(item_id))
    assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)
    assert conn.execute("SELECT raw_id, raw_blob_hash FROM source_item_raw_members").fetchone() == (
        None,
        _BLOB_HASH,
    )
    conn.rollback()
    conn.close()
