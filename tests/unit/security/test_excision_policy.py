"""The source admission policy is a hash-only projection of durable intent."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import AssertionKind
from polylogue.security.excision import (
    ExcisionPolicyError,
    ExcisionPolicySnapshot,
    build_excision_policy_snapshot,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_snapshot_contains_identities_and_no_removed_literals(tmp_path: Path) -> None:
    initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
    initialize_archive_database(tmp_path / "audit.db", ArchiveTier.AUDIT)
    digest = hashlib.sha256(b"secret bytes").hexdigest()
    conn = sqlite3.connect(tmp_path / "user.db")
    try:
        with conn:
            conn.execute(
                "INSERT INTO assertions(assertion_id,target_ref,kind,value_json,status,created_at_ms,updated_at_ms) VALUES (?,?,?,?,?,?,?)",
                (
                    "excision-record-1",
                    "session:s1",
                    AssertionKind.EXCISION_RECORD.value,
                    json.dumps({"reason": "secret bytes", "removed_blob_hashes": [digest]}),
                    "active",
                    1,
                    1,
                ),
            )
    finally:
        conn.close()
    snapshot = build_excision_policy_snapshot(tmp_path, source_generation_id="g1")
    assert snapshot.removed_hashes == (bytes.fromhex(digest),)
    assert snapshot.assertion_refs == ("excision-record-1",)
    assert "secret bytes" not in json.dumps(snapshot, default=str)
    with pytest.raises(ExcisionPolicyError):
        snapshot.assert_admissible(bytes.fromhex(digest), source_path="reacquired.json")


def test_snapshot_digest_changes_when_durable_generation_changes() -> None:
    first = ExcisionPolicySnapshot((), (), 1, 0, "a" * 64, "g1")
    second = ExcisionPolicySnapshot((), (), 2, 0, "a" * 64, "g1")
    assert first.digest != second.digest
