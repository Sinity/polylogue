"""Resident execution for small CLI reads outside the query grammar."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from polylogue.operations.daemon_protocol import validate_operation_result
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder


def test_identity_reset_targets_resolve_on_the_pinned_archive(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "reset-targets").provider("codex").title("Target").save()
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        session_id = archive.list_summaries(limit=1)[0].session_id
        result = execute_read_operation(
            "session.identity-reset.targets",
            {"session": session_id},
            archive=archive,
            serving_identity="test",
        )
    validate_operation_result("session.identity-reset.targets", result)
    assert result["session_ids"] == [session_id]


def test_assertion_export_keeps_a_missing_user_tier_empty(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "assertion-export").provider("codex").title("Export").save()
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        result = execute_read_operation(
            "user.assertions.export",
            {},
            archive=archive,
            serving_identity="test",
        )
    validate_operation_result("user.assertions.export", result)
    assert result["items"] == []
    assert result["total"] == 0
    outcome = result["outcome"]
    assert isinstance(outcome, dict)
    assert outcome["state"] == "empty"


def test_identity_reset_source_path_uses_the_pinned_source_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "reset-source-pin").provider("codex").title("Target").save()
    raw_id = "raw-reset-source-pin"
    with sqlite3.connect(root / "index.db") as index:
        session_id = str(index.execute("SELECT session_id FROM sessions").fetchone()[0])
        index.execute("UPDATE sessions SET raw_id = ? WHERE session_id = ?", (raw_id, session_id))
    with sqlite3.connect(root / "source.db") as source:
        source.execute(
            "INSERT INTO raw_sessions(raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms) "
            "VALUES (?,?,?,?,?,?)",
            (raw_id, "codex", "/old/source.jsonl", hashlib.sha256(b"source").digest(), 6, 1),
        )

    with open_operation_read(root) as pinned:
        assert (
            pinned.archive._conn.execute("SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[
                0
            ]
            == raw_id
        )
        assert tuple(
            pinned.archive._conn.execute("SELECT raw_id, source_path FROM source_tier.raw_sessions").fetchone()
        ) == (
            raw_id,
            "/old/source.jsonl",
        )
        with sqlite3.connect(root / "source.db") as writer:
            writer.execute("UPDATE raw_sessions SET source_path = ? WHERE raw_id = ?", ("/new/source.jsonl", raw_id))
        old_result = execute_read_operation(
            "session.identity-reset.targets",
            {"source_path": "/old/source.jsonl"},
            archive=pinned.archive,
            serving_identity="test",
        )
        new_result = execute_read_operation(
            "session.identity-reset.targets",
            {"source_path": "/new/source.jsonl"},
            archive=pinned.archive,
            serving_identity="test",
        )
    assert old_result["session_ids"] == [session_id], old_result
    assert new_result["session_ids"] == []

    with open_operation_read(root) as fresh:
        current = execute_read_operation(
            "session.identity-reset.targets",
            {"source_path": "/new/source.jsonl"},
            archive=fresh.archive,
            serving_identity="test",
        )
    assert current["session_ids"] == [session_id]


def test_assertion_export_uses_the_pinned_user_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "assertion-pin").provider("codex").title("Target").save()

    with open_operation_read(root) as pinned:
        with sqlite3.connect(root / "user.db") as writer:
            writer.execute(
                "INSERT INTO assertions(assertion_id,target_ref,key,kind,value_json,created_at_ms,updated_at_ms) "
                "VALUES (?,?,?,?,?,?,?)",
                ("pinned-assertion", "session:assertion-pin", "neutral", "tag", "{}", 1000, 1000),
            )
        original = execute_read_operation("user.assertions.export", {}, archive=pinned.archive, serving_identity="test")
    assert original["items"] == []

    with open_operation_read(root) as fresh:
        current = execute_read_operation("user.assertions.export", {}, archive=fresh.archive, serving_identity="test")
    items = current["items"]
    assert isinstance(items, list) and len(items) == 1
    assert isinstance(items[0], dict)
    assert items[0]["assertion_id"] == "pinned-assertion"
