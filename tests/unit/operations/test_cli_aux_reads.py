"""Resident execution for small CLI reads outside the query grammar."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.operations.daemon_protocol import validate_operation_result
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.connection_profile import attach_readonly_database
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


def test_assertion_export_keeps_a_present_empty_user_tier_empty(tmp_path: Path) -> None:
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


@pytest.mark.parametrize("damage", ["missing", "unreadable", "wrong-root"])
@pytest.mark.parametrize("operation", ["user.assertions.export", "user.assertions.list"])
def test_assertion_export_refuses_unavailable_user_authority(tmp_path: Path, damage: str, operation: str) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "authority-export").provider("codex").save()
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        archive._conn.execute("DETACH DATABASE user_tier")
        if damage == "unreadable":
            attach_readonly_database(archive._conn, root / "user.db", alias="user_tier")
            archive._conn.set_authorizer(
                lambda action, _a, _b, _db, _trigger: (
                    sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_PRAGMA else sqlite3.SQLITE_OK
                )
            )
        elif damage == "wrong-root":
            other_root = tmp_path / "other"
            other_root.mkdir()
            SessionBuilder(other_root / "index.db", "other").provider("codex").save()
            attach_readonly_database(archive._conn, other_root / "user.db", alias="user_tier")
        with pytest.raises(ArchiveTierUnavailableError) as failure:
            execute_read_operation(operation, {}, archive=archive, serving_identity="test")
    assert failure.value.tier == "user.db"


@pytest.mark.parametrize("replacement", [False, True])
def test_assertion_list_uses_pinned_user_after_update_or_path_replacement(tmp_path: Path, replacement: bool) -> None:
    from polylogue.core.enums import AssertionKind
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "assertion-list-pin").provider("codex").save()
    with sqlite3.connect(root / "user.db") as user:
        upsert_assertion(
            user,
            assertion_id="neutral-handoff",
            target_ref="session:codex-session:assertion-list-pin",
            kind=AssertionKind.HANDOFF,
            body_text="old snapshot",
            now_ms=1000,
        )
    with open_operation_read(root) as pinned:
        with sqlite3.connect(root / "user.db") as writer:
            writer.execute("UPDATE assertions SET body_text='new snapshot' WHERE assertion_id='neutral-handoff'")
            writer.commit()
            if replacement:
                other = tmp_path / "replacement.db"
                with sqlite3.connect(other) as changed:
                    writer.backup(changed)
                    changed.execute("UPDATE assertions SET body_text='replacement snapshot'")
                other.replace(root / "user.db")
        result = execute_read_operation(
            "user.assertions.list", {"kinds": ["handoff"]}, archive=pinned.archive, serving_identity="test"
        )
        items = result["items"]
        assert isinstance(items, list) and len(items) == 1
        assert isinstance(items[0], dict)
        assert items[0]["body_text"] == "old snapshot"
