"""Resident execution for small CLI reads outside the query grammar."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.operations.assertion_export import AssertionExportImages
from polylogue.operations.daemon_protocol import validate_operation_result
from polylogue.operations.daemon_reads import DaemonReadDependencies, execute_read_operation
from polylogue.operations.mutation_transaction import MutationPrincipal
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.connection_profile import attach_readonly_database
from tests.infra.storage_records import SessionBuilder


@pytest.fixture
def export_deps() -> Iterator[DaemonReadDependencies]:
    images = AssertionExportImages()
    try:
        yield DaemonReadDependencies(
            assertion_exports=images,
            assertion_export_principal=MutationPrincipal("neutral-export", frozenset(), "cli"),
        )
    finally:
        images.close()


def test_assertion_export_keeps_a_present_empty_user_tier_empty(
    tmp_path: Path, export_deps: DaemonReadDependencies
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "assertion-export").provider("codex").title("Export").save()
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        result = execute_read_operation(
            "user.assertions.export",
            {},
            archive=archive,
            serving_identity="test",
            dependencies=export_deps,
        )
    validate_operation_result("user.assertions.export", result)
    assert result["items"] == []
    assert result["total"] == 0
    outcome = result["outcome"]
    assert isinstance(outcome, dict)
    assert outcome["state"] == "empty"


def test_identity_reset_source_path_uses_the_pinned_source_snapshot(
    tmp_path: Path, export_deps: DaemonReadDependencies
) -> None:
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
        from polylogue.operations.cli_aux_reads import _iter_sessions_from_source_path

        old_result = tuple(_iter_sessions_from_source_path(pinned.archive, Path("/old/source.jsonl")))
        new_result = tuple(_iter_sessions_from_source_path(pinned.archive, Path("/new/source.jsonl")))
    assert old_result == (session_id,)
    assert new_result == ()
    with open_operation_read(root) as fresh:
        current = tuple(_iter_sessions_from_source_path(fresh.archive, Path("/new/source.jsonl")))
    assert current == (session_id,)


def test_assertion_export_uses_the_pinned_user_snapshot(tmp_path: Path, export_deps: DaemonReadDependencies) -> None:
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
        original = execute_read_operation(
            "user.assertions.export", {}, archive=pinned.archive, serving_identity="test", dependencies=export_deps
        )
    assert original["items"] == []

    with open_operation_read(root) as fresh:
        current = execute_read_operation(
            "user.assertions.export", {}, archive=fresh.archive, serving_identity="test", dependencies=export_deps
        )
    items = current["items"]
    assert isinstance(items, list) and len(items) == 1
    assert isinstance(items[0], dict)
    assert items[0]["assertion_id"] == "pinned-assertion"


@pytest.mark.parametrize("damage", ["missing", "unreadable", "wrong-root"])
@pytest.mark.parametrize("operation", ["user.assertions.export", "user.assertions.list"])
def test_assertion_export_refuses_unavailable_user_authority(
    tmp_path: Path, damage: str, operation: str, export_deps: DaemonReadDependencies
) -> None:
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
            execute_read_operation(operation, {}, archive=archive, serving_identity="test", dependencies=export_deps)
    assert failure.value.tier == "user.db"


@pytest.mark.parametrize("replacement", [False, True])
def test_assertion_list_uses_pinned_user_after_update_or_path_replacement(
    tmp_path: Path, replacement: bool, export_deps: DaemonReadDependencies
) -> None:
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


def test_assertion_export_pages_complete_population_and_explicit_limit(
    tmp_path: Path, export_deps: DaemonReadDependencies
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "neutral-export-pages").provider("codex").save()
    with sqlite3.connect(root / "user.db") as user:
        user.executemany(
            "INSERT INTO assertions(assertion_id,target_ref,key,kind,value_json,created_at_ms,updated_at_ms) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (f"neutral-{index:04d}", "session:neutral", "neutral", "tag", "{}", index + 1, index + 1)
                for index in range(513)
            ],
        )
    seen: list[str] = []
    offset = 0
    epoch = None
    with open_operation_read(root) as pinned:
        while True:
            page = execute_read_operation(
                "user.assertions.export",
                {"page_size": 17, "offset": offset, "selection_ref": epoch},
                archive=pinned.archive,
                serving_identity="test",
                dependencies=export_deps,
            )
            validate_operation_result("user.assertions.export", page)
            assert page["total"] == 513
            items = page["items"]
            assert isinstance(items, list) and len(items) <= 17
            seen.extend(row["assertion_id"] for row in items)
            epoch = page["selection_ref"]
            if page["next_offset"] is None:
                break
            next_offset = page["next_offset"]
            assert isinstance(next_offset, int)
            offset = next_offset
        first = execute_read_operation(
            "user.assertions.export",
            {"page_size": 1},
            archive=pinned.archive,
            serving_identity="test",
            dependencies=export_deps,
        )
        past_end = execute_read_operation(
            "user.assertions.export",
            {"offset": 10**100, "page_size": 10**100, "selection_ref": first["selection_ref"]},
            archive=pinned.archive,
            serving_identity="test",
            dependencies=export_deps,
        )
        assert past_end["items"] == [] and past_end["next_offset"] is None
        for limit in (0, 9):
            page = execute_read_operation(
                "user.assertions.export",
                {"limit": limit},
                archive=pinned.archive,
                serving_identity="test",
                dependencies=export_deps,
            )
            assert page["total"] == limit
            limited_items = page["items"]
            assert isinstance(limited_items, list)
            assert len(limited_items) == limit
            assert page["next_offset"] is None
    assert seen == [f"neutral-{index:04d}" for index in range(513)]


def test_assertion_export_continuation_refuses_changed_view(
    tmp_path: Path, export_deps: DaemonReadDependencies
) -> None:
    from polylogue.archive.query.transaction import QueryContinuationStaleError

    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "neutral-export-view").provider("codex").save()
    with sqlite3.connect(root / "user.db") as user:
        user.executemany(
            "INSERT INTO assertions(assertion_id,target_ref,key,kind,value_json,created_at_ms,updated_at_ms) VALUES (?,?,?,?,?,?,?)",
            [(f"original-{i}", "session:neutral", "neutral", "tag", "{}", i + 1, i + 1) for i in range(2)],
        )
    with open_operation_read(root) as pinned:
        page = execute_read_operation(
            "user.assertions.export",
            {"page_size": 1},
            archive=pinned.archive,
            serving_identity="test",
            dependencies=export_deps,
        )
    with sqlite3.connect(root / "user.db") as user:
        user.execute(
            "INSERT INTO assertions(assertion_id,target_ref,key,kind,value_json,created_at_ms,updated_at_ms) "
            "VALUES ('neutral-new', 'session:neutral', 'neutral', 'tag', '{}', 1, 1)"
        )
    with open_operation_read(root) as fresh:
        with pytest.raises(QueryContinuationStaleError):
            execute_read_operation(
                "user.assertions.export",
                {"selection_ref": page["selection_ref"], "offset": 1},
                archive=fresh.archive,
                serving_identity="test",
                dependencies=export_deps,
            )


def test_assertion_export_cancel_checkpoint_precedes_rows(tmp_path: Path, export_deps: DaemonReadDependencies) -> None:
    from polylogue.operations.daemon_reads import DaemonReadDependencies

    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "neutral-export-cancel").provider("codex").save()

    def cancel() -> None:
        raise InterruptedError("neutral cancellation")

    with open_operation_read(root) as pinned:
        with pytest.raises(InterruptedError):
            execute_read_operation(
                "user.assertions.export",
                {},
                archive=pinned.archive,
                serving_identity="test",
                dependencies=DaemonReadDependencies(
                    raise_if_aborted=cancel,
                    assertion_exports=export_deps.assertion_exports,
                    assertion_export_principal=export_deps.assertion_export_principal,
                ),
            )


@pytest.mark.parametrize("payload", [{"page_size": 0}, {"offset": -1}, {"offset": 1}, {"limit": -1}])
def test_assertion_export_refuses_malformed_page_operands(payload: dict[str, object]) -> None:
    from pydantic import ValidationError

    from polylogue.operations.daemon_protocol import AssertionExportRequest

    with pytest.raises(ValidationError):
        AssertionExportRequest.model_validate(payload)
