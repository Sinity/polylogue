"""Focused production-route regressions for the lineage write state machine."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Origin, Provider
from polylogue.pipeline.services.ingest_batch._core import _append_delta_payload
from polylogue.pipeline.services.ingest_worker import SessionWritePayload
from polylogue.security.excision import apply_session_excision
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope, write_parsed_session_to_archive
from polylogue.storage.sqlite.queries.message_query_reads import get_messages


def _index(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _message(native_id: str | None, text: str, position: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=native_id or "",
        role=Role.USER,
        text=text,
        position=position,
        is_active_path=True,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _session(native_id: str, messages: list[ParsedMessage], *, parent: str | None = None) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        parent_session_provider_id=parent,
        branch_type=BranchType.FORK if parent else None,
        messages=messages,
    )


def _texts(path: Path, session_id: str) -> list[str | None]:
    async def read() -> list[str | None]:
        import aiosqlite

        async with aiosqlite.connect(path) as conn:
            conn.row_factory = aiosqlite.Row
            rows = await get_messages(conn, session_id)
            return [row.text for row in rows]

    return asyncio.run(read())


def test_state_parent_is_authoritative_before_prefix_normalization(tmp_path: Path) -> None:
    """Anti-vacuity: using parser parent before state evidence leaves the child dangling."""
    index = _index(tmp_path / "index.db")
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    index.execute(
        "INSERT INTO codex_thread_spawn_edges (parent_thread_id, child_thread_id, status, observed_at_ms) "
        "VALUES ('hook-parent', 'child', 'closed', 1)",
    )
    index.commit()
    write_parsed_session_to_archive(
        index, _session("hook-parent", [_message("h0", "hook prefix", 0)]), source_conn=source
    )
    write_parsed_session_to_archive(
        index, _session("parser-parent", [_message("p0", "parser prefix", 0)]), source_conn=source
    )
    child_id = write_parsed_session_to_archive(
        index,
        _session("child", [_message("c0", "hook prefix", 0), _message("c1", "child tail", 1)], parent="parser-parent"),
        source_conn=source,
    )

    stored = index.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (child_id,)).fetchone()[0]
    link = index.execute(
        "SELECT dst_native_id, inheritance FROM session_links WHERE src_session_id = ? AND method = ?",
        (child_id, "authoritative-hook-evidence"),
    ).fetchone()
    assert stored == 1
    assert tuple(link) == ("hook-parent", "prefix-sharing")
    assert _texts(tmp_path / "index.db", child_id) == ["hook prefix", "child tail"]
    source.close()
    index.close()


def test_positional_branch_point_alias_is_typed_dangling(tmp_path: Path) -> None:
    """Anti-vacuity: removing the content witness silently composes the wrong parent row."""
    index = _index(tmp_path / "index.db")
    parent = _session("parent", [_message(None, "A", 0), _message(None, "B", 1)])
    parent_id = write_parsed_session_to_archive(index, parent)
    child_id = write_parsed_session_to_archive(
        index,
        _session("child", [_message(None, "A", 0), _message(None, "B", 1), _message(None, "tail", 2)], parent="parent"),
    )
    replacement = _session(
        "parent",
        [_message(None, "X", 0), _message(None, "A", 1), _message(None, "B", 2)],
    ).model_copy(update={"updated_at": "2027-01-01T00:00:01Z"})
    write_parsed_session_to_archive(index, replacement)
    envelope = read_archive_session_envelope(index, child_id)
    assert [block.text for message in envelope.messages for block in message.blocks] == ["tail"]
    assert envelope.lineage_complete is False
    assert envelope.lineage_truncation_reason == "dangling_branch_point"
    assert parent_id
    index.close()


def test_append_delta_compares_composed_parent_prefix(tmp_path: Path) -> None:
    """Anti-vacuity: comparing only physical child rows re-appends replayed prefix content."""
    index = _index(tmp_path / "index.db")
    write_parsed_session_to_archive(index, _session("parent", [_message("p0", "shared", 0)]))
    child = _session("child", [_message("c0", "shared", 0), _message("c1", "old tail", 1)], parent="parent")
    child_id = write_parsed_session_to_archive(index, child)
    incoming = _session(
        "child",
        [_message("new0", "shared", 0), _message("new1", "old tail", 1), _message("new2", "new tail", 2)],
        parent="parent",
    )
    payload = SessionWritePayload(session_id=child_id, content_hash="", parsed_session=incoming)
    delta, skipped = _append_delta_payload(index, payload)
    assert delta is not None
    assert skipped == 2
    assert [message.text for message in delta.messages] == ["new tail"]
    index.close()


def test_provider_omitted_total_tokens_stays_unknown(tmp_path: Path) -> None:
    """Anti-vacuity: coercing the absent total back to zero makes this assertion fail."""
    index = _index(tmp_path / "index.db")
    session = _session("usage", [_message("m0", "usage", 0)])
    session = session.model_copy(
        update={
            "session_events": [
                ParsedSessionEvent(
                    event_type="message_usage",
                    source_message_provider_id="m0",
                    payload={"type": "message_usage", "last_token_usage": {"input_tokens": 9}},
                )
            ]
        }
    )
    session_id = write_parsed_session_to_archive(index, session)
    row = index.execute(
        "SELECT last_input_tokens, last_total_tokens, total_tokens FROM session_provider_usage_events WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    assert tuple(row) == (9, None, None)
    index.close()


def test_inherited_message_usage_is_not_counted_again_on_child_write(tmp_path: Path) -> None:
    """Anti-vacuity: retaining the inherited event makes the child total 30 instead of 20."""
    index = _index(tmp_path / "index.db")
    inherited_event = ParsedSessionEvent(
        event_type="message_usage",
        source_message_provider_id="p0",
        payload={"type": "message_usage", "last_token_usage": {"input_tokens": 10}},
    )
    write_parsed_session_to_archive(
        index,
        _session("usage-parent", [_message("p0", "shared", 0)]).model_copy(
            update={"session_events": [inherited_event]}
        ),
    )
    child_event = ParsedSessionEvent(
        event_type="message_usage",
        source_message_provider_id="c0",
        payload={"type": "message_usage", "last_token_usage": {"input_tokens": 20}},
    )
    child = _session(
        "usage-child",
        [_message("p0", "shared", 0), _message("c0", "tail", 1)],
        parent="usage-parent",
    ).model_copy(update={"session_events": [inherited_event, child_event]})
    child_id = write_parsed_session_to_archive(index, child)
    row = index.execute(
        "SELECT SUM(last_input_tokens) FROM session_provider_usage_events WHERE session_id = ?",
        (child_id,),
    ).fetchone()
    assert row[0] == 20
    index.close()


def test_cascade_excision_removes_rawless_inherited_child(tmp_path: Path) -> None:
    """Anti-vacuity: target resolution must count a session row without physical child messages."""
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    conn = sqlite3.connect(tmp_path / "index.db")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        "INSERT INTO sessions(native_id, origin, title, content_hash) VALUES (?, ?, ?, zeroblob(32))",
        ("parent", Origin.CODEX_SESSION.value, "parent"),
    )
    parent_id = conn.execute("SELECT session_id FROM sessions WHERE native_id = 'parent'").fetchone()[0]
    conn.execute(
        "INSERT INTO messages(session_id, native_id, position, role, content_hash) VALUES (?, ?, 0, 'user', zeroblob(32))",
        (parent_id, "p0"),
    )
    branch_point = conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (parent_id,)).fetchone()[0]
    conn.execute(
        "INSERT INTO sessions(native_id, origin, title, content_hash) VALUES (?, ?, ?, zeroblob(32))",
        ("child", Origin.CODEX_SESSION.value, "child"),
    )
    child_id = conn.execute("SELECT session_id FROM sessions WHERE native_id = 'child'").fetchone()[0]
    conn.execute(
        """
        INSERT INTO session_links(
            src_session_id, dst_origin, dst_native_id, link_type,
            resolved_dst_session_id, branch_point_message_id, inheritance, observed_at_ms
        ) VALUES (?, ?, 'parent', 'branch', ?, ?, 'prefix-sharing', 1)
        """,
        (child_id, Origin.CODEX_SESSION.value, parent_id, branch_point),
    )
    conn.commit()
    conn.close()
    receipt = apply_session_excision(tmp_path, parent_id, reason="test", cascade_lineage=True)
    assert child_id in receipt.cascaded_session_ids
    check = sqlite3.connect(tmp_path / "index.db")
    assert check.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    check.close()


def test_full_replace_recovers_orphaned_messages_without_touching_foreign_membership(tmp_path: Path) -> None:
    """Anti-vacuity: skipping orphan cleanup makes the target replay hit its message-coordinate UNIQUE key."""
    conn = _index(tmp_path / "index.db")
    target = _session("target", [_message("old-0", "old zero", 0), _message("old-1", "old one", 1)])
    foreign = _session("foreign", [_message("foreign-0", "foreign row", 0)])
    target_id = write_parsed_session_to_archive(conn, target, raw_id="target-raw")
    foreign_id = write_parsed_session_to_archive(conn, foreign, raw_id="foreign-raw")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (target_id,))
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")

    replacement = _session("target", [_message("new-0", "new zero", 0), _message("new-1", "new one", 1)])
    write_parsed_session_to_archive(conn, replacement, raw_id="target-raw")
    write_parsed_session_to_archive(conn, replacement, raw_id="target-raw")

    target_rows = conn.execute(
        "SELECT messages.native_id, messages.position, blocks.text FROM messages "
        "JOIN blocks USING (message_id) WHERE messages.session_id = ? ORDER BY messages.position",
        (target_id,),
    ).fetchall()
    foreign_rows = conn.execute(
        "SELECT messages.native_id, messages.position, blocks.text FROM messages "
        "JOIN blocks USING (message_id) WHERE messages.session_id = ? ORDER BY messages.position",
        (foreign_id,),
    ).fetchall()
    assert [tuple(row) for row in target_rows] == [("new-0", 0, "new zero"), ("new-1", 1, "new one")]
    assert [tuple(row) for row in foreign_rows] == [("foreign-0", 0, "foreign row")]
    assert conn.execute("SELECT COUNT(*) FROM blocks WHERE session_id = ?", (target_id,)).fetchone()[0] == 2
    conn.close()


@pytest.mark.parametrize("fault_target", ["_write_messages", "_write_attachments"])
def test_full_replace_faults_roll_back_to_complete_membership_then_retry_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_target: str
) -> None:
    """A fault before or after replacement rolls back the old cohort; a retry publishes one new cohort."""
    conn = _index(tmp_path / "index.db")
    original = _session("atomic", [_message("old-0", "old zero", 0), _message("old-1", "old one", 1)])
    session_id = write_parsed_session_to_archive(conn, original, raw_id="atomic-raw")
    replacement = _session("atomic", [_message("new-0", "new zero", 0), _message("new-1", "new one", 1)])
    original_writer = getattr(archive_tier_write, fault_target)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(f"injected {fault_target} crash")

    monkeypatch.setattr(archive_tier_write, fault_target, fail_write)
    with pytest.raises(RuntimeError, match=f"injected {fault_target} crash"):
        write_parsed_session_to_archive(conn, replacement, raw_id="atomic-raw")
    old_rows = conn.execute(
        "SELECT messages.native_id, messages.position, blocks.text FROM messages "
        "JOIN blocks USING (message_id) WHERE messages.session_id = ? ORDER BY messages.position",
        (session_id,),
    ).fetchall()
    assert [tuple(row) for row in old_rows] == [("old-0", 0, "old zero"), ("old-1", 1, "old one")]

    monkeypatch.setattr(archive_tier_write, fault_target, original_writer)
    write_parsed_session_to_archive(conn, replacement, raw_id="atomic-raw")
    write_parsed_session_to_archive(conn, replacement, raw_id="atomic-raw")
    new_rows = conn.execute(
        "SELECT messages.native_id, messages.position, blocks.text FROM messages "
        "JOIN blocks USING (message_id) WHERE messages.session_id = ? ORDER BY messages.position",
        (session_id,),
    ).fetchall()
    assert [tuple(row) for row in new_rows] == [("new-0", 0, "new zero"), ("new-1", 1, "new one")]
    assert conn.execute("SELECT COUNT(*) FROM blocks WHERE session_id = ?", (session_id,)).fetchone()[0] == 2
    conn.close()
