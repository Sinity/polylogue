"""polylogue-4x38n: a Claude Code ``fork-context-ref`` record is lineage.

The record names the parent session and the exact parent message the fork
diverged at. A subagent transcript never replays that prefix, so prefix
alignment can derive no branch point from the child's own bytes -- if this
record is dropped, the branch point is unrecoverable.

Record shape (synthetic values, field names and types as they appear on the
wire)::

    {"type": "fork-context-ref", "agentId": ..., "parentSessionId": ...,
     "parentLastUuid": ..., "contextLength": ...}

Anti-vacuity for the whole file: revert ``fork-context-ref`` to a record type
the parser does not know and every assertion below fails -- the record falls
through to ordinary message parsing, carries no ``message``/text, and is
counted into ``empty_dropped_by_record_type`` instead of producing an edge.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.parsers.claude import parse_code
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    ASSERTED_BRANCH_POINT_EVIDENCE_KEY,
    write_parsed_session_to_archive,
)
from tests.infra.identity import archive_message_id

_PARENT_NATIVE_ID = "11111111-2222-3333-4444-555555555555"
_BRANCH_POINT_UUID = "66666666-7777-8888-9999-aaaaaaaaaaaa"
_AGENT_ID = "abcdef0123456789a"
_COVERAGE_EVENT_TYPE = "claude_parse_coverage"


def _fork_context_ref(*, parent_last_uuid: str = _BRANCH_POINT_UUID) -> dict[str, object]:
    return {
        "type": "fork-context-ref",
        "agentId": _AGENT_ID,
        "parentSessionId": _PARENT_NATIVE_ID,
        "parentLastUuid": parent_last_uuid,
        "contextLength": 477,
    }


def _child_records(*, parent_last_uuid: str = _BRANCH_POINT_UUID) -> list[dict[str, object]]:
    """One subagent transcript: the lineage record then its own work.

    Mirrors the corpus shape -- the ``fork-context-ref`` record is the first
    line, and the transcript that follows starts at the dispatching tool_use
    rather than replaying any parent message.
    """
    return [
        _fork_context_ref(parent_last_uuid=parent_last_uuid),
        {
            "type": "assistant",
            "sessionId": _PARENT_NATIVE_ID,
            "uuid": "c0000000-0000-0000-0000-000000000001",
            "isSidechain": True,
            "timestamp": "2026-01-01T00:00:10Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "subagent reply"}]},
        },
    ]


def _parse_child(*, parent_last_uuid: str = _BRANCH_POINT_UUID) -> ParsedSession:
    return parse_code(_child_records(parent_last_uuid=parent_last_uuid), f"agent-{_AGENT_ID}")


def _parent_session(*, include_branch_point: bool = True) -> ParsedSession:
    messages = [
        ParsedMessage(
            provider_message_id="p0000000-0000-0000-0000-000000000001",
            role=Role.USER,
            text="dispatch the fork",
            position=0,
            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="dispatch the fork")],
        )
    ]
    if include_branch_point:
        messages.append(
            ParsedMessage(
                provider_message_id=_BRANCH_POINT_UUID,
                role=Role.ASSISTANT,
                text="last message before the fork",
                position=1,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="last message before the fork")],
            )
        )
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=_PARENT_NATIVE_ID,
        title="parent",
        messages=messages,
    )


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _link(conn: sqlite3.Connection, child_id: str) -> sqlite3.Row:
    rows = conn.execute(
        """
        SELECT resolved_dst_session_id, dst_native_id, branch_point_message_id,
               method, confidence, evidence_json
        FROM session_links WHERE src_session_id = ?
        """,
        (child_id,),
    ).fetchall()
    assert len(rows) == 1
    row: sqlite3.Row = rows[0]
    return row


def test_fork_context_ref_is_lineage_evidence_not_a_dropped_record() -> None:
    """The parser must carry both halves of the assertion off the record."""
    parsed = _parse_child()

    assert parsed.parent_session_provider_id == _PARENT_NATIVE_ID
    assert parsed.branch_point_provider_message_id == _BRANCH_POINT_UUID
    # The record is not chat content: only the transcript's own message is a row.
    assert len(parsed.messages) == 1

    coverage = next(e for e in parsed.session_events if e.event_type == _COVERAGE_EVENT_TYPE)
    assert coverage.payload["sidecar_persisted"] == {"fork-context-ref": 1}
    assert coverage.payload["empty_dropped_by_record_type"] == {}

    evidence = next(e for e in parsed.session_events if e.event_type == "claude_fork_context_ref")
    assert evidence.payload["parent_session_id"] == _PARENT_NATIVE_ID
    assert evidence.payload["parent_last_message_id"] == _BRANCH_POINT_UUID
    assert evidence.payload["context_length"] == 477


def test_branch_point_reaches_session_links_when_parent_is_already_stored(tmp_path: Path) -> None:
    """The wanted outcome: an edge naming the parent AND the divergence point."""
    conn = _connect(tmp_path / "index.db")
    parent_id = write_parsed_session_to_archive(conn, _parent_session())
    child_id = write_parsed_session_to_archive(conn, _parse_child())

    link = _link(conn, child_id)
    assert link["dst_native_id"] == _PARENT_NATIVE_ID
    assert link["resolved_dst_session_id"] == parent_id
    assert link["branch_point_message_id"] == archive_message_id(parent_id, _BRANCH_POINT_UUID, position=0)
    # Parser-asserted, at full confidence: the provider stated this edge.
    assert link["method"]
    assert link["confidence"] == 1.0
    assert json.loads(link["evidence_json"])[ASSERTED_BRANCH_POINT_EVIDENCE_KEY] == _BRANCH_POINT_UUID


def test_branch_point_binds_when_the_parent_arrives_after_the_child(tmp_path: Path) -> None:
    """Ingest order must not decide whether the branch point survives.

    Anti-vacuity: drop ``_refill_inbound_asserted_branch_points`` and this
    edge keeps a NULL branch point forever, because the child is never
    rewritten.
    """
    conn = _connect(tmp_path / "index.db")
    child_id = write_parsed_session_to_archive(conn, _parse_child())
    assert _link(conn, child_id)["branch_point_message_id"] is None

    parent_id = write_parsed_session_to_archive(conn, _parent_session())

    link = _link(conn, child_id)
    assert link["resolved_dst_session_id"] == parent_id
    assert link["branch_point_message_id"] == archive_message_id(parent_id, _BRANCH_POINT_UUID, position=0)


def test_unbacked_branch_point_is_retained_as_a_claim_not_a_dangling_id(tmp_path: Path) -> None:
    """A branch point with no message behind it must not be written.

    ``session_links.branch_point_message_id`` has no foreign key, so writing
    the composed id unconditionally would satisfy the test above while
    planting exactly the dangling reference ``archive_verification``'s
    lineage-sanity check counts. The claim stays in ``evidence_json``.
    """
    conn = _connect(tmp_path / "index.db")
    write_parsed_session_to_archive(conn, _parent_session(include_branch_point=False))
    child_id = write_parsed_session_to_archive(conn, _parse_child())

    link = _link(conn, child_id)
    assert link["branch_point_message_id"] is None
    assert json.loads(link["evidence_json"])[ASSERTED_BRANCH_POINT_EVIDENCE_KEY] == _BRANCH_POINT_UUID

    dangling = conn.execute(
        """
        SELECT COUNT(*) FROM session_links AS sl
        WHERE sl.branch_point_message_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM messages AS m WHERE m.message_id = sl.branch_point_message_id)
        """
    ).fetchone()[0]
    assert dangling == 0
