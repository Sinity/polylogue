"""Shared blackboard body codec + note model (#1697)."""

from __future__ import annotations

from polylogue.archive.blackboard import (
    BLACKBOARD_KINDS,
    UNRESOLVED_KINDS,
    build_blackboard_body,
    decode_blackboard_note,
    parse_blackboard_body,
)


def test_build_then_parse_round_trips_structured_fields() -> None:
    body = build_blackboard_body(
        kind="blocker",
        title="WAL checkpoint stalls",
        content="Reader holds the lock for 40s during catch-up.",
        scope_repo="polylogue",
        scope_issue=1614,
        scope_path="storage/sqlite",
        related_sessions=("claude-code:abc", "codex:def"),
    )
    parsed = parse_blackboard_body(body)
    assert parsed.kind == "blocker"
    assert parsed.title == "WAL checkpoint stalls"
    assert parsed.content == "Reader holds the lock for 40s during catch-up."
    assert parsed.scope_repo == "polylogue"
    # scope_issue/scope_path/related_sessions are encoded but not surfaced as
    # structured fields on parse — they must not leak into content.
    assert "scope_issue" not in parsed.content
    assert "related_sessions" not in parsed.content


def test_build_omits_absent_scope_lines() -> None:
    body = build_blackboard_body(kind="finding", title="t", content="c")
    parsed = parse_blackboard_body(body)
    assert parsed.scope_repo is None
    assert parsed.content == "c"


def test_parse_unrecognized_body_falls_back_to_observation() -> None:
    parsed = parse_blackboard_body("a hand-written note without a kind prefix")
    assert parsed.kind == "observation"
    assert parsed.title == "a hand-written note without a kind prefix"
    assert parsed.scope_repo is None


def test_decode_blackboard_note_carries_targets_and_timestamps() -> None:
    body = build_blackboard_body(kind="handoff", title="pick up #1697", content="MCP tools remain")
    note = decode_blackboard_note(
        note_id="note-1",
        body=body,
        target_type="session",
        target_id="claude-code:xyz",
        created_at_ms=111,
        updated_at_ms=222,
    )
    assert note.note_id == "note-1"
    assert note.kind == "handoff"
    assert note.title == "pick up #1697"
    assert note.target_type == "session"
    assert note.target_id == "claude-code:xyz"
    assert note.created_at_ms == 111
    assert note.updated_at_ms == 222


def test_unresolved_kinds_subset_of_all_kinds() -> None:
    assert UNRESOLVED_KINDS.issubset(BLACKBOARD_KINDS)
    assert "blocker" in UNRESOLVED_KINDS
    assert "question" in UNRESOLVED_KINDS
    assert "finding" not in UNRESOLVED_KINDS
