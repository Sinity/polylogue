"""Tests for ``~/.claude/todos/*.json`` plan-snapshot parsing (polylogue-t0p)."""

from __future__ import annotations

import json

from polylogue.core.enums import Provider
from polylogue.sources.origin_specs import artifact_rule_for_path
from polylogue.sources.parsers.claude.todos import (
    parse_claude_todo_artifact,
    session_and_agent_id_from_filename,
)

_SESSION_ID = "138e259e-435f-4259-8c68-dbd5aa9f9837"
_AGENT_ID = "0092150d-6b81-43f9-85e2-ceeb2d1c8773"


def test_session_id_recovered_from_plain_filename() -> None:
    session_id, agent_id = session_and_agent_id_from_filename(f"/home/x/.claude/todos/{_SESSION_ID}.json")
    assert session_id == _SESSION_ID
    assert agent_id is None


def test_session_and_agent_id_recovered_from_subagent_filename() -> None:
    session_id, agent_id = session_and_agent_id_from_filename(
        f"/home/x/.claude/todos/{_SESSION_ID}-agent-{_AGENT_ID}.json"
    )
    assert session_id == _SESSION_ID
    assert agent_id == _AGENT_ID


def test_non_uuid_filename_returns_no_identity() -> None:
    assert session_and_agent_id_from_filename("/home/x/.claude/todos/not-a-uuid.json") == (None, None)


def test_parse_claude_todo_artifact_preserves_order_and_fields() -> None:
    source_path = f"/home/x/.claude/todos/{_SESSION_ID}.json"
    payload = json.dumps(
        [
            {"content": "Write bootstrap.sh", "status": "completed", "priority": "high", "id": "t1"},
            {"content": "Improve README", "status": "in_progress", "priority": "medium", "id": "t2"},
            {"content": "Run adversarial tests", "status": "pending", "priority": "high", "id": "t3"},
        ]
    ).encode("utf-8")

    snapshot = parse_claude_todo_artifact(source_path, payload)

    assert snapshot is not None
    assert snapshot.session_id == _SESSION_ID
    assert snapshot.agent_id is None
    assert snapshot.parse_error is None
    assert [item.content for item in snapshot.items] == [
        "Write bootstrap.sh",
        "Improve README",
        "Run adversarial tests",
    ]
    assert [item.position for item in snapshot.items] == [0, 1, 2]
    assert snapshot.item_count == 3
    assert snapshot.completed_count == 1
    assert snapshot.completion_rate == 1 / 3


def test_parse_claude_todo_artifact_empty_plan_has_no_completion_rate() -> None:
    source_path = f"/home/x/.claude/todos/{_SESSION_ID}.json"
    snapshot = parse_claude_todo_artifact(source_path, b"[]")

    assert snapshot is not None
    assert snapshot.items == ()
    assert snapshot.item_count == 0
    assert snapshot.completion_rate is None


def test_parse_claude_todo_artifact_skips_malformed_entries() -> None:
    source_path = f"/home/x/.claude/todos/{_SESSION_ID}.json"
    payload = json.dumps(
        [
            {"content": "valid", "status": "pending", "id": "t1"},
            {"content": "missing status"},
            "not-a-dict",
            {"status": "completed"},
        ]
    ).encode("utf-8")

    snapshot = parse_claude_todo_artifact(source_path, payload)

    assert snapshot is not None
    assert [item.content for item in snapshot.items] == ["valid"]


def test_parse_claude_todo_artifact_reports_error_for_non_array_json() -> None:
    source_path = f"/home/x/.claude/todos/{_SESSION_ID}.json"
    snapshot = parse_claude_todo_artifact(source_path, b'{"not": "an array"}')

    assert snapshot is not None
    assert snapshot.items == ()
    assert snapshot.parse_error is not None
    assert snapshot.session_id == _SESSION_ID


def test_parse_claude_todo_artifact_reports_error_for_invalid_json() -> None:
    source_path = f"/home/x/.claude/todos/{_SESSION_ID}.json"
    snapshot = parse_claude_todo_artifact(source_path, b"{not json")

    assert snapshot is not None
    assert snapshot.items == ()
    assert snapshot.parse_error is not None


def test_origin_spec_admits_todos_directory_artifact_as_fact_tier() -> None:
    """Production dependency: OriginSpec must classify a real todos path as a fact artifact.

    Anti-vacuity mutation: removing the ``todo_snapshot`` OriginArtifactRule
    (or narrowing its ``path_pattern`` to exclude ``todos/*.json``) makes this
    assertion fail -- ``artifact_rule_for_path`` would return ``None``.
    """
    rule = artifact_rule_for_path(Provider.CLAUDE_CODE, f"/home/x/.claude/todos/{_SESSION_ID}.json")

    assert rule is not None
    assert rule.kind == "todo_snapshot"
    assert rule.parse_policy == "fact"
    assert rule.parser_path == "polylogue/sources/parsers/claude/todos.py:parse_claude_todo_artifact"
