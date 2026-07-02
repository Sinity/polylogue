import sqlite3
from typing import Any, cast

from scripts import agent_forensics


def _build_failure_followup_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE actions (
            session_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            tool_name TEXT,
            tool_command TEXT,
            is_error INTEGER,
            exit_code INTEGER
        );
        CREATE TABLE messages (
            session_id TEXT NOT NULL,
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            position INTEGER NOT NULL,
            model_name TEXT
        );
        CREATE TABLE blocks (
            message_id TEXT NOT NULL,
            text TEXT
        );
        """
    )
    messages = [
        ("s1", "tool-ack", "tool", 1, "claude-opus"),
        ("s1", "next-ack", "assistant", 2, "claude-sonnet"),
        ("s1", "tool-silent", "tool", 3, "claude-opus"),
        ("s1", "next-silent", "assistant", 4, "claude-haiku"),
        ("s1", "tool-short", "tool", 5, "claude-opus"),
        ("s1", "next-short", "assistant", 6, "claude-haiku"),
        ("s2", "tool-missing-next", "tool", 1, "deepseek-v4"),
    ]
    conn.executemany(
        "INSERT INTO messages(session_id, message_id, role, position, model_name) VALUES (?, ?, ?, ?, ?)",
        messages,
    )
    actions = [
        ("s1", "tool-ack", "Bash", "pytest", 1, None),
        ("s1", "tool-silent", "Bash", "ls missing", 0, 2),
        ("s1", "tool-short", "Edit", "patch", 1, None),
        ("s2", "tool-missing-next", "Read", "read file", 0, 1),
    ]
    conn.executemany(
        "INSERT INTO actions(session_id, message_id, tool_name, tool_command, is_error, exit_code) VALUES (?, ?, ?, ?, ?, ?)",
        actions,
    )
    blocks = [
        ("next-ack", "The command failed with an error, so I will fix it."),
        ("next-silent", "I will continue by inspecting the neighboring module now."),
        ("next-short", "ok"),
    ]
    conn.executemany("INSERT INTO blocks(message_id, text) VALUES (?, ?)", blocks)
    return conn


def test_classify_failed_followup_uses_only_explicit_acknowledgment_markers() -> None:
    assert agent_forensics._classify_failed_followup(None) == "ambiguous"
    assert agent_forensics._classify_failed_followup("ok") == "ambiguous"
    assert agent_forensics._classify_failed_followup("The command failed with exit code 2.") == "acknowledged"
    assert (
        agent_forensics._classify_failed_followup("I will continue by inspecting the next module now.")
        == "silent_proceed"
    )


def test_structured_failure_followups_aggregates_ref_backed_samples() -> None:
    conn = _build_failure_followup_db()
    result = agent_forensics._structured_failure_followups(conn, sample_limit=3)
    by_tool = cast(list[dict[str, Any]], result["by_tool"])
    by_model = cast(list[dict[str, Any]], result["by_model"])
    samples = cast(list[dict[str, Any]], result["samples"])
    stratified = cast(dict[str, list[dict[str, Any]]], result["samples_by_classification"])

    assert result["totals"] == {
        "failed_outcomes": 4,
        "acknowledged": 1,
        "silent_proceed": 1,
        "ambiguous": 2,
        "classified_outcomes": 2,
    }
    assert by_tool[0] == {
        "name": "Bash",
        "failed_outcomes": 2,
        "acknowledged": 1,
        "silent_proceed": 1,
        "ambiguous": 0,
        "classified_outcomes": 2,
        "silent_rate": 0.5,
        "silent_rate_among_classified": 0.5,
    }
    assert by_model[0]["name"] == "claude-haiku"
    assert by_model[0]["failed_outcomes"] == 2
    assert by_model[0]["classified_outcomes"] == 1
    assert by_model[0]["silent_rate"] == 0.5
    assert by_model[0]["silent_rate_among_classified"] == 1.0
    assert len(samples) == 3
    assert samples[0]["tool_message_ref"] == "message:tool-ack"
    assert samples[0]["next_message_ref"] == "message:next-ack"
    assert [item["classification"] for item in stratified["acknowledged"]] == ["acknowledged"]
    assert [item["classification"] for item in stratified["silent_proceed"]] == ["silent_proceed"]
    assert [item["classification"] for item in stratified["ambiguous"]] == ["ambiguous", "ambiguous"]


def test_structured_failure_followups_can_bound_failed_outcomes() -> None:
    conn = _build_failure_followup_db()

    result = agent_forensics._structured_failure_followups(conn, failed_outcome_limit=2)

    assert result["failed_outcome_limit"] == 2
    assert result["totals"] == {
        "failed_outcomes": 2,
        "acknowledged": 1,
        "silent_proceed": 1,
        "ambiguous": 0,
        "classified_outcomes": 2,
    }
