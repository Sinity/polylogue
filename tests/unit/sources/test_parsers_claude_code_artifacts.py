from __future__ import annotations

from polylogue.archive.message.types import MessageType
from polylogue.sources.parsers.claude import parse_code
from polylogue.sources.parsers.claude.common import normalize_timestamp


def test_normalize_timestamp_returns_canonical_iso_text() -> None:
    assert normalize_timestamp(1704067200) == "2024-01-01T00:00:00+00:00"
    assert normalize_timestamp(1704067200000) == "2024-01-01T00:00:00+00:00"


def test_parse_code_classifies_runtime_artifacts() -> None:
    items: list[object] = [
        {
            "type": "user",
            "uuid": "task-1",
            "sessionId": "sess-1",
            "timestamp": 1704067200,
            "message": {
                "role": "user",
                "content": "<task-notification><status>completed</status></task-notification>",
            },
        },
        {
            "type": "user",
            "uuid": "command-1",
            "sessionId": "sess-1",
            "timestamp": 1704067201,
            "message": {
                "role": "user",
                "content": "<command-name>status</command-name>\n<command-message>status</command-message>",
            },
        },
        {
            "type": "user",
            "uuid": "skill-1",
            "sessionId": "sess-1",
            "timestamp": 1704067202,
            "message": {
                "role": "user",
                "content": "Base directory for this skill: /tmp/skill\n\n# Skill Body",
            },
        },
        {
            "type": "user",
            "uuid": "prompt-1",
            "sessionId": "sess-1",
            "timestamp": 1704067203,
            "message": {
                "role": "user",
                "content": "<system-reminder>model-only context</system-reminder>\n\nActual user prompt.",
            },
        },
    ]

    result = parse_code(items, "fallback")

    by_id = {message.provider_message_id: message for message in result.messages}
    assert len(result.messages) == len(by_id) == 4
    assert by_id["task-1"].message_type is MessageType.PROTOCOL
    assert by_id["command-1"].message_type is MessageType.PROTOCOL
    assert by_id["skill-1"].message_type is MessageType.CONTEXT
    assert by_id["prompt-1"].message_type is MessageType.MESSAGE


def test_parse_code_drops_progress_hook_records() -> None:
    """#1617: ``type=progress`` is a hook lifecycle event, not message content."""
    items: list[object] = [
        {
            "type": "user",
            "uuid": "u-1",
            "sessionId": "sess-1",
            "timestamp": 1704067200,
            "message": {"role": "user", "content": "real prompt"},
        },
        {
            "type": "progress",
            "uuid": "prog-1",
            "sessionId": "sess-1",
            "timestamp": 1704067201,
            "data": {"hookEvent": "PreToolUse", "hookName": "test-hook", "command": "echo"},
            "parentToolUseID": "tool-1",
            "toolUseID": "tool-1",
        },
        {
            "type": "progress",
            "uuid": "prog-2",
            "sessionId": "sess-1",
            "timestamp": 1704067202,
            "data": {"hookEvent": "PostToolUse"},
        },
        {
            "type": "assistant",
            "uuid": "a-1",
            "sessionId": "sess-1",
            "timestamp": 1704067203,
            "message": {"role": "assistant", "content": "reply"},
        },
    ]

    result = parse_code(items, "fallback-progress")

    # Two real messages survive; both progress hook events are dropped.
    provider_ids = {m.provider_message_id for m in result.messages}
    assert provider_ids == {"u-1", "a-1"}
    assert all("prog" not in pid for pid in provider_ids)
