from __future__ import annotations

from polylogue.archive.message.types import MessageType
from polylogue.sources.parsers.claude import parse_code


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
    assert by_id["task-1"].message_type is MessageType.PROTOCOL
    assert by_id["command-1"].message_type is MessageType.PROTOCOL
    assert by_id["skill-1"].message_type is MessageType.CONTEXT
    assert by_id["prompt-1"].message_type is MessageType.MESSAGE
