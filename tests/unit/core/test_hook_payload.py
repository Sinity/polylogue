"""bd polylogue-cp806: hook readers must read both payload generations.

Claude Code emits a camelCase generation of every hook payload alongside the
snake_case one. A reader keyed on one spelling reads the other as a payload
whose fields were never sent, so the guard here is *parity*: the canonical
reader keys that resolve must not depend on which generation the harness
emitted.

Anti-vacuity: dropping ``sessionId``/``toolUseId``/``toolResult`` from
``payload_key_spellings`` makes the camelCase side of every parity case resolve
a strictly smaller key set, and
``test_no_reader_key_matches_an_undescribed_generation`` red-lines the "no
reader key matched at all" case these tests exist to catch.
"""

from __future__ import annotations

import pytest

from polylogue.core.hook_payload import (
    HOOK_READER_KEYS,
    hook_payload_field,
    hook_record_field,
    matched_reader_keys,
    payload_key_spellings,
)

#: One snake_case and one camelCase payload per record kind, following the
#: shapes Claude Code actually emits. Values are synthetic.
_GENERATION_PAIRS: dict[str, tuple[dict[str, object], dict[str, object]]] = {
    "SessionStart": (
        {
            "session_id": "sess-1",
            "transcript_path": "/tmp/sess-1.jsonl",
            "cwd": "/work",
            "hook_event_name": "SessionStart",
            "source": "startup",
            "model": "test-model",
            "permission_mode": "auto",
        },
        {
            "sessionId": "sess-1",
            "transcriptPath": "/tmp/sess-1.jsonl",
            "cwd": "/work",
            "hookEventName": "SessionStart",
            "source": "startup",
            "model": "test-model",
            "permissionMode": "auto",
            "workspaceRoot": "/work",
            "timestamp": "2026-05-07T12:00:00Z",
        },
    ),
    "UserPromptSubmit": (
        {
            "session_id": "sess-1",
            "prompt_id": "p-1",
            "permission_mode": "auto",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "Inspect [Pasted text #1]",
        },
        {
            "sessionId": "sess-1",
            "promptId": "p-1",
            "permissionMode": "auto",
            "hookEventName": "UserPromptSubmit",
            "prompt": "Inspect [Pasted text #1]",
            "timestamp": "2026-05-07T12:00:00Z",
        },
    ),
    "PreToolUse": (
        {
            "session_id": "sess-1",
            "permission_mode": "auto",
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_use_id": "toolu_1",
            "tool_input": {"command": "ls"},
            "agent_id": "agent-instance-1",
            "agent_type": "log-consolidator",
        },
        {
            "sessionId": "sess-1",
            "permissionMode": "auto",
            "hookEventName": "PreToolUse",
            "toolName": "Bash",
            "toolUseId": "toolu_1",
            "toolInput": {"command": "ls"},
            "agentId": "agent-instance-1",
            "agentType": "log-consolidator",
            "timestamp": "2026-05-07T12:00:00Z",
        },
    ),
    "PostToolUse": (
        {
            "session_id": "sess-1",
            "permission_mode": "auto",
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_use_id": "toolu_1",
            "tool_input": {"command": "ls"},
            "tool_response": "ok",
            "agent_id": "agent-instance-1",
            "agent_type": "log-consolidator",
        },
        {
            "sessionId": "sess-1",
            "permissionMode": "auto",
            "hookEventName": "PostToolUse",
            "toolName": "Bash",
            "toolUseId": "toolu_1",
            "toolInput": {"command": "ls"},
            "toolResult": "ok",
            "agentId": "agent-instance-1",
            "agentType": "log-consolidator",
            "timestamp": "2026-05-07T12:00:00Z",
        },
    ),
    "Stop": (
        {
            "session_id": "sess-1",
            "permission_mode": "auto",
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": "done",
        },
        {
            "sessionId": "sess-1",
            "permissionMode": "auto",
            "hookEventName": "Stop",
            "stopHookActive": False,
            "lastAssistantMessage": "done",
            "timestamp": "2026-05-07T12:00:00Z",
        },
    ),
}


def _record(event_type: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "event_id": "e1",
        "event_type": event_type,
        "session_id": "sess-1",
        "timestamp": "2026-05-07T12:00:00Z",
        "provider": "claude-code",
        "payload": payload,
    }


@pytest.mark.parametrize("event_type", sorted(_GENERATION_PAIRS))
def test_both_generations_expose_the_same_reader_keys(event_type: str) -> None:
    snake, camel = _GENERATION_PAIRS[event_type]
    assert matched_reader_keys(_record(event_type, snake)) == matched_reader_keys(_record(event_type, camel))


@pytest.mark.parametrize("event_type", sorted(_GENERATION_PAIRS))
def test_every_record_kind_matches_at_least_one_reader_key(event_type: str) -> None:
    """A record no reader key matches is a signal, not a no-op."""
    for payload in _GENERATION_PAIRS[event_type]:
        assert matched_reader_keys(_record(event_type, payload))


@pytest.mark.parametrize("event_type", sorted(_GENERATION_PAIRS))
def test_camelcase_payload_resolves_the_keys_readers_name(event_type: str) -> None:
    camel = _GENERATION_PAIRS[event_type][1]
    record = _record(event_type, camel)
    assert hook_record_field(record, "session_id") == "sess-1"
    if "toolUseId" in camel:
        assert hook_record_field(record, "tool_use_id") == "toolu_1"
        assert hook_record_field(record, "agent_id") == "agent-instance-1"
        assert hook_record_field(record, "agent_type") == "log-consolidator"
    if "toolResult" in camel:
        assert hook_record_field(record, "tool_response") == "ok"


def test_no_reader_key_matches_an_undescribed_generation() -> None:
    """A spelling this vocabulary does not describe resolves nothing at all."""
    record = _record("PreToolUse", {"tool.use.id": "toolu_1", "agent.id": "agent-instance-1"})
    assert matched_reader_keys(record) == frozenset()


def test_payload_only_records_resolve_without_an_envelope() -> None:
    """``hook_payload_field`` reads a bare payload, ``hook_record_field`` a wrapped one."""
    payload = _GENERATION_PAIRS["PreToolUse"][1]
    assert hook_payload_field(payload, "tool_use_id") == "toolu_1"
    assert hook_payload_field(payload, "session_id") == "sess-1"


def test_envelope_value_wins_over_the_payload() -> None:
    record = _record("PreToolUse", {"sessionId": "payload-session"})
    assert hook_record_field(record, "session_id") == "sess-1"


def test_declared_reader_keys_are_canonical_snake_case() -> None:
    for key in HOOK_READER_KEYS:
        assert key == key.lower()
        assert payload_key_spellings(key)[0] == key


def test_irregular_spelling_is_declared_not_derived() -> None:
    """``tool_response`` and ``toolResult`` are the same field under two names."""
    assert "toolResult" in payload_key_spellings("tool_response")
