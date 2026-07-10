from __future__ import annotations

from polylogue.archive.message.types import MessageType
from polylogue.core.enums import MaterialOrigin, Role
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
            "uuid": "stdout-1",
            "sessionId": "sess-1",
            "timestamp": 1704067201,
            "message": {
                "role": "user",
                "content": "<local-command-stdout>pytest passed</local-command-stdout>",
            },
        },
        {
            "type": "user",
            "uuid": "stderr-1",
            "sessionId": "sess-1",
            "timestamp": 1704067201.1,
            "message": {
                "role": "user",
                "content": "<local-command-stderr>warning: generated runtime stderr</local-command-stderr>",
            },
        },
        {
            "type": "user",
            "uuid": "bash-stdout-1",
            "sessionId": "sess-1",
            "timestamp": 1704067201.2,
            "message": {
                "role": "user",
                "content": "<bash-stdout>cargo test passed</bash-stdout>",
            },
        },
        {
            "type": "user",
            "uuid": "bash-stderr-1",
            "sessionId": "sess-1",
            "timestamp": 1704067201.3,
            "message": {
                "role": "user",
                "content": "<bash-stderr>traceback from shell wrapper</bash-stderr>",
            },
        },
        {
            "type": "user",
            "uuid": "command-1",
            "sessionId": "sess-1",
            "timestamp": 1704067202,
            "message": {
                "role": "user",
                "content": "<command-name>status</command-name>\n<command-message>status</command-message>",
            },
        },
        {
            "type": "user",
            "uuid": "skill-1",
            "sessionId": "sess-1",
            "timestamp": 1704067203,
            "message": {
                "role": "user",
                "content": "Base directory for this skill: /tmp/skill\n\n# Skill Body",
            },
        },
        {
            "type": "user",
            "uuid": "commit-pack-1",
            "sessionId": "sess-1",
            "timestamp": 1704067204,
            "message": {
                "role": "user",
                "content": "# Commit N: Generate all artifacts\n\nlarge generated context",
            },
        },
        {
            "type": "user",
            "uuid": "retro-pack-1",
            "sessionId": "sess-1",
            "timestamp": 1704067205,
            "message": {
                "role": "user",
                "content": "Generate a retrospective for: long generated analysis bundle",
            },
        },
        {
            "type": "user",
            "uuid": "prompt-1",
            "sessionId": "sess-1",
            "timestamp": 1704067206,
            "message": {
                "role": "user",
                "content": "<system-reminder>model-only context</system-reminder>\n\nActual user prompt.",
            },
        },
    ]

    result = parse_code(items, "fallback")

    by_id = {message.provider_message_id: message for message in result.messages}
    assert len(result.messages) == len(by_id) == 10
    assert by_id["task-1"].message_type is MessageType.PROTOCOL
    assert by_id["task-1"].material_origin is MaterialOrigin.RUNTIME_PROTOCOL
    assert by_id["stdout-1"].message_type is MessageType.PROTOCOL
    assert by_id["stdout-1"].material_origin is MaterialOrigin.RUNTIME_PROTOCOL
    assert by_id["stderr-1"].role is Role.USER
    assert by_id["stderr-1"].message_type is MessageType.PROTOCOL
    assert by_id["stderr-1"].material_origin is MaterialOrigin.RUNTIME_PROTOCOL
    assert by_id["bash-stdout-1"].role is Role.USER
    assert by_id["bash-stdout-1"].message_type is MessageType.PROTOCOL
    assert by_id["bash-stdout-1"].material_origin is MaterialOrigin.RUNTIME_PROTOCOL
    assert by_id["bash-stderr-1"].role is Role.USER
    assert by_id["bash-stderr-1"].message_type is MessageType.PROTOCOL
    assert by_id["bash-stderr-1"].material_origin is MaterialOrigin.RUNTIME_PROTOCOL
    assert by_id["command-1"].message_type is MessageType.PROTOCOL
    assert by_id["command-1"].material_origin is MaterialOrigin.OPERATOR_COMMAND
    assert by_id["skill-1"].message_type is MessageType.CONTEXT
    assert by_id["skill-1"].material_origin is MaterialOrigin.RUNTIME_CONTEXT
    assert by_id["commit-pack-1"].message_type is MessageType.MESSAGE
    assert by_id["commit-pack-1"].material_origin is MaterialOrigin.GENERATED_CONTEXT_PACK
    assert by_id["retro-pack-1"].message_type is MessageType.MESSAGE
    assert by_id["retro-pack-1"].material_origin is MaterialOrigin.GENERATED_ANALYSIS_PACK
    assert by_id["prompt-1"].message_type is MessageType.MESSAGE
    # Claude Code has provider-native provenance for real user turns: type=user,
    # !isMeta, no toolUseResult, no non-human origin.
    assert by_id["prompt-1"].material_origin is MaterialOrigin.HUMAN_AUTHORED
    assert result.title == "Actual user prompt."


def test_parse_code_preserves_tool_result_reclassification_material_origin() -> None:
    result = parse_code(
        [
            {
                "type": "user",
                "uuid": "tool-result-1",
                "sessionId": "sess-tool",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"}],
                },
            }
        ],
        "fallback-tool",
    )

    assert result.messages[0].role is Role.TOOL
    assert result.messages[0].material_origin is MaterialOrigin.TOOL_RESULT


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


def test_parse_code_drops_non_message_sidecars() -> None:
    items: list[object] = [
        {
            "type": "user",
            "uuid": "u-1",
            "sessionId": "sess-sidecar",
            "message": {"role": "user", "content": "real prompt"},
        },
        {"type": "attachment", "uuid": "att-1", "sessionId": "sess-sidecar", "attachment": {"name": "x.py"}},
        {"type": "mode", "mode": "default", "sessionId": "sess-sidecar"},
        {"type": "last-prompt", "lastPrompt": "real prompt", "sessionId": "sess-sidecar"},
        {"type": "ai-title", "aiTitle": "Title", "sessionId": "sess-sidecar"},
        {"type": "bridge-session", "sessionId": "sess-sidecar", "bridgeSessionId": "b1"},
        {"type": "permission-mode", "permissionMode": "acceptEdits", "sessionId": "sess-sidecar"},
        {"type": "pr-link", "sessionId": "sess-sidecar", "prNumber": 1},
    ]

    result = parse_code(items, "fallback-sidecar")

    assert [message.provider_message_id for message in result.messages] == ["u-1"]
    assert result.messages[0].material_origin is MaterialOrigin.HUMAN_AUTHORED


def test_parse_code_drops_empty_system_sidecars() -> None:
    result = parse_code(
        [
            {"type": "system", "uuid": "sys-empty", "sessionId": "sess-empty"},
            {
                "type": "user",
                "uuid": "u-1",
                "sessionId": "sess-empty",
                "message": {"role": "user", "content": "real prompt"},
            },
        ],
        "fallback-empty",
    )

    assert [message.provider_message_id for message in result.messages] == ["u-1"]


def test_parse_code_continuation_summary_is_runtime_context() -> None:
    result = parse_code(
        [
            {
                "type": "user",
                "uuid": "continued",
                "sessionId": "sess-continued",
                "message": {
                    "role": "user",
                    "content": "This session is being continued from a previous conversation that ran out of context.\n\nSummary...",
                },
            }
        ],
        "fallback-continued",
    )

    assert result.messages[0].message_type is MessageType.CONTEXT
    assert result.messages[0].material_origin is MaterialOrigin.RUNTIME_CONTEXT


def test_parse_code_compaction_summary_is_generated_context() -> None:
    result = parse_code(
        [
            {
                "type": "summary",
                "uuid": "summary-1",
                "sessionId": "sess-summary",
                "summary": "Compressed context for the next turn.",
                "timestamp": 1704067200,
            }
        ],
        "fallback-summary",
    )

    assert result.messages[0].message_type is MessageType.SUMMARY
    assert result.messages[0].material_origin is MaterialOrigin.GENERATED_CONTEXT_PACK


def test_parse_code_non_human_user_origin_is_runtime_protocol() -> None:
    result = parse_code(
        [
            {
                "type": "user",
                "uuid": "task-origin",
                "sessionId": "sess-origin",
                "origin": {"kind": "task-notification"},
                "message": {"role": "user", "content": "background task completed"},
            }
        ],
        "fallback-origin",
    )

    assert result.messages[0].message_type is MessageType.PROTOCOL
    assert result.messages[0].material_origin is MaterialOrigin.RUNTIME_PROTOCOL


def test_parse_code_interrupt_marker_is_protocol_not_human_prompt() -> None:
    result = parse_code(
        [
            {
                "type": "user",
                "uuid": "interrupt",
                "sessionId": "sess-interrupt",
                "message": {"role": "user", "content": "[Request interrupted by user]"},
            }
        ],
        "fallback-interrupt",
    )

    assert result.messages[0].message_type is MessageType.PROTOCOL
    assert result.messages[0].material_origin is MaterialOrigin.RUNTIME_PROTOCOL


def test_message_input_tokens_stays_raw_unlike_codex_disjoint_fix() -> None:
    """Control case for polylogue-f2qv.2: Anthropic's native ``input_tokens``
    already excludes the cached portion (unlike Codex's inclusive convention),
    so the Claude Code parser must NOT subtract ``cache_read_input_tokens``
    out of it -- doing so would under-count fresh input. ``input_tokens`` and
    ``cache_read_input_tokens`` are already disjoint additive lanes as
    reported."""
    items: list[object] = [
        {
            "type": "assistant",
            "uuid": "asst-1",
            "sessionId": "sess-cache",
            "timestamp": 1704067200,
            "message": {
                "role": "assistant",
                "content": "done",
                "model": "claude-opus-4-8",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 2400,
                    "cache_creation_input_tokens": 10,
                },
            },
        },
    ]

    result = parse_code(items, "fallback-cache")

    assert result.messages[0].input_tokens == 100
    assert result.messages[0].cache_read_tokens == 2400
    assert result.messages[0].cache_write_tokens == 10


def test_message_usage_payload_captures_server_tool_use() -> None:
    """web_search/web_fetch request counts are billed separately and must be
    preserved in the usage event payload (they ride into payload_json)."""
    from polylogue.sources.parsers.claude.code_parser import _message_usage_event_payload

    payload = _message_usage_event_payload(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "server_tool_use": {"web_search_requests": 3, "web_fetch_requests": 1},
        },
        model_name="claude-opus-4-8",
        model_effort=None,
    )
    assert payload["server_tool_use"] == {"web_search_requests": 3, "web_fetch_requests": 1}


def test_message_usage_payload_omits_all_zero_server_tool_use() -> None:
    """Most CLI sessions never call web tools; an all-zero sub-dict is dropped so
    payload_json is not bloated on every message."""
    from polylogue.sources.parsers.claude.code_parser import _message_usage_event_payload

    payload = _message_usage_event_payload(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
        },
        model_name=None,
        model_effort=None,
    )
    assert "server_tool_use" not in payload


def test_message_usage_payload_without_server_tool_use_key() -> None:
    from polylogue.sources.parsers.claude.code_parser import _message_usage_event_payload

    payload = _message_usage_event_payload(
        {"input_tokens": 10, "output_tokens": 5},
        model_name=None,
        model_effort=None,
    )
    assert "server_tool_use" not in payload
