"""Dedicated tests for the Codex JSONL parser.

Covers format detection, envelope/direct parsing, session metadata,
branch tracking, git context, and edge cases.
"""

from __future__ import annotations

from polylogue.archive.message.types import MessageType
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, MaterialOrigin, Role
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.codex import looks_like as _looks_like_impl
from polylogue.sources.parsers.codex import parse as _parse_impl
from polylogue.sources.parsers.codex import parse_stream


def looks_like(payload: object) -> bool:
    if not isinstance(payload, list):
        return False
    return _looks_like_impl(payload)


def parse(payload: object, fallback_id: str) -> ParsedSession:
    assert isinstance(payload, list)
    return _parse_impl(payload, fallback_id)


# =============================================================================
# Format Detection (looks_like)
# =============================================================================


class TestLooksLike:
    def test_envelope_format_detected(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2024-01-01"}},
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": []}},
        ]
        assert looks_like(payload)

    def test_direct_format_detected(self) -> None:
        payload = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ]
        assert looks_like(payload)

    def test_state_record_detected(self) -> None:
        payload = [{"record_type": "state"}]
        assert looks_like(payload)

    def test_intermediate_format_detected(self) -> None:
        """First line with id+timestamp (no type) is intermediate format."""
        payload = [{"id": "session-123", "timestamp": "2024-01-01T10:00:00Z"}]
        assert looks_like(payload)

    def test_empty_list_rejected(self) -> None:
        assert not looks_like([])

    def test_non_list_rejected(self) -> None:
        assert not looks_like({"type": "message"})

    def test_unrecognized_records_rejected(self) -> None:
        payload = [{"random": "data", "no_type": True}]
        assert not looks_like(payload)

    def test_non_codex_content_shape_rejected_before_validation(self) -> None:
        payload = [{"role": "user", "content": "synthetic-30495"}]
        assert not looks_like(payload)

    def test_non_dict_items_skipped(self) -> None:
        payload = ["string", 42, None, {"type": "message", "role": "user", "content": []}]
        assert looks_like(payload)  # The dict item matches


# =============================================================================
# Session Metadata
# =============================================================================


class TestSessionMetadata:
    def test_first_session_meta_sets_session_id(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "conv-abc", "timestamp": "2024-01-01"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.provider_session_id == "conv-abc"

    def test_second_session_meta_sets_parent(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "conv-abc", "timestamp": "2024-01-01"}},
            {"type": "session_meta", "payload": {"id": "parent-xyz", "timestamp": "2024-01-01"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.provider_session_id == "conv-abc"
        assert result.parent_session_provider_id == "parent-xyz"
        assert result.branch_type == BranchType.CONTINUATION

    def test_no_session_meta_uses_fallback(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]
        result = parse(payload, "my-fallback")
        assert result.provider_session_id == "my-fallback"
        assert result.parent_session_provider_id is None
        assert result.branch_type is None

    def test_forked_from_id_sets_unclassified_parent(self) -> None:
        # A user fork / resume records `forked_from_id` on the child's own meta.
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "child-1",
                    "forked_from_id": "parent-1",
                    "timestamp": "2024-01-01",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.provider_session_id == "child-1"
        assert result.parent_session_provider_id == "parent-1"
        # forked_from_id proves a parent but not fork-vs-resume, so the type is
        # left unclassified rather than over-claiming FORK.
        assert result.branch_type is None

    def test_subagent_thread_spawn_sets_subagent_parent(self) -> None:
        # A spawned subagent records `source.subagent.thread_spawn` in addition
        # to `forked_from_id`; that distinguishes it from a plain user fork.
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "child-2",
                    "forked_from_id": "parent-2",
                    "source": {
                        "subagent": {
                            "thread_spawn": {
                                "parent_thread_id": "parent-2",
                                "depth": 1,
                                "agent_role": "explorer",
                            }
                        }
                    },
                    "timestamp": "2024-01-01",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "exploring"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.provider_session_id == "child-2"
        assert result.parent_session_provider_id == "parent-2"
        assert result.branch_type == BranchType.SUBAGENT

    def test_forked_from_id_beats_legacy_second_meta_heuristic(self) -> None:
        # When the explicit marker is present, the embedded parent meta (second
        # session_meta = the copied parent's header) must not override it.
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "child-3",
                    "forked_from_id": "real-parent",
                    "timestamp": "2024-01-01",
                },
            },
            {"type": "session_meta", "payload": {"id": "real-parent", "timestamp": "2024-01-01"}},
        ]
        result = parse(payload, "fallback")
        assert result.parent_session_provider_id == "real-parent"
        # Explicit marker wins over the legacy second-meta heuristic; the
        # relationship type stays unclassified (not FORK) on forked_from_id.
        assert result.branch_type is None

    def test_duplicate_session_meta_id_not_counted_twice(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "same-id", "timestamp": "2024-01-01"}},
            {"type": "session_meta", "payload": {"id": "same-id", "timestamp": "2024-01-01"}},
        ]
        result = parse(payload, "fallback")
        assert result.provider_session_id == "same-id"
        assert result.parent_session_provider_id is None
        assert result.branch_type is None

    def test_intermediate_format_metadata(self) -> None:
        """Intermediate format: first line has id+timestamp."""
        payload = [
            {"id": "conv-xyz", "timestamp": "2024-01-01T12:00:00Z"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        ]
        result = parse(payload, "fallback")
        assert result.provider_session_id == "conv-xyz"


# =============================================================================
# Message Parsing
# =============================================================================


class TestMessageParsing:
    def test_envelope_message_parsed(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2024-01-01"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is 2+2?"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].text == "What is 2+2?"
        assert result.messages[0].material_origin is MaterialOrigin.HUMAN_AUTHORED

    def test_contextual_user_message_is_not_human_authored(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "# AGENTS.md instructions for /repo\n\n<INSTRUCTIONS>system context</INSTRUCTIONS>",
                        }
                    ],
                },
            },
        ]

        result = parse(payload, "fallback")

        assert result.messages[0].message_type is MessageType.CONTEXT
        assert result.messages[0].material_origin is MaterialOrigin.RUNTIME_CONTEXT

    def test_direct_message_parsed(self) -> None:
        payload = [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "The answer is 4."}]},
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1
        assert result.messages[0].role == "assistant"

    def test_function_call_output_captures_structured_exit_code(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": '{"output": "failed", "metadata": {"exit_code": 2}}',
                },
            },
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 1
        block = result.messages[0].blocks[0]
        assert block.type == "tool_result"
        assert block.tool_id == "call-1"
        assert block.is_error is True
        assert block.exit_code == 2

    def test_state_records_skipped(self) -> None:
        payload = [
            {"record_type": "state"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"record_type": "state"},
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1

    def test_multiple_content_blocks(self) -> None:
        payload = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Part 1"},
                    {"type": "input_text", "text": "Part 2"},
                ],
            },
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1
        # Text content should contain both parts
        text = result.messages[0].text or ""
        assert "Part 1" in text
        assert "Part 2" in text

    def test_empty_content_skipped(self) -> None:
        payload = [
            {"type": "message", "role": "user", "content": []},
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 0

    def test_message_without_text_skipped(self) -> None:
        """Message with only non-text blocks is skipped."""
        payload = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "tool_use", "name": "search"},
                ],
            },
        ]
        result = parse(payload, "fallback")
        # Message has structured content (tool_use) → now preserved even
        # without text, since tool_use/tool_result/thinking blocks are
        # independently meaningful.
        assert len(result.messages) == 1

    def test_message_role_normalization(self) -> None:
        """Roles are normalized via Role.normalize()."""
        payload = [
            {"type": "message", "role": "User", "content": [{"type": "input_text", "text": "hello"}]},
            {"type": "message", "role": "ASSISTANT", "content": [{"type": "output_text", "text": "hi"}]},
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"

    def test_envelope_payload_unwrapped(self) -> None:
        """response_item payloads are unwrapped and parsed as messages."""
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "query"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1
        assert result.messages[0].text == "query"

    def test_envelope_message_uses_wrapper_timestamp(self) -> None:
        payload = [
            {
                "type": "response_item",
                "timestamp": "2026-06-30T03:26:22.762Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "timed query"}],
                },
            },
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 1
        assert result.messages[0].timestamp == "2026-06-30T03:26:22.762Z"
        assert result.created_at is None
        assert result.updated_at == "2026-06-30T03:26:22.762Z"

    def test_envelope_message_inner_timestamp_beats_wrapper_timestamp(self) -> None:
        payload = [
            {
                "type": "response_item",
                "timestamp": "2026-06-30T03:26:22.762Z",
                "payload": {
                    "type": "message",
                    "timestamp": "2026-06-30T03:27:00.000Z",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "timed query"}],
                },
            },
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 1
        assert result.messages[0].timestamp == "2026-06-30T03:27:00.000Z"
        assert result.updated_at == "2026-06-30T03:27:00.000Z"

    def test_event_user_message_uses_wrapper_timestamp(self) -> None:
        payload = [
            {
                "type": "event_msg",
                "timestamp": "2026-06-30T03:26:22.762Z",
                "payload": {
                    "type": "user_message",
                    "client_id": "client-1",
                    "message": "please inspect the parser",
                },
            }
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 1
        assert result.messages[0].timestamp == "2026-06-30T03:26:22.762Z"
        assert result.updated_at == "2026-06-30T03:26:22.762Z"

    def test_tool_message_uses_wrapper_timestamp(self) -> None:
        payload = [
            {
                "type": "response_item",
                "timestamp": "2026-06-30T03:26:22.762Z",
                "payload": {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "shell",
                    "arguments": {"cmd": "date"},
                },
            }
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 1
        assert result.messages[0].timestamp == "2026-06-30T03:26:22.762Z"
        assert result.updated_at == "2026-06-30T03:26:22.762Z"

    def test_event_user_message_materializes_when_no_response_duplicate(self) -> None:
        payload = [
            {
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "client_id": "client-1",
                    "message": "please inspect the parser",
                },
            }
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 1
        assert result.messages[0].provider_message_id == "client-1"
        assert result.messages[0].role is Role.USER
        assert result.messages[0].material_origin is MaterialOrigin.HUMAN_AUTHORED

    def test_event_user_message_dedupes_matching_response_message(self) -> None:
        payload = [
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "client_id": "client-1", "message": "same prompt"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "same prompt"}],
                },
            },
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 1
        assert result.messages[0].text == "same prompt"
        assert result.messages[0].material_origin is MaterialOrigin.HUMAN_AUTHORED

    def test_custom_tool_call_and_output_materialize_as_action_pair(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "id": "ctc-1",
                    "call_id": "call-custom",
                    "name": "apply_patch",
                    "input": "*** Begin Patch\n*** End Patch",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "call-custom",
                    "output": "patch applied",
                },
            },
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 2
        use = result.messages[0].blocks[0]
        output = result.messages[1].blocks[0]
        assert use.type is BlockType.TOOL_USE
        assert use.tool_name == "apply_patch"
        assert use.tool_id == "call-custom"
        assert use.tool_input == {"arguments": "*** Begin Patch\n*** End Patch"}
        assert output.type is BlockType.TOOL_RESULT
        assert output.tool_id == "call-custom"
        assert output.text == "patch applied"

    def test_messages_do_not_keep_raw_provider_meta(self) -> None:
        payload = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
                "timestamp": "2024-01-01T00:00:01Z",
            },
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 2
        # provider_meta is gone from ParsedMessage — the typed contract enforces
        # this at the model level; no escape-hatch dict can exist.

    def test_parse_stream_matches_list_parse(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "conv-abc", "timestamp": "2024-01-01T00:00:00Z"}},
            {
                "type": "message",
                "id": "msg-1",
                "role": "user",
                "timestamp": "2024-01-01T00:00:01Z",
                "content": [{"type": "input_text", "text": "hello"}],
            },
            {
                "type": "message",
                "id": "msg-2",
                "role": "assistant",
                "timestamp": "2024-01-01T00:00:02Z",
                "content": [{"type": "output_text", "text": "hi"}],
            },
        ]

        from_list = parse(payload, "fallback")
        from_stream = parse_stream(iter(payload), "fallback")

        assert from_stream == from_list

    def test_system_developer_and_protocol_messages_are_typed_as_context_or_protocol(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "developer-context",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "<developer>runtime instruction</developer>"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "protocol-wrapper",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "<command-name>status</command-name>"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "real-user",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "actual request"}],
                },
            },
        ]

        result = parse(payload, "fallback")

        assert [message.role for message in result.messages] == ["system", "user", "user"]
        assert [message.message_type for message in result.messages] == [
            MessageType.CONTEXT,
            MessageType.PROTOCOL,
            MessageType.MESSAGE,
        ]

    def test_archive_tiers_contract_fields_from_turn_context_and_messages(self) -> None:
        payload = [
            {"type": "turn_context", "payload": {"cwd": "/repo/polylogue", "model": "gpt-5-codex", "effort": "high"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "content": [{"type": "input_text", "text": "run checks"}],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 4,
                    },
                    "duration_ms": 1250,
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "assistant-1",
                    "role": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "content": [{"type": "output_text", "text": "passed"}],
                    "model": "gpt-5-codex-mini",
                    "model_effort": "medium",
                    "tokens": {"input_tokens": 1, "output_tokens": 20},
                    "durationMs": "750",
                },
            },
        ]

        result = parse(payload, "fallback")

        assert result.active_leaf_message_provider_id == "assistant-1"
        assert [message.position for message in result.messages] == [0, 1]
        assert [message.variant_index for message in result.messages] == [0, 0]
        assert [message.is_active_path for message in result.messages] == [True, True]
        assert [message.is_active_leaf for message in result.messages] == [False, True]
        assert result.messages[0].occurred_at_ms == 1_767_225_600_000
        assert result.messages[0].model_name == "gpt-5-codex"
        assert result.messages[0].model_effort == "high"
        assert result.messages[0].input_tokens == 10
        assert result.messages[0].output_tokens == 2
        assert result.messages[0].cache_read_tokens == 3
        assert result.messages[0].cache_write_tokens == 4
        assert result.messages[0].duration_ms == 1250
        assert result.messages[1].model_name == "gpt-5-codex-mini"
        assert result.messages[1].model_effort == "medium"
        assert result.messages[1].input_tokens == 1
        assert result.messages[1].output_tokens == 20
        assert result.messages[1].duration_ms == 750

    def test_message_usage_accepts_codex_event_aliases(self) -> None:
        payload = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
                "usage": {
                    "inputTokenCount": 10,
                    "outputTokenCount": 4,
                    "cached_input_tokens": 3,
                    "cache_write_input_tokens": 2,
                },
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "again"}],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 5,
                    "cached_tokens": 7,
                    "cache_creation_input_tokens": 6,
                },
            },
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 2
        assert result.messages[0].input_tokens == 10
        assert result.messages[0].output_tokens == 4
        assert result.messages[0].cache_read_tokens == 3
        assert result.messages[0].cache_write_tokens == 2
        assert result.messages[1].input_tokens == 11
        assert result.messages[1].output_tokens == 5
        assert result.messages[1].cache_read_tokens == 7
        assert result.messages[1].cache_write_tokens == 6


# =============================================================================
# Git Context and Instructions
# =============================================================================


class TestGitContextAndInstructions:
    def test_git_context_from_session_meta(self) -> None:
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "s1",
                    "timestamp": "2024-01-01",
                    "git": {"branch": "main", "commit_hash": "abc123"},
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.git_branch == "main"
        assert result.git_commit_hash == "abc123"

    def test_instructions_from_session_meta(self) -> None:
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "s1",
                    "timestamp": "2024-01-01",
                    "instructions": "You are a helpful assistant.",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.instructions_text == "You are a helpful assistant."

    def test_git_context_from_intermediate_metadata(self) -> None:
        """Intermediate format: git context on first line."""
        payload = [
            {
                "id": "conv-xyz",
                "timestamp": "2024-01-01T12:00:00Z",
                "git": {"branch": "develop"},
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        ]
        result = parse(payload, "fallback")
        assert result.git_branch == "develop"

    def test_git_and_instructions_combined(self) -> None:
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "s1",
                    "timestamp": "2024-01-01",
                    "git": {"branch": "feature"},
                    "instructions": "Be concise.",
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.git_branch == "feature"
        assert result.instructions_text == "Be concise."

    def test_turn_context_cwd_feeds_working_directories(self) -> None:
        payload = [
            {"type": "turn_context", "payload": {"cwd": "/repo/polylogue"}},
            {"type": "turn_context", "payload": {"turn_context": {"cwd": "/repo/other"}}},
        ]

        result = parse(payload, "fallback")

        assert result.working_directories == ["/repo/other", "/repo/polylogue"]
        assert result.session_events[0].payload["cwd"] == "/repo/polylogue"

    def test_session_events_keep_compact_provenance_not_raw_payloads(self) -> None:
        payload = [
            {
                "type": "compacted",
                "payload": {
                    "message": "summary",
                    "replacement_history": [{"role": "user", "content": "large prior text"}],
                },
            },
            {"type": "turn_context", "payload": {"cwd": "/repo/polylogue", "large": "x" * 1024}},
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "id": "evt-1",
                    "call_id": "call-1",
                    "output": "large command output" * 1024,
                },
            },
        ]

        result = parse(payload, "fallback")

        assert [event.event_type for event in result.session_events] == [
            "compaction",
            "turn_context",
            "function_call_output",
        ]
        assert result.session_events[0].payload == {
            "source_index": 1,
            "summary": "summary",
            "replacement_history_count": 1,
        }
        assert result.session_events[1].payload == {
            "source_index": 2,
            "cwd": "/repo/polylogue",
        }
        assert result.session_events[2].payload == {
            "source_index": 3,
            "type": "function_call_output",
            "id": "evt-1",
            "call_id": "call-1",
            "output_chars": len("large command output" * 1024),
        }
        assert all("raw" not in event.payload for event in result.session_events)

    def test_function_call_output_omits_inline_image_data_urls_from_text(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-image",
                    "output": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64," + ("a" * 4096),
                        }
                    ],
                },
            }
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 1
        message = result.messages[0]
        assert "data:image/png;base64" not in (message.text or "")
        assert "<inline image omitted;" in (message.text or "")
        assert "mime=image/png" in (message.text or "")
        assert "sha256_base64=" in (message.text or "")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_empty_payload(self) -> None:
        result = parse([], "fallback")
        assert result.provider_session_id == "fallback"
        assert result.messages == []
        assert result.source_name == "codex"

    def test_all_state_records(self) -> None:
        payload = [{"record_type": "state"} for _ in range(5)]
        result = parse(payload, "fallback")
        assert len(result.messages) == 0

    def test_invalid_records_skipped(self) -> None:
        payload = [
            42,  # non-dict
            "string",  # non-dict
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1

    def test_provider_is_codex(self) -> None:
        result = parse([], "fallback")
        assert result.source_name == "codex"

    def test_timestamp_preserved_on_messages(self) -> None:
        """Message timestamps are preserved from record."""
        payload = [
            {
                "type": "message",
                "role": "user",
                "timestamp": "2024-03-15T10:30:00Z",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1
        assert result.messages[0].timestamp == "2024-03-15T10:30:00Z"

    def test_numeric_epoch_timestamp_is_normalized(self) -> None:
        """Numeric epoch timestamps survive typed validation and normalize to ISO text."""
        payload = [
            {
                "type": "message",
                "role": "user",
                "timestamp": 1705312200.0,
                "content": [{"type": "input_text", "text": "hello"}],
            },
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 1
        assert result.messages[0].timestamp == "2024-01-15T09:50:00+00:00"

    def test_session_updated_at_uses_latest_message_timestamp(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "conv-1", "timestamp": "2024-03-15T10:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "msg-1",
                    "role": "user",
                    "timestamp": "2024-03-15T10:30:00Z",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "msg-2",
                    "role": "assistant",
                    "timestamp": "2024-03-15T10:45:00Z",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
            },
        ]

        result = parse(payload, "fallback")

        assert result.created_at == "2024-03-15T10:00:00Z"
        assert result.updated_at == "2024-03-15T10:45:00Z"

    def test_message_id_fallback(self) -> None:
        """Message ID falls back to f'msg-{idx}' if not provided."""
        payload = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "first"}]},
            {
                "type": "message",
                "role": "user",
                "id": "explicit-id",
                "content": [{"type": "input_text", "text": "second"}],
            },
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 2
        # First message should have fallback ID
        assert result.messages[0].provider_message_id.startswith("msg-")
        # Second message should use explicit ID
        assert result.messages[1].provider_message_id == "explicit-id"

    def test_complex_real_world_payload(self) -> None:
        """Real-world example: envelope format with git, instructions, multiple messages."""
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "prod-session-001",
                    "timestamp": "2024-03-15T14:30:00Z",
                    "git": {"branch": "main", "commit_hash": "f1e2d3c"},
                    "instructions": "You are an expert Python developer.",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "How do I async/await?"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Use asyncio module."}],
                },
            },
            {"record_type": "state"},  # Ignored
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Show me an example."}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.provider_session_id == "prod-session-001"
        assert len(result.messages) == 3
        assert result.git_commit_hash == "f1e2d3c"
        assert result.instructions_text == "You are an expert Python developer."
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"
        assert result.messages[2].role == "user"

    def test_function_call_items_become_tool_messages_and_events(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd": "git status"}',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "clean",
                },
            },
            {
                "type": "response_item",
                "payload": {"type": "token_count", "input_tokens": 10, "output_tokens": 5},
            },
        ]

        result = parse(payload, "fallback")

        assert [event.event_type for event in result.session_events] == [
            "function_call",
            "function_call_output",
            "token_count",
        ]
        assert len(result.messages) == 2
        assert [message.position for message in result.messages] == [0, 1]
        assert result.active_leaf_message_provider_id == "call_1"
        assert result.messages[0].message_type is MessageType.TOOL_USE
        assert result.messages[0].blocks[0].type == "tool_use"
        assert result.messages[0].blocks[0].tool_name == "exec_command"
        assert result.messages[0].blocks[0].tool_input == {"cmd": "git status"}
        assert result.messages[1].message_type is MessageType.TOOL_RESULT
        assert result.messages[1].blocks[0].type == "tool_result"

    def test_event_msg_token_count_preserves_current_model_and_usage_extras(self) -> None:
        payload = [
            {
                "type": "turn_context",
                "payload": {"cwd": "/repo/polylogue", "model": "gpt-5-codex", "effort": "high"},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-01-01T00:00:02Z",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 10,
                            "cached_input_tokens": 2,
                            "cache_creation_input_tokens": 3,
                            "uncached_input_tokens": 8,
                            "output_tokens": 4,
                        },
                        "total_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 20,
                            "cache_creation_input_tokens": 30,
                            "uncached_input_tokens": 80,
                            "output_tokens": 40,
                        },
                        "model_context_window": 200000,
                    },
                },
            },
        ]

        result = parse(payload, "fallback")

        assert [event.event_type for event in result.session_events] == ["turn_context", "token_count"]
        usage_event = result.session_events[1]
        assert usage_event.timestamp == "2026-01-01T00:00:02Z"
        assert usage_event.payload["model"] == "gpt-5-codex"
        assert usage_event.payload["model_effort"] == "high"
        assert usage_event.payload["last_token_usage"] == {
            "input_tokens": 10,
            "cached_input_tokens": 2,
            "cache_write_tokens": 3,
            "uncached_input_tokens": 8,
            "output_tokens": 4,
        }
        assert usage_event.payload["total_token_usage"] == {
            "input_tokens": 100,
            "cached_input_tokens": 20,
            "cache_write_tokens": 30,
            "uncached_input_tokens": 80,
            "output_tokens": 40,
        }
        assert usage_event.payload["model_context_window"] == 200000

    def test_token_count_event_preserves_nested_usage_counters(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 111,
                            "cached_input_tokens": 22,
                            "output_tokens": 33,
                            "reasoning_output_tokens": 4,
                            "total_tokens": 170,
                        },
                        "total_token_usage": {
                            "input_tokens": 1000,
                            "cached_input_tokens": 9000,
                            "output_tokens": 300,
                            "reasoning_output_tokens": 40,
                            "total_tokens": 10340,
                        },
                        "model_context_window": 200000,
                    },
                },
            },
        ]

        result = parse(payload, "fallback")

        assert [event.event_type for event in result.session_events] == ["token_count"]
        assert result.session_events[0].payload == {
            "source_index": 1,
            "type": "token_count",
            "last_token_usage": {
                "input_tokens": 111,
                "cached_input_tokens": 22,
                "output_tokens": 33,
                "reasoning_output_tokens": 4,
                "total_tokens": 170,
            },
            "total_token_usage": {
                "input_tokens": 1000,
                "cached_input_tokens": 9000,
                "output_tokens": 300,
                "reasoning_output_tokens": 40,
                "total_tokens": 10340,
            },
            "model_context_window": 200000,
        }
