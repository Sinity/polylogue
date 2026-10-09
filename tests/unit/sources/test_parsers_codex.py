"""Dedicated tests for the Codex JSONL parser.

Covers format detection, envelope/direct parsing, session metadata,
branch tracking, git context, and edge cases.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.message.types import MessageType
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.session_revision_membership import MembershipRevision, classify_membership_revisions
from polylogue.core.enums import BlockType, MaterialOrigin, Role
from polylogue.pipeline.ids import session_content_hash, session_revision_projection
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.parsers.codex import _tool_input_from_arguments, is_supported_session_stream, parse_stream
from polylogue.sources.parsers.codex import looks_like as _looks_like_impl
from polylogue.sources.parsers.codex import parse as _parse_impl
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.storage_records import db_setup


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


class TestSessionStreamContract:
    def test_headerless_envelope_append_delta_uses_fallback_identity(self) -> None:
        from polylogue.sources.dispatch import parse_stream_payload

        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "append delta"}],
                },
            }
        ]

        assert is_supported_session_stream(payload)
        sessions = parse_stream_payload("codex", iter(payload), "rollout-fallback", source_path="/tmp/append.jsonl")

        assert [session.provider_session_id for session in sessions] == ["rollout-fallback"]
        assert len(sessions[0].messages) == 1

    def test_real_wire_stream_parses_and_passes_materialization_evidence_gate(self) -> None:
        from polylogue.sources.dispatch import admit_parsed_sessions_for_publication, parse_stream_payload

        payload = [
            {"type": "session_meta", "payload": {"id": "real-stream"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]

        assert is_supported_session_stream(payload)
        sessions = parse_stream_payload("codex", iter(payload), "fallback", source_path="/tmp/real-stream.jsonl")

        assert [session.provider_session_id for session in sessions] == ["real-stream"]
        assert (
            admit_parsed_sessions_for_publication(sessions, provider="codex", source_path="/tmp/real-stream.jsonl")
            == sessions
        )

    def test_session_header_plus_direct_messages_is_admitted_and_parsed(self) -> None:
        from polylogue.sources.dispatch import parse_stream_payload

        payload = [
            {"type": "session_meta", "payload": {"id": "legacy-stream"}},
            {
                "type": "message",
                "id": "legacy-user",
                "role": "user",
                "content": [{"type": "input_text", "text": "header plus direct"}],
            },
        ]

        assert is_supported_session_stream(payload)
        sessions = parse_stream_payload("codex", iter(payload), "fallback", source_path="/tmp/header-direct.jsonl")

        assert [session.provider_session_id for session in sessions] == ["legacy-stream"]
        assert len(sessions[0].messages) == 1
        assert sessions[0].messages[0].provider_message_id == "legacy-user"

    def test_legacy_direct_stream_uses_fallback_identity_and_passes_contract(self) -> None:
        from polylogue.sources.dispatch import admit_parsed_sessions_for_publication, parse_stream_payload

        payload = [
            {
                "type": "message",
                "id": "legacy-user",
                "timestamp": "2024-01-01T00:00:00Z",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
            {
                "type": "message",
                "id": "legacy-assistant",
                "timestamp": "2024-01-01T00:00:01Z",
                "role": "assistant",
                "content": [{"type": "text", "text": "world"}],
            },
        ]

        assert is_supported_session_stream(payload)
        sessions = parse_stream_payload("codex", iter(payload), "legacy-fallback", source_path="/tmp/legacy.jsonl")

        assert [session.provider_session_id for session in sessions] == ["legacy-fallback"]
        assert len(sessions[0].messages) == 2
        assert (
            admit_parsed_sessions_for_publication(sessions, provider="codex", source_path="/tmp/legacy.jsonl")
            == sessions
        )

    def test_bare_session_headers_fail_parser_contract_and_materialization_gate(self) -> None:
        from polylogue.sources.dispatch import admit_parsed_sessions_for_publication, parse_stream_payload

        payload = [{"type": "session_meta"}, {"type": "session_meta"}]

        assert not is_supported_session_stream(payload)
        sessions = parse_stream_payload("codex", iter(payload), "fallback", source_path="/tmp/bare-headers.jsonl")

        assert sessions[0].messages == []
        assert (
            admit_parsed_sessions_for_publication(sessions, provider="codex", source_path="/tmp/bare-headers.jsonl")
            == []
        )

    def test_mixed_envelope_and_direct_stream_fails_admission_and_evidence_gate(self) -> None:
        from polylogue.sources.dispatch import admit_parsed_sessions_for_publication, parse_stream_payload

        payload = [
            {"type": "session_meta", "payload": {"id": "mixed-stream"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "envelope"}],
                },
            },
            {
                "type": "message",
                "id": "legacy-user",
                "role": "user",
                "content": [{"type": "input_text", "text": "direct"}],
            },
        ]

        assert not is_supported_session_stream(payload)
        sessions = parse_stream_payload("codex", iter(payload), "fallback", source_path="/tmp/mixed.jsonl")

        assert sessions[0].messages
        assert (
            admit_parsed_sessions_for_publication(sessions, provider="codex", source_path="/tmp/mixed.jsonl")
            == sessions
        )


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

    def test_second_session_meta_with_structural_evidence_sets_continuation(self) -> None:
        # A genuine resume physically replays the parent conversation's own
        # session_meta as the second distinct meta in the file. Real Codex
        # rollout files (verified against multi-meta exports on disk) show
        # two structural facts about that replayed header: its timestamp
        # *precedes* the new session's own start time, and it reports the
        # same cwd/git remote, because the resume continues in the same
        # working tree.
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "conv-abc",
                    "timestamp": "2024-01-01T12:00:20Z",
                    "cwd": "/realm/project/sinnix",
                    "git": {"repository_url": "git@github.com:Sinity/sinnix.git"},
                },
            },
            {
                "type": "session_meta",
                "payload": {
                    "id": "parent-xyz",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "cwd": "/realm/project/sinnix",
                    "git": {"repository_url": "git@github.com:Sinity/sinnix.git"},
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
        assert result.provider_session_id == "conv-abc"
        assert result.parent_session_provider_id == "parent-xyz"
        assert result.branch_type == BranchType.CONTINUATION

    def test_second_session_meta_without_structural_evidence_stays_unclassified(self) -> None:
        # Two session_metas with no forked_from_id marker and no structural
        # relationship (different cwd/repo, and the second one's timestamp
        # does NOT precede the first's) must not be inferred as a
        # continuation from the bare count alone. Reverting to the old
        # count-based heuristic (`elif len(session_metas_seen) > 1:` with no
        # further check) makes this assertion fail because it would classify
        # this payload as CONTINUATION with parent "unrelated-session".
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "conv-abc",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "cwd": "/realm/project/sinnix",
                    "git": {"repository_url": "git@github.com:Sinity/sinnix.git"},
                },
            },
            {
                "type": "session_meta",
                "payload": {
                    "id": "unrelated-session",
                    "timestamp": "2024-06-15T09:00:00Z",
                    "cwd": "/realm/project/polylogue",
                    "git": {"repository_url": "git@github.com:Sinity/polylogue.git"},
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
        assert result.provider_session_id == "conv-abc"
        assert result.parent_session_provider_id is None
        assert result.branch_type is None

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

    def test_subagent_thread_spawn_alone_sets_subagent_parent(self) -> None:
        # The common real spawn shape: the parent is named only under
        # `source.subagent.thread_spawn`, with no `forked_from_id`.
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "child-3",
                    "source": {
                        "subagent": {
                            "thread_spawn": {
                                "parent_thread_id": "parent-3",
                                "depth": 1,
                                "agent_role": "explorer",
                                "agent_nickname": "Ironwood",
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
        assert result.provider_session_id == "child-3"
        assert result.parent_session_provider_id == "parent-3"
        assert result.branch_type == BranchType.SUBAGENT

    def test_top_level_parent_thread_id_sets_parent(self) -> None:
        # The rarest carrier: a top-level `parent_thread_id` with no subagent
        # block. It proves a parent but not the relationship type.
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "child-4",
                    "parent_thread_id": "parent-4",
                    "timestamp": "2024-01-01",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "resumed"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.parent_session_provider_id == "parent-4"
        assert result.branch_type is None

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

    def test_function_call_output_timed_out_is_error(self) -> None:
        """A structural ``"timed_out": true`` is itself an outcome signal
        (e.g. codex's ``wait``/``write_stdin`` timeout envelope) even when no
        exit_code/is_error field is present."""
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": '{"message": "Wait timed out.", "timed_out": true}',
                },
            },
        ]

        result = parse(payload, "fallback")

        block = result.messages[0].blocks[0]
        assert block.is_error is True
        assert block.exit_code is None

    def test_function_call_output_exec_envelope_success(self) -> None:
        """Codex's own unified-exec tool (``exec_command``/``write_stdin``)
        emits a fixed, CLI-generated text envelope rather than a JSON outcome
        object. This is the single largest unknown-outcome bucket in the live
        archive (polylogue-cuxz.8) -- read it as structure, not prose.
        """
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "Chunk ID: 04636b\nWall time: 0.6470 seconds\nProcess exited with code 0\nOutput:\nok\n",
                },
            },
        ]

        result = parse(payload, "fallback")

        block = result.messages[0].blocks[0]
        assert block.is_error is False
        assert block.exit_code == 0

    def test_function_call_output_exec_envelope_nonzero_exit(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "Wall time: 4.0933 seconds\nProcess exited with code 1\nOutput:\nboom\n",
                },
            },
        ]

        result = parse(payload, "fallback")

        block = result.messages[0].blocks[0]
        assert block.is_error is True
        assert block.exit_code == 1

    def test_function_call_output_exec_envelope_still_running_stays_unknown(self) -> None:
        """A long-lived chunked session that hasn't exited yet has no
        concluded outcome -- must stay NULL, never guessed as success."""
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "Chunk ID: 0f81ae\nWall time: 30.0011 seconds\nProcess running with session ID 3600\nOutput:\n...",
                },
            },
        ]

        result = parse(payload, "fallback")

        block = result.messages[0].blocks[0]
        assert block.is_error is None
        assert block.exit_code is None

    def test_function_call_output_exec_envelope_does_not_match_embedded_prose(self) -> None:
        """An unrelated occurrence of similar wording deep inside captured
        subprocess output (e.g. a CI log the command itself printed) must
        never be mistaken for the tool's own exit status -- only the exact
        preamble anchored at the very start of the field counts. The real
        (anchored) exit code here is 0; the embedded text says otherwise and
        must not override it.
        """
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": (
                        "Chunk ID: abc123\nWall time: 5.0 seconds\nProcess exited with code 0\n"
                        "Output:\n##[error]Process completed with exit code 1.\nsome CI log tail\n"
                    ),
                },
            },
        ]

        result = parse(payload, "fallback")

        block = result.messages[0].blocks[0]
        assert block.is_error is False
        assert block.exit_code == 0

    def test_state_records_skipped(self) -> None:
        payload = [
            {"record_type": "state"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"record_type": "state"},
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1

    def test_reasoning_summary_text_becomes_thinking_block(self) -> None:
        """polylogue-vf9x: real wire shape (anonymized from an operator rollout).

        Codex ships a standalone ``reasoning`` response_item with a
        human-readable ``summary`` (a list of ``summary_text`` parts) and an
        essentially-always-null ``content``, with the full trace only
        recoverable as ciphertext in ``encrypted_content``. Previously this
        record was read only by the generic session_event compactor -- which
        has no reasoning-specific branch -- so the summary text was silently
        discarded and never reached FTS/search.
        """
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "**Preparing to analyze git changes**"}],
                    "content": None,
                    "encrypted_content": "gAAAAABozSvL2wVUn4Ixsm34" * 4,
                },
            },
        ]
        result = parse(payload, "fallback")

        thinking_messages = [m for m in result.messages if m.message_type is MessageType.THINKING]
        assert len(thinking_messages) == 1
        message = thinking_messages[0]
        assert message.role == Role.ASSISTANT
        assert message.material_origin is MaterialOrigin.ASSISTANT_AUTHORED
        assert message.text == "**Preparing to analyze git changes**"
        assert len(message.blocks) == 1
        assert message.blocks[0].type == BlockType.THINKING
        assert message.blocks[0].text == "**Preparing to analyze git changes**"
        assert message.provider_message_id == ""

    def test_reasoning_reordering_keeps_synthetic_revision_identity(self) -> None:
        first = {
            "type": "response_item",
            "payload": {"type": "reasoning", "summary": [{"type": "summary_text", "text": "First"}]},
        }
        second = {
            "type": "response_item",
            "payload": {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Second"}]},
        }

        forward = parse([first, second], "codex-order")
        reordered = parse([second, first], "codex-order")

        assert (
            session_revision_projection(forward).message_contents
            == session_revision_projection(reordered).message_contents
        )

    def test_duplicate_reasoning_ids_keep_one_active_leaf(self) -> None:
        record = {
            "type": "response_item",
            "payload": {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Repeat"}]},
        }

        result = parse([record, record], "codex-duplicate-reasoning")

        assert result.messages[0].provider_message_id == result.messages[1].provider_message_id == ""
        assert result.active_leaf_message_provider_id is None
        assert sum(message.is_active_leaf is True for message in result.messages) == 1
        assert result.messages[-1].is_active_leaf is True

    def test_timestamp_less_duplicate_reasoning_is_membership_growth(self) -> None:
        record = {
            "type": "response_item",
            "payload": {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Repeat"}]},
        }
        one = parse([record], "codex-duplicate-membership")
        two = parse([record, record], "codex-duplicate-membership")

        classification = classify_membership_revisions(
            [
                MembershipRevision(raw_id="one", projection=session_revision_projection(one)),
                MembershipRevision(raw_id="two", projection=session_revision_projection(two)),
            ]
        )

        assert classification.accepted_raw_ids == ("one", "two")
        assert not classification.equivalent_raw_ids

    def test_idless_linear_turns_do_not_fabricate_parent_coordinates(self, workspace_env: Mapping[str, Path]) -> None:
        result = parse(
            [
                {
                    "type": "message",
                    "role": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "content": [{"type": "input_text", "text": "one"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "content": [{"type": "output_text", "text": "two"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "timestamp": "2026-01-01T00:00:02Z",
                    "content": [{"type": "input_text", "text": "three"}],
                },
            ],
            "codex-idless-parent-chain",
        )

        assert [message.provider_message_id for message in result.messages] == ["", "", ""]
        assert [message.parent_message_position for message in result.messages] == [None, None, None]

        with open_connection(db_setup(workspace_env)) as conn:
            write_fixture_index_session(conn, result, content_hash=session_content_hash(result))
            rows = conn.execute(
                "SELECT message_id, native_id, parent_message_id FROM messages ORDER BY position"
            ).fetchall()

        assert [row["native_id"] for row in rows] == [None, None, None]
        assert [row["parent_message_id"] for row in rows] == [None, None, None]

    def test_reasoning_with_only_encrypted_content_still_recorded(self) -> None:
        """No recoverable text (summary absent, content null) -- the block
        still exists with text=None so the FACT that reasoning occurred
        survives, matching the empty-body Claude Code thinking fix.
        """
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [],
                    "content": None,
                    "encrypted_content": "gAAAAABozSvL2wVUn4Ixsm34" * 4,
                },
            },
        ]
        result = parse(payload, "fallback")

        thinking_messages = [m for m in result.messages if m.message_type is MessageType.THINKING]
        assert len(thinking_messages) == 1
        message = thinking_messages[0]
        assert message.text is None
        assert len(message.blocks) == 1
        assert message.blocks[0].type == BlockType.THINKING
        assert message.blocks[0].text is None

    def test_reasoning_content_text_used_when_present(self) -> None:
        """`content` (the full trace) is read too, when the wire carries it
        as plain text rather than encrypted ciphertext."""
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "short summary"}],
                    "content": [{"type": "reasoning_text", "text": "the full reasoning trace"}],
                },
            },
        ]
        result = parse(payload, "fallback")

        thinking_messages = [m for m in result.messages if m.message_type is MessageType.THINKING]
        assert len(thinking_messages) == 1
        message = thinking_messages[0]
        block_texts = [b.text for b in message.blocks]
        assert block_texts == ["short summary", "the full reasoning trace"]

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

    def test_undated_event_user_message_preserves_unproved_response_mirror(self) -> None:
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

        assert len(result.messages) == 2
        assert [message.text for message in result.messages] == ["same prompt", "same prompt"]
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

    def test_parse_stream_uses_supplied_sink_and_preserves_duplicate_ids(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "conv-abc"}},
            {
                "type": "message",
                "id": "reused",
                "role": "user",
                "content": [{"type": "input_text", "text": "first"}],
            },
            {
                "type": "message",
                "id": "reused",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "second"}],
            },
        ]
        sink: list[ParsedMessage] = []

        streamed = parse_stream(iter(payload), "fallback", message_sink=sink)
        collected = parse(payload, "fallback")

        assert streamed.messages is sink
        assert streamed.model_copy(update={"messages": list(streamed.messages)}) == collected
        assert [message.is_active_leaf for message in sink] == [False, True]

    def test_stream_lookahead_pairs_repeated_code_mode_ids(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "conv-abc"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "functions.exec",
                    "call_id": "reused",
                    "arguments": 'await tools.exec_command({cmd: "first"})',
                },
            },
            {
                "type": "response_item",
                "payload": {"type": "function_call_output", "call_id": "reused", "output": "first output"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "functions.exec",
                    "call_id": "reused",
                    "arguments": 'await tools.exec_command({cmd: "second"})',
                },
            },
            {
                "type": "response_item",
                "payload": {"type": "function_call_output", "call_id": "reused", "output": "second output"},
            },
        ]

        streamed = parse_stream(iter(payload), "fallback")
        assert streamed == parse(payload, "fallback")
        child_outputs = [
            block.text
            for message in streamed.messages
            for block in message.blocks
            if block.type is BlockType.TOOL_RESULT and block.tool_id == "reused::polylogue-child::0"
        ]
        assert child_outputs == ["first output", "second output"]

    def test_sqlite_sinks_keep_late_event_updates_and_compaction_order(self, tmp_path: Path) -> None:
        from polylogue.sources.prepared_message_sink import SqliteMessageStore

        payload = [
            {"type": "session_meta", "payload": {"id": "conv-abc"}},
            {
                "type": "compacted",
                "payload": {
                    "message": "summary",
                    "replacement_history": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "history only"}],
                        }
                    ],
                },
            },
            {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "final only"}},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "retained reply"}],
            },
            {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "retained reply"}},
        ]
        store = SqliteMessageStore(tmp_path / "codex-sinks.db")
        try:
            messages = store.new_sink()
            events = store.new_event_sink()
            streamed = parse_stream(iter(payload), "fallback", message_sink=messages, event_sink=events)
            collected = parse(payload, "fallback")

            assert id(streamed.messages) == id(messages)
            assert id(streamed.session_events) == id(events)
            assert streamed.model_copy(update={"messages": list(messages), "session_events": list(events)}) == collected
            event_types = [event.event_type for event in events]
            compaction_index = event_types.index("compaction")
            assert event_types[compaction_index + 1] == "codex_replacement_context"
            completions = [event for event in events if event.event_type == "task_complete"]
            assert completions[0].payload["last_agent_message"] == "final only"
            assert completions[1].payload["last_agent_message_retained"] is True
        finally:
            store.close()

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
        # input_tokens=10 is inclusive of cache_read_tokens=3 per Codex's raw
        # convention; the parser subtracts cache out at the source so input
        # and cache_read are disjoint, additive billing lanes (7 + 3 == 10).
        assert result.messages[0].input_tokens == 7
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
        # Disjoint lanes: input_tokens is the raw provider value minus
        # cache_read_tokens (10-3=7, 11-7=4), not the raw inclusive value.
        assert result.messages[0].input_tokens == 7
        assert result.messages[0].output_tokens == 4
        assert result.messages[0].cache_read_tokens == 3
        assert result.messages[0].cache_write_tokens == 2
        assert result.messages[1].input_tokens == 4
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

    def test_known_orphan_response_item_types_pass_through_unchanged(self) -> None:
        """polylogue-fuky: audited-but-unextracted Codex types keep their wire name.

        ``thread_goal_updated`` and ``agent_reasoning`` are two of the
        previously-unaudited Codex response_item/event_msg types this bead's
        producer/consumer audit classified explicitly
        (``_CODEX_KNOWN_RESPONSE_ITEM_TYPES`` in ``sources/parsers/codex.py``).
        The classification is intentionally non-destructive at parse time --
        the parser still emits both under their own wire name; only
        ``agent_reasoning`` is later filtered out at write time
        (``_SESSION_EVENTS_REDUNDANT_TYPES``, storage/sqlite/archive_tiers/
        write.py) because it is a confirmed duplicate of the paired
        ``reasoning`` record's already-materialized THINKING-block message.
        """
        payload = [
            {
                "type": "event_msg",
                "payload": {"type": "thread_goal_updated", "goal": {"objective": "ship the thing"}},
            },
            {
                "type": "event_msg",
                "payload": {"type": "agent_reasoning", "text": "**Diagnosing the thing**"},
            },
        ]

        result = parse(payload, "fallback")

        assert [event.event_type for event in result.session_events] == [
            "thread_goal_updated",
            "agent_reasoning",
        ]

    def test_unaudited_response_item_type_routes_to_unclassified_bucket(self) -> None:
        """polylogue-fuky: a never-examined Codex type is fail-loud, not silent.

        Before this bead, any response_item/event_msg inner ``type`` this repo
        had never read joined the same vocabulary as every audited type,
        indistinguishable at the ``event_type`` level. A type outside
        ``_CODEX_KNOWN_RESPONSE_ITEM_TYPES`` must now route to the distinct,
        greppable ``codex_unclassified_response_item`` bucket instead --
        matching the ``claude_attachment_unclassified`` precedent in
        ``sources/parsers/claude/code_parser.py``. The original wire type
        string is not lost: it survives in the event payload's own ``type``
        field.
        """
        payload = [
            {
                "type": "event_msg",
                "payload": {"type": "a_type_this_repo_has_never_seen", "id": "evt-9"},
            },
        ]

        result = parse(payload, "fallback")

        assert len(result.session_events) == 1
        event = result.session_events[0]
        assert event.event_type == "codex_unclassified_response_item"
        assert event.payload["type"] == "a_type_this_repo_has_never_seen"

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

    def test_message_preserves_bounded_inline_image_evidence(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "message-image",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "inspect this image"},
                        {"type": "input_image", "image_url": "data:image/png;base64," + ("a" * 4096)},
                    ],
                },
            }
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 1
        message = result.messages[0]
        assert "data:image/png;base64" not in (message.text or "")
        assert message.text == "inspect this image"
        assert all("data:image/png;base64" not in (block.text or "") for block in message.blocks)
        image_blocks = [block for block in message.blocks if block.type is BlockType.IMAGE]
        assert len(image_blocks) == 1
        assert image_blocks[0].media_type == "image/png"
        assert "mime=image/png" in (image_blocks[0].text or "")
        assert "sha256_base64=" in (image_blocks[0].text or "")
        text_blocks = [block for block in message.blocks if block.type is BlockType.TEXT]
        assert [block.text for block in text_blocks] == ["inspect this image"]

    def test_whale_image_and_compaction_bytes_never_reach_search_text_or_fts(
        self, workspace_env: Mapping[str, Path]
    ) -> None:
        """polylogue-dhil: the two classes that dominate whale rollout bytes --
        inline base64 screenshots and the compaction snapshot that re-embeds the
        screenshot-laden prefix -- must not survive into the derived tier.

        Measured on a synthetic whale (2026-09-16): a 10,496,756-byte rollout
        that is 50 percent inline base64 and 50 percent compaction snapshot
        lowers to 3,602 bytes of ``blocks.text`` and 2,580 bytes of FTS, with
        zero base64-bearing blocks.

        Anti-vacuity: the parse-level assertions above pin the parser; this one
        pins the storage route those bytes would actually inflate. Dropping the
        ``_sanitize_codex_data_url`` call from ``_codex_inline_image_blocks``
        or from ``_codex_tool_output_text`` makes it red, and so does storing
        ``compacted.replacement_history`` content as message text, because
        ``blocks.search_text`` is a generated projection of ``blocks.text`` and
        ``messages_fts`` indexes it.
        """
        image_url = "data:image/png;base64," + ("a" * 200_000)
        history_entry = {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look at the screenshot"},
                {"type": "input_image", "image_url": image_url},
            ],
        }
        payload = [
            {"type": "session_meta", "payload": {"id": "whale-1", "timestamp": "2026-01-01T00:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m-user",
                    "role": "user",
                    "content": history_entry["content"],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-shot",
                    "output": [{"type": "input_image", "image_url": image_url}],
                },
            },
            {
                "type": "compacted",
                "timestamp": "2026-01-01T01:00:00Z",
                "payload": {"message": "compacted", "replacement_history": [history_entry] * 4},
            },
        ]

        result = parse(payload, "codex-whale-image")

        with open_connection(db_setup(workspace_env)) as conn:
            write_fixture_index_session(conn, result, content_hash=session_content_hash(result))
            leaked_blocks = conn.execute(
                "SELECT count(*) AS n FROM blocks WHERE search_text LIKE '%data:image/png;base64%'"
            ).fetchone()["n"]
            leaked_events = conn.execute(
                "SELECT count(*) AS n FROM session_events WHERE payload_json LIKE '%data:image/png;base64%'"
            ).fetchone()["n"]
            block_text_bytes = conn.execute(
                "SELECT coalesce(sum(length(coalesce(text, ''))), 0) AS n FROM blocks"
            ).fetchone()["n"]
            fts_bytes = conn.execute("SELECT coalesce(sum(length(block)), 0) AS n FROM messages_fts_data").fetchone()[
                "n"
            ]

        assert leaked_blocks == 0
        assert leaked_events == 0
        # Six inline copies of a 200 KB data URL entered the parse; the derived
        # tier keeps bounded placeholders, not the payload. The FTS shadow table
        # is bounded for the same reason -- it indexes ``blocks.search_text``,
        # which is a generated projection of ``blocks.text``.
        assert block_text_bytes < 10_000
        assert fts_bytes < 10_000

    def test_message_preserves_image_only_bounded_evidence(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "message-image-only",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "data:image/png;base64," + ("a" * 4096)},
                    ],
                },
            }
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 1
        message = result.messages[0]
        assert message.text == ""
        assert len(message.blocks) == 1
        assert message.blocks[0].type is BlockType.IMAGE
        assert message.blocks[0].media_type == "image/png"
        assert "data:image/png;base64" not in (message.blocks[0].text or "")
        assert "sha256_base64=" in (message.blocks[0].text or "")


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
        """An id-less message leaves the provider id empty for comparison fallback."""
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
        # First message should leave the provider id empty so the content
        # anchor in pipeline.ids handles comparison identity.
        assert result.messages[0].provider_message_id == ""
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
        # The call keeps its own id; the output repeats only call_id and so
        # takes the side-qualified native id, and its event names it.
        assert [message.provider_message_id for message in result.messages] == ["fc_1", "call_1::output"]
        assert [event.source_message_provider_id for event in result.session_events[:2]] == [
            "fc_1",
            "call_1::output",
        ]
        assert result.active_leaf_message_provider_id == "call_1::output"
        assert result.messages[0].message_type is MessageType.TOOL_USE
        assert result.messages[0].blocks[0].type == "tool_use"
        assert result.messages[0].blocks[0].tool_name == "exec_command"
        assert result.messages[0].blocks[0].tool_input == {
            "cmd": "git status",
            "command": "git status",
        }
        assert result.messages[1].message_type is MessageType.TOOL_RESULT
        assert result.messages[1].blocks[0].type == "tool_result"

    def test_exec_freeform_arguments_gain_canonical_command(self) -> None:
        script = 'const result = await tools.exec_command({"cmd":"polylogue find repo:polylogue"});'
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "call-exec",
                    "name": "exec",
                    "arguments": script,
                },
            }
        ]

        result = parse(payload, "fallback")

        block = result.messages[0].blocks[0]
        assert block.tool_name == "exec"
        assert block.tool_input == {"arguments": script, "command": script}

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
        assert "model_context_window" not in usage_event.payload

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
        }


# =============================================================================
# Newly-read wire fields (parser-diff triage, polylogue-t46-style unread-field pass)
# =============================================================================


class TestUnreadFieldTriage:
    """Fields identified by `devtools schema parser-diff --provider codex`
    that were previously parsed by name but silently dropped in value."""

    def test_turn_context_captures_truncation_policy_and_output_schema(self) -> None:
        payload = [
            {
                "type": "turn_context",
                "payload": {
                    "cwd": "/repo/polylogue",
                    "truncation_policy": {"mode": "tokens", "limit": 10000},
                    "final_output_json_schema": {
                        "type": "object",
                        "properties": {"shard_id": {"type": "string"}},
                        "required": ["shard_id"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        result = parse(payload, "fallback")
        turn_event = result.session_events[0]
        assert turn_event.event_type == "turn_context"
        assert turn_event.payload["truncation_policy"] == {"mode": "tokens", "limit": 10000}
        output_schema = cast(dict[str, Any], turn_event.payload["final_output_json_schema"])
        assert cast(dict[str, Any], output_schema["properties"])["shard_id"] == {"type": "string"}

    def test_turn_context_user_instructions_feed_session_instructions_text(self) -> None:
        payload = [
            {
                "type": "turn_context",
                "payload": {"cwd": "/repo/polylogue", "user_instructions": "# Sinnix Configuration\n..."},
            },
        ]
        result = parse(payload, "fallback")
        assert result.instructions_text == "# Sinnix Configuration\n..."

    def test_turn_context_user_instructions_do_not_override_legacy_instructions(self) -> None:
        payload = [
            {
                "type": "session_meta",
                "payload": {"id": "s1", "timestamp": "2024-01-01", "instructions": "Legacy prompt."},
            },
            {"type": "turn_context", "payload": {"user_instructions": "New-format prompt."}},
        ]
        result = parse(payload, "fallback")
        assert result.instructions_text == "Legacy prompt."

    def test_changed_turn_context_user_instructions_are_conserved(self) -> None:
        """A mid-session system-prompt edit must survive parsing.

        Anti-vacuity: restoring the first-wins guard
        (``if not session_instructions``) drops the second prompt entirely and
        leaves no ``codex_instructions_changed`` event, so the assertions fail.
        """
        payload = [
            {
                "type": "session_meta",
                "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"},
            },
            {
                "type": "turn_context",
                "payload": {"user_instructions": "First prompt."},
                "timestamp": "2026-01-01T00:00:01Z",
            },
            {
                "type": "message",
                "role": "user",
                "timestamp": "2026-01-01T00:00:02Z",
                "content": [{"type": "input_text", "text": "hello"}],
            },
            {
                "type": "turn_context",
                "payload": {"user_instructions": "Second prompt."},
                "timestamp": "2026-01-01T00:00:03Z",
            },
        ]
        result = parse(payload, "fallback")

        assert result.instructions_text == "First prompt."
        changes = [event for event in result.session_events if event.event_type == "codex_instructions_changed"]
        assert [event.payload["instructions"] for event in changes] == ["Second prompt."]
        assert changes[0].payload["instructions_kind"] == "user_instructions"
        assert changes[0].payload["revision"] == 2
        # Anchored at the next message, which the writer resolves to the first
        # message the new prompt applied to.
        assert changes[0].boundary_message_position == 1
        assert changes[0].timestamp == "2026-01-01T00:00:03Z"

    def test_repeated_turn_context_user_instructions_emit_no_change_event(self) -> None:
        """The dedup reason still holds: an identical re-declaration is not a change."""
        payload = [
            {"type": "turn_context", "payload": {"user_instructions": "Stable prompt."}},
            {"type": "turn_context", "payload": {"user_instructions": "Stable prompt."}},
            {"type": "turn_context", "payload": {"user_instructions": "Stable prompt."}},
        ]
        result = parse(payload, "fallback")

        assert result.instructions_text == "Stable prompt."
        assert [event.event_type for event in result.session_events] == ["turn_context"] * 3

    def test_recurring_user_instructions_value_is_recorded_once(self) -> None:
        """A prompt that alternates back is one distinct value, not a new one."""
        payload = [
            {"type": "turn_context", "payload": {"user_instructions": "A"}},
            {"type": "turn_context", "payload": {"user_instructions": "B"}},
            {"type": "turn_context", "payload": {"user_instructions": "A"}},
            {"type": "turn_context", "payload": {"user_instructions": "B"}},
            {"type": "turn_context", "payload": {"user_instructions": "C"}},
        ]
        result = parse(payload, "fallback")

        changes = [event for event in result.session_events if event.event_type == "codex_instructions_changed"]
        assert [event.payload["instructions"] for event in changes] == ["B", "C"]
        assert [event.payload["revision"] for event in changes] == [2, 3]

    def test_turn_context_user_instructions_differing_from_legacy_are_conserved(self) -> None:
        """The legacy field keeps ``instructions_text``; the new value is still kept."""
        payload = [
            {
                "type": "session_meta",
                "payload": {"id": "s1", "timestamp": "2024-01-01", "instructions": "Legacy prompt."},
            },
            {"type": "turn_context", "payload": {"user_instructions": "New-format prompt."}},
        ]
        result = parse(payload, "fallback")

        assert result.instructions_text == "Legacy prompt."
        changes = [event for event in result.session_events if event.event_type == "codex_instructions_changed"]
        assert [event.payload["instructions"] for event in changes] == ["New-format prompt."]

    def test_changed_developer_instructions_are_conserved(self) -> None:
        """``developer_instructions`` accumulates the same way as the user prompt."""
        payload = [
            {"type": "turn_context", "payload": {"developer_instructions": "You are an awaiter."}},
            {"type": "turn_context", "payload": {"developer_instructions": "You are an awaiter."}},
            {"type": "turn_context", "payload": {"developer_instructions": "You are a reviewer."}},
        ]
        result = parse(payload, "fallback")

        identity = [event for event in result.session_events if event.event_type == "codex_agent_identity"]
        assert [event.payload["developer_instructions"] for event in identity] == ["You are an awaiter."]
        changes = [event for event in result.session_events if event.event_type == "codex_instructions_changed"]
        assert [event.payload["instructions"] for event in changes] == ["You are a reviewer."]
        assert changes[0].payload["instructions_kind"] == "developer_instructions"

    def test_changed_user_instructions_are_retrievable_from_the_archive(
        self, workspace_env: Mapping[str, Path]
    ) -> None:
        """Both prompts must survive the production write route, not just the parse.

        Anti-vacuity: with the first-wins guard restored the second prompt is
        absent from both ``sessions.instructions_text`` and ``session_events``,
        so the union assertion fails.
        """
        result = parse(
            [
                {
                    "type": "session_meta",
                    "payload": {"id": "s-instructions", "timestamp": "2026-01-01T00:00:00Z"},
                },
                {
                    "type": "turn_context",
                    "payload": {"user_instructions": "First prompt."},
                    "timestamp": "2026-01-01T00:00:01Z",
                },
                {
                    "type": "message",
                    "role": "user",
                    "timestamp": "2026-01-01T00:00:02Z",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
                {
                    "type": "turn_context",
                    "payload": {"user_instructions": "Second prompt."},
                    "timestamp": "2026-01-01T00:00:03Z",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "timestamp": "2026-01-01T00:00:04Z",
                    "content": [{"type": "output_text", "text": "answered under the new prompt"}],
                },
            ],
            "codex-changed-instructions",
        )

        with open_connection(db_setup(workspace_env)) as conn:
            write_fixture_index_session(conn, result, content_hash=session_content_hash(result))
            stored_instructions = conn.execute("SELECT instructions_text FROM sessions").fetchone()[0]
            event_rows = conn.execute(
                "SELECT payload_json, boundary_message_id FROM session_events "
                "WHERE event_type = 'codex_instructions_changed' ORDER BY position"
            ).fetchall()
            anchored_text = conn.execute(
                "SELECT text FROM blocks WHERE message_id = ? ORDER BY position LIMIT 1",
                (event_rows[0]["boundary_message_id"],),
            ).fetchone()[0]

        assert stored_instructions == "First prompt."
        payloads = [json.loads(row["payload_json"]) for row in event_rows]
        assert [item["instructions"] for item in payloads] == ["Second prompt."]
        assert {stored_instructions} | {item["instructions"] for item in payloads} == {
            "First prompt.",
            "Second prompt.",
        }
        assert anchored_text == "answered under the new prompt"

    def test_session_meta_base_instructions_text_feeds_instructions_when_no_legacy_field(self) -> None:
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "s1",
                    "timestamp": "2024-01-01",
                    "base_instructions": {"text": "You are Codex, a coding agent."},
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.instructions_text == "You are Codex, a coding agent."

    def test_sandbox_policy_type_variant_and_exclude_flags_captured(self) -> None:
        payload = [
            {
                "type": "turn_context",
                "payload": {
                    "sandbox_policy": {
                        "type": "workspace-write",
                        "network_access": False,
                        "exclude_slash_tmp": True,
                        "exclude_tmpdir_env_var": False,
                    },
                },
            },
        ]
        result = parse(payload, "fallback")
        policy_events = [event for event in result.session_events if event.event_type == "agent_policy"]
        assert len(policy_events) == 1
        assert policy_events[0].payload == {
            "sandbox_policy": "workspace-write",
            "network_policy": "false",
            "exclude_slash_tmp": True,
            "exclude_tmpdir_env_var": False,
        }

    def test_session_meta_agent_role_and_developer_instructions_emit_one_identity_event(self) -> None:
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "s1",
                    "timestamp": "2024-01-01",
                    "agent_role": "awaiter",
                    "agent_nickname": "Ironwood",
                    "model_provider": "openai",
                },
            },
            {
                "type": "turn_context",
                "payload": {"developer_instructions": "You are an awaiter."},
            },
            {
                "type": "turn_context",
                "payload": {"developer_instructions": "You are an awaiter. (repeat turn)"},
            },
        ]
        result = parse(payload, "fallback")
        identity_events = [event for event in result.session_events if event.event_type == "codex_agent_identity"]
        assert len(identity_events) == 1
        assert identity_events[0].payload == {
            "agent_role": "awaiter",
            "agent_nickname": "Ironwood",
            "model_provider": "openai",
            "developer_instructions": "You are an awaiter.",
        }

    def test_token_count_captures_rate_limits(self) -> None:
        payload = [
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "rate_limits": {
                        "primary": {"used_percent": 1.0, "window_minutes": 299, "resets_in_seconds": 15211},
                        "secondary": {"used_percent": 34.0, "window_minutes": 10079, "resets_in_seconds": 210020},
                    },
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.session_events[0].payload["rate_limits"] == {
            "primary": {"used_percent": 1.0, "window_minutes": 299, "resets_in_seconds": 15211},
            "secondary": {"used_percent": 34.0, "window_minutes": 10079, "resets_in_seconds": 210020},
        }

    def test_reasoning_event_captures_metadata_turn_id(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "gAAAA...",
                    "metadata": {"turn_id": "019edbf0-a9e1-7842-939b-35838823eb5d"},
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.session_events[0].event_type == "reasoning"
        assert result.session_events[0].payload["turn_id"] == "019edbf0-a9e1-7842-939b-35838823eb5d"
        assert "encrypted_content" not in result.session_events[0].payload

    def test_ghost_snapshot_captures_ghost_commit(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "ghost_snapshot",
                    "ghost_commit": {
                        "id": "ae5788b8c19de5c4e52491004db8eab9b91910e1",
                        "parent": "51743ed0c6d39dc5191a69e7ba17e0e265e1b10c",
                        "preexisting_untracked_files": [],
                        "preexisting_untracked_dirs": [],
                    },
                },
            },
        ]
        result = parse(payload, "fallback")
        ghost_commit = cast(dict[str, Any], result.session_events[0].payload["ghost_commit"])
        assert ghost_commit["id"] == "ae5788b8c19de5c4e52491004db8eab9b91910e1"

    def test_exec_command_end_captures_process_id_and_parsed_cmd_not_duplicate_output(self) -> None:
        payload = [
            {
                "type": "event_msg",
                "payload": {
                    "type": "exec_command_end",
                    "call_id": "call-1",
                    "process_id": "57152",
                    "parsed_cmd": [{"type": "search", "cmd": "rg foo", "query": "foo", "path": "."}],
                    "aggregated_output": "duplicate of function_call_output text" * 10,
                    "formatted_output": "also duplicate",
                    "exit_code": 0,
                },
            },
        ]
        result = parse(payload, "fallback")
        event = result.session_events[0]
        assert event.payload["process_id"] == "57152"
        assert event.payload["parsed_cmd"] == [{"type": "search", "cmd": "rg foo", "query": "foo", "path": "."}]
        assert "aggregated_output" not in event.payload
        assert "formatted_output" not in event.payload

    def test_patch_apply_end_captures_success_and_structured_changes(self) -> None:
        payload = [
            {
                "type": "event_msg",
                "payload": {
                    "type": "patch_apply_end",
                    "call_id": "exec-1",
                    "turn_id": "turn-1",
                    "stdout": "Success. Updated the following files:\nM /repo/foo.py\n",
                    "stderr": "",
                    "success": True,
                    "changes": {
                        "/repo/foo.py": {
                            "type": "update",
                            "unified_diff": "@@ -1,1 +1,1 @@\n-old\n+new\n",
                            "move_path": None,
                        },
                        "/repo/renamed.py": {
                            "type": "update",
                            "unified_diff": "",
                            "move_path": "/repo/old_name.py",
                        },
                    },
                },
            },
        ]
        result = parse(payload, "fallback")
        event = result.session_events[0]
        assert event.event_type == "patch_apply_end"
        assert event.payload["success"] is True
        changes = cast(dict[str, Any], event.payload["changes"])
        assert changes["/repo/foo.py"] == {
            "type": "update",
            "unified_diff": "@@ -1,1 +1,1 @@\n-old\n+new\n",
        }
        assert changes["/repo/renamed.py"]["move_path"] == "/repo/old_name.py"
        # stdout/stderr duplicate the paired function_call_output tool_result
        # text, same dedup rule as exec_command -- deliberately not re-stored.
        assert "stdout" not in event.payload
        assert "stderr" not in event.payload

    def test_patch_apply_end_omits_changes_when_no_files_touched(self) -> None:
        payload = [
            {
                "type": "event_msg",
                "payload": {"type": "patch_apply_end", "call_id": "exec-2", "success": False, "changes": {}},
            },
        ]
        result = parse(payload, "fallback")
        event = result.session_events[0]
        assert event.payload["success"] is False
        assert "changes" not in event.payload

    def test_turn_context_captures_personality_reasoning_summary_collaboration_mode(self) -> None:
        payload = [
            {
                "type": "turn_context",
                "payload": {
                    "cwd": "/repo/polylogue",
                    "personality": "pragmatic",
                    "summary": "auto",
                    "collaboration_mode": {
                        "mode": "plan",
                        "settings": {"model": "gpt-5.4", "reasoning_effort": "medium"},
                    },
                },
            },
        ]
        result = parse(payload, "fallback")
        turn_event = result.session_events[0]
        assert turn_event.payload["personality"] == "pragmatic"
        assert turn_event.payload["reasoning_summary"] == "auto"
        assert turn_event.payload["collaboration_mode"] == "plan"

    def test_compacted_replacement_history_aggregates_phase_and_ghost_commit(self) -> None:
        payload = [
            {
                "type": "compacted",
                "payload": {
                    "message": "summary",
                    "replacement_history": [
                        # Codex carries `phase` on the history *entry*; content
                        # items are exactly {text,type} or {image_url,type}.
                        {
                            "type": "message",
                            "role": "assistant",
                            "phase": "final_answer",
                            "content": [
                                {"type": "output_text", "text": "final answer"},
                                {"type": "output_text", "text": "still the same answer"},
                            ],
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "phase": "draft",
                            "content": [{"type": "output_text", "text": "again"}],
                        },
                        {
                            "type": "ghost_snapshot",
                            "ghost_commit": {"id": "abc", "parent": "def"},
                        },
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
                        },
                    ],
                },
            },
        ]
        result = parse(payload, "fallback")
        event = result.session_events[0]
        assert event.payload["replacement_history_count"] == 4
        assert event.payload["replacement_history_phase_counts"] == {"draft": 1, "final_answer": 1}
        assert event.payload["replacement_history_ghost_commit_count"] == 1
        assert event.payload["replacement_history_image_count"] == 1

    def test_world_state_captures_environments_subagents(self) -> None:
        payload = [
            {
                "type": "world_state",
                "payload": {
                    "full": False,
                    "state": {"environments": {"subagents": "- backlog: Hubble\n- flagship: Dewey"}},
                },
            },
        ]
        result = parse(payload, "fallback")
        world_events = [event for event in result.session_events if event.event_type == "world_state"]
        assert len(world_events) == 1
        assert world_events[0].payload["environments"] == {"subagents": "- backlog: Hubble\n- flagship: Dewey"}

    def test_world_state_without_environments_emits_nothing(self) -> None:
        payload = [{"type": "world_state", "payload": {"full": True, "state": {"agents_md": "large text"}}}]
        result = parse(payload, "fallback")
        assert result.session_events == []

    def test_mcp_tool_call_end_produces_tool_use_and_tool_result_messages(self) -> None:
        payload = [
            {
                "type": "event_msg",
                "payload": {
                    "type": "mcp_tool_call_end",
                    "call_id": "call_Ep1dVZ1GUsjzkyKeRUEwlcdE",
                    "invocation": {
                        "server": "github",
                        "tool": "search_pull_requests",
                        "arguments": {"owner": "Sinity", "repo": "polylogue", "query": "is:merged"},
                    },
                    "duration": {"secs": 1, "nanos": 172352127},
                    "result": {"Ok": {"content": [{"type": "text", "text": '{"total_count": 2}'}]}},
                },
            },
        ]
        result = parse(payload, "fallback")

        assert len(result.messages) == 2
        use_message, result_message = result.messages
        assert use_message.role is Role.ASSISTANT
        assert use_message.blocks[0].type == "tool_use"
        assert use_message.blocks[0].tool_name == "mcp__github__search_pull_requests"
        assert use_message.blocks[0].tool_input == {
            "owner": "Sinity",
            "repo": "polylogue",
            "query": "is:merged",
        }
        assert result_message.role is Role.TOOL
        assert result_message.blocks[0].type == "tool_result"
        assert result_message.blocks[0].tool_id == use_message.blocks[0].tool_id
        assert result_message.blocks[0].is_error is False
        assert result_message.text is not None
        assert "total_count" in result_message.text

    def test_mcp_tool_call_end_err_result_marks_tool_result_as_error(self) -> None:
        payload = [
            {
                "type": "event_msg",
                "payload": {
                    "type": "mcp_tool_call_end",
                    "call_id": "call-err",
                    "invocation": {"server": "codex_apps", "tool": "github_fetch_issue", "arguments": {}},
                    "result": {"Err": "tool call error: token_expired"},
                },
            },
        ]
        result = parse(payload, "fallback")

        assert len(result.messages) == 2
        result_message = result.messages[1]
        assert result_message.blocks[0].is_error is True
        assert result_message.text == "tool call error: token_expired"


def test_standalone_apply_patch_exposes_the_operated_path_for_indexing() -> None:
    """polylogue-a9hx: apply_patch carries its payload as a PATCH-FORMAT STRING
    under ``arguments``, not JSON, so the operated-on path lives in a
    ``*** Update File:`` header where no ``json_extract`` can reach it.

    ``blocks.tool_path`` and ``blocks.search_text`` are both generated from
    ``$.file_path``/``$.path``, so every Codex file edit was invisible to
    structured path queries and to FTS -- measured 0.07% tool_path coverage
    for codex-session against 44% for claude-code-session, with apply_patch
    accounting for 95% of Codex tool calls.

    Mutation that fails this: remove the ``_PATCH_TOOL_NAMES`` branch from
    ``_tool_input_from_arguments`` -- ``path`` disappears and the generated
    column goes back to NULL.
    """
    patch = (
        "*** Begin Patch\n"
        "*** Update File: polylogue/sources/parsers/codex.py\n"
        "@@\n"
        "-old\n"
        "+new\n"
        "*** Add File: docs/notes.md\n"
        "+hello\n"
    )
    tool_input = _tool_input_from_arguments(patch, tool_name="apply_patch")

    assert tool_input["path"] == "polylogue/sources/parsers/codex.py"
    assert tool_input["paths"] == ["polylogue/sources/parsers/codex.py", "docs/notes.md"]
    assert tool_input["patch"] == patch


def test_non_patch_tool_arguments_are_left_alone() -> None:
    """A non-patch tool whose arguments happen to be a string must not be
    scanned for patch headers -- that would invent a path from prose."""
    tool_input = _tool_input_from_arguments("*** Update File: not-a-patch.txt", tool_name="add_issue_comment")

    assert "path" not in tool_input
    assert tool_input["arguments"] == "*** Update File: not-a-patch.txt"


# =============================================================================
# Codex lifecycle/event field conservation (polylogue-z1rdw/q0vka)
# =============================================================================


class TestCodexLifecycleEventFields:
    def test_declared_event_families_keep_semantic_fields(self) -> None:
        payload: list[dict[str, Any]] = [
            {
                "type": "event_msg",
                "payload": {
                    "type": "thread_goal_updated",
                    "goal": {"objective": "ship it", "status": "in_progress", "tokensUsed": 8, "timeUsedSeconds": 3.5},
                    "started_at": 10,
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "sub_agent_activity",
                    "agent_thread_id": "child",
                    "agent_path": "0.1",
                    "kind": "interacted",
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "task_started",
                    "turn_id": "turn-1",
                    "model_context_window": 200000,
                    "collaboration_mode_kind": "default",
                    "started_at": 11,
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "turn_id": "turn-1",
                    "last_agent_message": "unmirrored final",
                    "error": {"message": "limit", "codex_error_info": "usage_limit_exceeded"},
                    "completed_at": 12,
                },
            },
            {"type": "event_msg", "payload": {"type": "turn_aborted", "turn_id": "turn-2", "reason": "interrupted"}},
            {
                "type": "event_msg",
                "payload": {
                    "type": "thread_settings_applied",
                    "settings": {
                        "model": "gpt-test",
                        "reasoning_effort": "high",
                        "personality": "precise",
                        "collaboration_mode": {"kind": "agent", "settings": {"developer_instructions": "be precise"}},
                    },
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "collab_agent_interaction_end",
                    "receiver_thread_id": "child",
                    "receiver_agent_nickname": "Birch",
                    "receiver_agent_role": "worker",
                    "prompt": "inspect this",
                    "status": "completed",
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "entered_review_mode",
                    "target": {"instructions": "review", "user_facing_hint": "checking"},
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "exited_review_mode",
                    "review_output": {
                        "findings": ["none"],
                        "overall_correctness": "yes",
                        "overall_explanation": "good",
                        "overall_confidence_score": 0.9,
                    },
                },
            },
            {"type": "event_msg", "payload": {"type": "view_image_tool_call", "path": "/producer/image.png"}},
            {
                "type": "event_msg",
                "payload": {
                    "type": "web_search_end",
                    "query": "polylogue",
                    "action": {"queries": ["polylogue", "archive"]},
                },
            },
            {"type": "event_msg", "payload": {"type": "thread_rolled_back", "num_turns": 2}},
            {
                "type": "event_msg",
                "payload": {"type": "error", "message": "failed", "codex_error_info": "usage_limit_exceeded"},
            },
        ]
        result = parse(payload, "field-contract")
        by_type: dict[str, dict[str, Any]] = {
            event.event_type: cast(dict[str, Any], event.payload) for event in result.session_events
        }

        assert by_type["thread_goal_updated"]["goal"] == {
            "objective": "ship it",
            "status": "in_progress",
            "tokensUsed": 8,
            "timeUsedSeconds": 3.5,
        }
        assert by_type["sub_agent_activity"]["agent_thread_id"] == "child"
        assert by_type["task_started"]["turn_id"] == "turn-1"
        assert by_type["task_started"]["model_context_window"] == 200000
        assert by_type["task_complete"]["last_agent_message"] == "unmirrored final"
        assert by_type["task_complete"]["message"] == "limit"
        assert by_type["task_complete"]["codex_error_info"] == "usage_limit_exceeded"
        assert by_type["turn_aborted"]["turn_id"] == "turn-2"
        assert by_type["thread_settings_applied"]["collaboration_mode"] == {
            "kind": "agent",
            "developer_instructions": "be precise",
        }
        assert by_type["collab_agent_interaction_end"]["prompt"] == "inspect this"
        assert by_type["entered_review_mode"]["target"]["instructions"] == "review"
        assert by_type["exited_review_mode"]["review_output"]["findings"] == ["none"]
        assert by_type["view_image_tool_call"]["path"] == "/producer/image.png"
        assert by_type["web_search_end"]["action"] == {"queries": ["polylogue", "archive"]}
        assert by_type["thread_rolled_back"]["num_turns"] == 2
        assert by_type["error"]["codex_error_info"] == "usage_limit_exceeded"

    def test_turn_conflicts_and_user_references_are_explicit(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "phase": "commentary",
                    "metadata": {"turn_id": "metadata-turn"},
                    "internal_chat_message_metadata_passthrough": {"turn_id": "passthrough-turn"},
                    "content": [{"type": "output_text", "text": "hello"}],
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "look at this",
                    "local_images": [{"path": "/producer/private.png", "mime_type": "image/png"}],
                    "text_elements": [
                        {"type": "image", "start": 3, "end": 4, "placeholder": "[image]", "hidden": "not retained"}
                    ],
                },
            },
        ]
        result = parse(payload, "reference-contract")
        response = next(event for event in result.session_events if event.event_type == "response_item")
        assert response.payload["phase"] == "commentary"
        assert response.payload["turn_id"] == "metadata-turn"
        conflict = cast(dict[str, Any], response.payload["turn_id_conflict"])
        assert conflict["values"] == {
            "metadata.turn_id": "metadata-turn",
            "internal_chat_message_metadata_passthrough.turn_id": "passthrough-turn",
        }
        user = next(event for event in result.session_events if event.event_type == "user_message")
        local_images = cast(list[dict[str, Any]], user.payload["local_images"])
        assert local_images[0] == {
            "path": "/producer/private.png",
            "mime_type": "image/png",
            "source": "codex.user_message.local_images",
            "acquired_bytes": False,
            "path_disclosure": "provider_reference",
        }
        text_elements = cast(list[dict[str, Any]], user.payload["text_elements"])
        assert text_elements[0] == {
            "type": "image",
            "start": 3,
            "end": 4,
            "placeholder": "[image]",
            "source": "codex.user_message.text_elements",
            "content_policy": "range_only",
        }
        assert "hidden" not in text_elements[0]

    def test_top_level_records_have_bounded_named_payloads_and_are_supported(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "top-level", "timestamp": "2026-01-01T00:00:00Z"}},
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            },
            {
                "type": "inter_agent_communication_metadata",
                "payload": {"trigger_turn": True, "turn_id": "turn-3", "opaque": {"secret": "drop"}},
            },
            {
                "type": "token_usage_record",
                "payload": {"usage": {"input_tokens": 10, "output_tokens": 4}, "opaque": "drop"},
            },
        ]
        result = parse(payload, "top-level")
        assert is_supported_session_stream(payload)
        by_type: dict[str, dict[str, Any]] = {
            event.event_type: cast(dict[str, Any], event.payload) for event in result.session_events
        }
        assert by_type["inter_agent_communication_metadata"]["trigger_turn"] is True
        assert by_type["inter_agent_communication_metadata"]["turn_id"] == "turn-3"
        assert by_type["token_usage_record"]["usage"] == {"input_tokens": 10, "output_tokens": 4}
        assert "opaque" not in by_type["inter_agent_communication_metadata"]
        assert "opaque" not in by_type["token_usage_record"]

    def test_command_item_completed_does_not_duplicate_typed_result_text(self) -> None:
        payload = [
            {
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "item": {"type": "CommandExecution", "id": "cmd-1", "text": "stdout that has a typed result"},
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "item": {"type": "Plan", "id": "plan-1", "text": "full plan content"},
                },
            },
        ]
        result = parse(payload, "item-dedup")
        by_type: list[dict[str, Any]] = [
            cast(dict[str, Any], event.payload)
            for event in result.session_events
            if event.event_type == "item_completed"
        ]
        assert "text" not in by_type[0]["item"]
        assert by_type[1]["item"]["text"] == "full plan content"

    def test_lifecycle_completion_timing_and_named_ranges_are_conserved(self) -> None:
        payload = [
            {
                "type": "event_msg",
                "payload": {
                    "type": "task_started",
                    "turn_id": "turn-1",
                    "started_at": 1788727984,
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "turn_id": "turn-1",
                    "completed_at": 1788727987,
                    "duration_ms": 3326,
                    "last_agent_message": "only completion text",
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "text_elements": [
                        {
                            "type": "image",
                            "range": {"start": 1, "end": 2, "private": "drop"},
                            "placeholder": "[image]",
                        }
                    ],
                },
            },
            {
                "type": "token_usage_record",
                "input_tokens": 10,
                "output_tokens": 4,
                "opaque": "drop",
            },
        ]
        result = parse(payload, "timing-contract")
        by_type = {event.event_type: cast(dict[str, Any], event.payload) for event in result.session_events}
        assert by_type["task_started"]["started_at"] == 1788727984
        assert by_type["task_complete"]["completed_at"] == 1788727987
        assert by_type["task_complete"]["duration_ms"] == 3326
        assert by_type["user_message"]["text_elements"][0]["range"] == {"start": 1, "end": 2}
        assert by_type["token_usage_record"]["usage"] == {"input_tokens": 10, "output_tokens": 4}
        assert "opaque" not in by_type["token_usage_record"]


# =============================================================================
# Re-embedded context conservation (replacement_history, task_complete)
# =============================================================================


def _replacement_contexts(session: ParsedSession) -> list[dict[str, Any]]:
    return [dict(event.payload) for event in session.session_events if event.event_type == "codex_replacement_context"]


def _re_embedded_source_texts(payload: list[dict[str, Any]]) -> list[str]:
    """Every text value the raw records re-embed rather than emit live."""
    texts: list[str] = []
    for record in payload:
        inner = record.get("payload") if isinstance(record.get("payload"), dict) else record
        assert isinstance(inner, dict)
        if record.get("type") == "compacted":
            for entry in inner.get("replacement_history") or []:
                for item in entry.get("content") or []:
                    if isinstance(item.get("text"), str) and item["text"]:
                        texts.append(item["text"])
        if inner.get("type") == "task_complete" and isinstance(inner.get("last_agent_message"), str):
            texts.append(inner["last_agent_message"])
    return texts


def _retained_texts(session: ParsedSession) -> set[str]:
    """Every text value the parsed session holds, by any route."""

    def walk(value: object, sink: set[str]) -> None:
        if isinstance(value, str):
            if value:
                sink.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item, sink)
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item, sink)

    out: set[str] = set()
    if session.instructions_text:
        out.add(session.instructions_text)
    for message in session.messages:
        if message.text:
            out.add(message.text)
        for block in message.blocks:
            if block.text:
                out.add(block.text)
            if block.tool_input:
                walk(dict(block.tool_input), out)
    for event in session.session_events:
        walk(event.payload, out)
    return out


def _live_user_turn(text: str, timestamp: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "timestamp": timestamp,
        "content": [{"type": "input_text", "text": text}],
    }


def _history_entry(text: str, *, role: str = "user", phase: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}
    if phase is not None:
        entry["phase"] = phase
    return entry


def _compaction(*entries: dict[str, Any], message: str = "", timestamp: str = "2026-01-01T00:10:00Z") -> dict[str, Any]:
    return {
        "type": "compacted",
        "timestamp": timestamp,
        "payload": {"message": message, "replacement_history": list(entries)},
    }


ENVIRONMENT_CONTEXT = "<environment_context>\n  <cwd>/realm/project/polylogue</cwd>\n</environment_context>"


class TestReplacementHistoryConservation:
    """`compacted.replacement_history` re-embeds pre-compaction records.

    A value the session already holds must not be stored twice; a value it
    holds nowhere else must survive. Anti-vacuity for the whole class:
    dropping the residual entirely (the pre-polylogue-6ev92 behaviour, bounded
    aggregates only) leaves no ``codex_replacement_context`` event and every
    "stored once" assertion fails; storing the history verbatim instead makes
    every suppression assertion fail.
    """

    def test_replacement_only_user_content_is_stored_once(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
            _live_user_turn("what does the parser drop?", "2026-01-01T00:00:01Z"),
            _compaction(
                _history_entry("what does the parser drop?"),
                _history_entry(ENVIRONMENT_CONTEXT, phase="turn_construction"),
                _history_entry(ENVIRONMENT_CONTEXT),
                message="earlier turns summarized",
            ),
        ]
        result = parse(payload, "fallback")

        contexts = _replacement_contexts(result)
        assert [item["content"] for item in contexts] == [ENVIRONMENT_CONTEXT]
        assert contexts[0]["occurrences"] == 2
        assert contexts[0]["context_kind"] == "replacement_history"
        assert contexts[0]["role"] == "user"
        assert contexts[0]["phase"] == "turn_construction"
        assert contexts[0]["content_chars"] == len(ENVIRONMENT_CONTEXT)

    def test_live_stream_duplicates_are_suppressed(self) -> None:
        """The 97.8% the exclusion reason is built on must stay excluded."""
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
            _live_user_turn("first turn", "2026-01-01T00:00:01Z"),
            _live_user_turn("second turn", "2026-01-01T00:00:02Z"),
            _compaction(_history_entry("first turn"), _history_entry("second turn")),
        ]
        result = parse(payload, "fallback")

        assert _replacement_contexts(result) == []
        compaction = next(event for event in result.session_events if event.event_type == "compaction")
        assert compaction.payload["replacement_history_text_count"] == 2
        assert "replacement_history_context_count" not in compaction.payload

    def test_a_value_the_live_stream_reaches_only_later_is_suppressed(self) -> None:
        """Resolution is deferred to the end of the parse.

        Anti-vacuity: resolving at the compaction record instead -- against the
        prefix parsed so far -- stores this value and the assertion fails.
        """
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
            _compaction(
                _history_entry("a turn replayed before it is re-sent"),
                _history_entry(ENVIRONMENT_CONTEXT),
            ),
            _live_user_turn("a turn replayed before it is re-sent", "2026-01-01T00:20:00Z"),
        ]
        result = parse(payload, "fallback")

        # The residual in the same compaction is still stored, so a run that
        # stores nothing cannot pass this by doing nothing.
        assert [item["content"] for item in _replacement_contexts(result)] == [ENVIRONMENT_CONTEXT]

    def test_ancestor_prefix_duplicates_are_suppressed(self) -> None:
        """A fork replays its parent's prefix into its own file.

        The replayed prefix is parsed as this session's own messages, so a
        replacement_history value repeating it is already retained and must not
        be stored a second time -- while the fork's own replacement-only
        context still must be.
        """
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "child",
                    "timestamp": "2026-01-01T01:00:00Z",
                    "forked_from_id": "parent",
                },
            },
            _live_user_turn("inherited parent turn", "2026-01-01T00:00:01Z"),
            _live_user_turn("the fork's own turn", "2026-01-01T01:00:01Z"),
            _compaction(
                _history_entry("inherited parent turn"),
                _history_entry("the fork's own turn"),
                _history_entry(ENVIRONMENT_CONTEXT),
            ),
        ]
        result = parse(payload, "fallback")

        assert result.parent_session_provider_id == "parent"
        assert [item["content"] for item in _replacement_contexts(result)] == [ENVIRONMENT_CONTEXT]

    def test_context_events_follow_their_own_compaction_event(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
            _compaction(_history_entry("first residual"), message="first", timestamp="2026-01-01T00:10:00Z"),
            _live_user_turn("a turn between compactions", "2026-01-01T00:11:00Z"),
            _compaction(_history_entry("second residual"), message="second", timestamp="2026-01-01T00:20:00Z"),
        ]
        result = parse(payload, "fallback")

        ordered = [
            (event.event_type, event.payload.get("summary") or event.payload.get("content"))
            for event in result.session_events
        ]
        assert ordered == [
            ("compaction", "first"),
            ("codex_replacement_context", "first residual"),
            ("compaction", "second"),
            ("codex_replacement_context", "second residual"),
        ]
        for event in result.session_events:
            if event.event_type == "compaction":
                assert event.payload["replacement_history_context_count"] == 1

    def test_one_value_repeated_across_compactions_is_stored_once(self) -> None:
        """Successive compactions re-embed the same text; it is one value."""
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
            _compaction(_history_entry(ENVIRONMENT_CONTEXT), timestamp="2026-01-01T00:10:00Z"),
            _compaction(_history_entry(ENVIRONMENT_CONTEXT), timestamp="2026-01-01T00:20:00Z"),
        ]
        result = parse(payload, "fallback")

        contexts = _replacement_contexts(result)
        assert [item["content"] for item in contexts] == [ENVIRONMENT_CONTEXT]
        assert contexts[0]["occurrences"] == 2

    def test_large_replacement_text_is_stored_once_through_the_archive_write(
        self, workspace_env: Mapping[str, Path]
    ) -> None:
        """Size never demotes replacement-only text to a digest.

        Anti-vacuity: reinstate a per-value size ceiling and the stored event
        lacks the text, so the archive row does not contain it.
        """
        large = "repeated context " + ("x" * (256 * 1024 + 1))
        result = parse(
            [
                {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
                _compaction(_history_entry(large), message="summarized"),
                _compaction(_history_entry(large), message="summarized again"),
            ],
            "fallback",
        )

        contexts = _replacement_contexts(result)
        assert [item["content"] for item in contexts] == [large]
        assert contexts[0]["occurrences"] == 2

        with open_connection(db_setup(workspace_env)) as conn:
            write_fixture_index_session(conn, result, content_hash=session_content_hash(result))
            rows = conn.execute(
                "SELECT payload_json FROM session_events WHERE event_type = 'codex_replacement_context'"
            ).fetchall()
        assert len(rows) == 1
        assert large in str(rows[0]["payload_json"])

    def test_replacement_only_content_survives_the_archive_write(self, workspace_env: Mapping[str, Path]) -> None:
        """The production write route, not just the parse, must keep it."""
        result = parse(
            [
                {"type": "session_meta", "payload": {"id": "s-residual", "timestamp": "2026-01-01T00:00:00Z"}},
                _live_user_turn("a live turn", "2026-01-01T00:00:01Z"),
                _compaction(
                    _history_entry("a live turn"),
                    _history_entry(ENVIRONMENT_CONTEXT, role="developer"),
                    message="summarized",
                ),
            ],
            "codex-replacement-residual",
        )

        with open_connection(db_setup(workspace_env)) as conn:
            write_fixture_index_session(conn, result, content_hash=session_content_hash(result))
            rows = conn.execute(
                "SELECT payload_json FROM session_events "
                "WHERE event_type = 'codex_replacement_context' ORDER BY position"
            ).fetchall()

        payloads = [json.loads(row["payload_json"]) for row in rows]
        assert [item["content"] for item in payloads] == [ENVIRONMENT_CONTEXT]
        assert payloads[0]["role"] == "developer"


class TestTaskCompleteLastAgentMessage:
    """`task_complete.last_agent_message` repeats the turn's final reply.

    Measured over 270 real rollout files, all 1,565 occurrences were already
    stored from the same file's live stream. Anti-vacuity: storing the text
    unconditionally fails the suppression test; dropping the field entirely --
    the pre-polylogue-6ev92 behaviour -- fails the residual test.
    """

    def test_duplicate_of_the_live_reply_is_recorded_not_restored(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
            {
                "type": "message",
                "role": "assistant",
                "timestamp": "2026-01-01T00:00:01Z",
                "content": [{"type": "output_text", "text": "the final reply"}],
            },
            {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "the final reply"}},
        ]
        result = parse(payload, "fallback")

        event = next(event for event in result.session_events if event.event_type == "task_complete")
        assert event.payload["last_agent_message_retained"] is True
        assert event.payload["last_agent_message_chars"] == len("the final reply")
        assert "last_agent_message" not in event.payload

    def test_a_reply_the_live_stream_never_carried_is_stored_once(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z"}},
            {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "an unmirrored reply"}},
            {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "an unmirrored reply"}},
        ]
        result = parse(payload, "fallback")

        events = [event for event in result.session_events if event.event_type == "task_complete"]
        assert events[0].payload["last_agent_message"] == "an unmirrored reply"
        assert "last_agent_message" not in events[1].payload
        assert events[1].payload["last_agent_message_retained"] is True

    def test_task_completion_alias_preserves_each_wire_length(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1"}},
            {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "e\u0301"}},
            {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "\u00e9"}},
        ]

        session = parse_stream(iter(payload), "fallback")
        events = [event for event in session.session_events if event.event_type == "task_complete"]

        assert events[0].payload["last_agent_message"] == "e\u0301"
        assert events[1].payload["last_agent_message_retained"] is True
        assert [event.payload["last_agent_message_chars"] for event in events] == [2, 1]


class TestReEmbeddedTextConservation:
    """Source-to-parser conservation: no re-embedded value goes missing.

    The same check the measurement ran over real rollouts, on a fixture that
    composes every shape at once. Anti-vacuity: dropping either residual route
    leaves its value unreachable and the assertion names it.
    """

    def test_every_re_embedded_value_is_reachable_from_the_parsed_session(self) -> None:
        payload: list[dict[str, Any]] = [
            {
                "type": "session_meta",
                "payload": {"id": "child", "timestamp": "2026-01-01T01:00:00Z", "forked_from_id": "parent"},
            },
            {"type": "turn_context", "payload": {"user_instructions": "First prompt."}},
            _live_user_turn("inherited parent turn", "2026-01-01T00:00:01Z"),
            _live_user_turn("the fork's own turn", "2026-01-01T01:00:01Z"),
            {
                "type": "message",
                "role": "assistant",
                "timestamp": "2026-01-01T01:00:02Z",
                "content": [{"type": "output_text", "text": "the mirrored reply"}],
            },
            {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "the mirrored reply"}},
            {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "an unmirrored reply"}},
            _compaction(
                _history_entry("inherited parent turn"),
                _history_entry("the fork's own turn"),
                _history_entry(ENVIRONMENT_CONTEXT, phase="turn_construction"),
                _history_entry("<skills_instructions>use the archive</skills_instructions>", role="developer"),
                _history_entry("a user turn that only the history kept"),
                message="summarized",
            ),
            {"type": "turn_context", "payload": {"user_instructions": "Second prompt."}},
        ]
        result = parse(payload, "fallback")

        retained = _retained_texts(result)
        missing = [text for text in _re_embedded_source_texts(payload) if text not in retained]
        assert missing == []
        assert {"First prompt.", "Second prompt."} <= retained | {result.instructions_text}
        # Conserved, not duplicated: only the values with no other home.
        assert sorted(item["content"] for item in _replacement_contexts(result)) == sorted(
            [
                ENVIRONMENT_CONTEXT,
                "<skills_instructions>use the archive</skills_instructions>",
                "a user turn that only the history kept",
            ]
        )


def test_event_msg_message_rollout_both_classifies_and_parses() -> None:
    """polylogue-1wom1: a Codex rollout whose conversational content arrives as
    ``event_msg`` payloads of type ``user_message``/``agent_message`` is a
    supported stream. ``parse`` has always materialized those two turns, but
    ``is_supported_session_stream`` refused the stream, so artifact
    classification declined ``parse_as_session`` and the messages reached the
    archive from nowhere -- silent evidence loss for a shape the parser
    handles.

    Anti-vacuity: dropping the ``event_msg`` branch from ``_message_record``
    leaves ``has_message`` False in ``_session_stream_supported`` and the
    classification assertion goes red while the parse assertion still passes --
    exactly the disagreement this test exists to forbid.
    """
    records: list[object] = [
        {"type": "session_meta", "payload": {"id": "event-msg-sess", "timestamp": "2026-04-01T00:00:00Z"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "hello there"}},
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "hi back"}},
    ]

    assert is_supported_session_stream(records) is True

    session = parse(records, "fallback")
    assert [(message.role, message.text) for message in session.messages] == [
        (Role.USER, "hello there"),
        (Role.ASSISTANT, "hi back"),
    ]


def test_world_state_state_keys_are_stored_or_declared_exempt() -> None:
    """polylogue-w54q3: ``world_state.state`` keeps every key it carries except
    the declared context-file text in ``_WORLD_STATE_INSTRUCTION_TEXT_KEYS``.
    ``host_skills``, ``permissions`` and ``collaboration_mode`` were previously
    dropped with no reason at all; the exemption set is now the one declaration
    and a new upstream key is stored rather than silently discarded.

    Anti-vacuity: dropping ``permissions`` back out of the stored payload, or
    widening the exemption set to cover an unnamed key, makes the
    stored-or-exempt partition assertion go red.
    """
    from polylogue.sources.parsers.codex import _WORLD_STATE_INSTRUCTION_TEXT_KEYS

    state = {
        "environments": {"local": {"cwd": "/tmp", "status": "available"}},
        "permissions": {"approved_command_prefixes": [["ls"]]},
        "collaboration_mode": {"mode": "default"},
        "host_skills": {"body": "## Skills\nrepeated catalog text"},
        "agents_md": {"text": "repeated context-file text"},
        "skills": {"includeInstructions": True},
    }
    records: list[object] = [
        {"type": "session_meta", "payload": {"id": "world-state-sess", "timestamp": "2026-04-01T00:00:00Z"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "go"}},
        {"type": "world_state", "payload": {"state": state}},
    ]

    session = parse(records, "fallback")
    event = next(event for event in session.session_events if event.event_type == "world_state")
    stored = set(event.payload) - {"source_index"}

    assert stored | _WORLD_STATE_INSTRUCTION_TEXT_KEYS >= set(state)
    assert stored & _WORLD_STATE_INSTRUCTION_TEXT_KEYS == set()
    assert {"environments", "permissions", "collaboration_mode"} <= stored


def test_replacement_candidates_hold_their_text_once_under_fixed_size_keys() -> None:
    """A large replacement-only value costs one scratch copy, keyed by digest.

    Anti-vacuity: key the scratch tables by the pickled text again and the
    key, lookup-value and context columns each hold another full copy.
    """
    import sqlite3
    import unicodedata

    from polylogue.sources.parsers.codex import _CodexLookaheadIndex, _CodexTextConservation

    connection = sqlite3.connect(":memory:")
    conservation = _CodexTextConservation(_CodexLookaheadIndex(connection))
    large = "é" + "x" * 200_000
    key, new = conservation.add(large)
    assert new and key is not None and len(key) == 32
    conservation.add_context(key, insert_at=1, timestamp=None, source_index=0, entry_type=None, role=None, phase=None)
    assert conservation.add(large) == (key, False)
    assert conservation._candidate(unicodedata.normalize("NFC", large)) == key
    assert conservation._candidate("x" * 200_001) is None

    scratch = connection.execute(
        """SELECT
               (SELECT SUM(length(key) + length(text)) FROM codex_replacement_texts),
               (SELECT SUM(length(value) + length(candidate_key)) FROM codex_replacement_keys),
               (SELECT SUM(length(key)) FROM codex_replacement_contexts)"""
    ).fetchone()
    texts_bytes, keys_bytes, contexts_bytes = scratch
    assert texts_bytes < len(large.encode()) + 1024
    assert keys_bytes == 4 * 32 and contexts_bytes == 32


def test_replacement_context_annotations_keep_a_lone_surrogate() -> None:
    """Anti-vacuity: bind the annotations as scratch TEXT again and a lone
    surrogate raises ``UnicodeEncodeError`` before the event is built."""
    import sqlite3

    from polylogue.sources.parsers.base import ParsedSessionEvent
    from polylogue.sources.parsers.codex import _CodexLookaheadIndex, _CodexTextConservation

    conservation = _CodexTextConservation(_CodexLookaheadIndex(sqlite3.connect(":memory:")))
    key, _new = conservation.add("replacement value")
    assert key is not None
    conservation.add_context(
        key, insert_at=1, timestamp=None, source_index=0, entry_type="t\ud800", role="r\ud800", phase=None
    )
    events = [ParsedSessionEvent(event_type="compaction", payload={})]
    conservation.finish_replacement_contexts(events)

    (context,) = [event for event in events if event.event_type == "codex_replacement_context"]
    assert (context.payload["entry_type"], context.payload["role"]) == ("t\ud800", "r\ud800")


def test_candidate_digest_is_windowed_and_never_rereads_the_stored_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: pickle the whole text to hash it, or unpickle the stored
    text on every lookup, and the guarded ``pickle`` calls fail."""
    import sqlite3

    from polylogue.sources.parsers import codex as codex_module

    monkeypatch.setattr(codex_module, "_DIGEST_WINDOW_CHARS", 7)
    conservation = codex_module._CodexTextConservation(codex_module._CodexLookaheadIndex(sqlite3.connect(":memory:")))
    large = "x\ud800" * 50
    key, _new = conservation.add(large)
    assert key == codex_module._text_digest(large) and len(key) == 32

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the full candidate was materialized again")

    import pickle

    monkeypatch.setattr(pickle, "loads", refuse)
    assert conservation._candidate(large) == key
    assert conservation.add(large) == (key, False)


def test_candidate_lookup_normalizes_only_after_the_exact_probe_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: build the NFC copy before probing the exact digest and the
    guarded ``normalize`` is called for a text registered verbatim."""
    import sqlite3
    import unicodedata

    from polylogue.sources.parsers import codex as codex_module

    conservation = codex_module._CodexTextConservation(codex_module._CodexLookaheadIndex(sqlite3.connect(":memory:")))
    decomposed = "é" * 50
    key, _new = conservation.add(decomposed)
    real = unicodedata.normalize

    def guarded(form: Any, value: str) -> str:
        assert value != decomposed, "normalized a text whose exact digest matches"
        return real(form, value)

    from types import SimpleNamespace

    monkeypatch.setattr(
        "polylogue.sources.parsers.codex.unicodedata",
        SimpleNamespace(normalize=guarded, combining=unicodedata.combining),
    )
    assert conservation._candidate(decomposed) == key


def test_js_literal_escaped_surrogate_pair_is_one_character() -> None:
    """Anti-vacuity: keep the two code units and the stored JSON escapes them
    adjacently, which every JSON reader combines into U+1F600."""
    from polylogue.sources.parsers.codex import _parse_js_literal

    value, ok = _parse_js_literal('"a\\uD83D\\uDE00b \\uD800"')
    assert ok and value == "a\U0001f600b \ud800"


def test_sink_restores_surrogates_inside_the_owner_coordinate() -> None:
    """Anti-vacuity: restore only strings, dicts and lists and the owner
    coordinate's stable key keeps the random surrogate marker."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.message_owner import MessageOwnerCoordinate
    from polylogue.sources.parsers.base import ParsedMessage
    from polylogue.sources.prepared_message_sink import _from_text_json, _message_json

    message = ParsedMessage(provider_message_id="m\ud800", role=Role.USER, text="t\ud800")
    message = message.model_copy(update={"owner_coordinate": MessageOwnerCoordinate(stable_key="s\ud800")})
    restored = _from_text_json(ParsedMessage, _message_json(message))
    assert restored.owner_coordinate is not None
    assert restored.owner_coordinate.stable_key == "s\ud800"
    assert restored.text == "t\ud800"


def test_a_cesu8_pair_in_provider_bytes_decodes_to_one_character() -> None:
    """Anti-vacuity: keep the two surrogatepass code units and the stored
    escaped JSON re-reads as a different value."""
    from polylogue.core.json import decode_provider_utf8

    assert decode_provider_utf8(b"a\xed\xa0\xbd\xed\xb8\x80 \xed\xa0\x80") == "a\U0001f600 \ud800"


def test_nfc_candidate_digest_streams_without_a_full_normalized_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: normalize the whole candidate to hash its NFC form and the
    guarded ``normalize`` sees a window larger than the digest window."""
    import unicodedata

    from polylogue.sources.parsers import codex as codex_module

    monkeypatch.setattr(codex_module, "_DIGEST_WINDOW_CHARS", 7)
    text = "é" * 40
    expected = codex_module._text_digest(unicodedata.normalize("NFC", text))
    real = unicodedata.normalize

    def bounded(form: Any, value: str) -> str:
        assert len(value) <= 16, "normalized more than one window at once"
        return real(form, value)

    from types import SimpleNamespace

    proxy = SimpleNamespace(normalize=bounded, combining=unicodedata.combining)
    monkeypatch.setattr("polylogue.sources.parsers.codex.unicodedata", proxy)
    assert codex_module._nfc_text_digest(text) == expected


def test_streamed_jsonl_decoder_reads_a_cesu8_pair_as_one_character() -> None:
    """Anti-vacuity: leave the streamed decoder on the lenient guess and the
    record is skipped or its pair split."""
    import io

    from polylogue.sources.decoders import _iter_json_stream

    line = b'{"type": "note", "text": "a\xed\xa0\xbd\xed\xb8\x80 \xed\xa0\x80"}\n'
    (record,) = list(_iter_json_stream(io.BytesIO(line), "rollout.jsonl"))
    assert record["text"] == "a\U0001f600 \ud800"  # type: ignore[index,call-overload]


def test_core_loads_still_reads_a_utf8_bom_document() -> None:
    """Anti-vacuity: hand a BOM-prefixed document to the provider decoder and
    the stdlib parse of the resulting string rejects the leading U+FEFF."""
    from polylogue.core.json import loads

    assert loads(b'\xef\xbb\xbf{"a": 1}') == {"a": 1}
    assert loads(b'{"a": "x\xed\xa0\xbd\xed\xb8\x80"}') == {"a": "x\U0001f600"}


def test_a_starter_free_run_keeps_its_nfc_key_in_bounded_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run of combining marks longer than the unsettled limit spills to disk.

    Anti-vacuity: give up the key past the limit and repeated U+0344 no longer
    matches its canonically equivalent U+0308 U+0301 spelling; carry the run
    unbounded and one normalization call sees all of it.
    """
    import unicodedata
    from types import SimpleNamespace

    from polylogue.sources.parsers import codex as codex_module

    monkeypatch.setattr(codex_module, "_DIGEST_WINDOW_CHARS", 7)
    monkeypatch.setattr(codex_module, "_NFC_UNSETTLED_LIMIT_CHARS", 28)
    real = unicodedata.normalize

    def bounded(form: Any, value: str) -> str:
        # A canonical decomposition is at most four characters per character.
        assert len(value) <= 4 * (28 + 7), "normalized an unsettled run whole"
        return real(form, value)

    proxy = SimpleNamespace(normalize=bounded, combining=unicodedata.combining)
    for text, equivalent in [
        ("\u0344" * 200, "\u0308\u0301" * 200),
        ("e" + "\u0301" + "\u0327" * 90 + "x", "\u0229" + "\u0327" * 89 + "\u0301x"),
    ]:
        expected = codex_module._text_digest(real("NFC", text))
        with monkeypatch.context() as scoped:
            scoped.setattr("polylogue.sources.parsers.codex.unicodedata", proxy)
            assert codex_module._nfc_text_digest(text) == expected
            assert codex_module._nfc_text_digest(equivalent) == expected


def test_core_loads_still_reads_bomless_utf16_and_utf32() -> None:
    """Anti-vacuity: hand valid UTF-8-decodable wide bytes to the provider decode
    and the NUL-bearing string is rejected by the stdlib parse."""
    from polylogue.core.json import loads

    for encoding in ("utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"):
        assert loads('{"a": 1}'.encode(encoding)) == {"a": 1}


def test_core_loads_reads_utf16_bytes_the_provider_decode_also_accepts() -> None:
    """Anti-vacuity: trust the provider decode once it succeeds and a BOM-less
    UTF-16LE document holding an ``ED A0 80`` triple is refused."""
    from polylogue.core.json import loads

    document = '{"a":"ꃭ\u0080"}'
    raw = document.encode("utf-16-le")
    assert b"\xed\xa0\x80" in raw
    assert loads(raw) == {"a": "ꃭ\u0080"}


def test_provider_decode_consumes_a_bom_before_a_surrogate() -> None:
    """Anti-vacuity: keep U+FEFF on the surrogatepass path and the document
    fails every later ``json.loads``."""
    from polylogue.core.json import decode_provider_utf8, loads

    raw = b'\xef\xbb\xbf{"a":"\xed\xa0\x80"}'
    assert decode_provider_utf8(raw) == '{"a":"\ud800"}'
    assert loads(raw) == {"a": "\ud800"}


def test_provider_decode_pairs_cesu8_in_one_decode() -> None:
    """Encoded pairs are combined while decoding, by the registered error
    handler, so a whale record is never held as a surrogatepass string and a
    second, combined copy.

    Anti-vacuity: leave the pair to ``surrogatepass`` and the result holds two
    code units instead of U+1F600; treat any encoded surrogate as malformed and
    the lone one raises.
    """
    from polylogue.core import json as core_json

    raw = b'{"a": "' + b"x" * 4096 + b'\xed\xa0\xbd\xed\xb8\x80 \xed\xa0\x80"}'
    assert core_json.decode_provider_utf8(raw) == '{"a": "' + "x" * 4096 + '\U0001f600 \ud800"}'


def test_js_literal_pairs_surrogates_as_consumed_across_runs() -> None:
    """Plain runs are consumed whole, and a pair formed by an escaped high half
    and the following low half is still one character.

    Anti-vacuity: stop pairing as the low half is consumed and the escaped
    ``\\uD83D`` before ``\\uDE00`` stays two code units; pair across a plain
    run and ``\\uD83D`` before ``x\\uDE00`` wrongly becomes one character.
    """
    from polylogue.sources.parsers.codex import _parse_js_literal

    value, ok = _parse_js_literal('"plain run \\uD83D\\uDE00 then \\uD83Dx\\uDE00 end"')
    assert ok and value == "plain run \U0001f600 then \ud83dx\ude00 end"
    template, ok = _parse_js_literal("`cost $5 \\uD83D\\uDE00`")
    assert ok and template == "cost $5 \U0001f600"


def test_markdown_rendering_escapes_a_lone_surrogate() -> None:
    """A session holding a lone surrogate still renders to UTF-8.

    Anti-vacuity: render the exact code point and encoding the Markdown for a
    UTF-8 destination raises ``UnicodeEncodeError``.
    """
    from polylogue.rendering.semantic_card_models import SemanticTranscript, SemanticTranscriptEntry, TranscriptProse
    from polylogue.rendering.semantic_markdown import render_semantic_transcript_markdown

    prose = TranscriptProse(message_id="m1", role="user", message_type="message", text="edit \ud800 here")
    rendered = render_semantic_transcript_markdown(
        SemanticTranscript(session_id="s1", entries=(SemanticTranscriptEntry(prose=prose),))
    )
    rendered.encode("utf-8")
    assert "\\ud800" in rendered


# Records whose parse depends on the lookahead pass (repeated code-mode call
# ids pair calls with outputs; a compaction's replacement history is resolved
# against later live text), so a replay tier that dropped, reordered, or
# re-decoded a record would change the parsed session.
_REPLAY_TIER_PAYLOAD: list[dict[str, Any]] = [
    {"type": "session_meta", "payload": {"id": "replay-tier", "timestamp": "2024-01-01T00:00:00Z"}},
    {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "functions.exec",
            "call_id": "reused",
            "arguments": 'await tools.exec_command({cmd: "first"})',
        },
    },
    {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "reused", "output": "first"}},
    {
        "type": "compacted",
        "payload": {
            "message": "summary",
            "replacement_history": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "history only"}]}
            ],
        },
    },
    {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "functions.exec",
            "call_id": "reused",
            "arguments": 'await tools.exec_command({cmd: "second"})',
        },
    },
    {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "reused", "output": "second"}},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "retained reply"}]},
    {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "retained reply"}},
    {"type": "future_record_kind"},
]


def _spy_spool(monkeypatch: pytest.MonkeyPatch) -> list[list[object]]:
    """Record every stream the parser spools to its scratch index."""
    from polylogue.sources.parsers import codex as codex_module

    spooled: list[list[object]] = []
    spool = codex_module._CodexLookaheadIndex.spool_records

    def recording(self: object, records: Any) -> None:
        captured = list(records)
        spooled.append(captured)
        spool(cast(Any, self), captured)

    monkeypatch.setattr(codex_module._CodexLookaheadIndex, "spool_records", recording)
    return spooled


@pytest.mark.parametrize("tier", ["memory", "spill-midstream", "spill-first"])
def test_stream_replay_tiers_parse_every_prefix_like_the_list_parse(monkeypatch: pytest.MonkeyPatch, tier: str) -> None:
    """Both replay tiers, and a spill after records were already retained, parse identically.

    Every prefix is an incrementally appended rollout. Anti-vacuity: drop the
    retained records on spill, or spool them after the rest of the stream,
    and the ``spill-midstream`` parse differs from the list parse; spool
    every stream regardless of size and the ``memory`` tier records a spool.
    """
    import sys

    from polylogue.sources.parsers import codex as codex_module

    retained_head = 2
    budget = {
        "memory": codex_module._CODEX_REPLAY_MEMORY_BUDGET_BYTES,
        "spill-midstream": sys.getsizeof([])
        + sum(
            codex_module._LIST_REFERENCE_BYTES + codex_module._retained_record_bytes(record)
            for record in _REPLAY_TIER_PAYLOAD[:retained_head]
        ),
        "spill-first": 0,
    }[tier]
    monkeypatch.setattr(codex_module, "_CODEX_REPLAY_MEMORY_BUDGET_BYTES", budget)
    spooled = _spy_spool(monkeypatch)

    for end in range(1, len(_REPLAY_TIER_PAYLOAD) + 1):
        prefix = _REPLAY_TIER_PAYLOAD[:end]
        spooled.clear()
        assert parse_stream(iter(prefix), "fallback") == parse(prefix, "fallback"), end
        if tier == "memory" or (tier == "spill-midstream" and end <= retained_head):
            assert spooled == []
        else:
            assert spooled == [prefix]


def test_stream_replay_budget_charges_nested_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """One record whose bulk sits deep in its payload still spills past the budget.

    Anti-vacuity: charge only the outer mapping (``sys.getsizeof(record)``)
    and the nested 64 KiB text fits a 16 KiB budget, so nothing is spooled.
    """
    from polylogue.sources.parsers import codex as codex_module

    monkeypatch.setattr(codex_module, "_CODEX_REPLAY_MEMORY_BUDGET_BYTES", 16 * 1024)
    spooled = _spy_spool(monkeypatch)
    payload = [
        {"type": "session_meta", "payload": {"id": "nested"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "n" * (64 * 1024)}],
            },
        },
    ]

    assert parse_stream(iter(payload), "fallback") == parse(payload, "fallback")
    assert spooled == [payload]


@pytest.mark.parametrize("tier", ["memory", "spill-first"])
def test_stream_parse_reads_its_records_once_then_replays_once(monkeypatch: pytest.MonkeyPatch, tier: str) -> None:
    """The lookahead is fed while the stream is retained, so the only replay is materialization.

    Anti-vacuity: run the lookahead as its own pass over the retained records
    (the predecessor shape) and the spooled tier replays twice.
    """
    from polylogue.sources.parsers import codex as codex_module

    if tier == "spill-first":
        monkeypatch.setattr(codex_module, "_CODEX_REPLAY_MEMORY_BUDGET_BYTES", 0)
    replays = 0
    original_replay = codex_module._CodexLookaheadIndex.replay_records

    def counting_replay(self: Any) -> Any:
        nonlocal replays
        replays += 1
        return original_replay(self)

    monkeypatch.setattr(codex_module._CodexLookaheadIndex, "replay_records", counting_replay)
    pulled = 0

    def source() -> Any:
        nonlocal pulled
        for record in _REPLAY_TIER_PAYLOAD:
            pulled += 1
            yield record

    assert parse_stream(source(), "fallback") == parse(_REPLAY_TIER_PAYLOAD, "fallback")
    assert pulled == len(_REPLAY_TIER_PAYLOAD)
    assert replays == (1 if tier == "spill-first" else 0)


def test_lookahead_echo_keys_are_fixed_size_digests() -> None:
    """Scratch lookup keys stay fixed-size and each mirror consumes one occurrence."""
    import sqlite3

    from polylogue.sources.parsers import codex as codex_module

    with sqlite3.connect("") as connection:
        index = codex_module._CodexLookaheadIndex(connection)
        long_text = "w" * 200_000
        evidence = (None, "2026-01-01T00:00:00+00:00", None)
        index.add_message_echo(1, ("user", long_text), evidence)
        index.add_message_echo(2, ("assistant", "short"), evidence)
        lengths = {row[0] for row in connection.execute("SELECT length(signature) FROM codex_message_echoes")}
        assert lengths == {32}
        assert not index.consume_message_echo(("assistant", long_text), evidence)
        assert not index.consume_message_echo(("user", long_text + "x"), evidence)
        assert index.consume_message_echo(("user", long_text), evidence)
        assert not index.consume_message_echo(("user", long_text), evidence)
        assert index.consume_message_echo(("assistant", "short"), evidence)
        index.close()


@pytest.mark.parametrize("record_type", ["tool_search_output", "function_call_output"])
@pytest.mark.parametrize("present", [True, False])
def test_codex_empty_tool_output_survives_parser_and_reopened_index(
    tmp_path: Path, record_type: str, present: bool
) -> None:
    """An empty provider vector remains evidence, independently of a reported verdict."""
    import sqlite3
    from contextlib import closing

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import open_connection as open_owned_connection
    from polylogue.storage.sqlite.write_lease import write_lease

    payload: dict[str, object] = {"type": record_type, "call_id": "empty-call"}
    if record_type == "tool_search_output":
        payload.update({"status": "completed", "execution": "client"})
    if present:
        payload["tools" if record_type == "tool_search_output" else "output"] = []
    records = [{"type": "response_item", "payload": payload}]
    expected = "[]" if present else None
    for label, parsed in (
        ("eager", parse(records, "empty-session")),
        ("streamed", parse_stream(iter(records), "empty-session")),
    ):
        assert parsed is not None
        results = [
            block for message in parsed.messages for block in message.blocks if block.type is BlockType.TOOL_RESULT
        ]
        assert len(results) == 1
        assert results[0].tool_id == "empty-call"
        assert results[0].text == expected
        assert results[0].is_error is None
        assert results[0].outcome_unknown_reason == "not_reported"
        index_path = tmp_path / label / "index.db"
        index_path.parent.mkdir()
        with write_lease("Codex parser persistence fixture", archive_root=index_path.parent):
            initialize_active_archive_root(index_path.parent)
            with closing(
                open_owned_connection(index_path, tier=ArchiveTier.INDEX, archive_root=index_path.parent)
            ) as index:
                index.row_factory = sqlite3.Row
                with index:
                    session_id = write_fixture_index_session(index, parsed, content_hash=session_content_hash(parsed))
        with closing(sqlite3.connect(index_path)) as index:
            stored = index.execute(
                "SELECT tool_id,text,tool_outcome FROM blocks WHERE session_id=? AND block_type='tool_result'",
                (session_id,),
            ).fetchall()
        assert stored == [("empty-call", expected, "unknown")]


@pytest.mark.parametrize(
    ("result", "expected_error", "expected_reason"),
    [
        ({"Ok": {"content": [{"type": "text", "text": "neutral output"}]}}, False, None),
        ({"Ok": {"content": [], "isError": None}}, False, None),
        ({"Ok": {"content": [], "isError": False}}, False, None),
        ({"Ok": {"content": [{"type": "text", "text": "failed"}], "isError": True}}, True, None),
        *[
            (
                {"Ok": {"content": [{"type": "text", "text": "retained"}], "isError": value}},
                None,
                "unsupported_construct",
            )
            for value in cast(tuple[object, ...], ("true", 0, 1, [], {}))
        ],
        ({"Err": "transport unavailable"}, True, None),
        ({"Ok": 0}, None, "unsupported_construct"),
        ({"Ok": {}}, None, "unsupported_construct"),
        *[
            ({"Ok": {"content": value}}, None, "unsupported_construct")
            for value in cast(tuple[object, ...], (None, "invalid", {}, False, 0))
        ],
        ({"Ok": {"content": [], "future_field": "neutral extension"}}, False, None),
    ],
)
def test_codex_mcp_application_verdict_survives_parser_and_reopened_index(
    tmp_path: Path, result: dict[str, object], expected_error: bool | None, expected_reason: str | None
) -> None:
    """Outer transport success cannot replace the provider's typed inner verdict."""
    import sqlite3
    from contextlib import closing

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import open_connection as open_owned_connection
    from polylogue.storage.sqlite.write_lease import write_lease

    records = [
        {
            "type": "event_msg",
            "payload": {
                "type": "mcp_tool_call_end",
                "call_id": "mcp-call",
                "invocation": {"server": "fixture", "tool": "count", "arguments": {}},
                "result": result,
            },
        }
    ]
    for label, parsed in (
        ("eager", parse(records, "mcp-session")),
        ("streamed", parse_stream(iter(records), "mcp-session")),
    ):
        assert parsed is not None
        use, output = parsed.messages
        assert use.blocks[0].tool_id == output.blocks[0].tool_id == "mcp-call"
        assert use.blocks[0].tool_name == "mcp__fixture__count"
        assert output.blocks[0].is_error is expected_error
        assert output.blocks[0].outcome_unknown_reason == expected_reason
        if "Err" in result:
            assert output.blocks[0].text == "transport unavailable"
        else:
            assert output.blocks[0].text is not None
            assert json.loads(output.blocks[0].text) == result["Ok"]
        index_path = tmp_path / label / "index.db"
        index_path.parent.mkdir()
        with write_lease("Codex parser persistence fixture", archive_root=index_path.parent):
            initialize_active_archive_root(index_path.parent)
            with closing(
                open_owned_connection(index_path, tier=ArchiveTier.INDEX, archive_root=index_path.parent)
            ) as index:
                index.row_factory = sqlite3.Row
                with index:
                    session_id = write_fixture_index_session(index, parsed, content_hash=session_content_hash(parsed))
        with closing(sqlite3.connect(index_path)) as index:
            rows = index.execute(
                "SELECT tool_id,text,tool_result_is_error,tool_outcome FROM blocks "
                "WHERE session_id=? AND block_type='tool_result'",
                (session_id,),
            ).fetchall()
        assert len(rows) == 1
        tool_id, text, is_error, outcome = rows[0]
        assert tool_id == "mcp-call"
        assert is_error == expected_error
        assert outcome == ("unknown" if expected_error is None else "error" if expected_error else "ok")
        assert text == output.blocks[0].text
        if "Err" not in result:
            assert json.loads(text) == result["Ok"]


@pytest.mark.parametrize(
    ("value", "expected"), [(None, None), (False, "false"), (0, "0"), ([], "[]"), ({}, "{}"), ("", "")]
)
def test_codex_structured_output_representation_distinguishes_falsy_values_from_absence(
    value: object, expected: str | None
) -> None:
    """Representation law only; scalar values do not assert MCP producer fidelity."""
    from polylogue.sources.parsers.codex import _codex_tool_output_text

    assert _codex_tool_output_text(value) == expected


# =============================================================================
# Legacy top-level response records, retained context and realtime markers
# =============================================================================

_CODEX_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "codex"


def _codex_fixture(name: str) -> list[object]:
    return [json.loads(line) for line in (_CODEX_FIXTURES / name).read_text().splitlines() if line.strip()]


def _wrap_legacy_response_records(records: list[object]) -> list[object]:
    """The same stream with each unwrapped response record inside a response_item."""
    wrapped: list[object] = []
    for record in records:
        if isinstance(record, dict) and record.get("type") in {"function_call", "function_call_output", "reasoning"}:
            wrapped.append({"type": "response_item", "payload": record})
        else:
            wrapped.append(record)
    return wrapped


def test_legacy_top_level_response_records_lower_exactly_like_wrapped_ones() -> None:
    """Unwrapping a 2025 function_call/output/reasoning record must not change the session.

    Red if the parser stops routing an unwrapped record through the
    response-item lowering (the records would surface as unknown outer
    records and the tool, reasoning and outcome evidence would differ).
    """
    legacy = _codex_fixture("legacy-response-records.jsonl")
    wrapped = _wrap_legacy_response_records(legacy)
    assert wrapped != legacy

    legacy_session = parse(legacy, "legacy-fallback")
    wrapped_session = parse(wrapped, "legacy-fallback")

    assert legacy_session.messages == wrapped_session.messages
    assert legacy_session.session_events == wrapped_session.session_events
    assert session_content_hash(legacy_session) == session_content_hash(wrapped_session)
    assert legacy_session.unit_accounting is not None
    assert not legacy_session.unit_accounting.outcomes
    assert all(event.event_type != "codex_unknown_outer_record" for event in legacy_session.session_events)

    blocks = [
        (block.type, block.tool_id, block.is_error, block.exit_code)
        for message in legacy_session.messages
        for block in message.blocks
    ]
    assert (BlockType.TOOL_USE, "call_sample_ok", None, None) in blocks
    assert (BlockType.TOOL_RESULT, "call_sample_ok", False, 0) in blocks
    assert (BlockType.TOOL_USE, "call_sample_fail", None, None) in blocks
    assert (BlockType.TOOL_RESULT, "call_sample_fail", True, 1) in blocks
    thinking = [
        block.text
        for message in legacy_session.messages
        for block in message.blocks
        if block.type is BlockType.THINKING
    ]
    assert thinking == ["Listing the directory first."]
    assert [message.role for message in legacy_session.messages] == [
        Role.USER,
        Role.ASSISTANT,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
    ]

    streamed = parse_stream(iter(legacy), "legacy-fallback")
    assert streamed.messages == legacy_session.messages
    assert streamed.session_events == legacy_session.session_events


@pytest.mark.parametrize("wrapped", [False, True], ids=["legacy", "wrapped"])
def test_legacy_response_tool_outcomes_survive_the_index_write(tmp_path: Path, wrapped: bool) -> None:
    """Both generations publish the same tool calls and outcomes through the index writer.

    Red if an unwrapped output stops carrying its exit-code evidence (the
    failed call would no longer store ``error``), or if an id-less call and
    its output collide on one message id at the write.
    """
    import sqlite3
    from contextlib import closing

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import open_connection as open_owned_connection
    from polylogue.storage.sqlite.write_lease import write_lease

    records = _codex_fixture("legacy-response-records.jsonl")
    parsed = parse(_wrap_legacy_response_records(records) if wrapped else records, "legacy-fallback")
    index_path = tmp_path / "index.db"
    with write_lease("Codex legacy response fixture", archive_root=tmp_path):
        initialize_active_archive_root(tmp_path)
        with closing(open_owned_connection(index_path, tier=ArchiveTier.INDEX, archive_root=tmp_path)) as index:
            index.row_factory = sqlite3.Row
            with index:
                session_id = write_fixture_index_session(index, parsed, content_hash=session_content_hash(parsed))
    with closing(sqlite3.connect(index_path)) as index:
        results = index.execute(
            "SELECT tool_id, tool_outcome FROM blocks WHERE session_id=? AND block_type='tool_result' ORDER BY tool_id",
            (session_id,),
        ).fetchall()
        uses = index.execute(
            "SELECT tool_id FROM blocks WHERE session_id=? AND block_type='tool_use' ORDER BY tool_id", (session_id,)
        ).fetchall()
    assert results == [("call_sample_fail", "error"), ("call_sample_ok", "ok")]
    assert uses == [("call_sample_fail",), ("call_sample_ok",)]


def test_verified_answer_and_realtime_markers_become_session_events() -> None:
    """retained_context/verified_answer keeps its question/answer pairs; realtime markers keep their ids.

    Red if either envelope stops being lowered: the records would arrive as
    ``codex_unknown_outer_record`` events and the ledger would not be clean.
    """
    session = parse(_codex_fixture("retained-context-and-realtime.jsonl"), "retained-fallback")

    assert session.unit_accounting is not None
    assert not session.unit_accounting.outcomes
    by_type = {event.event_type: event for event in session.session_events}
    assert "codex_unknown_outer_record" not in by_type

    answer = by_type["verified_answer"]
    assert answer.source_message_provider_id == "call_sample_question"
    assert answer.timestamp is not None
    assert answer.payload["call_id"] == "call_sample_question"
    assert answer.payload["turn_id"] == "turn-sample-1"
    assert answer.payload["acceptance_order"] == 1
    assert answer.payload["questions"] == [{"question": "Which colour should the banner use?", "answer": "Green"}]

    started = by_type["realtime_session_started"]
    closed = by_type["realtime_session_closed"]
    assert started.payload["realtime_session_id"] == closed.payload["realtime_session_id"] == "realtime-session-sample"
    assert started.payload["id"] == "realtime-marker-1"
    assert closed.payload["outcome"] == "ended"
    assert "outcome" not in started.payload

    # The surrounding tool call still pairs with its output.
    tool_ids = {(block.type, block.tool_id) for message in session.messages for block in message.blocks}
    assert (BlockType.TOOL_USE, "call_sample_question") in tool_ids
    assert (BlockType.TOOL_RESULT, "call_sample_question") in tool_ids


def test_undeclared_payload_type_under_a_declared_envelope_stays_visible() -> None:
    from polylogue.sources.parsers.base_models import AdmissionDisposition, AdmissionUnit, AdmissionUnknownReason

    records = _codex_fixture("retained-context-and-realtime.jsonl")
    records.insert(3, {"type": "retained_context", "payload": {"type": "future_retained_kind", "value": 1}})
    session = parse(records, "retained-fallback")

    assert session.unit_accounting is not None
    outcomes = session.unit_accounting.outcomes
    assert [(o.unit, o.ordinal, o.key, o.disposition, o.reason) for o in outcomes] == [
        (
            AdmissionUnit.OUTER_RECORD,
            3,
            "retained_context:future_retained_kind",
            AdmissionDisposition.TYPED_UNKNOWN,
            AdmissionUnknownReason.UNRECOGNIZED_TYPE,
        )
    ]
    unknown = [event for event in session.session_events if event.event_type == "codex_unknown_outer_record"]
    assert len(unknown) == 1
    assert unknown[0].payload["wire_type"] == "retained_context"


@pytest.mark.parametrize("fixture", ["legacy-response-records.jsonl", "retained-context-and-realtime.jsonl"])
def test_new_record_shapes_are_supported_session_streams(fixture: str) -> None:
    from polylogue.sources.parsers.codex import is_supported_outer_record

    records = _codex_fixture(fixture)
    assert all(is_supported_outer_record(record) for record in records)
    assert is_supported_session_stream(records)


@pytest.mark.parametrize(
    "record",
    [
        {"type": "future_outer_kind", "payload": {"type": "anything"}},
        {"type": "future_outer_kind"},
        # A legacy response type is supported only unwrapped; an envelope of
        # that name is not a shape the parser knows.
        {"type": "function_call", "payload": {"type": "function_call", "call_id": "c"}},
    ],
)
def test_unknown_outer_records_still_refuse_the_stream(record: dict[str, object]) -> None:
    from polylogue.sources.parsers.codex import is_supported_outer_record

    assert not is_supported_outer_record(record)
    for fixture in ("legacy-response-records.jsonl", "retained-context-and-realtime.jsonl"):
        records = _codex_fixture(fixture)
        records.insert(2, record)
        assert not is_supported_session_stream(records)


_TOOL_RESULT_HEADER_EXPECTATIONS: dict[str, tuple[bool | None, int | None]] = {
    # Unified-exec envelope behind a one-line ``Command:`` preamble.
    "call_command_ok": (False, 0),
    "call_command_fail": (True, 2),
    # A command rendered over several lines: the line after ``Command:`` is
    # not the envelope, so nothing is read.
    "call_command_multiline": (None, None),
    # A still-attached session has no outcome yet.
    "call_stdin_running": (None, None),
    # Freeform ``Exit code:`` header (apply_patch, shell_command).
    "call_patch_fail": (True, 1),
    "call_shell_ok": (False, 0),
    # The code-mode exec status item states no exit code; the script's
    # commands carry their own outcomes, so the transport stays unknown.
    "call_script_ok": (None, None),
    "call_script_fail": (None, None),
    # The same words not at the anchored position carry no outcome.
    "call_prose_decoy": (None, None),
    "call_script_decoy": (None, None),
}


def test_cli_written_result_headers_resolve_tool_outcomes() -> None:
    """Each CLI-written header resolves exactly its declared verdict and nothing else.

    Red if the ``Command:`` preamble or the freeform ``Exit code:`` header stops
    being read, or if either starts matching a decoy that is not the anchored,
    complete header.
    """
    session = parse(_codex_fixture("tool-result-headers.jsonl"), "headers-fallback")
    results = {
        block.tool_id: (block.is_error, block.exit_code)
        for message in session.messages
        for block in message.blocks
        if block.type is BlockType.TOOL_RESULT
    }
    assert results == _TOOL_RESULT_HEADER_EXPECTATIONS


def test_cli_written_result_headers_reach_the_written_tool_outcome() -> None:
    from polylogue.core.enums import Origin, ToolOutcome
    from polylogue.sources.tool_outcomes import derive_tool_outcomes

    session = parse(_codex_fixture("tool-result-headers.jsonl"), "headers-fallback")
    derived = derive_tool_outcomes(session.messages, session.session_events, origin=Origin.CODEX_SESSION)
    assert isinstance(derived, list)
    outcomes = {
        block.tool_id: block.tool_outcome
        for message in derived
        for block in message.blocks
        if block.type is BlockType.TOOL_RESULT
    }
    expected = {
        tool_id: ToolOutcome.UNKNOWN if is_error is None else ToolOutcome.ERROR if is_error else ToolOutcome.OK
        for tool_id, (is_error, _exit_code) in _TOOL_RESULT_HEADER_EXPECTATIONS.items()
    }
    assert outcomes == expected
