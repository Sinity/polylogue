"""Dedicated tests for the Codex JSONL parser.

Covers format detection, envelope/direct parsing, session metadata,
branch tracking, git context, and edge cases.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from polylogue.archive.message.types import MessageType
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.session_revision_membership import MembershipRevision, classify_membership_revisions
from polylogue.core.enums import BlockType, MaterialOrigin, Role
from polylogue.pipeline.ids import session_content_hash, session_revision_projection
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.codex import _tool_input_from_arguments, is_supported_session_stream, parse_stream
from polylogue.sources.parsers.codex import looks_like as _looks_like_impl
from polylogue.sources.parsers.codex import parse as _parse_impl
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection import open_connection
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
        from polylogue.sources.dispatch import parse_stream_payload, require_positive_conversational_evidence

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
            require_positive_conversational_evidence(sessions, provider="codex", source_path="/tmp/real-stream.jsonl")
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
        from polylogue.sources.dispatch import parse_stream_payload, require_positive_conversational_evidence

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
            require_positive_conversational_evidence(sessions, provider="codex", source_path="/tmp/legacy.jsonl")
            == sessions
        )

    def test_bare_session_headers_fail_parser_contract_and_materialization_gate(self) -> None:
        from polylogue.sources.dispatch import parse_stream_payload, require_positive_conversational_evidence

        payload = [{"type": "session_meta"}, {"type": "session_meta"}]

        assert not is_supported_session_stream(payload)
        sessions = parse_stream_payload("codex", iter(payload), "fallback", source_path="/tmp/bare-headers.jsonl")

        assert sessions[0].messages == []
        assert (
            require_positive_conversational_evidence(sessions, provider="codex", source_path="/tmp/bare-headers.jsonl")
            == []
        )

    def test_mixed_envelope_and_direct_stream_fails_admission_and_evidence_gate(self) -> None:
        from polylogue.sources.dispatch import parse_stream_payload, require_positive_conversational_evidence

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
            require_positive_conversational_evidence(sessions, provider="codex", source_path="/tmp/mixed.jsonl")
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
        assert message.provider_message_id.startswith("synthetic-")

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

        assert result.messages[0].provider_message_id == result.messages[1].provider_message_id
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

    def test_idless_linear_turns_resolve_parent_coordinates_in_archive(self, workspace_env: Mapping[str, Path]) -> None:
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
        assert [message.parent_message_position for message in result.messages] == [None, 0, 1]

        with open_connection(db_setup(workspace_env)) as conn:
            write_parsed_session_to_archive(conn, result, content_hash=session_content_hash(result))
            rows = conn.execute(
                "SELECT message_id, native_id, parent_message_id FROM messages ORDER BY position"
            ).fetchall()

        assert [row["native_id"] for row in rows] == [None, None, None]
        assert [row["parent_message_id"] for row in rows] == [None, rows[0]["message_id"], rows[1]["message_id"]]

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
        assert "inspect this image" in (message.text or "")
        assert "mime=image/png" in (message.text or "")
        assert "sha256_base64=" in (message.text or "")
        assert any("sha256_base64=" in (block.text or "") for block in message.blocks)


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
        assert result.active_leaf_message_provider_id == "call_1"
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
    """Fields identified by `devtools lab schema parser-diff --provider codex`
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
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "final answer", "phase": "final_answer"},
                                {"type": "output_text", "text": "draft", "phase": "draft"},
                            ],
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "again", "phase": "final_answer"}],
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
        assert event.payload["replacement_history_phase_counts"] == {"draft": 1, "final_answer": 2}
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
