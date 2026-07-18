"""Codex event-stream contract tests.

These tests pin the on-disk semantics of the Codex JSONL "streaming envelope"
shape — what the persisted session records look like once a streaming response
has been realized. They are explicitly contract-shaped: if OpenAI changes the
envelope keys, payload type tokens, or call_id pairing semantics, these tests
should fail with a precise reason naming the broken field.

Catalog payloads live under ``tests/data/codex_event_stream/`` and cover three
realistic streaming shapes:

- ``text_only_stream.jsonl`` — session_meta + user/assistant text turn +
  trailing token_count.
- ``tool_call_stream.jsonl`` — single function_call paired with a
  function_call_output via ``call_id`` and a concluding assistant message.
- ``interleaved_stream.jsonl`` — multiple tool turns, mixed assistant
  messages, turn_context cwd changes, and a compaction event.

Ref #1296.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, MaterialOrigin
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedSession
from polylogue.sources.parsers.codex import looks_like as _looks_like_impl
from polylogue.sources.parsers.codex import parse as _parse_impl
from polylogue.sources.parsers.codex import parse_stream

CATALOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "codex_event_stream"


# ---------------------------------------------------------------------------
# Catalog access helpers
# ---------------------------------------------------------------------------


def _load_catalog(name: str) -> list[object]:
    path = CATALOG_DIR / name
    records: list[object] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _parse(payload: list[object], fallback_id: str = "fallback") -> ParsedSession:
    return _parse_impl(payload, fallback_id)


def _looks_like(payload: list[object]) -> bool:
    return _looks_like_impl(payload)


# ---------------------------------------------------------------------------
# Catalog smoke — every shape must be detected and parsed without raising
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture",
    ["text_only_stream.jsonl", "tool_call_stream.jsonl", "interleaved_stream.jsonl"],
)
def test_catalog_streams_are_detected_as_codex(fixture: str) -> None:
    records = _load_catalog(fixture)
    assert _looks_like(records), f"{fixture} must satisfy looks_like()"


@pytest.mark.parametrize(
    "fixture",
    ["text_only_stream.jsonl", "tool_call_stream.jsonl", "interleaved_stream.jsonl"],
)
def test_catalog_streams_parse_without_raising(fixture: str) -> None:
    records = _load_catalog(fixture)
    session = _parse(records, fixture)
    assert session.source_name == "codex"
    # Each fixture must contribute at least one message — otherwise the
    # streaming envelope contract is silently dropping content.
    assert session.messages, f"{fixture} produced zero messages"


# ---------------------------------------------------------------------------
# response_item envelope shape — the "streaming envelope" contract
# ---------------------------------------------------------------------------


class TestStreamingEnvelopeShape:
    """Pin the on-disk streaming envelope contract.

    The persisted Codex JSONL wraps every realized streaming event in a
    ``response_item`` envelope with a typed ``payload``. These tests pin the
    contract: the outer ``type`` token, the inner payload ``type`` tokens we
    consume, and the message-vs-event dispatch.
    """

    def test_response_item_message_payload_routes_to_messages(self) -> None:
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
            }
        ]
        session = _parse(records)
        assert len(session.messages) == 1
        assert session.messages[0].role == Role.ASSISTANT
        assert session.messages[0].material_origin is MaterialOrigin.ASSISTANT_AUTHORED
        assert session.session_events == []

    def test_response_item_non_message_payload_routes_to_events(self) -> None:
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {"type": "token_count", "input_tokens": 1, "output_tokens": 2},
            }
        ]
        session = _parse(records)
        assert session.messages == []
        assert [event.event_type for event in session.session_events] == ["token_count"]

    def test_unknown_inner_payload_type_still_recorded_as_session_event(self) -> None:
        """Unknown payload type tokens must not be silently dropped.

        This is the canary for OpenAI introducing a new event class — the
        parser must still surface it as a session event so downstream
        consumers can see something is happening.
        """
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {"type": "future_event_kind", "data": 7},
            }
        ]
        session = _parse(records)
        assert [event.event_type for event in session.session_events] == ["future_event_kind"]


# ---------------------------------------------------------------------------
# response.completed equivalent — terminal message state
# ---------------------------------------------------------------------------


class TestResponseCompletedContract:
    """The on-disk equivalent of streaming ``response.completed``.

    In a live stream OpenAI emits a sequence of ``response.delta`` events
    terminated by ``response.completed``. The Codex CLI realizes those into a
    single persisted ``response_item`` whose payload is a fully-formed
    ``message`` (the completed state). The contract this pins:

    - The realized message text equals the concatenation of all output_text
      segments — there are no partial-token artifacts left over.
    - A subsequent ``token_count`` event marks the end-of-response boundary
      and does not corrupt the prior message.
    - The terminal message timestamp drives ``session.updated_at``.
    """

    def test_completed_message_concatenates_all_output_text_segments(self) -> None:
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello "},
                        {"type": "output_text", "text": "world."},
                    ],
                },
            }
        ]
        session = _parse(records)
        assert len(session.messages) == 1
        text = session.messages[0].text or ""
        assert "Hello" in text and "world." in text

    def test_token_count_after_message_does_not_mutate_prior_message(self) -> None:
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "timestamp": "2025-01-15T10:00:08Z",
                    "content": [{"type": "output_text", "text": "Paris."}],
                },
            },
            {
                "type": "response_item",
                "payload": {"type": "token_count", "input_tokens": 5, "output_tokens": 2},
            },
        ]
        session = _parse(records)
        assert len(session.messages) == 1
        assert session.messages[0].text == "Paris."
        assert session.messages[0].timestamp == "2025-01-15T10:00:08Z"
        assert [event.event_type for event in session.session_events] == ["token_count"]

    def test_terminal_message_drives_session_updated_at(self) -> None:
        records = _load_catalog("text_only_stream.jsonl")
        session = _parse(records)
        assert session.created_at == "2025-01-15T10:00:00Z"
        # Updated_at must come from the last assistant message, not from the
        # trailing token_count event (which has no timestamp).
        assert session.updated_at == "2025-01-15T10:00:08Z"


# ---------------------------------------------------------------------------
# Tool deltas → single tool_use block, paired by call_id
# ---------------------------------------------------------------------------


class TestToolCallStreamingContract:
    """A realized function_call envelope produces a single tool_use block.

    The streaming protocol may emit multiple tool-call deltas, but the
    persisted Codex JSONL collapses them into one ``function_call``
    response_item with the merged ``arguments`` JSON string. This pins that
    contract along with ``call_id`` round-trip semantics between
    ``function_call`` and ``function_call_output``.
    """

    def test_function_call_realizes_as_single_tool_use_block(self) -> None:
        records = _load_catalog("tool_call_stream.jsonl")
        session = _parse(records, "tool_call_stream")

        tool_use_msgs = [
            msg for msg in session.messages if any(block.type == BlockType.TOOL_USE for block in msg.blocks)
        ]
        assert len(tool_use_msgs) == 1
        tool_use_block = next(block for block in tool_use_msgs[0].blocks if block.type == BlockType.TOOL_USE)
        assert tool_use_block.tool_name == "exec_command"
        assert tool_use_block.tool_id == "call_abc"
        assert tool_use_block.tool_input == {"cmd": "ls /tmp", "command": "ls /tmp"}

    def test_function_call_output_produces_paired_tool_result(self) -> None:
        records = _load_catalog("tool_call_stream.jsonl")
        session = _parse(records, "tool_call_stream")

        tool_results = [
            (msg, block) for msg in session.messages for block in msg.blocks if block.type == BlockType.TOOL_RESULT
        ]
        assert len(tool_results) == 1
        result_msg, result_block = tool_results[0]
        assert result_msg.role == Role.TOOL
        assert result_block.tool_id == "call_abc"
        assert result_block.text == "file1.txt\nfile2.txt"

    def test_call_id_pairs_tool_use_and_tool_result(self) -> None:
        """The cross-message contract: tool_use.tool_id == tool_result.tool_id."""
        records = _load_catalog("interleaved_stream.jsonl")
        session = _parse(records, "interleaved_stream")

        tool_use_ids = [
            block.tool_id for msg in session.messages for block in msg.blocks if block.type == BlockType.TOOL_USE
        ]
        tool_result_ids = [
            block.tool_id for msg in session.messages for block in msg.blocks if block.type == BlockType.TOOL_RESULT
        ]
        assert tool_use_ids == ["call_git", "call_read"]
        assert tool_result_ids == ["call_git", "call_read"]


# ---------------------------------------------------------------------------
# Truncated / out-of-order / malformed streams
# ---------------------------------------------------------------------------


class TestStreamResilience:
    """Contract for incomplete and disordered streams.

    The Codex parser is forgiving by design — it does not raise on truncation
    or reordering, and it must not silently drop content. These tests pin the
    observable behavior so a future change to "reject with typed error" would
    fail loudly and intentionally.
    """

    def test_truncated_stream_keeps_tool_use_without_paired_result(self) -> None:
        """A function_call with no following function_call_output must still
        appear in the parsed session. Silent drop would be a contract
        violation — the operator needs to see the partial turn."""
        records: list[object] = [
            {
                "type": "session_meta",
                "payload": {"id": "truncated", "timestamp": "2025-01-15T13:00:00Z"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "u1",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Run something."}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_truncated",
                    "name": "exec_command",
                    "arguments": '{"cmd": "echo hi"}',
                },
            },
            # function_call_output intentionally absent — stream truncated.
        ]
        session = _parse(records, "truncated")

        tool_use_ids = [
            block.tool_id for msg in session.messages for block in msg.blocks if block.type == BlockType.TOOL_USE
        ]
        tool_result_ids = [
            block.tool_id for msg in session.messages for block in msg.blocks if block.type == BlockType.TOOL_RESULT
        ]
        assert tool_use_ids == ["call_truncated"]
        assert tool_result_ids == []
        # The session_events surface must record the unpaired function_call
        # so downstream consumers can detect the truncation.
        assert "function_call" in {event.event_type for event in session.session_events}

    def test_out_of_order_tool_output_before_call_is_preserved(self) -> None:
        """Output arriving before its call must still be parsed — the parser
        does not enforce ordering, but it also must not drop either side."""
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_oo",
                    "output": "early-output",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "id": "fc_oo",
                    "call_id": "call_oo",
                    "name": "exec_command",
                    "arguments": '{"cmd": "noop"}',
                },
            },
        ]
        session = _parse(records, "out-of-order")

        tool_blocks = [
            (block.type, block.tool_id)
            for msg in session.messages
            for block in msg.blocks
            if block.type in (BlockType.TOOL_USE, BlockType.TOOL_RESULT)
        ]
        # Both halves of the pair must survive — order may be insertion order.
        assert (BlockType.TOOL_RESULT, "call_oo") in tool_blocks
        assert (BlockType.TOOL_USE, "call_oo") in tool_blocks

    def test_malformed_record_does_not_abort_stream(self) -> None:
        records: list[object] = [
            42,  # non-dict garbage
            {"definitely": "not-a-codex-record"},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "survivor",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "still here"}],
                },
            },
        ]
        session = _parse(records, "malformed")
        assert [msg.text for msg in session.messages] == ["still here"]

    def test_parse_stream_is_equivalent_to_parse_for_truncated_input(self) -> None:
        """Streaming consumption (iterator) and list consumption must agree
        even when the stream is truncated. This is the contract that lets the
        daemon parse mid-write JSONL files safely."""
        records = _load_catalog("tool_call_stream.jsonl")[:3]  # drop output + final msg + tokens
        from_list = _parse(records, "trunc")
        from_stream = parse_stream(iter(records), "trunc")
        assert from_list == from_stream


# ---------------------------------------------------------------------------
# Cross-message references (compaction, turn_context, parent session)
# ---------------------------------------------------------------------------


class TestCrossMessageReferences:
    """Pins the cross-message reference contract carried in the stream."""

    def test_second_session_meta_becomes_parent_continuation(self) -> None:
        records: list[object] = [
            {"type": "session_meta", "payload": {"id": "child", "timestamp": "2025-01-15T14:00:00Z"}},
            {"type": "session_meta", "payload": {"id": "parent", "timestamp": "2025-01-15T13:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "u1",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continued"}],
                },
            },
        ]
        session = _parse(records, "fallback")
        assert session.provider_session_id == "child"
        assert session.parent_session_provider_id == "parent"
        assert session.branch_type == BranchType.CONTINUATION

    def test_turn_context_cwd_changes_collected_into_working_directories(self) -> None:
        records = _load_catalog("interleaved_stream.jsonl")
        session = _parse(records, "interleaved_stream")
        assert session.working_directories == ["/repo/other", "/repo/polylogue"]

    def test_compaction_surfaced_with_replacement_history_flag(self) -> None:
        records = _load_catalog("interleaved_stream.jsonl")
        session = _parse(records, "interleaved_stream")
        compactions = [e for e in session.session_events if e.event_type == "compaction"]
        assert len(compactions) == 1
        assert compactions[0].payload["replacement_history_count"] == 1
        assert compactions[0].payload["summary"] == "Conversation compacted"


# ---------------------------------------------------------------------------
# Modern Code Mode functions.exec lowering
# ---------------------------------------------------------------------------


class TestFunctionsExecLowering:
    """Pin transport-preserving lowering of nested Code Mode operations.

    These tests exercise the production Codex parser. Removing the envelope
    pre-scan or child-block emission must collapse the asserted child actions
    back to the historical outer-only shape and fail this class.
    """

    @staticmethod
    def _blocks(session: ParsedSession, block_type: BlockType) -> list[ParsedContentBlock]:
        return [block for message in session.messages for block in message.blocks if block.type == block_type]

    @staticmethod
    def _mapping(value: object) -> dict[str, object]:
        assert isinstance(value, dict)
        return value

    def test_single_child_retains_transport_and_promotes_exact_structural_result(self) -> None:
        session = _parse(_load_catalog("functions_exec_single.jsonl"), "functions-exec-single")
        tool_uses = self._blocks(session, BlockType.TOOL_USE)
        tool_results = self._blocks(session, BlockType.TOOL_RESULT)

        assert [block.tool_name for block in tool_uses] == ["exec", "exec_command"]
        assert [block.tool_id for block in tool_uses] == [
            "exec-single",
            "exec-single::polylogue-child::0",
        ]
        child = tool_uses[1]
        assert child.tool_input is not None
        assert child.tool_input["command"] == "git status --short"
        assert child.tool_input["workdir"] == "/repo"
        assert child.tool_input["byte_count"] == 5
        provenance = child.tool_input["_polylogue"]
        assert provenance == {
            "kind": "codex.functions_exec_child",
            "registry_type": "exec_command",
            "parse_state": "parsed",
            "raw_tool_path": "tools.exec_command",
            "transport_child_index": 0,
            "transport": {
                "provider_message_id": "ctc-single",
                "tool_id": "exec-single",
                "tool_name": "exec",
                "block_position": 0,
            },
            "source_span": [21, 92],
            "structural_result_fields": {"byte_count": 5},
        }

        # The outer content-item envelope remains evidence. Only the exact JSON
        # child item gets a structured outcome; the script-status prose does not.
        assert len(tool_results) == 2
        assert tool_results[0].tool_id == "exec-single"
        assert tool_results[0].is_error is None
        assert tool_results[0].exit_code is None
        assert tool_results[1].tool_id == "exec-single::polylogue-child::0"
        assert tool_results[1].is_error is False
        assert tool_results[1].exit_code == 0
        assert tool_results[1].metadata == {
            "codex_functions_exec_child_index": 0,
            "codex_functions_exec_registry_type": "exec_command",
            "byte_count": 5,
        }

    def test_multiple_children_preserve_order_registry_paths_and_unknown_states(self) -> None:
        session = _parse(_load_catalog("functions_exec_multiple.jsonl"), "functions-exec-multiple")
        tool_uses = self._blocks(session, BlockType.TOOL_USE)
        tool_results = self._blocks(session, BlockType.TOOL_RESULT)
        children = tool_uses[1:]

        assert tool_uses[0].tool_name == "functions.exec"
        assert [block.tool_name for block in children] == [
            "exec_command",
            "apply_patch",
            "write_stdin",
            "update_plan",
            "wait",
            "web",
            "image",
            "mcp.repo_memory.search",
            "future_tool",
            "exec_command",
        ]
        provenances = [
            self._mapping(block.tool_input["_polylogue"]) for block in children if block.tool_input is not None
        ]
        assert [value["registry_type"] for value in provenances] == [
            "exec_command",
            "apply_patch",
            "write_stdin",
            "update_plan",
            "wait",
            "web",
            "image",
            "mcp",
            "unknown",
            "exec_command",
        ]
        assert [value["transport_child_index"] for value in provenances] == list(range(10))
        assert [value["parse_state"] for value in provenances] == ["parsed"] * 9 + ["malformed"]

        command_input = children[0].tool_input
        patch_input = children[1].tool_input
        unknown_input = children[8].tool_input
        malformed_input = children[-1].tool_input
        assert command_input is not None and command_input["command"] == "git status --short"
        assert patch_input is not None
        assert patch_input["paths"] == ["src/example.py", "src/renamed.py"]
        assert patch_input["path"] == "src/example.py"
        patch_command = patch_input["command"]
        assert isinstance(patch_command, str) and patch_command.startswith("*** Begin Patch")
        assert self._mapping(patch_input["_polylogue"])["structural_result_fields"] == {
            "paths": ["src/renamed.py"],
            "byte_count": 42,
        }
        assert unknown_input is not None
        assert unknown_input["raw_arguments"] == "{opaque: true}"
        assert self._mapping(unknown_input["_polylogue"])["registry_type"] == "unknown"
        assert malformed_input is not None
        assert malformed_input["raw_arguments"] == "dynamicArgs;"
        assert "command" not in malformed_input
        assert self._mapping(malformed_input["_polylogue"])["parse_state"] == "malformed"

        child_results = tool_results[1:]
        assert [block.tool_id for block in child_results] == [
            f"exec-multiple::polylogue-child::{index}" for index in range(10)
        ]
        assert [(block.is_error, block.exit_code) for block in child_results] == [
            (True, 2),
            (None, None),
            (None, None),
            (None, None),
            (None, None),
            (None, None),
            (None, None),
            (False, 0),
            (None, None),
            (None, None),
        ]
        result_1_metadata = child_results[1].metadata
        result_6_metadata = child_results[6].metadata
        assert result_1_metadata is not None
        assert result_1_metadata["paths"] == ["src/renamed.py"]
        assert result_1_metadata["byte_count"] == 42
        assert result_6_metadata is not None
        assert result_6_metadata["paths"] == ["artifacts/preview.png"]
        assert result_6_metadata["byte_count"] == 100
        assert "exit_code=7" in (child_results[8].text or "")
        assert "exit_code=9" in (child_results[9].text or "")

    def test_repeated_transport_ids_pair_children_by_occurrence_rank(self) -> None:
        records: list[object] = []
        for occurrence, exit_code in enumerate((7, 0), start=1):
            records.extend(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "custom_tool_call",
                            "id": f"call-{occurrence}",
                            "call_id": "repeated-exec",
                            "name": "exec",
                            "input": f'tools.exec_command({{cmd: "echo {occurrence}"}});',
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "custom_tool_call_output",
                            "id": f"result-{occurrence}",
                            "call_id": "repeated-exec",
                            "output": [
                                {"type": "input_text", "text": "Script completed\nOutput:\n"},
                                {
                                    "type": "input_text",
                                    "text": json.dumps({"exit_code": exit_code, "output": f"result-{occurrence}"}),
                                },
                            ],
                        },
                    },
                ]
            )

        session = _parse(records, "repeated-exec")
        child_uses = [
            block
            for block in self._blocks(session, BlockType.TOOL_USE)
            if block.tool_id == "repeated-exec::polylogue-child::0"
        ]
        child_results = [
            block
            for block in self._blocks(session, BlockType.TOOL_RESULT)
            if block.tool_id == "repeated-exec::polylogue-child::0"
        ]
        assert [block.tool_input["command"] for block in child_uses if block.tool_input is not None] == [
            "echo 1",
            "echo 2",
        ]
        assert [(block.text, block.exit_code) for block in child_results] == [
            ('{"exit_code": 7, "output": "result-1"}', 7),
            ('{"exit_code": 0, "output": "result-2"}', 0),
        ]

        # Exercise the production action-pair SQL rather than reproducing its
        # join in the test. The repeated deterministic child id must pair Nth
        # use to Nth result, never fan out into a cross product.
        import sqlite3

        from polylogue.storage.sqlite.action_pairs import refresh_action_pairs

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                position INTEGER NOT NULL,
                variant_index INTEGER NOT NULL
            );
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                block_type TEXT NOT NULL,
                tool_name TEXT,
                semantic_type TEXT,
                tool_command TEXT,
                tool_path TEXT,
                tool_input TEXT,
                tool_id TEXT,
                text TEXT,
                tool_result_is_error INTEGER,
                tool_result_exit_code INTEGER
            );
            CREATE TABLE action_pairs (
                tool_use_block_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                tool_id TEXT,
                use_rank INTEGER,
                tool_name TEXT,
                semantic_type TEXT,
                tool_command TEXT,
                tool_path TEXT,
                tool_input TEXT,
                tool_result_block_id TEXT,
                output_text TEXT,
                is_error INTEGER,
                exit_code INTEGER
            );
            """
        )
        for message in session.messages:
            message_id = message.provider_message_id
            conn.execute(
                "INSERT INTO messages(message_id, position, variant_index) VALUES (?, ?, ?)",
                (message_id, message.position, message.variant_index or 0),
            )
            for block_position, block in enumerate(message.blocks):
                tool_input = dict(block.tool_input or {})
                conn.execute(
                    """
                    INSERT INTO blocks(
                        block_id, session_id, message_id, position, block_type,
                        tool_name, tool_command, tool_path, tool_input, tool_id,
                        text, tool_result_is_error, tool_result_exit_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{message_id}:{block_position}",
                        "session",
                        message_id,
                        block_position,
                        block.type.value,
                        block.tool_name,
                        tool_input.get("command"),
                        tool_input.get("path") or tool_input.get("file_path"),
                        json.dumps(tool_input, sort_keys=True) if block.tool_input is not None else None,
                        block.tool_id,
                        block.text,
                        block.is_error,
                        block.exit_code,
                    ),
                )
        refresh_action_pairs(conn, "session")
        rows = conn.execute(
            """
            SELECT use_rank, tool_command, output_text, is_error, exit_code
            FROM action_pairs
            WHERE tool_id = 'repeated-exec::polylogue-child::0'
            ORDER BY use_rank
            """
        ).fetchall()
        assert [dict(row) for row in rows] == [
            {
                "use_rank": 1,
                "tool_command": "echo 1",
                "output_text": '{"exit_code": 7, "output": "result-1"}',
                "is_error": 1,
                "exit_code": 7,
            },
            {
                "use_rank": 2,
                "tool_command": "echo 2",
                "output_text": '{"exit_code": 0, "output": "result-2"}',
                "is_error": 0,
                "exit_code": 0,
            },
        ]

    def test_structured_content_item_result_collection_expands_in_child_order(self) -> None:
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "id": "collection-call",
                    "call_id": "collection-exec",
                    "name": "exec",
                    "input": 'tools.exec_command({cmd: "one"}); tools.exec_command({cmd: "two"});',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "id": "collection-output",
                    "call_id": "collection-exec",
                    "output": [
                        {"type": "input_text", "text": "Script completed\nOutput:\n"},
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {
                                    "results": [
                                        {"exit_code": 4, "output": "one"},
                                        {"exit_code": 0, "output": "two"},
                                    ]
                                }
                            ),
                        },
                    ],
                },
            },
        ]

        session = _parse(records, "collection-exec")
        child_results = [
            block
            for block in self._blocks(session, BlockType.TOOL_RESULT)
            if block.tool_id and "::polylogue-child::" in block.tool_id
        ]

        assert [(block.text, block.exit_code) for block in child_results] == [
            ('{"exit_code": 4, "output": "one"}', 4),
            ('{"exit_code": 0, "output": "two"}', 0),
        ]

    def test_missing_structural_child_result_keeps_use_unpaired(self) -> None:
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "id": "unpaired-call",
                    "call_id": "unpaired-exec",
                    "name": "exec",
                    "input": 'tools.exec_command({cmd: "still-running"});',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "id": "unpaired-output",
                    "call_id": "unpaired-exec",
                    "output": [{"type": "input_text", "text": "Script completed\nOutput:\n"}],
                },
            },
        ]

        session = _parse(records, "unpaired-exec")
        child_uses = [
            block
            for block in self._blocks(session, BlockType.TOOL_USE)
            if block.tool_id and "::polylogue-child::" in block.tool_id
        ]
        child_results = [
            block
            for block in self._blocks(session, BlockType.TOOL_RESULT)
            if block.tool_id and "::polylogue-child::" in block.tool_id
        ]

        assert len(child_uses) == 1
        assert child_uses[0].tool_input is not None
        assert child_uses[0].tool_input["command"] == "still-running"
        assert child_results == []

    def test_single_status_mapping_pairs_with_unknown_outcome(self) -> None:
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "id": "status-call",
                    "call_id": "status-exec",
                    "name": "exec",
                    "input": 'tools.wait({cell_id: "cell-9"});',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "id": "status-output",
                    "call_id": "status-exec",
                    "output": {"status": "completed"},
                },
            },
        ]

        session = _parse(records, "status-exec")
        child_results = [
            block
            for block in self._blocks(session, BlockType.TOOL_RESULT)
            if block.tool_id and "::polylogue-child::" in block.tool_id
        ]

        assert len(child_results) == 1
        assert child_results[0].text == '{"status": "completed"}'
        assert child_results[0].is_error is None
        assert child_results[0].exit_code is None

    def test_result_before_use_pairs_by_envelope_occurrence_without_recovery_guess(self) -> None:
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "id": "early-output",
                    "call_id": "early-exec",
                    "output": {"exit_code": 3, "output": "early"},
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "id": "late-call",
                    "call_id": "early-exec",
                    "name": "exec",
                    "input": 'tools.exec_command({cmd: "late"});',
                },
            },
        ]

        session = _parse(records, "early-exec")
        child_uses = [
            block
            for block in self._blocks(session, BlockType.TOOL_USE)
            if block.tool_id and "::polylogue-child::" in block.tool_id
        ]
        child_results = [
            block
            for block in self._blocks(session, BlockType.TOOL_RESULT)
            if block.tool_id and "::polylogue-child::" in block.tool_id
        ]

        assert len(child_uses) == 1
        assert child_uses[0].tool_input is not None
        assert child_uses[0].tool_input["command"] == "late"
        assert [(block.text, block.exit_code) for block in child_results] == [
            ('{"exit_code": 3, "output": "early"}', 3)
        ]

    def test_lowered_blocks_change_semantic_content_hash(self) -> None:
        from polylogue.pipeline.ids import session_content_hash

        lowered = _parse(_load_catalog("functions_exec_single.jsonl"), "functions-exec-hash")
        outer_only = lowered.model_copy(
            update={
                "messages": [message.model_copy(update={"blocks": message.blocks[:1]}) for message in lowered.messages]
            }
        )

        assert session_content_hash(lowered) != session_content_hash(outer_only)
