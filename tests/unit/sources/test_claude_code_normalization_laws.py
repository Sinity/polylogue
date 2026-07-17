"""Claude Code provider-family normalization laws.

The fixtures are reduced, privacy-safe JSONL wire shapes. Expected facts below
are literal provider-neutral witnesses; they do not call parser helpers to
construct the oracle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from polylogue.archive.session.branch_type import BranchType
from polylogue.config import Source
from polylogue.core.enums import BlockType, MaterialOrigin, Provider, Role
from polylogue.sources.dispatch import detect_provider, parse_payload, parse_stream_payload
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.source_parsing import iter_source_sessions_with_raw
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveSessionEnvelope,
    read_archive_session_envelope,
)

_FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "claude-code"
_FAMILY_FIXTURE = _FIXTURE_ROOT / "normalization-family.jsonl"
_AGENT_FIXTURE = _FIXTURE_ROOT / "normalization-agent.jsonl"
_PARENT_FIXTURE = _FIXTURE_ROOT / "normalization-lineage-parent.jsonl"
_ACOMPACT_FIXTURE = _FIXTURE_ROOT / "normalization-lineage-acompact.jsonl"

_MAIN_SESSION_ID = "claude-normalization-main"
_PARENT_SESSION_ID = "claude-lineage-parent"
_AGENT_FALLBACK_ID = "agent-normalization-proof"
_ACOMPACT_FALLBACK_ID = "agent-acompact-normalization-proof"
_ARCHIVE_SESSION_ID = "claude-code-session:claude-normalization-main"
_EXPECTED_ARCHIVE_MESSAGE_IDS = (
    "claude-code-session:claude-normalization-main:main-u1",
    "claude-code-session:claude-normalization-main:main-a1",
    "claude-code-session:claude-normalization-main:main-bg-start",
    "claude-code-session:claude-normalization-main:main-command",
    "claude-code-session:claude-normalization-main:main-context",
    "claude-code-session:claude-normalization-main:msg-7",
    "claude-code-session:claude-normalization-main:main-a2",
    "claude-code-session:claude-normalization-main:main-fg-result",
    "claude-code-session:claude-normalization-main:msg-11",
)
_EXPECTED_MAIN_A1_BLOCK_IDS = (
    "claude-code-session:claude-normalization-main:main-a1:0",
    "claude-code-session:claude-normalization-main:main-a1:1",
    "claude-code-session:claude-normalization-main:main-a1:2",
    "claude-code-session:claude-normalization-main:main-a1:3",
)

# Independent normalized-fact oracle. These values are authored alongside the
# wire fixture, not derived from parser output or provider helper functions.
_EXPECTED_MAIN_MESSAGES = (
    ("main-u1", "user", "message", "human_authored", "2026-07-01T10:00:00+00:00"),
    ("main-a1", "assistant", "tool_use", "assistant_authored", "2026-07-01T10:00:01+00:00"),
    ("main-bg-start", "tool", "tool_result", "tool_result", "2026-07-01T10:00:02+00:00"),
    ("main-command", "user", "protocol", "operator_command", "2026-07-01T10:00:03+00:00"),
    ("main-context", "user", "context", "runtime_context", "2026-07-01T10:00:05+00:00"),
    ("msg-7", "user", "protocol", "runtime_protocol", "2026-07-01T10:00:06+00:00"),
    ("main-a2", "assistant", "tool_use", "assistant_authored", "2026-07-01T10:00:07+00:00"),
    ("main-fg-result", "tool", "tool_result", "tool_result", "2026-07-01T10:00:08+00:00"),
    ("msg-11", "assistant", "message", "assistant_authored", "2026-07-01T10:00:10+00:00"),
)


def _records(path: Path) -> list[object]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _session_by_id(sessions: list[ParsedSession], provider_session_id: str) -> ParsedSession:
    return next(session for session in sessions if session.provider_session_id == provider_session_id)


def _message_by_id(session: ParsedSession, provider_message_id: str) -> ParsedMessage:
    return next(message for message in session.messages if message.provider_message_id == provider_message_id)


def _wire_texts(envelope: ArchiveSessionEnvelope) -> list[str]:
    messages = envelope.messages
    return ["".join(block.text or "" for block in message.blocks) for message in messages]


def test_family_fixture_detector_and_streaming_paths_preserve_one_normalized_identity() -> None:
    """Production dependencies: detector precedence, dispatch grouping, Claude parser.

    Anti-vacuity mutations: make the detector claim this as a generic/local
    artifact; reset stream record indexes per contiguous chunk; remove shared
    UUID state; or omit post-merge background reconciliation. Each mutation
    changes the literal facts or eager/stream equality asserted here.
    """
    records = _records(_FAMILY_FIXTURE)

    assert detect_provider([{"role": "user", "content": "generic transcript"}]) is None
    assert (
        detect_provider([{"type": "event_msg", "payload": {"type": "user_message", "message": "Codex row"}}])
        is Provider.CODEX
    )
    assert detect_provider(records) is Provider.CLAUDE_CODE
    eager = parse_payload(Provider.CLAUDE_CODE, records, "normalization-family")
    streamed = parse_stream_payload(Provider.CLAUDE_CODE, iter(records), "normalization-family")

    assert [session.model_dump(mode="json") for session in streamed] == [
        session.model_dump(mode="json") for session in eager
    ]
    assert [session.provider_session_id for session in eager] == [
        _MAIN_SESSION_ID,
        "claude-normalization-other",
    ]

    main = _session_by_id(eager, _MAIN_SESSION_ID)
    observed_facts = tuple(
        (
            message.provider_message_id,
            message.role.value,
            message.message_type.value,
            message.material_origin.value,
            message.timestamp,
        )
        for message in main.messages
    )
    assert observed_facts == _EXPECTED_MAIN_MESSAGES
    assert [message.position for message in main.messages] == list(range(len(_EXPECTED_MAIN_MESSAGES)))
    assert main.active_leaf_message_provider_id == "msg-11"
    assert main.branch_type is BranchType.SIDECHAIN
    assert main.title == "Review the parser."
    assert main.created_at == "2026-07-01T10:00:00+00:00"
    assert main.updated_at == "2026-07-01T10:00:10+00:00"
    assert main.reported_cost_usd == pytest.approx(0.15)
    assert main.reported_duration_ms == 1750
    assert main.models_used == ["claude-opus-4-6", "claude-sonnet-4-5-20250929"]
    assert main.working_directories == ["/workspace/polylogue"]

    assistant = _message_by_id(main, "main-a1")
    assert [block.type for block in assistant.blocks] == [
        BlockType.THINKING,
        BlockType.TEXT,
        BlockType.TOOL_USE,
        BlockType.TOOL_USE,
    ]
    assert [(block.tool_name, block.tool_id) for block in assistant.blocks if block.type is BlockType.TOOL_USE] == [
        ("Bash", "tool-bg"),
        ("Read", "tool-missing"),
    ]
    assert (
        assistant.input_tokens,
        assistant.output_tokens,
        assistant.cache_read_tokens,
        assistant.cache_write_tokens,
        assistant.model_name,
        assistant.duration_ms,
    ) == (120, 40, 900, 300, "claude-sonnet-4-5-20250929", 1500)

    background_result = _message_by_id(main, "main-bg-start").blocks[0]
    assert background_result.tool_id == "tool-bg"
    assert background_result.is_error is True
    assert background_result.exit_code == 7
    assert background_result.metadata == {
        "claude_background_task_id": "task-bg",
        "claude_background_completion_status": "failed",
        "claude_background_output_file": "/tmp/task-bg.output",
    }
    foreground_result = _message_by_id(main, "main-fg-result").blocks[0]
    assert foreground_result.tool_id == "tool-fg"
    assert foreground_result.is_error is True
    assert foreground_result.exit_code is None

    assert [(event.event_type, event.source_message_provider_id) for event in main.session_events] == [
        ("message_usage", "main-a1"),
        ("message_usage", "msg-11"),
        ("background_task_completion", "main-bg-notification"),
    ]
    assert main.session_events[-1].payload == {
        "task_id": "task-bg",
        "tool_use_id": "tool-bg",
        "output_file": "/tmp/task-bg.output",
        "status": "failed",
        "summary": 'Background command "python -m pytest -q tests/unit/sources" failed with exit code 7',
        "exit_code": 7,
    }
    # The conflicting repeated UUID appears after another session's chunk. The
    # first wire record wins exactly as it does in eager grouping.
    assert all("Conflicting duplicate" not in (message.text or "") for message in main.messages)


def test_family_fixture_survives_acquire_parse_store_read_and_action_pairing(tmp_path: Path) -> None:
    """Production route: filesystem acquisition -> streaming parse -> archive -> reads.

    Anti-vacuity mutations: drop material_origin/token/model columns in the
    writer, remove block order, pair actions only by adjacency, or discard
    unmatched tool uses. The stored/read oracle below then fails.
    """
    source_bytes = _FAMILY_FIXTURE.read_bytes()
    blob_root = tmp_path / "source-blobs"
    acquired = list(
        iter_source_sessions_with_raw(
            Source(name="claude-code", path=_FAMILY_FIXTURE),
            blob_root=blob_root,
        )
    )
    assert [session.provider_session_id for _raw, session in acquired] == [
        _MAIN_SESSION_ID,
        "claude-normalization-other",
    ]
    expected_hash = hashlib.sha256(source_bytes).hexdigest()
    assert all(
        raw is not None
        and raw.raw_bytes == b""
        and raw.blob_hash == expected_hash
        and raw.blob_size == len(source_bytes)
        and BlobStore(blob_root).read_all(expected_hash) == source_bytes
        for raw, _session in acquired
    )
    main = _session_by_id([session for _raw, session in acquired], _MAIN_SESSION_ID)

    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root) as archive:
        session_id = archive.write_parsed(main)
        envelope = read_archive_session_envelope(archive._conn, session_id)
        actions = archive.query_session_actions([session_id], limit=10)

        stored_messages = archive._conn.execute(
            """
            SELECT native_id, material_origin, model_name,
                   input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
            FROM messages
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        session_counts = archive._conn.execute(
            """
            SELECT message_count, authored_user_message_count, authored_user_word_count
            FROM sessions WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

    assert session_id == _ARCHIVE_SESSION_ID
    assert envelope.session_id == _ARCHIVE_SESSION_ID
    assert [message.message_id for message in envelope.messages] == list(_EXPECTED_ARCHIVE_MESSAGE_IDS)
    assert [message.native_id for message in envelope.messages] == [fact[0] for fact in _EXPECTED_MAIN_MESSAGES]
    assert [message.material_origin for message in envelope.messages] == [fact[3] for fact in _EXPECTED_MAIN_MESSAGES]
    assert [block.block_id for block in envelope.messages[1].blocks] == list(_EXPECTED_MAIN_A1_BLOCK_IDS)
    assert envelope.messages[2].blocks[0].block_id == ("claude-code-session:claude-normalization-main:main-bg-start:0")
    assert [block.block_type for block in envelope.messages[1].blocks] == [
        "thinking",
        "text",
        "tool_use",
        "tool_use",
    ]
    assert envelope.messages[2].blocks[0].tool_result_is_error == 1
    assert envelope.messages[2].blocks[0].tool_result_exit_code == 7
    assert envelope.messages[7].blocks[0].tool_result_is_error == 1
    assert envelope.messages[7].blocks[0].tool_result_exit_code is None

    assert tuple(session_counts) == (9, 1, 3)
    by_native_id = {row[0]: tuple(row[1:]) for row in stored_messages}
    assert by_native_id["main-a1"] == (
        "assistant_authored",
        "claude-sonnet-4-5-20250929",
        120,
        40,
        900,
        300,
    )
    assert by_native_id["msg-11"] == (
        "assistant_authored",
        "claude-opus-4-6",
        25,
        8,
        75,
        5,
    )

    by_command = {action.tool_command: action for action in actions if action.tool_command is not None}
    assert set(by_command) == {"python -m pytest -q tests/unit/sources", "false"}
    assert (
        by_command["python -m pytest -q tests/unit/sources"].is_error,
        by_command["python -m pytest -q tests/unit/sources"].exit_code,
    ) == (1, 7)
    assert by_command["python -m pytest -q tests/unit/sources"].tool_result_block_id is not None
    assert (by_command["false"].is_error, by_command["false"].exit_code) == (1, None)
    assert by_command["false"].tool_result_block_id is not None
    unmatched_read = next(action for action in actions if action.tool_name == "Read")
    assert unmatched_read.tool_path == "/workspace/polylogue/missing.py"
    assert unmatched_read.tool_result_block_id is None
    assert unmatched_read.is_error is None
    assert unmatched_read.exit_code is None


def test_agent_artifact_topology_prevents_generated_instruction_from_becoming_human() -> None:
    """Production dependency: fallback artifact identity reaches authoredness.

    Mutation: treat every origin-less ``type=user`` message as human-authored or
    remove the ``agent-*`` topology input. The first assertion becomes false and
    the child title/aggregate semantics regress with it.
    """
    parsed = parse_payload(Provider.CLAUDE_CODE, _records(_AGENT_FIXTURE), _AGENT_FALLBACK_ID)
    assert len(parsed) == 1
    child = parsed[0]

    assert child.provider_session_id == f"{_PARENT_SESSION_ID}:{_AGENT_FALLBACK_ID}"
    assert child.parent_session_provider_id == _PARENT_SESSION_ID
    assert child.branch_type is BranchType.SUBAGENT
    assert child.messages[0].role is Role.USER
    assert child.messages[0].material_origin is MaterialOrigin.GENERATED_CONTEXT_PACK
    assert child.messages[1].material_origin is MaterialOrigin.ASSISTANT_AUTHORED
    assert child.title == child.provider_session_id


@pytest.mark.parametrize("write_parent_first", [True, False])
def test_acompact_resume_replayed_prefix_is_stored_once_and_composed(
    tmp_path: Path,
    write_parent_first: bool,
) -> None:
    """Production dependencies: parser topology plus writer late-parent repair.

    Mutation: classify ``agent-acompact-*`` as a subagent, remove prefix
    signature alignment, or skip unresolved-link repair when the parent arrives.
    The branch type, physical tail, edge, or composed transcript then fails.
    """
    parent = parse_payload(Provider.CLAUDE_CODE, _records(_PARENT_FIXTURE), "lineage-parent")[0]
    child = parse_payload(Provider.CLAUDE_CODE, _records(_ACOMPACT_FIXTURE), _ACOMPACT_FALLBACK_ID)[0]
    assert child.parent_session_provider_id == _PARENT_SESSION_ID
    assert child.branch_type is BranchType.CONTINUATION

    with ArchiveStore(tmp_path / ("parent-first" if write_parent_first else "child-first")) as archive:
        if write_parent_first:
            archive.write_parsed(parent)
            child_id = archive.write_parsed(child)
        else:
            child_id = archive.write_parsed(child)
            archive.write_parsed(parent)

        positions = archive._conn.execute(
            "SELECT position FROM messages WHERE session_id = ? ORDER BY position",
            (child_id,),
        ).fetchall()
        link = archive._conn.execute(
            """
            SELECT inheritance, branch_point_message_id, resolved_dst_session_id
            FROM session_links WHERE src_session_id = ?
            """,
            (child_id,),
        ).fetchone()
        envelope = read_archive_session_envelope(archive._conn, child_id)

    assert [row[0] for row in positions] == [2]
    assert link[0] == "prefix-sharing"
    assert link[1] is not None
    assert link[2] is not None
    assert envelope.branch_type == "continuation"
    assert _wire_texts(envelope) == [
        "Plan the release.",
        "I will inspect the checks.",
        "Compressed release context.",
    ]
    assert [message.material_origin for message in envelope.messages] == [
        "human_authored",
        "assistant_authored",
        "generated_context_pack",
    ]


def test_subagent_child_arriving_before_parent_resolves_as_spawned_fresh(tmp_path: Path) -> None:
    """Production dependency: child-first topology resolution without prefix theft.

    Mutation: prepend a parent's transcript to every child or infer
    prefix-sharing from the parent link alone. The physical count, inheritance,
    and composed text assertions fail.
    """
    parent = parse_payload(Provider.CLAUDE_CODE, _records(_PARENT_FIXTURE), "lineage-parent")[0]
    child = parse_payload(Provider.CLAUDE_CODE, _records(_AGENT_FIXTURE), _AGENT_FALLBACK_ID)[0]

    with ArchiveStore(tmp_path / "subagent-child-first") as archive:
        child_id = archive.write_parsed(child)
        archive.write_parsed(parent)
        stored_count = archive._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (child_id,),
        ).fetchone()[0]
        link = archive._conn.execute(
            """
            SELECT inheritance, branch_point_message_id, resolved_dst_session_id
            FROM session_links WHERE src_session_id = ?
            """,
            (child_id,),
        ).fetchone()
        envelope = read_archive_session_envelope(archive._conn, child_id)

    assert stored_count == 2
    assert link[0] == "spawned-fresh"
    assert link[1] is None
    assert link[2] is not None
    assert envelope.branch_type == "subagent"
    assert _wire_texts(envelope) == [
        "Inspect the normalization shard and report only evidence.",
        "The shard is coherent.",
    ]
    assert envelope.messages[0].material_origin == "generated_context_pack"
