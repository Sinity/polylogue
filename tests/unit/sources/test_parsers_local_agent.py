from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.archive.artifact_taxonomy import classify_artifact, classify_artifact_path
from polylogue.archive.artifact_taxonomy.models import ArtifactKind
from polylogue.archive.raw_payload import build_raw_payload_envelope
from polylogue.config import Source
from polylogue.core.enums import BlockType, MaterialOrigin, MessageType, Provider
from polylogue.core.json import JSONDocument
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import _STREAMING_FULL_INGEST_BYTES, LiveBatchProcessor
from polylogue.sources.live.batch_support import _detect_provider_from_path_sample, _parse_path_as_session_artifact
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers import antigravity, hermes_state
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.source_parsing import iter_source_sessions, iter_source_sessions_with_raw
from polylogue.sources.source_walk import _resolve_source_paths
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection import open_connection
from polylogue.storage.sqlite.schema import _ensure_schema
from tests.infra.storage_records import db_setup


def _write_hermes_state_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                user_id TEXT,
                session_key TEXT,
                chat_id TEXT,
                chat_type TEXT,
                thread_id TEXT,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                parent_session_id TEXT,
                started_at REAL,
                ended_at REAL,
                end_reason TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                cwd TEXT,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT,
                pricing_version TEXT,
                title TEXT,
                api_call_count INTEGER DEFAULT 0,
                handoff_state TEXT,
                handoff_platform TEXT,
                handoff_error TEXT,
                compression_failure_cooldown_until REAL,
                compression_failure_error TEXT,
                rewind_count INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_count INTEGER,
                finish_reason TEXT,
                reasoning TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT,
                codex_message_items TEXT,
                platform_message_id TEXT,
                observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (
                id, model, model_config, system_prompt, parent_session_id,
                started_at, ended_at, end_reason, input_tokens, output_tokens,
                cache_read_tokens, cache_write_tokens, reasoning_tokens, cwd,
                estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                pricing_version, billing_provider, billing_base_url, billing_mode,
                title, api_call_count, rewind_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "hermes-root",
                "nous-hermes-test",
                "{}",
                "be precise",
                None,
                1_775_000_000.0,
                1_775_000_100.0,
                "completed",
                10,
                20,
                3,
                4,
                5,
                "/realm/project/polylogue",
                0.002,
                0.0015,
                "estimated",
                "litellm",
                "2026-07-10",
                "openrouter",
                "https://openrouter.ai/api/v1",
                "metered",
                "Hermes parser work",
                2,
                1,
            ),
        )
        conn.execute(
            """
            INSERT INTO sessions (
                id, model, model_config, parent_session_id, started_at,
                ended_at, end_reason, title
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "hermes-child",
                "nous-hermes-test",
                json.dumps({"_delegate_from": "hermes-root"}),
                "hermes-root",
                1_775_000_101.0,
                1_775_000_150.0,
                "completed",
                "Delegate child",
            ),
        )
        conn.execute(
            "INSERT INTO messages(session_id, role, content, timestamp, token_count) VALUES (?, ?, ?, ?, ?)",
            ("hermes-root", "user", "run pytest", 1_775_000_001.0, 10),
        )
        conn.execute(
            """
            INSERT INTO messages(session_id, role, content, timestamp, observed, active, compacted)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("hermes-root", "user", "ambient terminal output", 1_775_000_001.5, 1, 1, 0),
        )
        conn.execute(
            """
            INSERT INTO messages(
                session_id, role, content, tool_calls, timestamp, token_count,
                finish_reason, reasoning_content, reasoning_details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "hermes-root",
                "assistant",
                "running",
                json.dumps([{"id": "call-1", "function": {"name": "shell", "arguments": '{"cmd":"pytest"}'}}]),
                1_775_000_002.0,
                20,
                "tool_calls",
                "need test proof",
                json.dumps({"effort": "medium"}),
            ),
        )
        conn.execute(
            """
            INSERT INTO messages(session_id, role, content, tool_call_id, tool_name, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("hermes-root", "tool", "passed", "call-1", "shell", 1_775_000_003.0),
        )
        conn.execute(
            """
            INSERT INTO messages(session_id, role, content, timestamp, observed, active, compacted)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("hermes-root", "assistant", "rewound answer", 1_775_000_004.0, 0, 0, 0),
        )
        conn.execute(
            """
            INSERT INTO messages(session_id, role, content, timestamp, observed, active, compacted)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("hermes-root", "assistant", "compacted answer", 1_775_000_005.0, 0, 0, 1),
        )
        conn.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            ("hermes-child", "assistant", "delegated work", 1_775_000_102.0),
        )


def test_gemini_cli_session_document_parses_through_dispatch() -> None:
    payload: JSONDocument = {
        "sessionId": "gemini-session-1",
        "projectHash": "project-hash",
        "startTime": "2026-04-08T20:45:00.000Z",
        "lastUpdated": "2026-04-08T20:47:00.000Z",
        "kind": "chat",
        "summary": "Parser work",
        "messages": [
            {"id": "u1", "timestamp": "2026-04-08T20:45:01.000Z", "type": "user", "content": ["hello"]},
            {
                "id": "a1",
                "timestamp": "2026-04-08T20:45:02.000Z",
                "type": "gemini",
                "content": "response",
                "model": "gemini-test",
                "durationMs": 900,
                "tokens": {"total": 10},
                "thoughts": [{"text": "reasoned"}],
                "toolCalls": [{"id": "tool-1", "name": "read_file", "arguments": {"path": "README.md"}}],
            },
        ],
    }

    assert detect_provider(payload) is Provider.GEMINI_CLI
    classification = classify_artifact(payload, provider=Provider.GEMINI_CLI, source_path="chats/session.json")
    assert classification.parse_as_session is True

    [session] = parse_payload("gemini-cli", payload, "fallback")

    assert session.source_name is Provider.GEMINI_CLI
    assert session.provider_session_id == "gemini-session-1:chat:2026-04-08T20:45:00.000Z"
    assert session.created_at == "2026-04-08T20:45:00.000Z"
    assert session.updated_at == "2026-04-08T20:47:00.000Z"
    assert session.title == "Parser work"
    assert session.messages[1].role == "assistant"
    assert [message.position for message in session.messages] == [0, 1]
    assert [message.is_active_path for message in session.messages] == [True, True]
    assert [message.is_active_leaf for message in session.messages] == [False, True]
    assert session.active_leaf_message_provider_id == "a1"
    assert session.messages[1].model_name == "gemini-test"
    assert session.messages[1].duration_ms == 900
    assert session.messages[1].input_tokens == 0
    assert session.messages[1].output_tokens == 10
    assert {block.type for block in session.messages[1].blocks} >= {
        BlockType.TEXT,
        BlockType.THINKING,
        BlockType.TOOL_USE,
    }


def test_gemini_cli_display_content_is_preserved_alongside_expanded_content() -> None:
    """polylogue-2ow9p: the prompt as typed survives @-reference expansion.

    ``content`` is the model-facing payload with every ``@path`` reference
    replaced by the referenced file's text; ``displayContent`` is what the
    user actually wrote. Both are lists of ``{"text": ...}`` parts, and in
    every observed case they differ. Anti-vacuity: delete the
    ``displayContent`` branch and the message keeps only the expansion --
    the user's own sentence is gone from the archive.
    """
    payload: JSONDocument = {
        "sessionId": "gemini-session-3",
        "projectHash": "project-hash",
        "kind": "chat",
        "messages": [
            {
                "id": "u1",
                "timestamp": "2026-04-08T20:45:01.000Z",
                "type": "user",
                "content": [
                    {"text": "@notes/ summarize this"},
                    {"text": "\n--- Content from referenced files ---"},
                    {"text": "\nContent from @notes/a.md:\nexpanded body"},
                ],
                "displayContent": [{"text": "@notes/ summarize this"}],
            },
        ],
    }

    [session] = parse_payload("gemini-cli", payload, "fallback")

    message = session.messages[0]
    assert "expanded body" in (message.text or "")
    display_blocks = [block for block in message.blocks if (block.metadata or {}).get("gemini_display_content")]
    assert [block.text for block in display_blocks] == ["@notes/ summarize this"]
    # The typed prompt is stored once. A second admission path for the same
    # field doubles the operator's own words in every word count derived from
    # this message.
    assert [block.text for block in message.blocks].count("@notes/ summarize this") == 1


def test_gemini_cli_display_content_identical_to_content_adds_no_block() -> None:
    payload: JSONDocument = {
        "sessionId": "gemini-session-4",
        "projectHash": "project-hash",
        "kind": "chat",
        "messages": [
            {
                "id": "u1",
                "timestamp": "2026-04-08T20:45:01.000Z",
                "type": "user",
                "content": [{"text": "no references here"}],
                "displayContent": [{"text": "no references here"}],
            },
        ],
    }

    [session] = parse_payload("gemini-cli", payload, "fallback")

    assert all(not (block.metadata or {}).get("gemini_display_content") for block in session.messages[0].blocks)


def test_gemini_cli_contentless_turn_keeps_its_token_counts() -> None:
    """polylogue-auy4z: a turn with no content is still a billed turn.

    Gemini CLI writes ``content: ""`` with an empty ``thoughts`` list and a
    populated ``tokens`` block for turns that produced no text; the checkpoint
    file is the only place those counts exist. Anti-vacuity: restore the
    content-only drop condition in ``_parse_gemini_message`` and the message,
    its ``message_usage`` event and its share of the session's tokens all
    disappear.
    """
    payload: JSONDocument = {
        "sessionId": "gemini-session-5",
        "projectHash": "project-hash",
        "kind": "chat",
        "messages": [
            {
                "id": "u1",
                "timestamp": "2026-04-08T20:45:01.000Z",
                "type": "user",
                "content": "run it",
                "model": "gemini-test",
                "tokens": {"input": 10, "output": 0, "cached": 0, "thoughts": 0, "tool": 0, "total": 10},
            },
            {
                "id": "g1",
                "timestamp": "2026-04-08T20:45:09.000Z",
                "type": "gemini",
                "content": "",
                "thoughts": [],
                "model": "gemini-test",
                "tokens": {"input": 19029, "output": 782, "cached": 0, "thoughts": 0, "tool": 0, "total": 19811},
            },
        ],
    }

    [session] = parse_payload("gemini-cli", payload, "fallback")

    contentless = session.messages[1]
    assert contentless.provider_message_id == "g1"
    assert contentless.role == "assistant"
    assert contentless.blocks == []
    assert (contentless.input_tokens, contentless.output_tokens) == (19029, 782)
    usage_events = [event for event in session.session_events if event.event_type == "message_usage"]
    assert [event.source_message_provider_id for event in usage_events] == ["u1", "g1"]
    assert usage_events[1].payload["last_token_usage"] == {
        "input_tokens": 19029,
        "output_tokens": 782,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 19811,
    }


def test_gemini_cli_contentless_turn_tokens_reach_the_cost_rollup(workspace_env: Mapping[str, Path]) -> None:
    """The counts a contentless turn carries are billed tokens, so they must
    reach ``session_model_usage`` -- the rollup the cost model reads. A
    ``message_usage`` session event alone does not: that rollup walks the
    ``messages`` rows and the provider-usage fold reads ``token_count`` events
    only. Anti-vacuity: drop the turn again and the rollup reports 10 input
    tokens for this session instead of 19,039.
    """
    payload: JSONDocument = {
        "sessionId": "gemini-session-7",
        "projectHash": "project-hash",
        "kind": "chat",
        "messages": [
            {
                "id": "u1",
                "timestamp": "2026-04-08T20:45:01.000Z",
                "type": "user",
                "content": "run it",
                "model": "gemini-test",
                "tokens": {"input": 10, "output": 0, "cached": 0, "thoughts": 0, "tool": 0, "total": 10},
            },
            {
                "id": "g1",
                "timestamp": "2026-04-08T20:45:09.000Z",
                "type": "gemini",
                "content": "",
                "thoughts": [],
                "model": "gemini-test",
                "tokens": {"input": 19029, "output": 782, "cached": 0, "thoughts": 0, "tool": 0, "total": 19811},
            },
        ],
    }

    [session] = parse_payload("gemini-cli", payload, "fallback")

    with open_connection(db_setup(workspace_env)) as conn:
        write_parsed_session_to_archive(conn, session)
        rollup = conn.execute("SELECT model_name, input_tokens, output_tokens FROM session_model_usage").fetchall()
        usage_events = conn.execute(
            "SELECT source_message_id, last_input_tokens, last_output_tokens"
            " FROM session_provider_usage_events ORDER BY position"
        ).fetchall()

    assert [tuple(row) for row in rollup] == [("gemini-test", 19039, 782)]
    assert [row["source_message_id"] for row in usage_events] == [
        f"gemini-cli-session:{session.provider_session_id}:n:u1",
        f"gemini-cli-session:{session.provider_session_id}:n:g1",
    ]
    assert [row["last_input_tokens"] for row in usage_events] == [10, 19029]


def test_gemini_cli_empty_turn_without_tokens_is_still_dropped() -> None:
    """Only token evidence earns a blockless message.

    Anti-vacuity: widen the retention condition to every contentless record
    and this session gains an empty message with nothing to account for.
    """
    payload: JSONDocument = {
        "sessionId": "gemini-session-6",
        "projectHash": "project-hash",
        "kind": "chat",
        "messages": [
            {"id": "u1", "timestamp": "2026-04-08T20:45:01.000Z", "type": "user", "content": "run it"},
            {"id": "g1", "timestamp": "2026-04-08T20:45:09.000Z", "type": "gemini", "content": "", "thoughts": []},
            {
                "id": "g2",
                "timestamp": "2026-04-08T20:45:10.000Z",
                "type": "gemini",
                "content": "",
                "tokens": {"input": 0, "output": 0, "cached": 0, "thoughts": 0, "tool": 0, "total": 0},
            },
        ],
    }

    [session] = parse_payload("gemini-cli", payload, "fallback")

    assert [message.provider_message_id for message in session.messages] == ["u1"]


def test_gemini_cli_session_metadata_and_scratchpad_survive_as_session_events() -> None:
    """polylogue-5o05: userMessageCount/hasUserOrAssistantMessage/memoryScratchpad
    were parsed by nothing; they must now surface as session_events."""
    payload: JSONDocument = {
        "sessionId": "gemini-session-2",
        "startTime": "2026-04-08T20:45:00.000Z",
        "lastUpdated": "2026-04-08T20:47:00.000Z",
        "kind": "subagent",
        "hasUserOrAssistantMessage": True,
        "userMessageCount": 10,
        "memoryScratchpad": {
            "version": 1,
            "workflowSummary": "ran the test suite and fixed a regression",
            "toolSequence": ["read_file", "run_shell_command"],
            "touchedPaths": ["a.py", "b.py"],
            "validationStatus": "passed",
        },
        "messages": [
            {"id": "u1", "timestamp": "2026-04-08T20:45:01.000Z", "type": "user", "content": ["hello"]},
        ],
    }

    [session] = parse_payload("gemini-cli", payload, "fallback")

    metadata_event = next(
        event for event in session.session_events if event.event_type == "gemini_cli_session_metadata"
    )
    assert metadata_event.payload == {
        "parsed_message_count": 1,
        "has_user_or_assistant_message": True,
        "reported_user_message_count": 10,
    }
    scratchpad_event = next(
        event for event in session.session_events if event.event_type == "gemini_cli_memory_scratchpad"
    )
    assert scratchpad_event.payload["memory_scratchpad"] == payload["memoryScratchpad"]


@pytest.mark.parametrize(
    ("source_name", "payload"),
    [
        (
            "gemini-cli",
            {
                "sessionId": "idless-gemini",
                "kind": "chat",
                "messages": [
                    {"type": "user", "content": ["first"]},
                    {"type": "gemini", "content": "second"},
                ],
            },
        ),
        (
            "hermes",
            {
                "session_id": "idless-hermes",
                "platform": "linux",
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                ],
            },
        ),
    ],
)
def test_local_agent_idless_messages_have_exactly_one_active_leaf(
    source_name: str,
    payload: JSONDocument,
) -> None:
    [session] = parse_payload(source_name, payload, "fallback")

    assert [message.provider_message_id for message in session.messages] == ["", ""]
    assert [message.is_active_leaf for message in session.messages] == [False, True]


def test_hermes_session_document_parses_through_dispatch() -> None:
    payload: JSONDocument = {
        "session_id": "hermes-session-1",
        "model": "local-model",
        "base_url": "http://localhost",
        "platform": "linux",
        "session_start": "2026-05-07T08:39:43.000000",
        "last_updated": "2026-05-07T08:46:00.000000",
        "system_prompt": "be concise",
        "tools": [{"name": "shell"}],
        "message_count": 3,
        "messages": [
            {"role": "user", "content": "run checks"},
            {
                "role": "assistant",
                "content": "running",
                "model": "local-override",
                "durationMs": 1250,
                "usage": {"input_tokens": 4, "output_tokens": 6},
                "reasoning_content": "need tests",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "run_shell_command", "arguments": '{"cmd":"pytest"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "passed"},
        ],
    }

    assert detect_provider(payload) is Provider.HERMES
    classification = classify_artifact(payload, provider=Provider.HERMES, source_path="session_1.json")
    assert classification.parse_as_session is True

    [session] = parse_payload("hermes", payload, "fallback")

    assert session.source_name is Provider.HERMES
    assert session.provider_session_id == "hermes-session-1"
    assert session.created_at == "2026-05-07T08:39:43.000000"
    assert session.updated_at == "2026-05-07T08:46:00.000000"
    assert session.messages[0].role == "system"
    assert [message.position for message in session.messages] == [0, 1, 2, 3]
    assert session.messages[0].model_name == "local-model"
    assert session.messages[2].role == "assistant"
    assert session.messages[2].model_name == "local-override"
    assert session.messages[2].duration_ms == 1250
    assert session.messages[2].input_tokens == 4
    assert session.messages[2].output_tokens == 6
    assert {block.type for block in session.messages[2].blocks} >= {
        BlockType.TEXT,
        BlockType.THINKING,
        BlockType.TOOL_USE,
    }
    assert session.messages[3].role == "tool"
    assert session.messages[3].is_active_leaf is True
    assert session.active_leaf_message_provider_id == "call-1"
    assert [message.parent_message_provider_id for message in session.messages] == [None, None, None, None]
    assert [message.parent_message_position for message in session.messages] == [None, None, None, None]
    assert any(block.type is BlockType.TOOL_RESULT for block in session.messages[3].blocks)

    # polylogue-5o05: base_url/platform/message_count/tools were parsed by
    # nothing; they must now surface as session_events.
    metadata_event = next(event for event in session.session_events if event.event_type == "hermes_session_metadata")
    assert metadata_event.payload == {
        "parsed_message_count": 4,
        "base_url": "http://localhost",
        "platform": "linux",
        "reported_message_count": 3,
    }
    tool_event = next(event for event in session.session_events if event.event_type == "hermes_tool_availability")
    assert tool_event.payload == {"tools": [{"name": "shell"}], "tool_count": 1}


def test_hermes_message_wire_extras_survive_as_session_events() -> None:
    """polylogue-5o05: codex_reasoning_items/codex_message_items (~59% of
    documents), the low-volume _empty_recovery_synthetic/_db_persisted
    markers, and tool_calls[].extra_content were parsed by nothing in the
    JSON-snapshot shape (unlike hermes_state.py's SQLite path, which already
    captures the reasoning-item equivalent)."""
    payload: JSONDocument = {
        "session_id": "hermes-session-2",
        "model": "local-model",
        "session_start": "2026-05-07T08:39:43.000000",
        "last_updated": "2026-05-07T08:46:00.000000",
        "messages": [
            {"role": "user", "content": "run checks"},
            {
                "role": "assistant",
                "content": "running",
                "timestamp": "2026-05-07T08:40:00.000000",
                "reasoning_content": "need tests",
                "codex_reasoning_items": [{"id": "r1", "encrypted_content": "abc123"}],
                "codex_message_items": [{"phase": "final"}],
                "_empty_recovery_synthetic": True,
                "_db_persisted": False,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "run_shell_command", "arguments": '{"cmd":"pytest"}'},
                        "extra_content": {"google": {"thought_signature": "sig-abc"}},
                    }
                ],
            },
        ],
    }

    [session] = parse_payload("hermes", payload, "fallback")

    extras_event = next(event for event in session.session_events if event.event_type == "hermes_message_wire_extras")
    assistant_message = session.messages[1]
    assert extras_event.source_message_provider_id == assistant_message.provider_message_id
    assert extras_event.payload == {
        "codex_reasoning_items": [{"id": "r1", "encrypted_content": "abc123"}],
        "codex_message_items": [{"phase": "final"}],
        "_empty_recovery_synthetic": True,
        "_db_persisted": False,
        "tool_calls_extra_content": [
            {"tool_id": "call-1", "extra_content": {"google": {"thought_signature": "sig-abc"}}}
        ],
    }


def test_hermes_snapshot_codex_message_items_prose_reaches_a_block() -> None:
    """The JSON-snapshot path projects the same field as the state.db path.

    Goes red if ``_parse_hermes_message`` stops projecting
    ``codex_message_items``: a turn whose ``content`` is empty carries no
    block at all and is dropped outright, so neither the block-derived
    display text nor FTS reaches its prose. The wire-extras event keeps the
    full structured item either way.
    """
    codex_only = "The wrapper stayed untouched while the stale config migrated."
    already_carried = "Verification passed on both affected suites."

    def item(prose: str) -> JSONDocument:
        return {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": prose}],
            "id": "msg_0887",
            "phase": "commentary",
        }

    payload: JSONDocument = {
        "session_id": "hermes-session-3",
        "model": "local-model",
        "session_start": "2026-05-07T08:39:43.000000",
        "last_updated": "2026-05-07T08:46:00.000000",
        "messages": [
            {"role": "user", "content": "migrate the config"},
            {"role": "assistant", "content": "", "codex_message_items": [item(codex_only)]},
            {"role": "assistant", "content": already_carried, "codex_message_items": [item(already_carried)]},
        ],
    }

    [session] = parse_payload("hermes", payload, "fallback")

    projected, duplicated = session.messages[1], session.messages[2]
    assert [block.text for block in projected.blocks if block.type is BlockType.TEXT] == [codex_only]
    assert projected.text == codex_only
    # The item repeats what ``content`` already holds: one block, not two.
    assert [block.text for block in duplicated.blocks if block.type is BlockType.TEXT] == [already_carried]

    extras = [event for event in session.session_events if event.event_type == "hermes_message_wire_extras"]
    assert len(extras) == 2


def test_hermes_state_db_parses_authoritative_sessions(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)

    sessions = hermes_state.parse_state_db(db_path, fallback_id="fallback")

    assert [session.provider_session_id.split("@", 1)[0] for session in sessions] == ["hermes-root", "hermes-child"]
    assert all("@profile-" in session.provider_session_id for session in sessions)
    root = sessions[0]
    assert root.source_name is Provider.HERMES
    assert root.title == "Hermes parser work"
    assert root.instructions_text == "be precise"
    assert root.working_directories == ["/realm/project/polylogue"]
    assert root.git_branch is None
    assert root.git_repository_url is None
    assert root.reported_cost_usd == 0.0015
    assert root.messages[0].role == "system"
    assert root.messages[3].provider_message_id == "hermes-root:message:3"
    assert root.messages[3].output_tokens == 20
    assert {block.type for block in root.messages[3].blocks} >= {
        BlockType.TEXT,
        BlockType.THINKING,
        BlockType.TOOL_USE,
    }
    assert any(block.type is BlockType.TOOL_RESULT and block.tool_id == "call-1" for block in root.messages[4].blocks)
    assert root.messages[2].material_origin is MaterialOrigin.RUNTIME_CONTEXT
    assert root.messages[2].is_active_path is True
    assert root.messages[5].is_active_path is False
    assert root.messages[6].is_active_path is False
    assert root.active_leaf_message_provider_id == root.messages[4].provider_message_id
    assert root.messages[4].is_active_leaf is True
    assert root.messages[6].is_active_leaf is False
    assert all(message.parent_message_provider_id is None for message in root.messages)
    assert all(message.parent_message_position is None for message in root.messages)
    usage_events = [event for event in root.session_events if event.event_type == "token_count"]
    assert usage_events
    assert usage_events[0].payload["total_token_usage"] == {
        "input_tokens": 10,
        "output_tokens": 20,
        "cached_input_tokens": 3,
        "cache_write_tokens": 4,
        "reasoning_output_tokens": 5,
        "total_tokens": 42,
    }
    assert {
        key: usage_events[0].payload[key]
        for key in (
            "estimated_cost_usd",
            "actual_cost_usd",
            "cost_status",
            "cost_source",
            "pricing_version",
            "billing_provider",
            "billing_base_url",
            "billing_mode",
        )
    } == {
        "estimated_cost_usd": 0.002,
        "actual_cost_usd": 0.0015,
        "cost_status": "estimated",
        "cost_source": "litellm",
        "pricing_version": "2026-07-10",
        "billing_provider": "openrouter",
        "billing_base_url": "https://openrouter.ai/api/v1",
        "billing_mode": "metered",
    }
    state_events = [event for event in root.session_events if event.event_type == "hermes_message_state"]
    assert [event.payload["state"] for event in state_events] == [
        "active",
        "observed",
        "active",
        "active",
        "rewound",
        "compacted",
    ]
    identity = next(event for event in root.session_events if event.event_type == "hermes_identity")
    assert identity.payload["raw_session_id"] == "hermes-root"
    assert identity.payload["schema_version"] == 16
    session_capabilities = identity.payload["session_capabilities"]
    assert isinstance(session_capabilities, list)
    assert "repository" not in session_capabilities
    assert "cost_provenance" in session_capabilities

    child = sessions[1]
    assert child.parent_session_provider_id == root.provider_session_id
    assert child.branch_type is not None
    assert child.branch_type.value == "subagent"


def test_hermes_state_db_duplicate_platform_ids_keep_one_active_leaf(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE messages SET platform_message_id = ? WHERE id IN (?, ?)",
            ("duplicate-platform-id", 1, 4),
        )

    [root, _child] = hermes_state.parse_state_db(db_path, fallback_id="fallback")

    leaves = [message for message in root.messages if message.is_active_leaf]
    assert [message.provider_message_id for message in leaves] == ["duplicate-platform-id"]
    assert leaves == [root.messages[4]]
    assert root.active_leaf_message_provider_id == "duplicate-platform-id"


def test_hermes_state_db_retains_empty_rows_and_their_state(tmp_path: Path) -> None:
    from polylogue.pipeline.ids import session_content_hash

    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)
    baseline = hermes_state.parse_state_db(db_path)[1]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO messages(session_id, role, content, timestamp, observed, active, compacted)
            VALUES ('hermes-child', ?, ?, ?, ?, ?, ?)
            """,
            [
                ("assistant", "", 1_775_000_103.0, 0, 1, 0),
                ("user", None, 1_775_000_104.0, 1, 1, 0),
                ("assistant", "", 1_775_000_105.0, 0, 0, 0),
                ("assistant", None, 1_775_000_106.0, 0, 0, 1),
            ],
        )

    child = hermes_state.parse_state_db(db_path)[1]

    assert session_content_hash(child) != session_content_hash(baseline)
    assert len(child.messages) == 5
    empty_messages = child.messages[1:]
    assert all(message.text is None for message in empty_messages)
    assert all(message.blocks == [] for message in empty_messages)
    assert [message.material_origin for message in empty_messages] == [
        MaterialOrigin.ASSISTANT_AUTHORED,
        MaterialOrigin.RUNTIME_CONTEXT,
        MaterialOrigin.ASSISTANT_AUTHORED,
        MaterialOrigin.ASSISTANT_AUTHORED,
    ]
    assert [message.is_active_path for message in empty_messages] == [True, True, False, False]
    assert child.active_leaf_message_provider_id == empty_messages[1].provider_message_id
    assert empty_messages[1].is_active_leaf is True
    state_events = [event for event in child.session_events if event.event_type == "hermes_message_state"]
    assert [event.payload["state"] for event in state_events] == [
        "active",
        "active",
        "observed",
        "rewound",
        "compacted",
    ]

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE messages SET observed = 1 WHERE session_id = 'hermes-child' AND timestamp = ?",
            (1_775_000_103.0,),
        )
    state_changed = hermes_state.parse_state_db(db_path)[1]
    assert session_content_hash(state_changed) != session_content_hash(child)


def test_hermes_state_db_later_repository_capability_is_optional(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE sessions ADD COLUMN git_branch TEXT")
        conn.execute("ALTER TABLE sessions ADD COLUMN git_repo_root TEXT")
        conn.execute("UPDATE schema_version SET version = 17")
        conn.execute(
            "UPDATE sessions SET git_branch = ?, git_repo_root = ? WHERE id = ?",
            ("feature/hermes", "/realm/project/polylogue", "hermes-root"),
        )

    assert hermes_state.looks_like_state_db_path(db_path) is True
    root = hermes_state.parse_state_db(db_path)[0]

    assert root.git_branch == "feature/hermes"
    assert root.git_repository_url == "/realm/project/polylogue"
    identity = next(event for event in root.session_events if event.event_type == "hermes_identity")
    assert identity.payload["schema_version"] == 17
    session_capabilities = identity.payload["session_capabilities"]
    assert isinstance(session_capabilities, list)
    assert "repository" in session_capabilities


def test_hermes_branch_keeps_its_physical_prefix_without_compression_hydration(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                id, source, model, model_config, parent_session_id, started_at, title
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "hermes-branch",
                "cli",
                "nous-hermes-test",
                json.dumps({"_branched_from": "hermes-root"}),
                "hermes-root",
                1_775_000_151.0,
                "Branch child",
            ),
        )
        conn.executemany(
            """
            INSERT INTO messages(session_id, role, content, timestamp, active, compacted)
            VALUES ('hermes-branch', ?, ?, ?, 1, 0)
            """,
            [
                ("user", "copied prefix", 1_775_000_151.0),
                ("assistant", "branch diverges", 1_775_000_152.0),
            ],
        )

    sessions = hermes_state.parse_state_db(db_path)
    branch = next(session for session in sessions if session.provider_session_id.startswith("hermes-branch@"))

    assert branch.branch_type is not None
    assert branch.branch_type.value == "fork"
    assert [message.text for message in branch.messages] == ["copied prefix", "branch diverges"]


def test_hermes_state_db_profile_qualifies_identity_and_retains_raw_id(tmp_path: Path) -> None:
    first_path = tmp_path / "profile-a" / "state.db"
    second_path = tmp_path / "profile-b" / "state.db"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    _write_hermes_state_db(first_path)
    _write_hermes_state_db(second_path)
    retained_path = tmp_path / "blob-store" / "retained.db"
    retained_path.parent.mkdir()
    with sqlite3.connect(first_path) as source, sqlite3.connect(retained_path) as retained_conn:
        source.backup(retained_conn)

    first = hermes_state.parse_state_db(first_path)[0]
    second = hermes_state.parse_state_db(second_path)[0]
    retained = hermes_state.parse_state_db_payload(
        hermes_state.marker_payload(retained_path, profile_root=first_path.parent),
        fallback_id="unused",
    )[0]

    assert first.provider_session_id != second.provider_session_id
    assert retained.provider_session_id == first.provider_session_id
    assert first.provider_session_id.startswith("hermes-root@profile-")
    assert second.provider_session_id.startswith("hermes-root@profile-")
    for session in (first, second):
        identity = next(event for event in session.session_events if event.event_type == "hermes_identity")
        assert identity.payload["raw_session_id"] == "hermes-root"
        assert ParsedSession.model_validate(session.model_dump()).session_events == session.session_events


def test_hermes_state_db_rejects_tables_missing_required_core(tmp_path: Path) -> None:
    db_path = tmp_path / "not-hermes.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions(id TEXT PRIMARY KEY, started_at REAL);
            CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL);
            """
        )

    assert hermes_state.looks_like_state_db_path(db_path) is False
    with pytest.raises(ValueError, match="not a Hermes state.db"):
        hermes_state.parse_state_db(db_path)


def test_hermes_state_db_rejects_versioned_chat_database_lookalike(tmp_path: Path) -> None:
    db_path = tmp_path / "chat-app.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at REAL,
                source TEXT,
                model_config TEXT,
                parent_session_id TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp REAL,
                tool_calls TEXT
            );
            """
        )

    assert hermes_state.looks_like_state_db_path(db_path) is False
    with pytest.raises(ValueError, match="not a Hermes state.db"):
        hermes_state.parse_state_db(db_path)


def test_hermes_state_db_contract_matches_parser_capability_map() -> None:
    contract_path = (
        Path(__file__).parents[3] / "polylogue" / "schemas" / "providers" / "hermes" / "state_db_v16.contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert set(contract["tables"]["sessions"]["required_core"]) == hermes_state._REQUIRED_SESSION_COLUMNS
    assert set(contract["tables"]["messages"]["required_core"]) == hermes_state._REQUIRED_MESSAGE_COLUMNS
    assert set(contract["detection_signature"]["session_columns"]) == hermes_state._HERMES_SIGNATURE_SESSION_COLUMNS
    assert set(contract["detection_signature"]["message_columns"]) == hermes_state._HERMES_SIGNATURE_MESSAGE_COLUMNS
    assert {
        name: frozenset(fields) for name, fields in contract["tables"]["sessions"]["optional_capabilities"].items()
    } == hermes_state._SESSION_CAPABILITIES
    assert {
        name: frozenset(fields) for name, fields in contract["tables"]["messages"]["optional_capabilities"].items()
    } == hermes_state._MESSAGE_CAPABILITIES


def test_hermes_state_db_dispatch_marker_parses_multiple_sessions(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)
    payload = hermes_state.marker_payload(db_path)

    assert detect_provider(payload) is Provider.HERMES
    sessions = parse_payload("hermes", payload, "fallback", source_path=str(db_path))

    assert [session.provider_session_id.split("@", 1)[0] for session in sessions] == [
        "hermes-root",
        "hermes-child",
    ]


def test_hermes_state_db_source_iterator_captures_raw_blob(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    blob_root = tmp_path / "blob"
    _write_hermes_state_db(db_path)

    rows = list(
        iter_source_sessions_with_raw(
            Source(name="hermes", path=db_path),
            capture_raw=True,
            blob_root=blob_root,
        )
    )

    assert [session.provider_session_id.split("@", 1)[0] for _raw, session in rows] == [
        "hermes-root",
        "hermes-child",
    ]
    raw = rows[0][0]
    assert raw is not None
    assert raw.raw_bytes == b""
    assert raw.blob_hash
    assert raw.blob_size and raw.blob_size > 0


def test_hermes_configured_directory_admits_only_its_state_database(tmp_path: Path) -> None:
    """Enumeration is broad; admission is structural.

    Anti-vacuity: dropping the recognizer from admission would make
    ``unrelated.sqlite`` produce sessions (or a parse failure) instead of a
    typed non-session observation.
    """

    source_root = tmp_path / "hermes"
    source_root.mkdir()
    db_path = source_root / "state.db"
    unrelated_path = source_root / "unrelated.sqlite"
    text_path = source_root / "notes.txt"
    _write_hermes_state_db(db_path)
    text_path.write_text("not a database", encoding="utf-8")
    with sqlite3.connect(unrelated_path) as conn:
        conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")

    hermes_source = Source(name="hermes", path=source_root)
    rows = list(
        iter_source_sessions_with_raw(
            hermes_source,
            capture_raw=True,
            blob_root=tmp_path / "blob",
        )
    )

    assert _resolve_source_paths(hermes_source) == [db_path, unrelated_path]
    assert [session.provider_session_id.split("@", 1)[0] for _raw, session in rows] == [
        "hermes-root",
        "hermes-child",
    ]


def test_unrelated_configured_hermes_sqlite_is_enumerated_but_not_admitted(tmp_path: Path) -> None:
    db_path = tmp_path / "unrelated.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")

    assert _resolve_source_paths(Source(name="hermes", path=tmp_path)) == [db_path]
    assert (
        list(
            iter_source_sessions_with_raw(
                Source(name="hermes", path=tmp_path),
                capture_raw=False,
                blob_root=tmp_path / "blob",
            )
        )
        == []
    )


def test_hermes_state_db_source_iterator_snapshots_wal_before_parsing(tmp_path: Path) -> None:
    from polylogue.pipeline.ids import session_content_hash

    db_path = tmp_path / "state.db"
    blob_root = tmp_path / "blob"
    _write_hermes_state_db(db_path)

    writer = sqlite3.connect(db_path)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            ("hermes-root", "assistant", "committed only in WAL", 1_775_000_004.0),
        )
        writer.commit()
        assert db_path.with_name("state.db-wal").stat().st_size > 0

        rows = list(
            iter_source_sessions_with_raw(
                Source(name="hermes", path=db_path),
                capture_raw=True,
                blob_root=blob_root,
            )
        )
        root = next(session for _raw, session in rows if session.provider_session_id.split("@", 1)[0] == "hermes-root")
        raw = rows[0][0]
        assert raw is not None and raw.blob_hash is not None
        retained_path = BlobStore(blob_root).blob_path(raw.blob_hash)
        assert root.messages[-1].text == "committed only in WAL"
    finally:
        writer.close()

    db_path.unlink()
    db_path.with_name("state.db-wal").unlink(missing_ok=True)
    db_path.with_name("state.db-shm").unlink(missing_ok=True)
    reparsed = hermes_state.parse_state_db(retained_path, profile_root=db_path.parent)
    reparsed_root = next(
        session for session in reparsed if session.provider_session_id.split("@", 1)[0] == "hermes-root"
    )
    assert reparsed_root.provider_session_id == root.provider_session_id
    assert reparsed_root.messages[-1].text == "committed only in WAL"
    assert session_content_hash(reparsed_root) == session_content_hash(root)

    corrupted_path = tmp_path / "corrupted-retained.db"
    corrupted_path.write_bytes(retained_path.read_bytes())
    corrupted_path.write_bytes(b"not a SQLite database")
    with pytest.raises(sqlite3.DatabaseError):
        hermes_state.parse_state_db(corrupted_path, profile_root=db_path.parent)


def test_hermes_snapshot_parse_route_keeps_live_wal_sidecars_out_of_namespace(tmp_path: Path) -> None:
    """The live Hermes snapshot route must isolate SQLite work files from CAS paths."""
    from polylogue.sources.sqlite_export import read_export_header

    db_path = tmp_path / "state.db"
    blob_root = tmp_path / "blob"
    _write_hermes_state_db(db_path)
    writer = sqlite3.connect(db_path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("PRAGMA application_id=51966")
    writer.commit()
    assert db_path.with_name("state.db-wal").exists()
    assert db_path.with_name("state.db-shm").exists()

    try:
        rows = list(
            iter_source_sessions_with_raw(
                Source(name="hermes", path=db_path),
                capture_raw=True,
                blob_root=blob_root,
            )
        )
    finally:
        writer.close()

    raw = rows[0][0]
    assert raw is not None and raw.blob_hash is not None
    store = BlobStore(blob_root)
    entries = tuple(store.iter_namespace())
    assert [(entry.kind, entry.hash_hex) for entry in entries] == [("blob", raw.blob_hash)]
    header = read_export_header(store.blob_path(raw.blob_hash))
    assert {"sessions", "messages"} <= set(header.tables)
    assert not tuple(store.staging_root.iterdir())
    assert store.verify_all().passed is True


def test_hermes_state_db_raw_payload_envelope_uses_marker(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)

    envelope = build_raw_payload_envelope(
        db_path,
        source_path=db_path,
        fallback_provider="inbox",
        sqlite_immutable=False,
    )

    assert envelope.provider is Provider.HERMES
    assert envelope.artifact.parse_as_session is True
    assert envelope.payload == hermes_state.marker_payload(db_path, profile_root=db_path.parent, immutable=False)
    assert "sqlite_immutable" not in envelope.payload


def test_hermes_state_db_raw_payload_envelope_default_keeps_live_marker_semantics(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)

    envelope = build_raw_payload_envelope(db_path, source_path=db_path, fallback_provider="inbox")

    assert envelope.payload == hermes_state.marker_payload(db_path, profile_root=db_path.parent, immutable=False)
    assert "sqlite_immutable" not in envelope.payload


def test_hermes_state_db_live_batch_classifies_as_session_artifact(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)

    assert _detect_provider_from_path_sample(db_path, Provider.UNKNOWN) is Provider.HERMES
    assert _parse_path_as_session_artifact(db_path, provider=Provider.HERMES) is True


def test_agent_sidecars_are_classified_as_non_session() -> None:
    logs = classify_artifact_path("~/.gemini/tmp/polylogue/logs.json", provider=Provider.GEMINI_CLI)
    request_dump = classify_artifact_path("~/.hermes/sessions/request_dump_1.json", provider=Provider.HERMES)
    antigravity_pb = classify_artifact_path(
        "~/.gemini/antigravity/sessions/session.pb",
        provider=Provider.ANTIGRAVITY,
    )
    antigravity_resolved = classify_artifact_path(
        "~/.gemini/antigravity/brain/session/task.md.resolved.1",
        provider=Provider.ANTIGRAVITY,
    )
    hermes_sqlite_evidence = classify_artifact_path(
        "~/.hermes/sessions/verification_evidence.db",
        provider=Provider.HERMES,
    )

    assert logs is not None
    assert logs.parse_as_session is False
    assert request_dump is not None
    assert request_dump.parse_as_session is False
    assert antigravity_pb is not None
    assert antigravity_pb.parse_as_session is False
    assert antigravity_resolved is not None
    assert antigravity_resolved.parse_as_session is False
    assert hermes_sqlite_evidence is not None
    assert hermes_sqlite_evidence.parse_as_session is False
    assert hermes_sqlite_evidence.schema_eligible is False
    assert hermes_sqlite_evidence.reason == "Hermes SQLite evidence sidecar"


def test_runtime_provider_identity_keeps_gemini_surfaces_distinct() -> None:
    assert Provider.from_string("gemini") is Provider.GEMINI
    assert Provider.from_string("aistudio") is Provider.GEMINI
    assert Provider.from_string("gemini-cli") is Provider.GEMINI_CLI
    assert Provider.from_string("antigravity") is Provider.ANTIGRAVITY


def test_antigravity_brain_artifact_metadata_parses_sibling_markdown(tmp_path: Path) -> None:
    session_dir = tmp_path / "brain" / "03c22aa3-8b7f-438d-baa8-d12567249cd9"
    session_dir.mkdir(parents=True)
    artifact = session_dir / "implementation_plan.md"
    metadata = session_dir / "implementation_plan.md.metadata.json"
    artifact.write_text("# Implementation Plan\n\nDo the work.\n", encoding="utf-8")
    payload: JSONDocument = {
        "artifactType": "ARTIFACT_TYPE_OTHER",
        "summary": "Implementation plan for source ingestion",
        "updatedAt": "2026-01-07T19:08:15.216541610Z",
    }
    metadata.write_text(
        (
            '{"artifactType":"ARTIFACT_TYPE_OTHER",'
            '"summary":"Implementation plan for source ingestion",'
            '"updatedAt":"2026-01-07T19:08:15.216541610Z"}'
        ),
        encoding="utf-8",
    )

    # Per-artifact brain metadata is a path-classified sidecar, never a
    # primary session by default (polylogue-eo81): the generic walk must not
    # fragment one file per artifact into noise sessions now that the real
    # conversation is acquired via the language-server export route.
    classification = classify_artifact(payload, provider=Provider.ANTIGRAVITY, source_path=metadata)
    assert classification.parse_as_session is False
    assert classification.kind is ArtifactKind.AGENT_SIDECAR_META

    # The generic payload dispatcher has no session route for sidecars. The
    # live and batch routes retain them as typed artifacts instead.
    assert (
        parse_payload(
            Provider.ANTIGRAVITY,
            payload,
            "fallback",
            source_path=str(metadata),
        )
        == []
    )


def test_antigravity_metadata_sidecar_is_rejected_without_blocking_conversation_json(tmp_path: Path) -> None:
    metadata_path = tmp_path / "brain" / "work-session" / "plan.metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_payload: JSONDocument = {
        "artifactType": "ARTIFACT_TYPE_OTHER",
        # Keep this as a valid brain metadata document while forcing the
        # ``_ingest_full_paths_sync`` large-file branch. The sidecar must be
        # excluded before its bytes are copied or parsed as a session.
        "summary": "Plan " + ("x" * _STREAMING_FULL_INGEST_BYTES),
        "updatedAt": "2026-08-04T08:00:00Z",
    }
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")

    path_classification = classify_artifact_path(metadata_path, provider=Provider.ANTIGRAVITY)
    assert path_classification is not None
    assert path_classification.kind is ArtifactKind.AGENT_SIDECAR_META
    assert path_classification.parse_as_session is False
    assert (
        classify_artifact(
            metadata_payload,
            provider=Provider.ANTIGRAVITY,
            source_path=metadata_path,
        ).parse_as_session
        is False
    )
    assert _parse_path_as_session_artifact(metadata_path, provider=Provider.ANTIGRAVITY) is False

    conversation_payload = antigravity.markdown_export_payload(
        antigravity.AntigravitySessionSummary(cascade_id="cascade-json", title="Conversation"),
        "### User Input\n\nhello\n\n### Planner Response\n\nhi",
    )
    conversation_path = tmp_path / "conversations" / "cascade-json.json"
    conversation_path.parent.mkdir()
    conversation_path.write_text(json.dumps(conversation_payload), encoding="utf-8")

    assert _parse_path_as_session_artifact(conversation_path, provider=Provider.ANTIGRAVITY) is True
    [session] = parse_payload(
        Provider.ANTIGRAVITY,
        conversation_payload,
        "cascade-json",
        source_path=str(conversation_path),
    )
    assert session.provider_session_id == "cascade-json"
    assert [message.text for message in session.messages] == ["hello", "hi"]

    assert metadata_path.stat().st_size > _STREAMING_FULL_INGEST_BYTES
    assert conversation_path.stat().st_size < _STREAMING_FULL_INGEST_BYTES
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="antigravity", root=tmp_path),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    admission = processor._ingest_full_paths_sync(
        [metadata_path, conversation_path],
        source_name="antigravity",
    )

    assert admission.succeeded == [metadata_path, conversation_path]
    assert admission.failed == []
    assert str(metadata_path) not in processor._cursor.list_excluded()
    assert str(conversation_path) not in processor._cursor.list_excluded()


def test_antigravity_language_server_markdown_export_parses_turns() -> None:
    markdown = """# Chat Session

Note: _This is purely the output of the chat session._

### User Input

Run the checks.

*User accepted the command `pytest -q`*

### Planner Response

Checks passed.
"""
    summary = antigravity.AntigravitySessionSummary(
        cascade_id="e85783e3-f047-49b8-9035-4029f58dd04a",
        title="Refactoring and Executing Plan",
        workspace_name="Sinity/sinex",
        snippet="Run the checks.",
        last_modified_time="2026-03-05T04:21:34.468316671Z",
    )

    session = antigravity.parse_markdown_export(markdown, summary)

    assert session.source_name is Provider.ANTIGRAVITY
    assert session.provider_session_id == "e85783e3-f047-49b8-9035-4029f58dd04a"
    assert session.title == "Refactoring and Executing Plan"
    assert session.updated_at == "2026-03-05T04:21:34.468316671Z"
    assert [message.role for message in session.messages] == ["user", "assistant", "assistant"]
    assert session.messages[0].text == "Run the checks."
    activity = session.messages[1]
    assert activity.message_type is MessageType.TOOL_USE
    assert [block.type for block in activity.blocks] == [BlockType.TOOL_USE]
    assert activity.blocks[0].tool_name == "accepted_command"
    assert activity.blocks[0].tool_input == {"command": "pytest -q"}
    assert session.messages[2].text == "Checks passed."
    assert session.provider_session_id == "e85783e3-f047-49b8-9035-4029f58dd04a"


def test_antigravity_language_server_export_dispatches_as_session() -> None:
    payload = antigravity.markdown_export_payload(
        antigravity.AntigravitySessionSummary(
            cascade_id="cascade-1",
            title="Session",
            last_modified_time="2026-03-05T04:21:34Z",
        ),
        "### User Input\n\nhello\n\n### Planner Response\n\nhi",
    )

    assert detect_provider(payload) is Provider.ANTIGRAVITY
    classification = classify_artifact(payload, provider=Provider.ANTIGRAVITY)
    assert classification.parse_as_session is True

    [session] = parse_payload(Provider.ANTIGRAVITY, payload, "fallback")

    assert session.provider_session_id == "cascade-1"
    assert session.updated_at == "2026-03-05T04:21:34Z"
    assert [message.text for message in session.messages] == ["hello", "hi"]


def test_antigravity_source_walk_prefers_language_server_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "conversations").mkdir()
    exported = antigravity.parse_markdown_export(
        "### User Input\n\nhello\n\n### Planner Response\n\nhi",
        antigravity.AntigravitySessionSummary(cascade_id="cascade-1", title="Session"),
    )

    (tmp_path / "conversations" / "cascade-1.pb").write_bytes(b"opaque protobuf")

    def fake_exports(
        root: Path, *, only_cascade_ids: frozenset[str] | None = None
    ) -> list[antigravity.AntigravityExportOutcome]:
        assert root == tmp_path
        del only_cascade_ids
        return [antigravity.AntigravityExportOutcome(tmp_path / "conversations/cascade-1.pb", "cascade-1", exported)]

    monkeypatch.setattr(
        "polylogue.sources.source_parsing.antigravity.iter_language_server_export_results",
        fake_exports,
    )

    sessions = list(iter_source_sessions(Source(name="antigravity", path=tmp_path)))

    assert [session.provider_session_id for session in sessions] == ["cascade-1"]
    assert sessions[0].messages[0].text == "hello"


def test_antigravity_source_walk_leaves_metadata_artifact_only_without_conversations(tmp_path: Path) -> None:
    session_dir = tmp_path / "brain" / "session-1"
    session_dir.mkdir(parents=True)
    (session_dir / "task.md").write_text("Task artifact", encoding="utf-8")
    (session_dir / "task.md.metadata.json").write_text(
        '{"artifactType":"ARTIFACT_TYPE_OTHER","summary":"Task","updatedAt":"2026-01-07T00:00:00Z"}',
        encoding="utf-8",
    )
    (tmp_path / "mcp_config.json").write_text("{}", encoding="utf-8")

    sessions = list(iter_source_sessions(Source(name="antigravity", path=tmp_path)))

    assert sessions == []


def _hermes_snapshot_payload() -> JSONDocument:
    """The ``sessions/session_*.json`` view of the ``_write_hermes_state_db`` root session."""
    return {
        "session_id": "hermes-root",
        "model": "nous-hermes-test",
        "platform": "linux",
        "session_start": "2026-04-01T00:00:00",
        "last_updated": "2026-04-01T00:05:00",
        "system_prompt": "be precise",
        "messages": [
            {"role": "user", "content": "run pytest"},
            {
                "role": "assistant",
                "content": "running",
                "finish_reason": "tool_calls",
                "tool_calls": [{"id": "call-1", "function": {"name": "shell", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "passed"},
            {"role": "assistant", "content": "done", "finish_reason": "stop"},
        ],
    }


def test_hermes_snapshot_finish_reason_sets_end_turn_and_stop_reason() -> None:
    """polylogue-q5xr4: the JSON-snapshot path must conserve the turn-terminal signal.

    Anti-vacuity: deleting the ``finish_reason`` read in
    ``_parse_hermes_message`` leaves ``end_turn``/``stop_reason`` ``None`` on
    the tool-call and stop turns, which every assertion below rejects.
    """
    [session] = parse_payload("hermes", _hermes_snapshot_payload(), "fallback")

    system, user, tool_call_turn, tool_result, final = session.messages
    assert (system.end_turn, system.stop_reason) == (None, None)
    assert (user.end_turn, user.stop_reason) == (None, None)
    assert (tool_call_turn.end_turn, tool_call_turn.stop_reason) == (False, "tool_use")
    assert (tool_result.end_turn, tool_result.stop_reason) == (None, None)
    assert (final.end_turn, final.stop_reason) == (True, "end_turn")


def test_hermes_state_db_finish_reason_sets_end_turn_and_stop_reason(tmp_path: Path) -> None:
    """polylogue-q5xr4: the state.db path agrees with the snapshot path, field for field.

    Anti-vacuity: restoring ``finish_reason != "tool_calls"`` as the whole
    rule makes the ``None`` assertions red, because absence would again be
    read as a terminal turn.
    """
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)

    root = hermes_state.parse_state_db(db_path, fallback_id="fallback")[0]

    signals = [(message.role, message.end_turn, message.stop_reason) for message in root.messages]
    assert (signals[3][1], signals[3][2]) == (False, "tool_use"), signals
    assert [(end_turn, stop_reason) for _role, end_turn, stop_reason in signals if end_turn is not None] == [
        (False, "tool_use")
    ], signals


def test_hermes_snapshot_and_state_db_share_one_session_identity(tmp_path: Path) -> None:
    """polylogue-xfpa8: one logical Hermes session is one archive session row.

    Anti-vacuity: dropping the profile qualification in ``parse_hermes``
    makes the snapshot land under the bare raw id, so the archive holds two
    ``hermes-session`` rows instead of one.
    """
    profile_root = tmp_path / ".hermes"
    snapshot_dir = profile_root / "sessions"
    snapshot_dir.mkdir(parents=True)
    db_path = profile_root / "state.db"
    _write_hermes_state_db(db_path)
    snapshot_path = snapshot_dir / "session_hermes-root.json"
    snapshot_path.write_text(json.dumps(_hermes_snapshot_payload()), encoding="utf-8")

    state_root = hermes_state.parse_state_db(db_path, fallback_id="fallback")[0]
    [snapshot] = parse_payload(
        "hermes",
        _hermes_snapshot_payload(),
        "fallback",
        source_path=str(snapshot_path),
    )

    assert snapshot.provider_session_id == state_root.provider_session_id
    assert snapshot.title == "hermes-root"

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        _ensure_schema(conn)
        for parsed in (state_root, snapshot):
            write_parsed_session_to_archive(conn, parsed, content_hash=session_content_hash(parsed))
        rows = [
            row["native_id"]
            for row in conn.execute("SELECT native_id FROM sessions WHERE origin = 'hermes-session'").fetchall()
        ]
    finally:
        conn.close()

    assert rows == [state_root.provider_session_id]


def test_hermes_snapshot_without_source_path_stays_unqualified() -> None:
    """polylogue-xfpa8: no asserted profile means no invented profile key."""
    [session] = parse_payload("hermes", _hermes_snapshot_payload(), "fallback")

    assert session.provider_session_id == "hermes-root"
