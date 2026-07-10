from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.artifact_taxonomy import classify_artifact, classify_artifact_path
from polylogue.archive.raw_payload import build_raw_payload_envelope
from polylogue.config import Source
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.core.json import JSONDocument
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.live.batch_support import _detect_provider_from_path_sample, _parse_path_as_session_artifact
from polylogue.sources.parsers import antigravity, hermes_state
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.source_parsing import iter_source_sessions, iter_source_sessions_with_raw
from polylogue.storage.blob_store import BlobStore


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
    assert session.provider_session_id == "gemini-session-1"
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
    assert any(block.type is BlockType.TOOL_RESULT for block in session.messages[3].blocks)


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


def test_hermes_state_db_raw_payload_envelope_uses_marker(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)

    envelope = build_raw_payload_envelope(db_path, source_path=db_path, fallback_provider="inbox")

    assert envelope.provider is Provider.HERMES
    assert envelope.artifact.parse_as_session is True
    assert envelope.payload == hermes_state.marker_payload(db_path, profile_root=db_path.parent)


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

    assert logs is not None
    assert logs.parse_as_session is False
    assert request_dump is not None
    assert request_dump.parse_as_session is False
    assert antigravity_pb is not None
    assert antigravity_pb.parse_as_session is False
    assert antigravity_resolved is not None
    assert antigravity_resolved.parse_as_session is False


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

    classification = classify_artifact(payload, provider=Provider.ANTIGRAVITY, source_path=metadata)
    assert classification.parse_as_session is True

    [session] = parse_payload(
        Provider.ANTIGRAVITY,
        payload,
        "fallback",
        source_path=str(metadata),
    )

    assert session.source_name is Provider.ANTIGRAVITY
    assert session.provider_session_id == "03c22aa3-8b7f-438d-baa8-d12567249cd9:implementation_plan.md"
    assert session.updated_at == "2026-01-07T19:08:15.216541610Z"
    assert session.messages[0].text == "# Implementation Plan\n\nDo the work.\n"
    assert session.source_name is Provider.ANTIGRAVITY


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
    assert [message.role for message in session.messages] == ["user", "assistant"]
    assert "pytest -q" in (session.messages[0].text or "")
    assert session.messages[1].text == "Checks passed."
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
    (tmp_path / "sessions").mkdir()
    exported = antigravity.parse_markdown_export(
        "### User Input\n\nhello\n\n### Planner Response\n\nhi",
        antigravity.AntigravitySessionSummary(cascade_id="cascade-1", title="Session"),
    )

    def fake_exports(root: Path) -> list[ParsedSession]:
        assert root == tmp_path
        return [exported]

    monkeypatch.setattr("polylogue.sources.source_parsing.antigravity.iter_language_server_exports", fake_exports)

    sessions = list(iter_source_sessions(Source(name="antigravity", path=tmp_path)))

    assert [session.provider_session_id for session in sessions] == ["cascade-1"]
    assert sessions[0].messages[0].text == "hello"


def test_antigravity_source_walk_ingests_metadata_not_config(tmp_path: Path) -> None:
    session_dir = tmp_path / "brain" / "session-1"
    session_dir.mkdir(parents=True)
    (session_dir / "task.md").write_text("Task artifact", encoding="utf-8")
    (session_dir / "task.md.metadata.json").write_text(
        '{"artifactType":"ARTIFACT_TYPE_OTHER","summary":"Task","updatedAt":"2026-01-07T00:00:00Z"}',
        encoding="utf-8",
    )
    (tmp_path / "mcp_config.json").write_text("{}", encoding="utf-8")

    sessions = list(iter_source_sessions(Source(name="antigravity", path=tmp_path)))

    assert len(sessions) == 1
    assert sessions[0].source_name is Provider.ANTIGRAVITY
