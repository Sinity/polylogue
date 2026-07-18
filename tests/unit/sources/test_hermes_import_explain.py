from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.sources.import_explain import explain_import_path
from polylogue.sources.parsers.hermes_state import parse_state_db


def _write_state_db(path: Path, *, include_cost_provenance: bool = True) -> None:
    cost_columns = (
        """
        estimated_cost_usd REAL,
        actual_cost_usd REAL,
        cost_status TEXT,
        cost_source TEXT,
        pricing_version TEXT,
        billing_provider TEXT,
        billing_base_url TEXT,
        billing_mode TEXT,
    """
        if include_cost_provenance
        else ""
    )
    with sqlite3.connect(path) as conn:
        conn.executescript(
            f"""
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                model_config TEXT,
                parent_session_id TEXT,
                started_at REAL,
                ended_at REAL,
                end_reason TEXT,
                rewind_count INTEGER,
                archived INTEGER,
                {cost_columns}
                title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL,
                tool_calls TEXT,
                observed INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1,
                compacted INTEGER DEFAULT 0
            );
            """
        )
        columns = "id, source, model_config, parent_session_id, started_at, ended_at, end_reason, rewind_count, archived, title"
        values: list[object] = ["root", "cli", "{}", None, 1.0, 8.0, "completed", 1, 0, "root"]
        if include_cost_provenance:
            columns += ", estimated_cost_usd, actual_cost_usd, cost_status, cost_source, pricing_version, billing_provider, billing_base_url, billing_mode"
            values += [
                0.03,
                0.02,
                "estimated",
                "litellm",
                "2026-07-12",
                "openrouter",
                "https://example.invalid",
                "metered",
            ]
        conn.execute(f"INSERT INTO sessions ({columns}) VALUES ({','.join('?' for _ in values)})", values)
        conn.execute(
            "INSERT INTO sessions (id, source, model_config, parent_session_id, started_at, ended_at, end_reason, rewind_count, archived, title) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("child", "cli", json.dumps({"_delegate_from": "root"}), "root", 2.0, 7.0, "completed", 0, 0, "child"),
        )
        conn.executemany(
            "INSERT INTO messages (id, session_id, role, content, timestamp, tool_calls, observed, active, compacted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "root", "user", "run checks", 2.0, "[]", 0, 1, 0),
                (2, "root", "user", "ambient output", 3.0, "[]", 1, 1, 0),
                (3, "root", "assistant", "rewound", 4.0, "[]", 0, 0, 0),
                (4, "root", "assistant", "compacted", 5.0, "[]", 0, 0, 1),
                (5, "child", "assistant", "delegated", 6.0, "[]", 0, 1, 0),
            ],
        )


def test_hermes_state_db_explain_declares_v16_fidelity_and_coverage(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    _write_state_db(path)

    [entry] = explain_import_path(path, source_name="hermes").entries

    assert entry.detector == "hermes_state_db"
    assert entry.parser_mode == "sqlite_backup"
    assert entry.produced.sessions == 2
    assert entry.fidelity is not None
    assert entry.fidelity.schema_version == 16
    assert entry.fidelity.acquisition_method == "sqlite_backup"
    assert entry.fidelity.retained_blob_reproducibility.status == "exact"
    assert entry.fidelity.capabilities["message_state"].status == "exact"
    assert entry.fidelity.capabilities["message_state"].counts == {
        "active": 2,
        "observed": 1,
        "rewound": 1,
        "compacted": 1,
    }
    assert entry.fidelity.capabilities["material_origin"].status == "inferred"
    assert entry.fidelity.capabilities["cost_provenance"].status == "degraded"
    assert entry.fidelity.capabilities["cost_provenance"].observed == 1
    assert entry.fidelity.capabilities["cost_provenance"].expected == 2
    assert entry.fidelity.capabilities["runtime_spans"].status == "absent"
    assert any(caveat.startswith("runtime_spans:") for caveat in entry.caveats)


def test_hermes_state_db_explain_reports_later_schema_capability(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    _write_state_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE sessions ADD COLUMN cwd TEXT")
        conn.execute("ALTER TABLE sessions ADD COLUMN git_branch TEXT")
        conn.execute("ALTER TABLE sessions ADD COLUMN git_repo_root TEXT")
        conn.execute("UPDATE sessions SET cwd = '/repo', git_branch = 'feature/hermes', git_repo_root = '/repo'")
        conn.execute("UPDATE schema_version SET version = 17")

    [entry] = explain_import_path(path, source_name="unknown").entries

    assert entry.fidelity is not None
    assert entry.fidelity.schema_version == 17
    assert entry.fidelity.capabilities["repository"].status == "exact"


def test_hermes_state_db_v19_gateway_and_compression_recovery_fields_round_trip(tmp_path: Path) -> None:
    """Schema v19 (live-verified 2026-07-18 against ~/.hermes/state.db) adds
    gateway-routing identity (#9006 chat-platform bridging) and compression
    retry state columns not present in the v16/v17 fixtures above. Both must
    be admitted as typed capabilities and surfaced as metadata, not silently
    dropped -- this repo's CLI-only live install carries no gateway data, so
    this fixture proves the mapping the same way v16-vs-v17 already does."""

    path = tmp_path / "state.db"
    _write_state_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE sessions ADD COLUMN session_key TEXT")
        conn.execute("ALTER TABLE sessions ADD COLUMN chat_id TEXT")
        conn.execute("ALTER TABLE sessions ADD COLUMN chat_type TEXT")
        conn.execute("ALTER TABLE sessions ADD COLUMN thread_id TEXT")
        conn.execute("ALTER TABLE sessions ADD COLUMN display_name TEXT")
        conn.execute("ALTER TABLE sessions ADD COLUMN origin_json TEXT")
        conn.execute("ALTER TABLE sessions ADD COLUMN expiry_finalized INTEGER DEFAULT 0")
        conn.execute("ALTER TABLE sessions ADD COLUMN compression_failure_cooldown_until REAL")
        conn.execute("ALTER TABLE sessions ADD COLUMN compression_failure_error TEXT")
        conn.execute(
            "UPDATE sessions SET session_key = ?, chat_id = ?, chat_type = ?, thread_id = ?, "
            "display_name = ?, origin_json = ?, expiry_finalized = 1, "
            "compression_failure_cooldown_until = ?, compression_failure_error = ? WHERE id = 'root'",
            (
                "gw-session-1",
                "chat-1",
                "slack",
                "thread-1",
                "Ops Room",
                json.dumps({"platform": "slack"}),
                123.0,
                "rate_limited",
            ),
        )
        conn.execute("UPDATE schema_version SET version = 19")

    [entry] = explain_import_path(path, source_name="hermes").entries
    assert entry.fidelity is not None
    assert entry.fidelity.schema_version == 19
    assert entry.fidelity.capabilities["gateway_identity"].status == "exact"
    assert entry.fidelity.capabilities["compression_recovery"].status == "exact"

    sessions = parse_state_db(path)
    root = next(session for session in sessions if session.provider_session_id.startswith("root@profile-"))
    metadata_events = [event for event in root.session_events if event.event_type == "hermes_session_metadata"]
    assert len(metadata_events) == 1
    payload = metadata_events[0].payload
    assert payload["session_key"] == "gw-session-1"
    assert payload["chat_id"] == "chat-1"
    assert payload["chat_type"] == "slack"
    assert payload["thread_id"] == "thread-1"
    assert payload["display_name"] == "Ops Room"
    assert payload["origin_json"] == json.dumps({"platform": "slack"})
    assert payload["expiry_finalized"] == 1
    assert payload["compression_failure_cooldown_until"] == 123.0
    assert payload["compression_failure_error"] == "rate_limited"


def test_hermes_json_fallback_explain_declares_missing_state_evidence(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    path.write_text(
        json.dumps(
            {
                "session_id": "fallback-1",
                "model": "local-model",
                "platform": "linux",
                "session_start": "2026-07-12T12:00:00Z",
                "last_updated": "2026-07-12T12:01:00Z",
                "messages": [{"role": "user", "content": "inspect evidence"}],
            }
        ),
        encoding="utf-8",
    )

    [entry] = explain_import_path(path, source_name="hermes").entries

    assert entry.fidelity is not None
    assert entry.fidelity.acquisition_method == "json_fallback"
    assert entry.fidelity.retained_blob_reproducibility.status == "absent"
    assert entry.fidelity.capabilities["message_state"].status == "absent"
    assert entry.fidelity.capabilities["runtime_spans"].status == "absent"


def test_hermes_state_db_explain_marks_removed_cost_provenance_absent(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    _write_state_db(path, include_cost_provenance=False)

    [entry] = explain_import_path(path, source_name="hermes").entries

    assert entry.fidelity is not None
    cost = entry.fidelity.capabilities["cost_provenance"]
    assert cost.status == "absent"
    assert cost.observed == 0
    assert any(caveat.startswith("cost_provenance:") for caveat in entry.caveats)
