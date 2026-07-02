from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from devtools.claim_vs_evidence import build_report


def _seed_archive(root: Path) -> None:
    root.mkdir(parents=True)
    conn = sqlite3.connect(root / "index.db")
    conn.executescript(
        """
        PRAGMA user_version=22;
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            title TEXT,
            created_at_ms INTEGER,
            updated_at_ms INTEGER
        );
        CREATE TABLE messages (
            session_id TEXT NOT NULL,
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            position INTEGER NOT NULL,
            model_name TEXT
        );
        CREATE TABLE blocks (
            block_id TEXT GENERATED ALWAYS AS (message_id || ':' || position) STORED UNIQUE,
            message_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            block_type TEXT NOT NULL,
            text TEXT,
            tool_name TEXT,
            tool_id TEXT,
            tool_input TEXT,
            semantic_type TEXT,
            tool_result_is_error INTEGER,
            tool_result_exit_code INTEGER,
            tool_command TEXT GENERATED ALWAYS AS (json_extract(tool_input, '$.command')) VIRTUAL,
            tool_path TEXT GENERATED ALWAYS AS (
                COALESCE(json_extract(tool_input, '$.file_path'), json_extract(tool_input, '$.path'))
            ) VIRTUAL,
            PRIMARY KEY(message_id, position)
        );
        CREATE INDEX idx_blocks_type ON blocks(block_type);
        CREATE INDEX idx_blocks_tool_result_outcome
        ON blocks(block_type, tool_result_is_error, tool_result_exit_code, session_id, tool_id, message_id)
        WHERE block_type = 'tool_result';
        CREATE INDEX idx_blocks_tool_id ON blocks(tool_id) WHERE tool_id IS NOT NULL;
        CREATE INDEX idx_messages_session_position ON messages(session_id, position);
        CREATE VIEW actions AS
        SELECT
            u.session_id,
            u.message_id,
            u.block_id AS tool_use_block_id,
            u.tool_name,
            u.semantic_type,
            u.tool_command,
            u.tool_path,
            u.tool_input,
            r.text AS output_text,
            r.tool_result_is_error AS is_error,
            r.tool_result_exit_code AS exit_code,
            r.block_id AS tool_result_block_id
        FROM blocks u
        LEFT JOIN blocks r
            ON r.tool_id = u.tool_id
           AND r.session_id = u.session_id
           AND r.block_type = 'tool_result'
        WHERE u.block_type = 'tool_use';
        """
    )
    conn.executemany(
        "INSERT INTO sessions(session_id, origin, title, created_at_ms, updated_at_ms) VALUES (?, ?, ?, ?, ?)",
        [
            ("s1", "claude-code-session", "fixture one", 1, 4),
            ("s2", "codex-session", "fixture two", 1, 1),
        ],
    )
    conn.executemany(
        "INSERT INTO messages(session_id, message_id, role, position, model_name) VALUES (?, ?, ?, ?, ?)",
        [
            ("s1", "tool-ack", "tool", 1, "claude-opus"),
            ("s1", "next-ack", "assistant", 2, "claude-sonnet"),
            ("s1", "tool-silent", "tool", 3, "claude-opus"),
            ("s1", "next-silent", "assistant", 4, "claude-haiku"),
            ("s2", "tool-missing-next", "tool", 1, "codex"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO blocks(
            message_id, session_id, position, block_type, text, tool_name, tool_id,
            tool_input, tool_result_is_error, tool_result_exit_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("tool-ack", "s1", 0, "tool_use", None, "Bash", "t1", '{"command":"pytest"}', None, None),
            ("tool-ack", "s1", 1, "tool_result", "failed", None, "t1", None, 1, None),
            (
                "next-ack",
                "s1",
                0,
                "text",
                "The command failed with exit code 2, so I will fix it.",
                None,
                None,
                None,
                None,
                None,
            ),
            ("tool-silent", "s1", 0, "tool_use", None, "Bash", "t2", '{"command":"ls missing"}', None, None),
            ("tool-silent", "s1", 1, "tool_result", "missing", None, "t2", None, 0, 2),
            (
                "next-silent",
                "s1",
                0,
                "text",
                "I will continue by inspecting the neighboring module now.",
                None,
                None,
                None,
                None,
                None,
            ),
            ("tool-missing-next", "s2", 0, "tool_use", None, "Read", "t3", '{"path":"x"}', None, None),
            ("tool-missing-next", "s2", 1, "tool_result", "nope", None, "t3", None, 0, 1),
        ],
    )
    conn.commit()
    conn.close()


def test_claim_vs_evidence_builds_bounded_artifacts(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    out_dir = tmp_path / "out"
    _seed_archive(archive)

    report = build_report(
        argparse.Namespace(
            archive_root=archive,
            out_dir=out_dir,
            limit=3,
            sample_limit=2,
            json=False,
        )
    )

    assert report["index_schema_version"] == 22
    assert report["sample_frame"] == {
        "classification_scope": "immediately following assistant message only",
        "complete_failure_frame": True,
        "failure_predicate": "tool_result_is_error = 1 OR tool_result_exit_code != 0",
        "inspected_structured_failures": 3,
        "limit": 3,
        "selection_order": "session_id, tool_result_message_position, tool_result_message_id, tool_id",
        "total_by_origin": [
            {"failed_outcomes": 2, "origin": "claude-code-session"},
            {"failed_outcomes": 1, "origin": "codex-session"},
        ],
        "total_structured_failures": 3,
    }
    assert report["totals"] == {
        "failed_outcomes": 3,
        "acknowledged": 1,
        "silent_proceed": 1,
        "ambiguous": 1,
        "classified_outcomes": 2,
    }
    assert report["rates"]["silent_rate_lower_bound"] == 1 / 3
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["claim"]
    assert summary["non_claim"]
    assert summary["proof_report"]["failed_outcomes"] == 3
    assert summary["proof_report"]["complete_failure_frame"] is True
    assert (out_dir / "claim-vs-evidence.report.json").exists()
    assert "Claim-vs-Evidence" in (out_dir / "README.md").read_text()
