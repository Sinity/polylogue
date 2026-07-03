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
            ("s2", "next-prose", "assistant", 2, "codex"),
            ("s2", "tool-wordless", "tool", 3, "codex"),
            ("s2", "next-wordless", "assistant", 4, "codex"),
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
            (
                "next-prose",
                "s2",
                0,
                "text",
                "Ok.",
                None,
                None,
                None,
                None,
                None,
            ),
            ("tool-wordless", "s2", 0, "tool_use", None, "Read", "t4", '{"path":"y"}', None, None),
            ("tool-wordless", "s2", 1, "tool_result", "nope again", None, "t4", None, 0, 1),
            ("next-wordless", "s2", 0, "tool_use", None, "Read", "t5", '{"path":"z"}', None, None),
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
            limit=4,
            sample_limit=2,
            json=False,
        )
    )

    assert report["index_schema_version"] == 22
    assert report["sample_frame"] == {
        "classification_scope": "immediately following assistant message only",
        "complete_failure_frame": True,
        "failure_predicate": "tool_result_is_error = 1 OR tool_result_exit_code != 0",
        "inspected_structured_failures": 4,
        "limit": 4,
        "time_window": "entire archive (no since/until filter)",
        "sampled_by_origin": [
            {
                "inspected_structured_failures": 2,
                "origin": "claude-code-session",
                "requested_limit": 2,
                "total_structured_failures": 2,
            },
            {
                "inspected_structured_failures": 2,
                "origin": "codex-session",
                "requested_limit": 2,
                "total_structured_failures": 2,
            },
        ],
        "selection_order": "origin, session_id, tool_id, tool_result_message_id",
        "selection_strategy": (
            "origin-stratified bounded sample; at least one row per origin when limit allows, "
            "then proportional fill by origin failure count; each origin candidate frame is bounded "
            "before pairing to tool-use rows"
        ),
        "total_by_origin": [
            {"failed_outcomes": 2, "origin": "claude-code-session"},
            {"failed_outcomes": 2, "origin": "codex-session"},
        ],
        "total_structured_failures": 4,
        "unpaired_structured_failures": 0,
    }
    assert report["totals"] == {
        "failed_outcomes": 4,
        "acknowledged": 1,
        "silent_proceed": 1,
        "ambiguous": 2,
        "ambiguous_wordless_continuation": 1,
        "ambiguous_prose_no_marker": 1,
        "classified_outcomes": 2,
    }
    assert report["rates"]["silent_rate_lower_bound"] == 1 / 4
    assert set(report["samples_by_origin_classification"]) == {"claude-code-session", "codex-session"}
    assert report["samples_by_origin_classification"]["codex-session"]["ambiguous"][0]["origin"] == "codex-session"
    codex_ambiguous = report["samples_by_origin_classification"]["codex-session"]["ambiguous"]
    assert {sample["classification_reason"] for sample in codex_ambiguous} == {
        "prose_no_marker",
        "wordless_tool_continuation",
    }
    assert any(sample["next_has_tool_use"] for sample in codex_ambiguous)
    assert (
        report["samples_by_origin_classification"]["claude-code-session"]["acknowledged"][0]["next_text_preview"]
        == "The command failed with exit code 2, so I will fix it."
    )
    assert (
        report["samples_by_origin_classification"]["claude-code-session"]["acknowledged"][0]["classification_reason"]
        == "explicit_acknowledgment_marker"
    )
    assert report["samples_by_origin_classification"]["claude-code-session"]["acknowledged"][0]["matched_marker"] == (
        "failed"
    )
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["claim"]
    assert summary["non_claim"]
    assert summary["proof_report"]["failed_outcomes"] == 4
    assert summary["proof_report"]["complete_failure_frame"] is True
    assert summary["proof_report"]["ambiguous_wordless_continuation"] == 1
    assert summary["proof_report"]["ambiguous_prose_no_marker"] == 1
    assert summary["proof_report"]["time_window"] == "entire archive (no since/until filter)"
    assert summary["proof_report"]["sampled_by_origin"] == [
        {
            "inspected_structured_failures": 2,
            "origin": "claude-code-session",
            "requested_limit": 2,
            "total_structured_failures": 2,
        },
        {
            "inspected_structured_failures": 2,
            "origin": "codex-session",
            "requested_limit": 2,
            "total_structured_failures": 2,
        },
    ]
    assert (out_dir / "claim-vs-evidence.report.json").exists()
    readme = (out_dir / "README.md").read_text()
    assert "Claim-vs-Evidence" in readme
    assert "- time window: entire archive (no since/until filter)" in readme
    assert "- claude-code-session: inspected 2 / 2 structured failures (requested 2)" in readme
    assert "- codex-session: inspected 2 / 2 structured failures (requested 2)" in readme


def test_claim_vs_evidence_bounded_sample_is_origin_stratified(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    _seed_archive(archive)

    report = build_report(
        argparse.Namespace(
            archive_root=archive,
            out_dir=None,
            limit=2,
            sample_limit=2,
            json=False,
        )
    )

    assert report["sample_frame"]["complete_failure_frame"] is False
    assert report["sample_frame"]["sampled_by_origin"] == [
        {
            "inspected_structured_failures": 1,
            "origin": "claude-code-session",
            "requested_limit": 1,
            "total_structured_failures": 2,
        },
        {
            "inspected_structured_failures": 1,
            "origin": "codex-session",
            "requested_limit": 1,
            "total_structured_failures": 2,
        },
    ]
    assert {row["name"] for row in report["by_origin"]} == {"claude-code-session", "codex-session"}


def test_claim_vs_evidence_keeps_same_message_tool_result_identities(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    _seed_archive(archive)
    conn = sqlite3.connect(archive / "index.db")
    conn.executemany(
        """
        INSERT INTO blocks(
            message_id, session_id, position, block_type, text, tool_name, tool_id,
            tool_input, tool_result_is_error, tool_result_exit_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("tool-missing-next", "s2", 2, "tool_use", None, "Read", "t6", '{"path":"y"}', None, None),
            ("tool-missing-next", "s2", 3, "tool_result", "also nope", None, "t6", None, 0, 1),
        ],
    )
    conn.commit()
    conn.close()

    report = build_report(
        argparse.Namespace(
            archive_root=archive,
            out_dir=None,
            limit=10,
            sample_limit=10,
            json=False,
        )
    )

    assert report["sample_frame"]["total_structured_failures"] == 5
    assert {row["origin"]: row["failed_outcomes"] for row in report["sample_frame"]["total_by_origin"]} == {
        "claude-code-session": 2,
        "codex-session": 3,
    }
    codex_samples = report["samples_by_origin_classification"]["codex-session"]["ambiguous"]
    assert len(codex_samples) == 3
    assert {sample["tool_result_tool_id"] for sample in codex_samples} == {"t3", "t4", "t6"}
    assert {sample["tool_result_message_ref"] for sample in codex_samples} == {
        "message:tool-missing-next",
        "message:tool-wordless",
    }
