from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from devtools import affordance_usage


def _make_index_db(root: Path) -> Path:
    root.mkdir()
    db = root / "index.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            PRAGMA user_version = 18;
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                origin TEXT,
                title TEXT,
                sort_key_ms INTEGER
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT,
                occurred_at_ms INTEGER
            );
            CREATE TABLE blocks (
                session_id TEXT,
                message_id TEXT,
                block_type TEXT,
                tool_name TEXT,
                tool_id TEXT,
                tool_command TEXT,
                tool_path TEXT,
                tool_input TEXT,
                tool_result_is_error INTEGER,
                tool_result_exit_code INTEGER
            );
            CREATE INDEX idx_blocks_session_position ON blocks(session_id, message_id);
            CREATE VIEW actions AS
            SELECT
                u.session_id,
                u.message_id,
                NULL AS tool_use_block_id,
                u.tool_name,
                NULL AS semantic_type,
                u.tool_command,
                u.tool_path,
                u.tool_input,
                NULL AS output_text,
                r.tool_result_is_error AS is_error,
                r.tool_result_exit_code AS exit_code,
                NULL AS tool_result_block_id
            FROM blocks u
            LEFT JOIN blocks r
                ON r.tool_id = u.tool_id
               AND r.session_id = u.session_id
               AND r.block_type = 'tool_result'
            WHERE u.block_type = 'tool_use';
            INSERT INTO sessions VALUES
                ('s1', 'codex-session', 'Codex work', 4102444800000),
                ('s2', 'claude-code-session', 'Claude work', 4102444800000);
            INSERT INTO messages VALUES
                ('m1', 's1', 4102444800000),
                ('m2', 's2', 4102444800000),
                ('m3', 's2', 0);
            INSERT INTO blocks VALUES
                ('s1', 'm1', 'tool_use', 'mcp__serena__find_symbol', 't1', '', '/repo/a.py', '', NULL, NULL),
                ('s1', 'm1', 'tool_use', 'mcp__context7__query-docs', 't2', '', '', 'react', NULL, NULL),
                ('s2', 'm2', 'tool_use', 'mcp__plugin_context7_context7__query-docs', 't3', '', '', 'sqlite', NULL, NULL),
                ('s2', 'm2', 'tool_result', NULL, 't3', '', '', '', 1, NULL),
                ('s2', 'm3', 'tool_use', 'mcp__cclsp__find_definition', 't4', '', '/repo/lib.rs', '', NULL, NULL),
                ('s1', 'm1', 'tool_use', 'functions.exec_command', 't5', 'codebase-memory-mcp cli search_code', '', '', NULL, NULL),
                ('s1', 'm1', 'tool_use', 'functions.exec_command', 't6', 'codebase-memory-mcp cli search code', '', '', NULL, NULL),
                ('s1', 'm1', 'tool_use', 'search_code', 't7', '', '', 'search_code query', NULL, NULL);
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db


def test_affordance_usage_report_and_files(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _make_index_db(archive_root)
    out_dir = tmp_path / "out"
    args = affordance_usage.AffordanceUsageArgs(
        archive_root=archive_root,
        out_dir=out_dir,
        days=36500,
        family=("serena", "context7", "cclsp"),
        detail_pattern=(),
        sample_limit=10,
        json=True,
        all_time=False,
    )

    report = affordance_usage.build_report(args)

    assert report["archive_root"] == str(archive_root.resolve())
    assert report["index_schema_version"] == 18
    families = {row["family"]: row for row in report["family_counts"]}
    assert families["context7"]["actions"] == 2
    assert families["context7"]["errors"] == 1
    assert families["serena"]["actions"] == 1
    tool_counts = {row["tool_name"]: row for row in report["tool_counts"]}
    assert tool_counts["context7/query-docs"]["actions"] == 2
    assert tool_counts["context7/query-docs"]["raw_tool_name_count"] == 2
    evidence = {(row["family"], row["evidence_kind"]): row for row in report["evidence_kind_counts"]}
    assert evidence[("serena", "mcp_tool_call")]["actions"] == 1
    assert evidence[("context7", "mcp_tool_call")]["actions"] == 2
    assert report["recent_tool_counts"][0]["family"] in {"context7", "serena", "cclsp"}

    with (out_dir / "family-counts.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["family"] for row in rows} == {"cclsp", "context7", "serena"}
    with (out_dir / "evidence-kind-counts.csv").open(encoding="utf-8", newline="") as handle:
        evidence_rows = list(csv.DictReader(handle))
    assert {row["evidence_kind"] for row in evidence_rows} == {"mcp_tool_call"}
    written_report = json.loads((out_dir / "affordance-usage.report.json").read_text(encoding="utf-8"))
    assert written_report["family_counts"] == report["family_counts"]
    assert "recent" in (out_dir / "README.md").read_text(encoding="utf-8").lower()


def test_affordance_usage_rejects_nonpositive_recent_window(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _make_index_db(archive_root)
    args = affordance_usage.AffordanceUsageArgs(
        archive_root=archive_root,
        out_dir=None,
        days=0,
        family=("serena",),
        detail_pattern=(),
        sample_limit=10,
        json=True,
        all_time=False,
    )

    try:
        affordance_usage.build_report(args)
    except ValueError as exc:
        assert "--days must be positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_affordance_usage_can_match_shell_command_details(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _make_index_db(archive_root)
    args = affordance_usage.AffordanceUsageArgs(
        archive_root=archive_root,
        out_dir=None,
        days=36500,
        family=(),
        detail_pattern=("codebase-memory",),
        sample_limit=10,
        json=True,
        all_time=False,
    )

    report = affordance_usage.build_report(args)

    families = {row["family"]: row for row in report["family_counts"]}
    assert families["codebase-memory"]["actions"] == 2
    assert report["tool_counts"][0]["tool_name"] == "codebase-memory/command-detail"
    assert report["tool_counts"][0]["raw_tool_names"] == "functions.exec_command"
    assert report["tool_counts"][0]["evidence_kind"] == "command_detail"
    assert report["samples"][0]["matched_by"] == "detail"
    assert report["samples"][0]["normalized_tool"] == "codebase-memory/command-detail"
    assert report["detail_patterns"] == ["codebase-memory"]


def test_affordance_usage_treats_like_wildcards_as_literals(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _make_index_db(archive_root)
    args = affordance_usage.AffordanceUsageArgs(
        archive_root=archive_root,
        out_dir=None,
        days=36500,
        family=(),
        detail_pattern=("search_code",),
        sample_limit=10,
        json=True,
        all_time=False,
    )

    report = affordance_usage.build_report(args)

    assert report["family_counts"][0]["actions"] == 2
    tool_counts = {row["tool_name"]: row for row in report["tool_counts"]}
    assert tool_counts["codebase-memory/search_code"]["raw_tool_names"] == "search_code"
    assert tool_counts["codebase-memory/command-detail"]["raw_tool_names"] == "functions.exec_command"
    assert any("search_code" in str(row["detail"]) for row in report["samples"])
    assert all("search code" not in str(row["detail"]) for row in report["samples"])
