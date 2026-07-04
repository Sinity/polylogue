from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from devtools import lineage_validation
from devtools.command_catalog import COMMANDS


def _make_index_db(root: Path, *, with_gap: bool = False) -> Path:
    root.mkdir()
    db = root / "index.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            PRAGMA user_version = 24;
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                native_id TEXT,
                origin TEXT,
                title TEXT,
                session_kind TEXT DEFAULT 'standard',
                active_leaf_message_id TEXT,
                parent_session_id TEXT,
                root_session_id TEXT,
                branch_type TEXT,
                title_source TEXT,
                instructions_text TEXT,
                created_at_ms INTEGER,
                updated_at_ms INTEGER,
                git_branch TEXT,
                git_repository_url TEXT,
                provider_project_ref TEXT,
                message_count INTEGER DEFAULT 0
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                logical_session_id TEXT
            );
            CREATE TABLE session_links (
                src_session_id TEXT,
                dst_origin TEXT,
                dst_native_id TEXT,
                link_type TEXT,
                status TEXT,
                resolved_dst_session_id TEXT,
                branch_point_message_id TEXT,
                inheritance TEXT
            );
            CREATE TABLE session_working_dirs (
                session_id TEXT,
                position INTEGER,
                path TEXT
            );
            CREATE TABLE attachments (
                attachment_id TEXT,
                display_name TEXT,
                media_type TEXT,
                byte_count INTEGER
            );
            CREATE TABLE attachment_refs (
                session_id TEXT,
                message_id TEXT,
                attachment_id TEXT,
                upload_origin TEXT,
                source_url TEXT,
                caption TEXT
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT,
                native_id TEXT,
                role TEXT,
                position INTEGER,
                variant_index INTEGER DEFAULT 0,
                is_active_path INTEGER DEFAULT 1,
                is_active_leaf INTEGER DEFAULT 0,
                message_type TEXT DEFAULT 'message',
                material_origin TEXT DEFAULT 'human_authored',
                word_count INTEGER DEFAULT 1,
                has_tool_use INTEGER DEFAULT 0,
                has_thinking INTEGER DEFAULT 0,
                has_paste INTEGER DEFAULT 0,
                occurred_at_ms INTEGER,
                paste_boundary TEXT,
                duration_ms INTEGER,
                parent_message_id TEXT
            );
            CREATE TABLE blocks (
                block_id TEXT,
                message_id TEXT,
                block_type TEXT,
                text TEXT,
                tool_name TEXT,
                tool_id TEXT,
                semantic_type TEXT,
                tool_input TEXT,
                language TEXT,
                tool_result_is_error INTEGER,
                tool_result_exit_code INTEGER,
                position INTEGER
            );
            INSERT INTO sessions(session_id, native_id, origin, title, root_session_id, branch_type, message_count)
            VALUES
                ('parent', 'parent-native', 'codex-session', 'Parent', 'parent', NULL, 2),
                ('child', 'child-native', 'codex-session', 'Child', 'parent', 'continuation', 1),
                ('fresh', 'fresh-native', 'claude-code-session', 'Fresh', 'fresh', 'subagent', 1);
            INSERT INTO session_profiles VALUES
                ('parent', 'parent'),
                ('child', 'parent'),
                ('fresh', 'fresh');
            INSERT INTO messages(message_id, session_id, native_id, role, position)
            VALUES
                ('p1', 'parent', 'p1', 'user', 0),
                ('p2', 'parent', 'p2', 'assistant', 1),
                ('c3', 'child', 'c3', 'assistant', 2),
                ('f1', 'fresh', 'f1', 'assistant', 0);
            INSERT INTO blocks(block_id, message_id, block_type, text, position)
            VALUES
                ('bp1', 'p1', 'text', 'parent one', 0),
                ('bp2', 'p2', 'text', 'parent two', 0),
                ('bc3', 'c3', 'text', 'child tail', 0),
                ('bf1', 'f1', 'text', 'fresh', 0);
            INSERT INTO session_links VALUES
                ('child', 'codex-session', 'parent-native', 'continuation', 'resolved', 'parent', 'p2', 'prefix-sharing'),
                ('fresh', 'claude-code-session', 'parent-native', 'subagent', 'resolved', 'parent', NULL, 'spawned-fresh');
            """
        )
        if with_gap:
            conn.executescript(
                """
                DELETE FROM session_profiles WHERE session_id = 'fresh';
                UPDATE session_links
                SET branch_point_message_id = 'missing-message'
                WHERE src_session_id = 'child';
                """
            )
        conn.commit()
    finally:
        conn.close()
    return db


def _args(archive_root: Path, out_dir: Path | None = None) -> lineage_validation.LineageValidationArgs:
    return lineage_validation.LineageValidationArgs(
        archive_root=archive_root,
        out_dir=out_dir,
        sample_prefix_sharing=10,
        max_sample_stored_messages=500,
        json=True,
    )


def test_lineage_validation_clean_archive_is_citable(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _make_index_db(archive_root)

    report = lineage_validation.build_report(_args(archive_root))

    assert report["index_schema_version"] == 24
    assert report["counts"]["physical_sessions"] == 3
    assert report["counts"]["logical_sessions"] == 2
    assert report["counts"]["stored_messages"] == 4
    assert report["counts"]["missing_session_profile_rows"] == 0
    assert report["verdict"] == {"external_counts_citable": True, "reasons": []}
    sample = report["lineage"]["prefix_sharing_read_sample"]
    assert sample["sampled"] == 1
    assert sample["stored_messages"] == 1
    assert sample["composed_messages"] == 3
    assert sample["rows"][0]["served_exceeds_stored"] is True


def test_lineage_validation_reports_integrity_gaps(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _make_index_db(archive_root, with_gap=True)

    report = lineage_validation.build_report(_args(archive_root))

    assert report["verdict"]["external_counts_citable"] is False
    assert "1 sessions have no session_profiles row" in report["verdict"]["reasons"]
    assert "1 prefix-sharing branch points do not resolve to messages" in report["verdict"]["reasons"]
    assert report["lineage"]["integrity"]["dangling_branch_points"] == 1
    assert report["lineage"]["missing_profile_samples"][0]["session_id"] == "fresh"
    sample = report["lineage"]["prefix_sharing_read_sample"]
    assert sample["rows"][0]["composed_messages"] == 1
    assert sample["rows"][0]["served_exceeds_stored"] is False


def test_lineage_validation_writes_demo_artifacts(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _make_index_db(archive_root)
    out_dir = tmp_path / "out"

    report = lineage_validation.build_report(_args(archive_root, out_dir))

    written = json.loads((out_dir / "lineage-validation.report.json").read_text(encoding="utf-8"))
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert written["counts"] == report["counts"]
    assert summary["artifact"] == "lineage-validation"
    assert summary["proof_report"]["external_counts_citable"] is True
    assert "external counts citable: `true`" in readme


def test_lineage_validation_command_registered() -> None:
    spec = COMMANDS["workspace lineage-validation"]
    assert spec.module == "devtools.lineage_validation"


def test_lineage_validation_main_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive_root = tmp_path / "archive"
    _make_index_db(archive_root)

    exit_code = lineage_validation.main(["--archive-root", str(archive_root), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "devtools workspace lineage-validation"
