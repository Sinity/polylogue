from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from devtools import lineage_validation
from devtools.command_catalog import COMMANDS
from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.infra.frozen_clock import FrozenClock


def _make_index_db(root: Path, *, with_gap: bool = False, with_unresolved: bool = False) -> Path:
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
                title_ref TEXT,
                title_confidence REAL,
                instructions_text TEXT,
                created_at_ms INTEGER,
                updated_at_ms INTEGER,
                git_branch TEXT,
                git_repository_url TEXT,
                provider_project_ref TEXT,
                message_count INTEGER DEFAULT 0,
                reported_cost_usd REAL
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
                method TEXT,
                evidence_json TEXT,
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
                parent_message_id TEXT,
                stop_reason TEXT
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
                tool_result_outcome_unknown_reason TEXT,
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
                ('child', 'codex-session', 'parent-native', 'continuation', NULL, 'parent', 'parser-parent', '{}', 'p2', 'prefix-sharing'),
                ('fresh', 'claude-code-session', 'parent-native', 'subagent', NULL, 'parent', 'parent-tool-use-id', '{}', NULL, 'spawned-fresh');
            """
        )
        if with_unresolved:
            conn.executescript(
                """
                INSERT INTO sessions(session_id, native_id, origin, title, root_session_id, branch_type, message_count)
                VALUES ('orphan', 'orphan-native', 'codex-session', 'Orphan', 'orphan', 'continuation', 1);
                INSERT INTO session_profiles VALUES ('orphan', 'orphan');
                INSERT INTO messages(message_id, session_id, native_id, role, position)
                VALUES ('o1', 'orphan', 'o1', 'user', 0);
                INSERT INTO blocks(block_id, message_id, block_type, text, position)
                VALUES ('bo1', 'o1', 'text', 'orphan', 0);
                INSERT INTO session_links
                    (src_session_id, dst_origin, dst_native_id, link_type, status,
                     resolved_dst_session_id, method, evidence_json, branch_point_message_id, inheritance)
                VALUES ('orphan', 'codex-session', 'missing-parent', 'continuation', NULL,
                        NULL, 'parser-parent', '{}', NULL, 'spawned-fresh');
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


def _args(
    archive_root: Path,
    out_dir: Path | None = None,
    *,
    index_db: Path | None = None,
) -> lineage_validation.LineageValidationArgs:
    return lineage_validation.LineageValidationArgs(
        archive_root=archive_root,
        out_dir=out_dir,
        sample_prefix_sharing=10,
        max_sample_stored_messages=500,
        json=True,
        index_db=index_db,
    )


def _writer_message(provider_id: str, text: str, position: int, role: Role = Role.USER) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_id,
        role=role,
        text=text,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _make_writer_candidate(root: Path) -> Path:
    root.mkdir()
    db = root / "index.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    try:
        write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="parent",
                title="parent",
                messages=[
                    _writer_message("p0", "hello", 0),
                    _writer_message("p1", "world", 1, Role.ASSISTANT),
                ],
            ),
        )
        for provider_id, tail_text in (("child", "child tail"), ("sibling", "sibling tail")):
            write_parsed_session_to_archive(
                conn,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=provider_id,
                    title=provider_id,
                    parent_session_provider_id="parent",
                    branch_type=BranchType.FORK,
                    messages=[
                        _writer_message(f"{provider_id}-p0", "hello", 0),
                        _writer_message(f"{provider_id}-p1", "world", 1, Role.ASSISTANT),
                        _writer_message(f"{provider_id}-tail", tail_text, 2),
                    ],
                ),
            )
        write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="orphan",
                title="orphan",
                parent_session_provider_id="missing-parent",
                messages=[_writer_message("orphan-0", "orphan", 0)],
            ),
        )
    finally:
        conn.close()
    return db


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
    topology = report["lineage"]["topology"]
    assert topology["empty_effective_status_count"] == 0
    assert topology["empty_method_count"] == 0
    assert topology["effective_status_counts"] == {"resolved": 2}
    assert topology["raw_status_empty_count"] == 2
    assert lineage_validation._receipt_sha256(report) == report["receipt_sha256"]


def test_lineage_validation_proves_unresolved_reads_stay_child_local(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _make_index_db(archive_root, with_unresolved=True)

    report = lineage_validation.build_report(_args(archive_root))

    topology = report["lineage"]["topology"]
    assert topology["effective_status_counts"] == {"resolved": 2, "unresolved": 1}
    sample = topology["unresolved_read_sample"]
    assert sample["safe"] is True
    assert sample["sampled"] == 1
    assert sample["rows"][0]["read_status"] == "safe"
    assert report["verdict"]["external_counts_citable"] is True


def test_lineage_validation_proves_writer_candidate_and_snapshot_identity(tmp_path: Path) -> None:
    archive_root = tmp_path / "candidate"
    db = _make_writer_candidate(archive_root)

    report = lineage_validation.build_report(_args(archive_root, index_db=db))

    topology = report["lineage"]["topology"]
    assert topology["effective_status_counts"] == {"resolved": 2, "unresolved": 1}
    assert topology["empty_effective_status_count"] == 0
    assert topology["empty_method_count"] == 0
    assert topology["raw_status_empty_count"] == 3
    assert topology["method_counts"] == {"parser-parent": 3}
    assert topology["unresolved_read_sample"]["status"] == "safe"
    assert topology["unresolved_read_sample"]["sampled"] == 1
    assert report["index_db"] == str(db.resolve())
    assert report["snapshot_identity"]["stable"] is True
    assert report["snapshot_identity"]["before"]["sha256"] == report["snapshot_identity"]["after"]["sha256"]


def test_lineage_validation_rejects_unobserved_unresolved_reader_sample(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _make_index_db(archive_root, with_unresolved=True)
    args = lineage_validation.LineageValidationArgs(
        archive_root=archive_root,
        out_dir=None,
        sample_prefix_sharing=10,
        max_sample_stored_messages=500,
        json=True,
        sample_unresolved=0,
    )

    report = lineage_validation.build_report(args)

    sample = report["lineage"]["topology"]["unresolved_read_sample"]
    assert sample["status"] == "not_observed"
    assert sample["safe"] is False
    assert report["verdict"]["external_counts_citable"] is False
    assert "1 unresolved-parent links were not exercised through the reader" in report["verdict"]["reasons"]


@pytest.mark.frozen_clock_modules("devtools.lineage_validation")
def test_lineage_validation_receipt_reproduces_before_binding_mutation(
    tmp_path: Path, frozen_clock: FrozenClock
) -> None:
    configured_root = tmp_path / "configured"
    candidate_root = tmp_path / "candidate"
    _make_index_db(configured_root)
    candidate_db = _make_index_db(candidate_root)

    first = lineage_validation.build_report(_args(configured_root, index_db=candidate_db))
    assert first["index_db"] == str(candidate_db.resolve())
    unchanged = lineage_validation.build_report(_args(configured_root, index_db=candidate_db))
    assert unchanged["captured_at"] == first["captured_at"] == frozen_clock.now().isoformat()
    assert unchanged["snapshot_identity"] == first["snapshot_identity"]
    assert unchanged["receipt_sha256"] == first["receipt_sha256"]

    with sqlite3.connect(candidate_db) as conn:
        conn.execute("UPDATE session_links SET method = 'changed' WHERE src_session_id = 'child'")
        conn.commit()
    second = lineage_validation.build_report(_args(configured_root, index_db=candidate_db))

    assert second["index_db"] == str(candidate_db.resolve())
    assert second["snapshot_identity"]["before"]["sha256"] != first["snapshot_identity"]["before"]["sha256"]
    assert second["receipt_sha256"] != first["receipt_sha256"]


def test_lineage_validation_snapshot_is_stable_for_a_quiescent_wal_database(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    db = _make_index_db(archive_root)
    with sqlite3.connect(db) as writer:
        assert writer.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        writer.execute("UPDATE session_links SET method = 'wal-proof' WHERE src_session_id = 'child'")
        writer.commit()

        report = lineage_validation.build_report(_args(archive_root))

    assert report["snapshot_identity"]["stable"] is True
    assert report["verdict"]["external_counts_citable"] is True


def test_lineage_validation_unchecked_census_has_checked_schema(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    db = _make_index_db(archive_root)
    with sqlite3.connect(db) as checked_conn:
        checked = lineage_validation.census_topology_links(checked_conn, sample_unresolved=0)

    missing_db = tmp_path / "missing.db"
    with sqlite3.connect(missing_db) as missing_conn:
        missing_conn.execute("CREATE TABLE session_links (src_session_id TEXT)")
        unchecked = lineage_validation.census_topology_links(missing_conn, sample_unresolved=0)

    assert unchecked["checked"] is False
    assert set(unchecked) == set(checked)


def test_lineage_validation_catches_empty_method_mutation(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    db = _make_index_db(archive_root)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE session_links SET method = '' WHERE src_session_id = 'child'")
        conn.commit()

    report = lineage_validation.build_report(_args(archive_root))

    topology = report["lineage"]["topology"]
    assert topology["empty_method_count"] == 1
    assert report["verdict"]["external_counts_citable"] is False
    assert "1 topology links have an empty method" in report["verdict"]["reasons"]


def test_lineage_validation_catches_unknown_status_and_unsafe_reader_mutation(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    db = _make_index_db(archive_root, with_unresolved=True)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE session_links SET status = 'made-up' WHERE src_session_id = 'child'")
        conn.execute("UPDATE sessions SET parent_session_id = 'parent' WHERE session_id = 'orphan'")
        conn.commit()

    report = lineage_validation.build_report(_args(archive_root))

    topology = report["lineage"]["topology"]
    assert topology["unknown_effective_status_count"] == 1
    assert topology["unresolved_read_sample"]["safe"] is False
    assert report["verdict"]["external_counts_citable"] is False
    assert any("unknown effective states" in reason for reason in report["verdict"]["reasons"])
    assert "sampled unresolved-parent reads did not remain child-local" in report["verdict"]["reasons"]


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
    assert written["receipt_sha256"] == report["receipt_sha256"]
    assert lineage_validation._receipt_sha256(written) == written["receipt_sha256"]
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
