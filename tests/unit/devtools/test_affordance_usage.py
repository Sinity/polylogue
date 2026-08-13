from __future__ import annotations

import csv
import json
import os
import shutil
import sqlite3
import stat
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest

from devtools import affordance_usage
from polylogue.cli.click_app import cli
from polylogue.cli.command_inventory import iter_command_paths
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from tests.infra.mcp import EXPECTED_TOOL_NAMES


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
                semantic_type TEXT,
                tool_command TEXT,
                tool_path TEXT,
                tool_input TEXT,
                tool_result_is_error INTEGER,
                tool_result_exit_code INTEGER,
                tool_detail_text TEXT GENERATED ALWAYS AS (
                    lower(coalesce(tool_command, '') || ' ' || coalesce(tool_path, ''))
                ) VIRTUAL
            );
            CREATE INDEX idx_blocks_session_position ON blocks(session_id, message_id);
            CREATE VIRTUAL TABLE blocks_command_trigram USING fts5(
                tool_detail_text, tokenize='trigram', content='blocks', content_rowid='rowid'
            );
            CREATE TRIGGER blocks_command_trigram_ai
            AFTER INSERT ON blocks WHEN new.block_type = 'tool_use' AND new.tool_detail_text != ' ' BEGIN
                INSERT INTO blocks_command_trigram(rowid, tool_detail_text)
                VALUES (new.rowid, new.tool_detail_text);
            END;
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
                ('s1', 'm1', 'tool_use', 'mcp__serena__find_symbol', 't1', NULL, '', '/repo/a.py', '', NULL, NULL),
                ('s1', 'm1', 'tool_use', 'mcp__context7__query-docs', 't2', NULL, '', '', 'react', NULL, NULL),
                ('s2', 'm2', 'tool_use', 'mcp__plugin_context7_context7__query-docs', 't3', NULL, '', '', 'sqlite', NULL, NULL),
                ('s2', 'm2', 'tool_result', NULL, 't3', NULL, '', '', '', 1, NULL),
                ('s2', 'm3', 'tool_use', 'mcp__cclsp__find_definition', 't4', NULL, '', '/repo/lib.rs', '', NULL, NULL),
                ('s1', 'm1', 'tool_use', 'functions.exec_command', 't5', NULL, 'codebase-memory-mcp cli search_code', '', '', NULL, NULL),
                ('s1', 'm1', 'tool_use', 'functions.exec_command', 't6', NULL, 'codebase-memory-mcp cli search code', '', '', NULL, NULL),
                ('s1', 'm1', 'tool_use', 'search_code', 't7', NULL, '', '', 'search_code query', NULL, NULL),
                ('s1', 'm1', 'tool_use', 'mcp__polylogue__query', 't8', NULL, '', '', 'messages where text:timeout', NULL, NULL),
                ('s1', 'm1', 'tool_use', 'functions.exec_command', 't9', NULL, 'polylogue read session:s1 --view summary', '', '', NULL, NULL);
            """
        )
        conn.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(text)")
        conn.execute(
            """
            INSERT INTO messages_fts(rowid, text)
            SELECT rowid, lower(
                coalesce(tool_command, '') || ' ' || coalesce(tool_path, '') || ' ' || coalesce(tool_input, '')
            )
            FROM blocks
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
    assert report["evidence_root"] == str(archive_root.resolve())
    assert report["index_db"] == str((archive_root / "index.db").resolve())
    assert report["index_schema_version"] == 18
    assert report["snapshot_identity"]["stable"] is True
    assert report["snapshot_identity"]["size"] == (archive_root / "index.db").stat().st_size
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
    surface_rows = {(str(row["surface_type"]), str(row["surface_name"])): row for row in report["surface_inventory"]}
    command_paths = {command.display_name for command in iter_command_paths(cli, include_root=False)}
    assert {name for kind, name in surface_rows if kind == "mcp_tool"} == EXPECTED_TOOL_NAMES
    assert {name for kind, name in surface_rows if kind == "cli_command"} == command_paths
    assert surface_rows[("mcp_tool", "query")]["observed_actions"] == 1
    assert surface_rows[("mcp_tool", "query")]["classification"] == "keep"
    assert surface_rows[("cli_command", "read")]["observed_actions"] == 1
    assert surface_rows[("mcp_tool", "maintenance")]["operator_only_caveat"] is True
    assert surface_rows[("mcp_tool", "maintenance")]["classification"] == "keep"
    assert surface_rows[("mcp_tool", "status")]["classification"] == "kill"
    assert report["surface_inventory_summary"]

    with (out_dir / "family-counts.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["family"] for row in rows} == {"cclsp", "context7", "serena"}
    with (out_dir / "evidence-kind-counts.csv").open(encoding="utf-8", newline="") as handle:
        evidence_rows = list(csv.DictReader(handle))
    assert {row["evidence_kind"] for row in evidence_rows} == {"mcp_tool_call"}
    written_report = json.loads((out_dir / "affordance-usage.report.json").read_text(encoding="utf-8"))
    assert written_report["family_counts"] == report["family_counts"]
    written_summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert written_summary["artifact"] == "agent-affordance-usage"
    assert written_summary["evidence_root"] == report["evidence_root"]
    assert written_summary["index_db"] == report["index_db"]
    assert written_summary["snapshot_identity"] == report["snapshot_identity"]
    assert written_summary["index_schema_version"] == report["index_schema_version"]
    assert written_summary["top_families"] == report["summary"]["top_families"]
    assert written_summary["surface_inventory_summary"] == report["surface_inventory_summary"]
    assert written_summary["action_scope"] == report["action_scope"]
    with (out_dir / "surface-inventory.csv").open(encoding="utf-8", newline="") as handle:
        inventory_rows = list(csv.DictReader(handle))
    assert len(inventory_rows) == len(EXPECTED_TOOL_NAMES) + len(command_paths)
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "recent" in readme.lower()
    assert "surface inventory" in readme.lower()
    assert f"Evidence index: `{report['index_db']}`" in readme
    assert f"Evidence snapshot SHA-256: `{report['snapshot_identity']['sha256']}`" in readme
    assert "`summary.json`" in (out_dir / "README.md").read_text(encoding="utf-8")


def test_affordance_usage_selected_external_index_is_the_report_evidence_source(tmp_path: Path) -> None:
    configured_root = tmp_path / "configured"
    selected_root = tmp_path / "selected"
    _make_index_db(configured_root)
    selected_db = _make_index_db(selected_root)
    out_dir = tmp_path / "out"

    report = affordance_usage.build_report(
        affordance_usage.AffordanceUsageArgs(
            archive_root=configured_root,
            out_dir=out_dir,
            days=36500,
            family=("serena",),
            detail_pattern=(),
            sample_limit=10,
            json=True,
            all_time=False,
            index_db=selected_db,
        )
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["index_db"] == str(selected_db.resolve())
    assert report["index_db"] != str((configured_root / "index.db").resolve())
    assert report["evidence_root"] == str(selected_root.resolve())
    snapshot_identity = report["snapshot_identity"]
    assert snapshot_identity["before"]["path"] == str(selected_db.resolve())
    assert snapshot_identity["after"]["path"] == str(selected_db.resolve())
    assert snapshot_identity["before"]["index_db"] == str(selected_db.resolve())
    assert snapshot_identity["after"]["index_db"] == str(selected_db.resolve())
    assert snapshot_identity["before"]["sha256"] == snapshot_identity["after"]["sha256"]
    assert snapshot_identity["size"] == selected_db.stat().st_size
    assert summary["index_db"] == str(selected_db.resolve())
    assert summary["evidence_root"] == str(selected_root.resolve())
    assert summary["snapshot_identity"] == snapshot_identity


def test_affordance_usage_selected_sibling_index_bypasses_archive_store_fast_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    configured_db = _make_index_db(archive_root)
    selected_db = archive_root / "candidate.db"
    shutil.copy2(configured_db, selected_db)
    calls = 0

    class DivergentArchive(AbstractContextManager["DivergentArchive"]):
        def __enter__(self) -> DivergentArchive:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def list_tool_action_evidence_count_rows(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
            nonlocal calls
            del args, kwargs
            calls += 1
            return [
                {
                    "source_name": "wrong-index",
                    "origin": "codex-session",
                    "normalized_tool_name": "codebase-memory/command-detail",
                    "action_kind": "shell",
                    "evidence_kind": "command_detail",
                    "matched_by": "detail",
                    "call_count": 999,
                    "session_count": 1,
                    "error_count": 0,
                    "nonzero_exit_count": 0,
                }
            ]

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        lambda _root, **_kwargs: DivergentArchive(),
    )

    report = affordance_usage.build_report(
        affordance_usage.AffordanceUsageArgs(
            archive_root=archive_root,
            out_dir=None,
            days=36500,
            family=(),
            detail_pattern=("codebase-memory",),
            sample_limit=10,
            json=True,
            all_time=False,
            index_db=selected_db,
        )
    )

    assert calls == 0
    assert report["index_db"] == str(selected_db.resolve())
    assert {row["family"]: row["actions"] for row in report["family_counts"]}["codebase-memory"] == 2
    assert report["samples"]


def test_affordance_usage_rejects_product_fast_path_on_different_physical_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    selected_db = _make_index_db(archive_root)
    other_db = _make_index_db(tmp_path / "other")

    class DivergentArchive(AbstractContextManager["DivergentArchive"]):
        index_db_path = other_db

        def __enter__(self) -> DivergentArchive:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        lambda _root, **_kwargs: DivergentArchive(),
    )

    with pytest.raises(RuntimeError, match="different physical index"):
        affordance_usage.build_report(
            affordance_usage.AffordanceUsageArgs(
                archive_root=archive_root,
                out_dir=None,
                days=36500,
                family=(),
                detail_pattern=("codebase-memory",),
                sample_limit=10,
                json=True,
                all_time=False,
                index_db=selected_db,
            )
        )


def test_affordance_usage_product_fast_path_stays_pinned_across_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    old_db = _make_index_db(tmp_path / "old-generation")
    new_db = _make_index_db(tmp_path / "new-generation")
    with sqlite3.connect(new_db) as conn:
        conn.execute(
            "INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "s1",
                "m1",
                "tool_use",
                "functions.exec_command",
                "promoted-extra",
                None,
                "codebase-memory extra",
                "",
                "",
                None,
                None,
            ),
        )
    active = archive_root / "index.db"
    active.symlink_to(old_db)

    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    real_open_existing = ArchiveStore.open_existing
    promoted = False
    opened_index_path: Path | None = None

    def promote_before_open(
        root: Path,
        *,
        read_only: bool = True,
        read_timeout: float = 5.0,
        index_path: Path | None = None,
        opened_main_fd: int | None = None,
    ) -> ArchiveStore:
        nonlocal opened_index_path, promoted
        active.unlink()
        active.symlink_to(new_db)
        promoted = True
        opened_index_path = index_path
        return real_open_existing(
            root,
            read_only=read_only,
            read_timeout=read_timeout,
            index_path=index_path,
            opened_main_fd=opened_main_fd,
        )

    monkeypatch.setattr(ArchiveStore, "open_existing", promote_before_open)

    report = affordance_usage.build_report(
        affordance_usage.AffordanceUsageArgs(
            archive_root=archive_root,
            out_dir=None,
            days=36500,
            family=(),
            detail_pattern=("codebase-memory",),
            sample_limit=10,
            json=True,
            all_time=False,
        )
    )

    assert promoted is True
    assert opened_index_path == old_db.resolve()
    assert report["index_db"] == str(old_db.resolve())
    assert report["snapshot_identity"]["sha256"] == affordance_usage._snapshot_observation(old_db)["sha256"]
    assert {row["family"]: row["actions"] for row in report["family_counts"]}["codebase-memory"] == 2


def test_affordance_usage_product_fast_path_consumes_opened_inode_after_selected_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The product route must not reopen a selected path after its reader is pinned."""
    archive_root = tmp_path / "archive"
    selected_db = _make_index_db(archive_root)
    replacement_root = tmp_path / "replacement"
    saved_db = tmp_path / "saved-index.db"
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    real_open_existing = ArchiveStore.open_existing
    replaced = False

    def replace_selected_path(
        root: Path,
        *,
        read_only: bool = True,
        read_timeout: float = 5.0,
        index_path: Path | None = None,
        opened_main_fd: int | None = None,
    ) -> Any:
        nonlocal replaced
        replacement_db = _make_index_db(replacement_root)
        with sqlite3.connect(replacement_db) as replacement:
            replacement.execute("DELETE FROM blocks")
            replacement.commit()
        selected_db.rename(saved_db)
        replacement_db.rename(selected_db)
        try:
            archive = real_open_existing(
                root,
                read_only=read_only,
                read_timeout=read_timeout,
                index_path=index_path,
                opened_main_fd=opened_main_fd,
            )
        finally:
            selected_db.unlink()
            saved_db.rename(selected_db)
        replaced = True
        return archive

    monkeypatch.setattr(ArchiveStore, "open_existing", replace_selected_path)
    report = affordance_usage.build_report(
        affordance_usage.AffordanceUsageArgs(
            archive_root=archive_root,
            out_dir=None,
            days=36500,
            family=(),
            detail_pattern=("codebase-memory",),
            sample_limit=10,
            json=True,
            all_time=False,
            index_db=selected_db,
        )
    )

    assert replaced is True
    assert {row["family"]: row["actions"] for row in report["family_counts"]}["codebase-memory"] == 2
    assert report["snapshot_identity"]["stable"] is True


def test_affordance_usage_marks_selected_index_snapshot_unstable_after_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = tmp_path / "archive"
    selected_db = _make_index_db(archive_root)
    writer = sqlite3.connect(selected_db)
    assert writer.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
    writer.execute("PRAGMA wal_autocheckpoint = 0")
    real_observation = affordance_usage._snapshot_observation
    calls = 0

    def observe_with_change(
        path: Path,
        *,
        opened_main_fd: int | None = None,
        opened_sidecar_fds: dict[str, int] | None = None,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            writer.execute("INSERT INTO sessions VALUES ('concurrent', 'codex-session', 'change', 1)")
            writer.commit()
        return real_observation(
            path,
            opened_main_fd=opened_main_fd,
            opened_sidecar_fds=opened_sidecar_fds,
        )

    monkeypatch.setattr(affordance_usage, "_snapshot_observation", observe_with_change)
    try:
        report = affordance_usage.build_report(
            affordance_usage.AffordanceUsageArgs(
                archive_root=archive_root,
                out_dir=None,
                days=36500,
                family=("serena",),
                detail_pattern=(),
                sample_limit=10,
                json=True,
                all_time=False,
                index_db=selected_db,
            )
        )
    finally:
        writer.close()

    identity = report["snapshot_identity"]
    assert identity["stable"] is False
    assert identity["before"]["sha256"] != identity["after"]["sha256"]
    assert identity["file_set_stable"] is False
    assert identity["no_concurrent_commits"] is False


def test_affordance_usage_rejects_unlinked_selected_index_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An open SQLite handle does not make an unlinked evidence path citable."""
    archive_root = tmp_path / "archive"
    selected_db = _make_index_db(archive_root)
    real_observation = affordance_usage._snapshot_observation
    unlinked = False

    def unlink_before_observation(
        path: Path,
        *,
        opened_main_fd: int | None = None,
        opened_sidecar_fds: dict[str, int] | None = None,
    ) -> dict[str, object]:
        nonlocal unlinked
        if not unlinked:
            path.unlink()
            unlinked = True
        return real_observation(
            path,
            opened_main_fd=opened_main_fd,
            opened_sidecar_fds=opened_sidecar_fds,
        )

    monkeypatch.setattr(affordance_usage, "_snapshot_observation", unlink_before_observation)
    report = affordance_usage.build_report(
        affordance_usage.AffordanceUsageArgs(
            archive_root=archive_root,
            out_dir=None,
            days=36500,
            family=("serena",),
            detail_pattern=("codebase-memory",),
            sample_limit=10,
            json=True,
            all_time=False,
            index_db=selected_db,
        )
    )

    identity = report["snapshot_identity"]
    assert report["index_schema_version"] == 18
    assert identity["before"]["present"] is False
    assert identity["after"]["present"] is False
    assert identity["before"]["observation_complete"] is False
    assert identity["after"]["observation_complete"] is False
    assert identity["before"]["files"][0]["sha256"]
    assert identity["stable"] is False


def test_affordance_usage_rejects_selected_index_replacement_after_reader_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = tmp_path / "archive"
    selected_db = _make_index_db(archive_root)
    replacement_root = tmp_path / "replacement"
    real_open = open_readonly_connection
    opened_readers = 0

    def replace_after_reader_open(path: Path, *, opened_main_fd: int | None = None) -> sqlite3.Connection:
        nonlocal opened_readers
        connection = real_open(path, opened_main_fd=opened_main_fd)
        opened_readers += 1
        if opened_readers == 2:
            replacement_db = _make_index_db(replacement_root)
            selected_db.unlink()
            replacement_db.replace(selected_db)
        return connection

    monkeypatch.setattr(affordance_usage, "open_readonly_connection", replace_after_reader_open)

    with pytest.raises(RuntimeError, match="selected index path was replaced"):
        affordance_usage.build_report(
            affordance_usage.AffordanceUsageArgs(
                archive_root=archive_root,
                out_dir=None,
                days=36500,
                family=("serena",),
                detail_pattern=(),
                sample_limit=10,
                json=True,
                all_time=False,
                index_db=selected_db,
            )
        )


def test_affordance_usage_reader_stays_on_opened_inode_across_path_replacement_and_restoration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production reader must use the inode opened before the pathname mutation."""
    archive_root = tmp_path / "archive"
    selected_db = _make_index_db(archive_root)
    replacement_root = tmp_path / "replacement"
    original_path = tmp_path / "original-index.db"
    real_open = open_readonly_connection
    swapped = False

    def replace_before_reader_open(path: Path, *, opened_main_fd: int | None = None) -> sqlite3.Connection:
        nonlocal swapped
        if not swapped:
            replacement_db = _make_index_db(replacement_root)
            with sqlite3.connect(replacement_db) as replacement:
                replacement.execute("DELETE FROM blocks")
                replacement.commit()
            selected_db.rename(original_path)
            replacement_db.rename(selected_db)
            connection = real_open(path, opened_main_fd=opened_main_fd)
            selected_db.rename(replacement_db)
            original_path.rename(selected_db)
            swapped = True
            return connection
        return real_open(path, opened_main_fd=opened_main_fd)

    monkeypatch.setattr(affordance_usage, "open_readonly_connection", replace_before_reader_open)
    report = affordance_usage.build_report(
        affordance_usage.AffordanceUsageArgs(
            archive_root=archive_root,
            out_dir=None,
            days=36500,
            family=("serena",),
            detail_pattern=("codebase-memory",),
            sample_limit=10,
            json=True,
            all_time=False,
            index_db=selected_db,
        )
    )

    assert swapped is True
    assert {row["family"]: row["actions"] for row in report["family_counts"]}["codebase-memory"] == 2
    assert report["snapshot_identity"]["stable"] is True


def test_affordance_usage_rejects_replaced_wal_sidecar_and_accepts_restoration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sidecar hash must remain tied to the object SQLite opened, including restoration."""
    archive_root = tmp_path / "archive"
    selected_db = _make_index_db(archive_root)
    writer = sqlite3.connect(selected_db)
    assert writer.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
    writer.execute("PRAGMA wal_autocheckpoint = 0")
    writer.execute("INSERT INTO sessions VALUES ('wal-row', 'codex-session', 'WAL', 1)")
    writer.commit()
    wal_path = Path(f"{selected_db}-wal")
    saved_wal = tmp_path / "saved-index.wal"
    replacement_wal = tmp_path / "replacement-index.wal"
    real_open = open_readonly_connection
    swapped = False

    def replace_wal_after_reader_open(path: Path, *, opened_main_fd: int | None = None) -> sqlite3.Connection:
        nonlocal swapped
        connection = real_open(path, opened_main_fd=opened_main_fd)
        if not swapped:
            wal_path.rename(saved_wal)
            replacement_wal.write_bytes(b"replacement sidecar")
            replacement_wal.rename(wal_path)
            swapped = True
        return connection

    monkeypatch.setattr(affordance_usage, "open_readonly_connection", replace_wal_after_reader_open)
    with pytest.raises(RuntimeError, match="sidecar"):
        affordance_usage.build_report(
            affordance_usage.AffordanceUsageArgs(
                archive_root=archive_root,
                out_dir=None,
                days=36500,
                family=("serena",),
                detail_pattern=(),
                sample_limit=10,
                json=True,
                all_time=False,
                index_db=selected_db,
            )
        )

    wal_path.unlink()
    saved_wal.rename(wal_path)
    monkeypatch.setattr(affordance_usage, "open_readonly_connection", real_open)
    report = affordance_usage.build_report(
        affordance_usage.AffordanceUsageArgs(
            archive_root=archive_root,
            out_dir=None,
            days=36500,
            family=("serena",),
            detail_pattern=(),
            sample_limit=10,
            json=True,
            all_time=False,
            index_db=selected_db,
        )
    )
    writer.close()
    assert report["snapshot_identity"]["stable"] is True


def test_affordance_usage_rejects_symlinked_wal_sidecar(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    selected_db = _make_index_db(archive_root)
    wal_path = Path(f"{selected_db}-wal")
    wal_target = tmp_path / "wal-target"
    wal_target.write_bytes(b"unsafe sidecar")
    wal_path.symlink_to(wal_target)

    with pytest.raises(RuntimeError, match="sidecar"):
        affordance_usage.build_report(
            affordance_usage.AffordanceUsageArgs(
                archive_root=archive_root,
                out_dir=None,
                days=36500,
                family=("serena",),
                detail_pattern=(),
                sample_limit=10,
                json=True,
                all_time=False,
                index_db=selected_db,
            )
        )


def test_affordance_usage_snapshot_includes_a_quiescent_wal_file_set(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    selected_db = _make_index_db(archive_root)
    writer = sqlite3.connect(selected_db)
    try:
        assert writer.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        writer.execute("PRAGMA wal_autocheckpoint = 0")
        writer.execute("INSERT INTO sessions VALUES ('wal-row', 'codex-session', 'WAL', 1)")
        writer.commit()
        assert Path(f"{selected_db}-wal").is_file()

        report = affordance_usage.build_report(
            affordance_usage.AffordanceUsageArgs(
                archive_root=archive_root,
                out_dir=None,
                days=36500,
                family=("serena",),
                detail_pattern=(),
                sample_limit=10,
                json=True,
                all_time=False,
                index_db=selected_db,
            )
        )
    finally:
        writer.close()

    identity = report["snapshot_identity"]
    assert identity["stable"] is True
    before_files = {Path(row["path"]).name: row for row in identity["before"]["files"]}
    assert before_files["index.db-wal"]["present"] is True


def test_affordance_usage_captures_reader_created_sqlite_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WAL, SHM, and journal names created after reader open enter the authority set."""
    archive_root = tmp_path / "archive"
    selected_db = _make_index_db(archive_root)
    real_snapshot = affordance_usage._snapshot_observation
    snapshot_calls = 0

    def create_sidecars_before_after_snapshot(
        path: Path,
        *,
        opened_main_fd: int | None = None,
        opened_sidecar_fds: dict[str, int] | None = None,
    ) -> dict[str, object]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 2:
            for suffix in ("-wal", "-shm", "-journal"):
                Path(f"{path}{suffix}").write_bytes(b"created after reader open")
        return real_snapshot(
            path,
            opened_main_fd=opened_main_fd,
            opened_sidecar_fds=opened_sidecar_fds,
        )

    monkeypatch.setattr(affordance_usage, "_snapshot_observation", create_sidecars_before_after_snapshot)
    monkeypatch.setattr(affordance_usage, "_data_version", lambda _connection: 1)
    report = affordance_usage.build_report(
        affordance_usage.AffordanceUsageArgs(
            archive_root=archive_root,
            out_dir=None,
            days=36500,
            family=("serena",),
            detail_pattern=(),
            sample_limit=10,
            json=True,
            all_time=False,
            index_db=selected_db,
        )
    )

    before_files = {Path(row["path"]).name: row for row in report["snapshot_identity"]["before"]["files"]}
    after_files = {Path(row["path"]).name: row for row in report["snapshot_identity"]["after"]["files"]}
    assert all(not before_files[f"index.db{suffix}"]["present"] for suffix in ("-wal", "-shm", "-journal"))
    assert all(after_files[f"index.db{suffix}"]["present"] for suffix in ("-wal", "-shm", "-journal"))
    assert report["snapshot_identity"]["stable"] is False


@pytest.mark.parametrize(
    ("target", "kind"),
    [
        ("main", "directory"),
        ("main", "fifo"),
        ("sidecar", "directory"),
        ("sidecar", "fifo"),
        ("main", "device"),
        ("sidecar", "device"),
    ],
)
def test_affordance_usage_rejects_nonregular_main_and_sidecar_objects(
    tmp_path: Path,
    target: str,
    kind: str,
) -> None:
    archive_root = tmp_path / "archive"
    selected_db = _make_index_db(archive_root)
    object_path = selected_db if target == "main" else Path(f"{selected_db}-wal")
    if target == "main":
        selected_db.unlink()
    if kind == "directory":
        object_path.mkdir()
    elif kind == "fifo":
        os.mkfifo(object_path)
    else:
        try:
            os.mknod(object_path, stat.S_IFCHR | 0o600, os.makedev(1, 3))
        except PermissionError:
            pytest.skip("device nodes are unavailable in this test environment")

    with pytest.raises(RuntimeError, match="regular|safely|sidecar"):
        affordance_usage.build_report(
            affordance_usage.AffordanceUsageArgs(
                archive_root=archive_root,
                out_dir=None,
                days=36500,
                family=("serena",),
                detail_pattern=(),
                sample_limit=10,
                json=True,
                all_time=False,
                index_db=selected_db,
            )
        )


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
    selected_db = archive_root / "selected-index.db"
    shutil.copy2(archive_root / "index.db", selected_db)
    args = affordance_usage.AffordanceUsageArgs(
        archive_root=archive_root,
        out_dir=None,
        days=36500,
        family=(),
        detail_pattern=("codebase-memory",),
        sample_limit=10,
        json=True,
        all_time=False,
        index_db=selected_db,
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
    selected_db = archive_root / "selected-index.db"
    shutil.copy2(archive_root / "index.db", selected_db)
    args = affordance_usage.AffordanceUsageArgs(
        archive_root=archive_root,
        out_dir=None,
        days=36500,
        family=(),
        detail_pattern=("search_code",),
        sample_limit=10,
        json=True,
        all_time=False,
        index_db=selected_db,
    )

    report = affordance_usage.build_report(args)

    assert report["family_counts"][0]["actions"] == 2
    tool_counts = {row["tool_name"]: row for row in report["tool_counts"]}
    assert tool_counts["codebase-memory/search_code"]["raw_tool_names"] == "search_code"
    assert tool_counts["codebase-memory/command-detail"]["raw_tool_names"] == "functions.exec_command"
    assert any("search_code" in str(row["detail"]) for row in report["samples"])
    assert all("search code" not in str(row["detail"]) for row in report["samples"])


def test_affordance_usage_detail_fast_path_splits_mixed_known_families(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    _make_index_db(archive_root)
    calls: list[tuple[str, ...]] = []

    class FakeArchive(AbstractContextManager["FakeArchive"]):
        index_db_path = archive_root / "index.db"

        def __enter__(self) -> FakeArchive:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def list_tool_action_evidence_count_rows(
            self,
            query: object = None,
            *,
            detail_patterns: tuple[str, ...],
            since_ms: int | None = None,
        ) -> list[dict[str, Any]]:
            del query, since_ms
            calls.append(detail_patterns)
            if detail_patterns == ("serena",):
                return [
                    {
                        "source_name": "claude-code",
                        "origin": "claude-code-session",
                        "normalized_tool_name": "serena/command-detail",
                        "action_kind": "shell",
                        "evidence_kind": "command_detail",
                        "matched_by": "detail",
                        "call_count": 4,
                        "session_count": 2,
                        "error_count": 1,
                        "nonzero_exit_count": 0,
                    }
                ]
            if detail_patterns == ("codebase-memory", "search_code"):
                return [
                    {
                        "source_name": "claude-code",
                        "origin": "claude-code-session",
                        "normalized_tool_name": "codebase-memory/command-detail",
                        "action_kind": "shell",
                        "evidence_kind": "command_detail",
                        "matched_by": "detail",
                        "call_count": 2,
                        "session_count": 1,
                        "error_count": 0,
                        "nonzero_exit_count": 0,
                    }
                ]
            return []

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        lambda _root, **_kwargs: FakeArchive(),
    )
    args = affordance_usage.AffordanceUsageArgs(
        archive_root=archive_root,
        out_dir=None,
        days=36500,
        family=(),
        detail_pattern=("serena", "codebase-memory", "search_code", "find_symbol"),
        sample_limit=10,
        json=True,
        all_time=False,
    )

    report = affordance_usage.build_report(args)

    assert ("serena",) in calls
    assert ("codebase-memory", "search_code") in calls
    assert report["report_version"] == 2
    assert report["action_scope"] == "product-action-evidence-recent-window-known-family-patterns"
    families = {row["family"]: row for row in report["family_counts"]}
    assert families["serena"]["actions"] == 4
    assert families["codebase-memory"]["actions"] == 2
    assert report["samples"] == []
