"""Tests for the `polylogue ops excise` command (polylogue-27m).

Mirrors tests/unit/cli/test_reset.py's CliRunner-with-patched-paths pattern.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _seed_session(archive_root: Path, *, native_id: str) -> str:
    archive_root.mkdir(parents=True, exist_ok=True)
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)

    source_conn = sqlite3.connect(source_db)
    source_conn.execute("PRAGMA foreign_keys = ON")
    try:
        raw_id = write_source_raw_session(
            source_conn,
            origin="codex-session",
            source_path=f"/fake/{native_id}.jsonl",
            source_index=0,
            payload=f"payload-{native_id}".encode(),
            acquired_at_ms=1_000,
            native_id=native_id,
        )
        source_conn.commit()
    finally:
        source_conn.close()

    index_conn = sqlite3.connect(index_db)
    index_conn.execute("PRAGMA foreign_keys = ON")
    try:
        index_conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms) "
            "VALUES (?, 'codex-session', ?, ?, zeroblob(32), 1000, 2000)",
            (native_id, raw_id, f"Session {native_id}"),
        )
        index_conn.commit()
        session_id = index_conn.execute("SELECT session_id FROM sessions WHERE native_id = ?", (native_id,)).fetchone()[
            0
        ]
    finally:
        index_conn.close()
    return str(session_id)


def _seed_lineage_pair(archive_root: Path) -> tuple[str, str]:
    """Seed a parent session and a prefix-sharing child `session_links` row.

    Returns ``(parent_session_id, child_session_id)``.
    """
    parent_id = _seed_session(archive_root, native_id="lineage-parent")
    child_id = _seed_session(archive_root, native_id="lineage-child")

    index_conn = sqlite3.connect(archive_root / "index.db")
    index_conn.execute("PRAGMA foreign_keys = ON")
    try:
        index_conn.execute(
            "INSERT INTO messages (session_id, native_id, position, role, content_hash) "
            "VALUES (?, 'm1', 0, 'user', zeroblob(32))",
            (parent_id,),
        )
        branch_point = index_conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ?", (parent_id,)
        ).fetchone()[0]
        index_conn.execute(
            """
            INSERT INTO session_links (
                src_session_id, dst_origin, dst_native_id, link_type,
                resolved_dst_session_id, branch_point_message_id, inheritance,
                status, method, confidence, evidence_json, observed_at_ms, resolved_at_ms
            ) VALUES (?, 'codex-session', 'lineage-parent', 'branch', ?, ?, 'prefix-sharing',
                      NULL, NULL, 1.0, '[]', 1000, NULL)
            """,
            (child_id, parent_id, branch_point),
        )
        index_conn.commit()
    finally:
        index_conn.close()
    return parent_id, child_id


class TestExciseStandalone:
    def test_missing_session_reports_not_found(self, tmp_path: Path) -> None:
        with patch("polylogue.cli.commands.excise.archive_root", return_value=tmp_path / "archive"):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["ops", "excise", "--session", "codex-session:nope", "--reason", "r", "--yes", "--json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "not_found"

    def test_dry_run_reports_plan_without_mutating(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_session(archive_root, native_id="dry-run-1")
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["ops", "excise", "--session", session_id, "--reason", "r", "--dry-run", "--json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "preview"
        assert payload["plan"]["found"] is True
        assert payload["plan"]["index_sessions"] == 1

        # dry-run must not have mutated anything.
        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            count = index_conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[
                0
            ]
        finally:
            index_conn.close()
        assert count == 1

    def test_without_yes_aborts_in_json_mode(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_session(archive_root, native_id="no-yes-1")
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["ops", "excise", "--session", session_id, "--reason", "r", "--json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "aborted"

        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            count = index_conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[
                0
            ]
        finally:
            index_conn.close()
        assert count == 1

    def test_yes_applies_excision(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_session(archive_root, native_id="apply-1")
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["ops", "excise", "--session", session_id, "--reason", "secret leak", "--yes", "--json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assert payload["detail"]  # receipt assertion id

        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            count = index_conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[
                0
            ]
        finally:
            index_conn.close()
        assert count == 0

        user_conn = sqlite3.connect(archive_root / "user.db")
        try:
            receipt_count = user_conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE assertion_id = ?", (payload["detail"],)
            ).fetchone()[0]
        finally:
            user_conn.close()
        assert receipt_count == 1


class TestExciseMirrorPrimary:
    def test_mirror_dry_run_does_not_write_a_request(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "ops",
                    "excise",
                    "--session",
                    "codex-session:whatever",
                    "--reason",
                    "r",
                    "--mode",
                    "mirror",
                    "--dry-run",
                    "--json",
                ],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "preview"
        assert not (archive_root / "user.db").exists()

    def test_primary_yes_creates_pending_request_without_touching_local_content(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_session(archive_root, native_id="primary-1")
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "ops",
                    "excise",
                    "--session",
                    session_id,
                    "--reason",
                    "leak",
                    "--mode",
                    "primary",
                    "--yes",
                    "--json",
                ],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assertion_id = payload["detail"]

        user_conn = sqlite3.connect(archive_root / "user.db")
        try:
            row = user_conn.execute(
                "SELECT kind, target_ref FROM assertions WHERE assertion_id = ?", (assertion_id,)
            ).fetchone()
        finally:
            user_conn.close()
        assert row is not None
        assert row[0] == "excision_request"
        assert row[1] == f"session:{session_id}"

        # Local content is untouched by mirror/primary mode.
        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            count = index_conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[
                0
            ]
        finally:
            index_conn.close()
        assert count == 1


class TestExciseLineageSafety:
    """CLI coverage for the polylogue-27m fix-round lineage-safety guard."""

    def test_dry_run_surfaces_lineage_dependents(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        parent_id, child_id = _seed_lineage_pair(archive_root)
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["ops", "excise", "--session", parent_id, "--reason", "r", "--dry-run", "--json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["plan"]["lineage_dependent_session_ids"] == [child_id]

    def test_without_cascade_flag_refuses_and_does_not_mutate(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        parent_id, child_id = _seed_lineage_pair(archive_root)
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["ops", "excise", "--session", parent_id, "--reason", "r", "--yes", "--json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "aborted"

        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            remaining = index_conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            index_conn.close()
        assert remaining == 2  # neither parent nor child touched

    def test_with_cascade_flag_removes_parent_and_dependents(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        parent_id, child_id = _seed_lineage_pair(archive_root)
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "ops",
                    "excise",
                    "--session",
                    parent_id,
                    "--reason",
                    "r",
                    "--yes",
                    "--cascade-lineage",
                    "--json",
                ],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        # affected_count is index_sessions summed across the whole cascade.
        assert payload["affected_count"] == 2

        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            remaining = index_conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            index_conn.close()
        assert remaining == 0

        user_conn = sqlite3.connect(archive_root / "user.db")
        try:
            receipt_count = user_conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE kind = 'excision_record'"
            ).fetchone()[0]
        finally:
            user_conn.close()
        assert receipt_count == 2  # one durable audit receipt per removed session
