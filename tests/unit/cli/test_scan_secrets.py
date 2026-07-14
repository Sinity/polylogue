"""Tests for the `polylogue ops scan-secrets` command (polylogue-27m fix round).

This is the CLI surface an operator actually runs. Mirrors
tests/unit/cli/test_excise.py's CliRunner-with-patched-paths pattern.
Anti-vacuity: ``test_yes_finds_and_records_a_credential_shaped_span`` drives
the real command end to end (real index.db block read, real regex/entropy
rules, real write chokepoint into user.db); reverting the CLI wiring to a
no-op (or to zero production callers, the bug this command fixes) makes it
fail because no SECRET_CANDIDATE assertion would be written.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.core.enums import AssertionKind
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _seed_session_with_block_text(archive_root: Path, *, native_id: str, text: str) -> str:
    archive_root.mkdir(parents=True, exist_ok=True)
    index_db = archive_root / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    conn = sqlite3.connect(index_db)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms) "
            "VALUES (?, 'codex-session', ?, zeroblob(32), 1000, 2000)",
            (native_id, f"Session {native_id}"),
        )
        session_id = conn.execute("SELECT session_id FROM sessions WHERE native_id = ?", (native_id,)).fetchone()[0]
        conn.execute(
            "INSERT INTO messages (session_id, native_id, position, role, content_hash) "
            "VALUES (?, 'm1', 0, 'user', zeroblob(32))",
            (session_id,),
        )
        message_id = conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
        conn.execute(
            "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, 0, 'text', ?)",
            (message_id, session_id, text),
        )
        conn.commit()
    finally:
        conn.close()
    return str(session_id)


class TestScanSecrets:
    def test_missing_session_reports_not_found(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        initialize_archive_database(archive_root / "index.db", ArchiveTier.INDEX)
        with patch("polylogue.cli.commands.scan_secrets.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "scan-secrets", "--session", "codex-session:nope", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "not_found"
        assert payload["found"] is False

    def test_finds_and_records_a_credential_shaped_span(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_session_with_block_text(
            archive_root, native_id="scan-1", text="ANTHROPIC_API_KEY=sk-ant-api03-" + "a" * 60
        )
        with patch("polylogue.cli.commands.scan_secrets.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "scan-secrets", "--session", session_id, "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assert payload["found"] is True
        assert payload["blocks_scanned"] == 1
        assert payload["candidates_found"] >= 1
        assert len(payload["written_assertion_ids"]) == payload["candidates_found"]

        # The matched literal must never appear in the plain-text CLI output.
        assert "sk-ant-api03-" + "a" * 60 not in result.output

        user_conn = sqlite3.connect(archive_root / "user.db")
        try:
            count = user_conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE kind = ?", (AssertionKind.SECRET_CANDIDATE.value,)
            ).fetchone()[0]
        finally:
            user_conn.close()
        assert count == payload["candidates_found"]

    def test_no_candidates_for_ordinary_text(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_session_with_block_text(archive_root, native_id="scan-2", text="just a normal message")
        with patch("polylogue.cli.commands.scan_secrets.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "scan-secrets", "--session", session_id, "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["candidates_found"] == 0

    def test_plain_text_output_reports_counts(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_session_with_block_text(
            archive_root, native_id="scan-3", text="AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP"
        )
        with patch("polylogue.cli.commands.scan_secrets.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "scan-secrets", "--session", session_id])
        assert result.exit_code == 0
        assert "blocks scanned: 1" in result.output
        assert "secret candidates found: 1" in result.output
        assert "AKIAABCDEFGHIJKLMNOP" not in result.output
