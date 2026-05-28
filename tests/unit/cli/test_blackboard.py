"""Tests for the polylogue blackboard CLI commands."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.paths import db_path as resolve_db_path

BLACKBOARD_DDL = """
CREATE TABLE IF NOT EXISTS blackboard_notes (
    note_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    scope_repo TEXT,
    scope_session TEXT,
    scope_issue INTEGER,
    scope_path TEXT,
    related_session_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    materialized_at TEXT,
    resolved_at TEXT
);
"""


def _seed_db(workspace: dict[str, Path]) -> Path:
    """Create a database with the blackboard_notes table at the workspace path.

    Monkeypatches XDG_DATA_HOME so db_path() resolves inside the workspace.
    Returns the database path.
    """
    db = resolve_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.executescript(BLACKBOARD_DDL)
    conn.commit()
    conn.close()
    return db


def _insert_note(workspace: dict[str, Path], **kwargs: object) -> str:
    """Insert a blackboard note directly and return its note_id."""

    db = resolve_db_path()
    note_id = kwargs.pop("note_id", "test-note-1")
    kind = kwargs.pop("kind", "finding")
    title = kwargs.pop("title", "Test Note")
    content = kwargs.pop("content", "Test content")
    scope_repo = kwargs.pop("scope_repo", None)
    scope_session = kwargs.pop("scope_session", None)
    scope_issue = kwargs.pop("scope_issue", None)
    scope_path = kwargs.pop("scope_path", None)
    related_json = kwargs.pop("related_json", "[]")
    created_at = kwargs.pop("created_at", "2026-05-01T00:00:00Z")
    resolved_at = kwargs.pop("resolved_at", None)

    conn = sqlite3.connect(str(db))
    conn.execute(
        """INSERT INTO blackboard_notes
           (note_id, kind, title, content, scope_repo, scope_session,
            scope_issue, scope_path, related_session_ids_json,
            created_at, materialized_at, resolved_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(note_id),
            str(kind),
            str(title),
            str(content),
            scope_repo,
            scope_session,
            scope_issue,
            scope_path,
            str(related_json),
            str(created_at),
            str(created_at),
            resolved_at,
        ),
    )
    conn.commit()
    conn.close()
    return str(note_id)


class TestBlackboardPost:
    """Tests for `polylogue blackboard post`."""

    def test_post_creates_note(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Posting a valid note creates it in the database."""
        _seed_db(workspace_env)

        result = cli_runner.invoke(
            cli,
            [
                "blackboard",
                "post",
                "--kind",
                "finding",
                "--title",
                "Test Finding",
                "--content",
                "Something interesting was found.",
            ],
        )
        assert result.exit_code == 0, f"post failed: {result.output}"
        assert "Posted finding note" in result.output

        # Verify the note is in the database.
        db = resolve_db_path()
        conn = sqlite3.connect(str(db))
        row = conn.execute(
            "SELECT kind, title, content FROM blackboard_notes WHERE title = ?",
            ("Test Finding",),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "finding"
        assert row[1] == "Test Finding"
        assert row[2] == "Something interesting was found."

    def test_post_with_scope_fields(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Posting with scope fields stores them correctly."""
        _seed_db(workspace_env)

        result = cli_runner.invoke(
            cli,
            [
                "blackboard",
                "post",
                "--kind",
                "blocker",
                "--title",
                "Scope Test",
                "--content",
                "Scoped content.",
                "--scope-repo",
                "polylogue",
                "--scope-issue",
                "1721",
                "--scope-path",
                "src/main.py",
                "--related-sessions",
                "session-1",
                "--related-sessions",
                "session-2",
            ],
        )
        assert result.exit_code == 0, f"post failed: {result.output}"

        db = resolve_db_path()
        conn = sqlite3.connect(str(db))
        row = conn.execute(
            "SELECT kind, title, scope_repo, scope_issue, scope_path, "
            "related_session_ids_json FROM blackboard_notes WHERE title = ?",
            ("Scope Test",),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "blocker"
        assert row[2] == "polylogue"
        assert row[3] == 1721
        assert row[4] == "src/main.py"
        import json

        related = json.loads(row[5])
        assert "session-1" in related
        assert "session-2" in related

    def test_post_with_invalid_kind_rejected(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Posting with an invalid kind produces a Click error."""
        _seed_db(workspace_env)

        result = cli_runner.invoke(
            cli,
            [
                "blackboard",
                "post",
                "--kind",
                "invalid-kind",
                "--title",
                "Bad Kind",
                "--content",
                "Should fail.",
            ],
        )
        assert result.exit_code != 0
        assert "invalid choice" in result.output.lower() or "usage" in result.output.lower()

    def test_post_without_title_rejected(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Posting without --title produces a Click error (required option)."""
        _seed_db(workspace_env)

        result = cli_runner.invoke(
            cli,
            [
                "blackboard",
                "post",
                "--kind",
                "finding",
                "--content",
                "Missing title.",
            ],
        )
        assert result.exit_code != 0
        assert "missing option" in result.output.lower() or "error" in result.output.lower()

    def test_post_without_content_rejected(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Posting without --content produces a Click error (required option)."""
        _seed_db(workspace_env)

        result = cli_runner.invoke(
            cli,
            [
                "blackboard",
                "post",
                "--kind",
                "finding",
                "--title",
                "Missing Content",
            ],
        )
        assert result.exit_code != 0
        assert "missing option" in result.output.lower() or "error" in result.output.lower()

    def test_post_without_kind_rejected(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Posting without --kind produces a Click error (required option)."""
        _seed_db(workspace_env)

        result = cli_runner.invoke(
            cli,
            [
                "blackboard",
                "post",
                "--title",
                "No Kind",
                "--content",
                "Missing kind.",
            ],
        )
        assert result.exit_code != 0
        assert "missing option" in result.output.lower() or "error" in result.output.lower()

    def test_post_no_database(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Posting when no archive database exists shows an error."""
        # Don't seed — let the db file be absent.
        result = cli_runner.invoke(
            cli,
            [
                "blackboard",
                "post",
                "--kind",
                "finding",
                "--title",
                "No DB",
                "--content",
                "Should fail gracefully.",
            ],
        )
        assert result.exit_code != 0
        assert "no archive database" in result.output.lower()

    def test_post_allowed_kinds(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """All six allowed kinds are accepted."""
        allowed = ["finding", "blocker", "decision", "handoff", "question", "observation"]
        _seed_db(workspace_env)

        for kind in allowed:
            result = cli_runner.invoke(
                cli,
                [
                    "blackboard",
                    "post",
                    "--kind",
                    kind,
                    "--title",
                    f"Kind {kind}",
                    "--content",
                    f"Testing kind {kind}.",
                ],
            )
            assert result.exit_code == 0, f"kind {kind} rejected: {result.output}"


class TestBlackboardList:
    """Tests for `polylogue blackboard list`."""

    def test_list_returns_notes(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Listing when notes exist shows them."""
        _seed_db(workspace_env)
        _insert_note(workspace_env, note_id="n1", kind="finding", title="First", content="Content 1")
        _insert_note(workspace_env, note_id="n2", kind="blocker", title="Second", content="Content 2")

        result = cli_runner.invoke(cli, ["blackboard", "list"])
        assert result.exit_code == 0, f"list failed: {result.output}"
        assert "First" in result.output
        assert "Second" in result.output
        assert "finding" in result.output
        assert "blocker" in result.output

    def test_list_empty_gracefully(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Listing when no notes exist shows a dim message."""
        _seed_db(workspace_env)

        result = cli_runner.invoke(cli, ["blackboard", "list"])
        assert result.exit_code == 0, f"list failed: {result.output}"
        assert "no matching notes" in result.output.lower() or "no blackboard notes yet" in result.output.lower()

    def test_list_filter_by_kind(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """--kind filter only shows matching notes."""
        _seed_db(workspace_env)
        _insert_note(workspace_env, note_id="n1", kind="finding", title="Finding Note", content="C1")
        _insert_note(workspace_env, note_id="n2", kind="blocker", title="Blocker Note", content="C2")

        result = cli_runner.invoke(cli, ["blackboard", "list", "--kind", "finding"])
        assert result.exit_code == 0
        assert "Finding Note" in result.output
        assert "Blocker Note" not in result.output

    def test_list_filter_unresolved(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """--unresolved filter only shows notes without resolved_at."""
        _seed_db(workspace_env)
        _insert_note(workspace_env, note_id="n1", kind="blocker", title="Open Blocker", content="C1", resolved_at=None)
        _insert_note(
            workspace_env,
            note_id="n2",
            kind="blocker",
            title="Resolved Blocker",
            content="C2",
            resolved_at="2026-05-02T00:00:00Z",
        )

        result = cli_runner.invoke(cli, ["blackboard", "list", "--unresolved"])
        assert result.exit_code == 0
        assert "Open Blocker" in result.output
        assert "Resolved Blocker" not in result.output

    def test_list_no_database(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Listing when no archive database exists shows an error."""
        result = cli_runner.invoke(cli, ["blackboard", "list"])
        assert result.exit_code != 0
        assert "no archive database" in result.output.lower()
