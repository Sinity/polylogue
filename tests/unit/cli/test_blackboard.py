"""Tests for the polylogue blackboard CLI commands."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_blackboard_note


def _seed_db(workspace: dict[str, Path]) -> Path:
    """Create an archive user database with blackboard tables."""

    user_db = workspace["archive_root"] / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    return user_db


def _insert_note(workspace: dict[str, Path], **kwargs: object) -> str:
    """Insert an archive blackboard note directly and return its note_id."""

    user_db = _seed_db(workspace)
    note_id = str(kwargs.pop("note_id", "test-note-1"))
    kind = str(kwargs.pop("kind", "finding"))
    title = str(kwargs.pop("title", "Test Note"))
    content = str(kwargs.pop("content", "Test content"))
    scope_repo = kwargs.pop("scope_repo", None)
    scope_session = kwargs.pop("scope_session", None)
    scope_issue = kwargs.pop("scope_issue", None)
    scope_path = kwargs.pop("scope_path", None)

    body_lines = [f"[{kind}] {title}", "", content]
    scope_lines = []
    if scope_repo:
        scope_lines.append(f"scope_repo: {scope_repo}")
    if scope_issue:
        scope_lines.append(f"scope_issue: {scope_issue}")
    if scope_path:
        scope_lines.append(f"scope_path: {scope_path}")
    if scope_lines:
        body_lines.extend(["", *scope_lines])

    conn = sqlite3.connect(user_db)
    try:
        upsert_blackboard_note(
            conn,
            "\n".join(body_lines),
            target_type="session" if scope_session else None,
            target_id=str(scope_session) if scope_session else None,
            note_id=note_id,
        )
        conn.commit()
    finally:
        conn.close()
    return note_id


class TestBlackboardPost:
    """Tests for `polylogue blackboard post`."""

    def test_post_creates_note(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Posting a valid note creates it in archive user.db."""
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

        user_db = workspace_env["archive_root"] / "user.db"
        conn = sqlite3.connect(str(user_db))
        row = conn.execute(
            "SELECT target_type, target_id, body FROM blackboard_notes WHERE body LIKE ?",
            ("[finding] Test Finding%",),
        ).fetchone()
        conn.close()
        assert row == (None, None, "[finding] Test Finding\n\nSomething interesting was found.")

    def test_post_writes_archive_user_db(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Posting a scoped note writes directly to archive user.db."""

        result = cli_runner.invoke(
            cli,
            [
                "blackboard",
                "post",
                "--kind",
                "decision",
                "--title",
                "Archive",
                "--content",
                "Store in user db.",
                "--scope-session",
                "codex-session:one",
            ],
        )

        assert result.exit_code == 0, f"post failed: {result.output}"
        conn = sqlite3.connect(workspace_env["archive_root"] / "user.db")
        row = conn.execute("SELECT target_type, target_id, body FROM blackboard_notes").fetchone()
        conn.close()
        assert row == ("session", "codex-session:one", "[decision] Archive\n\nStore in user db.")

    def test_post_with_scope_fields(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Posting with scope fields stores them in the archive note body."""
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

        conn = sqlite3.connect(workspace_env["archive_root"] / "user.db")
        row = conn.execute("SELECT body FROM blackboard_notes WHERE body LIKE ?", ("[blocker] Scope Test%",)).fetchone()
        conn.close()
        assert row is not None
        assert "Scoped content." in row[0]
        assert "scope_repo: polylogue" in row[0]
        assert "scope_issue: 1721" in row[0]
        assert "scope_path: src/main.py" in row[0]
        assert "related_sessions: session-1, session-2" in row[0]

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
        """Posting without --title produces a Click error."""
        _seed_db(workspace_env)

        result = cli_runner.invoke(cli, ["blackboard", "post", "--kind", "finding", "--content", "Missing title."])

        assert result.exit_code != 0
        assert "missing option" in result.output.lower() or "error" in result.output.lower()

    def test_post_without_content_rejected(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Posting without --content produces a Click error."""
        _seed_db(workspace_env)

        result = cli_runner.invoke(cli, ["blackboard", "post", "--kind", "finding", "--title", "Missing Content"])

        assert result.exit_code != 0
        assert "missing option" in result.output.lower() or "error" in result.output.lower()

    def test_post_without_kind_rejected(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Posting without --kind produces a Click error."""
        _seed_db(workspace_env)

        result = cli_runner.invoke(cli, ["blackboard", "post", "--title", "No Kind", "--content", "Missing kind."])

        assert result.exit_code != 0
        assert "missing option" in result.output.lower() or "error" in result.output.lower()

    def test_post_initializes_user_database(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Posting initializes archive user.db when it is absent."""

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
                "Create user tier.",
            ],
        )

        assert result.exit_code == 0
        assert (workspace_env["archive_root"] / "user.db").exists()

    def test_post_ignores_sibling_index_outside_configured_root(
        self,
        cli_runner: CliRunner,
        workspace_env: dict[str, Path],
    ) -> None:
        """Posting writes user.db at the configured archive root, ignoring a
        sibling index.db placed outside it. The configured archive root is
        authoritative and sibling-index discovery is removed, so a
        stray index.db elsewhere must not redirect the user-overlay write."""
        outside_root = workspace_env["data_root"] / "polylogue"
        outside_root.mkdir(parents=True)
        (outside_root / "index.db").write_text("index.db", encoding="utf-8")

        result = cli_runner.invoke(
            cli,
            [
                "blackboard",
                "post",
                "--kind",
                "handoff",
                "--title",
                "Active Root",
                "--content",
                "Use the active archive root.",
            ],
        )

        assert result.exit_code == 0, f"post failed: {result.output}"
        assert (workspace_env["archive_root"] / "user.db").exists()
        assert not (outside_root / "user.db").exists()
        with sqlite3.connect(workspace_env["archive_root"] / "user.db") as conn:
            row = conn.execute("SELECT body FROM blackboard_notes").fetchone()
        assert row == ("[handoff] Active Root\n\nUse the active archive root.",)

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
        _insert_note(workspace_env, note_id="n1", kind="finding", title="Finding Note", content="C1")
        _insert_note(workspace_env, note_id="n2", kind="blocker", title="Blocker Note", content="C2")

        result = cli_runner.invoke(cli, ["blackboard", "list", "--kind", "finding"])

        assert result.exit_code == 0
        assert "Finding Note" in result.output
        assert "Blocker Note" not in result.output

    def test_list_filter_unresolved(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """--unresolved filter only shows blocker/question notes."""
        _insert_note(workspace_env, note_id="n1", kind="blocker", title="Open Blocker", content="C1")
        _insert_note(workspace_env, note_id="n2", kind="finding", title="Finding", content="C2")

        result = cli_runner.invoke(cli, ["blackboard", "list", "--unresolved"])

        assert result.exit_code == 0
        assert "Open Blocker" in result.output
        assert "Finding" not in result.output

    def test_list_no_database(self, cli_runner: CliRunner, workspace_env: dict[str, Path]) -> None:
        """Listing when no user database exists reports an empty blackboard."""

        result = cli_runner.invoke(cli, ["blackboard", "list"])

        assert result.exit_code == 0
        assert "no blackboard notes yet" in result.output.lower()

    def test_list_ignores_sibling_index_outside_configured_root(
        self,
        cli_runner: CliRunner,
        workspace_env: dict[str, Path],
    ) -> None:
        """Listing reads the configured archive root's user.db and ignores a
        sibling index.db (and its user.db) placed outside it. A note seeded only
        in that outside root must not surface — the configured root is the single
        authoritative user-overlay home."""
        outside_root = workspace_env["data_root"] / "polylogue"
        outside_root.mkdir(parents=True)
        (outside_root / "index.db").write_text("index.db", encoding="utf-8")
        outside_user_db = outside_root / "user.db"
        initialize_archive_database(outside_user_db, ArchiveTier.USER)
        with sqlite3.connect(outside_user_db) as conn:
            upsert_blackboard_note(
                conn,
                "[question] Outside List\n\nSeeded outside the configured root.",
                note_id="outside-list",
            )
            conn.commit()

        result = cli_runner.invoke(cli, ["blackboard", "list"])

        assert result.exit_code == 0, f"list failed: {result.output}"
        assert "Outside List" not in result.output
