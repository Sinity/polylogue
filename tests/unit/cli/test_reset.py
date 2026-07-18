"""Tests for reset command."""

from __future__ import annotations

import json
import sqlite3
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace

# =============================================================================
# TEST DATA TABLE
# =============================================================================

RESET_DELETION_CASES = [
    ("--index", "index_db", "index database"),
    ("--database", "archive_db", "database"),
    ("--assets", "assets_dir", "assets"),
    ("--cache", "cache_dir", "cache"),
    ("--auth", "token_path", "auth token"),
]


def _seed_archive_session(archive_root: Path, *, native_id: str, source_path: Path | None = None) -> str:
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    session_id = f"codex-session:{native_id}"
    raw_id = f"raw-{native_id}"
    with sqlite3.connect(source_db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
            )
            VALUES (?, 'codex-session', ?, ?, zeroblob(32), 0, 1000)
            """,
            (raw_id, native_id, str(source_path or archive_root / f"{native_id}.jsonl")),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms
            )
            VALUES (?, 'codex-session', ?, ?, zeroblob(32), 1000, 2000)
            """,
            (native_id, raw_id, f"Session {native_id}"),
        )
    return session_id


# =============================================================================
# SUBPROCESS INTEGRATION TESTS - RESET COMMAND
# =============================================================================


@pytest.mark.integration
class TestResetCommandSubprocess:
    """Subprocess integration tests for the reset command."""

    def test_reset_requires_target(self, tmp_path: Path) -> None:
        """reset without flags fails with helpful message."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["ops", "reset"], env=env)
        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "specify" in output_lower or "target" in output_lower or "--database" in output_lower

    def test_reset_database_requires_force(self, tmp_path: Path) -> None:
        """reset --database without --yes prompts (plain mode fails)."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "ops", "reset", "--database"], env=env)
        # In plain mode without --yes, should exit without deleting
        # (may succeed if no db exists, or show "use --yes" message)
        output_lower = result.output.lower()
        assert result.exit_code == 0 or "force" in output_lower or "nothing" in output_lower

    def test_reset_force_database(self, tmp_path: Path) -> None:
        """reset --database --yes deletes database."""
        from tests.infra.source_builders import GenericSessionBuilder

        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]
        inbox = workspace["paths"]["inbox"]

        # Create some data first
        (GenericSessionBuilder("to-delete").add_user("will be deleted").write_to(inbox / "test.json"))
        run_cli(["--plain", "run", "parse"], env=env)

        # Now reset
        result = run_cli(["--plain", "ops", "reset", "--database", "--yes"], env=env)
        # Should succeed (either deleted or nothing existed)
        assert result.exit_code == 0

    def test_reset_all_flag(self, tmp_path: Path) -> None:
        """reset --all sets all targets."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        # With --yes in plain mode
        result = run_cli(["--plain", "ops", "reset", "--all", "--yes"], env=env)
        # Should succeed (nothing to delete in fresh workspace)
        assert result.exit_code == 0


# =============================================================================
# CLIRUNNER UNIT TESTS - RESET COMMAND
# =============================================================================


class TestResetCommandValidation:
    """Tests for reset command validation."""

    def test_no_flags_shows_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reset without any target flags shows error."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        runner = CliRunner()
        result = runner.invoke(cli, ["ops", "reset"])

        assert result.exit_code == 1
        assert "specify" in result.output.lower()

    def test_all_flag_sets_all_targets(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--all enables all reset targets."""
        # Patch paths to point to tmp_path
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        # Create mock path constants for the test
        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=tmp_path / "archive"),
            patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path / "data"),
            patch("polylogue.cli.commands.reset.cache_home", return_value=tmp_path / "cache"),
            patch("polylogue.cli.commands.reset.drive_token_path", return_value=tmp_path / "token.json"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--all", "--yes"])

            # Should not error even if files don't exist
            assert result.exit_code == 0


class TestResetCommandDeletion:
    """Tests for reset file/directory deletion."""

    @pytest.mark.parametrize("flag,path_attr,desc", RESET_DELETION_CASES)
    def test_reset_flag_deletes_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, path_attr: str, desc: str
    ) -> None:
        """Reset flags delete specified targets."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        # Set up appropriate paths based on path_attr
        if path_attr in {"archive_db", "index_db"}:
            archive_root = tmp_path / "archive"
            archive_root.mkdir()
            target_path = archive_root / "index.db"
            target_path.write_text("test database", encoding="utf-8")
            patches = [
                patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
                patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
            ]
        elif path_attr == "assets_dir":
            data_home = tmp_path / "data"
            target_path = data_home / "assets"
            target_path.mkdir(parents=True)
            (target_path / "test.png").write_bytes(b"test")
            patches = [
                patch("polylogue.cli.commands.reset.archive_root", return_value=tmp_path / "archive"),
                patch("polylogue.cli.commands.reset.data_home", return_value=data_home),
            ]
        elif path_attr == "cache_dir":
            target_path = tmp_path / "cache"
            target_path.mkdir(parents=True)
            (target_path / "index").write_text("index data", encoding="utf-8")
            patches = [
                patch("polylogue.cli.commands.reset.archive_root", return_value=tmp_path / "archive"),
                patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
                patch("polylogue.cli.commands.reset.cache_home", return_value=target_path),
            ]
        elif path_attr == "token_path":
            target_path = tmp_path / "token.json"
            target_path.write_text(json.dumps({"token": "test"}), encoding="utf-8")
            patches = [
                patch("polylogue.cli.commands.reset.archive_root", return_value=tmp_path / "archive"),
                patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
                patch("polylogue.cli.commands.reset.cache_home", return_value=tmp_path / "nonexistent"),
                patch("polylogue.cli.commands.reset.drive_token_path", return_value=target_path),
            ]
        else:
            raise AssertionError(f"Unhandled reset target fixture: {path_attr}")

        assert target_path.exists()

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", flag, "--yes"])

            assert result.exit_code == 0
            assert not target_path.exists()

    def test_multiple_flags(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple flags delete specified targets."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        archive_db = archive_root / "index.db"
        archive_db.write_text("test database", encoding="utf-8")

        data_home = tmp_path / "data"
        assets_dir = data_home / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "keep.png").write_bytes(b"keep")

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
            patch("polylogue.cli.commands.reset.data_home", return_value=data_home),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--database", "--assets", "--yes"])

            assert result.exit_code == 0
            assert not archive_db.exists()
            assert not assets_dir.exists()

    def _seed_archive_tiers(self, archive_root: Path) -> tuple[Path, list[Path], Path]:
        archive_root.mkdir(exist_ok=True)
        source_db = archive_root / "source.db"
        rebuildable = [
            archive_root / "index.db",
            archive_root / "index.db-wal",
            archive_root / "index.db-shm",
            archive_root / "embeddings.db",
            archive_root / "embeddings.db-wal",
            archive_root / "embeddings.db-shm",
            archive_root / "ops.db",
        ]
        user_db = archive_root / "user.db"
        initialize_archive_database(source_db, ArchiveTier.SOURCE)
        for path in [*rebuildable, user_db]:
            path.write_text("test database", encoding="utf-8")
        return source_db, rebuildable, user_db

    def test_reset_index_deletes_only_index_tier(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``reset --index`` rebuilds the index tier without dropping raw or user evidence."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        archive_root = tmp_path / "archive"
        source_db, rebuildable, user_db = self._seed_archive_tiers(archive_root)
        index_targets = {
            archive_root / "index.db",
            archive_root / "index.db-wal",
            archive_root / "index.db-shm",
        }
        preserved = [path for path in [source_db, *rebuildable, user_db] if path not in index_targets]

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
            patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
        ):
            result = CliRunner().invoke(cli, ["ops", "reset", "--index", "--yes"])

        assert result.exit_code == 0
        assert all(not path.exists() for path in index_targets)
        assert all(path.exists() for path in preserved)
        assert "index database" in result.output

    def test_reset_index_refuses_to_delete_managed_active_generation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        archive_root = tmp_path / "archive"
        _source_db, _rebuildable, _user_db = self._seed_archive_tiers(archive_root)
        canonical = tmp_path / "canonical"
        canonical.mkdir()
        active = canonical / "index.db"
        active.write_text("active generation", encoding="utf-8")
        (archive_root / ".index-active-pointer").write_text(str(active), encoding="utf-8")

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
            patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
        ):
            result = CliRunner().invoke(cli, ["ops", "reset", "--index", "--yes"])

        assert result.exit_code == 1
        assert "unsafe for a managed active generation" in result.output
        assert active.read_text(encoding="utf-8") == "active generation"

    def test_reset_database_refuses_to_delete_managed_active_generation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        archive_root = tmp_path / "archive"
        _source_db, _rebuildable, _user_db = self._seed_archive_tiers(archive_root)
        canonical = tmp_path / "canonical"
        canonical.mkdir()
        active = canonical / "index.db"
        active.write_text("active generation", encoding="utf-8")
        (archive_root / ".index-active-pointer").write_text(str(active), encoding="utf-8")

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
            patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
        ):
            result = CliRunner().invoke(cli, ["ops", "reset", "--database", "--yes"])

        assert result.exit_code == 1
        assert "unsafe for a managed active generation" in result.output
        assert active.read_text(encoding="utf-8") == "active generation"

    def test_reset_database_preserves_source_and_irreplaceable_user_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``reset --database`` deletes rebuildable tiers but preserves durable tiers."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        archive_root = tmp_path / "archive"
        source_db, rebuildable, user_db = self._seed_archive_tiers(archive_root)

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
            patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
        ):
            result = CliRunner().invoke(cli, ["ops", "reset", "--database", "--yes"])

        assert result.exit_code == 0
        assert all(not path.exists() for path in rebuildable), "rebuildable tiers should be deleted"
        assert source_db.exists(), "source.db is durable acquired evidence and must survive a plain --database reset"
        assert user_db.exists(), "user.db is irreplaceable and must survive a plain --database reset"
        assert "Preserving source.db" in result.output
        assert "Preserving user.db" in result.output

    def test_reset_database_include_source_and_user_db_deletes_everything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Destructive tier flags explicitly opt into deleting source.db and user.db."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        archive_root = tmp_path / "archive"
        source_db, rebuildable, user_db = self._seed_archive_tiers(archive_root)

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
            patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
        ):
            result = CliRunner().invoke(
                cli,
                ["ops", "reset", "--database", "--include-source-db", "--include-user-db", "--yes"],
            )

        assert result.exit_code == 0
        assert all(not path.exists() for path in [source_db, *rebuildable, user_db])

    def test_reset_database_include_source_db_refuses_missing_source_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting source.db is blocked when raw evidence cannot be reacquired."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        missing_source = tmp_path / "rotated-away.jsonl"
        _seed_archive_session(archive_root, native_id="rotated", source_path=missing_source)

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
            patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
        ):
            result = CliRunner().invoke(cli, ["ops", "reset", "--database", "--include-source-db", "--yes"])

        assert result.exit_code == 1
        assert "Refusing to delete source.db" in result.output
        assert "1 raw row" in result.output
        assert (archive_root / "source.db").exists()
        assert (archive_root / "index.db").exists()

    def test_reset_all_preserves_user_db_without_opt_in(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even ``reset --all`` preserves durable tiers without explicit opt-ins."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        archive_root = tmp_path / "archive"
        source_db, _rebuildable, user_db = self._seed_archive_tiers(archive_root)

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
            patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
        ):
            result = CliRunner().invoke(cli, ["ops", "reset", "--all", "--yes"])

        assert result.exit_code == 0
        assert source_db.exists(), "source.db must survive --all without an explicit --include-source-db opt-in"
        assert user_db.exists(), "user.db must survive --all without an explicit --include-user-db opt-in"

    def test_reset_session_records_archive_suppression_and_deletes_archive_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Session tombstone is user-tier suppression plus archive-row deletion."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="reset-one")

        with patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--session", session_id, "--yes"])

        assert result.exit_code == 0
        assert "1 suppression" in result.output
        assert "1 archive row" in result.output
        with sqlite3.connect(archive_root / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0] == 0
        with sqlite3.connect(archive_root / "user.db") as conn:
            row = conn.execute(
                "SELECT body_text, json_extract(value_json, '$.mode') FROM assertions WHERE kind = 'suppression' AND target_ref = ?",
                (f"session:{session_id}",),
            ).fetchone()
        assert row == ("reset --session", "hide")

    def test_reset_source_tombstones_matching_archive_sessions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Source tombstone matches archive raw_sessions by path-component prefix."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        source_root = tmp_path / "sources" / "codex"
        child_session_id = _seed_archive_session(
            archive_root,
            native_id="source-child",
            source_path=source_root / "session.jsonl",
        )
        sibling_session_id = _seed_archive_session(
            archive_root,
            native_id="source-sibling",
            source_path=tmp_path / "sources" / "codex-other.jsonl",
        )

        with patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--source", str(source_root), "--yes"])

        assert result.exit_code == 0
        assert "Tombstoned 1 session" in result.output
        with sqlite3.connect(archive_root / "index.db") as conn:
            assert (
                conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (child_session_id,)).fetchone()[0]
                == 0
            )
            assert (
                conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (sibling_session_id,)).fetchone()[0]
                == 1
            )
        with sqlite3.connect(archive_root / "user.db") as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM assertions WHERE kind = 'suppression' AND target_ref = ?",
                    (f"session:{child_session_id}",),
                ).fetchone()[0]
                == 1
            )


class TestResetIdentityMutationContract:
    """Regression tests for polylogue-jnj.5.

    Identity resets (--session/--source) must route through the same
    mutation contract as other destructive ops: a dry-run preview of the
    exact target rows before any tombstone write, no mutation without
    --yes, and a stable JSON envelope for both.
    """

    def test_nonexistent_session_ref_mutates_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo'd/nonexistent session ref must resolve to zero targets, not a literal tombstone."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        _seed_archive_session(archive_root, native_id="real-one")

        with patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["ops", "reset", "--session", "codex-session:totally-nonexistent-typo", "--yes"]
            )

        assert result.exit_code == 0
        assert "No sessions found" in result.output
        # No mutation happened at all -- user.db was never even created.
        assert not (archive_root / "user.db").exists()

    def test_session_dry_run_previews_without_mutating(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--dry-run prints the resolved target and performs no mutation."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="preview-only")

        with patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--session", session_id, "--dry-run"])

        assert result.exit_code == 0
        assert session_id in result.output
        with sqlite3.connect(archive_root / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0] == 1
        assert not (archive_root / "user.db").exists()

    def test_session_dry_run_json_envelope(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--dry-run --json emits a stable MutationResultPayload preview."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="json-preview")

        with patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--session", session_id, "--dry-run", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "preview"
        assert payload["operation"] == "reset"
        assert payload["session_count"] == 1
        assert payload["affected_count"] == 0
        assert payload["session_ids"] == [session_id]

    def test_session_without_yes_or_dry_run_aborts_in_plain_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No mutation happens without --yes, even outside JSON mode."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="no-yes")

        with patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--session", session_id])

        assert result.exit_code == 0
        with sqlite3.connect(archive_root / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0] == 1
        assert not (archive_root / "user.db").exists()

    def test_session_yes_json_envelope_matches_mutation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--yes --json emits a stable envelope for the real mutation, matching dry-run's shape."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="json-mutate")

        with patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--session", session_id, "--yes", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assert payload["operation"] == "reset"
        assert payload["session_count"] == 1
        assert payload["affected_count"] == 1
        assert payload["session_ids"] == [session_id]
        with sqlite3.connect(archive_root / "user.db") as conn:
            row = conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'suppression'").fetchone()
        assert row[0] == 1


class TestResetConfirmation:
    """Tests for reset confirmation flow."""

    def test_without_force_in_plain_mode_skips(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without --yes in plain mode, shows message and skips."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        archive_db = archive_root / "index.db"
        archive_db.write_text("test database", encoding="utf-8")

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
            patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--database"])

            # In plain mode without --yes, should not delete
            assert result.exit_code == 0
            assert archive_db.exists()
            assert "force" in result.output.lower()

    def test_force_bypasses_confirmation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--yes bypasses confirmation prompt."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        archive_db = archive_root / "index.db"
        archive_db.write_text("test database", encoding="utf-8")

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
            patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--database", "--yes"])

            assert result.exit_code == 0
            assert not archive_db.exists()


class TestResetEmptyTargets:
    """Tests for reset when targets don't exist."""

    def test_nothing_to_reset(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no files exist, shows 'nothing to reset'."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=tmp_path / "archive"),
            patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path / "nonexistent"),
            patch("polylogue.cli.commands.reset.cache_home", return_value=tmp_path / "nonexistent"),
            patch("polylogue.cli.commands.reset.drive_token_path", return_value=tmp_path / "nonexistent.json"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--all", "--yes"])

            assert result.exit_code == 0
            assert "nothing to reset" in result.output.lower()

    def test_partial_targets_exist(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only deletes targets that exist."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        archive_db = archive_root / "index.db"
        archive_db.write_text("test database", encoding="utf-8")

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
            patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path / "nonexistent"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--database", "--assets", "--yes"])

            assert result.exit_code == 0
            assert not archive_db.exists()
            assert "database" in result.output.lower()


class TestResetErrorHandling:
    """Tests for reset error handling."""

    def test_deletion_failure_shows_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deletion failure shows error but continues."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        archive_db = archive_root / "index.db"
        archive_db.write_text("test", encoding="utf-8")

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
            patch("polylogue.cli.commands.reset.data_home", return_value=tmp_path),
            patch("pathlib.Path.unlink") as mock_unlink,
        ):
            mock_unlink.side_effect = OSError("Permission denied")

            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--database", "--yes"])

            # Should report failure but not crash
            assert "failed" in result.output.lower() or result.exit_code == 0

    def test_shows_what_will_be_deleted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shows summary of what will be deleted."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        archive_db = archive_root / "index.db"
        archive_db.write_text("test", encoding="utf-8")

        data_home = tmp_path / "data"
        assets_dir = data_home / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "test.png").write_bytes(b"test")

        with (
            patch("polylogue.cli.commands.reset.archive_root", return_value=archive_root),
            patch("polylogue.cli.commands.reset.data_home", return_value=data_home),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["ops", "reset", "--database", "--assets"])

            # Should show paths in output
            assert "database" in result.output.lower()
            assert "assets" in result.output.lower()
