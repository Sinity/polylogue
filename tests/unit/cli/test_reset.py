"""Tests for reset command."""

from __future__ import annotations

import contextlib
import importlib
import json
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace
from tests.infra.daemon_operations import DaemonOperationStack, cli_daemon_archive


def test_reset_session_resolution_uses_readonly_database_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_module = importlib.import_module("polylogue.cli.commands.reset")
    index_db = tmp_path / "index.db"
    with sqlite3.connect(index_db) as writer:
        writer.execute("CREATE TABLE sessions (session_id TEXT)")
        writer.execute("INSERT INTO sessions VALUES ('selected')")

    real_open = reset_module.open_readonly_connection
    observed = []

    def checked_open(path: Path, **kwargs: Any) -> sqlite3.Connection:
        conn = cast(sqlite3.Connection, real_open(path, **kwargs))
        observed.append(path)
        for statement in (
            "INSERT INTO sessions VALUES ('wrong')",
            "UPDATE sessions SET session_id = 'wrong'",
            "DELETE FROM sessions",
            "CREATE TABLE unwanted (value TEXT)",
            "PRAGMA query_only = OFF",
            "ATTACH DATABASE ':memory:' AS writable",
        ):
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute(statement)
        return conn

    monkeypatch.setattr(reset_module, "_index_db_path", lambda: index_db)
    monkeypatch.setattr(reset_module, "open_readonly_connection", checked_open)
    assert reset_module._resolve_archive_session_ids(["selected"]) == ["selected"]
    assert observed == [index_db]


def _no_seed(_root: Path) -> None:
    """Default seed for cases that only need a bootstrapped archive."""
    return None


@contextlib.contextmanager
def _daemon_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed: Callable[[Path], Any] = _no_seed,
) -> Iterator[tuple[DaemonOperationStack, Any]]:
    """Run `ops reset` against a real daemon rooted at ``tmp_path/archive``.

    `ops reset` is no longer a writer: it previews and confirms locally, then
    lowers to `maintenance.reset` / `mutation.identity-reset`, which
    `configured_mutation_operation` will only send to a resident daemon. Cases
    that assert what a reset actually deletes therefore have to supply that
    daemon.

    ``HOME`` and the XDG roots are redirected under ``tmp_path`` rather than
    patching ``polylogue.cli.commands.reset.data_home`` and friends: the
    deleting code is ``_reset_targets`` in the daemon, which resolves its own
    paths through :mod:`polylogue.paths`. Patching the CLI module attribute
    would relocate only the preview and leave the writer aimed at the real
    home directory.
    """

    seeded: dict[str, Any] = {}

    def _seed(root: Path) -> None:
        seeded["value"] = seed(root)

    with cli_daemon_archive(tmp_path / "archive", monkeypatch, seed_archive=_seed, home=tmp_path / "home") as stack:
        yield stack, seeded.get("value")


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


def _assert_no_suppression_recorded(archive_root: Path) -> None:
    """Assert the reset recorded no durable user-tier mutation.

    These cases used to assert ``user.db`` did not exist, reading a missing
    tier file as proof that nothing happened. That stopped being a signal once
    archive initialization began materializing all six tiers up front — the
    seeding helper here calls ``initialize_active_archive_root`` itself, so the
    file is present before the command under test even runs. Counting the
    suppression rows the mutation would have written tests the actual contract
    and matches how the positive cases in this module check the other side.
    """
    user_db = archive_root / "user.db"
    if not user_db.exists():
        return
    with sqlite3.connect(user_db) as conn:
        recorded = conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'suppression'").fetchone()[0]
    assert recorded == 0


def _seed_archive_session(archive_root: Path, *, native_id: str, source_path: Path | None = None) -> str:
    initialize_active_archive_root(archive_root)
    source_db = archive_root / "source.db"
    index_db = ArchiveLocation.resolve(archive_root).active_index_path
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

    def test_reset_force_database_without_a_daemon_refuses(self, tmp_path: Path) -> None:
        """(a) typed refusal -- the end-to-end offline case.

        These two subprocess cases are the one place in this file answered with
        (a) rather than (b), and deliberately so: an isolated CLI subprocess
        with no `polylogued run` behind it *is* the offline scenario, and it is
        the only check here that exercises a genuinely separate process. They
        are this file's strongest anti-vacuity guard -- reintroduce an
        in-process reset writer and both commands start succeeding offline,
        turning these red.
        """
        from tests.infra.source_builders import GenericSessionBuilder

        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]
        inbox = workspace["paths"]["inbox"]

        (GenericSessionBuilder("to-delete").add_user("will be deleted").write_to(inbox / "test.json"))
        run_cli(["--plain", "run", "parse"], env=env)

        archive_db = Path(workspace["paths"]["archive_root"]) / "index.db"
        assert archive_db.exists()

        result = run_cli(["--plain", "ops", "reset", "--database", "--yes"], env=env)

        assert result.exit_code != 0
        assert "daemon is unavailable; it must execute maintenance.reset" in result.output
        assert archive_db.exists(), "an offline CLI must not delete the archive it was refused permission to touch"

    def test_reset_all_flag_without_a_daemon_refuses(self, tmp_path: Path) -> None:
        """(a) typed refusal: --all is lowered to the daemon like any other target set."""
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["--plain", "ops", "reset", "--all", "--yes"], env=env)

        assert result.exit_code != 0
        assert "daemon is unavailable; it must execute maintenance.reset" in result.output


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
        """(b) daemon route: --all enables every reset target and the daemon applies them."""
        with _daemon_reset(tmp_path, monkeypatch) as (_stack, _seeded):
            result = CliRunner().invoke(cli, ["ops", "reset", "--all", "--yes"])

            assert result.exit_code == 0, result.output


class TestResetCommandDeletion:
    """Tests for reset file/directory deletion."""

    @pytest.mark.parametrize("flag,path_attr,desc", RESET_DELETION_CASES)
    def test_reset_flag_deletes_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, path_attr: str, desc: str
    ) -> None:
        """(b) daemon route: each flag's target is deleted by the daemon that owns the write.

        The target is materialised at the location ``_reset_targets`` resolves
        in the daemon (``polylogue.paths`` under the redirected XDG roots),
        not at a path patched into the CLI module -- the CLI no longer does the
        deleting, so a CLI-side patch would prove nothing.
        """
        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            from polylogue.paths import cache_home, data_home, drive_token_path

            if path_attr in {"archive_db", "index_db"}:
                target_path = stack.archive_root / "index.db"
            elif path_attr == "assets_dir":
                target_path = data_home() / "assets"
                target_path.mkdir(parents=True, exist_ok=True)
                (target_path / "test.png").write_bytes(b"test")
            elif path_attr == "cache_dir":
                target_path = cache_home()
                target_path.mkdir(parents=True, exist_ok=True)
                (target_path / "index").write_text("index data", encoding="utf-8")
            elif path_attr == "token_path":
                target_path = drive_token_path()
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(json.dumps({"token": "test"}), encoding="utf-8")
            else:
                raise AssertionError(f"Unhandled reset target fixture: {path_attr}")

            assert target_path.exists()

            result = CliRunner().invoke(cli, ["ops", "reset", flag, "--yes"])

            assert result.exit_code == 0, result.output
            assert not target_path.exists()

    def test_multiple_flags(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """(b) daemon route: several flags in one request are all applied."""
        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            from polylogue.paths import data_home

            archive_db = stack.archive_root / "index.db"
            assets_dir = data_home() / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)
            (assets_dir / "keep.png").write_bytes(b"keep")
            assert archive_db.exists()

            result = CliRunner().invoke(cli, ["ops", "reset", "--database", "--assets", "--yes"])

            assert result.exit_code == 0, result.output
            assert not archive_db.exists()
            assert not assets_dir.exists()

    def _seed_archive_tiers(self, archive_root: Path) -> tuple[Path, list[Path], Path]:
        """Seed placeholder tier files for the cases that never open them.

        Only the managed-active-generation refusals still use this: they are
        refused by the CLI before anything is dispatched or opened, so cheap
        placeholder files are enough. Every case that reaches the daemon uses
        :meth:`_tier_paths` against a really bootstrapped archive instead.
        """
        archive_root.mkdir(exist_ok=True)
        source_db = archive_root / "source.db"
        rebuildable = [
            archive_root / "index.db",
            archive_root / "index.db-wal",
            archive_root / "index.db-shm",
            archive_root / "ops.db",
        ]
        user_db = archive_root / "user.db"
        initialize_archive_database(source_db, ArchiveTier.SOURCE)
        for path in [*rebuildable, user_db]:
            path.write_text("test database", encoding="utf-8")
        return source_db, rebuildable, user_db

    @staticmethod
    def _tier_paths(archive_root: Path) -> tuple[Path, list[Path], Path]:
        """Return (source.db, rebuildable tiers that exist, user.db).

        This used to fabricate each tier by writing the text "test database"
        over it, which was only viable while nothing opened the archive. The
        daemon that now performs the deletion opens every tier at startup, so
        the tiers have to be real databases -- the bootstrapped ones the
        fixture already creates. Resolving the list after the daemon is up also
        catches the -wal/-shm sidecars it opened, which the old pre-seeded list
        could not.
        """
        candidates = [
            archive_root / "index.db",
            archive_root / "index.db-wal",
            archive_root / "index.db-shm",
            archive_root / "ops.db",
        ]
        return archive_root / "source.db", [path for path in candidates if path.exists()], archive_root / "user.db"

    def test_reset_index_deletes_only_index_tier(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``reset --index`` rebuilds the index tier without dropping raw or user evidence."""
        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            archive_root = stack.archive_root
            source_db, rebuildable, user_db = self._tier_paths(archive_root)
            index_targets = {
                archive_root / "index.db",
                archive_root / "index.db-wal",
                archive_root / "index.db-shm",
            }
            preserved = [path for path in [source_db, *rebuildable, user_db] if path not in index_targets]

            result = CliRunner().invoke(cli, ["ops", "reset", "--index", "--yes"])

        assert result.exit_code == 0, result.output
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
        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            source_db, rebuildable, user_db = self._tier_paths(stack.archive_root)
            result = CliRunner().invoke(cli, ["ops", "reset", "--database", "--yes"])

        assert result.exit_code == 0, result.output
        assert all(not path.exists() for path in rebuildable), "rebuildable tiers should be deleted"
        assert source_db.exists(), "source.db is durable acquired evidence and must survive a plain --database reset"
        assert user_db.exists(), "user.db is irreplaceable and must survive a plain --database reset"
        assert "Preserving source.db" in result.output
        assert "Preserving user.db" in result.output

    def test_reset_database_preserves_the_expensive_embeddings_tier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``reset --database`` must not delete embeddings.db, and must say so.

        Anti-vacuity: re-adding ``embeddings.db`` to the reset target list --
        in the CLI preview or in the daemon's ``_reset_targets``, which is what
        actually unlinks -- deletes the file and turns the existence assertion
        red; dropping the preservation line turns the output assertion red.
        Nothing replays those vectors from source.db, so a silent delete is a
        repurchase billed to the operator, not a rebuild.
        """
        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            embeddings_db = stack.archive_root / "embeddings.db"
            assert embeddings_db.exists(), "fixture must bootstrap the embeddings tier"
            result = CliRunner().invoke(cli, ["ops", "reset", "--database", "--yes"])

        assert result.exit_code == 0, result.output
        assert embeddings_db.exists(), "embeddings.db is expensive_rebuild and must survive --database"
        assert "Preserving embeddings.db" in result.output
        assert "embedding-preservation" in result.output

    def test_reset_database_include_source_and_user_db_deletes_everything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Destructive tier flags explicitly opt into deleting source.db and user.db."""
        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            source_db, rebuildable, user_db = self._tier_paths(stack.archive_root)
            result = CliRunner().invoke(
                cli,
                ["ops", "reset", "--database", "--include-source-db", "--include-user-db", "--yes"],
            )

        assert result.exit_code == 0, result.output
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
        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            source_db, _rebuildable, user_db = self._tier_paths(stack.archive_root)
            result = CliRunner().invoke(cli, ["ops", "reset", "--all", "--yes"])

        assert result.exit_code == 0, result.output
        assert source_db.exists(), "source.db must survive --all without an explicit --include-source-db opt-in"
        assert user_db.exists(), "user.db must survive --all without an explicit --include-user-db opt-in"

    def test_reset_session_records_archive_suppression_and_deletes_archive_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Session tombstone is user-tier suppression plus archive-row deletion."""

        def seed(root: Path) -> tuple[str, Path]:
            # A managed generation lives under the archive root; a pointer naming
            # a target outside it is refused, which is what the clone hardening
            # added.
            active = root / ".index-generations" / "gen-reset" / "index.db"
            active.parent.mkdir(parents=True, exist_ok=True)
            (root / ".index-active-pointer").write_text(str(active), encoding="utf-8")
            return _seed_archive_session(root, native_id="reset-one"), active

        with _daemon_reset(tmp_path, monkeypatch, seed) as (stack, seeded):
            archive_root = stack.archive_root
            session_id, active_index = seeded
            result = CliRunner().invoke(cli, ["ops", "reset", "--session", session_id, "--yes"])

        assert result.exit_code == 0, result.output
        assert "1 suppression" in result.output
        assert "1 archive row" in result.output
        with sqlite3.connect(active_index) as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0] == 0
        # The active index is redirected by .index-active-pointer, so the
        # reset must act on the generation the pointer names -- asserted above.
        # The root-level index.db is a bootstrapped empty tier, not a second
        # live copy: it must not be carrying the session that was just reset.
        # (This previously asserted the root file did not exist at all, which
        # was a fact about the old hand-rolled fixture rather than about reset;
        # a real archive root always has the tier present.)
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
        source_root = tmp_path / "sources" / "codex"

        def seed(root: Path) -> tuple[str, str]:
            return (
                _seed_archive_session(root, native_id="source-child", source_path=source_root / "session.jsonl"),
                _seed_archive_session(
                    root, native_id="source-sibling", source_path=tmp_path / "sources" / "codex-other.jsonl"
                ),
            )

        with _daemon_reset(tmp_path, monkeypatch, seed) as (stack, seeded):
            archive_root = stack.archive_root
            child_session_id, sibling_session_id = seeded
            result = CliRunner().invoke(cli, ["ops", "reset", "--source", str(source_root), "--yes"])

        assert result.exit_code == 0, result.output
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
        _assert_no_suppression_recorded(archive_root)

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
        _assert_no_suppression_recorded(archive_root)

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
        _assert_no_suppression_recorded(archive_root)

    def test_session_yes_json_envelope_matches_mutation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--yes --json emits a stable envelope for the real mutation, matching dry-run's shape."""
        with _daemon_reset(
            tmp_path, monkeypatch, lambda root: _seed_archive_session(root, native_id="json-mutate")
        ) as (stack, seeded):
            archive_root = stack.archive_root
            session_id = str(seeded)
            result = CliRunner().invoke(cli, ["ops", "reset", "--session", session_id, "--yes", "--json"])

        assert result.exit_code == 0, result.output
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
        """(b) daemon route: --yes bypasses the prompt and the daemon performs the delete."""
        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            archive_db = stack.archive_root / "index.db"
            assert archive_db.exists()
            result = CliRunner().invoke(cli, ["ops", "reset", "--database", "--yes"])

            assert result.exit_code == 0, result.output
            assert not archive_db.exists()


class TestResetEmptyTargets:
    """Tests for reset when targets don't exist."""

    def test_nothing_to_reset(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """(b) daemon route: a request whose targets are all absent is a no-op.

        The flags are narrowed to the two targets that genuinely do not exist
        in a fresh workspace. ``--all`` no longer expresses "nothing exists":
        a real archive root always has its six tiers, so --all would have real
        work to do and the assertion would be testing the fixture, not reset.

        Both halves of "nothing to do" are covered, because they are now two
        different code paths: the "Nothing to reset" message belongs to the
        CLI's preview branch (reachable only without --yes), while a confirmed
        request goes to the daemon and comes back reporting nothing deleted.
        """
        with _daemon_reset(tmp_path, monkeypatch) as (_stack, _seeded):
            from polylogue.paths import cache_home, drive_token_path

            assert not cache_home().exists()
            assert not drive_token_path().exists()

            # Without --yes this is a pure preview and never reaches a writer:
            # the CLI reports that the selected targets do not exist.
            preview = CliRunner().invoke(cli, ["ops", "reset", "--cache", "--auth"])
            assert preview.exit_code == 0, preview.output
            assert "nothing to reset" in preview.output.lower()

            # With --yes the confirmed intent is lowered to the daemon, which
            # re-resolves the same empty target set and deletes nothing.
            result = CliRunner().invoke(cli, ["ops", "reset", "--cache", "--auth", "--yes"])

            assert result.exit_code == 0, result.output
            assert "0 item(s) deleted" in result.output

    def test_partial_targets_exist(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """(b) daemon route: only the targets that exist are deleted."""
        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            from polylogue.paths import data_home

            archive_db = stack.archive_root / "index.db"
            assert archive_db.exists()
            assert not (data_home() / "assets").exists()

            result = CliRunner().invoke(cli, ["ops", "reset", "--database", "--assets", "--yes"])

            assert result.exit_code == 0, result.output
            assert not archive_db.exists()
            assert "database" in result.output.lower()


class TestResetErrorHandling:
    """Tests for reset error handling."""

    def test_deletion_failure_is_reported_not_swallowed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """(b) daemon route: a delete that cannot be performed surfaces as a refusal.

        The failure is produced for real -- the archive directory is made
        unwritable so the unlink genuinely fails -- rather than by patching
        ``pathlib.Path.unlink``. A global patch is no longer usable here: the
        unlink happens inside the daemon, in this same process, so the mock
        would break the daemon's own bookkeeping instead of the target delete.

        The old assertion was the disjunction ``"failed" in output or exit_code
        == 0``, which any successful run satisfied. The contract asserted now
        is the one the daemon actually provides: the failure is reported and
        the command does not claim success.
        """
        with _daemon_reset(tmp_path, monkeypatch) as (_stack, _seeded):
            from polylogue.paths import data_home

            # The assets tree is the target, so the failure is confined to it:
            # making the archive root itself unwritable would break the daemon's
            # own journals before it ever reached a delete.
            assets = data_home() / "assets"
            locked = assets / "locked"
            locked.mkdir(parents=True)
            (locked / "pinned.png").write_bytes(b"pinned")
            locked.chmod(0o500)
            try:
                result = CliRunner().invoke(cli, ["ops", "reset", "--assets", "--yes"])
            finally:
                locked.chmod(0o700)

            assert result.exit_code != 0
            assert "maintenance.reset" in result.output
            assert "reset complete" not in result.output.lower()
            assert locked.exists(), "the undeletable target must still be there"

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
