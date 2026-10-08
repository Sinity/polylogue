"""Tests for reset command."""

from __future__ import annotations

import contextlib
import importlib
import json
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace
from tests.infra.daemon_operations import DaemonOperationStack, cli_daemon_archive
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture
from tests.infra.operation_recovery import recover_on_admitted_owner


def test_reset_session_resolution_uses_server_owned_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_module = importlib.import_module("polylogue.cli.commands.reset")
    seen: list[tuple[str, dict[str, object]]] = []

    def submit(_env: object, operation: str, payload: dict[str, object]) -> dict[str, object]:
        seen.append((operation, payload))
        return {"result": {"preview_ref": "preview:neutral", "session_count": 1}}

    monkeypatch.setattr(reset_module, "_submit", submit)
    env = SimpleNamespace(config=object())
    assert reset_module._identity_reset_targets(env, conv_id="selected", source_path=None) == (
        "preview:neutral",
        1,
        "session 'selected'",
    )
    assert seen == [("mutation.identity-reset.preview", {"session": "selected", "reason": "reset --session"})]


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
        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

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
        with _daemon_reset(tmp_path, monkeypatch) as (_stack, _seeded):
            from polylogue.paths import cache_home, data_home, drive_token_path

            if path_attr == "assets_dir":
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

    def test_confirmed_reset_without_a_preview_is_not_an_empty_preview(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An omitted ``expected_targets`` asserts nothing; ``[]`` asserts no targets.

        Anti-vacuity: default the protocol field to ``[]`` and the omitted
        request is compared against an empty preview and refused, leaving the
        cache in place.
        """
        from polylogue.daemon_client import DaemonOperationRejectedError

        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            from polylogue.paths import cache_home

            cache = cache_home()
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "index").write_text("index data", encoding="utf-8")

            try:
                refused = stack.client.operation_to_completion(
                    "maintenance.reset",
                    {"cache": True, "confirm": True, "expected_targets": []},
                    archive_root=str(stack.archive_root),
                )
            except DaemonOperationRejectedError:
                refused = None
            assert refused is None or refused["outcome"] != "completed"
            assert cache.exists()

            applied = stack.client.operation_to_completion(
                "maintenance.reset", {"cache": True, "confirm": True}, archive_root=str(stack.archive_root)
            )
            assert applied is not None and applied["outcome"] == "completed", applied
            assert not cache.exists()

    def test_confirmation_names_databases_not_their_transient_sidecars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A WAL sidecar appearing after the preview does not count as a changed target set.

        Anti-vacuity: compare sidecars as part of the confirmed identity and the
        index preview taken before ``index.db-wal`` exists is refused as a
        changed target set instead of staged; the added cache directory must
        still be refused.
        """
        from polylogue.daemon_client import DaemonOperationRejectedError

        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            from polylogue.paths import cache_home

            index_db = stack.archive_root / "index.db"
            preview = [str(index_db.resolve())]
            index_wal = index_db.with_name("index.db-wal")
            if not index_wal.exists():
                index_wal.write_bytes(b"")

            cache = cache_home()
            cache.mkdir(parents=True, exist_ok=True)
            try:
                changed = stack.client.operation_to_completion(
                    "maintenance.reset",
                    {"index": True, "cache": True, "confirm": True, "expected_targets": preview},
                    archive_root=str(stack.archive_root),
                )
            except DaemonOperationRejectedError:
                changed = None
            assert changed is None or changed["outcome"] == "rejected", changed
            assert cache.exists()

            staged = stack.client.operation_to_completion(
                "maintenance.reset",
                {"index": True, "confirm": True, "expected_targets": preview},
                archive_root=str(stack.archive_root),
            )
            assert staged is not None and staged["outcome"] == "completed", staged
            assert staged["result"]["result"]["state"] == "staged"
            assert index_db.exists()
            assert index_wal.exists()

    def test_multiple_flags(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """(b) daemon route: several flags in one request are all applied."""
        with _daemon_reset(tmp_path, monkeypatch) as (_stack, _seeded):
            from polylogue.paths import cache_home, data_home

            cache_dir = cache_home()
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "index").write_text("index data", encoding="utf-8")
            assets_dir = data_home() / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)
            (assets_dir / "keep.png").write_bytes(b"keep")

            result = CliRunner().invoke(cli, ["ops", "reset", "--cache", "--assets", "--yes"])

            assert result.exit_code == 0, result.output
            assert not cache_dir.exists()
            assert not assets_dir.exists()

    def _seed_archive_tiers(self, archive_root: Path) -> tuple[Path, list[Path], Path]:
        """Seed placeholder tier files for the cases that never open them.

        Only the managed-active-generation refusals still use this: they are
        refused by the CLI before anything is dispatched or opened, so cheap
        placeholder files are enough. Every case that reaches the daemon uses
        :meth:`_tier_files` against a really bootstrapped archive instead.
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
        initialize_runtime_source_fixture(source_db)
        for path in [*rebuildable, user_db]:
            path.write_text("test database", encoding="utf-8")
        return source_db, rebuildable, user_db

    @staticmethod
    def _tier_files(archive_root: Path) -> list[Path]:
        """Every archive tier database and sidecar the serving daemon has open."""
        return sorted(
            path
            for tier in ArchiveTier
            for suffix in ("", "-wal", "-shm")
            if (path := archive_root / f"{tier.value}.db{suffix}").exists()
        )

    @staticmethod
    def _reset_runs(archive_root: Path) -> list[tuple[str, str]]:
        with sqlite3.connect(archive_root / "audit.db") as conn:
            return [
                (str(row[0]), str(row[1]))
                for row in conn.execute(
                    "SELECT operation_id, status FROM operation_runs WHERE operation_name = 'mutate-filesystem-reset'"
                )
            ]

    @staticmethod
    def _end_attempt_owner(archive_root: Path, operation_id: str) -> None:
        """Stand in for the daemon process that staged the reset having exited."""
        with sqlite3.connect(archive_root / "audit.db") as conn:
            conn.execute(
                "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
            )

    @pytest.mark.parametrize(
        ("flags", "deleted_tiers"),
        [
            (["--index"], ("index.db",)),
            (["--database"], ("index.db", "ops.db")),
            (["--all"], ("index.db", "ops.db")),
            (["--database", "--assets"], ("index.db", "ops.db")),
        ],
        ids=["index", "database", "all", "database-and-assets"],
    )
    def test_reset_of_live_archive_tiers_is_staged_and_applied_before_the_next_start_opens_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], deleted_tiers: tuple[str, ...]
    ) -> None:
        """The daemon never unlinks tier databases it holds open (polylogue-9kemf 07.F012).

        Unlinking ``index.db``/``ops.db`` and their WAL/SHM sidecars under the
        daemon's watcher cursor store, status registry and readers left them
        writing to deleted inodes while new connections created fresh empty
        files, and the reset reported success. The live request now records
        the authorized plan and a running attempt and deletes nothing -- not
        even a non-tier target named beside a tier -- and says so. Ordinary
        recovery leaves it pending; the startup seam, which runs before any
        tier opens, deletes the whole plan, and bootstrap recreates the tiers.

        Anti-vacuity: apply the plan in ``maintenance_reset`` and the tier
        inodes change under the serving daemon; drop the
        ``archive_tiers_are_closed`` deferral in ``recover`` and ordinary
        recovery deletes the tiers; skip sidecars of a deleted database and a
        ``-wal`` survives beside the recreated file.
        """
        from polylogue.operations.mutation_replay import apply_staged_archive_resets
        from polylogue.storage.sqlite.write_guard import declared_unguarded_write

        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            from polylogue.paths import data_home

            archive_root = stack.archive_root
            assets_dir = data_home() / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)
            (assets_dir / "keep.png").write_bytes(b"keep")
            tier_files = self._tier_files(archive_root)
            inodes = {path: path.stat().st_ino for path in tier_files}
            assert archive_root / "index.db" in tier_files
            assert archive_root / "ops.db" in tier_files

            result = CliRunner().invoke(cli, ["ops", "reset", *flags, "--yes"])

            assert result.exit_code == 0, result.output
            assert "Reset staged" in result.output
            assert "reset complete" not in result.output.lower()
            assert {path: path.stat().st_ino for path in tier_files} == inodes
            assert assets_dir.exists()
            # The daemon that staged the reset still serves from its handles.
            read = stack.client.operation("cli.query", {"params": {}}, archive_root=str(archive_root))
            assert read is not None and read["outcome"] == "completed", read

        [(operation_id, status)] = self._reset_runs(archive_root)
        assert status == "running"
        self._end_attempt_owner(archive_root, operation_id)
        # Ordinary recovery runs after tiers open: it must leave the plan pending.
        with declared_unguarded_write("test: startup recovery outside the tier seam"):
            recover_on_admitted_owner(archive_root)
        # It may mark the dead attempt interrupted, but the plan stays nonterminal.
        [(_operation_id, status)] = self._reset_runs(archive_root)
        assert status in {"running", "interrupted"}
        assert all((archive_root / name).exists() for name in deleted_tiers)

        with declared_unguarded_write("test: daemon startup seam"):
            assert apply_staged_archive_resets(archive_root) == (operation_id,)

        [(_operation_id, status)] = self._reset_runs(archive_root)
        assert status not in {"running", "interrupted"}
        for name in deleted_tiers:
            for suffix in ("", "-wal", "-shm"):
                assert not (archive_root / f"{name}{suffix}").exists(), f"{name}{suffix}"
        for name in ("source.db", "user.db", "audit.db", "embeddings.db"):
            assert (archive_root / name).exists(), name
        assert assets_dir.exists() is ("--assets" not in flags and "--all" not in flags)
        with declared_unguarded_write("test: re-bootstrap after the staged reset"):
            initialize_active_archive_root(archive_root)
        assert (archive_root / "index.db").exists()
        assert (archive_root / "ops.db").exists()

    def test_reset_target_holding_a_durable_tier_is_refused_before_any_audit_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A directory target that holds tier databases is never staged or deleted.

        Anti-vacuity: drop the ``unresettable`` refusal in ``maintenance_reset``
        and the request stages a plan whose recovery would delete audit.db,
        source.db and user.db with the directory.
        """
        from polylogue.daemon_client import DaemonOperationRejectedError

        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            archive_root = stack.archive_root
            with patch("polylogue.paths.blob_store_root", return_value=archive_root):
                try:
                    envelope = stack.client.operation_to_completion(
                        "maintenance.reset",
                        {"blob": True, "confirm": True},
                        archive_root=str(archive_root),
                    )
                except DaemonOperationRejectedError as exc:
                    code = f"{exc.outcome} {exc.detail}"
                else:
                    assert envelope is not None and envelope["outcome"] == "rejected", envelope
                    code = str(envelope["error"]["code"])
            assert "reset_unresettable_archive_tier" in code
            assert self._reset_runs(archive_root) == []
            for name in ("source.db", "user.db", "audit.db", "index.db"):
                assert (archive_root / name).exists(), name

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
        """``--database`` names rebuildable tiers only; durable tiers are never among its targets."""
        from polylogue.operations.daemon_mutations import _reset_targets

        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            names = {name for name, _path in _reset_targets(stack.archive_root, {"database": True})}
            result = CliRunner().invoke(cli, ["ops", "reset", "--database", "--yes"])

        assert {"index database", "ops database"} <= names
        assert not {"source database", "user database"} & names
        assert (stack.archive_root / "source.db").exists()
        assert (stack.archive_root / "user.db").exists()
        assert "Preserving source.db" in result.output
        assert "Preserving user.db" in result.output

    def test_reset_database_preserves_the_expensive_embeddings_tier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No reset target set names embeddings.db, and the CLI says so.

        Anti-vacuity: re-adding ``embeddings.db`` to the daemon's
        ``_reset_targets`` names it among the resolved targets; dropping the
        preservation line turns the output assertion red. Nothing replays those
        vectors from source.db, so a delete is a repurchase billed to the
        operator, not a rebuild.
        """
        from polylogue.operations.daemon_mutations import _reset_targets

        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            embeddings_db = stack.archive_root / "embeddings.db"
            assert embeddings_db.exists(), "fixture must bootstrap the embeddings tier"
            resolved = _reset_targets(stack.archive_root, {"reset_all": True})
            result = CliRunner().invoke(cli, ["ops", "reset", "--database", "--yes"])

        assert all(path.resolve() != embeddings_db.resolve() for _name, path in resolved)
        assert embeddings_db.exists(), "embeddings.db is expensive_rebuild and must survive --database"
        assert "Preserving embeddings.db" in result.output
        assert "reused after an index rebuild" in result.output

    def test_reset_all_names_no_durable_tier(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``reset --all`` names no durable tier: an archive missing one refuses to open."""
        from polylogue.operations.daemon_mutations import _reset_targets

        with _daemon_reset(tmp_path, monkeypatch) as (stack, _seeded):
            paths = {path.resolve() for _name, path in _reset_targets(stack.archive_root, {"reset_all": True})}

        for name in ("source.db", "user.db", "audit.db"):
            assert (stack.archive_root / name).resolve() not in paths, name

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
        with _daemon_reset(tmp_path, monkeypatch, lambda root: _seed_archive_session(root, native_id="real-one")) as (
            stack,
            _seeded,
        ):
            archive_root = stack.archive_root
            result = CliRunner().invoke(
                cli, ["ops", "reset", "--session", "codex-session:totally-nonexistent-typo", "--yes"]
            )

        assert result.exit_code == 0
        assert "No sessions found" in result.output
        _assert_no_suppression_recorded(archive_root)

    def test_session_dry_run_previews_without_mutating(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--dry-run prints the resolved target and performs no mutation."""
        with _daemon_reset(
            tmp_path, monkeypatch, lambda root: _seed_archive_session(root, native_id="preview-only")
        ) as (stack, seeded):
            archive_root = stack.archive_root
            session_id = str(seeded)
            result = CliRunner().invoke(cli, ["ops", "reset", "--session", session_id, "--dry-run"])

        assert result.exit_code == 0
        assert session_id in result.output
        with sqlite3.connect(archive_root / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0] == 1
        _assert_no_suppression_recorded(archive_root)

    def test_session_dry_run_json_envelope(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--dry-run --json emits a stable MutationResultPayload preview."""
        with _daemon_reset(
            tmp_path, monkeypatch, lambda root: _seed_archive_session(root, native_id="json-preview")
        ) as (stack, seeded):
            session_id = str(seeded)
            result = CliRunner().invoke(cli, ["ops", "reset", "--session", session_id, "--dry-run", "--json"])

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
        with _daemon_reset(tmp_path, monkeypatch, lambda root: _seed_archive_session(root, native_id="no-yes")) as (
            stack,
            seeded,
        ):
            archive_root = stack.archive_root
            session_id = str(seeded)
            result = CliRunner().invoke(cli, ["ops", "reset", "--session", session_id])

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
        with _daemon_reset(tmp_path, monkeypatch) as (_stack, _seeded):
            from polylogue.paths import cache_home

            cache_dir = cache_home()
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "index").write_text("index data", encoding="utf-8")
            result = CliRunner().invoke(cli, ["ops", "reset", "--cache", "--yes"])

            assert result.exit_code == 0, result.output
            assert not cache_dir.exists()


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
        with _daemon_reset(tmp_path, monkeypatch) as (_stack, _seeded):
            from polylogue.paths import cache_home, data_home

            cache_dir = cache_home()
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "index").write_text("index data", encoding="utf-8")
            assert not (data_home() / "assets").exists()

            result = CliRunner().invoke(cli, ["ops", "reset", "--cache", "--assets", "--yes"])

            assert result.exit_code == 0, result.output
            assert not cache_dir.exists()
            assert "cache/indexes" in result.output


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
