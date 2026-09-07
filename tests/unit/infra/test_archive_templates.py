"""Regression coverage for reusable SQLite archive templates."""

from __future__ import annotations

import contextlib
import sqlite3
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.infra.archive_templates import clone_archive_template, finalize_archive_template


def test_clone_refuses_a_template_holding_a_symlink(tmp_path: Path) -> None:
    """A symlinked tier would let a clone reach outside its own tree.

    Anti-vacuity: accepting the link makes the clone's contents depend on a
    path the fixture does not own, so this assertion is red the moment the
    template route stops going through the shared publication discipline.
    """
    template = tmp_path / "template"
    template.mkdir()
    source_file = template / "source.db"
    with contextlib.closing(sqlite3.connect(source_file)) as conn, conn:
        conn.execute("CREATE TABLE entries (value TEXT)")
        conn.execute("INSERT INTO entries VALUES ('immutable-template')")
    (template / "source-link.db").symlink_to(source_file.name)

    with pytest.raises(ValueError, match="symlink"):
        finalize_archive_template(template)
    clone = tmp_path / "clone"
    with pytest.raises(ValueError, match="symlink"):
        clone_archive_template(template, clone)
    assert list(clone.iterdir()) == []


def test_clone_rebinds_durable_bootstrap_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reusing the source store or global identity would make this reopen unsafe."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    template = tmp_path / "template"
    clone = tmp_path / "clone"
    marker = Path(".maintenance-state/durable-change-trains/.bootstrap")
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path / "configured")
    with ArchiveStore(template):
        pass
    source_identity = template.joinpath(marker).read_bytes()

    clone_archive_template(template, clone)

    assert clone.joinpath(marker).read_bytes() != source_identity
    with ArchiveStore(template):
        pass
    with ArchiveStore(clone):
        pass
    assert template.joinpath(marker).read_bytes() == source_identity


def _leave_crash_recovered_wal(database: Path) -> None:
    writer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import sqlite3
import sys

conn = sqlite3.connect(sys.argv[1])
assert conn.execute(\"PRAGMA journal_mode=WAL\").fetchone() == (\"wal\",)
conn.execute(\"CREATE TABLE entries (value TEXT)\")
conn.execute(\"INSERT INTO entries VALUES ('written-through-wal')\")
conn.commit()
print(\"ready\", flush=True)
sys.stdin.read()
""",
            str(database),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert writer.stdout is not None
    assert writer.stdout.readline() == "ready\n"
    writer.terminate()
    assert writer.wait(timeout=5) != 0


def test_finalized_template_checkpoints_real_wal_before_freezing_and_cloning(tmp_path: Path) -> None:
    """A crash-left WAL must become a self-contained, immutable clone source.

    Anti-vacuity: removing the quiescence phase leaves the real ``-wal`` and
    ``-shm`` files plus WAL journal mode behind, so the sidecar and journal
    assertions below fail before the production clone path can read the row.
    """
    template = tmp_path / "template"
    template.mkdir()
    database = template / "source.db"
    _leave_crash_recovered_wal(database)

    assert (template / "source.db-wal").exists()
    assert (template / "source.db-shm").exists()

    finalize_archive_template(template)
    clone = tmp_path / "clone"
    clone_archive_template(template, clone)

    for root in (template, clone):
        assert not (root / "source.db-wal").exists()
        assert not (root / "source.db-shm").exists()
    assert not (database.stat().st_mode & stat.S_IWUSR)
    assert clone.joinpath("source.db").stat().st_mode & stat.S_IWUSR

    with contextlib.closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    with contextlib.closing(sqlite3.connect(clone / "source.db")) as conn:
        assert conn.execute("SELECT value FROM entries").fetchall() == [("written-through-wal",)]

    # ``stat.S_IWUSR``, not ``os.W_OK``: the latter is 2, which as a mode
    # bit is ``S_IWOTH``, so the check would pass with owner-write set --
    # exactly the bit that decides whether this template is immutable.
    assert not (template.stat().st_mode & stat.S_IWUSR)


def test_clone_requests_reflink_before_copy_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The normal clone path must not silently turn CoW into a full copy."""
    template = tmp_path / "template"
    destination = tmp_path / "clone"
    template.mkdir()
    (template / "index.db").write_bytes(b"snapshot")
    calls: list[list[str]] = []

    def no_reflink(argv: list[str], **kwargs: object) -> None:
        calls.append(argv)
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", no_reflink)
    assert clone_archive_template(template, destination) == "copy"

    assert [argv[:4] for argv in calls] == [["cp", "-a", "--reflink=always", str(template)]]
    assert (destination / "index.db").read_bytes() == b"snapshot"
    assert (destination / "index.db").stat().st_mode & stat.S_IWUSR


def test_clone_fallback_replaces_a_read_only_partial_reflink_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed reflink may leave immutable directories that the fallback replaces."""
    template = tmp_path / "template"
    destination = tmp_path / "clone"
    template.mkdir()
    source = template / "source.db"
    with contextlib.closing(sqlite3.connect(source)) as connection, connection:
        connection.execute("CREATE TABLE entries (value TEXT)")
        connection.execute("INSERT INTO entries VALUES ('complete-template')")
    finalize_archive_template(template)

    def partial_reflink(argv: list[str], **_kwargs: object) -> None:
        target = Path(argv[-1])
        partial = target / "nested" / "partial.db"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"incomplete")
        for path in (partial, partial.parent, target):
            path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", partial_reflink)
    clone_archive_template(template, destination)

    with contextlib.closing(sqlite3.connect(destination / "source.db")) as connection:
        assert connection.execute("SELECT value FROM entries").fetchall() == [("complete-template",)]
    assert not destination.joinpath("nested", "partial.db").exists()


def test_clone_refuses_a_template_that_changed_after_it_was_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A clone is authenticated against the tree it copied, not merely attempted.

    Anti-vacuity: without the file-set check the mutated byte reaches the
    consumer silently, and the clone reports success.
    """
    template = tmp_path / "template"
    destination = tmp_path / "clone"
    template.mkdir()
    (template / "index.db").write_bytes(b"snapshot")

    def divergent_copy(argv: list[str], **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, argv)

    def tampered_copy(source: Path, target: Path) -> None:
        target.mkdir(parents=True)
        (target / "index.db").write_bytes(b"tampered")

    monkeypatch.setattr(subprocess, "run", divergent_copy)
    monkeypatch.setattr("tests.infra.workload_artifacts._copy_tree", tampered_copy)

    with pytest.raises(ValueError, match="authenticated file-set validation"):
        clone_archive_template(template, destination)
    assert list(destination.iterdir()) == []
    assert list(destination.parent.glob(".clone.*")) == []


@pytest.fixture
def bootstrap_template_root(tmp_path: Path) -> Iterator[Path]:
    """Redirect bootstrap cloning at a private run root for one test.

    The session registers its own root for every worker; a test that left the
    process pointing at ``None`` would silently put every later archive in that
    worker back on the production route.
    """
    from tests.infra.archive_templates import register_bootstrap_template_root

    run_root = tmp_path / "run"
    previous = register_bootstrap_template_root(run_root)
    try:
        yield run_root
    finally:
        register_bootstrap_template_root(previous)


def _archive_state(root: Path) -> dict[str, object]:
    """Every tier's schema, version and rows, minus what is bound to the path.

    The durable bootstrap marker names the tree it belongs to, so it is compared
    for presence rather than content; :func:`clone_archive_template` rebinds it.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS

    state: dict[str, object] = {}
    for spec in ARCHIVE_TIER_SPECS.values():
        path = root / spec.filename
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA busy_timeout = 5000")
            objects = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
            state[f"{spec.filename}:objects"] = objects
            state[f"{spec.filename}:user_version"] = conn.execute("PRAGMA user_version").fetchone()[0]
            for kind, name, _sql in objects:
                if kind != "table" or name.startswith("sqlite_"):
                    continue
                with contextlib.suppress(sqlite3.DatabaseError):
                    state[f"{spec.filename}:{name}:rows"] = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
    state["files"] = sorted(entry.name for entry in root.iterdir() if entry.name != ".archive-ownership.lock")
    state["bootstrap_marker"] = root.joinpath(".maintenance-state/durable-change-trains/.bootstrap").is_file()
    return state


def test_bootstrap_clone_reproduces_the_production_bootstrap(tmp_path: Path, bootstrap_template_root: Path) -> None:
    """A cloned root is what the production bootstrap builds, or the clone is a lie.

    Anti-vacuity: seeding the template through any route that adds state the
    production bootstrap does not create -- a completed raw-authority census,
    an extra ops row -- makes the two states diverge and this red.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from tests.infra.archive_templates import bootstrap_archive_root

    produced = tmp_path / "produced"
    initialize_active_archive_root(produced)

    cloned = bootstrap_archive_root(tmp_path / "cloned")

    assert (bootstrap_template_root / ".bootstrap-archive-template").is_dir()
    assert _archive_state(cloned) == _archive_state(produced)


def test_bootstrap_falls_back_to_the_production_route_for_a_seeded_root(
    tmp_path: Path,
    bootstrap_template_root: Path,
) -> None:
    """A destination that already holds state must not be replaced by a clone.

    Anti-vacuity: cloning over it would drop the planted row, so the read below
    fails the moment the pristine-destination guard stops deciding the route.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from tests.infra.archive_templates import bootstrap_archive_root

    seeded = tmp_path / "seeded"
    initialize_active_archive_root(seeded)
    with contextlib.closing(sqlite3.connect(seeded / "ops.db")) as conn, conn:
        conn.execute("CREATE TABLE planted (value TEXT)")
        conn.execute("INSERT INTO planted VALUES ('kept')")

    bootstrap_archive_root(seeded)

    with contextlib.closing(sqlite3.connect(seeded / "ops.db")) as conn:
        assert conn.execute("SELECT value FROM planted").fetchall() == [("kept",)]


def test_bootstrap_without_a_registered_run_root_uses_the_production_route(tmp_path: Path) -> None:
    """No template location means no clone, never a half-built archive."""
    from tests.infra.archive_templates import bootstrap_archive_root, register_bootstrap_template_root

    previous = register_bootstrap_template_root(None)
    try:
        root = bootstrap_archive_root(tmp_path / "plain")
    finally:
        register_bootstrap_template_root(previous)

    assert (root / "index.db").is_file()
    assert not list(tmp_path.glob(".bootstrap-archive-template*"))
