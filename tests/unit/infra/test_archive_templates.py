"""Regression coverage for reusable SQLite archive templates."""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.infra.archive_templates import _template_key, clone_archive_template, finalize_archive_template
from tests.infra.workload_artifacts import ImmutableTreeArtifact


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
    # Source validation refuses before reserving or copying a destination.
    assert not clone.exists()


def test_clone_preserves_original_proof_and_both_owned_roots_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clone retains original proof and executes a destination-owned train."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.durable_tier_fixtures import ship_synthetic_source_train

    template = tmp_path / "template"
    clone = tmp_path / "clone"
    marker = Path(".maintenance-state/durable-change-trains/source-002.json")
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path / "configured")
    ship_synthetic_source_train(tmp_path / "train-package", monkeypatch)
    with ArchiveStore(template):
        pass
    finalize_archive_template(template)
    source_identity = template.joinpath(marker).read_bytes()

    clone_archive_template(template, clone)

    assert clone.joinpath(marker).read_bytes() != source_identity
    with ArchiveStore.open_existing(template, read_only=True):
        pass
    with ArchiveStore(clone):
        pass
    assert template.joinpath(marker).read_bytes() == source_identity
    original = template / ".maintenance-state/durable-change-trains/source-002.json"
    regenerated = clone / ".maintenance-state/durable-change-trains/source-002.json"
    assert original.read_bytes() != regenerated.read_bytes()
    source_manifest_id = ImmutableTreeArtifact.adopt(template, key=_template_key(template)).manifest_id
    source_namespace = hashlib.sha256(source_manifest_id.encode()).hexdigest()
    provenance = clone / ".archive-population-provenance" / source_namespace / "original-history/source-002.json"
    assert provenance.read_bytes() == original.read_bytes()
    provenance_record = json.loads((provenance.parent.parent / "source.json").read_text())
    assert provenance_record["source_manifest_id"] == source_manifest_id
    assert provenance_record["owning_artifact"] is None


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

    assert len(calls) == 1
    assert calls[0][:3] == ["cp", "-a", "--reflink=always"]
    assert str(template / "index.db") in calls[0][3:-1]
    assert calls[0][-1] == str(destination)
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

    def tampered_copy(source: Path, target: Path, **_kwargs: object) -> None:
        (target / "index.db").write_bytes(b"tampered")

    monkeypatch.setattr(subprocess, "run", divergent_copy)
    monkeypatch.setattr("tests.infra.workload_artifacts._copy_tree", tampered_copy)

    with pytest.raises(ValueError, match="authenticated file-set validation"):
        clone_archive_template(template, destination)
    assert not destination.exists()


def _archive_state(root: Path) -> dict[str, object]:
    """Compare tier schemas and rows while each root retains its own train history."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS

    state: dict[str, object] = {}
    for spec in ARCHIVE_TIER_SPECS.values():
        path = root / spec.filename
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA busy_timeout = 5000")
            objects = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
            state[f"{spec.filename}:objects"] = objects
            state[f"{spec.filename}:user_version"] = conn.execute("PRAGMA user_version").fetchone()[0]
            state[f"{spec.filename}:journal_mode"] = conn.execute("PRAGMA journal_mode").fetchone()[0]
            for kind, name, _sql in objects:
                if kind != "table" or name.startswith("sqlite_"):
                    continue
                with contextlib.suppress(sqlite3.DatabaseError):
                    state[f"{spec.filename}:{name}:rows"] = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
    # SQLite coordination files carry no persistent archive inventory;
    # table rows above compare any WAL-visible data.
    sqlite_sidecars = {
        f"{spec.filename}{suffix}" for spec in ARCHIVE_TIER_SPECS.values() for suffix in ("-wal", "-shm")
    }
    state["files"] = sorted(
        entry.name
        for entry in root.iterdir()
        if entry.name != ".archive-ownership.lock" and entry.name not in sqlite_sidecars
    )
    state["bootstrap_marker"] = root.joinpath(".maintenance-state/durable-change-trains/.bootstrap").is_file()
    return state


def test_fixture_bootstrap_executes_the_canonical_baseline_and_train(tmp_path: Path) -> None:
    """Each destination must carry its own actually executed train history."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from tests.infra.archive_templates import bootstrap_archive_root

    produced = tmp_path / "produced"
    initialize_active_archive_root(produced)

    cloned = bootstrap_archive_root(tmp_path / "cloned")

    assert _archive_state(cloned) == _archive_state(produced)
    with sqlite3.connect(cloned / "source.db") as conn:
        assert conn.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    assert _archive_state(cloned) != _archive_state(produced)
    with sqlite3.connect(cloned / "source.db") as conn:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    (cloned / "extra-durable-member").write_bytes(b"extra")
    assert _archive_state(cloned) != _archive_state(produced)


def test_fixture_bootstrap_preserves_an_existing_seeded_root(tmp_path: Path) -> None:
    """A destination that already holds state must not be replaced by a clone.

    Anti-vacuity: cloning over it would drop the planted row, so the read below
    fails if fixture construction replaces an already populated destination.
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


def test_fixture_bootstrap_creates_no_copied_template_history(tmp_path: Path) -> None:
    from tests.infra.archive_templates import bootstrap_archive_root

    root = bootstrap_archive_root(tmp_path / "plain")
    assert (root / "index.db").is_file()
    assert not list(tmp_path.glob(".bootstrap-archive-template*"))


def test_clone_accepts_only_an_empty_destination_reservation(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "payload").write_bytes(b"synthetic fixture")
    destination = tmp_path / "empty"
    destination.mkdir()
    clone_archive_template(template, destination)
    assert (destination / "payload").read_bytes() == b"synthetic fixture"

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "custody").write_bytes(b"unrelated evidence")
    with pytest.raises(ValueError, match="not empty"):
        clone_archive_template(template, occupied)
    assert {path.name for path in occupied.iterdir()} == {"custody"}
    assert (occupied / "custody").read_bytes() == b"unrelated evidence"
    with pytest.raises(ValueError, match="same"):
        clone_archive_template(template, template)
    assert (template / "payload").read_bytes() == b"synthetic fixture"


def test_clone_refuses_symlink_ancestor_before_touching_destination(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "payload").write_bytes(b"synthetic fixture")
    original = tmp_path / "original"
    original.mkdir()
    (original / "custody").write_bytes(b"unrelated evidence")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        clone_archive_template(template, alias / "original")
    assert (original / "custody").read_bytes() == b"unrelated evidence"


def test_clone_exclusive_recreation_refuses_a_competing_creator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "payload").write_bytes(b"synthetic fixture")
    destination = tmp_path / "empty"
    destination.mkdir()
    actual = Path.rmdir

    def competing_creator(path: Path) -> None:
        actual(path)
        if path == destination:
            path.mkdir()
            (path / "custody").write_bytes(b"concurrent creator")

    monkeypatch.setattr(Path, "rmdir", competing_creator)
    with pytest.raises(FileExistsError):
        clone_archive_template(template, destination)
    assert {path.name for path in destination.iterdir()} == {"custody"}
    assert (destination / "custody").read_bytes() == b"concurrent creator"


def test_generic_clone_rejects_a_borrowed_manifest_before_making_it_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    import tests.infra.workload_artifacts as artifacts

    tree = artifacts.build_immutable_tree(
        cache_root=tmp_path / "cache",
        key="literal-tree",
        builder=lambda root: (root / "payload").write_bytes(b"synthetic fixture"),
    )
    source_manifest = tree.root / "manifest.json"
    original_bytes = source_manifest.read_bytes()
    original_mode = source_manifest.stat().st_mode
    copy = artifacts._copy_tree

    def borrowed_manifest(source: Path, target: Path, **kwargs: Any) -> None:
        copy(source, target, **kwargs)
        (target / "manifest.json").unlink()
        os.link(source / "manifest.json", target / "manifest.json")

    monkeypatch.setattr(artifacts, "_copy_tree", borrowed_manifest)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["cp"])),
    )
    with pytest.raises(ValueError, match="inode was not detached"):
        artifacts.clone_immutable_tree(tree, tmp_path / "clone")
    assert source_manifest.read_bytes() == original_bytes
    assert source_manifest.stat().st_mode == original_mode


def test_empty_template_cannot_be_consumed_as_its_own_reservation(tmp_path: Path) -> None:
    template = tmp_path / "empty-template"
    template.mkdir()
    with pytest.raises(ValueError, match="same"):
        clone_archive_template(template, template)
    assert template.is_dir()
    assert list(template.iterdir()) == []
