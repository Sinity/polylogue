"""Lossless source-cut laws through the production snapshot interface."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.config import Config, Source
from polylogue.maintenance.rebuild_index import freeze_candidate_source_inputs, verify_frozen_candidate_source_inputs
from polylogue.maintenance.source_manifest_continuity import SourceDeclaration, SourceRole
from polylogue.sources import source_snapshot
from polylogue.sources.source_snapshot import (
    CandidateCohortError,
    SnapshotMode,
    SourceCutPolicy,
    SourceMutationError,
    SourceSnapshotError,
    execute_source_cut,
    preflight_source_cut,
    reacquire_candidate,
)
from polylogue.sources.sqlite_snapshot import snapshot_sqlite_database, sqlite_logical_revision


def test_cut_publishes_immutable_candidate_and_carry_forward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    first = root / "first.jsonl"
    first.write_text("before\n", encoding="utf-8")
    declaration = SourceDeclaration("sessions", SourceRole.APPEND_JSONL, root, True)

    preflight = preflight_source_cut([declaration], request_id="cut-1")
    original_observe = source_snapshot._observe
    calls = 0

    def observe_with_arrival(binding: source_snapshot.SourceCutBinding) -> tuple[source_snapshot.CutItem, ...]:
        nonlocal calls
        calls += 1
        if calls == 2:
            (root / "arrived.jsonl").write_text("after\n", encoding="utf-8")
            first.write_text("before\nafter\n", encoding="utf-8")
        return original_observe(binding)

    # The first post-cut inventory observes both the append and the new file.
    # The callback is installed only around execution, so preflight never
    # claims that mutable bytes remain unchanged.
    monkeypatch.setattr(source_snapshot, "_observe", observe_with_arrival)
    result = execute_source_cut(preflight, tmp_path / "published")

    assert result.counts.conserved
    assert {item.coordinate for item in result.carry_forward_manifest.items} == {"arrived.jsonl", "first.jsonl"}
    assert next(item for item in result.carry_forward_manifest.items if item.coordinate == "first.jsonl").readmission
    assert next(
        item for item in result.carry_forward_manifest.items if item.coordinate == "arrived.jsonl"
    ).post_cut_arrival
    assert result.counts.observed_bytes == len("before\n") + len("after\n")
    candidate = reacquire_candidate(result)
    assert candidate[0].path.read_text(encoding="utf-8") == "before\n"
    with pytest.raises(CandidateCohortError):
        reacquire_candidate(result, coordinates=["not-in-cut.jsonl"])


def test_directory_cut_readmits_a_grown_live_file_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: counting a grown directory member twice inflates observed bytes."""
    root = tmp_path / "source"
    root.mkdir()
    path = root / "live.jsonl"
    path.write_text("first\n", encoding="utf-8")
    preflight = preflight_source_cut([SourceDeclaration("source", SourceRole.DIRECTORY, root, True)])
    original_observe = source_snapshot._observe
    calls = 0

    def observe_after_growth(binding: source_snapshot.SourceCutBinding) -> tuple[source_snapshot.CutItem, ...]:
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_text("first\nextra\n", encoding="utf-8")
        return original_observe(binding)

    monkeypatch.setattr(source_snapshot, "_observe", observe_after_growth)
    result = execute_source_cut(preflight, tmp_path / "cut")

    assert result.counts.conserved
    assert result.counts.observed_bytes == len("first\n")
    assert result.counts.candidate_bytes == len("first\n")
    assert result.counts.carry_forward_bytes == 0
    assert result.carry_forward_manifest.items[0].readmission is True


def test_candidate_bytes_are_checked_after_publication(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "one.json").write_text("one", encoding="utf-8")
    result = execute_source_cut(
        preflight_source_cut([SourceDeclaration("source", SourceRole.IMMUTABLE_EXPORT, root)]), tmp_path / "cut"
    )
    candidate_path = result.candidate_root / "source" / "one.json"
    candidate_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(SourceMutationError, match="candidate snapshot mutated"):
        reacquire_candidate(result)


def test_completion_marker_fsyncs_its_destination_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: syncing only the parent can lose a marker written inside the destination."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "one.json").write_text("one", encoding="utf-8")
    destination = tmp_path / "cut"
    original_write = source_snapshot._write_durable
    original_fsync_directory = source_snapshot._fsync_directory
    marker_written = False
    fsyncs_after_marker: list[Path] = []

    def write_marker(path: Path, payload: str) -> None:
        nonlocal marker_written
        original_write(path, payload)
        if path.name == ".source-cut-complete":
            marker_written = True

    def record_directory_fsync(path: Path) -> None:
        if marker_written:
            fsyncs_after_marker.append(path)
        original_fsync_directory(path)

    monkeypatch.setattr(source_snapshot, "_write_durable", write_marker)
    monkeypatch.setattr(source_snapshot, "_fsync_directory", record_directory_fsync)
    execute_source_cut(
        preflight_source_cut([SourceDeclaration("source", SourceRole.IMMUTABLE_EXPORT, root)]), destination
    )

    assert destination in fsyncs_after_marker


def test_verify_frozen_candidate_source_inputs_rejects_tampered_candidate(tmp_path: Path) -> None:
    """Mutation: accepting a changed cut file would let candidate planning read unsealed bytes."""
    source = tmp_path / "configured-source"
    source.mkdir()
    (source / "session.jsonl").write_text("before\n", encoding="utf-8")
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[Source("configured", source)])
    destination = tmp_path / "cut"
    frozen = freeze_candidate_source_inputs(
        config,
        destination=destination,
        request_id="tamper-check",
        fallback_source_path=tmp_path / "source.db",
    )
    (frozen.candidate_root / "configured-0" / "session.jsonl").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(SourceMutationError, match="candidate snapshot mutated"):
        verify_frozen_candidate_source_inputs(destination)


def test_repeating_a_published_cut_reuses_its_manifest(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "one.json").write_text("one", encoding="utf-8")
    declaration = SourceDeclaration("source", SourceRole.IMMUTABLE_EXPORT, root)
    preflight = preflight_source_cut([declaration], request_id="repeat")
    first = execute_source_cut(preflight, tmp_path / "cut")
    (root / "later.json").write_text("later", encoding="utf-8")
    second = execute_source_cut(preflight, tmp_path / "cut")
    assert second.cut_identity == first.cut_identity
    assert second.candidate_manifest.digest == first.candidate_manifest.digest
    assert second.carry_forward_manifest.digest == first.carry_forward_manifest.digest


def test_preflight_binds_root_identity_and_strategy_without_bytes(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    declaration = SourceDeclaration("source", SourceRole.REWRITE_JSONL, root, True)
    preflight = preflight_source_cut(
        [declaration],
        policies={"source": SourceCutPolicy(SnapshotMode.COMPLETE_COPY, adapter_version="rewrite-v2")},
    )
    assert preflight.bindings[0].policy.mode is SnapshotMode.COMPLETE_COPY
    (root / "new.jsonl").write_text("new", encoding="utf-8")
    assert preflight.bindings[0].root_identity.inode == root.stat().st_ino


def test_replacement_with_identical_bytes_is_carry_forward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "source"
    root.mkdir()
    path = root / "one.jsonl"
    path.write_text("same", encoding="utf-8")
    preflight = preflight_source_cut(
        [SourceDeclaration("source", SourceRole.APPEND_JSONL, root, True)], request_id="replace"
    )
    original_observe = source_snapshot._observe
    calls = 0

    def observe_after_replacement(binding: source_snapshot.SourceCutBinding) -> tuple[source_snapshot.CutItem, ...]:
        nonlocal calls
        calls += 1
        if calls == 2:
            path.unlink()
            path.write_text("same", encoding="utf-8")
        return original_observe(binding)

    monkeypatch.setattr(source_snapshot, "_observe", observe_after_replacement)
    result = execute_source_cut(preflight, tmp_path / "cut")
    assert result.carry_forward_manifest.item_count == 1
    assert result.counts.conserved


def test_archive_members_and_sqlite_use_declared_strategies(tmp_path: Path) -> None:
    import zipfile

    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("one.json", "one")
    database = tmp_path / "state.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE state (value TEXT)")
        conn.execute("INSERT INTO state VALUES ('stable')")
        conn.commit()
    declarations = (
        SourceDeclaration("archive", SourceRole.ARCHIVE_MEMBER, archive),
        SourceDeclaration("state", SourceRole.MUTABLE_SQLITE, database, True),
    )
    preflight = preflight_source_cut(declarations)
    assert [binding.policy.mode for binding in preflight.bindings] == [
        SnapshotMode.ARCHIVE_MEMBER,
        SnapshotMode.SQLITE_BACKUP,
    ]
    result = execute_source_cut(preflight, tmp_path / "cut")
    assert result.counts.conserved
    assert {item.source_id for item in result.candidate_manifest.items} == {"archive", "state"}
    assert {item.source_id for item in result.carry_forward_manifest.items} == set()


def test_cut_verification_rejects_an_inventory_item_owned_by_neither_side(tmp_path: Path) -> None:
    """Mutation: omitting a measured item from both manifests must fail conservation."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "one.jsonl").write_text("one\n", encoding="utf-8")
    result = execute_source_cut(
        preflight_source_cut([SourceDeclaration("source", SourceRole.APPEND_JSONL, root, True)]), tmp_path / "cut"
    )
    missing = replace(result, candidate_manifest=source_snapshot._manifest("candidate", ()))

    with pytest.raises(SourceSnapshotError, match="conservation"):
        missing.verify()


def test_cut_verification_rejects_an_inventory_item_owned_by_both_sides(tmp_path: Path) -> None:
    """Mutation: assigning one measured item to both cohorts must fail conservation."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "one.jsonl").write_text("one\n", encoding="utf-8")
    result = execute_source_cut(
        preflight_source_cut([SourceDeclaration("source", SourceRole.APPEND_JSONL, root, True)]), tmp_path / "cut"
    )
    duplicate = replace(
        result,
        carry_forward_manifest=source_snapshot._manifest("carry-forward", result.candidate_manifest.items),
    )

    with pytest.raises(SourceSnapshotError, match="conservation"):
        duplicate.verify()


def test_published_cut_refuses_a_different_preflight(tmp_path: Path) -> None:
    """Mutation: reusing a cut for a differently bound request must be rejected."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "one.jsonl").write_text("one\n", encoding="utf-8")
    destination = tmp_path / "cut"
    execute_source_cut(
        preflight_source_cut([SourceDeclaration("source", SourceRole.APPEND_JSONL, root, True)], request_id="first"),
        destination,
    )

    with pytest.raises(SourceSnapshotError, match="binding"):
        execute_source_cut(
            preflight_source_cut(
                [SourceDeclaration("other", SourceRole.APPEND_JSONL, root, True)], request_id="second"
            ),
            destination,
        )


def test_sqlite_cut_refuses_a_commit_during_online_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: a post-backup revision change must not occupy either cohort."""
    database = tmp_path / "state.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE state (value TEXT)")
        conn.commit()

    original_backup = snapshot_sqlite_database

    def backup_then_commit(source: Path, destination: Path) -> None:
        original_backup(source, destination)
        with sqlite3.connect(source) as conn:
            conn.execute("INSERT INTO state VALUES ('after-cut')")
            conn.commit()

    monkeypatch.setattr("polylogue.sources.source_snapshot.snapshot_sqlite_database", backup_then_commit)
    with pytest.raises(SourceMutationError, match="SQLite source changed"):
        execute_source_cut(
            preflight_source_cut([SourceDeclaration("state", SourceRole.MUTABLE_SQLITE, database, True)]),
            tmp_path / "cut",
        )


def test_sqlite_cut_refuses_a_backup_with_a_different_logical_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: publishing a backup from another logical state must fail."""
    database = tmp_path / "state.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE state (value TEXT)")
        conn.execute("INSERT INTO state VALUES ('source')")
        conn.commit()

    original_backup = snapshot_sqlite_database

    def backup_with_different_content(source: Path, destination: Path) -> None:
        original_backup(source, destination)
        with sqlite3.connect(destination) as conn:
            conn.execute("INSERT INTO state VALUES ('not-source')")
            conn.commit()

    monkeypatch.setattr("polylogue.sources.source_snapshot.snapshot_sqlite_database", backup_with_different_content)
    with pytest.raises(SourceMutationError, match="backup does not match source logical revision"):
        execute_source_cut(
            preflight_source_cut([SourceDeclaration("state", SourceRole.MUTABLE_SQLITE, database, True)]),
            tmp_path / "cut",
        )


def test_sqlite_cut_manifest_hashes_the_published_backup_bytes(tmp_path: Path) -> None:
    """Mutation: a logical revision cannot replace the retained-byte hash."""
    database = tmp_path / "state.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE state (value TEXT)")
        conn.execute("INSERT INTO state VALUES ('source')")
        conn.commit()

    result = execute_source_cut(
        preflight_source_cut([SourceDeclaration("state", SourceRole.MUTABLE_SQLITE, database, True)]),
        tmp_path / "cut",
    )

    candidate = result.candidate_manifest.items[0]
    retained = reacquire_candidate(result)[0]
    assert candidate.content_sha256 == hashlib.sha256(retained.path.read_bytes()).hexdigest()


def test_sqlite_continuity_uses_logical_rows_not_page_layout(tmp_path: Path) -> None:
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    for path, rows in ((first, ((2, "b"), (1, "a"))), (second, ((1, "a"), (2, "b")))):
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE values_table (id INTEGER PRIMARY KEY, value TEXT)")
            conn.executemany("INSERT INTO values_table VALUES (?, ?)", rows)
    assert sqlite_logical_revision(first) == sqlite_logical_revision(second)


def test_sqlite_logical_revision_ignores_declared_nocase_collation_in_row_order(tmp_path: Path) -> None:
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    for path, rows in ((first, ((1, "a"), (2, "A"))), (second, ((2, "A"), (1, "a")))):
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE values_table (id INTEGER PRIMARY KEY, value TEXT COLLATE NOCASE)")
            conn.executemany("INSERT INTO values_table VALUES (?, ?)", rows)
    assert sqlite_logical_revision(first) == sqlite_logical_revision(second)


def test_sqlite_logical_revision_orders_rows_by_storage_class(tmp_path: Path) -> None:
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    for path, rows in ((first, ((1, 1), (2, 1.0))), (second, ((2, 1.0), (1, 1)))):
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE values_table (id INTEGER PRIMARY KEY, value)")
            conn.executemany("INSERT INTO values_table VALUES (?, ?)", rows)
    assert sqlite_logical_revision(first) == sqlite_logical_revision(second)


def test_sqlite_logical_revision_includes_implicit_rowids(tmp_path: Path) -> None:
    database = tmp_path / "rowid.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE values_table (value TEXT)")
        conn.executemany("INSERT INTO values_table VALUES (?)", (("first",), ("second",)))
    before = sqlite_logical_revision(database)
    with sqlite3.connect(database) as conn:
        conn.execute("DELETE FROM values_table WHERE rowid = 1")
        conn.execute("INSERT INTO values_table VALUES ('first')")
    assert sqlite_logical_revision(database) != before


def test_sqlite_logical_revision_excludes_without_rowid_tables(tmp_path: Path) -> None:
    database = tmp_path / "without-rowid.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE values_table (id TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID")
        conn.execute("INSERT INTO values_table VALUES ('one', 'value')")
    assert sqlite_logical_revision(database)


def test_sqlite_logical_revision_skips_unavailable_virtual_table_modules(tmp_path: Path) -> None:
    database = tmp_path / "virtual.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE values_table (value TEXT)")
        conn.execute("INSERT INTO values_table VALUES ('retained')")
        conn.execute("PRAGMA writable_schema = ON")
        conn.execute(
            "INSERT INTO sqlite_master(type, name, tbl_name, rootpage, sql) "
            "VALUES ('table', 'extension_table', 'extension_table', 0, "
            "'CREATE VIRTUAL TABLE extension_table USING unavailable_module')"
        )
        conn.execute("PRAGMA writable_schema = OFF")
    assert sqlite_logical_revision(database)


def test_sqlite_logical_revision_preserves_invalid_utf8_text(tmp_path: Path) -> None:
    database = tmp_path / "invalid-text.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE values_table (value TEXT)")
        conn.execute("INSERT INTO values_table VALUES (CAST(X'80' AS TEXT))")
    assert sqlite_logical_revision(database)


@pytest.mark.parametrize("role", [SourceRole.SPOOL, SourceRole.QUEUE])
def test_spool_handoff_leaves_a_new_empty_active_generation(tmp_path: Path, role: SourceRole) -> None:
    """Mutation: treating a spool as a copied directory would leave writers on the frozen generation."""
    spool = tmp_path / "spool"
    spool.mkdir()
    (spool / "event.json").write_text("event", encoding="utf-8")

    result = execute_source_cut(preflight_source_cut([SourceDeclaration("spool", role, spool, True)]), tmp_path / "cut")

    assert spool.is_dir()
    assert list(spool.iterdir()) == []
    assert (result.candidate_root / "spool" / "event.json").read_text(encoding="utf-8") == "event"


def test_cut_reclaims_incomplete_publication_and_orphaned_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: a crash before the final marker must be retried as absent output."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "one.jsonl").write_text("one\n", encoding="utf-8")
    preflight = preflight_source_cut(
        [SourceDeclaration("source", SourceRole.APPEND_JSONL, root, True)], request_id="crash-boundary"
    )
    destination = tmp_path / "cut"
    orphan = tmp_path / ".crash-boundary.orphan"
    orphan.mkdir()
    original_write = source_snapshot._write_durable

    def crash_before_marker(path: Path, payload: str) -> None:
        if path.name == ".source-cut-complete":
            raise OSError("simulated crash")
        original_write(path, payload)

    monkeypatch.setattr(source_snapshot, "_write_durable", crash_before_marker)
    with pytest.raises(OSError, match="simulated crash"):
        execute_source_cut(preflight, destination)
    assert not (destination / ".source-cut-complete").exists()
    with pytest.raises(FileNotFoundError):
        source_snapshot.load_source_cut(destination)

    monkeypatch.setattr(source_snapshot, "_write_durable", original_write)
    recovered = execute_source_cut(preflight, destination)
    assert recovered.counts.conserved
    assert not orphan.exists()

    valid_orphan = tmp_path / ".crash-boundary.valid-orphan"
    valid_orphan.mkdir()
    assert execute_source_cut(preflight, destination).cut_identity == recovered.cut_identity
    assert not valid_orphan.exists()

    (destination / ".source-cut-complete").write_text("partial", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        source_snapshot.load_source_cut(destination)
    assert execute_source_cut(preflight, destination).cut_identity == recovered.cut_identity


def test_spool_handoff_recovers_retired_generation_after_marker_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: reclaiming a markerless spool cut before restoration loses its only pre-cut events."""
    spool = tmp_path / "spool"
    spool.mkdir()
    (spool / "before.json").write_text("before", encoding="utf-8")
    preflight = preflight_source_cut(
        [SourceDeclaration("spool", SourceRole.SPOOL, spool, True)], request_id="spool-marker-crash"
    )
    destination = tmp_path / "cut"
    original_write = source_snapshot._write_durable

    def crash_before_marker(path: Path, payload: str) -> None:
        if path.name == ".source-cut-complete":
            raise OSError("simulated marker crash")
        original_write(path, payload)

    monkeypatch.setattr(source_snapshot, "_write_durable", crash_before_marker)
    with pytest.raises(OSError, match="simulated marker crash"):
        execute_source_cut(preflight, destination)

    retired = tmp_path / ".spool.spool.cut"
    assert retired.is_dir()
    assert (retired / "before.json").read_text(encoding="utf-8") == "before"
    (spool / "after.json").write_text("after", encoding="utf-8")

    monkeypatch.setattr(source_snapshot, "_write_durable", original_write)
    recovered = execute_source_cut(preflight, destination)

    assert recovered.counts.conserved
    assert {item.coordinate for item in recovered.candidate_manifest.items} == {
        ".spool.spool.arrivals/after.json",
        "before.json",
    }
    assert not retired.exists()


def test_source_id_cannot_escape_candidate_staging(tmp_path: Path) -> None:
    """Mutation: an absolute source ID must be rejected before any staging path is joined."""
    root = tmp_path / "source"
    root.mkdir()
    escaped = tmp_path / "escaped"
    with pytest.raises(SourceSnapshotError, match="source_id"):
        preflight_source_cut([SourceDeclaration(str(escaped), SourceRole.DIRECTORY, root, True)])
    assert not escaped.exists()
