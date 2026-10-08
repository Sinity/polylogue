"""Lossless source-cut laws through the production snapshot interface."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from polylogue.maintenance.source_manifest_continuity import (
    FrontierState,
    SourceDeclaration,
    SourceRole,
    build_source_frontier,
)
from polylogue.sources import source_snapshot, sqlite_export
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
from polylogue.sources.sqlite_export import logical_export_bytes, looks_like_logical_export_path, read_export_header
from polylogue.sources.sqlite_snapshot import sqlite_logical_revision, sqlite_member_revision


@pytest.mark.parametrize("kind", ["file", "zip", "sqlite"])
def test_source_observation_cancellation_stops_actual_byte_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import threading
    import zipfile

    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.core.compute_cancel import compute_cancel

    cancelled = threading.Event()
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"synthetic\n" * 150000)
    role = SourceRole.IMMUTABLE_EXPORT
    if kind == "zip":
        archive = tmp_path / "source.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.write(source, "session.jsonl")
        source = archive
        role = SourceRole.ARCHIVE_MEMBER
        original_member_read = zipfile.ZipExtFile.read

        def read_member(self: zipfile.ZipExtFile, *args: Any, **kwargs: Any) -> bytes:
            result = original_member_read(self, *args, **kwargs)
            if result:
                cancelled.set()
            return result

        monkeypatch.setattr(zipfile.ZipExtFile, "read", read_member)
    elif kind == "sqlite":
        source = tmp_path / "source.sqlite"
        with sqlite3.connect(source) as connection:
            connection.execute("CREATE TABLE evidence (value TEXT)")
            connection.execute("INSERT INTO evidence VALUES ('synthetic')")
        role = SourceRole.MUTABLE_SQLITE
        original_write = sqlite_export._HashingSink.write

        def write_chunk(self: Any, chunk: bytes) -> int:
            result = original_write(self, chunk)
            if chunk:
                cancelled.set()
            return result

        monkeypatch.setattr(sqlite_export._HashingSink, "write", write_chunk)
    else:
        original_file_read = os.read

        def read_file(descriptor: int, size: int) -> bytes:
            result = original_file_read(descriptor, size)
            if result:
                cancelled.set()
            return result

        monkeypatch.setattr(os, "read", read_file)
    declaration = SourceDeclaration("source", role, source, mutable=kind == "sqlite")
    token = compute_cancel.set(cancelled)
    try:
        with pytest.raises(DaemonOperationCancelled):
            source_snapshot.observe_source_members(declaration)
        assert cancelled.is_set()
    finally:
        compute_cancel.reset(token)
    assert len(source_snapshot.observe_source_members(declaration)) == 1


def test_source_cut_cancellation_during_copy_leaves_no_published_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.core.compute_cancel import compute_cancel

    source = tmp_path / "source.jsonl"
    source.write_bytes(b"synthetic\n" * 250000)
    preflight = preflight_source_cut(
        [SourceDeclaration("source", SourceRole.APPEND_JSONL, source, True)],
        policies={"source": source_snapshot.SourceCutPolicy(SnapshotMode.COMPLETE_COPY, prefer_reflink=False)},
    )
    destination = tmp_path / "candidate"
    cancelled = threading.Event()
    original_copy = source_snapshot._copy_file
    original_read = os.read
    in_copy = False
    armed = True

    def copy_file(*args: Any, **kwargs: Any) -> None:
        nonlocal in_copy
        in_copy = True
        try:
            original_copy(*args, **kwargs)
        finally:
            in_copy = False

    def read_chunk(descriptor: int, size: int) -> bytes:
        nonlocal armed
        result = original_read(descriptor, size)
        if in_copy and armed and result:
            armed = False
            cancelled.set()
        return result

    monkeypatch.setattr(source_snapshot, "_copy_file", copy_file)
    monkeypatch.setattr(os, "read", read_chunk)
    token = compute_cancel.set(cancelled)
    try:
        with pytest.raises(DaemonOperationCancelled):
            source_snapshot.execute_source_cut(preflight, destination)
        assert not destination.exists()
        assert not list(tmp_path.glob(".source-cut.*"))
    finally:
        compute_cancel.reset(token)
    result = source_snapshot.execute_source_cut(preflight, destination)
    assert len(result.candidate_manifest.items) == 1


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


def test_member_hash_uses_one_descriptor_and_captured_append_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A growth after descriptor stat cannot pair the old size with a longer digest.

    Anti-vacuity: a path-based read-to-EOF hashes the appended bytes while
    retaining the pre-append size from the initial inventory.
    """
    root = tmp_path / "append-root"
    root.mkdir()
    member = root / "events.jsonl"
    original = b"first\n"
    member.write_bytes(original)
    real_fstat = os.fstat
    member_inode = member.stat().st_ino
    captured = False

    def append_after_capture(fd: int) -> Any:
        nonlocal captured
        info = real_fstat(fd)
        if not captured and info.st_ino == member_inode:
            captured = True
            with member.open("ab") as output:
                output.write(b"later\n")
        return info

    monkeypatch.setattr(os, "fstat", append_after_capture)
    observed = source_snapshot.observe_source_members(SourceDeclaration("append", SourceRole.APPEND_JSONL, root, True))

    assert len(observed) == 1
    assert observed[0].size_bytes == len(original)
    assert observed[0].content_sha256 == hashlib.sha256(original).hexdigest()


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
        SnapshotMode.SQLITE_LOGICAL_EXPORT,
    ]
    result = execute_source_cut(preflight, tmp_path / "cut")
    assert result.counts.conserved
    assert {item.source_id for item in result.candidate_manifest.items} == {"archive", "state"}
    assert {item.source_id for item in result.carry_forward_manifest.items} == set()
    sqlite_candidate = reacquire_candidate(result, source_id="state")[0]
    assert sqlite_candidate.path.suffix == ".jsonl"
    assert looks_like_logical_export_path(sqlite_candidate.path)


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


def test_sqlite_cut_refuses_a_commit_during_logical_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: a post-export revision change must not occupy either cohort."""
    database = tmp_path / "state.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE state (value TEXT)")
        conn.commit()

    original_export = sqlite_export._write_logical_export_bound

    def export_then_commit(source: Path, handle: sqlite_export.BinaryWriteSink, **kwargs: Any) -> None:
        original_export(source, handle, **kwargs)
        with sqlite3.connect(source) as conn:
            conn.execute("INSERT INTO state VALUES ('after-cut')")
            conn.commit()

    monkeypatch.setattr("polylogue.sources.source_snapshot._write_logical_export_bound", export_then_commit)
    with pytest.raises(SourceMutationError, match="SQLite source changed during logical export"):
        execute_source_cut(
            preflight_source_cut([SourceDeclaration("state", SourceRole.MUTABLE_SQLITE, database, True)]),
            tmp_path / "cut",
        )


def test_sqlite_cut_refuses_a_logical_export_with_a_different_logical_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: publishing a backup from another logical state must fail."""
    database = tmp_path / "state.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE state (value TEXT)")
        conn.execute("INSERT INTO state VALUES ('source')")
        conn.commit()

    different = tmp_path / "different.sqlite"
    with sqlite3.connect(different) as conn:
        conn.execute("CREATE TABLE state (value TEXT)")
        conn.execute("INSERT INTO state VALUES ('not-source')")
        conn.commit()

    def export_different_content(_source: Path, handle: sqlite_export.BinaryWriteSink, **kwargs: Any) -> None:
        kwargs.pop("expected_identity", None)
        kwargs.pop("parent_anchor", None)
        kwargs.pop("source_binding", None)
        logical_export = logical_export_bytes(different, **kwargs)
        handle.write(logical_export)

    monkeypatch.setattr("polylogue.sources.source_snapshot._write_logical_export_bound", export_different_content)
    with pytest.raises(SourceMutationError, match="logical export does not match source logical revision"):
        execute_source_cut(
            preflight_source_cut([SourceDeclaration("state", SourceRole.MUTABLE_SQLITE, database, True)]),
            tmp_path / "cut",
        )


def test_sqlite_cut_manifest_hashes_the_published_logical_export(tmp_path: Path) -> None:
    """Mutation: restoring a page image must not satisfy the published candidate contract."""
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
    assert candidate.content_sha256 == sqlite_member_revision(database)
    assert looks_like_logical_export_path(retained.path)
    assert read_export_header(retained.path).tables == ("state",)


def test_sqlite_cut_rejects_legacy_page_backup_mode() -> None:
    """Mutation: an old page-image policy must not become a compatibility route."""
    with pytest.raises(ValueError, match="sqlite-online-backup"):
        SnapshotMode("sqlite-online-backup")


def test_sqlite_cut_bounds_the_streamed_logical_export(tmp_path: Path) -> None:
    """Mutation: a hex-expanded blob export must not exceed the declared candidate capacity."""
    database = tmp_path / "state.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE state (value BLOB)")
        conn.execute("INSERT INTO state VALUES (?)", (b"x" * (128 * 1024),))
        conn.commit()
    capacity = database.stat().st_size
    assert len(logical_export_bytes(database)) > capacity

    with pytest.raises(SourceSnapshotError, match="logical SQLite export"):
        execute_source_cut(
            preflight_source_cut(
                [SourceDeclaration("state", SourceRole.MUTABLE_SQLITE, database, True)],
                policies={"state": SourceCutPolicy(SnapshotMode.SQLITE_LOGICAL_EXPORT, capacity_bytes=capacity)},
            ),
            tmp_path / "cut",
        )
    assert not (tmp_path / "cut").exists()


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


def test_spool_handoff_preserves_declared_coordinate_exclusions(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    (spool / "event.json").write_text("event", encoding="utf-8")
    (spool / "excluded.json").write_text("excluded", encoding="utf-8")

    declaration = SourceDeclaration(
        "spool",
        SourceRole.SPOOL,
        spool,
        mutable=True,
        exclude_coordinates=("excluded.json",),
    )
    result = execute_source_cut(preflight_source_cut([declaration]), tmp_path / "cut")

    assert {item.coordinate for item in result.candidate_manifest.items} == {"event.json"}
    assert (result.candidate_root / "spool" / "event.json").read_text(encoding="utf-8") == "event"
    assert not (result.candidate_root / "spool" / "excluded.json").exists()


@pytest.mark.parametrize("role", [SourceRole.SPOOL, SourceRole.QUEUE])
@pytest.mark.parametrize("replace_active", [False, True])
def test_handoff_observes_only_the_producer_bound_active_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: SourceRole, replace_active: bool
) -> None:
    """A fresh-stat rebind would accept an unrelated active root after copying."""
    spool = tmp_path / "spool"
    spool.mkdir()
    (spool / "event.json").write_text("event", encoding="utf-8")
    original_copy = source_snapshot._copy_candidates

    def copy_with_arrival(
        binding: source_snapshot.SourceCutBinding, baseline: tuple[source_snapshot.CutItem, ...], destination: Path
    ) -> tuple[source_snapshot.CutItem, ...]:
        copied = original_copy(binding, baseline, destination)
        if replace_active:
            spool.rename(tmp_path / "displaced-active")
            spool.mkdir()
        (spool / "arrival.json").write_text("arrival", encoding="utf-8")
        return copied

    monkeypatch.setattr(source_snapshot, "_copy_candidates", copy_with_arrival)
    preflight = preflight_source_cut([SourceDeclaration("spool", role, spool, True)])
    if replace_active:
        with pytest.raises(source_snapshot.SourceMutationError):
            execute_source_cut(preflight, tmp_path / "cut")
    else:
        result = execute_source_cut(preflight, tmp_path / "cut")
        assert {item.coordinate for item in result.carry_forward_manifest.items} == {"arrival.json"}
        assert (result.candidate_root / "spool" / "event.json").read_text(encoding="utf-8") == "event"


def test_cut_reclaims_only_staging_it_owns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash before the final marker is retried as absent output.

    Anti-vacuity: reclaim by request-id prefix alone and the unowned
    ``.crash-boundary.backup`` directory is deleted; skip reclamation and the
    owned crashed staging directory survives.
    """
    root = tmp_path / "source"
    root.mkdir()
    (root / "one.jsonl").write_text("one\n", encoding="utf-8")
    preflight = preflight_source_cut(
        [SourceDeclaration("source", SourceRole.APPEND_JSONL, root, True)], request_id="crash-boundary"
    )
    destination = tmp_path / "cut"
    unowned = tmp_path / ".crash-boundary.backup"
    unowned.mkdir()
    (unowned / "unowned.txt").write_text("keep", encoding="utf-8")
    owned = tmp_path / ".crash-boundary.crashed"
    owned.mkdir()
    (owned / source_snapshot._STAGING_OWNER_MARKER).write_text("crash-boundary\n", encoding="utf-8")
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
    assert not owned.exists()

    monkeypatch.setattr(source_snapshot, "_write_durable", original_write)
    recovered = execute_source_cut(preflight, destination)
    assert recovered.counts.conserved
    assert (unowned / "unowned.txt").read_text(encoding="utf-8") == "keep"
    assert not (destination / source_snapshot._STAGING_OWNER_MARKER).exists()
    assert execute_source_cut(preflight, destination).cut_identity == recovered.cut_identity
    assert unowned.is_dir()

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


def test_spool_handoff_recovers_when_destination_was_never_created(tmp_path: Path) -> None:
    """Anti-vacuity: a crash after rename but before staging must retain pre-cut events."""
    spool = tmp_path / "spool"
    spool.mkdir()
    (spool / "before.json").write_text("before", encoding="utf-8")
    preflight = preflight_source_cut(
        [SourceDeclaration("spool", SourceRole.SPOOL, spool, True)], request_id="missing-destination"
    )
    retired = tmp_path / ".spool.spool.cut"
    spool.rename(retired)

    recovered = execute_source_cut(preflight, tmp_path / "cut")

    assert recovered.counts.conserved
    assert (recovered.candidate_root / "spool" / "before.json").read_text(encoding="utf-8") == "before"
    assert spool.is_dir()


def test_source_id_cannot_escape_candidate_staging(tmp_path: Path) -> None:
    """Mutation: an absolute source ID must be rejected before any staging path is joined."""
    root = tmp_path / "source"
    root.mkdir()
    escaped = tmp_path / "escaped"
    with pytest.raises(SourceSnapshotError, match="source_id"):
        preflight_source_cut([SourceDeclaration(str(escaped), SourceRole.DIRECTORY, root, True)])
    assert not escaped.exists()


@pytest.mark.parametrize("alias", [False, True])
def test_cut_refuses_staging_inside_source_before_mutating_it(tmp_path: Path, alias: bool) -> None:
    """Without the overlap guard, the census includes its own staging output."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "one.jsonl").write_text("one\n", encoding="utf-8")
    parent = root
    if alias:
        parent = tmp_path / "source-alias"
        parent.symlink_to(root, target_is_directory=True)
    preflight = preflight_source_cut([SourceDeclaration("source", SourceRole.APPEND_JSONL, root, True)])

    with pytest.raises(SourceSnapshotError):
        execute_source_cut(preflight, parent / "nested" / "cut")

    assert sorted(path.name for path in root.iterdir()) == ["one.jsonl"]


def test_sqlite_cut_keeps_dotted_source_ids_distinct(tmp_path: Path) -> None:
    """Replacing a suffix makes state.first and state.second overwrite state.jsonl."""
    declarations = []
    for name in ("first", "second"):
        database = tmp_path / f"{name}.sqlite"
        with sqlite3.connect(database) as conn:
            conn.execute("CREATE TABLE state (value TEXT)")
            conn.execute("INSERT INTO state VALUES (?)", (name,))
        declarations.append(SourceDeclaration(f"state.{name}", SourceRole.MUTABLE_SQLITE, database, True))

    result = execute_source_cut(preflight_source_cut(declarations), tmp_path / "cut")
    inputs = reacquire_candidate(result)

    assert {item.path.name for item in inputs} == {"state.first.jsonl", "state.second.jsonl"}
    assert len({item.path.read_bytes() for item in inputs}) == 2


def test_archive_cut_reacquires_member_bytes_and_detects_member_mutation(tmp_path: Path) -> None:
    """Comparing the ZIP file digest to a member digest rejects an intact cut."""
    import zipfile

    source = tmp_path / "export.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("one.json", "one")
        archive.writestr("nested/two.json", "two")
    result = execute_source_cut(
        preflight_source_cut([SourceDeclaration("export", SourceRole.ARCHIVE_MEMBER, source)]), tmp_path / "cut"
    )

    inputs = reacquire_candidate(result)
    assert {item.coordinate for item in inputs} == {"export.zip!one.json", "export.zip!nested/two.json"}
    assert {item.size_bytes for item in inputs} == {3}
    assert len({item.path for item in inputs}) == 1
    assert reacquire_candidate(result, coordinates=[inputs[0].coordinate]) == (inputs[0],)

    with zipfile.ZipFile(inputs[0].path, "w") as archive:
        archive.writestr("one.json", "changed")
        archive.writestr("nested/two.json", "two")
    with pytest.raises(SourceMutationError):
        reacquire_candidate(result)


@pytest.mark.parametrize("container_name", ["export.zip", "export!copy.zip"])
def test_archive_cut_preserves_distinct_duplicate_named_members(tmp_path: Path, container_name: str) -> None:
    """Name-based reopening selects the last duplicate instead of the captured member."""
    import zipfile

    source = tmp_path / container_name
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("nested/item!part.json", "first")
        with pytest.warns(UserWarning):
            archive.writestr("nested/item!part.json", "second")
    result = execute_source_cut(
        preflight_source_cut([SourceDeclaration("export", SourceRole.ARCHIVE_MEMBER, source)]), tmp_path / "cut"
    )
    inputs = reacquire_candidate(result)
    assert result.counts.conserved
    assert result.candidate_manifest.item_count == 2
    assert {item.content_sha256 for item in inputs} == {
        hashlib.sha256(b"first").hexdigest(),
        hashlib.sha256(b"second").hexdigest(),
    }
    assert len(inputs) == 2
    assert reacquire_candidate(result, coordinates=[inputs[0].coordinate]) == inputs


@pytest.mark.skipif(os.geteuid() == 0, reason="Permission test requires an unprivileged reader")
def test_frontier_refuses_whole_root_when_hidden_directory_is_unreadable(tmp_path: Path) -> None:
    """Mutation: rglob silently skips denied directories and publishes partial PRESENT."""
    root = tmp_path / "declared"
    hidden = root / "hidden"
    hidden.mkdir(parents=True)
    (root / "public.json").write_bytes(b"{}")
    (hidden / "session.jsonl").write_bytes(b"private\n")
    hidden.chmod(0)
    try:
        frontier = build_source_frontier([SourceDeclaration("declared", SourceRole.DIRECTORY, root, True)])
    finally:
        hidden.chmod(0o700)
    assert frontier.root_states["declared"] is FrontierState.UNAVAILABLE
    assert frontier.members == ()
    assert not frontier.complete
    assert frontier.blockers
    frontier.verify_integrity()


@pytest.mark.skipif(os.geteuid() == 0, reason="requires an unprivileged directory reader")
def test_candidate_sync_refuses_an_unreadable_nested_directory(tmp_path: Path) -> None:
    """A walk that silently omits a directory cannot prove the candidate tree synced."""
    root = tmp_path / "candidate"
    hidden = root / "hidden"
    hidden.mkdir(parents=True)
    (hidden / "member.jsonl").write_bytes(b"{}\n")
    hidden.chmod(0)
    try:
        with pytest.raises(SourceSnapshotError):
            source_snapshot._fsync_tree(root)
    finally:
        hidden.chmod(0o700)


@pytest.mark.parametrize("replacement", ["symlink", "regular", "parent-symlink"])
def test_frontier_refuses_member_substitution_between_enumeration_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    """Mutation: path-open follows an external symlink or assigns a new inode to the old coordinate."""
    root = tmp_path / "declared"
    directory = root / "nested"
    directory.mkdir(parents=True)
    member = directory / "session.jsonl"
    member.write_bytes(b"declared\n")
    external = tmp_path / "external"
    external.mkdir()
    target = external / member.name
    target.write_bytes(b"unrelated\n")
    original = source_snapshot._snapshot_regular_file

    def substitute(path: Path, expected: os.stat_result, *, anchor: int, coordinate: str) -> tuple[str, int, str]:
        if replacement == "parent-symlink":
            member.unlink()
            directory.rmdir()
            directory.symlink_to(external, target_is_directory=True)
        else:
            member.rename(directory / "previous")
            if replacement == "symlink":
                member.symlink_to(target)
            else:
                member.write_bytes(b"replacement\n")
        return original(path, expected, anchor=anchor, coordinate=coordinate)

    monkeypatch.setattr(source_snapshot, "_snapshot_regular_file", substitute)
    frontier = build_source_frontier([SourceDeclaration("declared", SourceRole.APPEND_JSONL, root, True)])
    assert frontier.root_states["declared"] is FrontierState.UNAVAILABLE
    assert frontier.members == ()
    assert not frontier.complete
    frontier.verify_integrity()


@pytest.mark.parametrize("replacement", ["symlink", "regular"])
def test_cut_refuses_equal_byte_substitution_before_copy_even_if_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    """Mutation: reopening the source path copies external equal bytes with the original inode's identity."""
    root = tmp_path / "declared"
    root.mkdir()
    member = root / "session.jsonl"
    member.write_bytes(b"equal bytes\n")
    external = tmp_path / "unrelated.jsonl"
    external.write_bytes(member.read_bytes())
    held = tmp_path / "original.jsonl"
    original = source_snapshot._copy_file

    def substituted_copy(
        source: Path,
        destination: Path,
        policy: SourceCutPolicy,
        *,
        expected: tuple[int, int],
        captured_size: int | None,
        anchor: int,
        coordinate: str,
    ) -> None:
        member.rename(held)
        if replacement == "symlink":
            member.symlink_to(external)
        else:
            member.write_bytes(external.read_bytes())
        try:
            return original(
                source,
                destination,
                policy,
                expected=expected,
                captured_size=captured_size,
                anchor=anchor,
                coordinate=coordinate,
            )
        finally:
            member.unlink()
            held.rename(member)

    monkeypatch.setattr(source_snapshot, "_copy_file", substituted_copy)
    destination = tmp_path / "cut"
    with pytest.raises(SourceSnapshotError):
        execute_source_cut(
            preflight_source_cut([SourceDeclaration("declared", SourceRole.APPEND_JSONL, root, True)]), destination
        )
    assert not destination.exists()
    assert member.read_bytes() == b"equal bytes\n"


def test_sqlite_frontier_refuses_persistent_symlink_substitution_during_logical_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: path-resolved SQLite revision publishes an external database under the declared coordinate."""
    database = tmp_path / "declared.sqlite"
    external = tmp_path / "external.sqlite"
    for path, value in ((database, "declared"), (external, "unrelated")):
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE state (value TEXT)")
            conn.execute("INSERT INTO state VALUES (?)", (value,))
    original = sqlite_export._logical_export_digest_bound

    def substitute(path: Path, **kwargs: Any) -> str:
        database.rename(tmp_path / "original.sqlite")
        database.symlink_to(external)
        return original(path, **kwargs)

    monkeypatch.setattr(source_snapshot, "_logical_export_digest_bound", substitute)
    frontier = build_source_frontier([SourceDeclaration("database", SourceRole.MUTABLE_SQLITE, database, True)])
    assert frontier.root_states["database"] is FrontierState.UNAVAILABLE
    assert frontier.members == ()
    assert not frontier.complete


@pytest.mark.parametrize("file_root", [False, True])
def test_stable_declared_parent_alias_preserves_observation_and_cut(tmp_path: Path, file_root: bool) -> None:
    """Mutation: no-following every absolute ancestor rejects a stable Documents alias."""
    actual = tmp_path / "actual-documents"
    actual.mkdir()
    (actual / "sessions").mkdir()
    (actual / "sessions" / "session.jsonl").write_bytes(b"declared\n")
    alias = tmp_path / "Documents"
    alias.symlink_to(actual, target_is_directory=True)
    root = alias / "sessions" / "session.jsonl" if file_root else alias / "sessions"
    declaration = SourceDeclaration("declared", SourceRole.APPEND_JSONL, root, True)
    frontier = build_source_frontier([declaration])
    assert frontier.complete
    assert frontier.root_states["declared"] is FrontierState.PRESENT
    assert len(frontier.members) == 1
    assert frontier.members[0].content_sha256 == hashlib.sha256(b"declared\n").hexdigest()
    cut = execute_source_cut(preflight_source_cut([declaration]), tmp_path / "cut")
    assert cut.counts.conserved
    assert reacquire_candidate(cut)[0].path.read_bytes() == b"declared\n"


def test_substituted_declared_parent_alias_cannot_publish_external_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: a root anchored for reads without rechecking its declaration publishes an old root under a new alias."""
    actual = tmp_path / "actual-documents"
    external = tmp_path / "external-documents"
    for directory in (actual, external):
        (directory / "sessions").mkdir(parents=True)
        (directory / "sessions" / "session.jsonl").write_bytes(b"equal bytes\n")
    alias = tmp_path / "Documents"
    alias.symlink_to(actual, target_is_directory=True)
    root = alias / "sessions"
    original = source_snapshot._snapshot_regular_file

    def substitute(path: Path, expected: os.stat_result, *, anchor: int, coordinate: str) -> tuple[str, int, str]:
        alias.unlink()
        alias.symlink_to(external, target_is_directory=True)
        return original(path, expected, anchor=anchor, coordinate=coordinate)

    monkeypatch.setattr(source_snapshot, "_snapshot_regular_file", substitute)
    frontier = build_source_frontier([SourceDeclaration("declared", SourceRole.APPEND_JSONL, root, True)])
    assert frontier.root_states["declared"] is FrontierState.UNAVAILABLE
    assert frontier.members == ()


@pytest.mark.uses_real_clock("SQLite process locks are verified by an external writer")
@pytest.mark.parametrize("operation", ["frontier", "shape", "export"])
def test_sqlite_observation_preserves_another_connections_process_locks(tmp_path: Path, operation: str) -> None:
    """Mutation: closing an ordinary SQLite guard fd releases a concurrent reader's POSIX lock."""
    database = tmp_path / "declared.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE state (value TEXT)")
        conn.execute("INSERT INTO state VALUES ('declared')")
    reader = sqlite3.connect(database)
    try:
        reader.execute("BEGIN")
        assert reader.execute("SELECT value FROM state").fetchone() == ("declared",)
        if operation == "frontier":
            frontier = build_source_frontier([SourceDeclaration("database", SourceRole.MUTABLE_SQLITE, database, True)])
            assert frontier.complete
        elif operation == "shape":
            assert sqlite_export.logical_source_shape(database) == {"state": ("value",)}
        else:
            assert b"declared" in sqlite_export.logical_export_bytes(database)
        writer = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sqlite3\nimport sys\nconn = sqlite3.connect(sys.argv[1], timeout=0)\ntry:\n    conn.execute(\"UPDATE state SET value = 'foreign'\")\n    conn.commit()\nexcept sqlite3.OperationalError as exc:\n    if exc.sqlite_errorcode == sqlite3.SQLITE_BUSY:\n        sys.exit(42)\n    raise\n",
                str(database),
            ],
            capture_output=True,
            text=True,
        )
        assert writer.returncode == 42
        assert reader.execute("SELECT value FROM state").fetchone() == ("declared",)
    finally:
        reader.rollback()
        reader.close()


@pytest.mark.parametrize("target_name", ["a.jsonl", "z.jsonl"])
@pytest.mark.parametrize("parent_alias", [False, True])
def test_contained_file_alias_is_excluded_only_after_target_observation(
    tmp_path: Path, target_name: str, parent_alias: bool
) -> None:
    """Rejecting every internal alias makes an otherwise complete provider root unavailable."""
    physical = tmp_path / "physical" / "sessions"
    project = physical / "-project"
    project.mkdir(parents=True)
    target = project / target_name
    target.write_bytes(b"session\n")
    (project / "m.jsonl").symlink_to(target if parent_alias else target.name)
    root = tmp_path / "declared" / "sessions" if parent_alias else physical
    if parent_alias:
        root.parent.symlink_to(physical.parent, target_is_directory=True)
    from polylogue.sources.live.discovery import _source_path_steps
    from polylogue.sources.live.watcher import WatchSource
    from polylogue.sources.source_layout import source_layout_for

    source = WatchSource("claude-code", root, layout=source_layout_for("claude-code"))
    ordinary_paths = [path for path in _source_path_steps(source, (source,), after=None) if path is not None]
    assert ordinary_paths == [root / "-project" / target_name]
    declaration = SourceDeclaration("declared", SourceRole.DIRECTORY, root, True, layout_name="claude-code")
    frontier = build_source_frontier([declaration])
    assert frontier.complete
    assert [member.coordinate for member in frontier.members] == [f"-project/{target_name}"]
    cut = execute_source_cut(preflight_source_cut([declaration]), tmp_path / "cut")
    assert cut.counts.conserved
    assert [item.coordinate for item in cut.candidate_manifest.items] == [f"-project/{target_name}"]
    assert reacquire_candidate(cut)[0].path.read_bytes() == b"session\n"


@pytest.mark.parametrize("target_kind", ["external", "dangling", "directory", "chain", "unselected", "excluded"])
def test_selected_file_alias_requires_contained_independently_selected_regular_target(
    tmp_path: Path, target_kind: str
) -> None:
    """A blanket symlink exclusion would hide missing or undeclared source material."""
    root = tmp_path / "source"
    project = root / "-project"
    project.mkdir(parents=True)
    target = project / "target.jsonl"
    target.write_bytes(b"session\n")
    if target_kind == "external":
        target = tmp_path / "external.jsonl"
        target.write_bytes(b"external\n")
    elif target_kind == "dangling":
        target = project / "missing.jsonl"
    elif target_kind == "directory":
        target = project / "directory"
        target.mkdir()
    elif target_kind == "chain":
        chain = project / "chain.jsonl"
        chain.symlink_to(target.name)
        target = chain
    elif target_kind == "unselected":
        target = project / "other.txt"
        target.write_bytes(b"unselected\n")
    (project / "alias.jsonl").symlink_to(target)
    declaration = SourceDeclaration(
        "declared",
        SourceRole.DIRECTORY,
        root,
        True,
        layout_name="claude-code",
        exclude_coordinates=("-project/target.jsonl",) if target_kind == "excluded" else (),
    )
    frontier = build_source_frontier([declaration])
    assert frontier.root_states["declared"] is FrontierState.UNAVAILABLE
    assert frontier.members == ()
    assert not frontier.complete


@pytest.mark.parametrize("mutation", ["alias", "target"])
def test_contained_alias_mutation_after_target_read_refuses_whole_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Checking containment once would publish a changed alias or target as a complete cut."""
    root = tmp_path / "source"
    root.mkdir()
    target = root / "z.jsonl"
    target.write_bytes(b"session\n")
    alias = root / "a.jsonl"
    alias.symlink_to(target.name)
    original = source_snapshot._snapshot_regular_file

    def mutate(path: Path, expected: os.stat_result, *, anchor: int, coordinate: str) -> tuple[str, int, str]:
        result = original(path, expected, anchor=anchor, coordinate=coordinate)
        if mutation == "alias":
            alias.unlink()
            alias.symlink_to("missing.jsonl")
        else:
            target.unlink()
            target.write_bytes(b"replacement\n")
        return result

    monkeypatch.setattr(source_snapshot, "_snapshot_regular_file", mutate)
    frontier = build_source_frontier([SourceDeclaration("declared", SourceRole.APPEND_JSONL, root, True)])
    assert frontier.root_states["declared"] is FrontierState.UNAVAILABLE
    assert frontier.members == ()
    assert not frontier.complete
