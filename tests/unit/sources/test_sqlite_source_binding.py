"""Actual isolated SQLite descriptors decide source attribution and settlement."""

from __future__ import annotations

import errno
import os
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from typing import IO, Any, cast

import pytest

from polylogue.maintenance.source_manifest_continuity import (
    FrontierState,
    SourceDeclaration,
    SourceRole,
    build_source_frontier,
)
from polylogue.sources import sqlite_export, sqlite_snapshot
from polylogue.storage.blob_store import BlobStore

pytestmark = pytest.mark.uses_real_clock("source worker settlement uses actual OS processes")


def _database(path: Path, value: str) -> None:
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE state (value TEXT)")
        conn.execute("INSERT INTO state VALUES (?)", (value,))


def test_profile_parent_alias_capture_keeps_the_existing_namespace(tmp_path: Path) -> None:
    from polylogue.sources.parsers.hermes_identity import profile_key

    original = tmp_path / "original"
    original.mkdir()
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    _database(original / "bundle.data", "same")
    _database(alternate / "bundle.data", "same")
    (original / "state.db").symlink_to(original / "bundle.data")
    (alternate / "state.db").symlink_to(alternate / "bundle.data")
    alias = tmp_path / "profile"
    alias.symlink_to(original, target_is_directory=True)
    store = BlobStore(tmp_path / "blobs")
    first = sqlite_snapshot.snapshot_sqlite_to_blob(alias / "state.db", store)
    alias.unlink()
    alias.symlink_to(alternate, target_is_directory=True)
    second = sqlite_snapshot.snapshot_sqlite_to_blob(alias / "state.db", store)

    assert first.blob_hash == second.blob_hash
    assert first.captured_profile_key == profile_key(original)
    assert second.captured_profile_key == profile_key(alternate)
    assert first.captured_profile_key != second.captured_profile_key
    assert first.source_path == original / "state.db"
    assert second.source_path == alternate / "state.db"
    assert sqlite_snapshot.hermes_profile_raw_id(
        first.source_path,
        0,
        first.source_revision,
        identity_path=first.captured_profile_source_path,
        profile_identity=first.captured_profile_key,
    ) != sqlite_snapshot.hermes_profile_raw_id(
        second.source_path,
        0,
        second.source_revision,
        identity_path=second.captured_profile_source_path,
        profile_identity=second.captured_profile_key,
    )


def test_cross_parent_file_alias_keeps_its_declared_profile(tmp_path: Path) -> None:
    from polylogue.sources.parsers.hermes_identity import profile_key

    profile = tmp_path / "profile"
    profile.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    _database(external / "state.db", "same")
    alias = profile / "state.db"
    alias.symlink_to(external / "state.db")
    snapshot = sqlite_snapshot.snapshot_sqlite_to_blob(alias, BlobStore(tmp_path / "blobs"))

    assert snapshot.source_path == alias
    assert snapshot.identity_path == external / "state.db"
    assert snapshot.captured_profile_root == profile
    assert snapshot.captured_profile_key == profile_key(profile)
    assert snapshot.captured_profile_key != profile_key(external)


def test_nested_family_directory_alias_keeps_the_shared_profile_namespace(tmp_path: Path) -> None:
    from polylogue.sources.parsers.hermes_identity import profile_key

    profile = tmp_path / "profile"
    profile.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    _database(external / "state.db", "same")
    (profile / "sessions").symlink_to(external, target_is_directory=True)
    source = profile / "sessions" / "state.db"
    snapshot = sqlite_snapshot.snapshot_sqlite_to_blob(source, BlobStore(tmp_path / "blobs"))

    assert snapshot.identity_path == external / "state.db"
    assert snapshot.captured_profile_source_path == profile / "sessions" / "state.db"
    assert snapshot.captured_profile_key == profile_key(profile)
    sqlite_snapshot.hermes_profile_raw_id(
        snapshot.source_path,
        0,
        snapshot.source_revision,
        identity_path=snapshot.captured_profile_source_path,
        profile_identity=snapshot.captured_profile_key,
    )


def _attack_worker(monkeypatch: pytest.MonkeyPatch, source: Path, external: Path, attack: str) -> None:
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "sqlite_source" / "worker_attack.py"
    monkeypatch.setenv("POLYLOGUE_TEST_SQLITE_SOURCE", str(source))
    monkeypatch.setenv("POLYLOGUE_TEST_SQLITE_EXTERNAL", str(external))
    monkeypatch.setenv("POLYLOGUE_TEST_SQLITE_ATTACK", attack)
    monkeypatch.setenv("POLYLOGUE_TEST_SQLITE_SQL_MARKER", str(source.parent / "foreign-sql"))
    monkeypatch.setattr(sqlite_export, "_WORKER_COMMAND", f"import runpy; runpy.run_path({str(fixture)!r})")


@pytest.mark.parametrize("attack", ["main-symlink", "main-regular"])
@pytest.mark.parametrize("operation", ["frontier", "export", "shape", "backup", "staging"])
def test_actual_connect_aba_cannot_attribute_external_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str, operation: str
) -> None:
    """Metadata before/after connect would accept the restored declared path."""
    source, external = tmp_path / "declared.sqlite", tmp_path / "external.sqlite"
    _database(source, "declared")
    _database(external, "unrelated")
    _attack_worker(monkeypatch, source, external, attack)
    if operation == "frontier":
        frontier = build_source_frontier([SourceDeclaration("source", SourceRole.MUTABLE_SQLITE, source, True)])
        assert frontier.root_states["source"] is FrontierState.UNAVAILABLE
        assert not frontier.complete
        assert frontier.members == ()
    else:
        with pytest.raises(OSError) as failure:
            if operation == "export":
                sqlite_export.logical_export_bytes(source)
            elif operation == "shape":
                sqlite_export.logical_source_shape(source)
            elif operation == "backup":
                sqlite_snapshot.snapshot_sqlite_database(source, tmp_path / "backup.sqlite")
            else:
                sqlite_snapshot.stage_sqlite_snapshot(source, tmp_path / "backup.sqlite")
        assert failure.value.errno == errno.ESTALE
        assert not (tmp_path / "backup.sqlite").exists()
        assert not sqlite_snapshot.sqlite_staging_metadata_path(tmp_path / "backup.sqlite").exists()
    assert not (tmp_path / "foreign-sql").exists()
    with closing(sqlite3.connect(source)) as conn:
        assert conn.execute("SELECT value FROM state").fetchone() == ("declared",)


@pytest.mark.parametrize("operation", ["export", "shape", "backup"])
def test_actual_wal_sidecar_aba_cannot_publish_a_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Binding only the main fd misses unrelated WAL/SHM opened during first SQL."""
    source, external = tmp_path / "declared.sqlite", tmp_path / "external.sqlite"
    _database(source, "declared")
    _database(external, "unrelated")
    with closing(sqlite3.connect(source)) as writer, closing(sqlite3.connect(external)) as foreign:
        for conn in (writer, foreign):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA wal_autocheckpoint=0")
            conn.execute("INSERT INTO state VALUES ('wal-only')")
            conn.commit()
        _attack_worker(monkeypatch, source, external, "sidecar")
        with pytest.raises((OSError, sqlite3.Error)):
            if operation == "export":
                sqlite_export.logical_export_bytes(source)
            elif operation == "shape":
                sqlite_export.logical_source_shape(source)
            else:
                sqlite_snapshot.snapshot_sqlite_database(source, tmp_path / "backup.sqlite")
        assert not (tmp_path / "backup.sqlite").exists()


@pytest.mark.parametrize("wal", [False, True])
def test_bound_worker_preserves_logical_bytes_shape_and_staged_backup(tmp_path: Path, wal: bool) -> None:
    """Dropping WAL from the source proof loses a committed row absent from the main file."""
    source = tmp_path / "declared.sqlite"
    _database(source, "declared")
    with closing(sqlite3.connect(source)) as writer:
        if wal:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("INSERT INTO state VALUES ('wal-only')")
            writer.commit()
        payload = sqlite_export.logical_export_bytes(source)
        assert b"declared" in payload
        assert (b"wal-only" in payload) is wal
        assert sqlite_export.logical_source_shape(source) == {"state": ("value",)}
        destination = tmp_path / "backup.sqlite"
        sqlite_snapshot.snapshot_sqlite_database(source, destination)
        assert sqlite_export.logical_export_bytes(destination) == payload


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_sink_failure_reaps_the_reader_blocked_on_its_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    """Returning before child settlement leaves a live source transaction and zombie."""
    source = tmp_path / "declared.sqlite"
    _database(source, "declared")
    original_directory = sqlite_export.tempfile.TemporaryDirectory
    scratch_paths: list[Path] = []

    def temporary_directory(*args: Any, **kwargs: Any) -> Any:
        directory = original_directory(*args, **kwargs)
        scratch_paths.append(Path(directory.name))
        return directory

    monkeypatch.setattr(sqlite_export.tempfile, "TemporaryDirectory", temporary_directory)
    original = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        child = original(*args, **kwargs)
        children.append(child)
        return child

    class FailingSink:
        def write(self, payload: bytes) -> int:
            raise failure()

    monkeypatch.setattr(sqlite_export.subprocess, "Popen", launch)
    with pytest.raises(failure):
        sqlite_export.write_logical_export(source, FailingSink())
    assert len(children) == 2
    assert all(child.returncode is not None and child.poll() is not None for child in children)
    assert scratch_paths and all(not path.exists() for path in scratch_paths)


@pytest.mark.parametrize("frame", [b"X" + bytes(8), b"S" + bytes(8) + b"extra", b"D" + bytes(8)])
def test_malformed_worker_frames_refuse_and_reap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: bytes) -> None:
    """An invalid frame or premature success must not produce a shape result."""
    source = tmp_path / "declared.sqlite"
    _database(source, "declared")
    monkeypatch.setattr(
        sqlite_export,
        "_WORKER_COMMAND",
        "import sys; from polylogue.sources import sqlite_export as owner; "
        "kind,size=owner._FRAME_HEADER.unpack(owner._read_exact(sys.stdin.buffer,owner._FRAME_HEADER.size)); "
        "owner._read_exact(sys.stdin.buffer,size); "
        f"sys.stdout.buffer.write({frame!r}); sys.stdout.flush()",
    )
    with pytest.raises(OSError) as failure:
        sqlite_export.logical_source_shape(source)
    assert failure.value.errno == errno.EPROTO


def test_final_binding_failure_discards_the_private_blob_prefix(tmp_path: Path) -> None:
    """Final descriptor proof must complete before a blob is published."""
    source, external = tmp_path / "declared.sqlite", tmp_path / "external.sqlite"
    _database(source, "declared")
    _database(external, "unrelated")
    store = BlobStore(tmp_path / "blobs")
    changed = False

    def write(handle: IO[bytes]) -> None:
        class SubstitutingSink:
            def write(self, payload: bytes) -> int:
                nonlocal changed
                if not changed:
                    changed = True
                    source.rename(tmp_path / "held.sqlite")
                    source.symlink_to(external)
                return handle.write(payload)

        sqlite_export.write_logical_export(source, SubstitutingSink())

    with pytest.raises(OSError):
        store.write_from_writer(write)
    assert changed
    assert not any(path.is_file() for path in store.root.rglob("*"))


@pytest.mark.skipif(
    os.geteuid() == 0 or not hasattr(os, "O_PATH"), reason="requires unprivileged metadata-only directory open"
)
def test_file_source_under_execute_only_parent_remains_readable(tmp_path: Path) -> None:
    """A read-only directory anchor adds a permission requirement the source does not have."""
    parent = tmp_path / "parent"
    parent.mkdir()
    source = parent / "declared.sqlite"
    _database(source, "declared")
    parent.chmod(0o111)
    try:
        assert b"declared" in sqlite_export.logical_export_bytes(source)
        assert sqlite_export.logical_source_shape(source) == {"state": ("value",)}
    finally:
        parent.chmod(0o700)


@pytest.mark.parametrize("attack", ["unknown", "unlinked"])
def test_unknown_regular_descriptor_cannot_hide_as_sorter_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
) -> None:
    """Exempting unlinked files could hide a substituted source WAL descriptor."""
    source, external = tmp_path / "declared.sqlite", tmp_path / "external.sqlite"
    _database(source, "declared")
    _database(external, "unrelated")
    _attack_worker(monkeypatch, source, external, attack)
    with pytest.raises(OSError) as failure:
        sqlite_export.logical_export_bytes(source)
    assert failure.value.errno == errno.ESTALE


def test_staged_backup_provenance_uses_its_accepted_parent_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolving provenance after the completed read attributes it to a replacement alias."""
    original_parent = tmp_path / "original"
    replacement_parent = tmp_path / "replacement"
    for parent in (original_parent, replacement_parent):
        parent.mkdir()
        _database(parent / "declared.sqlite", parent.name)
    alias = tmp_path / "Documents"
    alias.symlink_to(original_parent, target_is_directory=True)
    source = alias / "declared.sqlite"
    backup = sqlite_snapshot._snapshot_sqlite_database_bound

    def replace_alias_after_backup(source: Path, destination: Path) -> tuple[Path, tuple[int, int]]:
        accepted = backup(source, destination)
        alias.unlink()
        alias.symlink_to(replacement_parent, target_is_directory=True)
        return accepted

    monkeypatch.setattr(sqlite_snapshot, "_snapshot_sqlite_database_bound", replace_alias_after_backup)
    staged = tmp_path / "staged.sqlite"
    sqlite_snapshot.stage_sqlite_snapshot(source, staged)
    with sqlite_snapshot.bind_sqlite_source(staged) as binding:
        assert binding.source_path == original_parent / "declared.sqlite"
    assert b"original" in sqlite_export.logical_export_bytes(staged)


def _trajectory_database(path: Path, identity: str) -> None:
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript(
            "CREATE TABLE trajectory_meta(trajectory_id TEXT,cascade_id TEXT);"
            "CREATE TABLE steps(idx INTEGER,step_type TEXT,step_format TEXT,step_payload TEXT);"
        )
        conn.execute("INSERT INTO trajectory_meta VALUES (?,?)", (identity, identity))
        conn.execute("INSERT INTO steps VALUES (1,'message','v1',?)", ('{"role":"user","text":"neutral message"}',))


@pytest.mark.parametrize("attack", ["main-symlink", "main-regular", "sidecar"])
@pytest.mark.parametrize("operation", ["explain", "preflight"])
def test_actual_preview_cannot_attribute_an_unrelated_restored_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str, operation: str
) -> None:
    """Both public preview routes previously detected A and parsed substituted B."""
    from polylogue.sources.import_explain import explain_import_path
    from polylogue.sources.import_preflight import ImportPreflightStatus, preflight_import_source

    source, external = tmp_path / "declared.sqlite", tmp_path / "external.sqlite"
    _trajectory_database(source, "declared-session")
    _trajectory_database(external, "external-session")
    with closing(sqlite3.connect(source)) as writer, closing(sqlite3.connect(external)) as foreign:
        if attack == "sidecar":
            for conn in (writer, foreign):
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA wal_autocheckpoint=0")
                conn.execute('INSERT INTO steps VALUES (2,\'message\',\'v1\',\'{"role":"user","text":"WAL message"}\')')
                conn.commit()
        _attack_worker(monkeypatch, source, external, attack)
        if operation == "explain":
            payload = explain_import_path(source, source_name="antigravity")
            assert payload.produced.sessions == 0
            assert payload.produced.session_refs == ()
            assert payload.skipped
        else:
            result = preflight_import_source(source)
            assert result.status is ImportPreflightStatus.MALFORMED
            assert result.supported_count == 0
        if attack.startswith("main-"):
            assert not (tmp_path / "foreign-sql").exists()


@pytest.mark.parametrize("semantics", ["nocase", "affinity", "view", "rowid"])
def test_bound_public_preview_preserves_native_sqlite_query_semantics(tmp_path: Path, semantics: str) -> None:
    """Replacing a live native read with untyped logical reconstruction drops these rows."""
    from polylogue.sources.import_explain import explain_import_path
    from polylogue.sources.import_preflight import ImportPreflightStatus, preflight_import_source

    source = tmp_path / "declared.sqlite"
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute("CREATE TABLE trajectory_meta(trajectory_id TEXT,cascade_id TEXT)")
        kind = "NUMERIC" if semantics == "affinity" else "TEXT COLLATE NOCASE"
        table = "step_values" if semantics == "view" else "steps"
        conn.execute(
            f"CREATE TABLE {table}(trajectory_id {kind},idx INTEGER,step_type TEXT,step_format TEXT,step_payload TEXT)"
        )
        if semantics == "view":
            conn.execute(
                "CREATE VIEW steps AS SELECT trajectory_id COLLATE NOCASE AS trajectory_id,idx,step_type,step_format,step_payload FROM step_values"
            )
        native_id, stored_key = ("123", "123.0") if semantics == "affinity" else ("ABC", "abc")
        conn.execute(
            "INSERT INTO trajectory_meta(rowid,trajectory_id,cascade_id) VALUES (400,?,?)", (native_id, native_id)
        )
        conn.execute(
            f"INSERT INTO {table} VALUES (?,1,'message','v1',?)", (stored_key, '{"role":"user","text":"native read"}')
        )
        if semantics == "rowid":
            conn.execute("INSERT INTO trajectory_meta(rowid,trajectory_id,cascade_id) VALUES (20,'FIRST','FIRST')")
            conn.execute("INSERT INTO steps VALUES ('first',2,'message','v1','{\"role\":\"user\",\"text\":\"first\"}')")
    payload = explain_import_path(source, source_name="antigravity")
    assert payload.produced.messages == (2 if semantics == "rowid" else 1)
    expected_refs = (
        ("session:antigravity:FIRST", "session:antigravity:ABC")
        if semantics == "rowid"
        else (f"session:antigravity:{native_id}",)
    )
    assert payload.produced.session_refs == expected_refs
    assert preflight_import_source(source).status is ImportPreflightStatus.SUPPORTED


def test_retained_export_preview_uses_the_existing_private_reconstruction(tmp_path: Path) -> None:
    """The worker must not replace retained export semantics with a native page copy."""
    from polylogue.sources.import_explain import explain_import_path
    from polylogue.sources.import_preflight import ImportPreflightStatus, preflight_import_source

    source = tmp_path / "source.sqlite"
    _trajectory_database(source, "retained-session")
    retained = tmp_path / "retained.sqlite"
    retained.write_bytes(sqlite_export.logical_export_bytes(source))
    payload = explain_import_path(retained)
    assert payload.produced.session_refs == ("session:antigravity:retained-session",)
    assert payload.produced.messages == 1
    assert preflight_import_source(retained).status is ImportPreflightStatus.SUPPORTED


def test_sqlite_directory_frontier_refuses_a_substituted_ancestor_with_the_same_main_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching main inode alone cannot prove containment under the declared root."""
    from polylogue.sources import source_snapshot

    root = tmp_path / "declared"
    parent = root / "nested"
    parent.mkdir(parents=True)
    source = parent / "state.sqlite"
    _database(source, "declared")
    external = tmp_path / "external"
    external.mkdir()
    os.link(source, external / source.name)
    original = source_snapshot._logical_export_digest_bound
    attacked = False

    def substitute(path: Path, **kwargs: Any) -> str:
        nonlocal attacked
        attacked = True
        held = tmp_path / "held-parent"
        parent.rename(held)
        parent.symlink_to(external, target_is_directory=True)
        try:
            assert path.stat().st_ino == (held / source.name).stat().st_ino
            return original(path, **kwargs)
        finally:
            parent.unlink()
            held.rename(parent)

    monkeypatch.setattr(source_snapshot, "_logical_export_digest_bound", substitute)
    frontier = build_source_frontier([SourceDeclaration("source", SourceRole.MUTABLE_SQLITE, root, True)])
    assert attacked
    assert frontier.root_states["source"] is FrontierState.UNAVAILABLE
    assert frontier.members == ()
    assert not frontier.complete


def test_malformed_callback_ack_refuses_reaps_and_removes_private_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ACK failure cannot leave a completed export or a live reader transaction."""
    source = tmp_path / "declared.sqlite"
    _database(source, "declared")
    original = subprocess.Popen
    original_directory = sqlite_export.tempfile.TemporaryDirectory
    children: list[subprocess.Popen[bytes]] = []
    scratch_paths: list[Path] = []

    class WrongAck:
        def __init__(self, stream: Any) -> None:
            self.stream = stream

        def write(self, payload: bytes) -> int:
            return int(self.stream.write(b"X" if payload == b"A" else payload))

        def flush(self) -> None:
            self.stream.flush()

        def close(self) -> None:
            self.stream.close()

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        child = original(*args, **kwargs)
        child.stdin = cast(IO[bytes], WrongAck(child.stdin))
        children.append(child)
        return child

    def temporary_directory(*args: Any, **kwargs: Any) -> Any:
        directory = original_directory(*args, **kwargs)
        scratch_paths.append(Path(directory.name))
        return directory

    monkeypatch.setattr(sqlite_export.subprocess, "Popen", launch)
    monkeypatch.setattr(sqlite_export.tempfile, "TemporaryDirectory", temporary_directory)
    with pytest.raises(OSError) as failure:
        sqlite_export.logical_export_bytes(source)
    assert failure.value.errno == errno.EPROTO
    assert len(children) == 2 and all(child.poll() is not None for child in children)
    assert scratch_paths and all(not path.exists() for path in scratch_paths)
    assert source.exists()


def test_staging_provenance_follows_the_actual_accepted_backup_coordinate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disconnected pre-backup resolve would label backed-up B as source A."""
    original_parent, replacement_parent = tmp_path / "original", tmp_path / "replacement"
    for parent in (original_parent, replacement_parent):
        parent.mkdir()
        _database(parent / "declared.sqlite", parent.name)
    alias = tmp_path / "Documents"
    alias.symlink_to(original_parent, target_is_directory=True)
    source = alias / "declared.sqlite"
    backup = sqlite_snapshot._snapshot_sqlite_database_bound

    def replace_alias_before_acceptance(source: Path, destination: Path) -> tuple[Path, tuple[int, int]]:
        alias.unlink()
        alias.symlink_to(replacement_parent, target_is_directory=True)
        try:
            return backup(source, destination)
        finally:
            alias.unlink()
            alias.symlink_to(original_parent, target_is_directory=True)

    monkeypatch.setattr(sqlite_snapshot, "_snapshot_sqlite_database_bound", replace_alias_before_acceptance)
    staged = tmp_path / "staged.sqlite"
    sqlite_snapshot.stage_sqlite_snapshot(source, staged)
    with sqlite_snapshot.bind_sqlite_source(staged) as binding:
        assert binding.source_path == replacement_parent / "declared.sqlite"
    assert b"replacement" in sqlite_export.logical_export_bytes(staged)
    assert b"original" not in sqlite_export.logical_export_bytes(staged)


@pytest.mark.parametrize("name", ["déclared:%#?.sqlite", "declared-\udcff.sqlite"])
def test_bound_worker_preserves_os_filename_bytes_and_uri_metacharacters(tmp_path: Path, name: str) -> None:
    """UTF-8-only control JSON or an unquoted relative URI refuses a valid source name."""
    source = tmp_path / name
    _database(source, "declared")
    assert b"declared" in sqlite_export.logical_export_bytes(source)
    assert sqlite_export.logical_source_shape(source) == {"state": ("value",)}


def test_staged_reader_refuses_both_publication_gaps_and_recovers_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata B cannot select B's tables or attribution while the opened database is A."""
    from polylogue.sources.import_explain import explain_import_path
    from polylogue.sources.import_preflight import ImportPreflightStatus, preflight_import_source

    original = tmp_path / "state_5.sqlite"
    replacement = tmp_path / "state.db"
    for source in (original, replacement):
        with closing(sqlite3.connect(source)) as connection, connection:
            connection.executescript(
                "CREATE TABLE threads(value TEXT); INSERT INTO threads VALUES ('thread');"
                "CREATE TABLE sessions(value TEXT); INSERT INTO sessions VALUES ('session');"
                "CREATE TABLE messages(value TEXT); INSERT INTO messages VALUES ('message');"
            )
    staged = tmp_path / "staged.sqlite"
    sqlite_snapshot.stage_sqlite_snapshot(original, staged)
    old_inode = staged.stat().st_ino
    metadata = sqlite_snapshot.sqlite_staging_metadata_path(staged)
    replace = os.replace
    refused = []

    def inspect_gap() -> None:
        with pytest.raises(OSError) as failure:
            sqlite_snapshot.snapshot_sqlite_to_blob(staged, BlobStore(tmp_path / "blobs"))
        assert failure.value.errno == errno.ESTALE
        explanation = explain_import_path(staged, source_name="hermes")
        assert explanation.produced.sessions == 0
        assert explanation.produced.session_refs == ()
        assert explanation.skipped
        assert preflight_import_source(staged).status is ImportPreflightStatus.MALFORMED
        refused.append(True)

    def replace_then_fail(source: str | Path, target: str | Path) -> None:
        if Path(target) == staged:
            inspect_gap()
            raise OSError(errno.EIO, "synthetic database publication failure")
        replace(source, target)
        if Path(target) == metadata:
            inspect_gap()

    monkeypatch.setattr(sqlite_snapshot.os, "replace", replace_then_fail)
    with pytest.raises(OSError) as failure:
        sqlite_snapshot.stage_sqlite_snapshot(replacement, staged)
    assert failure.value.errno == errno.EIO
    assert staged.stat().st_ino == old_inode
    assert len(refused) == 2
    inspect_gap()
    monkeypatch.setattr(sqlite_snapshot.os, "replace", replace)
    sqlite_snapshot.stage_sqlite_snapshot(replacement, staged)
    snapshot = sqlite_snapshot.snapshot_sqlite_to_blob(staged, BlobStore(tmp_path / "blobs"))
    assert snapshot.source_path == replacement
    header = sqlite_export.read_export_header(BlobStore(tmp_path / "blobs").blob_path(snapshot.blob_hash))
    assert header.member == "state.db"
    assert header.tables == ("messages", "sessions")
    assert "threads" not in header.tables


def test_captured_staging_binding_refuses_replaced_metadata_before_any_export(
    tmp_path: Path,
) -> None:
    """A pathname-only metadata check would apply the previous declaration to a new sidecar."""
    source = tmp_path / "declared.sqlite"
    _database(source, "declared")
    staged = tmp_path / "staged.sqlite"
    sqlite_snapshot.stage_sqlite_snapshot(source, staged)
    with sqlite_snapshot.bind_sqlite_source(staged) as binding:
        metadata = sqlite_snapshot.sqlite_staging_metadata_path(staged)
        metadata.write_text('{"version":1,"original_source_path":"/synthetic/state.db"}')
        with pytest.raises(OSError) as failure:
            sqlite_snapshot.snapshot_sqlite_to_blob(staged, BlobStore(tmp_path / "blobs"), source_binding=binding)
        assert failure.value.errno == errno.ESTALE
    assert staged.exists() and source.exists()


def test_metadata_substitution_cannot_release_a_parent_sqlite_readers_locks(tmp_path: Path) -> None:
    """Closing an ordinary metadata FD aliased to this DB would release the parent's read lock."""
    import sys

    source, locked = tmp_path / "declared.sqlite", tmp_path / "locked.sqlite"
    _database(source, "declared")
    _database(locked, "locked")
    metadata = sqlite_snapshot.sqlite_staging_metadata_path(source)
    with closing(sqlite3.connect(locked)) as reader:
        reader.execute("BEGIN").close()
        assert reader.execute("SELECT value FROM state").fetchone() == ("locked",)
        os.link(locked, metadata)
        with pytest.raises(OSError) as failure, sqlite_snapshot.bind_sqlite_source(source):
            pytest.fail("SQLite bytes cannot be staging JSON")
        assert failure.value.errno == errno.ESTALE
        writer = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sqlite3,sys; c=sqlite3.connect(sys.argv[1], timeout=0); "
                    "\ntry: c.execute('BEGIN EXCLUSIVE')"
                    "\nexcept sqlite3.OperationalError as e: sys.exit(0 if e.sqlite_errorcode==sqlite3.SQLITE_BUSY else 2)"
                    "\nelse: sys.exit(1)"
                ),
                str(locked),
            ],
            check=False,
            capture_output=True,
        )
        assert writer.returncode == 0, writer.stderr
        assert reader.execute("SELECT value FROM state").fetchone() == ("locked",)


def test_explicit_stable_sqlite_root_alias_uses_the_accepted_actual_root(tmp_path: Path) -> None:
    """Root aliases are legitimate declarations; enumerated member aliases remain refused."""
    actual_parent = tmp_path / "actual"
    actual_parent.mkdir()
    actual = actual_parent / "declared.sqlite"
    _database(actual, "declared")
    alias = tmp_path / "alias.sqlite"
    alias.symlink_to(actual)
    assert sqlite_export.logical_export_bytes(alias) == sqlite_export.logical_export_bytes(actual)
    snapshot = sqlite_snapshot.snapshot_sqlite_to_blob(alias, BlobStore(tmp_path / "blobs"))
    assert snapshot.source_path == alias
    assert snapshot.identity_path == actual
    frontier = build_source_frontier([SourceDeclaration("source", SourceRole.MUTABLE_SQLITE, alias, True)])
    assert frontier.root_states["source"] is FrontierState.PRESENT
    assert frontier.complete and len(frontier.members) == 1


def test_nonregular_staging_metadata_is_unavailable_before_read(tmp_path: Path) -> None:
    source = tmp_path / "declared.sqlite"
    _database(source, "declared")
    metadata = source.with_name(source.name + sqlite_snapshot._STAGING_METADATA_SUFFIX)
    os.mkfifo(metadata)
    with pytest.raises(OSError) as refused:
        with sqlite_snapshot.bind_sqlite_source(source):
            pytest.fail("nonregular provenance cannot bind a source")
    assert refused.value.errno == errno.ESTALE


@pytest.mark.parametrize(
    "payload",
    [b"not-json", b"{}", b'{"version":2,"original_source_path":"/synthetic/state.db","database_identity":[0,0]}'],
)
def test_present_invalid_staging_provenance_never_falls_back_to_filename(tmp_path: Path, payload: bytes) -> None:
    source = tmp_path / "declared.sqlite"
    _database(source, "declared")
    sqlite_snapshot.sqlite_staging_metadata_path(source).write_bytes(payload)
    with pytest.raises(OSError) as refused:
        sqlite_snapshot.snapshot_sqlite_to_blob(source, BlobStore(tmp_path / "blobs"))
    assert refused.value.errno == errno.ESTALE


@pytest.mark.skipif(os.geteuid() == 0, reason="requires unprivileged metadata read permissions")
def test_present_unreadable_provenance_is_unavailable(tmp_path: Path) -> None:
    source = tmp_path / "declared.sqlite"
    _database(source, "declared")
    metadata = sqlite_snapshot.sqlite_staging_metadata_path(source)
    metadata.write_text("{}")
    metadata.chmod(0)
    try:
        with pytest.raises(OSError) as refused:
            with sqlite_snapshot.bind_sqlite_source(source):
                pytest.fail("unreadable provenance cannot become an ordinary source")
        assert refused.value.errno == errno.EACCES
    finally:
        metadata.chmod(0o600)


def test_absent_provenance_preserves_ordinary_declared_semantics(tmp_path: Path) -> None:
    source = tmp_path / "declared.sqlite"
    _database(source, "declared")
    with sqlite_snapshot.bind_sqlite_source(source) as binding:
        assert not binding.staged
        assert binding.source_path == source
        assert binding.identity_path == source
