"""Mutable SQLite sources are identified by logical content, not page bytes.

A live database's page image changes after every commit, checkpoint and
vacuum, so byte identity re-acquires whole snapshots of content the archive
already holds. These tests drive the production acquisition route
(``snapshot_sqlite_to_blob`` and ``LiveBatchProcessor.ingest_files``) and the
production freshness gate (``LiveWatcher._needs_work_from_state``) and assert
that identity, idempotency and continuity follow the declared logical
revision.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import polylogue.sources.live.watcher as live_watcher
import polylogue.sources.sqlite_snapshot as sqlite_snapshot
from polylogue import Polylogue
from polylogue.core.enums import Provider
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import LiveWatcher
from polylogue.sources.origin_specs import database_capability_for_provider
from polylogue.sources.sqlite_snapshot import (
    codex_state_raw_id,
    hermes_profile_raw_id,
    retained_content_revision,
    snapshot_sqlite_to_blob,
    sqlite_logical_revision,
    sqlite_source_revision,
)
from polylogue.storage.blob_store import BlobStore

_STATE_DB_SCHEMA = """
CREATE TABLE schema_version(version INTEGER NOT NULL);
INSERT INTO schema_version(version) VALUES (19);
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT,
    model_config TEXT,
    parent_session_id TEXT,
    started_at REAL,
    ended_at REAL,
    end_reason TEXT,
    title TEXT
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    timestamp REAL NOT NULL,
    tool_calls TEXT,
    observed INTEGER DEFAULT 0,
    active INTEGER DEFAULT 1,
    compacted INTEGER DEFAULT 0
);
"""


def _write_state_db(path: Path, *, sessions: int = 1, wal: bool = False) -> None:
    with closing(sqlite3.connect(path)) as conn, conn:
        if wal:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_STATE_DB_SCHEMA)
        for index in range(sessions):
            conn.execute(
                "INSERT INTO sessions (id, source, model_config, started_at, ended_at, end_reason, title) "
                "VALUES (?, 'cli', '{}', 1.0, 8.0, 'completed', ?)",
                (f"session-{index}", f"title-{index}"),
            )
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, 'user', ?, 2.0)",
                (index + 1, f"session-{index}", f"hello {index}"),
            )


def _blob_store(tmp_path: Path) -> BlobStore:
    return BlobStore(tmp_path / "blob")


# ---------------------------------------------------------------------------
# Identity: logical content, never page bytes
# ---------------------------------------------------------------------------


def test_a_repaged_database_keeps_the_identity_its_content_earns(tmp_path: Path) -> None:
    """Anti-vacuity: key ``codex_state_raw_id`` on ``snapshot.blob_hash``
    again and the two ids diverge here, which is the 13-snapshot residue.

    Vacuuming to a new page size rewrites every page and every offset in the
    file while leaving the schema and every row exactly where they were.
    """
    source = tmp_path / "state_5.sqlite"
    _write_state_db(source, sessions=3)
    store = _blob_store(tmp_path)

    before = snapshot_sqlite_to_blob(source, store)
    with closing(sqlite3.connect(source)) as conn:
        conn.execute("PRAGMA page_size=8192")
        conn.execute("VACUUM")
    after = snapshot_sqlite_to_blob(source, store)

    assert before.blob_hash != after.blob_hash, "sanity: the page image was rewritten"
    assert before.source_revision == after.source_revision
    assert codex_state_raw_id(source, before.source_revision) == codex_state_raw_id(source, after.source_revision)


def test_freelist_churn_from_an_insert_and_delete_changes_no_identity(tmp_path: Path) -> None:
    """The ordinary live shape: rows written and removed between acquisitions.

    Anti-vacuity: identify the source by its bytes and this mints a second
    revision for a database holding exactly the rows already acquired.
    """
    source = tmp_path / "state_5.sqlite"
    _write_state_db(source, sessions=3)
    store = _blob_store(tmp_path)
    before = snapshot_sqlite_to_blob(source, store)

    with closing(sqlite3.connect(source)) as conn, conn:
        conn.executemany(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, 'session-0', 'user', ?, 9.0)",
            [(1000 + index, "x" * 400) for index in range(500)],
        )
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute("DELETE FROM messages WHERE id >= 1000")

    after = snapshot_sqlite_to_blob(source, store)

    assert before.blob_hash != after.blob_hash, "sanity: the freelist moved the page image"
    assert before.source_revision == after.source_revision
    assert codex_state_raw_id(source, before.source_revision) == codex_state_raw_id(source, after.source_revision)


def test_one_changed_row_invalidates_identity_under_byte_similarity(tmp_path: Path) -> None:
    """Anti-vacuity: digest the file bytes instead of the typed rows and a
    same-length overwrite reports the database as unchanged.

    The replacement title is the same length, so page count, file size and
    every page offset are unchanged; only one row's value differs.
    """
    source = tmp_path / "state_5.sqlite"
    _write_state_db(source, sessions=3)
    store = _blob_store(tmp_path)
    before = snapshot_sqlite_to_blob(source, store)
    size_before = source.stat().st_size

    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute("UPDATE sessions SET title = 'title-X' WHERE id = 'session-1'")

    after = snapshot_sqlite_to_blob(source, store)

    assert source.stat().st_size == size_before, "sanity: the overwrite kept the file length"
    assert before.source_revision != after.source_revision
    assert codex_state_raw_id(source, before.source_revision) != codex_state_raw_id(source, after.source_revision)


def test_restoring_an_older_page_image_does_not_pass_as_the_state_it_replaced(tmp_path: Path) -> None:
    """A byte-level restore cannot mint continuity for content it does not carry.

    The restore keeps the replaced state's size and mtime, so the filesystem
    fingerprint still reports that state. Anti-vacuity: identify the source by
    ``sqlite_source_revision`` and the restored file passes as the newer
    content it no longer holds.
    """
    source = tmp_path / "state_5.sqlite"
    _write_state_db(source, sessions=3)
    original_bytes = source.read_bytes()
    original_revision = sqlite_logical_revision(source)

    with closing(sqlite3.connect(source)) as conn, conn:
        # Same length as 'title-2', so the file size never moves.
        conn.execute("UPDATE sessions SET title = 'title-Y' WHERE id = 'session-2'")
    changed_revision = sqlite_logical_revision(source)
    changed_stat = source.stat()
    changed_fingerprint = sqlite_source_revision(source)
    assert changed_revision != original_revision

    source.write_bytes(original_bytes)
    os.utime(source, ns=(changed_stat.st_atime_ns, changed_stat.st_mtime_ns))

    assert source.stat().st_size == changed_stat.st_size
    assert sqlite_source_revision(source) == changed_fingerprint, "sanity: the restore is invisible to the filesystem"
    assert sqlite_logical_revision(source) == original_revision
    assert sqlite_logical_revision(source) != changed_revision


def test_schema_change_moves_the_logical_revision(tmp_path: Path) -> None:
    """Anti-vacuity: digest rows only and an added column or table is invisible."""
    source = tmp_path / "state_5.sqlite"
    _write_state_db(source, sessions=1)
    baseline = sqlite_logical_revision(source)

    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute("ALTER TABLE sessions ADD COLUMN workspace TEXT")
    added_column = sqlite_logical_revision(source)
    assert added_column != baseline

    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute("CREATE TABLE thread_goals (thread_id TEXT PRIMARY KEY, objective TEXT)")
    assert sqlite_logical_revision(source) != added_column


def test_two_empty_members_of_one_profile_keep_distinct_identities(tmp_path: Path) -> None:
    """Anti-vacuity: drop the member filename from the Hermes domain and two
    empty declared members of one profile collapse onto one raw id."""
    profile = tmp_path / "hermes"
    profile.mkdir()
    state = profile / "state.db"
    verification = profile / "verification_evidence.db"
    for path in (state, verification):
        sqlite3.connect(path).close()

    revision = sqlite_logical_revision(state)
    assert sqlite_logical_revision(verification) == revision, "sanity: both members are logically empty"
    assert hermes_profile_raw_id(state, 0, revision) != hermes_profile_raw_id(verification, 0, revision)


def test_retained_blob_yields_the_same_content_term_as_live_acquisition(tmp_path: Path) -> None:
    """The import and replay routes must not mint a second identity.

    Anti-vacuity: return ``blob_hash`` unconditionally from
    ``retained_content_revision`` and the imported raw id stops matching the
    live one for the same database state.
    """
    source = tmp_path / "state.db"
    _write_state_db(source, sessions=2)
    store = _blob_store(tmp_path)
    snapshot = snapshot_sqlite_to_blob(source, store)

    assert retained_content_revision(store.blob_path(snapshot.blob_hash), snapshot.blob_hash) == (
        snapshot.source_revision
    )


def test_non_sqlite_material_is_identified_by_its_bytes(tmp_path: Path) -> None:
    """Anti-vacuity: try to open every blob as SQLite and a Hermes ATOF
    stream's raw identity raises instead of resolving."""
    store = _blob_store(tmp_path)
    blob_hash, _size = store.write_from_bytes(b'{"event": "atof"}\n')

    assert retained_content_revision(store.blob_path(blob_hash), blob_hash) == blob_hash


# ---------------------------------------------------------------------------
# Consistency: WAL, concurrent commit, corruption
# ---------------------------------------------------------------------------


def test_wal_source_with_an_uncommitted_writer_snapshots_committed_state_only(tmp_path: Path) -> None:
    """Anti-vacuity: drop the ``BEGIN`` in ``sqlite_logical_revision`` or the
    ``mode=ro`` backup and the open write transaction leaks into the snapshot.
    """
    source = tmp_path / "state.db"
    _write_state_db(source, sessions=2, wal=True)
    store = _blob_store(tmp_path)

    writer = sqlite3.connect(source)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO sessions (id, source, model_config, started_at, ended_at, end_reason, title) "
            "VALUES ('uncommitted', 'cli', '{}', 1.0, 8.0, 'completed', 'uncommitted')"
        )
        snapshot = snapshot_sqlite_to_blob(source, store)
    finally:
        writer.rollback()
        writer.close()

    blob = store.blob_path(snapshot.blob_hash)
    with closing(sqlite3.connect(f"file:{blob.resolve()}?mode=ro&immutable=1", uri=True)) as conn:
        ids = {str(row[0]) for row in conn.execute("SELECT id FROM sessions")}
    assert ids == {"session-0", "session-1"}
    assert sqlite_logical_revision(blob, immutable=True) == snapshot.source_revision


def test_a_commit_during_the_backup_refuses_the_acquisition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A torn observation is refused, never published.

    Anti-vacuity: delete the post-backup revision comparison in
    ``snapshot_sqlite_to_blob`` and this acquisition succeeds while carrying
    a revision that describes neither the state before nor after the commit.
    """
    source = tmp_path / "state.db"
    _write_state_db(source, sessions=2, wal=True)
    store = _blob_store(tmp_path)
    real_backup = sqlite_snapshot.snapshot_sqlite_database

    def backup_then_commit(src: Path, destination: Path) -> None:
        real_backup(src, destination)
        with closing(sqlite3.connect(src)) as conn, conn:
            conn.execute(
                "INSERT INTO sessions (id, source, model_config, started_at, ended_at, end_reason, title) "
                "VALUES ('raced', 'cli', '{}', 1.0, 8.0, 'completed', 'raced')"
            )

    monkeypatch.setattr(sqlite_snapshot, "snapshot_sqlite_database", backup_then_commit)

    with pytest.raises(OSError, match="SQLite source changed during backup"):
        snapshot_sqlite_to_blob(source, store)


def test_a_corrupt_database_fails_typed_and_publishes_nothing(tmp_path: Path) -> None:
    """Anti-vacuity: swallow the SQLite error and a damaged database is
    acquired as an empty logical revision, which reads as a healthy source
    that lost all its content."""
    source = tmp_path / "state.db"
    _write_state_db(source, sessions=2)
    payload = bytearray(source.read_bytes())
    # Keep the file-format header intact so the failure is a database error
    # rather than a "not a database" refusal, and damage every page the
    # schema and its tables are rooted in.
    page_size = int.from_bytes(payload[16:18], "big") or 4096
    payload[page_size:] = b"\xff" * (len(payload) - page_size)
    source.write_bytes(bytes(payload))
    store = _blob_store(tmp_path)

    with pytest.raises(sqlite3.DatabaseError):
        snapshot_sqlite_to_blob(source, store)


def test_a_write_locked_database_is_still_acquirable(tmp_path: Path) -> None:
    """Acquisition never waits on a writer: it reads a snapshot, not a lock.

    Anti-vacuity: open the source read-write in ``snapshot_sqlite_database``
    and this raises ``database is locked``.
    """
    source = tmp_path / "state.db"
    _write_state_db(source, sessions=2, wal=True)
    store = _blob_store(tmp_path)

    holder = sqlite3.connect(source)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        snapshot = snapshot_sqlite_to_blob(source, store)
    finally:
        holder.rollback()
        holder.close()

    assert snapshot.source_revision == sqlite_logical_revision(source)


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", [Provider.CODEX, Provider.HERMES])
def test_every_admitted_database_member_names_its_logical_product(provider: Provider) -> None:
    """Anti-vacuity: drop ``logical_tables`` from any admitted member and the
    declaration no longer says what the snapshot is acquired for."""
    capability = database_capability_for_provider(provider)
    assert capability is not None
    for member in capability.members:
        if member.disposition == "out-of-scope":
            assert member.logical_tables == ()
            assert member.consumer is None
            continue
        assert member.logical_tables, member.filename
        if member.disposition == "acquire":
            assert member.consumer, member.filename


@pytest.mark.parametrize("provider", [Provider.CODEX, Provider.HERMES])
def test_declared_revision_identity_is_the_logical_one(provider: Provider) -> None:
    """Anti-vacuity: restore the ``dev/inode/size/mtime_ns`` wording and the
    declaration describes a fingerprint the acquisition route stopped using."""
    capability = database_capability_for_provider(provider)
    assert capability is not None
    assert "sqlite_logical_revision" in capability.revision_identity
    assert "logical revision" in capability.raw_id_strategy


def test_declared_consumers_resolve_to_real_symbols() -> None:
    """Anti-vacuity: rename a consumer without updating the declaration and
    the member claims a reader that no longer exists."""
    import importlib

    for provider in (Provider.CODEX, Provider.HERMES):
        capability = database_capability_for_provider(provider)
        assert capability is not None
        for member in capability.members:
            if not member.consumer:
                continue
            module_path, _, symbol = member.consumer.partition(":")
            module = importlib.import_module(module_path.removesuffix(".py").replace("/", "."))
            assert hasattr(module, symbol), member.consumer


def test_declared_logical_tables_exist_in_the_parsed_schema(tmp_path: Path) -> None:
    """The Hermes state member's declared tables are the ones its parser needs.

    Anti-vacuity: declare a table the schema does not have and this is red.
    """
    source = tmp_path / "state.db"
    _write_state_db(source, sessions=1)
    with sqlite3.connect(source) as conn:
        present = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    capability = database_capability_for_provider(Provider.HERMES)
    assert capability is not None
    member = capability.member("state.db")
    assert member is not None
    assert set(member.logical_tables) <= present


# ---------------------------------------------------------------------------
# Production route: ingest and the freshness gate
# ---------------------------------------------------------------------------


def _hermes_source(root: Path) -> WatchSource:
    return WatchSource(name="hermes", root=root, suffixes=(".json", ".jsonl", ".db", ".sqlite", ".sqlite3"))


def _processor(
    workspace_env: dict[str, Path], root_name: str, db_name: str
) -> tuple[Any, Any, Path, CursorStore, Path]:
    root = workspace_env["data_root"] / root_name
    root.mkdir(parents=True)
    db_path = workspace_env["data_root"] / db_name
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (_hermes_source(root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    return archive, processor, root, cursor, db_path


def _raw_rows(archive_root: Path, source_path: Path) -> list[str]:
    source_db = archive_root / "source.db"
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as conn:
        return [
            str(row[0])
            for row in conn.execute(
                "SELECT raw_id FROM raw_sessions WHERE source_path = ? ORDER BY raw_id",
                (str(source_path),),
            )
        ]


@pytest.mark.asyncio
async def test_reingesting_a_repaged_database_adds_no_raw_revision(
    workspace_env: dict[str, Path],
) -> None:
    """Anti-vacuity: key raw identity on the blob hash and the second ingest
    writes a second raw row for content the archive already holds -- the
    shape that accumulated 13 snapshots over five databases.
    """
    archive, processor, root, _cursor, _db = _processor(workspace_env, "hermes-vacuum", "hermes-vacuum-cursor.db")
    source_path = root / "state.db"
    try:
        _write_state_db(source_path, sessions=2)
        first = await processor.ingest_files([source_path], emit_event=False)
        assert first.failed_file_count == 0
        after_first = _raw_rows(workspace_env["archive_root"], source_path)
        assert len(after_first) == 1

        with closing(sqlite3.connect(source_path)) as conn:
            conn.execute("PRAGMA page_size=8192")
            conn.execute("VACUUM")

        second = await processor.ingest_files([source_path], emit_event=False)
        assert second.failed_file_count == 0
        assert _raw_rows(workspace_env["archive_root"], source_path) == after_first
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_one_changed_row_produces_exactly_one_new_raw_revision(
    workspace_env: dict[str, Path],
) -> None:
    """Anti-vacuity: collapse identity onto the source path alone and the
    changed state never gets its own revision."""
    archive, processor, root, _cursor, _db = _processor(workspace_env, "hermes-delta", "hermes-delta-cursor.db")
    source_path = root / "state.db"
    try:
        _write_state_db(source_path, sessions=2)
        await processor.ingest_files([source_path], emit_event=False)
        after_first = _raw_rows(workspace_env["archive_root"], source_path)

        with closing(sqlite3.connect(source_path)) as conn, conn:
            conn.execute("UPDATE sessions SET title = 'renamed' WHERE id = 'session-1'")

        await processor.ingest_files([source_path], emit_event=False)
        after_second = _raw_rows(workspace_env["archive_root"], source_path)

        assert len(after_first) == 1
        assert len(after_second) == 2
        assert set(after_first) < set(after_second)
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_the_freshness_gate_skips_a_checkpointed_but_unchanged_database(
    workspace_env: dict[str, Path],
) -> None:
    """A WAL checkpoint moves every page and no logical row.

    Anti-vacuity: restore ``cursor.tail_hash != sqlite_source_revision(path)``
    as the whole gate and this returns True, re-snapshotting the database on
    every checkpoint.
    """
    archive, processor, root, cursor, db_path = _processor(
        workspace_env, "hermes-checkpoint", "hermes-checkpoint-cursor.db"
    )
    source_path = root / "state.db"
    try:
        _write_state_db(source_path, sessions=2, wal=True)
        await processor.ingest_files([source_path], emit_event=False)

        with closing(sqlite3.connect(source_path)) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        os.utime(source_path, ns=(0, 0))

        watcher = LiveWatcher(
            cast(
                Any,
                SimpleNamespace(
                    archive_root=workspace_env["archive_root"],
                    backend=SimpleNamespace(db_path=db_path),
                    config=None,
                ),
            ),
            (_hermes_source(root),),
            cursor=cursor,
            debounce_s=0.0,
        )
        record = cursor.get_record(source_path)
        assert record is not None
        assert record.tail_hash != sqlite_source_revision(source_path), "sanity: the filesystem state moved"

        assert watcher._needs_work_from_state(source_path, stat=source_path.stat(), cursor=record) is False
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_the_freshness_gate_reopens_a_logically_changed_database(
    workspace_env: dict[str, Path],
) -> None:
    """Anti-vacuity: return False whenever the recorded fingerprint is set and
    a changed database is never re-read."""
    archive, processor, root, cursor, db_path = _processor(workspace_env, "hermes-changed", "hermes-changed-cursor.db")
    source_path = root / "state.db"
    try:
        _write_state_db(source_path, sessions=2, wal=True)
        await processor.ingest_files([source_path], emit_event=False)

        with closing(sqlite3.connect(source_path)) as conn, conn:
            conn.execute("UPDATE sessions SET title = 'renamed' WHERE id = 'session-0'")

        watcher = LiveWatcher(
            cast(
                Any,
                SimpleNamespace(
                    archive_root=workspace_env["archive_root"],
                    backend=SimpleNamespace(db_path=db_path),
                    config=None,
                ),
            ),
            (_hermes_source(root),),
            cursor=cursor,
            debounce_s=0.0,
        )
        record = cursor.get_record(source_path)
        assert record is not None

        assert watcher._needs_work_from_state(source_path, stat=source_path.stat(), cursor=record) is True
    finally:
        await archive.close()


def test_an_out_of_scope_database_is_not_a_declared_codex_member(tmp_path: Path) -> None:
    """Anti-vacuity: admit every ``~/.codex`` SQLite file and the 627 MB log
    database enters acquisition."""
    root = tmp_path / "codex"
    root.mkdir()
    cursor = CursorStore(tmp_path / "cursor.db")
    watcher = LiveWatcher(
        cast(
            Any,
            SimpleNamespace(
                archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "cursor.db"), config=None
            ),
        ),
        (WatchSource(name="codex-state", root=root, suffixes=(".sqlite", ".db")),),
        cursor=cursor,
        debounce_s=0.0,
    )

    assert watcher._is_declared_codex_database(root / "state_5.sqlite") is True
    assert watcher._is_declared_codex_database(root / "goals_1.sqlite") is True
    assert watcher._is_declared_codex_database(root / "logs_2.sqlite") is False
    assert watcher._is_declared_codex_database(root / "codex-dev.db") is False
    assert watcher._is_declared_codex_database(root / "history.jsonl") is False


# ---------------------------------------------------------------------------
# Scale: the residue cohort was 551.87 MB across five databases
# ---------------------------------------------------------------------------

# Larger than the 551.87 MB protected source-divergent residue this bead was
# filed against, so the acquisition route is exercised past the size that
# produced it. Held in few large rows: the byte volume is what the backup,
# the digest and the memory ceiling see, and the row count is not.
_COHORT_EXCEEDING_BYTES = 600 * 1024 * 1024
_SCALE_ROW_BYTES = 4 * 1024 * 1024


@pytest.mark.slow
@pytest.mark.storage_scale
def test_a_database_larger_than_the_residue_cohort_acquires_in_bounded_memory(tmp_path: Path) -> None:
    """Anti-vacuity: materialize the rows (``fetchall``, or digesting the whole
    file into one buffer) and this exceeds the ceiling below; key identity on
    the page image and the checkpointed copy stops matching the source.
    """
    import resource

    source = tmp_path / "state.db"
    row_count = _COHORT_EXCEEDING_BYTES // _SCALE_ROW_BYTES
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE pages (id INTEGER PRIMARY KEY, body BLOB NOT NULL)")
        for index in range(row_count):
            conn.execute(
                "INSERT INTO pages (id, body) VALUES (?, ?)",
                (index, os.urandom(64) + bytes(_SCALE_ROW_BYTES - 64)),
            )
    assert source.stat().st_size > 551_870_000, source.stat().st_size

    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    revision = sqlite_logical_revision(source)

    copy = tmp_path / "copy.db"
    sqlite_snapshot.snapshot_sqlite_database(source, copy)
    assert sqlite_logical_revision(copy, immutable=True) == revision

    with closing(sqlite3.connect(source)) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert sqlite_logical_revision(source) == revision

    growth_bytes = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before_kb) * 1024
    # One row plus its hex and JSON encodings, with headroom; never the file.
    assert growth_bytes < 16 * _SCALE_ROW_BYTES, growth_bytes
