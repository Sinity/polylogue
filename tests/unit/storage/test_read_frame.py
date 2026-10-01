"""Read-frame lifetime, rebinding, continuation and read-only enforcement.

Anti-vacuity: the mutants these tests catch are a frame that keeps serving its
connection past the declared maximum snapshot age, a rebind that does not move
the generation identity, a continuation that resumes across a moved generation
whose anchor no longer holds, and any nominally read-only production route that
can still reach the write path.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from polylogue.storage.sqlite import wal_checkpoint
from polylogue.storage.sqlite.connection_profile import (
    READ_PROFILES,
    SEALED_READ_CONNECTION_PROFILE,
    TIMEOUT_CLASSES,
    WRITE_PROFILES,
    LiveGenerationImmutableError,
    ReadContinuation,
    ReadFrame,
    ReadFrameCancelledError,
    ReadFrameExpiredError,
    StaleContinuationError,
    attach_readonly_database,
    one_shot_diagnostic_read,
    open_profiled_connection,
    open_readonly_connection,
    read_frame,
)
from tests.infra.sqlite_cursor_settlement import (
    native_settlement_connections,  # noqa: F401  # Pytest fixture discovery.
)


@pytest.fixture
def index_db(tmp_path: Path) -> Path:
    db = tmp_path / "index.db"
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(db, ArchiveTier.INDEX)
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE rows_ (position INTEGER PRIMARY KEY, body TEXT NOT NULL)")
        conn.executemany("INSERT INTO rows_ VALUES (?, ?)", [(n, f"row-{n}") for n in range(1, 11)])
        conn.commit()
    finally:
        conn.close()
    return db


def _commit(db: Path, sql: str, params: tuple[object, ...] = ()) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


# -- declared profile completeness -------------------------------------------


def test_every_read_profile_declares_its_frame_policy() -> None:
    for name, profile in READ_PROFILES.items():
        assert profile.role == "read", name
        assert profile.query_only, name
        assert profile.cancellation_supported, name
        assert profile.generation_identity == "live", name
        # A live generation can change under the reader, so it must declare the
        # age past which the frame is rebound rather than pinning WAL forever.
        assert profile.max_snapshot_age_s is not None, name
        assert not profile.immutable, name
        assert profile.busy_timeout_ms == int(TIMEOUT_CLASSES[name] * 1000), name


def test_sealed_profile_is_the_only_immutable_one() -> None:
    assert SEALED_READ_CONNECTION_PROFILE.generation_identity == "sealed"
    assert SEALED_READ_CONNECTION_PROFILE.immutable
    assert SEALED_READ_CONNECTION_PROFILE.max_snapshot_age_s is None
    assert not any(profile.immutable for profile in READ_PROFILES.values())


def test_named_write_classes_do_not_shadow_the_read_ones() -> None:
    assert set(WRITE_PROFILES) <= set(TIMEOUT_CLASSES)
    assert all(profile.role == "write" for profile in WRITE_PROFILES.values())
    assert READ_PROFILES["offline-bulk"] is not WRITE_PROFILES["offline-bulk"]


def test_immutable_is_reserved_for_a_sealed_generation(index_db: Path) -> None:
    live_profile = READ_PROFILES["background-read"]
    with pytest.raises(ValueError, match="sealed-generation"):
        open_readonly_connection(
            index_db,
            immutable=True,
            validate_schema=False,
            profile=replace(live_profile, cache_size_kib=1),
        )


def test_asking_for_immutability_selects_the_sealed_profile(index_db: Path) -> None:
    conn = open_readonly_connection(index_db, immutable=True, validate_schema=False, timeout_class="offline-bulk")
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone() == (SEALED_READ_CONNECTION_PROFILE.busy_timeout_ms,)
    finally:
        conn.close()


# -- live versus sealed generation -------------------------------------------


@pytest.fixture
def live_writer(index_db: Path) -> Iterator[sqlite3.Connection]:
    """A writer that stays open with a committed row living only in the WAL."""
    writer = sqlite3.connect(index_db)
    writer.execute("PRAGMA wal_autocheckpoint = 0")
    writer.execute("INSERT INTO rows_ VALUES (11, 'wal-only')")
    writer.commit()
    assert index_db.with_name("index.db-wal").stat().st_size > 0, "the row must live only in the WAL"
    try:
        yield writer
    finally:
        writer.close()


def test_a_live_read_sees_a_committed_wal_only_row(index_db: Path, live_writer: sqlite3.Connection) -> None:
    conn = open_readonly_connection(index_db, validate_schema=False)
    try:
        assert conn.execute("SELECT body FROM rows_ WHERE position = 11").fetchone() == ("wal-only",)
    finally:
        conn.close()


def test_marking_a_changing_database_immutable_is_refused(
    index_db: Path, tmp_path: Path, live_writer: sqlite3.Connection
) -> None:
    """``immutable=1`` would read the main file alone and skip the WAL-only row.

    Anti-vacuity: without the owner's sidecar refusal each of these opens
    succeeds and reports ten rows while the database holds eleven.
    """
    with pytest.raises(LiveGenerationImmutableError, match="-wal"):
        open_readonly_connection(index_db, immutable=True, validate_schema=False)
    with pytest.raises(LiveGenerationImmutableError):
        ReadFrame(index_db, profile=SEALED_READ_CONNECTION_PROFILE)

    host = tmp_path / "host.db"
    sqlite3.connect(host).close()
    reader = open_readonly_connection(host, validate_schema=False)
    try:
        with pytest.raises(LiveGenerationImmutableError):
            attach_readonly_database(reader, index_db, alias="changing", immutable=True)
        assert reader.execute("PRAGMA database_list").fetchall()[-1][1] != "changing"
    finally:
        reader.close()


def test_a_frozen_snapshot_carries_the_wal_state_and_its_generation(
    index_db: Path, live_writer: sqlite3.Connection
) -> None:
    """Freezing is an exclusive checkpoint into the main file, then a sealed read.

    The writer stays open but idle, so its last-connection close cannot be what
    folds the WAL back: the exclusive checkpoint has to.
    """
    observation = wal_checkpoint.checkpoint_wal(
        index_db, reason="seal", escalation="exclusive", warn_bytes=1, escalation_bytes=1
    )
    assert observation.mode == "truncate"
    assert observation.wal_bytes_after == 0

    stat = index_db.stat()
    frame = ReadFrame(index_db, profile=SEALED_READ_CONNECTION_PROFILE)
    try:
        assert (frame.generation.device, frame.generation.inode) == (stat.st_dev, stat.st_ino)
        assert frame.connection.execute("SELECT body FROM rows_ WHERE position = 11").fetchone()[0] == "wal-only"
    finally:
        frame.close()


# -- frame lifetime ----------------------------------------------------------


def test_frame_serves_its_connection_until_the_declared_age(index_db: Path) -> None:
    with read_frame(index_db, timeout_class="interactive-read") as frame:
        assert frame.connection.execute("SELECT count(*) FROM rows_").fetchone()[0] == 10
        assert not frame.expired


def test_expired_frame_refuses_its_connection(index_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = read_frame(index_db, timeout_class="interactive-read")
    try:
        monkeypatch.setattr(type(frame), "age_s", property(lambda _self: 1_000.0))
        assert frame.expired
        with pytest.raises(ReadFrameExpiredError, match="maximum"):
            _ = frame.connection
    finally:
        frame.close()


def test_rebinding_releases_the_frame_and_sees_later_commits(index_db: Path) -> None:
    frame = read_frame(index_db, timeout_class="interactive-read")
    try:
        before = frame.generation
        assert frame.connection.execute("SELECT count(*) FROM rows_").fetchone()[0] == 10
        _commit(index_db, "INSERT INTO rows_ VALUES (11, 'row-11')")
        # The commit landed in the same generation, so the file identity is
        # unchanged; what moved is the content this frame was opened on.
        assert not frame.revalidate()
        assert frame.rebind() == before
        assert frame.epoch == 1
        assert frame.revalidate()
        assert frame.connection.execute("SELECT count(*) FROM rows_").fetchone()[0] == 11
    finally:
        frame.close()


def test_generation_identity_moves_when_the_pointer_is_swapped(index_db: Path, tmp_path: Path) -> None:
    replacement = tmp_path / "replacement.db"
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(replacement, ArchiveTier.INDEX)
    conn = sqlite3.connect(replacement)
    try:
        conn.execute("CREATE TABLE rows_ (position INTEGER PRIMARY KEY, body TEXT NOT NULL)")
        conn.commit()
    finally:
        conn.close()

    frame = read_frame(index_db, timeout_class="interactive-read")
    try:
        before = frame.generation
        replacement.replace(index_db)
        assert not frame.revalidate()
        assert frame.rebind() != before
    finally:
        frame.close()


def test_sealed_frame_has_nothing_to_rebind_to(index_db: Path) -> None:
    frame = ReadFrame(index_db, profile=SEALED_READ_CONNECTION_PROFILE)
    try:
        assert not frame.expired
        with pytest.raises(ValueError, match="nothing to rebind"):
            frame.rebind()
    finally:
        frame.close()


def test_cancelled_frame_refuses_further_reads(index_db: Path) -> None:
    frame = read_frame(index_db, timeout_class="interactive-read")
    try:
        frame.cancel()
        with pytest.raises(ReadFrameCancelledError):
            _ = frame.connection
        frame.rebind()
        assert frame.connection.execute("SELECT count(*) FROM rows_").fetchone()[0] == 10
    finally:
        frame.close()


def test_frame_refuses_cancellation_a_profile_does_not_declare(index_db: Path) -> None:
    profile = replace(READ_PROFILES["background-read"], cancellation_supported=False)
    frame = ReadFrame(index_db, profile=profile)
    try:
        with pytest.raises(ValueError, match="cancellation"):
            frame.cancel()
    finally:
        frame.close()


def test_frame_requires_a_query_only_read_profile(index_db: Path) -> None:
    with pytest.raises(ValueError, match="query-only read profile"):
        ReadFrame(index_db, profile=WRITE_PROFILES["publication"])


def test_frame_over_a_missing_tier_preserves_the_caller_error(tmp_path: Path) -> None:
    with pytest.raises(sqlite3.OperationalError):
        read_frame(tmp_path / "absent.db", timeout_class="interactive-read")


def test_frame_preserves_missing_table_semantics(index_db: Path) -> None:
    with read_frame(index_db, timeout_class="interactive-read") as frame, pytest.raises(sqlite3.OperationalError):
        frame.connection.execute("SELECT 1 FROM never_created")


# -- continuations -----------------------------------------------------------


_ANCHOR = "SELECT position FROM rows_ WHERE position = ?"


def test_unmoved_generation_resumes_unchanged(index_db: Path) -> None:
    with read_frame(index_db, timeout_class="interactive-read") as frame:
        continuation = frame.bind(ReadContinuation(position=5, anchor_sql=_ANCHOR, anchor_params=(5,)))
        assert frame.resume(continuation) == continuation


def test_rebound_frame_reproves_the_anchor_rather_than_trusting_identity(index_db: Path) -> None:
    """A rebind starts a new incarnation, so the fast path must not fire.

    ``PRAGMA data_version`` is meaningless across connections and the file
    identity is unchanged by an ordinary commit, so without the epoch a
    continuation would be waved through against content it never saw.
    """
    with read_frame(index_db, timeout_class="interactive-read") as frame:
        continuation = frame.bind(ReadContinuation(position=5, anchor_sql=_ANCHOR, anchor_params=(5,)))
        _commit(index_db, "DELETE FROM rows_ WHERE position = 5")
        frame.rebind()
        assert frame.generation == continuation.generation
        assert frame.epoch != continuation.epoch
        with pytest.raises(StaleContinuationError):
            frame.resume(continuation)


def test_moved_generation_resumes_when_the_anchor_still_holds(index_db: Path) -> None:
    with read_frame(index_db, timeout_class="interactive-read") as frame:
        continuation = frame.bind(ReadContinuation(position=5, anchor_sql=_ANCHOR, anchor_params=(5,)))
        _commit(index_db, "INSERT INTO rows_ VALUES (12, 'row-12')")
        frame.rebind()
        resumed = frame.resume(continuation)
        assert resumed.position == 5
        assert resumed.generation == frame.generation


def test_moved_generation_refuses_when_the_anchor_is_gone(index_db: Path) -> None:
    with read_frame(index_db, timeout_class="interactive-read") as frame:
        continuation = frame.bind(ReadContinuation(position=5, anchor_sql=_ANCHOR, anchor_params=(5,)))
        _commit(index_db, "DELETE FROM rows_ WHERE position = 5")
        frame.rebind()
        with pytest.raises(StaleContinuationError, match="anchor row no longer holds"):
            frame.resume(continuation)


def test_expired_frame_rebinds_before_resuming(index_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = read_frame(index_db, timeout_class="interactive-read")
    try:
        continuation = frame.bind(ReadContinuation(position=5, anchor_sql=_ANCHOR, anchor_params=(5,)))
        ages = iter([1_000.0])
        monkeypatch.setattr(type(frame), "age_s", property(lambda _self: next(ages, 0.0)))
        resumed = frame.resume(continuation)
        assert resumed.position == 5
        assert not frame.expired
    finally:
        frame.close()


def test_one_shot_diagnostic_read_is_readonly_and_releases_its_connection(index_db: Path) -> None:
    """A diagnostic probe cannot become a hidden long-lived or writable reader."""
    with one_shot_diagnostic_read(index_db) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized|readonly|read.only"):
            conn.execute("INSERT INTO rows_ VALUES (99, 'injected')")
        assert conn.execute("SELECT count(*) FROM rows_").fetchone()[0] == 10

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")


# -- production-route read-only enforcement ----------------------------------


def _migrated_readers(index_db: Path) -> Iterator[tuple[str, Callable[[], AbstractContextManager[sqlite3.Connection]]]]:
    """One opener per production route this bead migrated or already owned."""
    yield "api/cli interactive", lambda: closing(open_readonly_connection(index_db, validate_schema=False))
    yield (
        "operations/debt background",
        lambda: closing(open_readonly_connection(index_db, validate_schema=False, timeout_class="background-read")),
    )
    yield (
        "security/excision profiled",
        lambda: closing(open_profiled_connection(index_db, profile=READ_PROFILES["background-read"])),
    )
    yield (
        "daemon/backup sealed",
        lambda: closing(open_readonly_connection(index_db, validate_schema=False, immutable=True)),
    )

    @contextmanager
    def frame_connection() -> Iterator[sqlite3.Connection]:
        with read_frame(index_db, timeout_class="background-read") as frame:
            yield frame.connection

    yield "cli streaming read frame", frame_connection


_WRITE_ATTEMPTS = (
    ("insert", "INSERT INTO rows_ VALUES (99, 'injected')"),
    ("update", "UPDATE rows_ SET body = 'injected'"),
    ("delete", "DELETE FROM rows_"),
    ("schema", "CREATE TABLE injected (x TEXT)"),
    ("writable pragma", "PRAGMA user_version = 4242"),
)


@pytest.mark.parametrize(
    "statement", [sql for _label, sql in _WRITE_ATTEMPTS], ids=[label for label, _ in _WRITE_ATTEMPTS]
)
def test_migrated_readers_refuse_writes_at_the_database_boundary(index_db: Path, statement: str) -> None:
    for label, opener in _migrated_readers(index_db):
        with opener() as conn:
            with pytest.raises(sqlite3.DatabaseError, match="not authorized|readonly|read.only"):
                conn.execute(statement)
            assert conn.execute("SELECT count(*) FROM rows_").fetchone()[0] == 10, label


def test_migrated_readers_cannot_write_through_an_attached_database(index_db: Path, tmp_path: Path) -> None:
    sibling = tmp_path / "sibling.db"
    conn = sqlite3.connect(sibling)
    try:
        conn.execute("CREATE TABLE target (x TEXT)")
        conn.commit()
    finally:
        conn.close()

    for label, opener in _migrated_readers(index_db):
        with opener() as reader:
            try:
                reader.execute("ATTACH DATABASE ? AS sibling", (str(sibling),))
                with pytest.raises(sqlite3.DatabaseError, match="not authorized|readonly|read.only"):
                    reader.execute("INSERT INTO sibling.target VALUES ('injected')")
            except sqlite3.DatabaseError as exc:  # a refused ATTACH is also correct
                assert any(reason in str(exc) for reason in ("not authorized", "readonly", "read-only")), label

    survivor = sqlite3.connect(sibling)
    try:
        assert survivor.execute("SELECT count(*) FROM target").fetchone()[0] == 0
    finally:
        survivor.close()


def test_migrated_readers_preserve_their_declared_timeout(index_db: Path) -> None:
    for name, profile in READ_PROFILES.items():
        conn = open_readonly_connection(index_db, validate_schema=False, timeout_class=name)
        try:
            assert conn.execute("PRAGMA busy_timeout").fetchone() == (profile.busy_timeout_ms,), name
            assert conn.execute("PRAGMA query_only").fetchone() == (1,), name
        finally:
            conn.close()


def test_a_reader_does_not_block_writer_progress(index_db: Path) -> None:
    """A bounded frame must not be the reason a writer cannot commit."""
    with read_frame(index_db, timeout_class="background-read") as frame:
        frame.connection.execute("SELECT count(*) FROM rows_").fetchone()
        _commit(index_db, "INSERT INTO rows_ VALUES (13, 'row-13')")
        frame.rebind()
        assert frame.connection.execute("SELECT count(*) FROM rows_").fetchone()[0] == 11


def test_resume_releases_a_held_snapshot_before_proving_the_anchor(index_db: Path) -> None:
    """Old snapshot proof finds row 5 even after a concurrent delete."""
    with read_frame(index_db) as frame:
        frame.connection.execute("BEGIN DEFERRED")
        assert frame.connection.execute(_ANCHOR, (5,)).fetchone()[0] == 5
        continuation = frame.bind(ReadContinuation(position=5, anchor_sql=_ANCHOR, anchor_params=(5,)))
        _commit(index_db, "DELETE FROM rows_ WHERE position = 5")
        with pytest.raises(StaleContinuationError):
            frame.resume(continuation)


def test_resume_refuses_while_a_stream_pins_the_snapshot(index_db: Path) -> None:
    """Anti-vacuity: proving the anchor beside an in-flight stream reads the
    stream's snapshot, so a concurrently deleted anchor row was accepted."""
    with read_frame(index_db) as frame:
        continuation = frame.bind(ReadContinuation(position=5, anchor_sql=_ANCHOR, anchor_params=(5,)))
        rows = frame.stream("SELECT * FROM rows_ ORDER BY position")
        try:
            assert next(rows)[0] == 1
            _commit(index_db, "DELETE FROM rows_ WHERE position = 5")
            with pytest.raises(ReadFrameExpiredError, match="stream is in flight"):
                frame.resume(continuation)
        finally:
            rows.close()


def test_stream_can_be_closed_after_its_frame(index_db: Path) -> None:
    """Pre-fix generator finalization closes a cursor on an already closed DB."""
    with read_frame(index_db) as frame:
        rows = frame.stream("SELECT * FROM rows_ ORDER BY position")
        assert next(rows)[0] == 1
    rows.close()
    assert not frame.streaming


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), -float("inf")])
def test_read_frame_rejects_nonfinite_snapshot_bounds(index_db: Path, bound: float) -> None:
    """NaN/inf bypass a <= 0 comparison and disable the declared lifetime."""
    with pytest.raises(ValueError):
        with read_frame(index_db, max_snapshot_age_s=bound, reason="nonfinite regression"):
            pass
    with pytest.raises(ValueError):
        with ReadFrame(index_db, profile=replace(READ_PROFILES["interactive-read"], max_snapshot_age_s=bound)):
            pass


@pytest.mark.uses_real_clock("actual read handle cleanup preserves physical exclusion")
@pytest.mark.parametrize("failure_point", ["initial", "rebind", "close"])
def test_read_frame_retains_actual_handle_until_all_cleanup_settles(
    index_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    from polylogue.storage.sqlite import connection_profile as profiles
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.archive_custody_probe import archive_custody_available
    from tests.infra.sqlite_cursor_settlement import SettlementConnection, arm_settlement

    original_open = profiles.open_readonly_connection
    original_version = profiles._data_version
    handles: list[SettlementConnection] = []
    frames: list[ReadFrame] = []
    fault = failure_point == "initial"
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    failed: ControlledCursor | None = None
    successful: ControlledCursor | None = None

    def controlled_open(*args: object, **kwargs: object) -> sqlite3.Connection:
        handle = arm_settlement(original_open(*args, **kwargs))  # type: ignore[arg-type]
        handles.append(handle)
        return handle

    def version(connection: sqlite3.Connection) -> int:
        if fault:
            raise ValueError("synthetic frame metadata failure")
        return original_version(connection)

    monkeypatch.setattr(profiles, "open_readonly_connection", controlled_open)
    monkeypatch.setattr(profiles, "_data_version", version)
    context = write_lease("test.frame_cleanup", archive_root=index_db.parent)
    context.__enter__()
    try:
        if failure_point == "initial":
            with pytest.raises(profiles.NativeConnectionSettlementError) as refused:
                read_frame(index_db)
        else:
            frame = read_frame(index_db)
            frames.append(frame)
            if failure_point == "rebind":
                handles[0].allow_cleanup.set()
                fault = True
                with pytest.raises(profiles.NativeConnectionSettlementError) as refused:
                    frame.rebind()
            else:
                from tests.infra.sqlite_cursor_settlement import ControlledCursor

                failed = frame._conn.cursor(factory=ControlledCursor)
                successful = frame._conn.cursor(factory=ControlledCursor)
                failed.execute("SELECT 1 UNION ALL SELECT 2")
                successful.execute("SELECT 1 UNION ALL SELECT 2")
                failed.allow_cleanup.clear()
                frame._cursors.update((failed, successful))
                with pytest.raises(profiles.NativeConnectionSettlementError) as refused:
                    frame.close()
                assert failed.close_attempts == successful.close_attempts == 1
                assert {id(cursor) for cursor in frame._cursors} == {id(failed)}
                failed.allow_cleanup.set()
        owner = refused.value.owner
        assert owner.frame is not None
        assert cast(object, owner.connection) is handles[-1]
        assert not archive_custody_available(index_db.parent)
        handles[-1].allow_cleanup.set()
        owner.close()
        assert owner.frame is None
        if failure_point == "close":
            assert failed is not None and successful is not None
            assert failed.close_attempts == 2
            assert successful.close_attempts == 1
        assert owner.connection is None
        with pytest.raises(sqlite3.ProgrammingError):
            handles[-1].execute("SELECT 1")
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        for frame in frames:
            frame.close()
        context.__exit__(None, None, None)
    assert archive_custody_available(index_db.parent)


def test_independent_frame_census_retains_discarded_failed_cleanup(
    index_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc

    from polylogue.storage.sqlite import connection_profile as profiles
    from tests.infra.sqlite_cursor_settlement import SettlementConnection, arm_settlement

    actual_open = profiles.open_readonly_connection
    handles: list[SettlementConnection] = []

    def controlled_open(*args: object, **kwargs: object) -> sqlite3.Connection:
        handle = arm_settlement(actual_open(*args, **kwargs))  # type: ignore[arg-type]
        handles.append(handle)
        return handle

    monkeypatch.setattr(profiles, "open_readonly_connection", controlled_open)
    try:
        frame = read_frame(index_db)
        identity = id(frame)
        with pytest.raises(profiles.NativeConnectionSettlementError):
            frame.close()
        del frame
        gc.collect()
        retained = next(frame for frame in profiles._LIVE_READ_FRAMES if id(frame) == identity)
        assert retained._sql_owner.custody is None
        with pytest.raises(profiles.NativeConnectionSettlementError):
            _ = retained.connection
        with pytest.raises(profiles.NativeConnectionSettlementError):
            retained.revalidate()
        assert cast(object, retained._sql_owner.connection) is handles[0]
        handles[0].allow_cleanup.set()
        retained.close()
        assert all(id(frame) != identity for frame in profiles._LIVE_READ_FRAMES)
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        for frame in tuple(profiles._LIVE_READ_FRAMES):
            if frame._path == index_db:
                frame.close()


@pytest.mark.uses_real_clock("fork child must refuse inherited SQL before any SQLite call")
def test_forked_frame_refuses_inherited_sql_and_resets_usable_census(index_db: Path) -> None:
    import json
    import os

    from polylogue.storage.sqlite import connection_profile as profiles

    if not hasattr(os, "fork"):
        pytest.skip("process fork is unavailable")
    with read_frame(index_db) as frame:
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                refused = 0
                for operation in (
                    lambda: frame.connection.execute("SELECT 1"),
                    frame.cancel,
                    frame.revalidate,
                    frame.close,
                ):
                    try:
                        operation()
                    except RuntimeError:
                        refused += 1
                assert not profiles.live_read_frames()
                assert profiles._FORK_ABANDONED_READ_FRAMES
                with read_frame(index_db) as fresh:
                    assert fresh.connection.execute("SELECT COUNT(*) FROM rows_").fetchone()[0] == 10
                assert not profiles.live_read_frames()
                os.write(write_fd, json.dumps({"refused": refused}).encode())
                os._exit(0)
            except BaseException:
                os._exit(1)
        os.close(write_fd)
        try:
            result = os.read(read_fd, 1024)
        finally:
            os.close(read_fd)
        waited, status = os.waitpid(pid, 0)
        assert waited == pid and os.waitstatus_to_exitcode(status) == 0
        assert json.loads(result) == {"refused": 4}
        assert frame.connection.execute("SELECT COUNT(*) FROM rows_").fetchone()[0] == 10


@pytest.mark.uses_real_clock("fork/async task custody uses actual owner identities")
@pytest.mark.parametrize("foreign_owner", ["task", "fork"])
@pytest.mark.parametrize("action", ["resume", "close"])
def test_started_frame_stream_refuses_foreign_step_and_cleanup(
    index_db: Path,
    foreign_owner: str,
    action: str,
) -> None:
    import asyncio
    import json
    import os

    if foreign_owner == "fork" and not hasattr(os, "fork"):
        pytest.skip("process fork is unavailable")

    async def run() -> None:
        with read_frame(index_db) as frame:
            actual = frame._conn
            counts = {"step": 0, "close": 0}

            class Cursor(sqlite3.Cursor):
                def __next__(self) -> sqlite3.Row:
                    counts["step"] += 1
                    return cast(sqlite3.Row, super().__next__())

                def close(self) -> None:
                    counts["close"] += 1
                    super().close()

            class Connection:
                def execute(self, sql: str, parameters: tuple[object, ...]) -> Cursor:
                    cursor = actual.cursor(factory=Cursor)
                    cursor.execute(sql, parameters)
                    return cursor

            frame._conn = Connection()  # type: ignore[assignment]
            stream = frame.stream("SELECT position FROM rows_ ORDER BY position")
            assert next(stream)[0] == 1
            assert counts == {"step": 1, "close": 0}

            def foreign_operation() -> None:
                with pytest.raises(RuntimeError):
                    next(stream) if action == "resume" else stream.close()
                assert counts == {"step": 1, "close": 0}
                assert frame._cursors

            if foreign_owner == "task":

                async def foreign_task() -> None:
                    foreign_operation()

                await asyncio.create_task(foreign_task())
            else:
                read_fd, write_fd = os.pipe()
                pid = os.fork()
                if pid == 0:
                    os.close(read_fd)
                    try:
                        foreign_operation()
                        os.write(write_fd, json.dumps(counts).encode())
                        os._exit(0)
                    except BaseException:
                        os._exit(1)
                os.close(write_fd)
                try:
                    result = os.read(read_fd, 1024)
                finally:
                    os.close(read_fd)
                waited, status = os.waitpid(pid, 0)
                assert waited == pid and os.waitstatus_to_exitcode(status) == 0
                assert json.loads(result) == {"step": 1, "close": 0}
                # The parent's copied iterator still belongs to this task.
                stream.close()
            frame.close()
            assert not frame._cursors
            assert not frame.streaming
            assert counts["close"] == 1
            frame.rebind()
            assert frame.connection.execute("SELECT COUNT(*) FROM rows_").fetchone()[0] == 10
            assert not frame.streaming

    asyncio.run(run())


@pytest.mark.parametrize("exhaust", [False, True])
def test_stream_finalization_retires_cursor_before_native_connection_close(index_db: Path, exhaust: bool) -> None:
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    with read_frame(index_db) as frame:
        actual = frame._conn
        cursors: list[ControlledCursor] = []

        class Connection:
            def execute(self, sql: str, parameters: tuple[object, ...]) -> ControlledCursor:
                cursor = actual.cursor(factory=ControlledCursor)
                cursors.append(cursor)
                cursor.execute(sql, parameters)
                return cursor

        frame._conn = Connection()  # type: ignore[assignment]
        stream = frame.stream("SELECT position FROM rows_ ORDER BY position")
        if exhaust:
            assert len(list(stream)) == 10
        else:
            assert next(stream)[0] == 1
            stream.close()
        assert not frame.streaming
        assert cursors[0].close_attempts == 1
        frame.close()
        stream.close()
        assert cursors[0].close_attempts == 1
