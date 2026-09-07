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
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.storage.sqlite.connection_profile import (
    READ_PROFILES,
    SEALED_READ_CONNECTION_PROFILE,
    TIMEOUT_CLASSES,
    WRITE_PROFILES,
    ReadContinuation,
    ReadFrame,
    ReadFrameCancelledError,
    ReadFrameExpiredError,
    StaleContinuationError,
    open_profiled_connection,
    open_readonly_connection,
    read_frame,
)


@pytest.fixture
def index_db(tmp_path: Path) -> Path:
    db = tmp_path / "index.db"
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
        assert not frame.revalidate()
        after = frame.rebind()
        assert after != before
        assert frame.revalidate()
        assert frame.connection.execute("SELECT count(*) FROM rows_").fetchone()[0] == 11
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


# -- production-route read-only enforcement ----------------------------------


def _migrated_readers(index_db: Path) -> Iterator[tuple[str, Callable[[], sqlite3.Connection]]]:
    """One opener per production route this bead migrated or already owned."""
    yield "api/cli interactive", lambda: open_readonly_connection(index_db, validate_schema=False)
    yield (
        "operations/debt background",
        lambda: open_readonly_connection(index_db, validate_schema=False, timeout_class="background-read"),
    )
    yield (
        "security/excision profiled",
        lambda: open_profiled_connection(index_db, profile=READ_PROFILES["background-read"]),
    )
    yield (
        "daemon/backup sealed",
        lambda: open_readonly_connection(index_db, validate_schema=False, immutable=True),
    )
    yield "cli streaming read frame", lambda: read_frame(index_db, timeout_class="background-read").connection


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
        conn = opener()
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly|read.only"):
                conn.execute(statement)
            assert conn.execute("SELECT count(*) FROM rows_").fetchone()[0] == 10, label
        finally:
            conn.close()


def test_migrated_readers_cannot_write_through_an_attached_database(index_db: Path, tmp_path: Path) -> None:
    sibling = tmp_path / "sibling.db"
    conn = sqlite3.connect(sibling)
    try:
        conn.execute("CREATE TABLE target (x TEXT)")
        conn.commit()
    finally:
        conn.close()

    for label, opener in _migrated_readers(index_db):
        reader = opener()
        try:
            reader.execute("ATTACH DATABASE ? AS sibling", (str(sibling),))
            with pytest.raises(sqlite3.OperationalError, match="readonly|read.only"):
                reader.execute("INSERT INTO sibling.target VALUES ('injected')")
        except sqlite3.OperationalError as exc:  # an ATTACH the profile refuses outright is also correct
            assert "readonly" in str(exc) or "read-only" in str(exc), label
        finally:
            reader.close()

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
