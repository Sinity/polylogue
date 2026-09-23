"""The cold build's durability policy, proved on the handles it actually uses.

``source`` is durable and irreplaceable; ``ops`` is disposable. The cold build
is allowed to spend ``ops`` durability and is not allowed to spend ``source``
durability, and these tests pin both halves of that -- on the real writer
handles, not on a reopened inspection connection, which would report its own
connection-local pragmas rather than the writer's (polylogue-rk0it AC2).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    active_index_generation_is_empty,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.connection_profile import (
    WRITE_CONNECTION_PROFILE,
    write_connection_pragma_statements,
)

_SYNCHRONOUS_LEVELS = {"OFF": 0, "NORMAL": 1, "FULL": 2, "EXTRA": 3}


@pytest.fixture
def cold_build(tmp_path: Path) -> Iterator[ColdBuildGeneration]:
    assert active_index_generation_is_empty(tmp_path) is True
    generation = ColdBuildGeneration.begin(tmp_path, reason="durability policy")
    register_cold_build_generation(generation)
    try:
        yield generation
    finally:
        clear_cold_build_generation()


def _expected_synchronous() -> int:
    statement = next(
        statement
        for statement in write_connection_pragma_statements(WRITE_CONNECTION_PROFILE)
        if "synchronous" in statement
    )
    return _SYNCHRONOUS_LEVELS[statement.rsplit("=", maxsplit=1)[1].strip().upper()]


def _ops_write(archive_root: Path, name: str) -> None:
    """One ordinary one-shot ``ops.db`` write through the production route."""
    CursorStore(archive_root / "index.db").set(archive_root / name, 1)


def _ops_wal(archive_root: Path) -> Path:
    return archive_root / "ops.db-wal"


def test_cold_build_source_writer_policy(tmp_path: Path, cold_build: ColdBuildGeneration) -> None:
    """The durable source tier keeps its declared policy inside a cold build.

    The cold build takes ``journal_mode=MEMORY``/``synchronous=OFF`` on the
    *candidate index* because a corrupt candidate is discarded. It buys none
    of that on ``source.db``, which is the one tier a rebuild cannot
    reconstruct, so the handle the pass acquires through must still report the
    declared writer policy.

    Anti-vacuity: replacing ``open_source_tier_write_connection`` in
    ``ArchiveStore._ensure_source_conn`` with a bare ``sqlite3.connect`` leaves
    this handle at SQLite's defaults -- ``synchronous=FULL``, a 5 s busy
    timeout and foreign keys off -- and every field below is wrong.
    """
    with cold_build.open_writer() as archive:
        source = archive._ensure_source_conn()
        assert source is archive._ensure_source_conn()
        observed = (
            str(source.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
            int(source.execute("PRAGMA synchronous").fetchone()[0]),
            int(source.execute("PRAGMA busy_timeout").fetchone()[0]),
            int(source.execute("PRAGMA foreign_keys").fetchone()[0]),
        )
    assert observed == ("wal", _expected_synchronous(), WRITE_CONNECTION_PROFILE.busy_timeout_ms, 1)


def test_cold_build_retains_the_ops_wal(tmp_path: Path, cold_build: ColdBuildGeneration) -> None:
    """A one-shot ops writer inside a cold build does not checkpoint on close.

    That deferral *is* the cold-build ops policy. Without it every cursor,
    convergence-debt and stage-event write is the last connection to
    ``ops.db``, so its close runs a checkpoint: fsync the WAL, copy it into
    ``ops.db``, fsync that, delete the WAL, fsync the directory. Measured at
    246 of 284 ``fdatasync`` calls in a 16-file cold-build page.

    The commit itself is unchanged and still durable against a process crash:
    it is in the WAL, which the second assertion reads back through a fresh
    connection.

    Anti-vacuity: delete the ``_retain_ops_checkpoints()`` call from
    ``ColdBuildGeneration.open_writer`` and the close-time checkpoint fires
    again -- the WAL is drained and removed, and the first assertion is red.
    """
    cold_build.open_writer().close()
    _ops_write(tmp_path, "one.jsonl")

    wal = _ops_wal(tmp_path)
    assert wal.exists() and wal.stat().st_size > 0

    replayed = sqlite3.connect(f"file:{tmp_path / 'ops.db'}?mode=ro", uri=True)
    try:
        assert replayed.execute("SELECT COUNT(*) FROM ingest_cursor").fetchone()[0] == 1
    finally:
        replayed.close()


def test_settling_the_build_drains_the_wal(tmp_path: Path, cold_build: ColdBuildGeneration) -> None:
    """The deferral ends with the build; a settled archive keeps the live shape.

    The widened window is scoped to the build. Once the generation is promoted
    or discarded the holder is released, so the next one-shot ops writer is the
    last connection again and checkpoints on close exactly as it did before.

    Anti-vacuity: drop the ``_release_ops_checkpoint_holder()`` call from
    ``ColdBuildGeneration.discard`` and the WAL survives the settled build --
    the final assertion is red.
    """
    cold_build.open_writer().close()
    _ops_write(tmp_path, "one.jsonl")
    assert _ops_wal(tmp_path).exists()

    assert cold_build.discard() is True

    _ops_write(tmp_path, "two.jsonl")
    assert not _ops_wal(tmp_path).exists()


def test_the_holder_blocks_no_checkpoint(tmp_path: Path, cold_build: ColdBuildGeneration) -> None:
    """The holder defers checkpoints; it must never *prevent* one.

    Deferring ops' flush onto the daemon's recurring checkpoint is only honest
    if that checkpoint can still run. A holder that left a read transaction
    open would pin WAL frames and make even a TRUNCATE checkpoint report busy,
    which would turn a bounded window into an unbounded one.

    Anti-vacuity: open a transaction in ``_ops_holder_is_attached`` (``BEGIN``
    before the read, no commit) and ``wal_checkpoint(TRUNCATE)`` answers
    ``busy=1`` with frames left behind.
    """
    cold_build.open_writer().close()
    _ops_write(tmp_path, "one.jsonl")
    assert _ops_wal(tmp_path).exists()

    checkpointer = sqlite3.connect(tmp_path / "ops.db")
    try:
        busy, _log_frames, _checkpointed = checkpointer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        checkpointer.close()
    assert busy == 0
    assert _ops_wal(tmp_path).stat().st_size == 0


def test_failed_candidate_open_releases_the_ops_holder(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed pass open must not retain the build's checkpoint handle.

    The holder is acquired before the candidate open, so an open failure is a
    real cleanup boundary rather than an ordinary ``with ArchiveStore`` exit.
    Anti-vacuity: removing the exception cleanup in ``open_writer`` leaves the
    holder populated after this deliberate open failure.
    """
    from polylogue.sources.live import cold_build as cold_build_module

    def fail_open(*args: object, **kwargs: object) -> object:
        raise sqlite3.OperationalError("candidate open failed")

    monkeypatch.setattr(cold_build_module.ArchiveStore, "open_cold_build_generation", classmethod(fail_open))
    with pytest.raises(sqlite3.OperationalError, match="candidate open failed"):
        cold_build.open_writer()
    assert cold_build._ops_checkpoint_holder is None
    assert not _ops_wal(tmp_path).exists()
