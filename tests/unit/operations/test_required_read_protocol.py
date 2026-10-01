"""Internal read boundaries require real snapshots, guards and resource settlement."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.query.execution_control import (
    InterruptibleSQLiteRead,
    QueryExecutionContext,
    _close_store,
)
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root

pytestmark = pytest.mark.uses_real_clock("exercises read admission and cancellation receipts")


def test_operation_read_cannot_succeed_without_its_snapshot_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing the pin must fail, not report an authoritative empty version map."""
    bootstrap_archive_root(tmp_path)
    closed: list[ArchiveStore] = []
    original_close = ArchiveStore.close

    def close(store: ArchiveStore) -> None:
        original_close(store)
        closed.append(store)

    monkeypatch.setattr(ArchiveStore, "pin_operation_snapshot", None)
    monkeypatch.setattr(ArchiveStore, "close", close)
    with pytest.raises(TypeError):
        with open_operation_read(tmp_path):
            pytest.fail("an unpinned store reached the consumer")
    assert len(closed) == 1


def test_operation_read_closes_acquired_handles_when_pin_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure after acquiring the source sibling still ends and closes the read."""
    bootstrap_archive_root(tmp_path)
    failure = RuntimeError("synthetic pin failure")
    original_pin = ArchiveStore.pin_operation_snapshot
    original_end = ArchiveStore.end_read_snapshot
    original_close = ArchiveStore.close
    events: list[str] = []
    connections: list[sqlite3.Connection] = []

    def pin(store: ArchiveStore) -> tuple[dict[str, int], tuple[str, ...]]:
        original_pin(store)
        assert store.index_connection is not None
        connections.extend([store.index_connection, store.source_connection])
        raise failure

    def end(store: ArchiveStore) -> None:
        events.append("end")
        original_end(store)

    def close(store: ArchiveStore) -> None:
        events.append("close")
        original_close(store)

    monkeypatch.setattr(ArchiveStore, "pin_operation_snapshot", pin)
    monkeypatch.setattr(ArchiveStore, "end_read_snapshot", end)
    monkeypatch.setattr(ArchiveStore, "close", close)
    context = QueryExecutionContext.create(query_text="synthetic pin failure", timeout_s=None)
    with pytest.raises(RuntimeError) as raised:
        with open_operation_read(tmp_path, execution_context=context):
            pytest.fail("a failed pin reached the consumer")
    assert raised.value is failure
    assert events.index("end") < events.index("close")
    assert context.receipt.cleanup_complete
    assert len(connections) == 2
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


@pytest.mark.parametrize("failed_step", ["clear_read_progress_guard", "end_read_snapshot"])
def test_store_close_still_runs_after_a_prior_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_step: str
) -> None:
    """Reverting to sequential cleanup leaks the connection after either failure."""
    bootstrap_archive_root(tmp_path)
    store = ArchiveStore.open_existing(tmp_path, read_only=True)
    connection = store.index_connection
    assert connection is not None
    store.pin_operation_snapshot()
    failure = RuntimeError(f"synthetic {failed_step} failure")

    def fail() -> None:
        raise failure

    monkeypatch.setattr(store, failed_step, fail)
    with pytest.raises(RuntimeError) as raised:
        _close_store(store)
    assert raised.value is failure
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_synchronous_read_forwards_its_declared_open_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bootstrap_archive_root(tmp_path)
    original_factory = ArchiveStore.open_existing
    received: list[tuple[float, bool]] = []

    def open_existing(
        cls: type[ArchiveStore], archive_root: Path, *, read_timeout: float, read_only: bool
    ) -> ArchiveStore:
        received.append((read_timeout, read_only))
        return original_factory(archive_root, read_timeout=read_timeout, read_only=read_only)

    monkeypatch.setattr(ArchiveStore, "open_existing", classmethod(open_existing))
    context = QueryExecutionContext.create(query_text="synthetic open options", timeout_s=None)
    with InterruptibleSQLiteRead(context).open_context(tmp_path, read_timeout=0.25) as store:
        assert store.index_connection is not None
        assert store.index_connection.in_transaction
    assert received == [(0.25, True)]
    assert context.receipt.cleanup_complete
