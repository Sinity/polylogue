"""Exact parser census receipt bytes stay on their original measurement owner."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.revision_authority import ParserCensusIdentityMeasurement, parser_census_identity_measurement
from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.compute_cancel import compute_cancel
from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError, NativeSQLCustodyOwner
from polylogue.storage.sqlite.managed_connection import sqlite_connection


@contextmanager
def _measurement(keys: list[str]) -> Iterator[ParserCensusIdentityMeasurement]:
    with parser_census_identity_measurement(
        raw_logical_key=None,
        revision_kind="unknown",
        membership_logical_keys=keys,
        observed_logical_keys=keys,
        observed_are_receipt=True,
    ) as measured:
        yield measured


@pytest.mark.parametrize("sqlite_encoding", [False, True])
@pytest.mark.parametrize(
    "keys",
    [
        [],
        ["chatgpt-export:one"],
        [
            'chatgpt-export:雪😀"\\\n\x00',
            "chatgpt-export:duplicate",
            "chatgpt-export:duplicate",
            "chatgpt-export:" + "雪😀\\\n" * 20000,
        ],
    ],
)
def test_census_stream_preserves_exact_predecessor_spelling_and_bounded_chunks(
    keys: list[str], sqlite_encoding: bool
) -> None:
    if sqlite_encoding:
        with sqlite_connection(":memory:") as predecessor:
            predecessor.execute("CREATE TABLE original_keys(logical_key TEXT PRIMARY KEY)")
            predecessor.executemany("INSERT OR IGNORE INTO original_keys VALUES (?)", ((key,) for key in keys))
            expected = (
                predecessor.execute(
                    "SELECT json_group_array(logical_key) FROM (SELECT logical_key FROM original_keys ORDER BY logical_key)"
                )
                .fetchone()[0]
                .encode("utf-8")
            )
    else:
        expected = json.dumps(sorted(keys)).encode("ascii")
    with _measurement(keys) as measured:
        assert measured.owner.connection is measured.connection
        with measured.keys_json_stream(sqlite_encoding=sqlite_encoding) as (length, chunks):
            received = bytearray()
            for chunk in chunks:
                assert len(chunk) <= 65536
                received.extend(chunk)
            assert length == len(expected) == len(received)
            assert received == expected
        assert measured.owner.connection is measured.connection
        if sqlite_encoding:
            assert measured.owner._incremental_blobs == []


@pytest.mark.parametrize("sqlite_encoding", [False, True])
def test_census_stream_early_exit_closes_actual_reader_before_measurement(sqlite_encoding: bool) -> None:
    with _measurement(["chatgpt-export:" + "x" * 100000]) as measured:
        with measured.keys_json_stream(sqlite_encoding=sqlite_encoding) as (_length, chunks):
            owner = measured.owner
            assert next(chunks)
            while not owner._incremental_blobs:
                assert next(chunks)
            assert owner.connection is measured.connection
            blobs = tuple(owner._incremental_blobs)
        assert next(chunks, None) is None
        for blob in blobs:
            with pytest.raises(sqlite3.ProgrammingError):
                len(blob)
        assert owner.connection is measured.connection and not owner._settled
    assert owner._settled


@pytest.mark.parametrize("sqlite_encoding", [False, True])
@pytest.mark.parametrize("cancellation", ["callback", "compute"])
def test_census_stream_mid_delivery_preserves_original_cancellation(sqlite_encoding: bool, cancellation: str) -> None:
    event = threading.Event()
    refusal = DaemonOperationCancelled("synthetic original census stop")

    def check() -> None:
        if event.is_set() and cancellation == "callback":
            raise refusal

    token = compute_cancel.set(event if cancellation == "compute" else None)
    try:
        with _measurement(["chatgpt-export:" + "x" * 100000]) as measured:
            with pytest.raises(DaemonOperationCancelled) as stopped:
                with measured.keys_json_stream(sqlite_encoding=sqlite_encoding, check_stop=check) as (_length, chunks):
                    assert next(chunks)
                    while not measured.owner._incremental_blobs:
                        assert next(chunks)
                    event.set()
                    next(chunks)
            if cancellation == "callback":
                assert stopped.value is refusal
            owner = measured.owner
            assert owner._incremental_blobs == []
        assert owner._settled
    finally:
        compute_cancel.reset(token)


def test_census_sqlite_aggregate_preserves_callback_stop_before_native_blob_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusal = DaemonOperationCancelled("synthetic actual aggregate stop")
    calls = 0

    def check() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise refusal

    def forbidden_open(*_args: object, **_kwargs: object) -> None:
        pytest.fail("aggregate cancellation reached a native Blob open")

    with _measurement([f"chatgpt-export:{index:06}" for index in range(10000)]) as measured:
        monkeypatch.setattr(NativeSQLCustodyOwner, "readonly_blob", forbidden_open)
        with pytest.raises(DaemonOperationCancelled) as stopped:
            with measured.keys_json_stream(sqlite_encoding=True, check_stop=check):
                pytest.fail("aggregate cancellation yielded a stream")
        assert stopped.value is refusal and calls == 2


@pytest.mark.parametrize("sqlite_encoding", [False, True])
def test_census_stream_failed_actual_blob_close_retains_same_owner_and_scratch_until_retry(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_encoding: bool,
) -> None:
    original_close = NativeSQLCustodyOwner.close_incremental_blob
    blocked = False
    owner: NativeSQLCustodyOwner | None = None
    selected_blob: sqlite3.Blob | None = None
    artifact: Path | None = None

    def fail_close(selected: NativeSQLCustodyOwner, blob: sqlite3.Blob) -> None:
        if selected is owner and blocked:
            raise OSError("synthetic original census Blob close fault")
        original_close(selected, blob)

    monkeypatch.setattr(NativeSQLCustodyOwner, "close_incremental_blob", fail_close)
    try:
        with pytest.raises(NativeConnectionSettlementError):
            with _measurement(["chatgpt-export:retained"]) as measured:
                owner = measured.owner
                assert owner.scratch_directory is not None
                artifact = Path(owner.scratch_directory.name)
                with measured.keys_json_stream(sqlite_encoding=sqlite_encoding) as (_length, chunks):
                    assert next(chunks)
                    while not owner._incremental_blobs:
                        assert next(chunks)
                    selected_blob = owner._incremental_blobs[0]
                    blocked = True
        assert owner is not None and selected_blob is not None and artifact is not None
        assert artifact.exists() and owner.connection is measured.connection and not owner._settled
        assert owner._incremental_blobs == [selected_blob]
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        assert artifact.exists() and len(selected_blob) > 0
        blocked = False
        owner.close()
        assert owner._settled and not artifact.exists()
        with pytest.raises(sqlite3.ProgrammingError):
            len(selected_blob)
    finally:
        blocked = False
        if owner is not None:
            owner.close()


@pytest.mark.parametrize("sqlite_encoding", [False, True])
def test_census_stream_never_fetches_a_valid_eight_mib_key_as_python_text(
    monkeypatch: pytest.MonkeyPatch, sqlite_encoding: bool
) -> None:
    from polylogue.storage.io_phase_metrics import _MeasuredCursor

    key = "chatgpt-export:" + "x" * (8 * 1024 * 1024)
    expected = json.dumps([key]).encode("ascii")
    expected_digest = hashlib.sha256(expected).digest()
    expected_length = len(expected)
    original_next, original_fetchone = _MeasuredCursor.__next__, _MeasuredCursor.fetchone
    original_fetchmany, original_fetchall = _MeasuredCursor.fetchmany, _MeasuredCursor.fetchall

    def verify_row(row: object) -> None:
        if row is not None:
            assert isinstance(row, (tuple, sqlite3.Row))
            assert not any(isinstance(value, str) and len(value) > 65536 for value in row)

    def next_row(cursor: _MeasuredCursor) -> object:
        row = original_next(cursor)
        verify_row(row)
        return row

    def fetchone(cursor: _MeasuredCursor) -> object:
        row = original_fetchone(cursor)
        verify_row(row)
        return row

    def fetchmany(cursor: _MeasuredCursor, *args: Any, **kwargs: Any) -> Any:
        rows = original_fetchmany(cursor, *args, **kwargs)
        for row in rows:
            verify_row(row)
        return rows

    def fetchall(cursor: _MeasuredCursor) -> Any:
        rows = original_fetchall(cursor)
        for row in rows:
            verify_row(row)
        return rows

    with _measurement([key]) as measured:
        monkeypatch.setattr(_MeasuredCursor, "__next__", next_row)
        monkeypatch.setattr(_MeasuredCursor, "fetchone", fetchone)
        monkeypatch.setattr(_MeasuredCursor, "fetchmany", fetchmany)
        monkeypatch.setattr(_MeasuredCursor, "fetchall", fetchall)
        with measured.keys_json_stream(sqlite_encoding=sqlite_encoding) as (length, chunks):
            digest = hashlib.sha256()
            received = 0
            for chunk in chunks:
                assert len(chunk) <= 65536
                digest.update(chunk)
                received += len(chunk)
            assert length == received == expected_length
            assert digest.digest() == expected_digest
