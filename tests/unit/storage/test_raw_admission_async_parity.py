"""Durable parity checks for pending raw admission's sync and async adapters."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.enums import Origin, Provider
from polylogue.pipeline.services.acquisition_persistence import persist_raw_record
from polylogue.pipeline.services.acquisition_records import make_raw_record
from polylogue.pipeline.stage_models import AcquireResult
from polylogue.sources.parsers.base import RawSessionData
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.raw_admission import (
    PendingPreParseRawAdmissionRequest,
    acquisition_timestamp_ms,
    execute_raw_admission_plan_sync,
    plan_raw_admission,
)
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.queries import raw_writes


def _request(
    *,
    source_path: str = "/captures/source.json",
    source_index: int = 0,
    raw_id: str | None = None,
    acquired_at_ms: int = 0,
    receipt: str | None = "receipt-1",
) -> PendingPreParseRawAdmissionRequest:
    payload = b'{"pending":"raw"}'
    return PendingPreParseRawAdmissionRequest(
        origin=Origin.CHATGPT_EXPORT,
        capture_mode=Provider.CHATGPT,
        source_path=source_path,
        source_index=source_index,
        blob_hash=hashlib.sha256(payload).digest(),
        blob_size=len(payload),
        acquired_at_ms=acquired_at_ms,
        file_mtime_ms=17,
        raw_id=raw_id,
        blob_publication_receipt_id=receipt,
    )


def _reserve(conn: sqlite3.Connection, request: PendingPreParseRawAdmissionRequest) -> None:
    assert request.blob_publication_receipt_id is not None
    conn.execute(
        """
        INSERT INTO blob_publication_reservations
            (publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms)
        VALUES (?, ?, ?, 'test', 1)
        """,
        (request.blob_publication_receipt_id, request.blob_hash, request.blob_size),
    )
    conn.commit()


def _snapshot(path: Path) -> dict[str, list[tuple[object, ...]]]:
    with sqlite3.connect(path) as conn:
        return {
            "raw": list(
                conn.execute(
                    """
                    SELECT raw_id, origin, capture_mode, native_id, source_path, source_index,
                           blob_hash, blob_size, acquired_at_ms, file_mtime_ms, logical_source_key,
                           revision_kind, source_revision, acquisition_generation, revision_authority
                    FROM raw_sessions ORDER BY raw_id
                    """
                )
            ),
            "capture": list(
                conn.execute(
                    "SELECT raw_id, capture_mode, first_observed_at_ms FROM raw_capture_observations ORDER BY raw_id"
                )
            ),
            "refs": list(
                conn.execute(
                    "SELECT blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms FROM blob_refs ORDER BY ref_id"
                )
            ),
            "reservations": list(conn.execute("SELECT publication_id FROM blob_publication_reservations")),
        }


async def test_pending_preparse_admission_sync_and_aiosqlite_have_identical_durable_effects(tmp_path: Path) -> None:
    sync_root = tmp_path / "sync"
    async_root = tmp_path / "async"
    initialize_active_archive_root(sync_root)
    initialize_active_archive_root(async_root)
    request = _request()
    plan = plan_raw_admission(request)

    with sqlite3.connect(sync_root / "source.db") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        _reserve(conn, request)
        first = execute_raw_admission_plan_sync(conn, plan)
        duplicate = execute_raw_admission_plan_sync(conn, plan)
    assert first.raw_id == duplicate.raw_id == plan.raw_id

    with sqlite3.connect(async_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO blob_publication_reservations
                (publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms)
            VALUES (?, ?, ?, 'test', 1)
            """,
            (request.blob_publication_receipt_id, request.blob_hash, request.blob_size),
        )
    async_backend = SQLiteBackend(db_path=async_root / "index.db")
    async with async_backend.bulk_connection():
        first_async = await async_backend.admit_raw(request)
        duplicate_async = await async_backend.admit_raw(request)
    await async_backend.close()
    assert first_async.inserted is True
    assert duplicate_async.inserted is False
    assert first_async.result.raw_id == duplicate_async.result.raw_id == plan.raw_id
    assert _snapshot(sync_root / "source.db") == _snapshot(async_root / "source.db")
    durable = _snapshot(async_root / "source.db")
    assert durable["capture"] == [(plan.raw_id, Provider.CHATGPT.value, 0)]
    assert len(durable["refs"]) == 1
    assert durable["reservations"] == []


async def test_pending_preparse_conflict_has_no_async_side_effects_after_rollback(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    request = _request(raw_id="immutable-raw", receipt=None)
    backend = SQLiteBackend(db_path=root / "index.db")
    async with backend.bulk_connection():
        await backend.admit_raw(request)
    before = _snapshot(root / "source.db")
    conflicting = _request(
        source_path="/captures/substituted.json",
        raw_id="immutable-raw",
        acquired_at_ms=99,
        receipt=None,
    )
    async with backend.bulk_connection():
        with pytest.raises(ValueError, match="acquisition evidence"):
            async with backend.transaction():
                await backend.admit_raw(conflicting)
    await backend.close()
    assert _snapshot(root / "source.db") == before


async def test_two_acquisition_coordinates_share_one_blob_but_not_raw_identity(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    first = plan_raw_admission(_request(source_path="/captures/a.json", source_index=0, receipt=None))
    second = plan_raw_admission(_request(source_path="/captures/b.json", source_index=1, receipt=None))
    assert first.raw_id != second.raw_id
    assert first.request.blob_hash == second.request.blob_hash
    backend = SQLiteBackend(db_path=root / "index.db")
    async with backend.bulk_connection():
        await backend.admit_raw(first.request)
        await backend.admit_raw(second.request)
    await backend.close()
    with sqlite3.connect(root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (2,)
        assert conn.execute("SELECT COUNT(DISTINCT blob_hash) FROM raw_sessions").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'raw_payload'").fetchone() == (2,)


async def test_epoch_zero_is_valid_and_invalid_acquisition_times_are_refused_before_sql(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    plan = plan_raw_admission(_request(acquired_at_ms=0, receipt=None))
    with sqlite3.connect(root / "source.db") as conn:
        execute_raw_admission_plan_sync(conn, plan)
        assert conn.execute("SELECT acquired_at_ms FROM raw_sessions WHERE raw_id = ?", (plan.raw_id,)).fetchone() == (
            0,
        )

    assert acquisition_timestamp_ms("0") == 0
    for invalid in (None, "not-a-time", -1):
        with pytest.raises(ValueError):
            acquisition_timestamp_ms(invalid)

    missing = _request(receipt=None)
    object.__setattr__(missing, "acquired_at_ms", cast(int, None))
    with pytest.raises(ValueError, match="acquired_at_ms"):
        plan_raw_admission(missing)

    backend = SQLiteBackend(db_path=root / "index.db")
    for invalid_request in (
        _request(acquired_at_ms=-1, receipt=None),
        missing,
    ):
        with pytest.raises(ValueError):
            await backend.admit_raw(invalid_request)
    await backend.close()
    with sqlite3.connect(root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)


async def test_acquisition_persistence_enters_plan_executor_and_failure_leaves_no_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: the production persistence route reaches the plan executor.

    The injected executor failure occurs before its first source-tier mutation;
    the real acquisition helper records the failure and leaves no raw row.
    """
    backend = SQLiteBackend(db_path=tmp_path / "index.db")
    from polylogue.storage.repository import SessionRepository

    repository = SessionRepository(backend=backend)
    record = make_raw_record(
        RawSessionData(raw_bytes=b'{"route":"entered"}', source_path="/captures/route.json"),
        "chatgpt",
        blob_root=tmp_path / "blob",
    )
    entered = False

    async def fail_before_mutation(*args: object, **kwargs: object) -> object:
        nonlocal entered
        del args, kwargs
        entered = True
        raise RuntimeError("injected plan executor failure")

    monkeypatch.setattr(raw_writes, "execute_raw_admission_plan_async", fail_before_mutation)
    result = AcquireResult()
    await persist_raw_record(repository, record, result=result)
    await backend.close()

    assert entered is True
    assert result.errors == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)
