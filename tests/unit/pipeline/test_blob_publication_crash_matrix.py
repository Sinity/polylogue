"""Deterministic crash-injection matrix for the blob publication lifecycle.

polylogue-0puw AC3: inject a failure after each publication boundary in the
real acquire->publish->reference->commit->finalize chain (not a toy replica --
these tests drive `_process_ingest_batch_sync` and `write_source_raw_session`,
the actual production entry points) and assert the surviving state either (a)
resumes safely on retry, or (b) lands in the exact classification bucket
`reconcile_blob_publication_reservations` already names (missing / referenced
/ unresolved) so polylogue-qs0a's reconciler has evidence for every reachable
crash state instead of speculation.

Boundaries covered, matching the design's named chain:
1. reservation -> blob write        (test_crash_after_reservation...)
2. blob write -> reference (index)  (test_crash_after_blob_write_before_index_write...)
3. blob write -> reference (source) (test_crash_during_source_commit_transaction...)
4. reference/commit -> finalize     (test_crash_after_index_commit_before_finalization...)
5. finalize atomicity               (test_finalization_transaction_is_atomic...)
"""

from __future__ import annotations

import sqlite3
from hashlib import sha256
from pathlib import Path

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
import polylogue.storage.sqlite.archive_tiers.source_write as source_write
from polylogue.core.enums import Origin
from polylogue.pipeline.services.ingest_batch import _process_ingest_batch_sync
from polylogue.pipeline.services.ingest_worker import IngestRecordResult, SessionWritePayload
from polylogue.storage.blob_publication import (
    ArchiveBlobPublisher,
    consume_blob_publication_receipt,
    exclude_archive_blob_publishers,
    reconcile_blob_publication_reservations,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.connection import open_connection
from tests.unit.pipeline.test_ingest_batch import (
    _attachment_ref_tuple,
    _attachment_tuple,
    _message_tuple,
    _session_data,
)


def _reservation_rows(source_db: Path, blob_hash: bytes) -> list[tuple[str]]:
    with sqlite3.connect(source_db) as conn:
        return conn.execute(
            "SELECT publication_id FROM blob_publication_reservations WHERE blob_hash = ?",
            (blob_hash,),
        ).fetchall()


def test_crash_after_reservation_before_blob_write_leaves_missing_classified_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary 1: reservation durably committed, blob write crashes.

    ``reserve_many`` commits the reservation row in its own transaction
    before ``publish_many`` ever runs, so a crash here must never lose the
    reservation -- and since no attachment row can exist yet, the blob is
    provably unreferenced. Mutation-sensitive: without the crash, the
    reservation is consumed and the table ends empty (see the retry
    assertion below); the fault proves the row survives instead.

    ``_write_session_entry`` catches and logs per-session write exceptions
    (_core.py's "Error writing session" path) rather than propagating them,
    so a raise here does NOT fail the whole batch -- it is recorded as a
    parse failure for this raw_id and the batch call returns normally. That
    swallowing is itself real production resilience (one bad session must
    not abort an entire ingest batch); this test asserts against it rather
    than a fabricated raise.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    db_path = archive_root / "index.db"
    source_path = tmp_path / "raw.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    raw_id = "raw-crash-reservation"
    session_id = "codex-session:crash-reservation"
    payload = b"crash after reservation, before blob write"
    expected_hash = sha256(payload).digest()

    def build_session() -> SessionWritePayload:
        # A fresh payload per call: _process_ingest_batch_sync mutates (clears)
        # session contents in place after draining it (discard_session_data_payload),
        # so retrying with the SAME SessionWritePayload instance would silently
        # see an already-emptied session -- not a production bug, a shared-fixture
        # trap this harness must avoid to keep the retry assertion meaningful.
        attachment = _attachment_tuple("att-1", mime_type="text/plain", inline_bytes=payload)
        return _session_data(
            session_id,
            content_hash="crash-reservation",
            raw_id=raw_id,
            message_tuples=[_message_tuple("msg-1", session_id, role="user", text="x", content_hash="m", sort_key=0.0)],
            attachment_tuples=[attachment],
            attachment_ref_tuples=[_attachment_ref_tuple("att-1", session_id, "msg-1")],
        )

    raw_record = RawSessionRecord(
        raw_id=raw_id,
        source_name="codex",
        source_path=str(source_path),
        blob_size=source_path.stat().st_size,
        acquired_at="2026-04-02T00:00:00Z",
    )

    def fake_ingest_record(
        record: RawSessionRecord,
        archive_root_str: str,
        validation_mode: str,
        measure_ingest_result_size: bool,
        *,
        blob_root_str: str | None,
    ) -> IngestRecordResult:
        del archive_root_str, validation_mode, measure_ingest_result_size, blob_root_str
        assert record.raw_id == raw_id
        return IngestRecordResult(raw_id=record.raw_id, sessions=[build_session()])

    def boom_publish(_store: object, _prepared: object) -> None:
        raise RuntimeError("simulated crash: after reservation, before blob write")

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest_record)

    with monkeypatch.context() as mp:
        mp.setattr(BlobStore, "publish_many", boom_publish)
        summary = _process_ingest_batch_sync(
            [raw_record],
            db_path=db_path,
            archive_root_str=str(archive_root),
            blob_root_str=str(archive_root / "blob"),
            validation_mode="off",
            ingest_workers=1,
            measure_ingest_result_size=False,
        )

    assert raw_id in summary.failed_raw_ids
    assert "after reservation, before blob write" in summary.failed_raw_ids[raw_id]
    assert len(_reservation_rows(archive_root / "source.db", expected_hash)) == 1
    assert not BlobStore(archive_root / "blob").exists(expected_hash.hex())
    with open_connection(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM attachments WHERE blob_hash = ?", (expected_hash,)).fetchone()[0] == 0

    # Unexcluded reconciliation classifies but must not delete.
    retained = reconcile_blob_publication_reservations(archive_root / "source.db", archive_root / "blob")
    assert retained.retained_missing == 1
    assert retained.cleared_missing == 0
    assert retained.unresolved == 0
    assert len(_reservation_rows(archive_root / "source.db", expected_hash)) == 1

    # Excluded reconciliation clears the terminal, blob-missing obligation.
    with exclude_archive_blob_publishers(archive_root / "source.db") as exclusion:
        cleared = reconcile_blob_publication_reservations(
            archive_root / "source.db",
            archive_root / "blob",
            index_db_path=db_path,
            writer_exclusion=exclusion,
        )
    assert cleared.cleared_missing == 1
    assert _reservation_rows(archive_root / "source.db", expected_hash) == []

    # Safe resume: retrying the same attempt (fault removed) converges cleanly
    # and its own finalization leaves zero outstanding reservations.
    _process_ingest_batch_sync(
        [raw_record],
        db_path=db_path,
        archive_root_str=str(archive_root),
        blob_root_str=str(archive_root / "blob"),
        validation_mode="off",
        ingest_workers=1,
        measure_ingest_result_size=False,
    )
    assert BlobStore(archive_root / "blob").exists(expected_hash.hex())
    with open_connection(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM attachments WHERE blob_hash = ?", (expected_hash,)).fetchone()[0] == 1
    assert _reservation_rows(archive_root / "source.db", expected_hash) == []


def test_crash_after_blob_write_before_index_write_lands_in_unresolved_bucket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary 2: blob durably published, the index reference write crashes.

    Reproduces qs0a's named ambiguous case from real code: reservation
    present, blob bytes present, nothing references it yet. Confirms
    (as a regression, not speculation) that this bucket stays retained even
    under full writer exclusion -- reconcile_blob_publication_reservations's
    ``unresolved`` branch never checks ``may_clear`` -- so a future
    classification change has a real fixture to change against.

    Like boundary 1, this raise is caught per-session by
    ``_write_session_entry`` -- the batch call itself does not raise; the
    fault surfaces as a recorded parse failure for this raw_id instead.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    db_path = archive_root / "index.db"
    source_path = tmp_path / "raw.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    raw_id = "raw-crash-index-write"
    session_id = "codex-session:crash-index-write"
    payload = b"crash after blob write, before index reference"
    expected_hash = sha256(payload).digest()
    attachment = _attachment_tuple("att-1", mime_type="text/plain", inline_bytes=payload)
    session = _session_data(
        session_id,
        content_hash="crash-index-write",
        raw_id=raw_id,
        message_tuples=[_message_tuple("msg-1", session_id, role="user", text="x", content_hash="m", sort_key=0.0)],
        attachment_tuples=[attachment],
        attachment_ref_tuples=[_attachment_ref_tuple("att-1", session_id, "msg-1")],
    )
    raw_record = RawSessionRecord(
        raw_id=raw_id,
        source_name="codex",
        source_path=str(source_path),
        blob_size=source_path.stat().st_size,
        acquired_at="2026-04-02T00:00:00Z",
    )

    def fake_ingest_record(
        record: RawSessionRecord,
        archive_root_str: str,
        validation_mode: str,
        measure_ingest_result_size: bool,
        *,
        blob_root_str: str | None,
    ) -> IngestRecordResult:
        del archive_root_str, validation_mode, measure_ingest_result_size, blob_root_str
        assert record.raw_id == raw_id
        return IngestRecordResult(raw_id=record.raw_id, sessions=[session])

    def boom_write_parsed(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated crash: after blob write, before index reference")

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest_record)
    monkeypatch.setattr(ingest_batch_core, "write_parsed_session_to_archive", boom_write_parsed)

    summary = _process_ingest_batch_sync(
        [raw_record],
        db_path=db_path,
        archive_root_str=str(archive_root),
        blob_root_str=str(archive_root / "blob"),
        validation_mode="off",
        ingest_workers=1,
        measure_ingest_result_size=False,
    )

    assert raw_id in summary.failed_raw_ids
    assert "after blob write, before index reference" in summary.failed_raw_ids[raw_id]
    assert len(_reservation_rows(archive_root / "source.db", expected_hash)) == 1
    assert BlobStore(archive_root / "blob").exists(expected_hash.hex())
    with open_connection(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM attachments WHERE blob_hash = ?", (expected_hash,)).fetchone()[0] == 0

    with exclude_archive_blob_publishers(archive_root / "source.db") as exclusion:
        outcome = reconcile_blob_publication_reservations(
            archive_root / "source.db",
            archive_root / "blob",
            index_db_path=db_path,
            writer_exclusion=exclusion,
        )
    assert outcome.unresolved == 1
    assert outcome.cleared_missing == 0
    assert outcome.cleared_referenced == 0
    # The unresolved bucket is retained even under full exclusion -- current,
    # deliberate behavior (see reconcile_blob_publication_reservations), not
    # a bug this bead fixes; this is the evidence qs0a's classification work
    # consumes.
    assert len(_reservation_rows(archive_root / "source.db", expected_hash)) == 1


def test_crash_during_source_commit_transaction_rolls_back_atomically_to_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary 3: the source-tier "source commit" step crashes mid-transaction.

    write_source_raw_session runs the raw_sessions insert and the receipt
    consumption (_insert_blob_ref) inside one ``with conn:`` block; a crash
    between them must roll back BOTH atomically, not just skip the receipt
    consumption -- proving the raw-acquisition path lands in the exact same
    classification bucket (unresolved) as the index-attachment path above,
    from an independent code path.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    source_db = archive_root / "source.db"
    payload = b"crash mid source-commit transaction"
    publisher = ArchiveBlobPublisher(source_db, archive_root / "blob")
    blob_hash_hex, _size = publisher.write_from_bytes(payload)
    receipt_id = publisher.receipt_id(blob_hash_hex)
    assert receipt_id is not None
    publisher.flush()
    expected_hash = bytes.fromhex(blob_hash_hex)

    def boom_insert_blob_ref(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated crash: during source commit transaction")

    monkeypatch.setattr(source_write, "_insert_blob_ref", boom_insert_blob_ref)

    with sqlite3.connect(source_db) as conn, pytest.raises(RuntimeError, match="during source commit transaction"):
        source_write.write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="raw.jsonl",
            source_index=0,
            payload=payload,
            acquired_at_ms=1,
            raw_id="raw-source-commit-crash",
            blob_publication_receipt_id=receipt_id,
        )

    with sqlite3.connect(source_db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", ("raw-source-commit-crash",)).fetchone()[
                0
            ]
            == 0
        )
        assert conn.execute("SELECT COUNT(*) FROM blob_refs WHERE blob_hash = ?", (expected_hash,)).fetchone()[0] == 0
    assert len(_reservation_rows(source_db, expected_hash)) == 1
    assert BlobStore(archive_root / "blob").exists(blob_hash_hex)

    with exclude_archive_blob_publishers(source_db) as exclusion:
        outcome = reconcile_blob_publication_reservations(source_db, archive_root / "blob", writer_exclusion=exclusion)
    assert outcome.unresolved == 1
    assert len(_reservation_rows(source_db, expected_hash)) == 1


def test_crash_after_index_commit_before_finalization_self_heals_via_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary 4: index commit durable, finalization (receipt consumption) crashes.

    The attachment row is committed through ArchiveWriteGateway before
    ``pending_attachment_receipts`` is drained (_core.py), so a crash in that
    drain leaves a referenced-but-unconsumed reservation. This is exactly the
    startup-reconciliation leak polylogue-qs0a's PR #3104 fixed: proves the
    now-fixed ``reconcile_blob_publication_reservations_under_exclusion``
    path actually clears this specific crash's residue.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    db_path = archive_root / "index.db"
    source_path = tmp_path / "raw.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    raw_id = "raw-crash-finalization"
    session_id = "codex-session:crash-finalization"
    payload = b"crash after index commit, before finalization"
    expected_hash = sha256(payload).digest()
    attachment = _attachment_tuple("att-1", mime_type="text/plain", inline_bytes=payload)
    session = _session_data(
        session_id,
        content_hash="crash-finalization",
        raw_id=raw_id,
        message_tuples=[_message_tuple("msg-1", session_id, role="user", text="x", content_hash="m", sort_key=0.0)],
        attachment_tuples=[attachment],
        attachment_ref_tuples=[_attachment_ref_tuple("att-1", session_id, "msg-1")],
    )
    raw_record = RawSessionRecord(
        raw_id=raw_id,
        source_name="codex",
        source_path=str(source_path),
        blob_size=source_path.stat().st_size,
        acquired_at="2026-04-02T00:00:00Z",
    )

    def fake_ingest_record(
        record: RawSessionRecord,
        archive_root_str: str,
        validation_mode: str,
        measure_ingest_result_size: bool,
        *,
        blob_root_str: str | None,
    ) -> IngestRecordResult:
        del archive_root_str, validation_mode, measure_ingest_result_size, blob_root_str
        assert record.raw_id == raw_id
        return IngestRecordResult(raw_id=record.raw_id, sessions=[session])

    def boom_consume(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated crash: after index commit, before finalization")

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest_record)
    monkeypatch.setattr(ingest_batch_core, "consume_blob_publication_receipt", boom_consume)

    with pytest.raises(RuntimeError, match="after index commit, before finalization"):
        _process_ingest_batch_sync(
            [raw_record],
            db_path=db_path,
            archive_root_str=str(archive_root),
            blob_root_str=str(archive_root / "blob"),
            validation_mode="off",
            ingest_workers=1,
            measure_ingest_result_size=False,
        )

    # The index commit itself is unaffected by the later finalization crash --
    # verify against a fresh connection so a would-be uncommitted rollback of
    # the *finalization* transaction can't be mistaken for the index write.
    with open_connection(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM attachments WHERE blob_hash = ?", (expected_hash,)).fetchone()[0] == 1
    assert len(_reservation_rows(archive_root / "source.db", expected_hash)) == 1

    with exclude_archive_blob_publishers(archive_root / "source.db") as exclusion:
        outcome = reconcile_blob_publication_reservations(
            archive_root / "source.db",
            archive_root / "blob",
            index_db_path=db_path,
            writer_exclusion=exclusion,
        )
    assert outcome.cleared_referenced == 1
    assert _reservation_rows(archive_root / "source.db", expected_hash) == []


def test_finalization_transaction_is_atomic_across_multiple_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary 5: a crash mid-finalization-loop rolls back every receipt, not just the failing one.

    Two inline attachments in one session share the deferred finalization
    loop (_core.py:~1315-1319). A crash on the second receipt must not leave
    the first one silently consumed while the second survives -- the whole
    drain runs in one ``BEGIN IMMEDIATE`` transaction. Mutation-sensitive:
    narrowing the finalization loop to commit per-receipt instead of once
    would make this test observe exactly one surviving row instead of two.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    db_path = archive_root / "index.db"
    source_path = tmp_path / "raw.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    raw_id = "raw-crash-finalization-atomic"
    session_id = "codex-session:crash-finalization-atomic"
    payload_a = b"finalization atomicity attachment a"
    payload_b = b"finalization atomicity attachment b"
    hash_a = sha256(payload_a).digest()
    hash_b = sha256(payload_b).digest()
    attachment_a = _attachment_tuple("att-a", mime_type="text/plain", inline_bytes=payload_a)
    attachment_b = _attachment_tuple("att-b", mime_type="text/plain", inline_bytes=payload_b)
    session = _session_data(
        session_id,
        content_hash="crash-finalization-atomic",
        raw_id=raw_id,
        message_tuples=[_message_tuple("msg-1", session_id, role="user", text="x", content_hash="m", sort_key=0.0)],
        attachment_tuples=[attachment_a, attachment_b],
        attachment_ref_tuples=[
            _attachment_ref_tuple("att-a", session_id, "msg-1"),
            _attachment_ref_tuple("att-b", session_id, "msg-1"),
        ],
    )
    raw_record = RawSessionRecord(
        raw_id=raw_id,
        source_name="codex",
        source_path=str(source_path),
        blob_size=source_path.stat().st_size,
        acquired_at="2026-04-02T00:00:00Z",
    )

    def fake_ingest_record(
        record: RawSessionRecord,
        archive_root_str: str,
        validation_mode: str,
        measure_ingest_result_size: bool,
        *,
        blob_root_str: str | None,
    ) -> IngestRecordResult:
        del archive_root_str, validation_mode, measure_ingest_result_size, blob_root_str
        assert record.raw_id == raw_id
        return IngestRecordResult(raw_id=record.raw_id, sessions=[session])

    call_count = 0

    def boom_on_second(conn: sqlite3.Connection, publication_id: str | None, blob_hash: bytes) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated crash: mid-finalization-loop")
        consume_blob_publication_receipt(conn, publication_id, blob_hash)

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest_record)
    monkeypatch.setattr(ingest_batch_core, "consume_blob_publication_receipt", boom_on_second)

    with pytest.raises(RuntimeError, match="mid-finalization-loop"):
        _process_ingest_batch_sync(
            [raw_record],
            db_path=db_path,
            archive_root_str=str(archive_root),
            blob_root_str=str(archive_root / "blob"),
            validation_mode="off",
            ingest_workers=1,
            measure_ingest_result_size=False,
        )

    assert call_count == 2
    # Both receipts survive -- the first call's DELETE was never committed
    # because the second call's raise rolled back the whole transaction.
    assert len(_reservation_rows(archive_root / "source.db", hash_a)) == 1
    assert len(_reservation_rows(archive_root / "source.db", hash_b)) == 1
    with open_connection(db_path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM attachments WHERE blob_hash IN (?, ?)", (hash_a, hash_b)).fetchone()[0]
            == 2
        )

    with exclude_archive_blob_publishers(archive_root / "source.db") as exclusion:
        outcome = reconcile_blob_publication_reservations(
            archive_root / "source.db",
            archive_root / "blob",
            index_db_path=db_path,
            writer_exclusion=exclusion,
        )
    assert outcome.cleared_referenced == 2
    assert _reservation_rows(archive_root / "source.db", hash_a) == []
    assert _reservation_rows(archive_root / "source.db", hash_b) == []
