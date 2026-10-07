"""Deterministic crash-injection matrix for the blob publication lifecycle.

polylogue-0puw AC3: inject a failure at each publication boundary of the real
canonical chain -- retained preparation reserves and publishes attachment
bytes, the Index write references them, the Source durable-reference
transaction consumes the reservation -- and assert the surviving state either
resumes safely on retry or lands in a bucket
``reconcile_blob_publication_reservations`` names (missing / referenced /
unresolved). The chain is driven through the canonical retained owner
(``RawObservationConvergenceOwner.ingest_retained_raw_ids``) with a real
inline-attachment capture, not a replica.

Boundaries covered:
1. reservation -> blob write        (BlobStore.publish_many fails)
2. Source reference -> Index write (the Index attachment write fails)
3. Source durable-reference commit (the receipt consumption fails)
Atomic multi-receipt finalization is owned by
tests/unit/storage/test_archive_tiers_source_write.py::
test_source_reference_commit_atomically_consumes_publication_reservation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import polylogue.storage.blob_publication as blob_publication
import polylogue.storage.sqlite.archive_tiers.write as archive_write
from polylogue.core.enums import Provider
from polylogue.storage.blob_publication import (
    BlobPublicationReconciliation,
    abandon_blob_publication_receipts,
    exclude_archive_blob_publishers,
    reconcile_blob_publication_reservations,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write, run_off_event_loop
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.live_provider_proof import native_proof_artifact


def _reservation_rows(source_db: Path, blob_hash: bytes) -> list[tuple[str]]:
    with sqlite3.connect(source_db) as conn:
        return conn.execute(
            "SELECT publication_id FROM blob_publication_reservations WHERE blob_hash = ?",
            (blob_hash,),
        ).fetchall()


def _all_reservations(source_db: Path) -> list[tuple[str, bytes]]:
    with sqlite3.connect(source_db) as conn:
        return [
            (str(row[0]), bytes(row[1]))
            for row in conn.execute("SELECT publication_id, blob_hash FROM blob_publication_reservations")
        ]


def _attachment_hashes(index_db: Path) -> set[bytes]:
    with sqlite3.connect(index_db) as conn:
        return {bytes(row[0]) for row in conn.execute("SELECT blob_hash FROM attachments WHERE blob_hash IS NOT NULL")}


def _reconcile_excluded(root: Path) -> BlobPublicationReconciliation:
    """Excluded reconciliation deletes Source rows, so it runs as the archive writer does."""

    def reconcile() -> BlobPublicationReconciliation:
        with (
            write_lease("test.blob.reconcile", archive_root=root),
            exclude_archive_blob_publishers(root / "source.db") as exclusion,
        ):
            return reconcile_blob_publication_reservations(
                root / "source.db", root / "blob", index_db_path=root / "index.db", writer_exclusion=exclusion
            )

    return run_off_event_loop(reconcile)


async def _acquire_inline_attachment_capture(tmp_path: Path) -> tuple[Path, str, dict[bytes, int]]:
    """Acquire a real Grok capture whose canonical parse yields inline attachments."""
    root = tmp_path / "archive"
    envelope, _messages, attachments = native_proof_artifact(
        tmp_path, "native-inline-attachment-v1.json", Provider.GROK
    )
    assert attachments > 0
    contents = [base64.b64decode(entry["content_base64"]) for entry in envelope["session"]["attachments"]]
    expected = {hashlib.sha256(content).digest(): len(content) for content in contents}
    payload = json.dumps(envelope).encode()

    def acquire() -> str:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.GROK,
                payload=payload,
                source_path="neutral-grok-capture.json",
                canonical_source_path="neutral-grok-capture.json",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(root, acquire)
    return root, raw_id, expected


async def _ingest(root: Path, raw_id: str) -> BaseException | None:
    """Run the canonical owner once; return the surfaced failure, if any."""
    try:
        async with prepared_live_convergence_owner(root) as owner:
            receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
    except BaseException as exc:  # the crash under test may surface as any typed failure
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return exc
    if any(receipt.quarantined for receipt in receipts):
        return RuntimeError("retained owner quarantined the raw")
    return None


def _assert_crash_consistent(root: Path, expected: dict[bytes, int]) -> None:
    """The durable laws every reachable crash state must satisfy.

    No Index row references bytes that are not on disk; every published
    attachment blob is either referenced or still reserved (never an
    unreserved orphan); unexcluded reconciliation classifies without deleting.
    """
    from polylogue.storage.blob_liveness import LivenessState, inspect_blob_liveness

    store = BlobStore(root / "blob")
    referenced = _attachment_hashes(root / "index.db")
    reserved = {blob_hash for _publication, blob_hash in _all_reservations(root / "source.db")}
    for blob_hash in referenced:
        assert store.exists(blob_hash.hex()), "an Index attachment references missing bytes"
    with sqlite3.connect(root / "source.db") as source, sqlite3.connect(root / "index.db") as index:
        for blob_hash in expected:
            if store.exists(blob_hash.hex()):
                live = (
                    inspect_blob_liveness(source, blob_hash.hex(), index_conn=index, require_index=True).state
                    is LivenessState.LIVE
                )
                assert live or blob_hash in reserved, "published bytes are neither referenced nor reserved"
    before = _all_reservations(root / "source.db")
    retained = reconcile_blob_publication_reservations(root / "source.db", root / "blob")
    assert retained.cleared_missing == 0
    assert retained.cleared_referenced == 0
    assert _all_reservations(root / "source.db") == before


async def _assert_retry_converges(root: Path, raw_id: str, expected: dict[bytes, int]) -> None:
    assert await _ingest(root, raw_id) is None
    store = BlobStore(root / "blob")
    with sqlite3.connect(root / "index.db") as conn:
        rows = conn.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchall()
    assert {bytes(row[0]): int(row[1]) for row in rows} == expected
    assert all(row[2] == "acquired" for row in rows)
    assert all(store.exists(blob_hash.hex()) for blob_hash in expected)
    assert _all_reservations(root / "source.db") == []


def test_reconciliation_keeps_same_hash_receipts_without_their_exact_consuming_transaction(tmp_path: Path) -> None:
    """A live attachment cannot consume unrelated same-content publications."""
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    store = BlobStore(archive_root / "blob")
    blob_hash, size = store.write_from_bytes(b"same retained bytes")
    with sqlite3.connect(source_db) as source:
        source.executemany(
            """INSERT INTO blob_publication_reservations
            (publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms) VALUES (?, ?, ?, 'test', 1)""",
            [("publication-a", bytes.fromhex(blob_hash), size), ("publication-b", bytes.fromhex(blob_hash), size)],
        )
    with sqlite3.connect(index_db) as index:
        index.execute(
            "INSERT INTO attachments (attachment_id, blob_hash, acquisition_status) VALUES ('attachment', ?, 'acquired')",
            (bytes.fromhex(blob_hash),),
        )

    outcome = _reconcile_excluded(archive_root)

    assert outcome.cleared_referenced == 0
    assert outcome.retained_referenced == 2
    assert _reservation_rows(source_db, bytes.fromhex(blob_hash)) == [("publication-a",), ("publication-b",)]

    with sqlite3.connect(index_db) as index:
        index.execute("DELETE FROM attachments WHERE attachment_id = 'attachment'")
    with write_lease("test.blob.abandon", archive_root=archive_root):
        abandonment = abandon_blob_publication_receipts(
            source_db, store.root, ["publication-a"], confirmed=True, index_db_path=index_db
        )
    assert abandonment.abandoned == 1
    assert _reservation_rows(source_db, bytes.fromhex(blob_hash)) == [("publication-b",)]


def test_reconciliation_retains_receipts_when_required_index_is_unavailable(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    source_db = archive_root / "source.db"
    store = BlobStore(archive_root / "blob")
    blob_hash, size = store.write_from_bytes(b"awaiting index")
    with sqlite3.connect(source_db) as source:
        source.execute(
            """INSERT INTO blob_publication_reservations
            (publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms) VALUES ('publication', ?, ?, 'test', 1)""",
            (bytes.fromhex(blob_hash), size),
        )

    with exclude_archive_blob_publishers(source_db) as exclusion:
        outcome = reconcile_blob_publication_reservations(
            source_db, store.root, index_db_path=archive_root / "missing-index.db", writer_exclusion=exclusion
        )

    assert outcome.retained_blocked == 1
    assert any("index tier is unavailable" in blocker for blocker in outcome.blockers)
    assert _reservation_rows(source_db, bytes.fromhex(blob_hash)) == [("publication",)]


@pytest.mark.asyncio
async def test_crash_after_reservation_before_blob_write_leaves_missing_classified_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary 1: reservation durably committed, the blob write crashes.

    The reservation commits in its own transaction before ``publish_many``
    runs, so the crash must not lose it, and no attachment can reference the
    absent bytes. Excluded reconciliation clears the blob-missing obligation;
    a retry without the fault converges. Anti-vacuity: without the fault the
    reservation is consumed and the retry assertions alone hold.
    """
    root, raw_id, expected = await _acquire_inline_attachment_capture(tmp_path)

    def boom_publish(_store: object, _prepared: object) -> None:
        raise RuntimeError("simulated crash: after reservation, before blob write")

    with monkeypatch.context() as patch:
        patch.setattr(BlobStore, "publish_many", boom_publish)
        failure = await _ingest(root, raw_id)

    assert failure is not None
    store = BlobStore(root / "blob")
    for blob_hash in expected:
        assert not store.exists(blob_hash.hex())
        assert len(_reservation_rows(root / "source.db", blob_hash)) == 1
    assert not (_attachment_hashes(root / "index.db") & set(expected))
    _assert_crash_consistent(root, expected)

    cleared = _reconcile_excluded(root)
    assert cleared.cleared_missing == len(expected)
    assert _all_reservations(root / "source.db") == []

    await _assert_retry_converges(root, raw_id, expected)


@pytest.mark.asyncio
async def test_crash_after_blob_write_before_index_write_keeps_the_source_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary 2: blob published and Source reference committed, the Index write crashes.

    The Source phase precedes Index publication: its transaction writes the
    retained ``blob_refs`` attachment row and consumes the exact receipt
    together. An Index crash therefore leaves no receipt, and the bytes stay
    live through that durable Source reference (never an unreserved,
    unreferenced orphan). A retry converges the Index from the same Source
    evidence.
    """
    from polylogue.storage.blob_liveness import LivenessState, inspect_blob_liveness

    root, raw_id, expected = await _acquire_inline_attachment_capture(tmp_path)

    def boom_attachments(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated crash: after blob write, before index reference")

    with monkeypatch.context() as patch:
        patch.setattr(archive_write, "_write_attachments", boom_attachments)
        failure = await _ingest(root, raw_id)

    assert failure is not None
    store = BlobStore(root / "blob")
    assert not (_attachment_hashes(root / "index.db") & set(expected))
    assert _all_reservations(root / "source.db") == []
    with sqlite3.connect(root / "source.db") as source, sqlite3.connect(root / "index.db") as index:
        for blob_hash in expected:
            assert store.exists(blob_hash.hex())
            liveness = inspect_blob_liveness(source, blob_hash.hex(), index_conn=index, require_index=True)
            assert liveness.state is LivenessState.LIVE
            assert liveness.surfaces == ("source.db.blob_refs",)
    _assert_crash_consistent(root, expected)

    await _assert_retry_converges(root, raw_id, expected)


@pytest.mark.asyncio
async def test_crash_during_durable_reference_consumption_keeps_a_classified_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary 3: the Source durable-reference consumption crashes.

    The Source transaction rolls back atomically, so each reservation is still
    present. If the Index reference committed first, the receipt lands in the
    referenced bucket and its bytes survive; otherwise it is unresolved. Either
    way nothing is deleted without exclusion and a retry converges.

    The claim identity is derived from the retained raw, the attachment
    coordinate and the blob, so the retry re-adopts the crashed attempt's
    exact reservations and its reference transaction consumes them.
    Anti-vacuity: a random per-attempt identity leaves the crashed receipts
    behind the retry's own, and the converged reservation set is not empty.
    """
    root, raw_id, expected = await _acquire_inline_attachment_capture(tmp_path)

    def boom_consume(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated crash: during durable reference consumption")

    with monkeypatch.context() as patch:
        patch.setattr(blob_publication, "blob_publication_receipt_delete", boom_consume)
        failure = await _ingest(root, raw_id)

    assert failure is not None
    store = BlobStore(root / "blob")
    for blob_hash in expected:
        assert store.exists(blob_hash.hex())
        assert len(_reservation_rows(root / "source.db", blob_hash)) == 1
    _assert_crash_consistent(root, expected)
    crashed_receipts = _all_reservations(root / "source.db")
    assert crashed_receipts and all(publication.startswith("claim-") for publication, _hash in crashed_receipts)
    referenced = _attachment_hashes(root / "index.db") & set(expected)
    outcome = _reconcile_excluded(root)
    assert outcome.cleared_missing == 0
    assert outcome.unresolved + outcome.cleared_referenced + outcome.retained_referenced == len(expected)
    if not referenced:
        assert outcome.unresolved == len(expected)

    await _assert_retry_converges(root, raw_id, expected)
