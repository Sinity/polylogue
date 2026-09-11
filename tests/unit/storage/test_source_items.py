"""Production source-tier source-item authority laws."""

import hashlib
import sqlite3

import pytest

from polylogue.core.enums import IngestOutcome
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.source_items import (
    AcquisitionDisposition,
    complete_source_item_enumeration,
    publish_source_generation,
    record_source_item_raw_member,
    seal_source_generation,
    source_generation_census,
    source_item_id,
    transition_source_item,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _source() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    initialize_archive_tier(conn, ArchiveTier.SOURCE)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _frozen_item(conn: sqlite3.Connection, generation: str = "frozen") -> str:
    (item,) = publish_source_generation(
        conn,
        source_generation_id=generation,
        manifest_digest="a" * 64,
        addressing_mode="physical-file-v1",
        coordinates=("export.json",),
        input_blob_hashes={"export.json": b"i" * 32},
        enumeration_fingerprint="b" * 64,
        observed_at_ms=1,
    )
    return item


def _raw_member(conn: sqlite3.Connection, item: str, coordinate: str, raw_id: str = "raw") -> None:
    conn.execute(
        "INSERT INTO raw_sessions(raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms) "
        "VALUES (?, 'codex-session', '/synthetic/export.json', ?, 1, 1) ON CONFLICT(raw_id) DO NOTHING",
        (raw_id, b"r" * 32),
    )
    record_source_item_raw_member(
        conn,
        source_generation_id="frozen",
        source_item_id=item,
        record_coordinate=coordinate,
        raw_id=raw_id,
        raw_blob_hash=b"r" * 32,
    )


def test_raw_and_membership_rollback_together_including_deduplicated_records() -> None:
    """An internal membership commit would retain a partially admitted export."""
    conn = _source()
    item = _frozen_item(conn)
    conn.execute("BEGIN")
    _raw_member(conn, item, "record:0")
    _raw_member(conn, item, "record:1")
    assert conn.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
    assert source_generation_census(conn, "frozen")["enumeration_pending"] == 1


def test_interrupted_enumeration_cannot_claim_empty_or_complete() -> None:
    conn = _source()
    item = _frozen_item(conn)
    conn.execute("BEGIN")
    _raw_member(conn, item, "record:0")
    for coordinates in ((), ("record:0", "record:1")):
        with pytest.raises(ValueError, match="missing or unexpected"):
            complete_source_item_enumeration(
                conn,
                source_generation_id="frozen",
                source_item_id=item,
                enumeration_fingerprint="b" * 64,
                record_coordinates=coordinates,
                enumerated_at_ms=2,
            )
    assert source_generation_census(conn, "frozen")["enumeration_pending"] == 1
    complete_source_item_enumeration(
        conn,
        source_generation_id="frozen",
        source_item_id=item,
        enumeration_fingerprint="b" * 64,
        record_coordinates=("record:0",),
        enumerated_at_ms=2,
    )
    assert source_generation_census(conn, "frozen")["enumeration_pending"] == 0


def test_retired_raw_preserves_enumeration_digest_and_refuses_readmission() -> None:
    conn = _source()
    item = _frozen_item(conn)
    conn.execute("BEGIN")
    _raw_member(conn, item, "record:0")
    digest = complete_source_item_enumeration(
        conn,
        source_generation_id="frozen",
        source_item_id=item,
        enumeration_fingerprint="b" * 64,
        record_coordinates=("record:0",),
        enumerated_at_ms=2,
    )
    conn.commit()
    conn.execute("DELETE FROM raw_sessions WHERE raw_id='raw'")
    assert conn.execute("SELECT raw_id, raw_blob_hash FROM source_item_raw_members").fetchone() == (None, b"r" * 32)
    assert conn.execute("SELECT enumeration_digest FROM source_items").fetchone()[0] == digest
    assert source_generation_census(conn, "frozen")["retired_raw_members"] == 1
    with pytest.raises(ValueError, match="retired; readmission is forbidden"):
        record_source_item_raw_member(
            conn,
            source_generation_id="frozen",
            source_item_id=item,
            record_coordinate="record:0",
            raw_id="raw",
            raw_blob_hash=b"r" * 32,
        )
    assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0


def test_frozen_manifest_retry_cannot_change_input_bytes() -> None:
    conn = _source()
    item = _frozen_item(conn)
    with pytest.raises(ValueError, match="input binding changed"):
        publish_source_generation(
            conn,
            source_generation_id="frozen",
            manifest_digest="a" * 64,
            addressing_mode="physical-file-v1",
            coordinates=("export.json",),
            input_blob_hashes={"export.json": b"x" * 32},
            enumeration_fingerprint="b" * 64,
            observed_at_ms=2,
        )
    with pytest.raises(ValueError, match="frozen source input blob"):
        transition_source_item(
            conn,
            source_generation_id="frozen",
            source_item_id=item,
            request_id="changed",
            disposition=AcquisitionDisposition.ADMITTED,
            outcome_code=IngestOutcome.SUCCESS,
            stage="acquire",
            observed_at_ms=2,
            blob_hash=b"x" * 32,
        )


def test_manifest_is_published_before_read_and_identity_is_generation_bound() -> None:
    conn = _source()
    digest = hashlib.sha256(b"manifest").hexdigest()
    ids = publish_source_generation(
        conn,
        source_generation_id="g1",
        manifest_digest=digest,
        addressing_mode="zip-member",
        coordinates=("export.zip:a.json", "export.zip:b.json"),
        observed_at_ms=1,
    )
    assert len(ids) == 2
    assert conn.execute("SELECT COUNT(*) FROM source_items WHERE disposition='pending'").fetchone()[0] == 2
    assert source_item_id(source_generation_id="g1", logical_coordinate="x", addressing_mode="path") != source_item_id(
        source_generation_id="g2", logical_coordinate="x", addressing_mode="path"
    )


def test_manifest_rejects_unknown_origin_before_persisting_it() -> None:
    conn = _source()

    with pytest.raises(ValueError, match="origin must be one of"):
        publish_source_generation(
            conn,
            source_generation_id="invalid-origin",
            manifest_digest="d" * 64,
            addressing_mode="path",
            coordinates=("a.json",),
            observed_at_ms=1,
            origin="not-an-origin",
        )

    assert conn.execute("SELECT COUNT(*) FROM source_generations").fetchone()[0] == 0


def test_transition_is_idempotent_and_census_blocks_missing_or_admitted_without_raw() -> None:
    conn = _source()
    publish_source_generation(
        conn,
        source_generation_id="g1",
        manifest_digest="a" * 64,
        addressing_mode="path",
        coordinates=("a.json", "b.json"),
        observed_at_ms=1,
    )
    item = source_item_id(source_generation_id="g1", logical_coordinate="a.json", addressing_mode="path")
    assert (
        transition_source_item(
            conn,
            source_generation_id="g1",
            source_item_id=item,
            request_id="r1",
            disposition=AcquisitionDisposition.ADMITTED,
            outcome_code=IngestOutcome.SUCCESS,
            stage="raw_admission",
            observed_at_ms=2,
        )
        == 1
    )
    assert (
        transition_source_item(
            conn,
            source_generation_id="g1",
            source_item_id=item,
            request_id="r1",
            disposition=AcquisitionDisposition.ADMITTED,
            outcome_code=IngestOutcome.SUCCESS,
            stage="raw_admission",
            observed_at_ms=3,
        )
        == 1
    )
    census = source_generation_census(conn, "g1")
    assert census["missing"] == 0
    assert census["pending"] == 1
    assert census["admitted_without_raw"] == 1
    assert census["sealable"] is False


def test_mixed_batch_remains_structurally_mixed() -> None:
    conn = _source()
    publish_source_generation(
        conn,
        source_generation_id="g1",
        manifest_digest="b" * 64,
        addressing_mode="jsonl-record",
        coordinates=("x:0", "x:1"),
        observed_at_ms=1,
    )
    for coordinate, disposition, outcome in (
        ("x:0", AcquisitionDisposition.ADMITTED, IngestOutcome.SUCCESS),
        ("x:1", AcquisitionDisposition.CORRUPT, IngestOutcome.CORRUPT_INPUT),
    ):
        transition_source_item(
            conn,
            source_generation_id="g1",
            source_item_id=source_item_id(
                source_generation_id="g1", logical_coordinate=coordinate, addressing_mode="jsonl-record"
            ),
            request_id=coordinate,
            disposition=disposition,
            outcome_code=outcome,
            stage="decode",
            observed_at_ms=2,
        )
    census = source_generation_census(conn, "g1")
    assert census["admitted"] == 1
    assert census["deliberate"] == 1
    assert census["sealable"] is False


def test_seal_requires_every_item_and_payload_backing() -> None:
    conn = _source()
    publish_source_generation(
        conn,
        source_generation_id="g1",
        manifest_digest="c" * 64,
        addressing_mode="path",
        coordinates=("empty.txt",),
        observed_at_ms=1,
    )
    item = source_item_id(source_generation_id="g1", logical_coordinate="empty.txt", addressing_mode="path")
    transition_source_item(
        conn,
        source_generation_id="g1",
        source_item_id=item,
        request_id="r1",
        disposition=AcquisitionDisposition.EMPTY,
        outcome_code=IngestOutcome.UNSUPPORTED_SHAPE,
        stage="detect",
        observed_at_ms=2,
    )
    seal_source_generation(conn, source_generation_id="g1", sealed_at_ms=3)
    assert conn.execute("SELECT sealed_at_ms FROM source_generations WHERE source_generation_id='g1'").fetchone() == (
        3,
    )


def test_manifest_and_transition_participate_in_caller_transaction() -> None:
    """An internal attachment/item commit would strand authority after rollback."""
    conn = _source()
    conn.execute("BEGIN")
    (item,) = publish_source_generation(
        conn,
        source_generation_id="atomic",
        manifest_digest="a" * 64,
        addressing_mode="path",
        coordinates=("empty.json",),
        observed_at_ms=1,
        commit=False,
    )
    transition_source_item(
        conn,
        source_generation_id="atomic",
        source_item_id=item,
        request_id="empty-detected",
        disposition=AcquisitionDisposition.EMPTY,
        outcome_code=IngestOutcome.UNSUPPORTED_SHAPE,
        stage="detect",
        observed_at_ms=2,
        expected_revision=0,
        commit=False,
    )
    seal_source_generation(conn, source_generation_id="atomic", sealed_at_ms=3, commit=False)
    assert conn.in_transaction
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM source_generations").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM source_items").fetchone()[0] == 0


def test_stale_item_transition_cannot_overwrite_a_newer_observation() -> None:
    """Removing the revision check admits an out-of-order acquisition result."""
    conn = _source()
    (item,) = publish_source_generation(
        conn,
        source_generation_id="ordered",
        manifest_digest="a" * 64,
        addressing_mode="path",
        coordinates=("empty.json",),
        observed_at_ms=1,
    )
    args = {
        "source_generation_id": "ordered",
        "source_item_id": item,
        "disposition": AcquisitionDisposition.EMPTY,
        "outcome_code": IngestOutcome.UNSUPPORTED_SHAPE,
        "stage": "detect",
        "observed_at_ms": 2,
    }
    assert transition_source_item(conn, request_id="newer", expected_revision=0, **args) == 1
    assert transition_source_item(conn, request_id="newer", expected_revision=0, **args) == 1
    with pytest.raises(ValueError, match="revision changed"):
        transition_source_item(conn, request_id="older", expected_revision=0, **args)
    assert conn.execute("SELECT revision, request_id FROM source_items").fetchone() == (1, "newer")
