"""Production source-tier source-item authority laws."""

import hashlib
import sqlite3

import pytest

from polylogue.core.enums import IngestOutcome
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.source_items import (
    AcquisitionDisposition,
    publish_source_generation,
    seal_source_generation,
    source_generation_census,
    source_item_id,
    transition_source_item,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _source() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    initialize_archive_tier(conn, ArchiveTier.SOURCE)
    return conn


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
