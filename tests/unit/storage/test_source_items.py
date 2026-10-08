"""Production source-tier source-item authority laws."""

import asyncio
import hashlib
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import TypedDict

import pytest

from polylogue.core.enums import IngestOutcome
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate, MemberAddressingMode
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier, initialize_runtime_tier_probe
from polylogue.storage.sqlite.archive_tiers.source_attachments import SourceAttachment
from polylogue.storage.sqlite.archive_tiers.source_items import (
    AcquisitionDisposition,
    FrozenSourceInput,
    FrozenSourceManifest,
    SourceItemMemberDisposition,
    abort_prepared_source_manifest,
    append_prepared_source_inputs,
    begin_prepared_source_manifest,
    complete_source_item_enumeration,
    page_retained_source_inputs,
    prepare_source_manifest,
    publish_sealed_source_manifest,
    publish_source_generation,
    record_source_item_member_disposition,
    record_source_item_raw_member,
    seal_prepared_source_manifest,
    seal_source_generation,
    source_generation_census,
    source_item_id,
    transition_source_item,
)
from polylogue.storage.sqlite.archive_tiers.source_write import record_raw_container_coordinate
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


class _TransitionArgs(TypedDict):
    source_generation_id: str
    source_item_id: str
    disposition: AcquisitionDisposition
    outcome_code: IngestOutcome
    stage: str
    observed_at_ms: int


def _source(*, baseline: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    if baseline:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
    else:
        initialize_runtime_tier_probe(conn, ArchiveTier.SOURCE)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _reservation(conn: sqlite3.Connection, receipt_id: str, blob_hash: bytes) -> None:
    conn.execute(
        "INSERT INTO blob_publication_reservations VALUES (?, ?, 1, 'synthetic', 1)",
        (receipt_id, blob_hash),
    )


@pytest.mark.parametrize(
    "first_import",
    ["polylogue.security.excision_policy", "polylogue.storage.runtime", "polylogue.storage.sqlite.archive_tiers"],
)
def test_cold_archive_initialization_preserves_source_item_and_policy_models(first_import: str) -> None:
    """Cold archive loading must support raw admission and canonical policy identity."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    f"import {first_import}",
                    "from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER",
                    "from polylogue.storage.runtime import RawSessionRecord",
                    "from polylogue.storage.sqlite.archive_tiers.source_items import SourceItemAdmission",
                    "from polylogue.security.excision_policy import ExcisionPolicySnapshot",
                    "from polylogue.storage.sqlite.archive_tiers.bootstrap import archive_tier_spec",
                    "from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier",
                    "member = SourceItemAdmission('generation', 'item', 'record:0')",
                    "raw = RawSessionRecord(raw_id='raw', source_path='/synthetic/source.json', source_item=member, blob_size=0, acquired_at='2026-01-01T00:00:00Z')",
                    "assert raw.source_item == member",
                    "policy = ExcisionPolicySnapshot((), (), 0, 0, 'head', None)",
                    "assert policy.schema_identity == ';'.join(f'{tier.value}:{archive_tier_spec(tier).version}' for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT))",
                    "assert ArchiveTier.SOURCE in ARCHIVE_DDL_BY_TIER",
                )
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_fresh_source_v1_contains_prepared_manifest_tables() -> None:
    """Fresh campaign bootstrap carries the new relations without a version step."""
    conn = _source(baseline=True)
    assert conn.execute("PRAGMA user_version").fetchone() == (1,)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name LIKE 'prepared_source_manifest%'"
        )
    }
    assert tables == {"prepared_source_manifests", "prepared_source_manifest_members"}


def test_sealed_prepared_manifest_preserves_v1_digest_and_binds_custody() -> None:
    conn = _source()
    inputs = (
        FrozenSourceInput("a", "/tmp/α", "a" * 64, "receipt:a"),
        FrozenSourceInput("b", "/tmp/β", "b" * 64, "receipt:b"),
    )
    for item in inputs:
        _reservation(conn, item.publication_receipt_id, bytes.fromhex(item.blob_hash))
    conn.commit()
    conn.execute("BEGIN")
    begin_prepared_source_manifest(
        conn,
        source_generation_id="synthetic",
        publisher_id="synthetic-publisher",
        enumeration_fingerprint="d" * 64,
        source_name="codex",
    )
    append_prepared_source_inputs(conn, "synthetic", 0, inputs)
    ref = seal_prepared_source_manifest(conn, "synthetic", sealed_at_ms=1)
    assert ref.manifest_digest == FrozenSourceManifest("synthetic", "d" * 64, inputs, "codex").manifest_digest
    assert conn.execute("SELECT COUNT(*) FROM source_generations").fetchone()[0] == 0
    publish_sealed_source_manifest(conn, ref, prepared_at_ms=2)
    assert conn.execute("SELECT COUNT(*) FROM source_items").fetchone()[0] == 2
    publish_sealed_source_manifest(conn, ref, prepared_at_ms=2)
    assert conn.execute("SELECT COUNT(*) FROM source_items").fetchone()[0] == 2
    with pytest.raises(ValueError, match="accepted source generation cannot be aborted"):
        abort_prepared_source_manifest(conn, source_generation_id="synthetic", publisher_id="synthetic-publisher")
    assert conn.execute("SELECT COUNT(*) FROM source_items").fetchone()[0] == 2


def test_sealed_manifest_pages_beyond_old_total_bound() -> None:
    conn = _source()
    conn.execute("BEGIN")
    begin_prepared_source_manifest(
        conn,
        source_generation_id="large",
        publisher_id="large-publisher",
        enumeration_fingerprint="d" * 64,
        source_name=None,
    )
    for start in range(0, 10_241, 256):
        batch = tuple(
            FrozenSourceInput(f"input:{number:05d}", f"/synthetic/{number:05d}", "a" * 64, "shared")
            for number in range(start, min(start + 256, 10_241))
        )
        append_prepared_source_inputs(conn, "large", start, batch)
    _reservation(conn, "shared", bytes.fromhex("a" * 64))
    ref = seal_prepared_source_manifest(conn, "large", sealed_at_ms=1)
    assert ref.input_count == 10_241
    publish_sealed_source_manifest(conn, ref, prepared_at_ms=2)
    seen = 0
    cursor = None
    while page := page_retained_source_inputs(conn, "large", after=cursor):
        assert len(page) <= 256
        seen += len(page)
        cursor = (page[-1][0].coordinate, page[-1][0].source_item_id)
    assert seen == 10_241


def test_public_manifest_preparation_streams_more_than_ten_thousand_inputs() -> None:
    conn = _source()
    conn.execute("BEGIN")
    _reservation(conn, "shared", bytes.fromhex("a" * 64))
    ref = prepare_source_manifest(
        conn,
        source_generation_id="public-large",
        publisher_id="publisher",
        enumeration_fingerprint="d" * 64,
        source_name="codex",
        sealed_at_ms=1,
        inputs=(FrozenSourceInput(str(n), f"/synthetic/{n}", "a" * 64, "shared") for n in range(10_241)),
    )
    publish_sealed_source_manifest(conn, ref, prepared_at_ms=2)
    assert ref.input_count == 10_241
    assert conn.execute("SELECT COUNT(*) FROM source_items").fetchone() == (10_241,)


def test_completion_uses_disk_rows_and_same_uncommitted_source_membership(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.storage.sqlite.archive_tiers import source_items

    conn = _source()
    item = _frozen_item(conn)
    conn.execute("BEGIN")
    for number in range(4097):
        _raw_member(conn, item, f"record:{number:05d}")
    from polylogue.storage.sqlite.connection_profile import scratch_connection_context

    actual_factory = scratch_connection_context
    observed: list[tuple[str, int, int, str]] = []

    @contextmanager
    def observe_factory(*, prefix: str, filename: str) -> Iterator[sqlite3.Connection]:
        with actual_factory(prefix=prefix, filename=filename) as scratch:
            try:
                yield scratch
            finally:
                observed.append(
                    (
                        scratch.execute("PRAGMA journal_mode").fetchone()[0],
                        scratch.execute("SELECT COUNT(*) FROM main.records").fetchone()[0],
                        scratch.execute("SELECT COUNT(*) FROM sqlite_temp_schema").fetchone()[0],
                        scratch.execute("PRAGMA database_list").fetchone()[2],
                    )
                )

    monkeypatch.setattr(source_items, "scratch_connection_context", observe_factory)
    complete_source_item_enumeration(
        conn,
        source_generation_id="frozen",
        source_item_id=item,
        enumeration_fingerprint="b" * 64,
        record_coordinates=(f"record:{number:05d}" for number in range(4097)),
        enumerated_at_ms=2,
    )
    assert observed[0][:3] == ("delete", 4097, 0)
    assert observed[0][3]
    assert conn.in_transaction
    conn.rollback()
    assert source_generation_census(conn, "frozen")["enumeration_pending"] == 1


def test_completion_cancellation_cannot_publish_completion() -> None:
    conn = _source()
    item = _frozen_item(conn)
    conn.execute("BEGIN")
    _raw_member(conn, item, "record:0")

    class CancelledError(Exception):
        pass

    calls = 0

    def stop() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise CancelledError

    with pytest.raises(CancelledError):
        complete_source_item_enumeration(
            conn,
            source_generation_id="frozen",
            source_item_id=item,
            enumeration_fingerprint="b" * 64,
            record_coordinates=iter(("record:0",)),
            enumerated_at_ms=2,
            check_stop=stop,
        )
    assert source_generation_census(conn, "frozen")["enumeration_pending"] == 1
    conn.rollback()


@pytest.mark.asyncio
async def test_async_completion_cancellation_drains_worker_and_rolls_back() -> None:
    import aiosqlite

    from polylogue.storage.sqlite.queries.raw_writes import complete_acquired_zip_input

    async with aiosqlite.connect(":memory:") as owner:

        def initialize() -> str:
            initialize_runtime_tier_probe(owner._conn, ArchiveTier.SOURCE)
            owner._conn.execute("PRAGMA foreign_keys=ON")
            item = _frozen_item(owner._conn)
            owner._conn.execute("BEGIN")
            _raw_member(owner._conn, item, "record:0")
            return item

        execute: Callable[[Callable[[], str]], Awaitable[str]] = owner._execute
        item = await execute(initialize)
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def coordinates() -> Iterator[str]:
            loop.call_soon_threadsafe(entered.set)
            release.wait()
            yield "record:0"

        task = asyncio.create_task(
            complete_acquired_zip_input(
                owner,
                source_generation_id="frozen",
                source_item_id=item,
                enumeration_fingerprint="b" * 64,
                record_coordinates=coordinates(),
                member_count=1,
                observed_at_ms=2,
                transaction_depth=1,
            )
        )
        try:
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)  # Deliver cancellation to the owning adapter.
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with owner.execute("SELECT enumerated_at_ms FROM source_items") as cursor:
            assert await cursor.fetchone() == (None,)
        async with owner.execute("SELECT COUNT(*) FROM source_item_raw_members") as cursor:
            assert await cursor.fetchone() == (1,)
        await owner.rollback()


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
    assert source_generation_census(conn, "frozen")["enumeration_pending"] == 1


def test_transition_refuses_invalid_domain_vocabularies_before_idempotency_lookup() -> None:
    conn = _source()
    item = _frozen_item(conn)
    for field, value in (("disposition", "not-a-disposition"), ("outcome_code", "not-an-outcome")):
        args = {
            "source_generation_id": "frozen",
            "source_item_id": item,
            "request_id": field,
            "disposition": AcquisitionDisposition.ADMITTED,
            "outcome_code": IngestOutcome.SUCCESS,
            "stage": "test",
            "observed_at_ms": 2,
        }
        args[field] = value
        with pytest.raises(ValueError, match=field):
            transition_source_item(conn, **args)  # type: ignore[arg-type]
    assert tuple(conn.execute("SELECT disposition, outcome_code, revision FROM source_items").fetchone()) == (
        "pending",
        "interrupted",
        0,
    )


def test_manifest_preflights_attachment_vocabulary_before_publishing_generation() -> None:
    conn = _source()
    with pytest.raises(ValueError, match="origin"):
        publish_source_generation(
            conn,
            source_generation_id="invalid-attachments",
            manifest_digest="c" * 64,
            addressing_mode="physical-file-v1",
            coordinates=("export.json",),
            observed_at_ms=2,
            attachments=(SourceAttachment("attachment", "not-an-origin", "drive"),),
        )
    assert conn.execute("SELECT COUNT(*) FROM source_generations").fetchone()[0] == 0


def test_sql_bypass_can_store_an_unowned_source_item_origin() -> None:
    """The nullable origin column has no membership CHECK by design."""
    conn = _source()
    item = _frozen_item(conn)
    conn.execute(
        "UPDATE source_items SET origin = 'not-an-origin' WHERE source_generation_id='frozen' AND source_item_id=?",
        (item,),
    )
    assert (
        conn.execute("SELECT origin FROM source_items WHERE source_item_id=?", (item,)).fetchone()[0] == "not-an-origin"
    )


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


def test_zip_member_disposition_completes_full_denominator_and_retry_is_idempotent() -> None:
    """A refused central member is durable evidence, not an omitted record."""
    conn = _source()
    item = _frozen_item(conn)
    conn.execute("BEGIN")
    _raw_member(conn, item, "record:0")
    record_raw_container_coordinate(
        conn,
        "raw",
        coordinate_format="zip-v2",
        entry_ordinal=0,
        split_index=0,
        addressing_mode="whole_member",
        # Declared synthetic namespace; this control exercises the denominator,
        # not a claim that an external container was acquired.
        captured_coordinate=CapturedZipMemberCoordinate(
            "/fixture/export.zip",
            "/fixture/export.zip",
            "session.json",
            0,
            0,
            MemberAddressingMode.WHOLE_MEMBER,
            "ab" * 32,
            "cd" * 32,
        ),
        manage_transaction=False,
    )
    record_source_item_member_disposition(
        conn,
        source_generation_id="frozen",
        source_item_id=item,
        entry_ordinal=1,
        member_name="skipped.html",
        disposition=SourceItemMemberDisposition.UNSELECTED,
        diagnostic="not selected",
        observed_at_ms=2,
    )
    # A retry may have a new observation time but must not rewrite the
    # disposition or reject the already-recorded member.
    record_source_item_member_disposition(
        conn,
        source_generation_id="frozen",
        source_item_id=item,
        entry_ordinal=1,
        member_name="skipped.html",
        disposition=SourceItemMemberDisposition.UNSELECTED,
        diagnostic="not selected",
        observed_at_ms=3,
    )
    complete_source_item_enumeration(
        conn,
        source_generation_id="frozen",
        source_item_id=item,
        enumeration_fingerprint="b" * 64,
        record_coordinates=("record:0",),
        enumerated_at_ms=2,
        member_ordinals=(0, 1),
        member_count=2,
    )
    census = source_generation_census(conn, "frozen")
    assert census["member_unselected"] == 1
    assert census["member_dispositions"] == 1
    assert census["sealable"] is False


def test_member_disposition_preserves_exact_identity_and_bounds_diagnostic() -> None:
    conn = _source()
    item = _frozen_item(conn)
    conn.execute("BEGIN")
    record_source_item_member_disposition(
        conn,
        source_generation_id="frozen",
        source_item_id=item,
        entry_ordinal=0,
        member_name="n" * 20_000,
        disposition=SourceItemMemberDisposition.REFUSED,
        diagnostic="d" * 20_000,
        observed_at_ms=2,
    )
    name, diagnostic = conn.execute("SELECT member_name, diagnostic FROM source_item_member_dispositions").fetchone()
    assert name == "n" * 20_000
    assert len(diagnostic) <= 4096


def test_zip_enumeration_without_member_denominator_stays_incomplete() -> None:
    conn = _source()
    item = _frozen_item(conn)
    conn.execute("BEGIN")
    _raw_member(conn, item, '["zip-v2",0,0,"whole_member"]')
    record_raw_container_coordinate(
        conn,
        "raw",
        coordinate_format="zip-v2",
        entry_ordinal=0,
        split_index=0,
        addressing_mode="whole_member",
        # Declared synthetic namespace; this control exercises the denominator,
        # not a claim that an external container was acquired.
        captured_coordinate=CapturedZipMemberCoordinate(
            "/fixture/export.zip",
            "/fixture/export.zip",
            "session.json",
            0,
            0,
            MemberAddressingMode.WHOLE_MEMBER,
            "ab" * 32,
            "cd" * 32,
        ),
        manage_transaction=False,
    )
    with pytest.raises(ValueError, match="central-directory denominator"):
        complete_source_item_enumeration(
            conn,
            source_generation_id="frozen",
            source_item_id=item,
            enumeration_fingerprint="b" * 64,
            record_coordinates=('["zip-v2",0,0,"whole_member"]',),
            enumerated_at_ms=2,
        )


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
    args: _TransitionArgs = {
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


@pytest.mark.parametrize("baseline", [False, True])
@pytest.mark.parametrize("member_name", [" ", " " * 4096 + "member.jsonl"])
def test_member_disposition_preserves_exact_whitespace_zip_names(baseline: bool, member_name: str) -> None:
    from contextlib import closing

    with closing(_source(baseline=baseline)) as conn:
        if baseline:
            # Fresh DDL is tested with its own declared fixture columns, before
            # captured-input columns arrive through the immutable train.
            item = source_item_id(
                source_generation_id="frozen", logical_coordinate="export.json", addressing_mode="physical-file-v1"
            )
            conn.execute(
                "INSERT INTO source_generations(source_generation_id, manifest_digest, addressing_mode, "
                "item_count, created_at_ms) VALUES ('frozen', ?, 'physical-file-v1', 1, 1)",
                ("a" * 64,),
            )
            conn.execute(
                "INSERT INTO source_items(source_generation_id, source_item_id, logical_coordinate, "
                "addressing_mode, disposition, outcome_code, stage, observed_at_ms, updated_at_ms) "
                "VALUES ('frozen', ?, 'export.json', 'physical-file-v1', 'pending', 'interrupted', 'manifest', 1, 1)",
                (item,),
            )
        else:
            item = _frozen_item(conn)

        def record(name: str, ordinal: int = 0) -> None:
            record_source_item_member_disposition(
                conn,
                source_generation_id="frozen",
                source_item_id=item,
                entry_ordinal=ordinal,
                member_name=name,
                disposition=SourceItemMemberDisposition.UNSELECTED,
                diagnostic="not a declared artifact",
                observed_at_ms=2,
            )

        record(member_name)
        record(member_name)
        assert conn.execute("SELECT entry_ordinal, member_name FROM source_item_member_dispositions").fetchall() == [
            (0, member_name)
        ]
        with pytest.raises(ValueError):
            record("", 1)
        assert conn.execute("SELECT COUNT(*) FROM source_item_member_dispositions").fetchone() == (1,)


@pytest.mark.parametrize(
    "outcome, expected",
    [
        (IngestOutcome.SUCCESS, False),
        (IngestOutcome.VALIDATION_REJECTED, False),
        (IngestOutcome.UNSUPPORTED_SHAPE, False),
        (IngestOutcome.CORRUPT_INPUT, False),
        (IngestOutcome.TRANSIENT_ERROR, True),
        (IngestOutcome.PARSER_DEFECT, False),
        (IngestOutcome.DOWNSTREAM_FAILURE, True),
        (IngestOutcome.CANCELED, True),
        (IngestOutcome.INTERRUPTED, True),
        (IngestOutcome.LEGACY_UNKNOWN, None),
    ],
)
def test_source_item_default_retryability_preserves_typed_outcome(
    outcome: IngestOutcome, expected: bool | None
) -> None:
    from contextlib import closing

    with closing(_source()) as conn:
        item = _frozen_item(conn)
        for _ in range(2):
            assert (
                transition_source_item(
                    conn,
                    source_generation_id="frozen",
                    source_item_id=item,
                    request_id="outcome",
                    disposition=AcquisitionDisposition.UNKNOWN_BLOCKING,
                    outcome_code=outcome,
                    stage="observation",
                    observed_at_ms=2,
                )
                == 1
            )
        assert conn.execute("SELECT outcome_code, retryable, revision FROM source_items").fetchone() == (
            outcome.value,
            None if expected is None else int(expected),
            1,
        )


@pytest.mark.parametrize("retryable", [False, True])
def test_source_item_explicit_retryability_is_preserved(retryable: bool) -> None:
    from contextlib import closing

    with closing(_source()) as conn:
        item = _frozen_item(conn)
        transition_source_item(
            conn,
            source_generation_id="frozen",
            source_item_id=item,
            request_id="explicit",
            disposition=AcquisitionDisposition.UNKNOWN_BLOCKING,
            outcome_code=IngestOutcome.TRANSIENT_ERROR,
            stage="observation",
            observed_at_ms=2,
            retryable=retryable,
        )
        assert conn.execute("SELECT retryable FROM source_items").fetchone() == (int(retryable),)
