"""Original acquired and excised claims follow direct, cohort and aggregate carrier rows."""

from __future__ import annotations

import hashlib
import json
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.identity_law import attachment_acquisition_coordinate
from polylogue.core.stage_admission import admit_stage_write
from polylogue.pipeline.ids import session_id
from polylogue.sources import prepared_merge
from polylogue.sources.parsers.base import ParsedAttachment, ParsedSession
from polylogue.sources.prepared_jsonl import PreparedJsonl, PreparedSessionSequence, prepare_jsonl_blob
from polylogue.sources.prepared_merge import (
    aggregate_attachment_blobs,
    aggregate_resident_attachment_blobs,
    prepare_retained_cohort_artifact,
)
from polylogue.sources.prepared_message_sink import SqliteMessageStore
from polylogue.storage.blob_publication import ArchiveBlobPublisher, ConnectionBlobPublicationRead
from polylogue.storage.blob_store import BlobStore, PreparedBlob
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_write import record_excised_blob_hash, write_source_blob_refs
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.source_builders import ChatGPTExportBuilder


@contextmanager
def _owned_artifact(artifact: PreparedJsonl) -> Generator[PreparedJsonl, None, None]:
    try:
        yield artifact
    except BaseException as primary:
        try:
            artifact.discard()
        except BaseException as cleanup:
            raise BaseExceptionGroup("artifact operation and physical cleanup failed", [primary, cleanup]) from None
        raise
    else:
        artifact.discard()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    [
        "direct",
        "cohort",
        "aggregate",
        "aggregate_acquired",
        "aggregate_missing",
        "aggregate_crossraw",
        "aggregate_wrongordinal",
    ],
)
async def test_original_attachment_claims_follow_direct_cohort_and_aggregate_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    root = tmp_path / "archive"
    asset = b"neutral attachment bytes"
    asset_hash = hashlib.sha256(asset).hexdigest()
    payload = json.dumps(ChatGPTExportBuilder("excised-attachment").add_node("user", "neutral").build()).encode()

    second_payload = json.dumps(ChatGPTExportBuilder("excised-attachment").add_node("user", "second").build()).encode()

    def acquire() -> tuple[str, str]:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            archive.write_raw_payload(
                provider=Provider.UNKNOWN,
                payload=asset,
                source_path="neutral-asset.bin",
                canonical_source_path="neutral-asset.bin",
                acquired_at_ms=1,
            )
            raw_id = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path="neutral-session.json",
                canonical_source_path="neutral-session.json",
                acquired_at_ms=2,
            )
            second_raw_id = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=second_payload,
                source_path="neutral-second-session.json",
                canonical_source_path="neutral-second-session.json",
                acquired_at_ms=3,
            )
            if route != "aggregate_acquired":
                record_excised_blob_hash(
                    archive._ensure_source_conn(),
                    blob_hash=bytes.fromhex(asset_hash),
                    reason="neutral test excision",
                    actor="test.prepared-attachment",
                    excised_at_ms=3,
                )
            archive.commit()
            return raw_id, second_raw_id

    raw_id, second_raw_id = await run_archive_fixture_write(root, acquire)
    original_prepare = ArchiveBlobPublisher.prepare_from_path

    def refuse_asset_read(
        self: ArchiveBlobPublisher,
        path: Path,
        *,
        heartbeat: Callable[[], None] | None = None,
        staging_directory: Path | None = None,
    ) -> PreparedBlob:
        if route != "aggregate_acquired" and Path(path) == self.blob_path(asset_hash):
            raise AssertionError("originally excised attachment must not be reacquired")
        return original_prepare(self, path, heartbeat=heartbeat, staging_directory=staging_directory)

    monkeypatch.setattr(ArchiveBlobPublisher, "prepare_from_path", refuse_asset_read)

    def attach(session: ParsedSession) -> ParsedSession:
        return session.model_copy(
            update={
                "attachments": [
                    ParsedAttachment(
                        provider_attachment_id="attachment",
                        name="neutral.bin",
                        size_bytes=len(asset),
                        precomputed_blob=(asset_hash, len(asset)),
                    )
                ]
            }
        )

    def cohort(sessions: PreparedSessionSequence) -> Iterable[ParsedSession]:
        return (attach(session) for session in sessions)

    if route in {"aggregate_missing", "aggregate_crossraw", "aggregate_wrongordinal"}:
        original_merge = prepared_merge._merge_into_store

        def corrupt_origin(
            ordered: Sequence[tuple[str, PreparedJsonl]], merged: ParsedSession, store: SqliteMessageStore
        ) -> ParsedSession:
            result = original_merge(ordered, merged, store)
            if route == "aggregate_missing":
                store.conn.execute("DELETE FROM prepared_attachment_origin WHERE attachment_ordinal=0")
            elif route == "aggregate_crossraw":
                store.conn.execute(
                    "UPDATE prepared_attachment_origin SET raw_id='not-an-accepted-raw' WHERE attachment_ordinal=0"
                )
            else:
                store.conn.execute(
                    "UPDATE prepared_attachment_origin SET original_attachment_ordinal=999 WHERE attachment_ordinal=0"
                )
            return result

        monkeypatch.setattr(prepared_merge, "_merge_into_store", corrupt_origin)

    async with prepared_live_convergence_owner(root) as owner:

        def prepare() -> None:
            with (
                PreparedIndexMutation.source_only(archive_root=root) as seal,
                ExitStack() as original_reads,
            ):
                original_reads.enter_context(seal.original_read_snapshot())
                original_reads.enter_context(seal.source_producer())
                reader = PreparedSessionSourceRead(seal, blob_store=BlobStore(root / "blob"))
                descriptor = reader.raw_revision_descriptor(raw_id)
                original = reader.raw_revision_blob_path(raw_id)
                assert original is not None
                publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
                directory = publisher._prepared_staging_directory(None) / "attachment-control"
                with ExitStack() as stack:
                    artifact = stack.enter_context(
                        _owned_artifact(
                            prepare_jsonl_blob(
                                str(original),
                                descriptor[2],
                                Provider.CHATGPT.value,
                                "excised-attachment",
                                is_stream=False,
                                shard_directory=str(directory),
                                publication_publisher=publisher,
                                publication_source_read=reader,
                                prepare_session=attach if route != "cohort" else None,
                                prepare_sessions=cohort if route == "cohort" else None,
                            )
                        )
                    )
                    originals = {raw_id: artifact}
                    source_paths = {raw_id: descriptor[2]}
                    observed_times = {raw_id: reader.raw_revision_observation_order(raw_id)[0]}
                    ordered_ids: tuple[str, ...] = (raw_id,)
                    original_key = artifact.session_sequence()[0].attachments[0].acquisition_key
                    if route.startswith("aggregate"):
                        second_descriptor = reader.raw_revision_descriptor(second_raw_id)
                        second_path = reader.raw_revision_blob_path(second_raw_id)
                        assert second_path is not None
                        second = stack.enter_context(
                            _owned_artifact(
                                prepare_jsonl_blob(
                                    str(second_path),
                                    second_descriptor[2],
                                    Provider.CHATGPT.value,
                                    "excised-attachment",
                                    is_stream=False,
                                    shard_directory=str(directory),
                                    publication_publisher=publisher,
                                    publication_source_read=reader,
                                    prepare_session=attach,
                                )
                            )
                        )
                        originals[second_raw_id] = second
                        source_paths[second_raw_id] = second_descriptor[2]
                        observed_times[second_raw_id] = reader.raw_revision_observation_order(second_raw_id)[0]
                        ordered_ids = (raw_id, second_raw_id)
                        artifact = stack.enter_context(
                            _owned_artifact(
                                prepare_retained_cohort_artifact(
                                    [(raw_id, originals[raw_id]), (second_raw_id, second)],
                                    directory,
                                )
                            )
                        )
                    assert artifact.error is None
                    sessions = artifact.session_sequence()
                    assert len(sessions) == 1
                    retained = sessions[0]
                    assert len(retained.attachments) == (2 if route.startswith("aggregate") else 1)
                    sid = str(session_id(retained.source_name, retained.provider_session_id))
                    if route == "aggregate_acquired":
                        # Finish the exact original read window before checking
                        # Source currency for publication on the same seal.
                        original_reads.close()
                        # Claim preparation is off-writer. Only original admitted
                        # publication creates the reservation consumed by Source.
                        for original_artifact in originals.values():
                            original_artifact.publish_blobs(reference_seal=reader._seal)

                        def validate_and_consume() -> None:
                            with ArchiveStore.open_existing(root, read_only=False) as archive:
                                source = archive._ensure_source_conn()
                                publication_read = ConnectionBlobPublicationRead(source)
                                mapping = aggregate_attachment_blobs(
                                    artifact,
                                    source_read=publication_read,
                                    session_id=sid,
                                    original_artifacts=originals,
                                    accepted_raw_ids=ordered_ids,
                                )
                                with connection_cursor(
                                    source, "SELECT COUNT(*) FROM blob_publication_reservations"
                                ) as rows:
                                    before = int(rows.fetchone()[0])
                                assert before == 2
                                for attachment in retained.attachments:
                                    assert attachment.precomputed_blob == (asset_hash, len(asset))
                                    assert mapping[attachment.acquisition_key] == (
                                        bytes.fromhex(asset_hash),
                                        len(asset),
                                        "acquired",
                                    )
                                assert tuple(mapping) == tuple(item.acquisition_key for item in retained.attachments)
                                with connection_cursor(
                                    source, "SELECT COUNT(*) FROM blob_publication_reservations"
                                ) as rows:
                                    assert rows.fetchone()[0] == before
                                # Match production: validate while the exact claim
                                # lives, then consume its receipt with the referent.
                                for original_raw_id, original_artifact in originals.items():
                                    write_source_blob_refs(
                                        source,
                                        original_raw_id,
                                        partial(
                                            original_artifact.iter_attachment_refs,
                                            source_path=source_paths[original_raw_id],
                                            acquired_at_ms=observed_times[original_raw_id],
                                            source_read=publication_read,
                                        ),
                                    )
                                with connection_cursor(
                                    source, "SELECT COUNT(*) FROM blob_publication_reservations"
                                ) as rows:
                                    assert rows.fetchone()[0] == 0
                                with connection_cursor(
                                    source,
                                    "SELECT COUNT(*) FROM blob_refs WHERE ref_type='attachment' AND blob_hash=?",
                                    (bytes.fromhex(asset_hash),),
                                ) as rows:
                                    assert rows.fetchone()[0] == 2
                                resident = aggregate_resident_attachment_blobs(
                                    artifact,
                                    source_read=publication_read,
                                    session_id=sid,
                                    original_artifacts=originals,
                                    accepted_raw_ids=ordered_ids,
                                )
                                for attachment in retained.attachments:
                                    assert resident[attachment.acquisition_key] == (
                                        bytes.fromhex(asset_hash),
                                        len(asset),
                                        "acquired",
                                    )
                                assert tuple(resident) == tuple(a.acquisition_key for a in retained.attachments)
                                first_original = originals[raw_id]
                                assert first_original.blob_hash is not None
                                original_claim_key = next(
                                    iter(
                                        first_original.resident_attachment_blobs(
                                            source_read=publication_read, session_id=sid, raw_id=raw_id
                                        )
                                    )
                                )
                                with pytest.raises(ValueError):
                                    first_original.resident_attachment_blobs(
                                        source_read=publication_read, session_id=sid, raw_id="not-an-acquired-raw"
                                    )[original_claim_key]
                                wrong_source = replace(first_original, blob_hash="00" * 32)
                                with pytest.raises(ValueError):
                                    wrong_source.resident_attachment_blobs(
                                        source_read=publication_read, session_id=sid, raw_id=raw_id
                                    )[original_claim_key]
                                publication_read.retained_attachment_reference(
                                    raw_id,
                                    bytes.fromhex(first_original.blob_hash),
                                    attachment_acquisition_coordinate(None, "attachment"),
                                    bytes.fromhex(asset_hash),
                                    len(asset),
                                )
                                with pytest.raises(ValueError):
                                    publication_read.retained_attachment_reference(
                                        raw_id,
                                        bytes.fromhex(first_original.blob_hash),
                                        "not-the-original-coordinate",
                                        bytes.fromhex(asset_hash),
                                        len(asset),
                                    )
                                with connection_cursor(
                                    source, "SELECT COUNT(*) FROM blob_publication_reservations"
                                ) as rows:
                                    assert rows.fetchone()[0] == 0
                                with connection_cursor(
                                    source, "SELECT COUNT(*) FROM blob_refs WHERE ref_type='attachment'"
                                ) as rows:
                                    assert rows.fetchone()[0] == 2
                                archive.commit()

                        admit_stage_write("test.prepared-attachment.claim-consumption", validate_and_consume)
                        assert BlobStore(root / "blob").read_all(asset_hash) == asset
                        return
                    with reader._seal.original_rows(
                        "source", "SELECT COUNT(*) FROM blob_publication_reservations"
                    ) as rows:
                        reservation_count = int(rows.fetchone()[0])
                    if route == "aggregate":
                        resident = aggregate_resident_attachment_blobs(
                            artifact,
                            source_read=reader,
                            session_id=sid,
                            original_artifacts=originals,
                            accepted_raw_ids=ordered_ids,
                        )
                        for attachment in retained.attachments:
                            assert resident[attachment.acquisition_key] == (None, len(asset), "unavailable")
                    elif route in {"direct", "cohort"}:
                        resident = artifact.resident_attachment_blobs(source_read=reader, session_id=sid, raw_id=raw_id)
                        for attachment in retained.attachments:
                            assert resident[attachment.acquisition_key] == (None, len(asset), "unavailable")
                    if route.startswith("aggregate"):
                        mapping = aggregate_attachment_blobs(
                            artifact,
                            source_read=reader,
                            session_id=sid,
                            original_artifacts=originals,
                            accepted_raw_ids=ordered_ids,
                        )
                        if route not in {"aggregate", "aggregate_acquired"}:
                            with pytest.raises(ValueError):
                                mapping[retained.attachments[0].acquisition_key]
                            assert BlobStore(root / "blob").read_all(asset_hash) == asset
                            return
                        with pytest.raises(ValueError):
                            aggregate_attachment_blobs(
                                artifact,
                                source_read=reader,
                                session_id=sid,
                                original_artifacts=originals,
                                accepted_raw_ids=tuple(reversed(ordered_ids)),
                            )
                        with pytest.raises(KeyError):
                            mapping[original_key]
                        aggregate_key = retained.attachments[0].acquisition_key
                        original_first = originals[raw_id]
                        del originals[raw_id]
                        try:
                            with pytest.raises(ValueError):
                                mapping.get(aggregate_key)
                            with pytest.raises(ValueError):
                                aggregate_attachment_blobs(
                                    artifact,
                                    source_read=reader,
                                    session_id=sid,
                                    original_artifacts=originals,
                                    accepted_raw_ids=ordered_ids,
                                )
                        finally:
                            originals[raw_id] = original_first
                        for invalid_key in (
                            (str(artifact.sessions_path), False, 0),
                            (str(artifact.sessions_path), 0, False),
                            (str(artifact.sessions_path), 0.0, 0),
                            (str(artifact.sessions_path), 0, 0.0),
                        ):
                            with pytest.raises(KeyError):
                                mapping[invalid_key]
                        originals[raw_id] = originals[second_raw_id]
                        try:
                            with pytest.raises(ValueError):
                                mapping[aggregate_key]
                        finally:
                            originals[raw_id] = original_first
                        replaced = dict(originals)
                        replaced[raw_id] = originals[second_raw_id]
                        with pytest.raises(ValueError):
                            aggregate_attachment_blobs(
                                artifact,
                                source_read=reader,
                                session_id=sid,
                                original_artifacts=replaced,
                                accepted_raw_ids=ordered_ids,
                            )
                    else:
                        mapping = artifact.attachment_blobs(source_read=reader, session_id=sid)
                    for attachment in retained.attachments:
                        assert attachment.precomputed_blob == (asset_hash, len(asset))
                        assert attachment.inline_bytes is None
                        assert mapping[attachment.acquisition_key] == (
                            (bytes.fromhex(asset_hash), len(asset), "acquired")
                            if route == "aggregate_acquired"
                            else (None, len(asset), "unavailable")
                        )
                    assert tuple(artifact.iter_attachment_claims()) == ()
                    assert tuple(mapping) == tuple(attachment.acquisition_key for attachment in retained.attachments)
                    with reader._seal.original_rows(
                        "source", "SELECT COUNT(*) FROM blob_publication_reservations"
                    ) as rows:
                        assert rows.fetchone()[0] == reservation_count
                    with reader._seal.original_rows(
                        "source",
                        "SELECT COUNT(*) FROM blob_refs WHERE ref_type='attachment' AND blob_hash=?",
                        (bytes.fromhex(asset_hash),),
                    ) as rows:
                        assert rows.fetchone()[0] == 0
                    assert BlobStore(root / "blob").read_all(asset_hash) == asset

        await owner.run_prepared_sync(
            "test.prepared-attachment.excised", prepare, settlement_owners=lambda: (), estimated_bytes=len(payload)
        )
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        source = archive._ensure_source_conn()
        with connection_cursor(source, "SELECT COUNT(*) FROM blob_publication_reservations") as rows:
            assert rows.fetchone()[0] == 0
        with connection_cursor(
            source,
            "SELECT COUNT(*) FROM blob_refs WHERE ref_type='attachment' AND blob_hash=?",
            (bytes.fromhex(asset_hash),),
        ) as rows:
            assert rows.fetchone()[0] == (2 if route == "aggregate_acquired" else 0)
    assert BlobStore(root / "blob").read_all(asset_hash) == asset
