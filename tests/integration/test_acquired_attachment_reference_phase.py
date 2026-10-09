"""One acquired attachment claim retains each exact original Raw reference."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import pytest

from polylogue.archive.revision_authority import RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.sources.prepared_jsonl import prepare_jsonl_blob
from polylogue.sources.revision_backfill import PreparedRetainedInput
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.derived.raw import _publish_acquired_attachment_refs
from polylogue.storage.raw_authority import raw_authority_parser_fingerprint
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.live_provider_proof import native_proof_artifact


@pytest.mark.asyncio
async def test_one_original_attachment_claim_retains_two_acquired_raw_references(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    envelope, _messages, attachment_count = native_proof_artifact(
        tmp_path, "native-inline-attachment-v1.json", Provider.GROK
    )
    payload = json.dumps(envelope).encode()
    digest = hashlib.sha256(payload).hexdigest()

    def acquire() -> tuple[str, str]:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            ids = tuple(
                archive.write_raw_payload(
                    provider=Provider.GROK,
                    payload=payload,
                    source_path=f"capture-{index}.json",
                    canonical_source_path=f"capture-{index}.json",
                    acquired_at_ms=index + 1,
                )
                for index in range(2)
            )
            assert len(ids) == 2 and ids[0] != ids[1]
            return ids[0], ids[1]

    raw_ids = await run_archive_fixture_write(root, acquire)
    async with prepared_live_convergence_owner(root) as owner:

        def prepare_and_publish() -> None:
            with PreparedIndexMutation.source_only(archive_root=root) as seal, ExitStack() as reads:
                reads.enter_context(seal.original_read_snapshot())
                reads.enter_context(seal.source_producer())
                reader = PreparedSessionSourceRead(seal, blob_store=BlobStore(root / "blob"))
                path = reader.raw_revision_blob_path(raw_ids[0])
                assert path is not None
                descriptor = reader.raw_revision_descriptor(raw_ids[0])
                publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
                artifact = prepare_jsonl_blob(
                    str(path),
                    descriptor[2],
                    Provider.GROK.value,
                    "native-conversation",
                    is_stream=False,
                    shard_directory=str(publisher._prepared_staging_directory(None) / "shared-claim"),
                    publication_publisher=publisher,
                    publication_source_read=reader,
                )
                seal.retain_preparation_payload(artifact.discard)
                assert artifact.error is None
                stat = path.stat()
                binding = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                inputs = {
                    raw_id: PreparedRetainedInput(
                        raw_id,
                        Provider.GROK,
                        digest,
                        reader.raw_revision_descriptor(raw_id)[2],
                        RawRevisionKind.FULL,
                        len(payload),
                        None,
                        raw_authority_parser_fingerprint(),
                        None,
                        binding,
                        validation_verdict=artifact.validation_verdict,
                        prepared_artifact=artifact,
                    )
                    for raw_id in raw_ids
                }
                reads.close()
                artifact.publish_blobs(reference_seal=seal)
                _publish_acquired_attachment_refs(seal, inputs, blob_store=BlobStore(root / "blob"))

        await owner.run_prepared_sync(
            "test.shared-acquired-claim",
            prepare_and_publish,
            settlement_owners=lambda: (),
            estimated_bytes=len(payload) * 2,
        )
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        source = archive.source_connection
        assert source is not None
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0
        refs = source.execute(
            "SELECT ref_id,source_path,blob_hash,size_bytes,acquired_at_ms FROM blob_refs WHERE ref_type='attachment' ORDER BY ref_id,source_path"
        ).fetchall()
        assert len(refs) == 2 * attachment_count
        first = [(row[1], row[2], row[3]) for row in refs if row[0] == raw_ids[0]]
        second = [(row[1], row[2], row[3]) for row in refs if row[0] == raw_ids[1]]
        assert first == second and len(first) == attachment_count
        assert {row[4] for row in refs if row[0] == raw_ids[0]} == {1}
        assert {row[4] for row in refs if row[0] == raw_ids[1]} == {2}


@pytest.mark.asyncio
async def test_incomplete_attachment_document_never_consumes_unpublished_claims(tmp_path: Path) -> None:
    from polylogue.core.raw_failure_evidence import RetainedRawDecodeRefusalError
    from polylogue.storage.derived.raw import RawObservationReplacement
    from tests.infra.retained_jsonl import run_retained_source_phase

    root = tmp_path / "archive"
    envelope, _messages, _attachments = native_proof_artifact(
        tmp_path, "native-inline-attachment-v1.json", Provider.GROK
    )
    payload = json.dumps(envelope).encode() + b"\n{\n"
    digest = hashlib.sha256(payload).digest()

    def acquire() -> str:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.GROK,
                payload=payload,
                source_path="incomplete-capture.jsonl",
                canonical_source_path="incomplete-capture.jsonl",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(root, acquire)
    observed: list[str] = []

    def inspect_original(replacement: RawObservationReplacement) -> None:
        assert replacement.prepared_inputs is not None
        original = replacement.prepared_inputs[raw_id]
        assert original.parser_error is not None
        assert original.prepared_artifact is None
        observed.append(raw_id)

    with pytest.raises(RetainedRawDecodeRefusalError):
        await run_retained_source_phase(root, (raw_id,), before_publication=inspect_original)
    assert observed == [raw_id]
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        source = archive.source_connection
        assert source is not None
        assert source.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone() == (digest,)
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0
        assert source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type='attachment'").fetchone()[0] == 0
        assert BlobStore(root / "blob").read_all(digest.hex()) == payload
