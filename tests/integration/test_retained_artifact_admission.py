"""Typed artifact acquisition and parser receipts use their actual owners."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.artifact_taxonomy import classify_artifact_path
from polylogue.core.enums import Provider
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["payload", "blob-ref"])
async def test_typed_artifact_acquisition_leaves_parser_receipt_to_original_retained_owner(
    tmp_path: Path,
    route: str,
) -> None:
    root = tmp_path / "archive"
    source_path = str(tmp_path / ".claude/projects/neutral/subagents/workflows/wf-neutral/journal.jsonl")
    payload = b'{"contentKey":"neutral-artifact","agentId":"neutral-agent"}\n'

    def acquire() -> str:
        bootstrap_archive_root(root)
        classification = classify_artifact_path(source_path, provider=Provider.CLAUDE_CODE)
        assert classification is not None and not classification.parse_as_session
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            if route == "payload":
                admitted = archive.admit_raw_artifact_payload(
                    provider=Provider.CLAUDE_CODE,
                    payload=payload,
                    source_path=source_path,
                    canonical_source_path=source_path,
                    acquired_at_ms=1,
                    classification=classification,
                )
            else:
                publisher = archive._blob_publisher
                assert publisher is not None
                digest, size = publisher.write_from_bytes(payload)
                receipt_id = publisher.receipt_id(digest)
                publisher.flush()
                admitted = archive.admit_raw_artifact_blob_ref(
                    provider=Provider.CLAUDE_CODE,
                    blob_hash_hex=digest,
                    blob_size=size,
                    source_path=source_path,
                    canonical_source_path=source_path,
                    acquired_at_ms=1,
                    classification=classification,
                    blob_publication_receipt_id=receipt_id,
                )
            source = archive._ensure_source_conn()
            with connection_cursor(source, "SELECT COUNT(*) FROM raw_authority_parser_census") as rows:
                assert rows.fetchone()[0] == 0
            with connection_cursor(source, "SELECT raw_id, parse_as_session FROM raw_artifacts") as rows:
                assert [tuple(row) for row in rows] == [(admitted.raw_id, 0)]
            return admitted.raw_id

    raw_id = await run_archive_fixture_write(root, acquire)
    async with prepared_live_convergence_owner(root) as owner:
        receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert sum(len(receipt.written_session_ids) for receipt in receipts) == 0
        assert sum(receipt.written_message_count for receipt in receipts) == 0
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        source = archive._ensure_source_conn()
        with connection_cursor(
            source, "SELECT status FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)
        ) as rows:
            receipt = rows.fetchone()
            assert receipt is not None and tuple(receipt) == ("complete",)
        with connection_cursor(source, "SELECT raw_id, parse_as_session FROM raw_artifacts") as rows:
            assert [tuple(row) for row in rows] == [(raw_id, 0)]
        index = archive.index_connection
        assert index is not None
        with connection_cursor(index, "SELECT COUNT(*) FROM sessions") as rows:
            assert rows.fetchone()[0] == 0
        with connection_cursor(index, "SELECT COUNT(*) FROM messages") as rows:
            assert rows.fetchone()[0] == 0
