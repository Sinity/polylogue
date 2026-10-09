"""Library state filters and HTTP rows use the same physical Source CAS proof."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.api import Polylogue
from polylogue.archive.query.transaction import archive_read_context
from polylogue.core.enums import Provider
from polylogue.daemon.http import DaemonAPIHandler
from polylogue.daemon.webui_data import attachment_to_envelope
from polylogue.operations.http_read_models import read_attachment_library_page
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.storage_records import seed_attachment_library_lineage_archive


def _seed(root: Path) -> dict[str, str]:
    ids = seed_attachment_library_lineage_archive(root)
    payload = b'{"neutral":"physical Source bytes"}'
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="neutral.json",
            canonical_source_path="neutral.json",
            acquired_at_ms=1,
        )
        archive.commit()
    with (
        write_lease("test.attachment-library-physical-state", archive_root=root),
        ArchiveStore.open_existing(root, read_only=False) as archive,
    ):
        archive._conn.execute(
            "UPDATE attachments SET blob_hash=?, acquisition_status='acquired', media_type='text/plain'",
            (hashlib.sha256(payload).digest(),),
        )
        archive._conn.execute(
            "UPDATE attachments SET blob_hash=? WHERE display_name='own.txt'",
            (hashlib.sha256(b"never published").digest(),),
        )
        archive._conn.execute("UPDATE attachments SET media_type='application/zip' WHERE display_name='post-cut.txt'")
        archive._conn.commit()
    return ids


@pytest.mark.asyncio
async def test_library_filters_physical_states_before_paging_and_http_count(tmp_path: Path) -> None:
    """Hash presence alone admits missing own.txt before real available prefix.txt.

    Restoring the SQL hash-only predicate either emits missing bytes as available,
    loses the real Source CAS positive row, or cuts pages before state filtering.
    """
    ids = run_off_event_loop(lambda: _seed(tmp_path))
    handler = cast(Any, DaemonAPIHandler.__new__(DaemonAPIHandler))
    async with Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db") as poly:
        expected = {
            "available": ["foreign.txt", "prefix.txt"],
            "missing-blob": ["own.txt"],
            "unsupported-kind": ["post-cut.txt"],
        }
        for state, names in expected.items():
            emitted: list[str] = []
            for offset in range(len(names) + 1):
                payload = await handler._do_attachment_library(
                    poly, limit=1, offset=offset, mime_filter="", state_filter=state, session_filter=""
                )
                assert all(item["state"] == state for item in payload["items"])
                emitted.extend(item["name"] for item in payload["items"])
                if offset == len(names) - 1:
                    assert payload["total"] == len(names)
                    assert payload["total_is_exact"]
                with archive_read_context(
                    tmp_path, operation="http.archive.read", arguments={}, projection="http-read"
                ) as archive:
                    rows = read_attachment_library_page(
                        archive, limit=1, offset=offset, mime_filter="", state_filter=state, session_filter=""
                    )
                sync_items = [
                    attachment_to_envelope(att, session_id=str(att.session_id), message_id=att.message_id)
                    for att, _title, _origin in rows
                ]
                assert [(item["name"], item["state"]) for item in sync_items] == [
                    (item["name"], item["state"]) for item in payload["items"]
                ]
                for item in sync_items:
                    assert item["availability"] is not None
                    assert item["can_fetch"] == (state != "missing-blob")
            assert emitted == names
        scoped = await handler._do_attachment_library(
            poly, limit=1, offset=0, mime_filter="", state_filter="available", session_filter=ids["child"]
        )
        assert [(item["name"], item["state"]) for item in scoped["items"]] == [("prefix.txt", "available")]
        assert scoped["total"] == 1
