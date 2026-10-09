"""Attachment records expose stable source references beside content versions."""

from __future__ import annotations

import aiosqlite
import pytest

from polylogue.storage.hydrators import attachment_from_record
from polylogue.storage.sqlite.queries.attachment_records import get_attachments


@pytest.mark.asyncio
async def test_attachment_record_keeps_stable_reference_and_content_version_separate() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await conn.executescript(
            "CREATE TABLE attachments(attachment_id TEXT, media_type TEXT, byte_count INTEGER, "
            "blob_hash BLOB, acquisition_status TEXT, display_name TEXT);"
            "CREATE TABLE attachment_refs(ref_id TEXT, attachment_id TEXT, session_id TEXT, message_id TEXT, "
            "position INTEGER, upload_origin TEXT, direction TEXT, producer_ref TEXT, source_url TEXT, caption TEXT, "
            "supplying_raw_id TEXT);"
            "CREATE TABLE attachment_native_ids(ref_id TEXT, id_kind TEXT, native_id TEXT);"
            "INSERT INTO attachments VALUES ('content-version-2', 'image/png', 4, NULL, 'unfetched', 'image.png');"
            "INSERT INTO attachment_refs VALUES ('message-a:attachment:0', 'content-version-2', 'session-a', "
            "'message-a', 0, NULL, 'user_input', NULL, NULL, NULL, 'raw-a');"
        )

        records = await get_attachments(conn, "session-a")

    assert len(records) == 1
    assert records[0].reference_id == "message-a:attachment:0"
    assert records[0].supplying_raw_id == "raw-a"
    domain_attachment = attachment_from_record(records[0])
    assert domain_attachment.id == "content-version-2"
    assert domain_attachment.reference_id == "message-a:attachment:0"
