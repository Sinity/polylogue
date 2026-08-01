"""Attachment byte preservation (#2468): inline bytes are stored in the blob
store with the true SHA-256 and 'acquired' status; attachments with no recoverable
bytes are honestly marked 'unfetched' with a NULL hash (no fabricated hash).
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
)
from polylogue.sources.parsers.claude import parse_ai
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _session_with_attachment(attachment: ParsedAttachment) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.GEMINI,
        provider_session_id="s1",
        title="s1",
        messages=[
            ParsedMessage(
                provider_message_id="m0",
                role=Role.USER,
                text="here is a file",
                position=0,
                variant_index=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="here is a file")],
            )
        ],
        attachments=[attachment],
    )


def _preacquired(store: BlobStore, session: ParsedSession) -> dict[int, tuple[bytes | None, int, str]]:
    acquired: dict[int, tuple[bytes | None, int, str]] = {}
    for attachment in session.attachments:
        if attachment.inline_bytes is None:
            continue
        blob_hash, size = store.write_from_bytes(attachment.inline_bytes)
        acquired[id(attachment)] = (bytes.fromhex(blob_hash), size, "acquired")
    return acquired


def test_inline_attachment_bytes_are_stored_with_true_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = BlobStore(tmp_path / "blob")
    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)

    payload = b"hello attachment bytes"
    conn = _connect(tmp_path / "index.db")
    session = _session_with_attachment(
        ParsedAttachment(
            provider_attachment_id="att-1",
            message_provider_id="m0",
            name="note.txt",
            mime_type="text/plain",
            inline_bytes=payload,
        )
    )
    write_parsed_session_to_archive(conn, session, preacquired_attachment_blobs=_preacquired(store, session))

    row = conn.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchone()
    assert row["acquisition_status"] == "acquired"
    assert row["byte_count"] == len(payload)
    assert bytes(row["blob_hash"]) == hashlib.sha256(payload).digest()
    # The bytes actually round-trip through the content-addressed store.
    assert store.read_all(hashlib.sha256(payload).hexdigest()) == payload


def test_attachment_without_bytes_is_marked_unfetched_not_faked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BlobStore(tmp_path / "blob")
    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)

    conn = _connect(tmp_path / "index.db")
    write_parsed_session_to_archive(
        conn,
        _session_with_attachment(
            ParsedAttachment(
                provider_attachment_id="att-2",
                message_provider_id="m0",
                name="remote.pdf",
                mime_type="application/pdf",
                size_bytes=4096,
                source_url="https://drive.example/remote.pdf",
            )
        ),
    )

    row = conn.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchone()
    assert row["acquisition_status"] == "unfetched"
    assert row["blob_hash"] is None
    assert row["byte_count"] == 4096


def test_claude_extracted_attachment_content_is_acquired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = BlobStore(tmp_path / "blob")
    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)

    session = parse_ai(
        {
            "uuid": "claude-attachment-session",
            "name": "Attachment payload",
            "chat_messages": [
                {
                    "uuid": "m0",
                    "sender": "human",
                    "text": "Please read this.",
                    "attachments": [
                        {
                            "file_name": "notes.md",
                            "file_type": "text/markdown",
                            "file_size": 11,
                            "extracted_content": "hello notes",
                        }
                    ],
                    "files": [
                        {
                            "file_uuid": "remote-file-1",
                            "file_name": "remote.tar.gz",
                            "size_bytes": 1024,
                        }
                    ],
                }
            ],
        },
        "fallback",
    )
    conn = _connect(tmp_path / "index.db")

    write_parsed_session_to_archive(conn, session, preacquired_attachment_blobs=_preacquired(store, session))

    rows = conn.execute(
        """
        SELECT display_name, blob_hash, byte_count, acquisition_status
        FROM attachments
        ORDER BY display_name
        """
    ).fetchall()
    by_name = {str(row["display_name"]): row for row in rows}
    acquired = by_name["notes.md"]
    assert acquired["acquisition_status"] == "acquired"
    assert acquired["byte_count"] == len(b"hello notes")
    assert bytes(acquired["blob_hash"]) == hashlib.sha256(b"hello notes").digest()
    assert store.read_all(hashlib.sha256(b"hello notes").hexdigest()) == b"hello notes"

    remote = by_name["remote.tar.gz"]
    assert remote["acquisition_status"] == "unfetched"
    assert remote["blob_hash"] is None
    assert remote["byte_count"] == 1024


@pytest.mark.parametrize("payload", [b"must be reserved first", b""], ids=["nonempty", "empty"])
def test_low_level_writer_rejects_inline_bytes_without_preacquisition(tmp_path: Path, payload: bytes) -> None:
    conn = _connect(tmp_path / "index.db")
    session = _session_with_attachment(
        ParsedAttachment(
            provider_attachment_id="att-unreserved",
            message_provider_id="m0",
            inline_bytes=payload,
        )
    )

    with pytest.raises(ValueError, match="preacquired_attachment_blobs"):
        write_parsed_session_to_archive(conn, session)


@pytest.mark.asyncio
async def test_orphaned_attachment_ref_is_swept_not_left_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-w06b: a full-replace re-ingest whose attachment can no
    longer be matched to a message (its owning message dropped out of this
    ingest's message set) must not leave a ref-less ``attachments`` row
    behind. Before the fix, ``_write_attachments`` recomputed
    ``ref_count = 0`` for the stale attachment via the UPDATE it already
    runs, but never deleted the row -- unlike the identical
    ``prune_attachments``/``delete_session_sql`` cleanups -- so the
    attachment (real acquired bytes and all) became permanently unreachable
    from the production read path ``get_attachments``, which INNER JOINs
    ``attachment_refs`` and therefore can never surface a ref-less
    ``attachments`` row for any session. Reverting the
    ``DELETE FROM attachments WHERE ref_count <= 0 ...`` statement added to
    ``_write_attachments`` makes this test fail: the orphaned row survives
    in the table (visible via a direct ``SELECT``) even though
    ``get_attachments`` still (correctly) can't reach it.
    """
    import aiosqlite

    from polylogue.storage.sqlite.queries.attachment_records import get_attachments

    store = BlobStore(tmp_path / "blob")
    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)

    attachment = ParsedAttachment(
        provider_attachment_id="att-orphan",
        message_provider_id="m0",
        name="orphan.txt",
        mime_type="text/plain",
        inline_bytes=b"will be orphaned",
    )
    attachment_id = _attachment_id_for_test(attachment)

    db_path = tmp_path / "index.db"
    conn = _connect(db_path)

    # First ingest: attachment lands on message m0 and is reachable.
    first_session = _session_with_attachment(attachment)
    write_parsed_session_to_archive(
        conn, first_session, preacquired_attachment_blobs=_preacquired(store, first_session)
    )

    row = conn.execute(
        "SELECT acquisition_status, ref_count FROM attachments WHERE attachment_id = ?",
        (attachment_id,),
    ).fetchone()
    assert row is not None
    assert row["acquisition_status"] == "acquired"
    assert row["ref_count"] == 1

    # Second ingest of the SAME session (full replace, no merge_append): the
    # provider's message set no longer includes m0 (e.g. it was renumbered,
    # or excluded as a duplicate native id on this pass), so the attachment
    # -- which still names message_provider_id="m0" -- can no longer be
    # resolved to any message this time and _write_attachments skips
    # re-writing its ref. The attachment_refs row for it was already dropped
    # by the full replace's message CASCADE.
    second_session = ParsedSession(
        source_name=first_session.source_name,
        provider_session_id=first_session.provider_session_id,
        title=first_session.title,
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="a different message, m0 is gone",
                position=0,
                variant_index=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="a different message, m0 is gone")],
            )
        ],
        attachments=[attachment],
    )
    write_parsed_session_to_archive(conn, second_session, force_replace=True)

    surviving_row = conn.execute(
        "SELECT 1 FROM attachments WHERE attachment_id = ?",
        (attachment_id,),
    ).fetchone()
    assert surviving_row is None, (
        "orphaned attachment row (acquired, real bytes) must be garbage-collected "
        "once its last ref is dropped, not left permanently unreachable"
    )

    async with aiosqlite.connect(db_path) as aconn:
        aconn.row_factory = aiosqlite.Row
        records = await get_attachments(aconn, "aistudio-drive:s1")
    assert all(record.attachment_id != attachment_id for record in records), (
        "get_attachments (the production session-attachment read path) must never surface the orphaned attachment"
    )


def _attachment_id_for_test(attachment: ParsedAttachment) -> str:
    """Mirror ``write.py:_attachment_id`` (identity hash, session-independent)
    without importing the private helper across module boundaries.
    """
    import hashlib as _hashlib

    def _hash_bytes(*parts: str) -> bytes:
        digest = _hashlib.sha256()
        for part in parts:
            digest.update(part.encode("utf-8", errors="surrogatepass"))
            digest.update(b"\0")
        return digest.digest()

    return _hash_bytes(
        "attachment",
        attachment.provider_attachment_id,
        attachment.provider_file_id or "",
        attachment.provider_drive_id or "",
        attachment.path or "",
        attachment.name or "",
        attachment.mime_type or "",
        str(attachment.size_bytes or 0),
    ).hex()
