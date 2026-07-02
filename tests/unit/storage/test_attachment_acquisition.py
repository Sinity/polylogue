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


def test_inline_attachment_bytes_are_stored_with_true_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = BlobStore(tmp_path / "blob")
    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)

    payload = b"hello attachment bytes"
    conn = _connect(tmp_path / "index.db")
    write_parsed_session_to_archive(
        conn,
        _session_with_attachment(
            ParsedAttachment(
                provider_attachment_id="att-1",
                message_provider_id="m0",
                name="note.txt",
                mime_type="text/plain",
                inline_bytes=payload,
            )
        ),
    )

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

    write_parsed_session_to_archive(conn, session)

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
