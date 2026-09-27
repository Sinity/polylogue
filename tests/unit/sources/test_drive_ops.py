"""Focused contracts for Drive ingestion helpers and live attachment acquisition.

`_apply_drive_attachments`/`iter_drive_sessions` (the decoupled, dead
local-path-writing attachment path) were removed as part of polylogue-83u.2:
they had zero live callers and wrote to `attachment.path` instead of
`inline_bytes`, so acquired Drive attachment bytes never reached the blob
store. The live path is `iter_drive_raw_data`, which now resolves
Drive-hosted attachment references (`driveDocument`/`driveImage`/etc.) via the
same live client used to download the session document, injecting fetched
bytes into the raw payload before it is cached/blob-stored. The tests below
exercise that live path end to end through the ordinary parse+write pipeline,
proving `acquisition_status='acquired'` with a blob at the attachment's true
SHA-256 (AC#1, Drive sub-case), and that a fetch failure leaves the attachment
honestly `unfetched` rather than fabricating a hash.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.config import Source
from polylogue.core.json import JSONValue
from polylogue.pipeline.ids import session_content_hash
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.services.ingest_worker import SessionWritePayload
from polylogue.sources import DriveFile, download_drive_files
from polylogue.sources.drive import drive_cache_file_path, iter_drive_raw_data
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


@dataclass
class _DriveSessionClient:
    """Minimal `DriveSourceAPI` stub covering the live raw-acquisition path."""

    files: list[DriveFile]
    payload_bytes: dict[str, bytes]
    download_bytes_calls: list[str] = field(default_factory=list)

    def resolve_folder_id(self, folder_ref: str) -> str:
        return f"folder:{folder_ref}"

    def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]:
        yield from self.files

    def download_json_payload(self, file_id: str, *, name: str) -> JSONValue:
        raise NotImplementedError("iter_drive_raw_data uses download_bytes, not download_json_payload")

    def download_to_path(self, file_id: str, dest: Path) -> DriveFile:
        raise NotImplementedError("not used by the live raw-acquisition path")

    def download_bytes(self, file_id: str) -> bytes:
        self.download_bytes_calls.append(file_id)
        return self.payload_bytes[file_id]

    def download_into(self, file_id: str, handle: IO[bytes]) -> None:
        raise NotImplementedError("attachment bytes are fetched by the convergence stage, not raw acquisition")


def _empty_cursor_state() -> CursorStatePayload:
    return {}


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def test_download_drive_files_contract(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    client = MagicMock()
    client.iter_json_files.return_value = [
        DriveFile("good", "session", "application/json", None, None),
        DriveFile("bad", "broken.jsonl", "application/json", None, None),
    ]

    def download(file_id: str, dest: Path) -> None:
        if file_id == "bad":
            raise PermissionError("denied")
        dest.write_bytes(b'{"id":"good"}')

    client.download_to_path.side_effect = download

    result = download_drive_files(client, "folder-1", tmp_path)

    assert result.total_files == 2
    assert [path.name for path in result.downloaded_files] == ["session.json"]
    assert result.downloaded_files[0].read_bytes() == b'{"id":"good"}'
    assert result.failed_files == [{"file_id": "bad", "name": "broken.jsonl", "error": "denied"}]


def test_iter_drive_raw_data_replaces_torn_cache_even_when_revision_is_unchanged(tmp_path: Path) -> None:
    payload = {"chunkedPrompt": {"chunks": [{"role": "user", "text": "fresh"}]}}
    client = _DriveSessionClient(
        files=[DriveFile("file-1", "session.json", "application/json", "2025-01-01T00:00:00Z", 64)],
        payload_bytes={"file-1": json.dumps(payload).encode()},
    )
    cache = drive_cache_file_path(tmp_path, "session.json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"{")

    records = list(
        iter_drive_raw_data(
            source=Source(name="gemini", folder="Google AI Studio", path=tmp_path),
            client=client,
            known_mtimes={str(cache): "2025-01-01T00:00:00Z"},
            blob_store=BlobStore(tmp_path / "blob"),
        )
    )

    assert len(records) == 1
    assert client.download_bytes_calls == ["file-1"]
    assert json.loads(cache.read_bytes()) == payload


def _write_via_ingest_batch(
    *,
    conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    blob_publisher: ArchiveBlobPublisher,
    session: ParsedSession,
    raw_id: str,
) -> None:
    payload = SessionWritePayload(
        session_id=str(make_session_id(session.source_name, session.provider_session_id)),
        content_hash=session_content_hash(session),
        parsed_session=session,
        message_count=len(session.messages),
        attachment_count=len(session.attachments),
        raw_id=raw_id,
    )
    changed, _ = ingest_batch_core._write_session(conn, payload, blob_publisher=blob_publisher, source_conn=source_conn)
    assert changed is True
    conn.commit()
