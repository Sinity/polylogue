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

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.config import Source
from polylogue.core.json import JSONValue
from polylogue.sources import DriveFile, download_drive_files
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.drive import iter_drive_raw_data
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


@dataclass
class _DriveSessionClient:
    """Minimal `DriveSourceAPI` stub covering the live raw-acquisition path."""

    files: list[DriveFile]
    payload_bytes: dict[str, bytes]
    attachment_bytes: dict[str, bytes] = field(default_factory=dict)
    attachment_failures: dict[str, Exception] = field(default_factory=dict)
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
        if file_id in self.attachment_failures:
            raise self.attachment_failures[file_id]
        if file_id in self.attachment_bytes:
            return self.attachment_bytes[file_id]
        return self.payload_bytes[file_id]


def _empty_cursor_state() -> CursorStatePayload:
    return {}


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _preacquired(store: BlobStore, session: ParsedSession) -> dict[int, tuple[bytes | None, int, str]]:
    acquired: dict[int, tuple[bytes | None, int, str]] = {}
    for attachment in session.attachments:
        if attachment.inline_bytes is None:
            continue
        blob_hash, size = store.write_from_bytes(attachment.inline_bytes)
        acquired[id(attachment)] = (bytes.fromhex(blob_hash), size, "acquired")
    return acquired


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


def test_iter_drive_raw_data_injects_live_attachment_bytes_into_raw_payload(tmp_path: Path) -> None:
    """Drive-hosted attachment bytes are fetched INSIDE the live client's
    iterator scope and land in the stored raw payload as a base64 sidecar."""
    payload = {
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hi"},
                {
                    "role": "model",
                    "text": "Here is the file",
                    "driveDocument": {"id": "att-1", "name": "doc.txt", "mimeType": "text/plain"},
                },
            ]
        }
    }
    client = _DriveSessionClient(
        files=[DriveFile("file-1", "chat.json", "application/json", "2025-01-01T00:00:00Z", 10)],
        payload_bytes={"file-1": json.dumps(payload).encode("utf-8")},
        attachment_bytes={"att-1": b"the actual drive attachment bytes"},
    )
    blob_store = BlobStore(tmp_path / "blob")
    cursor_state = _empty_cursor_state()

    records = list(
        iter_drive_raw_data(
            source=Source(name="gemini", folder="Google AI Studio", path=tmp_path),
            client=client,
            cursor_state=cursor_state,
            blob_store=blob_store,
        )
    )

    assert len(records) == 1
    assert client.download_bytes_calls == ["file-1", "att-1"]
    assert records[0].blob_hash is not None
    stored_bytes = blob_store.read_all(records[0].blob_hash)
    stored_payload = json.loads(stored_bytes)
    stored_doc = stored_payload["chunkedPrompt"]["chunks"][1]["driveDocument"]
    assert stored_doc["id"] == "att-1"
    assert stored_doc["name"] == "doc.txt"

    # The cache file written for future runs carries the same injected bytes.
    cache_path = Path(records[0].source_path)
    assert json.loads(cache_path.read_bytes()) == stored_payload


def test_drive_live_attachment_bytes_reach_acquired_blob_with_true_hash(tmp_path: Path) -> None:
    """End-to-end: live client -> raw injection -> ordinary parse -> write.

    Proves polylogue-83u.2 AC#1 for the Drive sub-case: a seeded fixture with
    a live Drive-hosted attachment reference produces
    acquisition_status='acquired' with a blob at the attachment's true
    SHA-256, using the same generic dispatch (`parse_payload`) and write
    (`write_parsed_session_to_archive`) path the daemon uses in production.
    """
    attachment_bytes = b"the actual drive attachment bytes"
    payload = {
        "title": "Drive Chat",
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hi"},
                {
                    "role": "model",
                    "text": "Here is the file",
                    "driveDocument": {"id": "att-1", "name": "doc.txt", "mimeType": "text/plain"},
                },
            ]
        },
    }
    client = _DriveSessionClient(
        files=[DriveFile("file-1", "chat.json", "application/json", "2025-01-01T00:00:00Z", 10)],
        payload_bytes={"file-1": json.dumps(payload).encode("utf-8")},
        attachment_bytes={"att-1": attachment_bytes},
    )
    blob_store = BlobStore(tmp_path / "blob")

    records = list(
        iter_drive_raw_data(
            source=Source(name="gemini", folder="Google AI Studio", path=tmp_path),
            client=client,
            blob_store=blob_store,
        )
    )
    assert len(records) == 1
    assert records[0].blob_hash is not None
    resolved_payload = json.loads(blob_store.read_all(records[0].blob_hash))

    # The subprocess parse stage has no live client — it only ever sees the
    # already-resolved raw payload, dispatched generically like production.
    sessions = parse_payload("gemini", resolved_payload, "fallback-id")
    assert len(sessions) == 1
    session = sessions[0]
    assert len(session.attachments) == 1
    attachment = session.attachments[0]
    assert attachment.upload_origin == "drive"
    assert attachment.inline_bytes == attachment_bytes

    conn = _connect(tmp_path / "index.db")
    write_parsed_session_to_archive(conn, session, preacquired_attachment_blobs=_preacquired(blob_store, session))

    row = conn.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchone()
    assert row["acquisition_status"] == "acquired"
    assert row["byte_count"] == len(attachment_bytes)
    assert bytes(row["blob_hash"]) == hashlib.sha256(attachment_bytes).digest()
    assert blob_store.read_all(hashlib.sha256(attachment_bytes).hexdigest()) == attachment_bytes


def test_drive_attachment_fetch_failure_stays_honestly_unfetched(tmp_path: Path) -> None:
    """A Drive attachment whose live fetch fails is NOT acquired and carries
    no synthetic hash — it stays `unfetched`, same as a genuinely-unfetchable
    handle (source_url-only). This is the negative-path complement to the
    acquired-blob test above."""
    payload = {
        "chunkedPrompt": {
            "chunks": [
                {
                    "role": "model",
                    "driveDocument": {"id": "att-dead", "name": "gone.bin", "mimeType": "application/octet-stream"},
                },
            ]
        }
    }
    client = _DriveSessionClient(
        files=[DriveFile("file-1", "chat.json", "application/json", None, 10)],
        payload_bytes={"file-1": json.dumps(payload).encode("utf-8")},
        attachment_failures={"att-dead": RuntimeError("file no longer accessible")},
    )
    blob_store = BlobStore(tmp_path / "blob")

    records = list(
        iter_drive_raw_data(
            source=Source(name="gemini", folder="Google AI Studio", path=tmp_path),
            client=client,
            blob_store=blob_store,
        )
    )
    assert records[0].blob_hash is not None
    resolved_payload = json.loads(blob_store.read_all(records[0].blob_hash))
    sessions = parse_payload("gemini", resolved_payload, "fallback-id")
    session = sessions[0]
    attachment = session.attachments[0]
    assert attachment.inline_bytes is None

    conn = _connect(tmp_path / "index.db")
    write_parsed_session_to_archive(conn, session)

    row = conn.execute("SELECT blob_hash, acquisition_status FROM attachments").fetchone()
    assert row["acquisition_status"] == "unfetched"
    assert row["blob_hash"] is None
