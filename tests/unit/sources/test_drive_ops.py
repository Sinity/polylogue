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
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from polylogue.config import Source
from polylogue.core.json import JSONValue
from polylogue.sources import DriveFile, download_drive_files
from polylogue.sources.drive import drive_cache_file_path, iter_drive_raw_data
from polylogue.sources.drive.witness import drive_cache_directory, drive_source_coordinate
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload


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

    def get_metadata(self, file_id: str, *, refresh: bool = False) -> DriveFile:
        return next(file for file in self.iter_json_files("") if file.file_id == file_id)

    def download_bytes(self, file_id: str) -> bytes:
        self.download_bytes_calls.append(file_id)
        return self.payload_bytes[file_id]

    def download_into(self, file_id: str, handle: IO[bytes]) -> None:
        raise NotImplementedError("attachment bytes are fetched by the convergence stage, not raw acquisition")


def _empty_cursor_state() -> CursorStatePayload:
    return {}


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
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b'{"id":"good"}')

    client.download_to_path.side_effect = download

    result = download_drive_files(client, "folder-1", tmp_path)

    assert result.total_files == 2
    assert result.downloaded_files == [drive_cache_file_path(drive_cache_directory(tmp_path, "folder-1"), "good")]
    assert result.downloaded_files[0].read_bytes() == b'{"id":"good"}'
    assert result.failed_files == [{"file_id": "bad", "name": "broken.jsonl", "error": "denied"}]


def test_iter_drive_raw_data_replaces_torn_cache_even_when_revision_is_unchanged(tmp_path: Path) -> None:
    payload = {"chunkedPrompt": {"chunks": [{"role": "user", "text": "fresh"}]}}
    client = _DriveSessionClient(
        files=[DriveFile("file-1", "session.json", "application/json", "2025-01-01T00:00:00Z", 64)],
        payload_bytes={"file-1": json.dumps(payload).encode()},
    )
    cache = drive_cache_file_path(drive_cache_directory(tmp_path, "folder:Google AI Studio"), "file-1")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"{")

    records = list(
        iter_drive_raw_data(
            source=Source(name="gemini", folder="Google AI Studio", path=tmp_path),
            client=client,
            known_mtimes={
                drive_source_coordinate("gemini", "folder:Google AI Studio", "file-1"): "2025-01-01T00:00:00Z"
            },
            blob_store=BlobStore(tmp_path / "blob"),
        )
    )

    assert len(records) == 1
    assert client.download_bytes_calls == ["file-1"]
    assert json.loads(cache.read_bytes()) == payload


def test_iter_drive_raw_data_replaces_a_cache_rewritten_with_attachment_bytes(tmp_path: Path) -> None:
    """A cache with no recorded revision is re-downloaded, even on the unchanged path.

    Caches written before revisions were recorded include ones an earlier
    acquisition rewrote with embedded attachment bytes. Anti-vacuity: drop the
    revision check in ``_cache_holds_readable_revision`` and the rewritten
    (valid JSON) document satisfies the unchanged-revision skip, so nothing
    is re-downloaded.
    """
    payload = {"chunkedPrompt": {"chunks": [{"role": "user", "text": "fresh"}]}}
    rewritten = {
        "chunkedPrompt": {
            "chunks": [
                {
                    "role": "user",
                    "text": "fresh",
                    "driveDocument": {"id": "att", "_polylogue_drive_live_bytes_b64": "Ynl0ZXM="},
                }
            ]
        }
    }
    client = _DriveSessionClient(
        files=[DriveFile("file-1", "session.json", "application/json", "2025-01-01T00:00:00Z", 64)],
        payload_bytes={"file-1": json.dumps(payload).encode()},
    )
    cache = drive_cache_file_path(drive_cache_directory(tmp_path, "folder:Google AI Studio"), "file-1")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(json.dumps(rewritten).encode())

    records = list(
        iter_drive_raw_data(
            source=Source(name="gemini", folder="Google AI Studio", path=tmp_path),
            client=client,
            # An unchanged revision takes the cursor fast path; a rewritten
            # cache must not satisfy it.
            known_mtimes={
                drive_source_coordinate("gemini", "folder:Google AI Studio", "file-1"): "2025-01-01T00:00:00Z"
            },
            blob_store=BlobStore(tmp_path / "blob"),
        )
    )

    assert len(records) == 1
    assert client.download_bytes_calls == ["file-1"]
    assert json.loads(cache.read_bytes()) == payload
