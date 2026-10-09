"""Drive source acquisition retains provider bytes through cache and CAS."""

from __future__ import annotations

import hashlib
import json
import tracemalloc
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import pytest

from polylogue.config import Source
from polylogue.core.compute import DaemonOperationCancelled
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
    download_into_calls: list[str] = field(default_factory=list)

    def resolve_folder_id(self, folder_ref: str) -> str:
        return f"folder:{folder_ref}"

    def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]:
        yield from self.files

    def download_json_payload(self, file_id: str, *, name: str) -> JSONValue:
        raise NotImplementedError("iter_drive_raw_data uses download_into, not download_json_payload")

    def download_to_path(self, file_id: str, dest: Path) -> DriveFile:
        raise NotImplementedError("not used by the live raw-acquisition path")

    def get_metadata(self, file_id: str, *, refresh: bool = False) -> DriveFile:
        return next(file for file in self.iter_json_files("") if file.file_id == file_id)

    def download_bytes(self, file_id: str) -> bytes:
        raise AssertionError("raw acquisition must stream")

    def download_into(self, file_id: str, handle: IO[bytes]) -> None:
        self.download_into_calls.append(file_id)
        handle.write(self.payload_bytes[file_id])


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
    assert client.download_into_calls == ["file-1"]
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
    assert client.download_into_calls == ["file-1"]
    assert json.loads(cache.read_bytes()) == payload


@pytest.mark.parametrize("cached", [False, True])
def test_large_drive_document_stays_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: bool) -> None:
    prefix = b'{"chunkedPrompt":{"chunks":[{"role":"user","text":"neutral"}]}}'
    chunk = b" " * 65536
    count = 256
    size = len(prefix) + len(chunk) * count
    digest = hashlib.sha256(prefix)
    for _ in range(count):
        digest.update(chunk)

    class StreamingClient(_DriveSessionClient):
        def download_into(self, file_id: str, handle: IO[bytes]) -> None:
            self.download_into_calls.append(file_id)
            handle.write(prefix)
            for _ in range(count):
                handle.write(chunk)

    client = StreamingClient(
        files=[DriveFile("large", "neutral.json", "application/json", "2026-01-01T00:00:00Z", size)],
        payload_bytes={},
    )
    source = Source(name="gemini", folder="neutral", path=tmp_path / "cache")
    assert source.path is not None
    cache = drive_cache_file_path(drive_cache_directory(source.path, "folder:neutral"), "large")
    if cached:
        cache.parent.mkdir(parents=True)
        with cache.open("wb") as handle:
            handle.write(prefix)
            for _ in range(count):
                handle.write(chunk)
        cache.with_name(cache.name + ".revision").write_text("2026-01-01T00:00:00Z")

    def refuse_resident_read(_path: Path) -> bytes:
        raise AssertionError("raw acquisition must never read a whole file")

    monkeypatch.setattr(Path, "read_bytes", refuse_resident_read)
    store = BlobStore(tmp_path / "blob")
    tracemalloc.start()
    try:
        records = list(iter_drive_raw_data(source=source, client=client, blob_store=store))
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < size // 2
    assert len(records) == 1
    assert records[0].blob_hash == digest.hexdigest()
    assert records[0].blob_size == size
    assert records[0].raw_bytes == b""
    assert client.download_into_calls == ([] if cached else ["large"])
    with cache.open("rb") as cache_reader:
        assert hashlib.file_digest(cache_reader, "sha256").hexdigest() == digest.hexdigest()
    with store.blob_path(digest.hexdigest()).open("rb") as blob_reader:
        assert hashlib.file_digest(blob_reader, "sha256").hexdigest() == digest.hexdigest()


def test_cancelled_drive_download_releases_private_stage(tmp_path: Path) -> None:
    class CancelledClient(_DriveSessionClient):
        def download_into(self, file_id: str, handle: IO[bytes]) -> None:
            handle.write(b"unfinished")
            raise DaemonOperationCancelled("neutral cancellation")

    client = CancelledClient(
        files=[DriveFile("file-1", "neutral.json", "application/json", "2026-01-01T00:00:00Z", 100)],
        payload_bytes={},
    )
    store = BlobStore(tmp_path / "blob")
    with pytest.raises(DaemonOperationCancelled):
        list(
            iter_drive_raw_data(
                source=Source(name="gemini", folder="neutral", path=tmp_path / "cache"), client=client, blob_store=store
            )
        )
    assert not list(tmp_path.rglob(".blob.*"))
    assert not list((tmp_path / "cache").rglob("*.json"))


@pytest.mark.parametrize("cached", [False, True])
def test_drive_private_stage_refuses_foreign_tail_before_publication(tmp_path: Path, cached: bool) -> None:
    """Both staging routes validate EOF before publishing any acquired raw."""
    payload = json.dumps(
        [
            {"chunkedPrompt": {"chunks": [{"role": "user", "text": "neutral"}]}},
            {"type": "session_meta", "payload": {"id": "neutral-codex"}},
        ]
    ).encode()
    revision = "2026-01-01T00:00:00Z"
    client = _DriveSessionClient(
        files=[DriveFile("file-1", "neutral.json", "application/json", revision, len(payload))],
        payload_bytes={"file-1": payload},
    )
    source = Source(name="gemini", folder="neutral", path=tmp_path / "cache")
    assert source.path is not None
    cache = drive_cache_file_path(drive_cache_directory(source.path, "folder:neutral"), "file-1")
    if cached:
        cache.parent.mkdir(parents=True)
        cache.write_bytes(payload)
        cache.with_name(cache.name + ".revision").write_text(revision)
    store = BlobStore(tmp_path / "blob")
    state = _empty_cursor_state()
    assert list(iter_drive_raw_data(source=source, client=client, blob_store=store, cursor_state=state)) == []
    assert state["error_count"] == 1
    assert client.download_into_calls == ([] if cached else ["file-1"])
    assert not store.blob_path(hashlib.sha256(payload).hexdigest()).exists()
    assert not list(tmp_path.rglob(".blob.*"))
    if cached:
        assert cache.read_bytes() == payload
    else:
        assert not cache.exists()
