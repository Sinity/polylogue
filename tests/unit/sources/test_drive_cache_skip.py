"""Drive cache revisions preserve exact bytes without resident payload reads."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import IO, cast

import pytest

from polylogue.config import Source
from polylogue.core.json import JSONValue, is_json_value
from polylogue.sources.drive import (
    _cache_document_is_readable,
    _cache_holds_readable_revision,
    drive_cache_file_path,
    iter_drive_raw_data,
)
from polylogue.sources.drive.source import DriveSourceAPI
from polylogue.sources.drive.types import DriveFile
from polylogue.sources.drive.witness import drive_cache_directory, drive_source_coordinate
from polylogue.sources.parsers.base import RawSessionData
from polylogue.storage.blob_store import BlobStore

_MTIME = "2026-01-01T00:00:00Z"


class _StubDriveClient:
    def __init__(self, files: list[DriveFile], payloads: dict[str, bytes]) -> None:
        self.files = files
        self.payloads = payloads
        self.downloaded: list[str] = []

    def resolve_folder_id(self, folder_ref: str) -> str:
        return f"folder:{folder_ref}"

    def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]:
        del folder_id
        yield from self.files

    def get_metadata(self, file_id: str, *, refresh: bool = False) -> DriveFile:
        return next(file for file in self.iter_json_files("") if file.file_id == file_id)

    def download_bytes(self, file_id: str) -> bytes:
        self.downloaded.append(file_id)
        return self.payloads[file_id]

    def download_into(self, file_id: str, handle: IO[bytes]) -> None:
        handle.write(self.download_bytes(file_id))

    def download_json_payload(self, file_id: str, *, name: str) -> JSONValue:
        del name
        payload = json.loads(self.download_bytes(file_id))
        assert is_json_value(payload)
        return payload

    def download_to_path(self, file_id: str, dest: Path) -> DriveFile:
        dest.write_bytes(self.download_bytes(file_id))
        return next(entry for entry in self.files if entry.file_id == file_id)


def _run(tmp_path: Path, cache_bytes: bytes) -> tuple[_StubDriveClient, list[Path], list[RawSessionData]]:
    cache_dir = tmp_path / "drive-cache"
    cache_dir.mkdir()
    source = Source(name="gemini", folder="AI Studio", path=cache_dir)
    cache_path = drive_cache_file_path(drive_cache_directory(cache_dir, "folder:AI Studio"), "f1")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(cache_bytes)
    cache_path.with_name(f"{cache_path.name}.revision").write_text(_MTIME)
    client = _StubDriveClient(
        [DriveFile("f1", "prompt.json", "application/json", _MTIME, len(cache_bytes))],
        {"f1": b'{"chunkedPrompt": {"chunks": []}}'},
    )
    items = list(
        iter_drive_raw_data(
            source=source,
            client=cast(DriveSourceAPI, client),
            known_mtimes={drive_source_coordinate("gemini", "folder:AI Studio", "f1"): _MTIME},
            blob_store=BlobStore(tmp_path / "blob"),
        )
    )
    return client, [cache_path], items


def test_unchanged_cache_is_not_materialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse_resident_read(_path: Path) -> bytes:
        raise AssertionError("cache bytes must remain on disk")

    monkeypatch.setattr(Path, "read_bytes", refuse_resident_read)
    client, _paths, items = _run(tmp_path, b'{"chunkedPrompt": {"chunks": [{"role": "user"}]}}')
    assert items == []
    assert client.downloaded == []


def test_corrupt_cache_is_reacquired(tmp_path: Path) -> None:
    client, _paths, items = _run(tmp_path, b'{"chunkedPrompt": {"chunks":')

    assert client.downloaded == ["f1"]
    assert len(items) == 1


def test_a_cache_from_an_earlier_revision_is_redownloaded(tmp_path: Path) -> None:
    """A document that changed on Drive is re-downloaded, not served from cache.

    Anti-vacuity: drop the revision comparison in ``_cache_holds_readable_revision`` and
    the stale cached document is returned while the grown one is never read.
    """
    source = Source(name="gemini", folder="Google AI Studio", path=tmp_path)
    cache_path = drive_cache_file_path(drive_cache_directory(tmp_path, "folder:Google AI Studio"), "f1")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b'{"chunkedPrompt": {"chunks": []}}')
    cache_path.with_name(f"{cache_path.name}.revision").write_text("2025-01-01T00:00:00Z")
    grown = b'{"chunkedPrompt": {"chunks": [{"role": "user", "text": "grown"}]}}'
    client = _StubDriveClient(
        [DriveFile("f1", "prompt.json", "application/json", _MTIME, len(grown))],
        {"f1": grown},
    )

    items = list(
        iter_drive_raw_data(
            source=source,
            client=cast(DriveSourceAPI, client),
            blob_store=BlobStore(tmp_path / "blob"),
        )
    )

    assert client.downloaded == ["f1"]
    assert len(items) == 1
    assert cache_path.read_bytes() == grown
    assert cache_path.with_name(f"{cache_path.name}.revision").read_text() == _MTIME


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("whole", b'{"a": [1, 2, {"b": null}]}'),
        ("scalar", b"null"),
        ("empty", b""),
        ("blank", b"   \n  "),
        ("truncated", b'{"a": [1, 2,'),
        ("stream.jsonl", b'{"a":1}\n\n{"b":2}\n'),
        ("bad.jsonl", b"{oops\n"),
        ("null.jsonl", b"null\n"),
        ("blank.jsonl", b"\n\n"),
    ],
    ids=["whole", "scalar", "empty", "blank", "truncated", "jsonl", "badjsonl", "nulljsonl", "blankjsonl"],
)
def test_validators_admit_the_same_documents(tmp_path: Path, name: str, payload: bytes) -> None:
    path = tmp_path / name
    path.write_bytes(payload)
    path.with_name(f"{path.name}.revision").write_text(_MTIME)

    readable = bool(payload.strip()) and name not in {"truncated", "bad.jsonl"}
    assert _cache_document_is_readable(path) is readable
    assert _cache_holds_readable_revision(path, _MTIME) is readable
