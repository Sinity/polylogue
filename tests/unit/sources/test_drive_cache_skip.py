"""An unchanged Drive revision must not decode its cache to prove the skip.

``iter_drive_raw_data`` called ``_read_valid_cache`` -- ``read_bytes`` plus a
whole-document ``json.loads`` -- before comparing the Drive revision against
``known_mtimes``. A large unchanged AI Studio document therefore paid its
complete object graph on every scan, which is what this memory-bounded
acquisition route exists to avoid. The skip only needs a decode, never the
decoded object, so it now runs ``_cache_document_is_readable`` (a streamed
``ijson``/line proof) and materializes bytes only when a payload is produced.

Anti-vacuity: move ``_read_valid_cache`` back above the revision comparison in
``polylogue/sources/drive/__init__.py`` and
``test_unchanged_cache_is_not_materialized`` goes red -- the recording stub
records a call. ``test_corrupt_cache_is_reacquired`` pins the opposite
direction so a validator that simply always says "readable" cannot pass, and
``test_validators_admit_the_same_documents`` pins that the streamed proof
admits exactly the documents the byte reader hands back.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pytest

from polylogue.config import Source
from polylogue.core.json import JSONValue, is_json_value
from polylogue.sources import drive as drive_module
from polylogue.sources.drive import (
    _cache_document_is_readable,
    _read_valid_cache,
    drive_cache_file_path,
    iter_drive_raw_data,
)
from polylogue.sources.drive.source import DriveSourceAPI
from polylogue.sources.drive.types import DriveFile
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

    def download_bytes(self, file_id: str) -> bytes:
        self.downloaded.append(file_id)
        return self.payloads[file_id]

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
    cache_path = drive_cache_file_path(cache_dir, "prompt.json")
    cache_path.write_bytes(cache_bytes)
    client = _StubDriveClient(
        [DriveFile("f1", "prompt.json", "application/json", _MTIME, len(cache_bytes))],
        {"f1": b'{"chunkedPrompt": {"chunks": []}}'},
    )
    items = list(
        iter_drive_raw_data(
            source=source,
            client=cast(DriveSourceAPI, client),
            known_mtimes={str(cache_path): _MTIME},
            blob_store=BlobStore(tmp_path / "blob"),
        )
    )
    return client, [cache_path], items


def test_unchanged_cache_is_not_materialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reads: list[Path] = []

    def _recording_read(path: Path) -> bytes | None:
        reads.append(path)
        return _read_valid_cache(path)

    monkeypatch.setattr(drive_module, "_read_valid_cache", _recording_read)

    client, _paths, items = _run(tmp_path, b'{"chunkedPrompt": {"chunks": [{"role": "user"}]}}')

    assert items == []
    assert client.downloaded == []
    assert reads == []


def test_corrupt_cache_is_reacquired(tmp_path: Path) -> None:
    client, _paths, items = _run(tmp_path, b'{"chunkedPrompt": {"chunks":')

    assert client.downloaded == ["f1"]
    assert len(items) == 1


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

    assert _cache_document_is_readable(path) is (_read_valid_cache(path) is not None)
