"""Contract tests for DriveSourceClient — folder resolution, iteration, and downloads."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.sources.drive_gateway import DriveServiceGateway
from polylogue.sources.drive_source import (
    DriveSourceClient,
    _build_folder_lookup_query,
    _is_supported_drive_payload,
    _looks_like_id,
    _needs_download,
    _parse_downloaded_json_payload,
    _parse_modified_time,
    _parse_size,
)
from polylogue.sources.drive_types import (
    GEMINI_PROMPT_MIME_TYPE,
    DriveFile,
    DriveNotFoundError,
)
from tests.infra.drive_mocks import (
    FakeDriveServiceGateway,
    MockDriveService,
    mock_drive_file,
)
from tests.infra.strategies import json_document_strategy


def _source_client(
    mock_service: MockDriveService | None = None,
    file_content: dict[str, bytes | str] | None = None,
    download_error: Exception | None = None,
) -> DriveSourceClient:
    gw = FakeDriveServiceGateway(
        mock_service=mock_service,
        file_content=file_content,
        download_error=download_error,
    )
    return DriveSourceClient(gateway=cast(DriveServiceGateway, gw))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_modified_time_contract() -> None:
    from datetime import datetime, timezone

    utc_ts = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc).timestamp()

    assert _parse_modified_time(None) is None
    assert _parse_modified_time("") is None
    assert _parse_modified_time("   ") is None
    assert _parse_modified_time("not a date") is None
    assert _parse_modified_time("12345") is None
    assert _parse_modified_time("2024-13-45T99:99:99Z") is None
    assert _parse_modified_time("2024-01-15T10:30:45Z") == pytest.approx(utc_ts, abs=1)
    assert _parse_modified_time("2024-01-15T10:30:45+00:00") == pytest.approx(utc_ts, abs=1)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (0, 0),
        (1024, 1024),
        (-1, -1),
        ("123", 123),
        ("  456  ", 456),
        ("0", 0),
        ("not a number", None),
        ("12.34", None),
        ("12a", None),
        ("", None),
    ],
)
def test_parse_size_contract(raw: str | int | None, expected: int | None) -> None:
    assert _parse_size(raw) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", False),
        ("hello world", False),
        (" test", False),
        ("test ", False),
        ("file.txt", False),
        ("file@home", False),
        ("a/b", False),
        ("abc-123-def", True),
        ("file_1_test", True),
        ("abc123", True),
        ("123", True),
        ("a", True),
        ("---", True),
    ],
)
def test_looks_like_id_contract(value: str, expected: bool) -> None:
    assert _looks_like_id(value) is expected


@pytest.mark.parametrize(
    ("folder_ref", "expected"),
    [
        ("Inbox", "name = 'Inbox' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"),
        (
            "O'Hare Folder",
            "name = 'O\\'Hare Folder' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
        ),
    ],
)
def test_build_folder_lookup_query_contract(folder_ref: str, expected: str) -> None:
    assert _build_folder_lookup_query(folder_ref) == expected


@pytest.mark.parametrize(
    ("name", "mime_type", "expected"),
    [
        ("chat.json", "application/json", True),
        ("chat.jsonl", "text/plain", True),
        ("chat.jsonl.txt", "application/octet-stream", True),
        ("chat.ndjson", "application/octet-stream", True),
        ("prompt.bin", GEMINI_PROMPT_MIME_TYPE, True),
        ("notes.md", "text/markdown", False),
    ],
)
def test_supported_drive_payload_contract(name: str, mime_type: str, expected: bool) -> None:
    assert _is_supported_drive_payload(name, mime_type) is expected


def test_needs_download_contract(tmp_path: Path) -> None:
    dest = tmp_path / "payload.json"
    dest.write_text("12345", encoding="utf-8")
    os.utime(dest, (1735689600.0, 1735689600.0))
    base = DriveFile(
        file_id="file-1",
        name="payload.json",
        mime_type="application/json",
        modified_time="2025-01-01T00:00:00Z",
        size_bytes=5,
    )

    assert _needs_download(base, dest) is False
    assert _needs_download(replace(base, size_bytes=6), dest) is True
    assert _needs_download(replace(base, modified_time="2025-01-01T00:00:05Z"), dest) is True
    assert _needs_download(base, tmp_path / "missing.json") is True


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------


@given(
    st.lists(json_document_strategy(), min_size=0, max_size=8),
    st.sampled_from(("session.jsonl", "session.jsonl.txt", "session.ndjson", "SESSION.JSONL")),
)
@settings(max_examples=35)
def test_parse_downloaded_json_payload_preserves_newline_delimited_documents(
    documents: list[dict[str, object]], name: str
) -> None:
    raw = b"\n\n".join(json.dumps(document).encode("utf-8") for document in documents)
    assert _parse_downloaded_json_payload(raw, name=name) == documents


@given(st.one_of(json_document_strategy(), st.lists(json_document_strategy(), min_size=0, max_size=6)))
@settings(max_examples=35)
def test_parse_downloaded_json_payload_round_trips_standard_json(payload: object) -> None:
    raw = json.dumps(payload).encode("utf-8")
    assert _parse_downloaded_json_payload(raw, name="payload.json") == payload


# ---------------------------------------------------------------------------
# iter_json_files — filtering contract
# ---------------------------------------------------------------------------


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=10_000),
            st.sampled_from((".json", ".jsonl", ".jsonl.txt", ".ndjson", ".txt", ".md", "")),
            st.sampled_from(("application/json", "text/plain", "application/octet-stream", GEMINI_PROMPT_MIME_TYPE)),
            st.booleans(),
        ),
        min_size=1,
        max_size=20,
        unique_by=lambda item: item[0],
    )
)
@settings(max_examples=30, deadline=None)
def test_iter_json_files_filters_supported_entries(
    file_specs: list[tuple[int, str, str, bool]],
) -> None:
    folder_id = "folder-law"
    service = MockDriveService()
    service._files_resource.files.clear()
    client = _source_client(mock_service=service)

    expected_ids: list[str] = []
    for file_num, suffix, mime_type, in_folder in file_specs:
        file_id = f"file-{file_num}"
        name = f"payload-{file_num}{suffix}"
        parents = [folder_id] if in_folder else ["other-folder"]
        service._files_resource.files[file_id] = mock_drive_file(
            file_id=file_id, name=name, mime_type=mime_type, parents=parents
        )
        if in_folder and (
            name.lower().endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson")) or mime_type == GEMINI_PROMPT_MIME_TYPE
        ):
            expected_ids.append(file_id)

    files = list(client.iter_json_files(folder_id))

    assert [f.file_id for f in files] == expected_ids
    assert list(client._meta_cache) == expected_ids


# ---------------------------------------------------------------------------
# Folder resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("folder_ref", "get_result", "list_result", "expected", "expect_get", "expect_error"),
    [
        (
            "folder-1",
            {"id": "folder-1", "mimeType": "application/vnd.google-apps.folder"},
            [],
            "folder-1",
            True,
            None,
        ),
        (
            "Target",
            Exception("skip id lookup"),
            [{"id": "folder-by-name", "name": "Target"}],
            "folder-by-name",
            False,
            None,
        ),
        (
            "O'Hare Folder",
            AssertionError("id lookup should not run"),
            [{"id": "folder-id", "name": "O'Hare"}],
            "folder-id",
            False,
            None,
        ),
        (
            "Missing",
            Exception("404"),
            [],
            None,
            False,
            DriveNotFoundError,
        ),
    ],
    ids=["by-id", "by-name", "with-quote", "not-found"],
)
def test_resolve_folder_id_contract(
    folder_ref: str,
    get_result: object,
    list_result: list[dict[str, str]],
    expected: str | None,
    expect_get: bool,
    expect_error: type[Exception] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from googleapiclient.errors import HttpError

    class _FakeHttpError(HttpError):
        def __init__(self, status: int) -> None:
            resp = SimpleNamespace(status=status, reason="")
            super().__init__(resp=resp, content=b"")

    service = MockDriveService()
    resource = service._files_resource
    get_mock = (
        MagicMock(return_value=SimpleNamespace(execute=lambda: get_result))
        if expect_get
        else MagicMock(side_effect=_FakeHttpError(404))
    )
    list_mock = MagicMock(return_value=SimpleNamespace(execute=lambda: {"files": list_result}))
    if expect_get:
        monkeypatch.setattr(resource, "get", get_mock)
    else:
        monkeypatch.setattr(resource, "get", get_mock)
    monkeypatch.setattr(resource, "list", list_mock)

    client = _source_client(mock_service=service)

    if expect_error is not None:
        with pytest.raises(expect_error, match="Folder not found"):
            client.resolve_folder_id(folder_ref)
    else:
        assert client.resolve_folder_id(folder_ref) == expected


def test_resolve_folder_helper_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MockDriveService()
    get_mock = MagicMock(
        return_value=SimpleNamespace(
            execute=lambda: {"id": "folder-1", "mimeType": "application/vnd.google-apps.folder"}
        )
    )
    list_mock = MagicMock(
        return_value=SimpleNamespace(execute=lambda: {"files": [{"id": "folder-2", "name": "Inbox"}]})
    )
    monkeypatch.setattr(service._files_resource, "get", get_mock)
    monkeypatch.setattr(service._files_resource, "list", list_mock)
    client = _source_client(mock_service=service)

    assert client._resolve_folder_by_id("folder-1") == "folder-1"
    assert client._resolve_folder_by_name("Inbox") == "folder-2"
    assert list_mock.call_args.kwargs["q"] == (
        "name = 'Inbox' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )


# ---------------------------------------------------------------------------
# download_bytes / download_json_payload
# ---------------------------------------------------------------------------


def test_download_bytes_and_json_payload_contract() -> None:
    client = _source_client(file_content={"file-1": b'{"hello":"world"}', "file-2": b'{"a":1}\n{"b":2}\n'})

    assert client.download_bytes("file-1") == b'{"hello":"world"}'
    assert client.download_json_payload("file-1", name="payload.json") == {"hello": "world"}
    assert client.download_json_payload("file-2", name="payload.jsonl") == [{"a": 1}, {"b": 2}]


# ---------------------------------------------------------------------------
# download_to_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("existing_bytes", "existing_mtime", "expect_write"),
    [
        (b"12345", 1735689600.0, False),  # unchanged — skip
        (None, None, True),  # new file — write
    ],
    ids=["skip-unchanged", "write-new"],
)
def test_download_to_path_contract(
    existing_bytes: bytes | None,
    existing_mtime: float | None,
    expect_write: bool,
    tmp_path: Path,
) -> None:
    metadata = DriveFile(
        file_id="file-1",
        name="payload.json",
        mime_type="application/json",
        modified_time="2025-01-01T00:00:00Z",
        size_bytes=5,
    )
    service = MockDriveService(file_content={"file-1": b"12345"})
    client = _source_client(mock_service=service)
    client._meta_cache["file-1"] = metadata

    dest = tmp_path / "payload.json"
    if existing_bytes is not None:
        dest.write_bytes(existing_bytes)
    if existing_mtime is not None:
        os.utime(dest, (existing_mtime, existing_mtime))

    result = client.download_to_path("file-1", dest)

    assert result.file_id == "file-1"
    if expect_write:
        assert dest.read_bytes() == b"12345"
        assert abs(dest.stat().st_mtime - 1735689600.0) < 1


def test_download_to_path_cleans_up_temp_file_when_download_fails(tmp_path: Path) -> None:
    metadata = DriveFile(
        file_id="file-1",
        name="payload.json",
        mime_type="application/json",
        modified_time=None,
        size_bytes=5,
    )
    client = _source_client(download_error=OSError("download blew up"))
    client._meta_cache["file-1"] = metadata

    dest = tmp_path / "payload.json"
    with pytest.raises(OSError, match="download blew up"):
        client.download_to_path("file-1", dest)

    assert not list(tmp_path.glob("tmp*"))
    assert not dest.exists()


# ---------------------------------------------------------------------------
# get_metadata — cache
# ---------------------------------------------------------------------------


def test_get_metadata_and_iteration_cache_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MockDriveService(
        files_data={"file-1": mock_drive_file(file_id="file-1", name="", mime_type="application/json", size=0)}
    )
    get_mock = MagicMock(wraps=service._files_resource.get)
    list_mock = MagicMock(
        side_effect=[
            SimpleNamespace(
                execute=lambda: {
                    "nextPageToken": "page-2",
                    "files": [
                        {
                            "id": "f1",
                            "name": "one.json",
                            "mimeType": "application/json",
                            "modifiedTime": None,
                            "size": "1",
                        },
                        {
                            "id": "skip",
                            "name": "readme.txt",
                            "mimeType": "text/plain",
                            "modifiedTime": None,
                            "size": "1",
                        },
                    ],
                }
            ),
            SimpleNamespace(
                execute=lambda: {
                    "files": [
                        {
                            "id": "f2",
                            "name": "two.ndjson",
                            "mimeType": "application/octet-stream",
                            "modifiedTime": None,
                            "size": "2",
                        },
                        {
                            "id": "f3",
                            "name": "prompt.bin",
                            "mimeType": GEMINI_PROMPT_MIME_TYPE,
                            "modifiedTime": None,
                            "size": "3",
                        },
                    ],
                }
            ),
        ]
    )
    monkeypatch.setattr(service._files_resource, "get", get_mock)
    monkeypatch.setattr(service._files_resource, "list", list_mock)
    client = _source_client(mock_service=service)

    first = client.get_metadata("file-1")
    second = client.get_metadata("file-1")
    files = list(client.iter_json_files("folder-1"))

    assert first.name == "file-1"
    assert second is first
    assert [f.file_id for f in files] == ["f1", "f2", "f3"]
    assert list(client._meta_cache) == ["file-1", "f1", "f2", "f3"]


def test_get_metadata_builds_file_with_fallback_id() -> None:
    service = MockDriveService(
        files_data={"file-9": mock_drive_file(file_id="file-9", name="", mime_type="application/json", size=7)}
    )
    client = _source_client(mock_service=service)

    meta = client.get_metadata("file-9")

    assert meta.file_id == "file-9"
    assert meta.name == "file-9"
    assert meta.mime_type == "application/json"
    assert meta.size_bytes == 7
