"""Focused contracts for live Drive-hosted attachment byte resolution (polylogue-83u.2).

`fetch_live_drive_attachment_bytes` is the piece that runs INSIDE the live
Drive client's iterator scope (see `iter_drive_raw_data`) and injects fetched
bytes into the raw payload so the ordinary chunk parser
(`drive_support_attachments.attachment_from_doc`) can turn them into
`ParsedAttachment.inline_bytes` with no further changes.
"""

from __future__ import annotations

import base64

import pytest

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.sources.drive.attachment_fetch import fetch_live_drive_attachment_bytes
from polylogue.sources.parsers.drive_support_attachments import (
    DRIVE_LIVE_FETCH_DATA_KEY,
    attachment_from_doc,
)


def _doc(value: JSONValue) -> JSONDocument:
    assert isinstance(value, dict)
    return value


def _items(value: JSONValue) -> list[JSONValue]:
    assert isinstance(value, list)
    return value


def test_fetch_resolves_dict_form_drive_document() -> None:
    payload: JSONDocument = {
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "hi"},
                {"role": "model", "driveDocument": {"id": "att-1", "name": "doc.txt", "mimeType": "text/plain"}},
            ]
        }
    }

    resolved, stats = fetch_live_drive_attachment_bytes(payload, lambda file_id: f"bytes-for-{file_id}".encode())

    assert stats.fetched_count == 1
    assert stats.failed_count == 0
    chunks = _items(_doc(_doc(resolved)["chunkedPrompt"])["chunks"])
    doc = _doc(_doc(chunks[1])["driveDocument"])
    assert doc["id"] == "att-1"
    assert doc["name"] == "doc.txt"
    fetched_b64 = doc[DRIVE_LIVE_FETCH_DATA_KEY]
    assert isinstance(fetched_b64, str)
    assert base64.b64decode(fetched_b64) == b"bytes-for-att-1"

    attachment = attachment_from_doc(doc, "msg-1")
    assert attachment is not None
    assert attachment.upload_origin == "drive"
    assert attachment.inline_bytes == b"bytes-for-att-1"
    assert attachment.provider_attachment_id == "att-1"


def test_fetch_resolves_bare_string_doc_id() -> None:
    payload: JSONDocument = {"driveDocument": "att-string-id"}

    resolved, stats = fetch_live_drive_attachment_bytes(payload, lambda file_id: b"raw-bytes")

    assert stats.fetched_count == 1
    doc = _doc(_doc(resolved)["driveDocument"])
    assert doc["id"] == "att-string-id"
    attachment = attachment_from_doc(doc, None)
    assert attachment is not None
    assert attachment.inline_bytes == b"raw-bytes"


def test_fetch_prefers_file_id_over_attachment_id_when_both_present() -> None:
    # attachment_from_doc's own native-id precedence (#1252): `id` is a
    # provider attachment handle, `fileId` is the real Drive file id that
    # DriveSourceClient.download_bytes needs. A live fetch must request the
    # SAME id the parser records as provider_file_id, or real payloads
    # carrying both fields would request the wrong handle and fail live.
    payload: JSONDocument = {"driveDocument": {"id": "attachment-handle-1", "fileId": "drive-file-1"}}
    requested_ids: list[str] = []

    def fetch(file_id: str) -> bytes:
        requested_ids.append(file_id)
        return b"drive-file-bytes"

    resolved, stats = fetch_live_drive_attachment_bytes(payload, fetch)

    assert requested_ids == ["drive-file-1"]
    assert stats.fetched_count == 1
    doc = _doc(_doc(resolved)["driveDocument"])
    fetched_b64 = doc[DRIVE_LIVE_FETCH_DATA_KEY]
    assert isinstance(fetched_b64, str)
    assert base64.b64decode(fetched_b64) == b"drive-file-bytes"


def test_fetch_never_uses_shared_drive_id_as_a_file_id() -> None:
    # driveId is the shared-drive CONTAINER id, never a downloadable file id.
    # A doc carrying only driveId (no id/fileId) has no live handle to fetch.
    payload: JSONDocument = {"driveDocument": {"driveId": "shared-drive-container-1", "name": "doc.txt"}}
    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return b"should-never-be-fetched"

    resolved, stats = fetch_live_drive_attachment_bytes(payload, fetch)

    assert calls == []
    assert stats.fetched_count == 0
    doc = _doc(_doc(resolved)["driveDocument"])
    assert DRIVE_LIVE_FETCH_DATA_KEY not in doc


@pytest.mark.parametrize("field", ["driveImage", "driveAudio", "driveVideo"])
def test_fetch_resolves_media_fields(field: str) -> None:
    payload: JSONDocument = {"chunk": {field: {"id": "media-1"}}}

    resolved, stats = fetch_live_drive_attachment_bytes(payload, lambda file_id: b"media-bytes")

    assert stats.fetched_count == 1
    doc = _doc(_doc(_doc(resolved)["chunk"])[field])
    fetched_b64 = doc[DRIVE_LIVE_FETCH_DATA_KEY]
    assert isinstance(fetched_b64, str)
    assert base64.b64decode(fetched_b64) == b"media-bytes"


def test_fetch_resolves_list_of_docs() -> None:
    payload: JSONDocument = {"driveDocuments": [{"id": "att-a"}, {"id": "att-b"}]}

    def fetch(file_id: str) -> bytes:
        return f"{file_id}-bytes".encode()

    resolved, stats = fetch_live_drive_attachment_bytes(payload, fetch)

    assert stats.fetched_count == 2
    docs = _items(_doc(resolved)["driveDocuments"])
    doc_a = _doc(docs[0])
    doc_b = _doc(docs[1])
    b64_a = doc_a[DRIVE_LIVE_FETCH_DATA_KEY]
    b64_b = doc_b[DRIVE_LIVE_FETCH_DATA_KEY]
    assert isinstance(b64_a, str)
    assert isinstance(b64_b, str)
    assert base64.b64decode(b64_a) == b"att-a-bytes"
    assert base64.b64decode(b64_b) == b"att-b-bytes"


def test_fetch_failure_leaves_attachment_honestly_unfetched() -> None:
    payload: JSONDocument = {"driveDocument": {"id": "att-broken", "name": "broken.bin"}}

    def fetch(file_id: str) -> bytes:
        raise RuntimeError(f"boom for {file_id}")

    resolved, stats = fetch_live_drive_attachment_bytes(payload, fetch)

    assert stats.fetched_count == 0
    assert stats.failed_count == 1
    assert "att-broken" in stats.failures[0]
    doc = _doc(_doc(resolved)["driveDocument"])
    assert DRIVE_LIVE_FETCH_DATA_KEY not in doc
    attachment = attachment_from_doc(doc, None)
    assert attachment is not None
    assert attachment.inline_bytes is None  # honest-unfetched: no synthetic bytes/hash


def test_fetch_skips_oversize_attachment() -> None:
    payload: JSONDocument = {"driveDocument": {"id": "att-huge"}}

    resolved, stats = fetch_live_drive_attachment_bytes(
        payload,
        lambda file_id: b"x" * 100,
        max_attachment_bytes=10,
    )

    assert stats.fetched_count == 0
    assert stats.skipped_too_large_count == 1
    doc = _doc(_doc(resolved)["driveDocument"])
    assert DRIVE_LIVE_FETCH_DATA_KEY not in doc


def test_fetch_does_not_touch_inline_or_file_data_or_youtube_fields() -> None:
    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return b"unused"

    payload: JSONDocument = {
        "inlineFile": {"mimeType": "text/plain", "data": base64.b64encode(b"already inline").decode()},
        "youtubeVideo": {"id": "yt-1"},
        "parts": [{"fileData": {"fileUri": "https://example.com/x", "mimeType": "text/plain"}}],
    }

    resolved, stats = fetch_live_drive_attachment_bytes(payload, fetch)

    assert calls == []
    assert stats.fetched_count == 0
    assert resolved == payload


def test_fetch_skips_already_resolved_doc() -> None:
    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return b"should-not-be-called-twice"

    already_resolved_b64 = base64.b64encode(b"cached bytes").decode()
    payload: JSONDocument = {"driveDocument": {"id": "att-1", DRIVE_LIVE_FETCH_DATA_KEY: already_resolved_b64}}

    resolved, stats = fetch_live_drive_attachment_bytes(payload, fetch)

    assert calls == []
    assert stats.fetched_count == 0
    doc = _doc(_doc(resolved)["driveDocument"])
    assert doc[DRIVE_LIVE_FETCH_DATA_KEY] == already_resolved_b64
