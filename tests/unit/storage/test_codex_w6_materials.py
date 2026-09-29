"""Production material-admission regressions for worker 6's review findings."""

from __future__ import annotations

import http.client
import json
import sqlite3
import urllib.error
import zipfile
from collections.abc import Iterator
from email.message import Message
from io import BytesIO
from pathlib import Path

import pytest

from polylogue.storage.blob_store import BlobStore
from polylogue.storage.materials import (
    acquire_material,
    admit_material,
    admit_material_file,
    get_material,
    link_material,
    list_materials,
    read_material,
)
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL


@pytest.fixture
def material_db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SOURCE_DDL)
    try:
        yield conn
    finally:
        conn.close()


def test_material_json_admission_reads_complete_document(material_db: sqlite3.Connection, tmp_path: Path) -> None:
    payload = json.dumps({"value": "x" * 2_000_100}).encode()
    observation = admit_material(
        material_db, blob_store=BlobStore(tmp_path / "blobs"),
        source_uri="https://example.test/large.json", referrer_ref="test:json",
        observed_at_ms=100, payload=payload, media_type="application/json",
    )
    assert observation.acquisition_state == "acquired"
    assert observation.extraction_manifest["json_type"] == "dict"
    assert "diagnostic" not in observation.extraction_manifest


def test_material_ndjson_admission_parses_each_record(material_db: sqlite3.Connection, tmp_path: Path) -> None:
    observation = admit_material(
        material_db, blob_store=BlobStore(tmp_path / "blobs"),
        source_uri="https://example.test/records.ndjson", referrer_ref="test:ndjson",
        observed_at_ms=100, payload=b'{"n":1}\n\n{"n":2}\n', media_type="application/ndjson",
    )
    assert observation.acquisition_state == "acquired"
    assert observation.extraction_manifest["record_count"] == 2


def test_material_zip_manifest_does_not_copy_filenames(material_db: sqlite3.Connection, tmp_path: Path) -> None:
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("private-name-" + "x" * 12_000, b"content")
    observation = admit_material(
        material_db, blob_store=BlobStore(tmp_path / "blobs"),
        source_uri="https://example.test/a.zip", referrer_ref="test:zip",
        observed_at_ms=100, payload=payload.getvalue(), media_type="application/zip",
    )
    manifest = observation.extraction_manifest
    assert manifest["entry_count"] == 1
    assert manifest["uncompressed_bytes"] == 7
    assert "entries" not in manifest
    assert "private-name" not in json.dumps(manifest)
    assert read_material(material_db, observation.material_id, blob_store=BlobStore(tmp_path / "blobs")) == payload.getvalue()


def test_material_readmission_returns_persisted_metadata_and_creation_time(material_db: sqlite3.Connection, tmp_path: Path) -> None:
    store = BlobStore(tmp_path / "blobs")
    common = dict(blob_store=store, source_uri="https://example.test/item", referrer_ref="test:metadata", payload=b"text")
    first = admit_material(material_db, observed_at_ms=100, filename="first.txt", media_type="text/plain", **common)
    second = admit_material(
        material_db, observed_at_ms=200, filename="renamed.md", media_type="text/markdown",
        media_charset="utf-8", privacy_classification="restricted", **common,
    )
    assert second == get_material(material_db, first.material_id)
    assert second.created_at_ms == 100
    assert second.acquired_at_ms == 200
    assert (second.filename, second.media_type, second.media_charset, second.privacy_classification) == (
        "renamed.md", "text/markdown", "utf-8", "restricted",
    )


@pytest.mark.parametrize("payload,media_type,expected", [(b"valid", "text/plain", "acquired"), (b"{", "application/json", "malformed")])
def test_material_readmission_does_not_duplicate_itself(
    material_db: sqlite3.Connection, tmp_path: Path, payload: bytes, media_type: str, expected: str,
) -> None:
    common = dict(blob_store=BlobStore(tmp_path / "blobs"), source_uri="https://example.test/same", referrer_ref="test:same",
                  observed_at_ms=100, payload=payload, media_type=media_type)
    first = admit_material(material_db, **common)
    repeated = admit_material(material_db, **common)
    assert repeated.material_id == first.material_id
    assert repeated.acquisition_state == expected


@pytest.mark.parametrize("url", ["http://[broken", "http://example.test:bad/path"])
def test_material_invalid_url_is_a_durable_malformed_claim(
    material_db: sqlite3.Connection, tmp_path: Path, url: str,
) -> None:
    observation = acquire_material(
        material_db, source_uri=url, referrer_ref="test:bad-url", observed_at_ms=100,
        blob_store=BlobStore(tmp_path / "blobs"),
    )
    assert observation.acquisition_state == "malformed"
    assert not observation.retryable
    assert get_material(material_db, observation.material_id) == observation


def test_material_invalid_http_target_is_a_durable_claim(
    material_db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_target(*args: object, **kwargs: object) -> None:
        raise http.client.InvalidURL("control character in target")
    monkeypatch.setattr("polylogue.storage.materials._acquire_response", invalid_target)
    observation = acquire_material(
        material_db, source_uri="https://example.test/bad", referrer_ref="test:bad-target", observed_at_ms=100,
        blob_store=BlobStore(tmp_path / "blobs"),
    )
    assert observation.acquisition_state == "malformed"
    assert get_material(material_db, observation.material_id) == observation


def test_material_evidence_listing_returns_one_observation_per_identity(material_db: sqlite3.Connection, tmp_path: Path) -> None:
    observation = admit_material(material_db, blob_store=None, source_uri="https://example.test/item",
                                 referrer_ref="test:link", observed_at_ms=100)
    for relation in ("supports", "refers_to"):
        link_material(material_db, observation.material_id, "work:one", relation=relation, observed_at_ms=100)
    assert [item.material_id for item in list_materials(material_db, evidence_ref="work:one")] == [observation.material_id]


def test_material_chunked_response_ignores_stale_content_length(
    material_db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response(BytesIO):
        headers: Message

    response = Response(b"complete")
    response.headers = Message()
    response.headers["Content-Type"] = "text/plain"
    response.headers["Transfer-Encoding"] = "chunked"
    response.headers["Content-Length"] = "999"
    monkeypatch.setattr("polylogue.storage.materials._acquire_response", lambda url, timeout: (response, url))
    observation = acquire_material(
        material_db, source_uri="https://example.test/chunked", referrer_ref="test:chunked", observed_at_ms=100,
        blob_store=BlobStore(tmp_path / "blobs"),
    )
    assert observation.acquisition_state == "acquired"
    assert observation.byte_size == 8


def test_material_http_error_retains_redirect_destination(
    material_db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_url = "https://example.test/final-gone"
    def error(*args: object, **kwargs: object) -> None:
        raise urllib.error.HTTPError(final_url, 410, "Gone", Message(), None)
    monkeypatch.setattr("polylogue.storage.materials._acquire_response", error)
    observation = acquire_material(
        material_db, source_uri="https://example.test/redirect", referrer_ref="test:redirect", observed_at_ms=100,
        blob_store=BlobStore(tmp_path / "blobs"),
    )
    assert observation.acquisition_state == "expired"
    assert observation.source_uri == "https://example.test/redirect"
    assert f"redirected to {final_url}" in observation.diagnostic


@pytest.mark.parametrize("exists", [True, False])
def test_material_relative_file_admission_retains_success_or_absence(
    material_db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exists: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path("material.txt")
    if exists:
        path.write_bytes(b"retained")
    observation = admit_material_file(
        material_db, path=path, referrer_ref="test:local", observed_at_ms=100,
        blob_store=BlobStore(tmp_path / "blobs"),
    )
    assert observation.source_uri == (tmp_path / "material.txt").as_uri()
    assert observation.acquisition_state == ("acquired" if exists else "unavailable")
    assert get_material(material_db, observation.material_id) == observation
