"""Material admission regressions through the production admission routes."""

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
    MaterialObservation,
    get_material,
    link_material,
    list_materials,
    prepare_material,
    prepare_material_acquisition,
    prepare_material_file,
    read_material,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.material_preparation import apply_material_preparation, material_publisher


@pytest.fixture
def material_db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    with ArchiveStore(tmp_path, initialize=True, read_only=False):
        pass
    conn = sqlite3.connect(tmp_path / "source.db")
    try:
        yield conn
    finally:
        conn.close()


def _admit(conn: sqlite3.Connection, tmp_path: Path, payload: bytes, media_type: str, name: str) -> MaterialObservation:
    return apply_material_preparation(
        conn,
        prepared=prepare_material(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri=f"https://example.test/{name}",
            referrer_ref=f"test:{name}",
            payload=payload,
            media_type=media_type,
        ),
        observed_at_ms=100,
    )


def test_material_json_admission_validates_the_complete_document(
    material_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """A valid document past the old 2 MB slice was declared malformed."""
    observation = _admit(
        material_db, tmp_path, json.dumps({"value": "x" * 2_000_100}).encode(), "application/json", "large.json"
    )
    assert observation.acquisition_state == "acquired"
    assert observation.extraction_manifest["json_type"] == "dict"
    # Anti-vacuity: the whole document is still validated, so trailing
    # garbage after a complete value keeps the malformed verdict.
    broken = _admit(material_db, tmp_path, b'{"a": 1} trailing', "application/json", "broken.json")
    assert broken.acquisition_state == "malformed"


def test_material_ndjson_admission_parses_each_record(material_db: sqlite3.Connection, tmp_path: Path) -> None:
    """A two-record stream is not one JSON value and must not read as 'Extra data'."""
    observation = _admit(material_db, tmp_path, b'{"n":1}\n\n{"n":2}\n', "application/ndjson", "records.ndjson")
    assert observation.acquisition_state == "acquired"
    assert observation.extraction_manifest["record_count"] == 2


def test_material_zip_manifest_does_not_copy_entry_names(material_db: sqlite3.Connection, tmp_path: Path) -> None:
    """One 12 KB entry name was copied verbatim into durable manifest metadata."""
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("private-name-" + "x" * 12_000, b"content")
    observation = _admit(material_db, tmp_path, payload.getvalue(), "application/zip", "a.zip")
    manifest = observation.extraction_manifest
    assert (manifest["entry_count"], manifest["uncompressed_bytes"]) == (1, 7)
    assert "private-name" not in json.dumps(manifest)
    assert (
        read_material(material_db, observation.material_id, blob_store=BlobStore(tmp_path / "blobs"))
        == payload.getvalue()
    )


def test_material_readmission_returns_the_persisted_row(material_db: sqlite3.Connection, tmp_path: Path) -> None:
    """Readmission reported its own arguments while the stored row kept the originals."""
    store = BlobStore(tmp_path / "blobs")
    first = apply_material_preparation(
        material_db,
        prepared=prepare_material(
            blob_store=material_publisher(material_db, store),
            source_uri="https://example.test/item",
            referrer_ref="test:metadata",
            payload=b"text",
            filename="first.txt",
            media_type="text/plain",
        ),
        observed_at_ms=100,
    )
    second = apply_material_preparation(
        material_db,
        prepared=prepare_material(
            blob_store=material_publisher(material_db, store),
            source_uri="https://example.test/item",
            referrer_ref="test:metadata",
            payload=b"text",
            filename="renamed.md",
            media_type="text/markdown",
            privacy_classification="restricted",
        ),
        observed_at_ms=200,
    )
    assert second == get_material(material_db, first.material_id)
    assert (second.created_at_ms, second.acquired_at_ms) == (100, 200)
    assert (second.filename, second.media_type, second.privacy_classification) == ("first.txt", "text/plain", "private")


@pytest.mark.parametrize(
    ("payload", "media_type", "expected"),
    [(b"valid", "text/plain", "acquired"), (b"{", "application/json", "malformed")],
)
def test_material_readmission_does_not_duplicate_itself(
    material_db: sqlite3.Connection, tmp_path: Path, payload: bytes, media_type: str, expected: str
) -> None:
    """The same observation matched its own blob and was rewritten as 'duplicate'."""
    first = _admit(material_db, tmp_path, payload, media_type, "same")
    repeated = _admit(material_db, tmp_path, payload, media_type, "same")
    assert repeated.material_id == first.material_id
    assert repeated.acquisition_state == expected
    # A distinct observation carrying the same bytes is still a duplicate.
    other = _admit(material_db, tmp_path, payload, media_type, "other")
    assert other.acquisition_state == "duplicate"


@pytest.mark.parametrize("url", ["http://[broken", "http://example.test:bad/path"])
def test_material_invalid_url_is_a_durable_malformed_claim(
    material_db: sqlite3.Connection, tmp_path: Path, url: str
) -> None:
    """URL parsing raised out of the API before any durable claim was written."""
    observation = apply_material_preparation(
        material_db,
        prepared=prepare_material_acquisition(
            source_uri=url,
            referrer_ref="test:bad-url",
            blob_store=material_publisher(material_db, BlobStore(tmp_path / "blobs")),
        ),
        observed_at_ms=100,
    )
    assert observation.acquisition_state == "malformed"
    assert not observation.retryable
    assert get_material(material_db, observation.material_id) == observation


def test_material_invalid_http_target_is_a_durable_claim(
    material_db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invalid_target(*_args: object, **_kwargs: object) -> None:
        raise http.client.InvalidURL("control character in target")

    monkeypatch.setattr("polylogue.storage.materials._acquire_response", invalid_target)
    observation = apply_material_preparation(
        material_db,
        prepared=prepare_material_acquisition(
            source_uri="https://example.test/bad",
            referrer_ref="test:bad-target",
            blob_store=material_publisher(material_db, BlobStore(tmp_path / "blobs")),
        ),
        observed_at_ms=100,
    )
    assert observation.acquisition_state == "malformed"
    assert get_material(material_db, observation.material_id) == observation


def test_material_evidence_listing_returns_one_observation_per_identity(
    material_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    """Two relations to one evidence ref listed the same observation twice."""
    observation = apply_material_preparation(
        material_db,
        prepared=prepare_material(
            blob_store=material_publisher(material_db, BlobStore(tmp_path / "blobs")),
            source_uri="https://example.test/item",
            referrer_ref="test:link",
        ),
        observed_at_ms=100,
    )
    link_material(material_db, observation.material_id, "work:one", relation="supports", observed_at_ms=100)
    link_material(material_db, observation.material_id, "work:one", relation="refers_to", observed_at_ms=100)
    assert [item.material_id for item in list_materials(material_db, evidence_ref="work:one")] == [
        observation.material_id
    ]


def test_material_chunked_response_ignores_stale_content_length(
    material_db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete chunk-framed body with a stale larger Content-Length was stored as partial."""

    class Response(BytesIO):
        headers: Message

    response = Response(b"complete")
    response.headers = Message()
    response.headers["Content-Type"] = "text/plain"
    response.headers["Transfer-Encoding"] = "chunked"
    response.headers["Content-Length"] = "999"
    monkeypatch.setattr("polylogue.storage.materials._acquire_response", lambda url, timeout: (response, url))
    observation = apply_material_preparation(
        material_db,
        prepared=prepare_material_acquisition(
            source_uri="https://example.test/chunked",
            referrer_ref="test:chunked",
            blob_store=material_publisher(material_db, BlobStore(tmp_path / "blobs")),
        ),
        observed_at_ms=100,
    )
    assert observation.acquisition_state == "acquired"
    assert observation.byte_size == 8


def test_material_http_error_retains_redirect_destination(
    material_db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A redirected hop's HTTP failure lost which destination failed."""
    final_url = "https://example.test/final-gone"

    def error(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError(final_url, 410, "Gone", Message(), None)

    monkeypatch.setattr("polylogue.storage.materials._acquire_response", error)
    observation = apply_material_preparation(
        material_db,
        prepared=prepare_material_acquisition(
            source_uri="https://example.test/redirect",
            referrer_ref="test:redirect",
            blob_store=material_publisher(material_db, BlobStore(tmp_path / "blobs")),
        ),
        observed_at_ms=100,
    )
    assert observation.acquisition_state == "expired"
    assert observation.source_uri == "https://example.test/redirect"
    assert f"redirected to {final_url}" in observation.diagnostic


@pytest.mark.parametrize("exists", [True, False])
def test_material_relative_file_admission_retains_success_or_absence(
    material_db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exists: bool
) -> None:
    """``Path.as_uri`` raised for a relative path before any observation was recorded."""
    monkeypatch.chdir(tmp_path)
    path = Path("material.txt")
    if exists:
        path.write_bytes(b"retained")
    observation = apply_material_preparation(
        material_db,
        prepared=prepare_material_file(
            path=path,
            referrer_ref="test:local",
            blob_store=material_publisher(material_db, BlobStore(tmp_path / "blobs")),
        ),
        observed_at_ms=100,
    )
    assert observation.source_uri == (tmp_path / "material.txt").as_uri()
    assert observation.acquisition_state == ("acquired" if exists else "unavailable")
    assert get_material(material_db, observation.material_id) == observation
