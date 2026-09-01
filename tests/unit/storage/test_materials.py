from __future__ import annotations

import sqlite3
import urllib.error
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.storage.blob_liveness import inspect_blob_liveness
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.materials import (
    acquire_material,
    admit_material,
    admit_material_file,
    link_material,
    read_material,
)
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL


def _source_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SOURCE_DDL)
    return conn


def test_material_retains_bytes_and_links_without_session(tmp_path: Path) -> None:
    conn = _source_db()
    observation = admit_material(
        conn,
        blob_store=BlobStore(tmp_path / "blobs"),
        source_uri="https://example.test/result.zip",
        referrer_ref="message:codex-session:abc:m1",
        observed_at_ms=100,
        payload=b"not a session export",
        media_type="application/zip",
        privacy_classification="private",
    )
    link_material(
        conn,
        observation.material_id,
        "work-attempt:attempt-1",
        relation="supports",
        authority="provider",
        confidence=0.8,
        observed_at_ms=101,
    )
    row = conn.execute("SELECT acquisition_state, custody, byte_size, blob_hash FROM material_observations").fetchone()
    assert row[:3] == ("malformed", "retained", len(b"not a session export"))
    assert len(row[3]) == 32
    assert conn.execute("SELECT COUNT(*) FROM material_evidence_links").fetchone()[0] == 1


def test_failed_claim_is_queryable_and_synthetic_raw_bytes_are_rejected(tmp_path: Path) -> None:
    conn = _source_db()
    observation = admit_material(
        conn,
        blob_store=BlobStore(tmp_path / "blobs"),
        source_uri="https://expired.example/file",
        referrer_ref="agent:worker-1",
        observed_at_ms=200,
        state="expired",
        diagnostic="HTTP 410 Gone",
        retryable=True,
    )
    assert observation.blob_hash is None
    assert conn.execute("SELECT acquisition_state, diagnostic, retryable FROM material_observations").fetchone() == (
        "expired",
        "HTTP 410 Gone",
        1,
    )
    with pytest.raises(ValueError, match="synthetic"):
        admit_material(
            conn,
            blob_store=BlobStore(tmp_path / "blobs"),
            source_uri="synthetic:test",
            referrer_ref="test",
            observed_at_ms=201,
            payload=b"raw",
            privacy_classification="synthetic",
        )


def test_duplicate_bytes_remain_separate_observations_and_are_liveness_protected(tmp_path: Path) -> None:
    conn = _source_db()
    store = BlobStore(tmp_path / "blobs")
    first = admit_material(
        conn,
        blob_store=store,
        source_uri="https://one.example/file",
        referrer_ref="message:one",
        observed_at_ms=1,
        payload=b"same bytes",
        media_type="text/plain",
    )
    second = admit_material(
        conn,
        blob_store=store,
        source_uri="https://two.example/file",
        referrer_ref="message:two",
        observed_at_ms=2,
        payload=b"same bytes",
        media_type="text/plain",
    )
    assert first.material_id != second.material_id
    assert second.acquisition_state == "duplicate"
    assert read_material(conn, first.material_id, blob_store=store) == b"same bytes"
    assert (
        inspect_blob_liveness(
            conn,
            blob_hash=first.blob_hash or "",
            index_conn=None,
        ).state.value
        == "live"
    )


def test_text_manifest_does_not_copy_raw_content(tmp_path: Path) -> None:
    conn = _source_db()
    observation = admit_material(
        conn,
        blob_store=BlobStore(tmp_path / "blobs"),
        source_uri="file:///private/notes.txt",
        referrer_ref="agent:worker",
        observed_at_ms=3,
        payload=b"secret text that must remain in the blob",
        media_type="text/plain",
    )
    assert "text_prefix" not in observation.extraction_manifest
    assert observation.extraction_manifest["text"] == {"available": True, "encoding": "text/plain"}


def test_url_acquisition_keeps_redirect_provenance_and_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Response:
        headers = SimpleNamespace(get_content_type=lambda: "text/markdown", get_content_charset=lambda: "utf-8")

        def __init__(self) -> None:
            self.done = False

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://cdn.example/result.md"

        def read(self, size: int) -> bytes:
            if self.done:
                return b""
            self.done = True
            return b"# retained"

    monkeypatch.setattr("polylogue.storage.materials.urllib.request.urlopen", lambda *args, **kwargs: Response())
    conn = _source_db()
    observation = acquire_material(
        conn,
        blob_store=BlobStore(tmp_path / "blobs"),
        source_uri="https://example.test/result.md",
        referrer_ref="message:codex:1",
        observed_at_ms=10,
    )
    assert observation.acquisition_state == "acquired"
    assert observation.diagnostic == "redirected to https://cdn.example/result.md"
    assert read_material(conn, observation.material_id, blob_store=BlobStore(tmp_path / "blobs")) == b"# retained"


def test_url_and_file_failures_are_queryable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    conn = _source_db()
    monkeypatch.setattr(
        "polylogue.storage.materials.urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.HTTPError("url", 410, "Gone", Message(), None)),
    )
    expired = acquire_material(
        conn,
        blob_store=BlobStore(tmp_path / "blobs"),
        source_uri="https://example.test/old",
        referrer_ref="agent:a",
        observed_at_ms=1,
    )
    missing = admit_material_file(
        conn,
        blob_store=BlobStore(tmp_path / "blobs"),
        path=tmp_path / "missing.patch",
        referrer_ref="agent:a",
        observed_at_ms=2,
    )
    assert expired.acquisition_state == "expired"
    assert missing.acquisition_state == "unavailable"
    material_ids = [row[0] for row in conn.execute("SELECT material_id FROM material_observations")]
    assert len(material_ids) == 2
    assert all(material_ids)
