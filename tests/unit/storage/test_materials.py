from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.blob_store import BlobStore
from polylogue.storage.materials import admit_material, link_material
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
