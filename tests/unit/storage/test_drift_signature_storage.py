"""Exact streamed storage for retained schema-drift signatures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.schemas.drift_sentinel import DriftSignature
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    iter_schema_drift_signature,
    list_schema_drift_samples,
    record_schema_drift_sample,
    summarize_schema_drift_since,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _ops_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    initialize_archive_tier(conn, ArchiveTier.OPS)
    return conn


def _record(conn: sqlite3.Connection, signature: DriftSignature, **overrides: object) -> str:
    fields: dict[str, object] = {
        "origin": "claude-code-session",
        "element_kind": "session_record",
        "classification": "field_changed",
        "signature_chunks": signature.iter_utf8_chunks(),
        "signature_byte_count": signature.byte_count,
        "native_id_example": "native-example",
        "raw_id": "raw-example",
        "observed_at_ms": 10_000,
    }
    fields.update(overrides)
    return record_schema_drift_sample(conn, **fields)  # type: ignore[arg-type]


def test_drift_signature_exactly_replays_and_compares_utf8_surrogatepass(tmp_path: Path) -> None:
    text = "α,β,\ud800,z"
    encoded = text.encode("utf-8", errors="surrogatepass")
    signature = DriftSignature.from_utf8_chunks(
        (encoded[:1], encoded[1:4], encoded[4:]),
        directory=tmp_path,
        inline_limit_bytes=2,
    )
    assert signature.byte_count == len(text.encode("utf-8", errors="surrogatepass"))
    assert b"".join(signature.iter_utf8_chunks(chunk_bytes=3)) == text.encode("utf-8", errors="surrogatepass")
    assert signature._path is not None and signature._path.parent == tmp_path
    assert signature.compare(DriftSignature.from_text(text, directory=tmp_path)) == 0
    assert signature.compare(DriftSignature.from_text("α,β,\ud800,y", directory=tmp_path)) > 0
    assert signature.compare(DriftSignature.from_text("", directory=tmp_path)) > 0
    assert DriftSignature.from_text("", directory=tmp_path).byte_count == 0


def test_storage_rechunks_large_signature_below_sqlite_cell_limit_and_metadata_omits_content(
    tmp_path: Path,
) -> None:
    conn = _ops_conn(tmp_path / "ops.db")
    try:
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 32 * 1024)
        text = "λ," * 24_000
        signature = DriftSignature.from_text(text, directory=tmp_path, inline_limit_bytes=64)
        sample_id = _record(conn, signature)

        metadata = list_schema_drift_samples(conn, limit=5)
        assert len(metadata) == 1
        assert metadata[0].signature_byte_count == len(text.encode("utf-8", errors="surrogatepass"))
        assert not hasattr(metadata[0], "unseen_key_signature")
        chunks = b"".join(iter_schema_drift_signature(conn, sample_id))
        assert chunks == text.encode("utf-8", errors="surrogatepass")
        assert conn.execute(
            "SELECT MAX(length(chunk_bytes)), SUM(length(chunk_bytes)) FROM schema_drift_signature_chunks"
        ).fetchone() == (4096, len(chunks))
        summary = summarize_schema_drift_since(conn, since_ms=0)
        assert [(item.origin, item.total, item.risky) for item in summary] == [("claude-code-session", 1, 1)]
        assert summary[0].example_native_ids == ("native-example",)
    finally:
        conn.close()


def test_signature_iterator_failure_rolls_back_sample_and_chunks(tmp_path: Path) -> None:
    conn = _ops_conn(tmp_path / "ops.db")

    def fail_after_one_chunk():
        yield b"a" * 8192
        raise RuntimeError("synthetic signature read failure")

    try:
        with pytest.raises(RuntimeError, match="synthetic signature read failure"):
            record_schema_drift_sample(
                conn,
                origin="claude-code-session",
                element_kind="session_record",
                classification="field_changed",
                signature_chunks=fail_after_one_chunk(),
                signature_byte_count=8192,
                native_id_example="native-failure",
                raw_id="raw-failure",
                observed_at_ms=10_000,
                sample_id="failing-sample",
            )
        assert conn.execute("SELECT COUNT(*) FROM schema_drift_samples").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM schema_drift_signature_chunks").fetchone()[0] == 0
    finally:
        conn.close()


def test_pruning_sample_cascades_signature_chunks(tmp_path: Path) -> None:
    conn = _ops_conn(tmp_path / "ops.db")
    try:
        old = DriftSignature.from_text("old-signature" * 700, directory=tmp_path)
        old_id = _record(conn, old, sample_id="old-sample", observed_at_ms=1)
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM schema_drift_signature_chunks WHERE sample_id = ?", (old_id,)
            ).fetchone()[0]
            > 1
        )

        current = DriftSignature.from_text("current", directory=tmp_path)
        _record(conn, current, sample_id="current-sample", observed_at_ms=5_000_000_000)
        assert (
            conn.execute("SELECT COUNT(*) FROM schema_drift_samples WHERE sample_id = ?", (old_id,)).fetchone()[0] == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM schema_drift_signature_chunks WHERE sample_id = ?", (old_id,)
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()
