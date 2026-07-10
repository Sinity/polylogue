"""Retained-artifact inspection contracts for Hermes SQLite state."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.core.enums import ArtifactSupportStatus, Provider
from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.runtime import RawSessionRecord


@pytest.fixture
def blob_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[BlobStore]:
    root = tmp_path / "blobs"
    store = BlobStore(root)
    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: root)
    monkeypatch.setattr("polylogue.storage.blob_store.blob_store_root", lambda: root, raising=False)
    reset_blob_store()
    yield store
    reset_blob_store()


def _write_hermes_v16(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version VALUES (16);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                source TEXT,
                model_config TEXT,
                parent_session_id TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                tool_calls TEXT,
                observed INTEGER,
                active INTEGER,
                compacted INTEGER
            );
            INSERT INTO sessions VALUES ('session-1', 1.0, 'cli', '{}', NULL);
            INSERT INTO messages VALUES (1, 'session-1', 'user', 'hello', 1.0, '[]', 0, 1, 0);
            """
        )


def _write_generic_sqlite_lookalike(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT,
                model TEXT,
                model_config TEXT,
                started_at REAL,
                title TEXT
            );
            CREATE TABLE messages (
                id INTEGER,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp REAL,
                tool_calls TEXT
            );
            """
        )


def _record(
    store: BlobStore,
    path: Path,
    *,
    raw_id: str,
    source_path: str = "/original/profile/state.db",
) -> RawSessionRecord:
    blob_hash, blob_size = store.write_from_path(path)
    assert raw_id != blob_hash
    return RawSessionRecord(
        raw_id=raw_id,
        blob_hash=blob_hash,
        payload_provider=Provider.HERMES,
        source_name=Provider.HERMES.value,
        source_path=source_path,
        blob_size=blob_size,
        acquired_at="2026-07-10T00:00:00+00:00",
    )


def test_retained_v16_snapshot_is_contract_backed_parseable(
    blob_store: BlobStore,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "retained.sqlite3"
    _write_hermes_v16(snapshot)

    observation = inspect_raw_artifact(
        _record(
            blob_store,
            snapshot,
            raw_id="hermes:profile-a:revision-1",
            source_path="/original/profile/arbitrary-session-name.sqlite3",
        )
    )

    assert observation.payload_provider is Provider.HERMES
    assert observation.artifact_kind == "session_document"
    assert observation.classification_reason == "Hermes state.db SQLite archive marker"
    assert observation.support_status is ArtifactSupportStatus.SUPPORTED_PARSEABLE
    assert observation.resolved_package_version == "state-db-v16"
    assert observation.resolved_element_kind == "state_db"
    assert observation.decode_error is None


def test_corrupt_retained_snapshot_is_decode_failed(blob_store: BlobStore, tmp_path: Path) -> None:
    snapshot = tmp_path / "state.db"
    snapshot.write_bytes(b"not a SQLite database")

    observation = inspect_raw_artifact(_record(blob_store, snapshot, raw_id="hermes:profile-a:corrupt"))

    assert observation.support_status is ArtifactSupportStatus.DECODE_FAILED
    assert observation.resolved_package_version is None
    assert observation.decode_error is not None


def test_generic_sqlite_lookalike_is_not_claimed_as_hermes(blob_store: BlobStore, tmp_path: Path) -> None:
    snapshot = tmp_path / "state.db"
    _write_generic_sqlite_lookalike(snapshot)

    observation = inspect_raw_artifact(_record(blob_store, snapshot, raw_id="hermes:profile-a:lookalike"))

    assert observation.support_status is ArtifactSupportStatus.DECODE_FAILED
    assert observation.artifact_kind == "unknown"
    assert observation.resolved_package_version is None
    assert observation.classification_reason != "Hermes state.db SQLite archive marker"
