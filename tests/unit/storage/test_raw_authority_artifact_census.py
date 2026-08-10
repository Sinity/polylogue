"""Focused tests for the quarantined raw-authority artifact census."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.storage.artifacts import raw_authority_census as census_module
from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.artifacts.raw_authority_census import (
    RawAuthorityBucket,
    scan_quarantined_raw_authority,
    write_artifact_observations,
)
from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    blob_root = root / "blob"
    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: blob_root)
    monkeypatch.setattr("polylogue.storage.blob_store.blob_store_root", lambda: blob_root, raising=False)
    reset_blob_store()
    yield root
    reset_blob_store()


def _chatgpt_payload(title: str) -> bytes:
    return json.dumps(
        {
            "title": title,
            "create_time": 1_700_000_000,
            "update_time": 1_700_000_100,
            "current_node": "m1",
            "mapping": {
                "root": {"id": "root", "message": None, "children": ["m1"]},
                "m1": {
                    "id": "m1",
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["hello"]},
                        "create_time": 1_700_000_050,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }
    ).encode()


def _tool_result_payload() -> bytes:
    return (
        b'{"messages":[{"role":"user","content":"hi"}],"chat_messages":[{"role":"user","text":"hi"}],'
        b'"mapping":{"a":{"message":{"role":"user","content":[{"type":"text","text":"hi"}]}}}}'
    )


def _write_raw(
    archive: ArchiveStore,
    *,
    raw_id: str,
    provider: Provider,
    payload: bytes,
    source_path: str,
    revision: RawRevisionEnvelope | None = None,
) -> None:
    archive.write_raw_payload(
        provider=provider,
        payload=payload,
        source_path=source_path,
        acquired_at_ms=1_700_000_000_000,
        raw_id=raw_id,
        revision=revision,
    )


def test_full_quarantine_census_is_mutually_exclusive_and_deterministic(archive: Path) -> None:
    duplicate_payload = _chatgpt_payload("duplicate")
    novel_payload = _chatgpt_payload("novel")
    unresolved_payload = _chatgpt_payload("unresolved")
    with ArchiveStore.open_existing(archive, read_only=False) as store:
        _write_raw(
            store,
            raw_id="raw-artifact",
            provider=Provider.CLAUDE_CODE,
            payload=_tool_result_payload(),
            source_path="/home/user/.claude/projects/p/tool-results/toolu.json",
        )
        _write_raw(
            store,
            raw_id="raw-duplicate",
            provider=Provider.CHATGPT,
            payload=duplicate_payload,
            source_path="/exports/duplicate.json",
        )
        _write_raw(
            store,
            raw_id="raw-indexed-twin",
            provider=Provider.CHATGPT,
            payload=duplicate_payload,
            source_path="/exports/twin.json",
        )
        _write_raw(
            store,
            raw_id="raw-novel",
            provider=Provider.CHATGPT,
            payload=novel_payload,
            source_path="/exports/novel.json",
        )
        _write_raw(
            store,
            raw_id="raw-unresolved",
            provider=Provider.CHATGPT,
            payload=unresolved_payload,
            source_path="/exports/unresolved.json",
            revision=RawRevisionEnvelope(
                logical_source_key="chatgpt:unresolved",
                kind=RawRevisionKind.FULL,
                source_revision="unresolved-v1",
                acquisition_generation=0,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        store.commit()
        with sqlite3.connect(archive / "source.db") as source_conn:
            source_conn.execute(
                "UPDATE raw_sessions SET revision_authority = 'asserted' WHERE raw_id = 'raw-indexed-twin'"
            )
        with sqlite3.connect(archive / "index.db") as index_conn:
            index_conn.execute(
                "INSERT INTO sessions (origin, native_id, content_hash, raw_id, created_at_ms, updated_at_ms) "
                "VALUES ('chatgpt-export', 'twin', ?, 'raw-indexed-twin', 0, 0)",
                (bytes.fromhex("01" * 32),),
            )
        # Missing bytes is a retained source row whose content-addressed file
        # was removed. The census must not ask inspect_raw_artifact to decode it.
        store.commit()
    with sqlite3.connect(archive / "source.db") as source_conn:
        source_conn.execute(
            "INSERT INTO raw_sessions (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms) "
            "VALUES ('raw-missing', 'chatgpt-export', '/exports/missing.json', 0, ?, 12, 1)",
            (bytes.fromhex("99" * 32),),
        )

    with (
        sqlite3.connect(f"file:{archive / 'source.db'}?mode=ro", uri=True) as source_conn,
        sqlite3.connect(f"file:{archive / 'index.db'}?mode=ro", uri=True) as index_conn,
    ):
        first = scan_quarantined_raw_authority(source_conn, index_conn, blob_store=BlobStore(archive / "blob"))
        second = scan_quarantined_raw_authority(source_conn, index_conn, blob_store=BlobStore(archive / "blob"))

    assert first.entries == second.entries
    assert first.total_quarantined_count == 5
    assert first.scanned_count == 5
    assert sum(first.counts().values()) == first.scanned_count
    assert first.counts() == {
        "artifact": 1,
        "terminal_byte_duplicate_superseded": 1,
        "novel_materialization_candidate": 1,
        "missing_bytes": 1,
        "unresolved_authority": 1,
    }
    assert first.entries_for(RawAuthorityBucket.TERMINAL_BYTE_DUPLICATE)[0].duplicate_of_raw_id == "raw-indexed-twin"
    assert [entry.raw_id for entry in first.entries] == sorted(entry.raw_id for entry in first.entries)


def test_artifact_apply_upserts_only_observations_and_preserves_authority(archive: Path) -> None:
    with ArchiveStore.open_existing(archive, read_only=False) as store:
        _write_raw(
            store,
            raw_id="raw-artifact",
            provider=Provider.CLAUDE_CODE,
            payload=_tool_result_payload(),
            source_path="/home/user/.claude/projects/p/tool-results/toolu.json",
        )
        store.commit()

    with sqlite3.connect(archive / "source.db") as conn:
        conn.row_factory = sqlite3.Row
        before = conn.execute(
            "SELECT raw_id, revision_authority, blob_hash FROM raw_sessions WHERE raw_id = 'raw-artifact'"
        ).fetchone()
        with sqlite3.connect(f"file:{archive / 'index.db'}?mode=ro", uri=True) as index_conn:
            census = scan_quarantined_raw_authority(conn, index_conn, blob_store=BlobStore(archive / "blob"))
        written = write_artifact_observations(conn, census.artifact_observations())
        conn.commit()
        after = conn.execute(
            "SELECT raw_id, revision_authority, blob_hash FROM raw_sessions WHERE raw_id = 'raw-artifact'"
        ).fetchone()
        observation = conn.execute(
            "SELECT raw_id, artifact_kind, parse_as_session FROM raw_artifacts WHERE raw_id = 'raw-artifact'"
        ).fetchone()

    assert written == 1
    assert before == after
    assert dict(observation) == {
        "raw_id": "raw-artifact",
        "artifact_kind": "tool_result_sidecar",
        "parse_as_session": 0,
    }


def test_logical_keyed_revision_with_indexed_twin_stays_unresolved(archive: Path) -> None:
    payload = _chatgpt_payload("logical-keyed")
    with ArchiveStore.open_existing(archive, read_only=False) as store:
        _write_raw(
            store,
            raw_id="raw-indexed-twin",
            provider=Provider.CHATGPT,
            payload=payload,
            source_path="/exports/indexed.json",
        )
        _write_raw(
            store,
            raw_id="raw-logical-keyed",
            provider=Provider.CHATGPT,
            payload=payload,
            source_path="/exports/revision.json",
            revision=RawRevisionEnvelope(
                logical_source_key="chatgpt:logical-keyed",
                kind=RawRevisionKind.FULL,
                source_revision="revision-1",
                acquisition_generation=1,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        store.commit()
    with sqlite3.connect(archive / "source.db") as source_conn:
        source_conn.execute("UPDATE raw_sessions SET revision_authority = 'asserted' WHERE raw_id = 'raw-indexed-twin'")
    with sqlite3.connect(archive / "index.db") as index_conn:
        index_conn.execute(
            "INSERT INTO sessions (origin, native_id, content_hash, raw_id, created_at_ms, updated_at_ms) "
            "VALUES ('chatgpt-export', 'indexed', ?, 'raw-indexed-twin', 0, 0)",
            (bytes.fromhex("02" * 32),),
        )

    with (
        sqlite3.connect(f"file:{archive / 'source.db'}?mode=ro", uri=True) as source_conn,
        sqlite3.connect(f"file:{archive / 'index.db'}?mode=ro", uri=True) as index_conn,
    ):
        report = scan_quarantined_raw_authority(source_conn, index_conn, blob_store=BlobStore(archive / "blob"))

    assert report.entries_for(RawAuthorityBucket.TERMINAL_BYTE_DUPLICATE) == ()
    unresolved = report.entries_for(RawAuthorityBucket.UNRESOLVED_AUTHORITY)
    assert [entry.raw_id for entry in unresolved] == ["raw-logical-keyed"]


def test_census_reuses_duplicate_authority_and_excludes_parser_failures(archive: Path) -> None:
    payload = _chatgpt_payload("duplicate")
    with ArchiveStore.open_existing(archive, read_only=False) as store:
        _write_raw(
            store,
            raw_id="raw-indexed-twin",
            provider=Provider.CHATGPT,
            payload=payload,
            source_path="/exports/indexed.json",
        )
        _write_raw(
            store,
            raw_id="raw-duplicate",
            provider=Provider.CHATGPT,
            payload=payload,
            source_path="/exports/duplicate.json",
        )
        _write_raw(
            store,
            raw_id="raw-parser-failed",
            provider=Provider.CHATGPT,
            payload=_chatgpt_payload("failed"),
            source_path="/exports/failed.json",
        )
        store.commit()
    with sqlite3.connect(archive / "source.db") as source_conn:
        source_conn.execute("UPDATE raw_sessions SET revision_authority = 'asserted' WHERE raw_id = 'raw-indexed-twin'")
        source_conn.execute("UPDATE raw_sessions SET parse_error = 'parser failed' WHERE raw_id = 'raw-parser-failed'")
    with sqlite3.connect(archive / "index.db") as index_conn:
        index_conn.execute(
            "INSERT INTO sessions (origin, native_id, content_hash, raw_id, created_at_ms, updated_at_ms) "
            "VALUES ('chatgpt-export', 'indexed', ?, 'raw-indexed-twin', 0, 0)",
            (bytes.fromhex("03" * 32),),
        )

    with (
        sqlite3.connect(f"file:{archive / 'source.db'}?mode=ro", uri=True) as source_conn,
        sqlite3.connect(f"file:{archive / 'index.db'}?mode=ro", uri=True) as index_conn,
    ):
        report = scan_quarantined_raw_authority(source_conn, index_conn, blob_store=BlobStore(archive / "blob"))

    assert [entry.raw_id for entry in report.entries] == ["raw-duplicate"]
    assert report.entries[0].bucket is RawAuthorityBucket.TERMINAL_BYTE_DUPLICATE


def test_census_passes_the_selected_archive_blob_store_to_inspection(
    archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"[]"
    with ArchiveStore.open_existing(archive, read_only=False) as store:
        _write_raw(
            store,
            raw_id="raw-explicit-blob-root",
            provider=Provider.CLAUDE_CODE,
            payload=payload,
            source_path="/exports/ordinary.json",
        )
        store.commit()

    with sqlite3.connect(archive / "source.db") as conn:
        blob_hash = bytes(conn.execute("SELECT blob_hash FROM raw_sessions").fetchone()[0]).hex()

    ambient_root = archive.parent / "ambient-blob"
    ambient_path = BlobStore(ambient_root).blob_path(blob_hash)
    ambient_path.parent.mkdir(parents=True, exist_ok=True)
    ambient_path.write_bytes(_chatgpt_payload("wrong ambient bytes"))
    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: ambient_root)
    monkeypatch.setattr("polylogue.storage.blob_store.blob_store_root", lambda: ambient_root)
    reset_blob_store()
    inspected_roots: list[Path] = []
    real_inspect_raw_artifact = inspect_raw_artifact

    def inspect_with_root(record: RawSessionRecord, *, blob_store: BlobStore | None = None) -> object:
        assert blob_store is not None
        inspected_roots.append(blob_store.root)
        return real_inspect_raw_artifact(record, blob_store=blob_store)

    monkeypatch.setattr(census_module, "inspect_raw_artifact", inspect_with_root)

    with (
        sqlite3.connect(f"file:{archive / 'source.db'}?mode=ro", uri=True) as source_conn,
        sqlite3.connect(f"file:{archive / 'index.db'}?mode=ro", uri=True) as index_conn,
    ):
        census = scan_quarantined_raw_authority(
            source_conn,
            index_conn,
            blob_store=BlobStore(archive / "blob"),
        )

    assert inspected_roots == [archive / "blob"]
    assert len(census.entries) == 1
