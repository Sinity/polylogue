"""Command and production-route coverage for raw-authority artifact census."""

from __future__ import annotations

import io
import json
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from devtools.command_catalog import COMMANDS
from devtools.raw_authority_artifact_census import main
from polylogue.archive.artifact_taxonomy import ArtifactClassification, ArtifactKind, classify_artifact
from polylogue.core.enums import Provider
from polylogue.maintenance.raw_authority_artifact_census import run_raw_authority_artifact_census
from polylogue.pipeline.services.ingest_worker import ingest_record
from polylogue.storage.artifacts.raw_authority_census import RawAuthorityBucket
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


def _artifact_payload() -> bytes:
    return (
        b'{"messages":[{"role":"user","content":"hi"}],"chat_messages":[{"role":"user","text":"hi"}],'
        b'"mapping":{"a":{"message":{"role":"user","content":[{"type":"text","text":"hi"}]}}}}'
    )


def _session_payload(title: str) -> bytes:
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


def _write_artifact(archive: Path) -> None:
    with ArchiveStore.open_existing(archive, read_only=False) as store:
        store.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=_artifact_payload(),
            source_path="/home/user/.claude/projects/p/tool-results/toolu.json",
            acquired_at_ms=1_700_000_000_000,
            raw_id="raw-artifact",
        )
        store.commit()


def test_census_is_catalogued_and_dry_run_receipt_is_immutable(archive: Path) -> None:
    _write_artifact(archive)
    receipt_path = archive.parent / "raw-authority-census.json"
    output = io.StringIO()

    assert "workspace raw-authority-artifact-census" in COMMANDS
    assert main(["--archive-root", str(archive), "--json"], stdout=output) == 1
    assert "requires --receipt" in output.getvalue()
    output = io.StringIO()
    assert main(["--archive-root", str(archive), "--json", "--receipt", str(receipt_path)], stdout=output) == 0
    payload = json.loads(output.getvalue())
    assert payload["receipt"]["mode"] == "dry_run"
    assert payload["receipt"]["counts"]["artifact"] == 1
    assert payload["receipt"]["counts"]["novel_materialization_candidate"] == 0
    assert "source_path" not in receipt_path.read_text(encoding="utf-8")

    output = io.StringIO()
    assert main(["--archive-root", str(archive), "--json", "--receipt", str(receipt_path)], stdout=output) == 1
    assert "immutable" in output.getvalue()
    with sqlite3.connect(archive / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)


def test_apply_requires_backup_and_writes_only_artifact_observation(
    archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_artifact(archive)
    with sqlite3.connect(archive / "source.db") as conn:
        raw_rows_before = conn.execute(
            "SELECT raw_id, revision_authority, blob_hash, blob_size, source_path FROM raw_sessions ORDER BY raw_id"
        ).fetchall()
    index_before = (archive / "index.db").read_bytes()
    blobs_before = {
        str(path.relative_to(archive / "blob")): path.read_bytes()
        for path in (archive / "blob").rglob("*")
        if path.is_file()
    }
    output = io.StringIO()
    assert main(["--archive-root", str(archive), "--apply", "--json"], stdout=output) == 1
    assert "backup-manifest" in output.getvalue()

    import polylogue.maintenance.raw_authority_artifact_census as maintenance

    monkeypatch.setattr(maintenance, "offline_maintenance_block_reason", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(maintenance, "_checkpoint_source_tier", lambda _conn: None)
    monkeypatch.setattr(maintenance, "validate_migration_backup_manifest", lambda *_args, **_kwargs: None)
    output = io.StringIO()
    assert (
        main(
            [
                "--archive-root",
                str(archive),
                "--apply",
                "--limit",
                "1",
                "--backup-manifest",
                str(archive.parent / "verified-manifest.json"),
                "--json",
            ],
            stdout=output,
        )
        == 0
    )
    payload = json.loads(output.getvalue())
    assert payload["observations_written"] == 1
    with sqlite3.connect(archive / "source.db") as conn:
        assert (
            conn.execute(
                "SELECT raw_id, revision_authority, blob_hash, blob_size, source_path FROM raw_sessions ORDER BY raw_id"
            ).fetchall()
            == raw_rows_before
        )
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        assert conn.execute("SELECT revision_authority FROM raw_sessions WHERE raw_id = 'raw-artifact'").fetchone() == (
            "quarantined",
        )
        assert conn.execute("SELECT artifact_kind, parse_as_session FROM raw_artifacts").fetchone() == (
            "tool_result_sidecar",
            0,
        )
    assert (archive / "index.db").read_bytes() == index_before
    assert {
        str(path.relative_to(archive / "blob")): path.read_bytes()
        for path in (archive / "blob").rglob("*")
        if path.is_file()
    } == blobs_before


def test_apply_rejects_receipt_option_before_mutating_source(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_artifact(archive)
    receipt_path = archive.parent / "apply-receipt.json"

    import polylogue.maintenance.raw_authority_artifact_census as maintenance

    monkeypatch.setattr(maintenance, "offline_maintenance_block_reason", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(maintenance, "_checkpoint_source_tier", lambda _conn: None)
    monkeypatch.setattr(maintenance, "validate_migration_backup_manifest", lambda *_args, **_kwargs: None)
    output = io.StringIO()

    assert (
        main(
            [
                "--archive-root",
                str(archive),
                "--apply",
                "--backup-manifest",
                str(archive.parent / "verified-manifest.json"),
                "--receipt",
                str(receipt_path),
                "--json",
            ],
            stdout=output,
        )
        == 1
    )
    assert "dry-run" in output.getvalue()
    with sqlite3.connect(archive / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)


def test_census_uses_active_index_pointer_for_duplicate_witness(archive: Path, tmp_path: Path) -> None:
    payload = _session_payload("duplicate")
    with ArchiveStore.open_existing(archive, read_only=False) as store:
        store.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=payload,
            source_path="/exports/duplicate.json",
            acquired_at_ms=1_700_000_000_000,
            raw_id="raw-duplicate",
        )
        store.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=payload,
            source_path="/exports/twin.json",
            acquired_at_ms=1_700_000_000_000,
            raw_id="raw-indexed-twin",
        )
        store.commit()

    with sqlite3.connect(archive / "source.db") as source_conn:
        source_conn.execute("UPDATE raw_sessions SET revision_authority = 'asserted' WHERE raw_id = 'raw-indexed-twin'")

    active_index = tmp_path / "index-generation" / "index.db"
    active_index.parent.mkdir()
    shutil.copy2(archive / "index.db", active_index)
    with sqlite3.connect(active_index) as index_conn:
        index_conn.execute(
            "INSERT INTO sessions (origin, native_id, content_hash, raw_id, created_at_ms, updated_at_ms) "
            "VALUES ('chatgpt-export', 'twin', ?, 'raw-indexed-twin', 0, 0)",
            (bytes.fromhex("01" * 32),),
        )
    (archive / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")

    report = run_raw_authority_artifact_census(archive, receipt_path=tmp_path / "pointer-receipt.json")
    duplicate = report.census.entries_for(RawAuthorityBucket.TERMINAL_BYTE_DUPLICATE)

    assert [entry.raw_id for entry in duplicate] == ["raw-duplicate"]
    assert duplicate[0].duplicate_of_raw_id == "raw-indexed-twin"


def test_production_ingest_route_excludes_artifact_and_bypass_is_load_bearing(
    archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = (
        b'{"type":"user","sessionId":"s1","uuid":"u1","message":{"role":"user","content":"hi"}}\n'
        b'{"type":"assistant","sessionId":"s1","uuid":"u2","parentUuid":"u1",'
        b'"message":{"role":"assistant","content":"hey"}}\n'
    )
    blob_root = archive / "blob"
    blob_hash, blob_size = BlobStore(blob_root).write_from_bytes(payload)
    record = RawSessionRecord(
        raw_id="raw-artifact-route",
        blob_hash=blob_hash,
        source_name="claude-code",
        payload_provider=Provider.CLAUDE_CODE,
        source_path="/home/user/.claude/projects/p/tool-results/toolu.jsonl",
        source_index=0,
        blob_size=blob_size,
        acquired_at="2026-08-05T00:00:00+00:00",
    )

    excluded = ingest_record(record, str(archive), "advisory", blob_root_str=str(blob_root))
    assert excluded.sessions == []
    assert excluded.validation_status == "skipped"

    excluded_fast = ingest_record(record, str(archive), "off", blob_root_str=str(blob_root))
    assert excluded_fast.sessions == []
    assert excluded_fast.validation_status == "skipped"

    real_classify_artifact = classify_artifact
    monkeypatch.setattr(
        "polylogue.pipeline.services.ingest_worker.classify_artifact",
        lambda *_args, **_kwargs: ArtifactClassification(
            provider=Provider.CLAUDE_CODE,
            kind=ArtifactKind.SESSION_RECORD_STREAM,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=1,
            reason="test bypass",
        ),
    )
    bypassed = ingest_record(record, str(archive), "advisory", blob_root_str=str(blob_root))
    assert bypassed.sessions, "removing the artifact classification gate must change the route"
    monkeypatch.setattr("polylogue.pipeline.services.ingest_worker.classify_artifact", real_classify_artifact)

    history_record = RawSessionRecord(
        raw_id="raw-file-history-route",
        blob_hash=None,
        source_name="claude-code",
        payload_provider=Provider.CLAUDE_CODE,
        source_path="/home/user/.claude/projects/p/12345678-1234-1234-1234-123456789abc.jsonl",
        source_index=0,
        blob_size=0,
        acquired_at="2026-08-05T00:00:00+00:00",
    )
    history_payload = b'{"type":"file-history-snapshot","messageId":"m1"}\n'
    history_hash, history_size = BlobStore(blob_root).write_from_bytes(history_payload)
    history_record = history_record.model_copy(update={"blob_hash": history_hash, "blob_size": history_size})
    history_result = ingest_record(history_record, str(archive), "off", blob_root_str=str(blob_root))
    assert history_result.sessions == []
    assert history_result.validation_status == "skipped"

    analysis_record = record.model_copy(
        update={
            "raw_id": "raw-analysis-session-route",
            "source_path": "/home/user/.claude/projects/p/analysis/session.jsonl",
        }
    )
    analysis_result = ingest_record(analysis_record, str(archive), "off", blob_root_str=str(blob_root))
    assert analysis_result.sessions, "positive session content must override the weak analysis path heuristic"
