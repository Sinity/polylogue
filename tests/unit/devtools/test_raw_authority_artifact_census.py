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
from polylogue.core.enums import Provider
from polylogue.daemon import backup as backup_module
from polylogue.maintenance.raw_authority_artifact_census import (
    RawAuthorityArtifactCensusError,
    run_raw_authority_artifact_census,
)
from polylogue.storage.artifacts.raw_authority_census import RawAuthorityBucket
from polylogue.storage.blob_store import reset_blob_store
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


def _write_artifact(archive: Path, *, raw_id: str = "raw-artifact") -> None:
    with ArchiveStore.open_existing(archive, read_only=False) as store:
        store.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=_artifact_payload(),
            source_path=f"/home/user/.claude/projects/p/tool-results/{raw_id}.json",
            acquired_at_ms=1_700_000_000_000,
            raw_id=raw_id,
        )
        store.commit()


def _real_backup_manifest(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, label: str = "backups"
) -> Path:
    monkeypatch.setattr(backup_module, "archive_root", lambda: archive)
    result = backup_module.backup_archive(output_dir=tmp_path / label, verify=True)
    assert result.ok
    assert result.output_path is not None
    return Path(result.output_path) / "manifest.json"


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
    assert payload["receipt"]["scope"]["physical_database_operations"] == []
    assert payload["receipt"]["counts"]["artifact"] == 1
    assert payload["receipt"]["counts"]["novel_materialization_candidate"] == 0
    assert "source_path" not in receipt_path.read_text(encoding="utf-8")

    output = io.StringIO()
    assert main(["--archive-root", str(archive), "--json", "--receipt", str(receipt_path)], stdout=output) == 1
    assert "immutable" in output.getvalue()
    with sqlite3.connect(archive / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)


def test_apply_pages_observations_through_real_backup_gates_and_records_receipts(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_artifact(archive)
    _write_artifact(archive, raw_id="raw-artifact-next")
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

    first_manifest = _real_backup_manifest(archive, tmp_path, monkeypatch, label="backups-first")
    output = io.StringIO()
    assert (
        main(
            [
                "--archive-root",
                str(archive),
                "--apply",
                "--backup-manifest",
                str(first_manifest),
                "--limit",
                "1",
                "--json",
            ],
            stdout=output,
        )
        == 0
    )
    payload = json.loads(output.getvalue())
    assert payload["observations_written"] == 1
    assert payload["receipt_id"] is not None
    assert payload["receipt"]["page"] == {"after_raw_id": None, "next_after_raw_id": "raw-artifact"}
    census_id = payload["receipt"]["evidence"]["checkpoint"]["census_id"]
    assert payload["receipt"]["evidence"].keys() == {"archive", "backup", "checkpoint", "command", "inventory"}
    assert (
        payload["receipt"]["evidence"]["inventory"]["before_sha256"]
        != payload["receipt"]["evidence"]["inventory"]["after_sha256"]
    )
    assert payload["receipt"]["scope"]["logical_database_operations"] == ["upsert raw_artifacts observations"]
    assert payload["receipt"]["scope"]["physical_database_operations"] == [
        "PRAGMA wal_checkpoint(TRUNCATE) on source.db after initial backup validation"
    ]
    with sqlite3.connect(archive / "source.db") as conn:
        assert (
            conn.execute(
                "SELECT raw_id, revision_authority, blob_hash, blob_size, source_path FROM raw_sessions ORDER BY raw_id"
            ).fetchall()
            == raw_rows_before
        )
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (2,)
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (1,)
        receipt = conn.execute(
            "SELECT receipt_id, receipt_sha256, receipt_json FROM raw_authority_artifact_census_receipts"
        ).fetchone()
        assert receipt is not None
        assert receipt[0] == payload["receipt_id"]
        assert json.loads(receipt[2])["receipt_sha256"] == receipt[1]
    assert (archive / "index.db").read_bytes() == index_before
    assert {
        str(path.relative_to(archive / "blob")): path.read_bytes()
        for path in (archive / "blob").rglob("*")
        if path.is_file()
    } == blobs_before

    second_manifest = _real_backup_manifest(archive, tmp_path, monkeypatch, label="backups-second")
    output = io.StringIO()
    assert (
        main(
            [
                "--archive-root",
                str(archive),
                "--apply",
                "--census-id",
                census_id,
                "--backup-manifest",
                str(second_manifest),
                "--limit",
                "1",
                "--after-raw-id",
                "raw-artifact",
                "--json",
            ],
            stdout=output,
        )
        == 0
    )
    second_payload = json.loads(output.getvalue())
    assert second_payload["receipt"]["page"] == {"after_raw_id": "raw-artifact", "next_after_raw_id": None}
    with sqlite3.connect(archive / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (2,)
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_artifact_census_receipts").fetchone() == (2,)
        checkpoint = conn.execute(
            "SELECT next_after_raw_id, last_receipt_id, completed_at_ms FROM raw_authority_artifact_census_checkpoints WHERE census_id = ?",
            (census_id,),
        ).fetchone()
        assert checkpoint is not None
        next_after_raw_id, last_receipt_id, completed_at_ms = checkpoint
        assert next_after_raw_id is None
        assert last_receipt_id == second_payload["receipt_id"]
        assert completed_at_ms is not None


def test_apply_refuses_replayed_or_unbound_page_cursor(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_artifact(archive, raw_id="raw-b")
    _write_artifact(archive, raw_id="raw-c")
    first_manifest = _real_backup_manifest(archive, tmp_path, monkeypatch, label="first")
    first = run_raw_authority_artifact_census(archive, apply=True, backup_manifest=first_manifest, limit=1)
    evidence = first.receipt["evidence"]
    assert isinstance(evidence, dict)
    checkpoint_evidence = evidence["checkpoint"]
    assert isinstance(checkpoint_evidence, dict)
    census_id = checkpoint_evidence["census_id"]
    assert isinstance(census_id, str)
    next_after = first.census.next_after_raw_id
    assert next_after == "raw-b"
    second_manifest = _real_backup_manifest(archive, tmp_path, monkeypatch, label="second")

    with pytest.raises(RawAuthorityArtifactCensusError, match="continuation cursor"):
        run_raw_authority_artifact_census(
            archive,
            apply=True,
            backup_manifest=second_manifest,
            limit=1,
            census_id=census_id,
            after_raw_id="raw-c",
        )
    with pytest.raises(RawAuthorityArtifactCensusError, match="requires --census-id"):
        run_raw_authority_artifact_census(
            archive,
            apply=True,
            backup_manifest=second_manifest,
            limit=1,
            after_raw_id=next_after,
        )

    _write_artifact(archive, raw_id="raw-a-added-after-checkpoint")
    third_manifest = _real_backup_manifest(archive, tmp_path, monkeypatch, label="third")
    resumed = run_raw_authority_artifact_census(
        archive,
        apply=True,
        backup_manifest=third_manifest,
        limit=1,
        census_id=census_id,
        after_raw_id=next_after,
    )
    assert resumed.census.next_after_raw_id is None
    with sqlite3.connect(archive / "source.db") as conn:
        assert {row[0] for row in conn.execute("SELECT raw_id FROM raw_artifacts ORDER BY raw_id")} == {
            "raw-b",
            "raw-c",
        }
    fourth_manifest = _real_backup_manifest(archive, tmp_path, monkeypatch, label="fourth")
    with pytest.raises(RawAuthorityArtifactCensusError, match="already complete"):
        run_raw_authority_artifact_census(
            archive,
            apply=True,
            backup_manifest=fourth_manifest,
            limit=1,
            census_id=census_id,
            after_raw_id=next_after,
        )
    fifth_manifest = _real_backup_manifest(archive, tmp_path, monkeypatch, label="fifth")
    next_census = run_raw_authority_artifact_census(archive, apply=True, backup_manifest=fifth_manifest, limit=1)
    assert [entry.raw_id for entry in next_census.census.entries] == ["raw-a-added-after-checkpoint"]


def test_apply_rejects_receipt_option_before_mutating_source(archive: Path) -> None:
    _write_artifact(archive)
    receipt_path = archive.parent / "apply-receipt.json"
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


def test_apply_rejects_missing_invalid_and_unattested_real_backups(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_artifact(archive)
    output = io.StringIO()
    missing = tmp_path / "missing-manifest.json"
    assert (
        main(
            ["--archive-root", str(archive), "--apply", "--backup-manifest", str(missing), "--json"],
            stdout=output,
        )
        == 1
    )
    assert "backup" in output.getvalue().lower()

    invalid = tmp_path / "invalid-manifest.json"
    invalid.write_text("{}\n", encoding="utf-8")
    output = io.StringIO()
    assert (
        main(
            ["--archive-root", str(archive), "--apply", "--backup-manifest", str(invalid), "--json"],
            stdout=output,
        )
        == 1
    )
    assert "backup" in output.getvalue().lower()

    manifest = _real_backup_manifest(archive, tmp_path, monkeypatch)
    verification_receipt = manifest.parent / "verification-receipt.json"
    receipt_payload = json.loads(verification_receipt.read_text(encoding="utf-8"))
    receipt_payload["attestations"] = []
    verification_receipt.write_text(json.dumps(receipt_payload, sort_keys=True), encoding="utf-8")
    output = io.StringIO()
    assert (
        main(
            ["--archive-root", str(archive), "--apply", "--backup-manifest", str(manifest), "--json"],
            stdout=output,
        )
        == 1
    )
    assert "attestation" in output.getvalue().lower()

    with sqlite3.connect(archive / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)


def test_invalid_backup_is_refused_before_checkpoint(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_artifact(archive)
    import polylogue.maintenance.raw_authority_artifact_census as maintenance

    def checkpoint_must_not_run(_conn: sqlite3.Connection) -> None:
        raise AssertionError("invalid backup must not checkpoint source.db")

    monkeypatch.setattr(maintenance, "_checkpoint_source_tier", checkpoint_must_not_run)
    with pytest.raises(maintenance.RawAuthorityArtifactCensusError, match="backup"):
        run_raw_authority_artifact_census(
            archive,
            apply=True,
            backup_manifest=archive.parent / "invalid-manifest.json",
        )


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
