"""Historical terminal raw-failure disposition stays receipt-bound and lossless."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import ArtifactSupportStatus, Origin, Provider
from polylogue.maintenance.raw_failure_disposition_apply import (
    TOOL_VERSION,
    RawFailureDispositionApplyError,
    apply_raw_failure_dispositions,
)
from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceArtifact, upsert_raw_artifact
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _manifest(path: Path, *entries: dict[str, str]) -> Path:
    path.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")
    return path


def _archive(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b"",
            source_path="/captures/empty.jsonl",
            acquired_at_ms=100,
        )
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET parse_error = 'captured JSONL payload ends before a complete record boundary'"
        )
        upsert_raw_artifact(
            conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="historical-empty",
                origin=Origin.CLAUDE_CODE_SESSION,
                source_path="/captures/empty.jsonl",
                source_index=0,
                artifact_kind="coordinator_session_stream",
                classification_reason="legacy-path-classification",
                support_status=ArtifactSupportStatus.SUPPORTED_PARSEABLE,
                first_observed_at_ms=100,
                last_observed_at_ms=100,
            ),
        )
        conn.commit()
    return root, raw_id


def _state(root: Path) -> tuple[str, str, int]:
    with sqlite3.connect(root / "source.db") as conn:
        artifact_kind, parse_error = conn.execute(
            "SELECT a.artifact_kind, r.parse_error FROM raw_sessions r JOIN raw_artifacts a ON a.raw_id = r.raw_id"
        ).fetchone()
        receipts = conn.execute("SELECT COUNT(*) FROM raw_failure_disposition_receipts").fetchone()[0]
    return str(artifact_kind), str(parse_error), int(receipts)


def test_dry_run_preserves_historical_raw_and_artifact(tmp_path: Path) -> None:
    root, raw_id = _archive(tmp_path)
    manifest = _manifest(
        tmp_path / "manifest.jsonl",
        {"raw_id": raw_id, "disposition_kind": "terminal_corrupt_input", "detail": "empty retained byte stream"},
    )
    before = _state(root)

    report = apply_raw_failure_dispositions(root, manifest_path=manifest)

    assert report.applied is False
    assert report.candidate_count == 1
    assert report.disposed_raw_ids == ()
    assert _state(root) == before


def test_apply_reclassifies_only_manifested_failed_raw_and_writes_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, raw_id = _archive(tmp_path)
    manifest = _manifest(
        tmp_path / "manifest.jsonl",
        {"raw_id": raw_id, "disposition_kind": "terminal_corrupt_input", "detail": "empty retained byte stream"},
    )
    backup = tmp_path / "backup" / "manifest.json"
    checked: list[tuple[Path, object]] = []

    def _validate(path: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        checked.append((path, tier))
        return path

    monkeypatch.setattr(
        "polylogue.maintenance.raw_failure_disposition_apply.validate_migration_backup_manifest", _validate
    )
    report = apply_raw_failure_dispositions(root, manifest_path=manifest, backup_manifest=backup, dry_run=False)

    assert report.applied is True
    assert report.disposed_raw_ids == (raw_id,)
    assert checked == [(backup, ArchiveTier.SOURCE), (backup, ArchiveTier.SOURCE)]
    artifact_kind, parse_error, receipt_count = _state(root)
    assert artifact_kind == "terminal_corrupt_input"
    assert "complete record boundary" in parse_error
    assert receipt_count == 1
    lifecycle = read_raw_failure_lifecycle(root / "source.db")
    assert lifecycle.terminal == 1
    assert lifecycle.unexplained == 0
    with sqlite3.connect(root / "source.db") as conn:
        receipt = conn.execute(
            "SELECT raw_id, previous_artifact_kind, previous_support_status, "
            "previous_classification_reason, disposition_kind, tool_version, detail "
            "FROM raw_failure_disposition_receipts"
        ).fetchone()
        classification_reason = conn.execute(
            "SELECT classification_reason FROM raw_artifacts WHERE raw_id = ?", (raw_id,)
        ).fetchone()[0]
    assert receipt == (
        raw_id,
        "coordinator_session_stream",
        "supported_parseable",
        "legacy-path-classification",
        "terminal_corrupt_input",
        TOOL_VERSION,
        "empty retained byte stream",
    )
    assert json.loads(classification_reason) == {
        "diagnostic": "empty retained byte stream",
        "evidence_ref": f"raw-failure-disposition:{hashlib.sha256(manifest.read_bytes()).hexdigest()}",
        "outcome_code": "corrupt_input",
        "provenance": "worker-disposition-v1",
        "remediation": "retain the reviewed terminal disposition until a forced reparse is authorized",
        "retryable": False,
    }


def test_apply_refuses_duplicate_or_nonterminal_manifest_entries(tmp_path: Path) -> None:
    root, raw_id = _archive(tmp_path)
    duplicate = _manifest(
        tmp_path / "duplicate.jsonl",
        {"raw_id": raw_id, "disposition_kind": "terminal_corrupt_input", "detail": "one"},
        {"raw_id": raw_id, "disposition_kind": "terminal_corrupt_input", "detail": "two"},
    )
    with pytest.raises(RawFailureDispositionApplyError, match="repeats raw_id"):
        apply_raw_failure_dispositions(root, manifest_path=duplicate)
    deferred = _manifest(
        tmp_path / "deferred.jsonl",
        {"raw_id": raw_id, "disposition_kind": "deferred_hot_jsonl_capture", "detail": "not terminal"},
    )
    with pytest.raises(RawFailureDispositionApplyError, match="not a terminal"):
        apply_raw_failure_dispositions(root, manifest_path=deferred)
