from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.commands.maintenance import _rebuild_index as maintenance_rebuild_index
from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.maintenance.replay import rebuild_index_from_source
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.blob_gc import read_gc_history
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSearchHit, ArchiveSessionSummary
from polylogue.storage.sqlite.archive_tiers.archive_init import (
    ArchiveInitResult,
    ArchiveTierInitResult,
)
from polylogue.storage.sqlite.archive_tiers.archive_plan import ArchiveInitAction, ArchiveInitPlan
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion

_ARCHIVE_TIERS = ("source.db", "index.db", "embeddings.db", "ops.db", "user.db")


def _stage_uninitialized_archive(cli_workspace: dict[str, Path]) -> None:
    """Reset the workspace to an uninitialized state for plan/init tests.

    ``cli_workspace`` bootstraps a full archive (all tiers present),
    which the archive planner would classify as already-initialized. Clear the
    pre-created tiers so the plan reports a ready, "create every tier"
    initialization.
    """
    archive_root = cli_workspace["archive_root"]
    for name in _ARCHIVE_TIERS:
        (archive_root / name).unlink(missing_ok=True)


def _write_gc_candidate(cli_workspace: dict[str, Path], blob_hash: str) -> Path:
    blob_root = cli_workspace["archive_root"] / "blob"
    path = blob_root / blob_hash[:2] / blob_hash[2:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"gc candidate")
    old_epoch_s = 946684800
    os.utime(path, (old_epoch_s, old_epoch_s))
    return path


def _seed_assertion_export_rows(archive_root: Path) -> None:
    with sqlite3.connect(archive_root / "user.db") as conn:
        conn.row_factory = sqlite3.Row
        initialize_archive_tier(conn, ArchiveTier.USER)
        upsert_assertion(
            conn,
            assertion_id="export-mark",
            target_ref="session:s-1",
            kind=AssertionKind.MARK,
            scope_ref="run:r-1",
            key="export/mark",
            value={"label": "important"},
            body_text="operator mark",
            author_ref="user:operator",
            author_kind="user",
            evidence_refs=["message:s-1:1"],
            status="active",
            visibility="private",
            now_ms=1_700_000_001_000,
        )
        upsert_assertion(
            conn,
            assertion_id="export-deleted-note",
            target_ref="session:s-2",
            kind=AssertionKind.NOTE,
            scope_ref="run:r-2",
            key="export/note",
            body_text="deleted note retained for backup",
            status="deleted",
            visibility="private",
            now_ms=1_700_000_002_000,
        )


def _seed_missing_blob_cursor(archive_root: Path, source: Path) -> None:
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('{"type":"session_meta","payload":{"id":"missing-blob"}}\n', encoding="utf-8")
    blob_hash = b"a" * 32
    with sqlite3.connect(archive_root / "source.db") as conn:
        write_source_raw_session_blob_ref(
            conn,
            origin="codex-session",
            source_path=str(source),
            source_index=0,
            blob_hash=blob_hash,
            blob_size=source.stat().st_size,
            acquired_at_ms=1,
            native_id="missing-blob",
        )
    stat = source.stat()
    CursorStore(archive_root / "ops.db").set(
        source,
        stat.st_size,
        byte_offset=stat.st_size,
        last_complete_newline=stat.st_size,
        parser_fingerprint="live-batched-v2",
        content_fingerprint=blob_hash.hex(),
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )


def _create_user_v3(path: Path) -> None:
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE assertions (
                assertion_id        TEXT PRIMARY KEY,
                scope_ref           TEXT,
                target_ref          TEXT NOT NULL,
                key                 TEXT,
                kind                TEXT NOT NULL,
                value_json          TEXT,
                body_text           TEXT,
                author_ref          TEXT DEFAULT 'user:local',
                author_kind         TEXT DEFAULT 'user',
                evidence_refs_json  TEXT DEFAULT '[]',
                status              TEXT DEFAULT 'active',
                visibility          TEXT DEFAULT 'private',
                confidence          REAL,
                staleness_json      TEXT,
                context_policy_json TEXT DEFAULT '{"inject":false}',
                supersedes_json     TEXT DEFAULT '[]',
                created_at_ms       INTEGER NOT NULL,
                updated_at_ms       INTEGER NOT NULL
            ) STRICT;
            CREATE INDEX idx_assertions_target_kind
            ON assertions(target_ref, kind);
            CREATE INDEX idx_assertions_kind_status_updated
            ON assertions(kind, status, updated_at_ms);
            CREATE INDEX idx_assertions_target_kind_status_visibility
            ON assertions(target_ref, kind, status, visibility);
            PRAGMA user_version = 3;
            """
        )


def _run_verified_backup_cli(cli_runner: CliRunner, output_dir: Path, *, profile: str) -> Path:
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "backup",
            "--output-dir",
            str(output_dir),
            "--profile",
            profile,
            "--verify",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    backup_line = next(line for line in result.output.splitlines() if line.startswith("Backup complete: "))
    backup_root = Path(backup_line.removeprefix("Backup complete: "))
    assert (backup_root / "verification-receipt.json").exists()
    return backup_root / "manifest.json"


def _seed_blob_reference_debt(archive_root: Path, source: Path) -> None:
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('{"title":"recoverable"}\n', encoding="utf-8")
    missing_raw_hash = b"b" * 32
    missing_ref_hash = b"c" * 32
    with sqlite3.connect(archive_root / "source.db") as conn:
        write_source_raw_session_blob_ref(
            conn,
            origin="chatgpt-export",
            source_path=str(source),
            source_index=0,
            blob_hash=missing_raw_hash,
            blob_size=source.stat().st_size,
            acquired_at_ms=1,
            native_id="recoverable-chat",
        )
        conn.execute(
            """
            INSERT INTO blob_refs (
                blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (missing_ref_hash, "raw-gone", "raw_payload", str(archive_root / "missing-browser-capture.json"), 10, 1),
        )


def test_archive_plan_cli_reports_tier_targets(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    _stage_uninitialized_archive(cli_workspace)
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "archive-plan", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert {tier["tier"]: tier["action"] for tier in payload["tiers"]} == {
        "source": "create",
        "index": "create",
        "embeddings": "create",
        "user": "create",
        "ops": "create",
    }
    assert {Path(tier["path"]).name for tier in payload["tiers"]} == {
        "source.db",
        "index.db",
        "embeddings.db",
        "user.db",
        "ops.db",
    }
    assert {tier["tier"]: tier["durability"] for tier in payload["tiers"]} == {
        "source": "irreplaceable",
        "index": "rebuildable",
        "embeddings": "expensive_rebuild",
        "user": "human",
        "ops": "disposable",
    }
    assert all(isinstance(tier["expected_user_version"], int) for tier in payload["tiers"])


def test_archive_plan_cli_surfaces_existing_target_blocker(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    source_target = cli_workspace["archive_root"] / "source.db"
    source_target.write_bytes(b"not an archive source")

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "archive-plan", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is False
    source_plan = next(tier for tier in payload["tiers"] if tier["tier"] == "source")
    assert source_plan["action"] == "blocked"
    assert any("source target already exists" in blocker for blocker in payload["blockers"])


def test_backup_plan_cli_reports_backup_profiles_and_tier_boundaries(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "backup-plan", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["mode"] == "backup_plan"
    assert payload["mutates"] is False
    assert payload["archive_root"] == str(cli_workspace["archive_root"])

    tiers = {tier["tier"]: tier for tier in payload["tiers"]}
    assert tiers["source"]["backup_class"] == "critical"
    assert tiers["source"]["backup_required"] is True
    assert tiers["index"]["backup_class"] == "warm_cache"
    assert tiers["index"]["backup_required"] is False
    assert tiers["embeddings"]["backup_policy"] == "back_up_when_present"
    assert tiers["user"]["backup_policy"] == "always_back_up"
    assert tiers["ops"]["backup_policy"] == "diagnostics_only"
    assert all(tier["present"] is True for tier in tiers.values())

    profiles = {profile["name"] for profile in payload["profiles"]}
    assert profiles == {
        "full_evidence",
        "user_overlays",
        "rebuildable_cache_exclude",
        "diagnostics_bundle",
    }
    assert payload["blob_store"]["path"] == str(cli_workspace["archive_root"] / "blob")
    assert payload["blob_store"]["backup_policy"] == "back_up_referenced_blobs_with_source_and_user_tiers"


def test_backup_plan_cli_surfaces_missing_tiers_and_wal_checkpoint_warning(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    archive_root = cli_workspace["archive_root"]
    (archive_root / "index.db").unlink()
    (archive_root / "user.db-wal").write_text("pending", encoding="utf-8")

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "backup-plan", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    tiers = {tier["tier"]: tier for tier in payload["tiers"]}
    assert tiers["index"]["present"] is False
    assert tiers["user"]["wal_present"] is True
    assert tiers["user"]["checkpoint_recommended"] is True
    assert payload["warnings"] == ["user.db-wal is present; checkpoint before copying user.db"]


def test_backup_plan_cli_renders_plain_summary(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "backup-plan"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Archive backup plan" in result.output
    assert "source.db: critical policy=back_up present" in result.output
    assert "full_evidence:" in result.output


def test_assertion_export_cli_emits_all_assertions_as_jsonl(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    _seed_assertion_export_rows(cli_workspace["archive_root"])

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "assertion-export"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.output.splitlines()]
    assert [row["assertion_id"] for row in rows] == ["export-mark", "export-deleted-note"]
    assert rows[0]["kind"] == "mark"
    assert rows[0]["value"] == {"label": "important"}
    assert rows[0]["evidence_refs"] == ["message:s-1:1"]
    assert rows[1]["status"] == "deleted"


def test_assertion_export_cli_filters_and_writes_json_file(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    _seed_assertion_export_rows(cli_workspace["archive_root"])
    out_path = cli_workspace["archive_root"] / "exports" / "assertions.json"

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "assertion-export",
            "--format",
            "json",
            "--kind",
            "note",
            "--status",
            "deleted",
            "--out",
            str(out_path),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert result.output == f"Exported 1 assertions to {out_path}\n"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "assertion_export"
    assert payload["count"] == 1
    assert [row["assertion_id"] for row in payload["assertions"]] == ["export-deleted-note"]


def test_blob_gc_cli_dry_run_reports_without_deleting(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    blob_hash = "aa" + "1" * 62
    candidate = _write_gc_candidate(cli_workspace, blob_hash)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-gc", "--max-batch", "5", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["mode"] == "blob_gc"
    assert payload["mutates"] is False
    assert payload["dry_run"] is True
    assert payload["candidate_count"] == 1
    assert payload["inspected_count"] == 1
    assert payload["would_delete_count"] == 1
    assert payload["deleted_count"] == 0
    assert payload["generation_written"] is False
    assert candidate.exists(), "dry-run must not delete the candidate"
    assert read_gc_history(cli_workspace["archive_root"] / "index.db", limit=1) == []


def test_blob_gc_cli_plain_preview_names_skip_counts(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    _write_gc_candidate(cli_workspace, "bb" + "2" * 62)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-gc", "--max-batch", "5"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Blob GC dry-run" in result.output
    assert "Candidates: 1" in result.output
    assert "Result:     would delete 1 blob(s)" in result.output
    assert "referenced=0 reserved=0 missing=0 unlink_error=0" in result.output


def test_blob_gc_cli_yes_deletes_and_records_generation(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    blob_hash = "cc" + "3" * 62
    candidate = _write_gc_candidate(cli_workspace, blob_hash)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-gc", "--yes", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mutates"] is True
    assert payload["dry_run"] is False
    assert payload["would_delete_count"] == 0
    assert payload["deleted_count"] == 1
    assert payload["reclaimed_bytes"] == len(b"gc candidate")
    assert payload["generation_written"] is True
    assert str(payload["generation_id"]).startswith("gc-")
    assert not candidate.exists()

    history = read_gc_history(cli_workspace["archive_root"] / "index.db", limit=1)
    assert len(history) == 1
    assert history[0].generation_id == payload["generation_id"]
    assert history[0].reclaimed_count == 1


def test_blob_publications_cli_requires_confirmation_to_abandon(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    archive_root = cli_workspace["archive_root"]
    publisher = ArchiveBlobPublisher(
        archive_root / "source.db",
        archive_root / "blob",
    )
    blob_hash, _ = publisher.write_from_bytes(b"operator-adjudicated receipt")
    receipt_id = publisher.receipt_id(blob_hash)
    publisher.flush()
    assert receipt_id is not None

    inspected = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-publications", "--output-format", "json"],
        catch_exceptions=False,
    )
    assert inspected.exit_code == 0
    payload = json.loads(inspected.output)
    assert payload["mutates"] is False
    assert payload["receipts"][0]["publication_id"] == receipt_id

    refused = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-publications", "--abandon", receipt_id],
    )
    assert refused.exit_code != 0
    assert "--yes is required" in refused.output

    abandoned = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "blob-publications",
            "--abandon",
            receipt_id,
            "--yes",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert abandoned.exit_code == 0
    payload = json.loads(abandoned.output)
    assert payload["abandonment"]["abandoned"] == 1
    assert payload["receipts"] == []


def test_blob_reference_debt_cli_classifies_missing_refs(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    source = cli_workspace["archive_root"] / "exports" / "recoverable.json"
    _seed_blob_reference_debt(cli_workspace["archive_root"], source)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "blob-reference-debt",
            "--sample-limit",
            "1",
            "--group-limit",
            "2",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "blob_reference_debt"
    assert payload["mutates"] is False
    assert payload["ok"] is False
    assert payload["missing_distinct_blobs"] == 2
    assert payload["missing_by_table"] == {"blob_refs": 2, "raw_sessions": 1}
    assert payload["missing_by_origin"] == {"(none)": 1, "chatgpt-export": 1}
    assert payload["missing_ref_id_join"] == {
        "ref_id_has_raw_session": 1,
        "ref_id_without_raw_session": 1,
    }
    assert payload["missing_source_path_presence"] == {
        "recoverable_source_path_exists": 1,
        "source_path_missing": 1,
    }
    assert len(payload["samples"]) == 1


def test_blob_reference_debt_cli_plain_output_names_read_only_debt(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    source = cli_workspace["archive_root"] / "exports" / "recoverable.json"
    _seed_blob_reference_debt(cli_workspace["archive_root"], source)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-reference-debt", "--sample-limit", "1"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Blob reference debt" in result.output
    assert "Status:       debt-present" in result.output
    assert "Source paths: recoverable_source_path_exists=1, source_path_missing=1" in result.output


def test_attachment_acquisition_debt_cli_reports_clean_empty_archive(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "attachment-acquisition-debt", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "attachment_acquisition_debt"
    assert payload["mutates"] is False
    assert payload["ok"] is True
    assert payload["total_attachments"] == 0
    assert payload["acquired_missing_blob_count"] == 0


def test_attachment_acquisition_debt_cli_plain_output_distinguishes_unfetched_from_debt(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "attachment-acquisition-debt"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Attachment acquisition debt" in result.output
    assert "Unfetched:         0 (honest floor, not missing blobs)" in result.output
    assert "Acquired missing:  0 (genuine debt)" in result.output
    assert "Status:            ok" in result.output


def test_blob_reference_recovery_plan_cli_writes_raw_backed_manifest(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    source = cli_workspace["archive_root"] / "exports" / "recoverable.json"
    _seed_blob_reference_debt(cli_workspace["archive_root"], source)
    manifest = cli_workspace["archive_root"] / "plans" / "raw-backed.jsonl"

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "blob-reference-recovery-plan",
            "--manifest-file",
            str(manifest),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "blob_reference_recovery_plan"
    assert payload["mutates"] is False
    assert payload["writes_manifest"] is True
    assert payload["missing_raw_backed_blobs"] == 1
    assert payload["by_origin"] == {"chatgpt-export": 1}
    assert payload["by_action"] == {"direct_source_hash_mismatch": 1}
    manifest_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["source_path"] == str(source)


def test_blob_reference_replace_from_source_cli_requires_manifest_for_apply(
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-reference-replace-from-source", "--yes"],
    )

    assert result.exit_code != 0
    assert "--manifest-file is required with --yes" in result.output


def test_blob_reference_replace_from_source_cli_applies_with_manifest(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    source = cli_workspace["archive_root"] / "exports" / "recoverable.json"
    _seed_blob_reference_debt(cli_workspace["archive_root"], source)
    manifest = cli_workspace["archive_root"] / "plans" / "replace.jsonl"

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "blob-reference-replace-from-source",
            "--yes",
            "--manifest-file",
            str(manifest),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "blob_reference_replace_from_source"
    assert payload["mutates"] is True
    assert payload["writes_manifest"] is True
    assert payload["candidate_rows"] == 1
    assert payload["replaced_rows"] == 1
    assert payload["skipped_error"] == 0
    manifest_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["old_blob_hash"] != manifest_rows[0]["new_blob_hash"]


def test_blob_reference_prune_orphans_cli_dry_run_keeps_refs(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    source = cli_workspace["archive_root"] / "exports" / "recoverable.json"
    _seed_blob_reference_debt(cli_workspace["archive_root"], source)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "blob-reference-prune-orphans",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "blob_reference_prune_orphans"
    assert payload["mutates"] is False
    assert payload["dry_run"] is True
    assert payload["missing_orphan_refs"] == 1
    assert payload["pruned_refs"] == 0
    with sqlite3.connect(cli_workspace["archive_root"] / "source.db") as conn:
        refs = conn.execute("SELECT source_path FROM blob_refs ORDER BY source_path").fetchall()
    assert refs == [(str(source),), (str(cli_workspace["archive_root"] / "missing-browser-capture.json"),)]


def test_blob_reference_prune_orphans_cli_apply_quarantines_deleted_refs(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    source = cli_workspace["archive_root"] / "exports" / "recoverable.json"
    _seed_blob_reference_debt(cli_workspace["archive_root"], source)
    quarantine_file = cli_workspace["archive_root"] / "quarantine" / "blob-refs.jsonl"

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "blob-reference-prune-orphans",
            "--yes",
            "--quarantine-file",
            str(quarantine_file),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mutates"] is True
    assert payload["dry_run"] is False
    assert payload["missing_orphan_refs"] == 1
    assert payload["pruned_refs"] == 1
    assert payload["quarantine_path"] == str(quarantine_file)
    exported = [json.loads(line) for line in quarantine_file.read_text(encoding="utf-8").splitlines()]
    assert exported[0]["ref_id"] == "raw-gone"
    assert exported[0]["source_path"].endswith("missing-browser-capture.json")
    with sqlite3.connect(cli_workspace["archive_root"] / "source.db") as conn:
        refs = conn.execute("SELECT source_path FROM blob_refs ORDER BY source_path").fetchall()
    assert refs == [(str(source),)]


def _seed_orphan_embedding_row(archive_root: Path) -> tuple[str, str]:
    """Seed embeddings.db with one vector row for a message that no longer
    exists under an otherwise-live session — standing in for a message
    dropped by an index rebuild (polylogue-1dk1) while the session survives.
    """

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, MaterialOrigin, Origin, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.embedding_write import (
        ArchiveEmbeddingWrite,
        upsert_message_embeddings,
    )
    from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

    long_text = "This live message keeps the session present in the rebuilt index."
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="orphan-cli-fixture",
                title="orphan reconcile fixture",
                messages=[
                    ParsedMessage(
                        provider_message_id="live",
                        role=Role.USER,
                        text=long_text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=long_text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            )
        )

    orphan_message_id = f"{session_id}:orphaned-message-no-longer-in-index"
    with sqlite3.connect(archive_root / "embeddings.db") as conn:
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            pytest.skip(f"sqlite-vec extension is unavailable: {error}")
        upsert_message_embeddings(
            conn,
            [
                ArchiveEmbeddingWrite(
                    message_id=orphan_message_id,
                    session_id=session_id,
                    origin=Origin.CODEX_SESSION,
                    embedding=[0.01] * EMBEDDING_DIMENSION,
                    model="voyage-4",
                    embedded_at_ms=1_700_000_000_000,
                    content_hash=b"o" * 32,
                )
            ],
        )
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES (?, 'codex-session', 1, 1700000000000, 0, NULL)
            """,
            (session_id,),
        )
        conn.commit()
    return session_id, orphan_message_id


def test_embedding_orphan_reconcile_default_quiet_window_matches_reconcile_module() -> None:
    """The CLI's hardcoded --help default (polylogue-sod7) must not drift from the real constant.

    _embeddings.py hardcodes _DEFAULT_QUIET_WINDOW_SECONDS instead of importing
    DEFAULT_QUIET_WINDOW_MS from polylogue.storage.embeddings.reconcile, so
    that constant -- and its heavy import chain -- isn't paid on the
    `--help` path. This test is the drift guard for that duplication.
    """
    from polylogue.cli.commands.maintenance._embeddings import _DEFAULT_QUIET_WINDOW_SECONDS
    from polylogue.storage.embeddings.reconcile import DEFAULT_QUIET_WINDOW_MS

    assert _DEFAULT_QUIET_WINDOW_SECONDS == DEFAULT_QUIET_WINDOW_MS // 1000


def test_embedding_orphan_reconcile_cli_dry_run_keeps_rows(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    _seed_orphan_embedding_row(cli_workspace["archive_root"])
    with sqlite3.connect(cli_workspace["archive_root"] / "embeddings.db") as conn:
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES ('codex-session:absent', 'codex-session', 0, 1700000000000, 0, NULL)
            """
        )

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "embedding-orphan-reconcile", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "embedding_orphan_reconcile"
    assert payload["mutates"] is False
    assert payload["dry_run"] is True
    assert payload["orphan_message_rows"] == 1
    assert payload["candidate_message_rows"] == 1
    assert payload["candidate_message_meta_rows"] == 1
    assert payload["candidate_vector_rows"] == 1
    assert payload["candidate_status_rows"] == 1
    assert payload["removed_message_rows"] == 0
    assert payload["removed_vector_rows"] == 0
    assert payload["removed_status_rows"] == 0
    with sqlite3.connect(cli_workspace["archive_root"] / "embeddings.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1


def test_embedding_orphan_reconcile_cli_plain_dry_run_reports_would_remove_counts(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    _seed_orphan_embedding_row(cli_workspace["archive_root"])
    with sqlite3.connect(cli_workspace["archive_root"] / "embeddings.db") as conn:
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES ('codex-session:absent', 'codex-session', 0, 1700000000000, 0, NULL)
            """
        )

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "embedding-orphan-reconcile"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Would remove:  1 meta row(s), 1 vector row(s), 1 status row(s)" in result.output
    assert "Removed:" not in result.output
    with sqlite3.connect(cli_workspace["archive_root"] / "embeddings.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1


def test_embedding_orphan_reconcile_cli_apply_removes_rows(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    session_id, message_id = _seed_orphan_embedding_row(cli_workspace["archive_root"])

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "embedding-orphan-reconcile",
            "--yes",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mutates"] is True
    assert payload["dry_run"] is False
    assert payload["removed_message_rows"] == 1
    assert payload["removed_vector_rows"] == 1
    assert payload["sessions_recounted"] == 1
    with sqlite3.connect(cli_workspace["archive_root"] / "embeddings.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM message_embeddings_meta WHERE message_id = ?", (message_id,)).fetchone()[
                0
            ]
            == 0
        )
        status = conn.execute(
            "SELECT message_count_embedded, needs_reindex FROM embedding_status WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert status is not None
        assert status[0] == 0
        assert status[1] == 1


def test_embedding_orphan_reconcile_cli_apply_is_bounded_by_default(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    session_id, _message_id = _seed_orphan_embedding_row(cli_workspace["archive_root"])
    embeddings_db = cli_workspace["archive_root"] / "embeddings.db"
    with sqlite3.connect(embeddings_db) as conn:
        conn.executemany(
            """
            INSERT INTO message_embeddings_meta (
                message_id, model, dimension, content_hash, embedded_at_ms, needs_reindex
            ) VALUES (?, 'voyage-4', 1024, ?, 1700000000000, 0)
            """,
            [(f"{session_id}:zz-orphan-{position:03d}", b"z" * 32) for position in range(500)],
        )

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "embedding-orphan-reconcile",
            "--yes",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["candidate_message_rows"] == 500
    assert payload["removed_message_rows"] == 500
    assert payload["more_pending"] is True
    with sqlite3.connect(embeddings_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1


def test_embedding_orphan_reconcile_cli_apply_requires_exclusive_offline_lease(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """Production CLI proof: a live/shared writer lease blocks break-glass apply."""
    from polylogue.storage.index_generation import ActiveWriterLease

    _seed_orphan_embedding_row(cli_workspace["archive_root"])
    writer = ActiveWriterLease(cli_workspace["archive_root"])
    writer.acquire()
    try:
        result = cli_runner.invoke(
            cli,
            ["--plain", "ops", "maintenance", "embedding-orphan-reconcile", "--yes"],
        )
    finally:
        writer.close()

    assert result.exit_code != 0
    assert "index rebuild lease is already held" in result.output


def test_embedding_orphan_reconcile_cli_apply_refuses_stale_index_schema(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """The break-glass writer cannot treat a pre-rebuild index as deletion truth."""
    _session_id, message_id = _seed_orphan_embedding_row(cli_workspace["archive_root"])
    index_db = cli_workspace["archive_root"] / "index.db"
    with sqlite3.connect(index_db) as conn:
        conn.execute("PRAGMA user_version = 32")

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "embedding-orphan-reconcile", "--yes"],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "active index is v32, packaged index is v" in str(result.exception)
    with sqlite3.connect(cli_workspace["archive_root"] / "embeddings.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM message_embeddings_meta WHERE message_id = ?", (message_id,)).fetchone()[
                0
            ]
            == 1
        )


def test_archive_init_cli_is_dry_run_without_yes(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    _stage_uninitialized_archive(cli_workspace)
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "archive-init", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["executed"] is False
    assert payload["ready"] is True
    assert not (cli_workspace["archive_root"] / "index.db").exists()


def test_archive_init_cli_executes_confirmed_initialization(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_init(plan: ArchiveInitPlan) -> ArchiveInitResult:
        return ArchiveInitResult(
            tier_results=(
                ArchiveTierInitResult(
                    tier="index",
                    path=plan.archive_root / "index.db",
                    action=ArchiveInitAction.CREATE,
                    backup_path=None,
                    initialized=True,
                ),
            ),
        )

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive_init.initialize_archive_tier_files_from_plan",
        fake_init,
    )

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "archive-init", "--yes", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["executed"] is True
    assert payload["tiers"] == [
        {
            "action": "create",
            "backup_path": None,
            "initialized": True,
            "path": str(cli_workspace["archive_root"] / "index.db"),
            "tier": "index",
        }
    ]


def test_migrate_tier_cli_allows_additive_source_ledgers_without_backup(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    source_db = cli_workspace["archive_root"] / "source.db"
    with sqlite3.connect(source_db) as conn:
        conn.executescript(SOURCE_DDL)
        conn.execute("DROP TABLE excised_content")
        conn.execute("DROP TABLE sinex_publication_obligations")
        conn.execute("PRAGMA user_version = 9")
        conn.commit()

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "migrate-tier", "source", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["backup_manifest"] is None
    assert payload["backup_receipt"] is None
    assert payload["applied_versions"] == [10, 11]


def test_backup_verify_then_migrate_tier_cli_applies_user_migration_with_receipt(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    user_db = cli_workspace["archive_root"] / "user.db"
    _create_user_v3(user_db)
    manifest = _run_verified_backup_cli(cli_runner, tmp_path / "backup", profile="user_overlays")

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "user",
            "--backup-manifest",
            str(manifest),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["tier"] == "user"
    assert payload["from_version"] == 3
    assert payload["to_version"] == USER_SCHEMA_VERSION
    assert payload["applied_versions"] == [4, 5]
    assert payload["backup_receipt"] == str(manifest.with_name("verification-receipt.json"))
    with sqlite3.connect(user_db) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").fetchone()
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='context_deliveries'"
        ).fetchone()


def test_migrate_tier_cli_rejects_unverified_backup_before_user_version_changes(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    user_db = cli_workspace["archive_root"] / "user.db"
    _create_user_v3(user_db)
    backup = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "backup",
            "--output-dir",
            str(tmp_path / "backup"),
            "--profile",
            "user_overlays",
        ],
        catch_exceptions=False,
    )
    assert backup.exit_code == 0, backup.output
    backup_line = next(line for line in backup.output.splitlines() if line.startswith("Backup complete: "))
    backup_root = Path(backup_line.removeprefix("Backup complete: "))

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "user",
            "--backup-manifest",
            str(backup_root / "manifest.json"),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "successful backup verification receipt" in json.loads(result.output)["error"]
    with sqlite3.connect(user_db) as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3


def test_migrate_tier_cli_rejects_one_byte_tampered_backup_before_user_version_changes(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    user_db = cli_workspace["archive_root"] / "user.db"
    _create_user_v3(user_db)
    manifest = _run_verified_backup_cli(cli_runner, tmp_path / "backup", profile="user_overlays")
    copied_tier = manifest.with_name("user.db")
    copied_bytes = bytearray(copied_tier.read_bytes())
    copied_bytes[-1] ^= 1
    copied_tier.write_bytes(copied_bytes)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "user",
            "--backup-manifest",
            str(manifest),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "tier artifact hash mismatch" in json.loads(result.output)["error"]
    with sqlite3.connect(user_db) as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3


def test_migrate_tier_cli_refuses_manifest_missing_target_tier(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    user_db = cli_workspace["archive_root"] / "user.db"
    _create_user_v3(user_db)
    manifest = _run_verified_backup_cli(cli_runner, tmp_path / "backup", profile="diagnostics_bundle")

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "user",
            "--backup-manifest",
            str(manifest),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert "does not include user.db" in payload["error"]
    with sqlite3.connect(user_db) as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").fetchone()


def test_archive_maintenance_help_omits_copy_activation_surface(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli, ["--plain", "ops", "maintenance", "--help"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "archive-plan" in result.output
    assert "archive-init" in result.output
    assert "archive-read" in result.output
    for removed in (
        "archive-copy-raw",
        "archive-copy-archive",
        "archive-copy-insights",
        "archive-copy-user",
        "archive-copy-all",
        "archive-copy-audit",
        "archive-activate",
    ):
        assert removed not in result.output


def test_missing_raw_blob_cursor_repair_dry_run_keeps_cursor(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    source = cli_workspace["archive_root"] / "watch" / "missing-blob.jsonl"
    _seed_missing_blob_cursor(cli_workspace["archive_root"], source)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "missing-raw-blob-cursors",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "dry-run"
    assert payload["candidate_count"] == 1
    assert payload["deleted_cursor_count"] == 0
    assert payload["candidates"][0]["source_path"] == str(source)
    with sqlite3.connect(cli_workspace["archive_root"] / "ops.db") as conn:
        assert conn.execute("SELECT 1 FROM ingest_cursor WHERE source_path = ?", (str(source),)).fetchone() == (1,)


def test_missing_raw_blob_cursor_repair_apply_deletes_only_cursor(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    source = cli_workspace["archive_root"] / "watch" / "missing-blob.jsonl"
    _seed_missing_blob_cursor(cli_workspace["archive_root"], source)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "missing-raw-blob-cursors",
            "--apply",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "apply"
    assert payload["candidate_count"] == 1
    assert payload["deleted_cursor_count"] == 1
    with sqlite3.connect(cli_workspace["archive_root"] / "ops.db") as conn:
        assert conn.execute("SELECT 1 FROM ingest_cursor WHERE source_path = ?", (str(source),)).fetchone() is None
    with sqlite3.connect(cli_workspace["archive_root"] / "source.db") as conn:
        assert conn.execute("SELECT 1 FROM raw_sessions WHERE source_path = ?", (str(source),)).fetchone() == (1,)


def test_quarantined_accepted_raw_repair_cli_dry_run_is_bounded_json(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    del cli_workspace
    raw_id = "a" * 64
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "quarantined-accepted-raws",
            "--raw-id",
            raw_id,
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "dry-run"
    assert payload["requested_count"] == 1
    assert payload["ineligible_count"] == 1
    assert len(payload["proof_digest"]) == 64
    assert payload["items"][0]["raw_id"] == raw_id

    plain = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "quarantined-accepted-raws", "--raw-id", raw_id],
        catch_exceptions=False,
    )
    assert plain.exit_code == 0
    assert f"Proof digest: {payload['proof_digest']}" in plain.output
    assert f"{raw_id} ineligible proof=unavailable" in plain.output


def test_quarantined_accepted_raw_repair_cli_apply_requires_receipt_and_proof(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    del cli_workspace
    raw_id = "a" * 64
    missing_receipt = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "quarantined-accepted-raws", "--raw-id", raw_id, "--apply"],
    )
    assert missing_receipt.exit_code == 2
    assert "--receipt" in missing_receipt.output
    missing_proof = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "quarantined-accepted-raws",
            "--raw-id",
            raw_id,
            "--apply",
            "--receipt",
            "repair.jsonl",
        ],
    )
    assert missing_proof.exit_code == 2
    assert "--proof-digest" in missing_proof.output


def test_browser_capture_origin_repair_cli_is_bounded_and_requires_receipt_proof(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    del cli_workspace
    raw_id = "b" * 64
    dry_run = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "browser-capture-origin-mismatches",
            "--raw-id",
            raw_id,
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert dry_run.exit_code == 0
    payload = json.loads(dry_run.output)
    assert payload["mode"] == "dry-run"
    assert payload["requested_count"] == 1
    assert payload["ineligible_count"] == 1
    assert len(payload["proof_digest"]) == 64

    missing_receipt = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "browser-capture-origin-mismatches",
            "--raw-id",
            raw_id,
            "--apply",
        ],
    )
    assert missing_receipt.exit_code == 2
    assert "--receipt" in missing_receipt.output
    missing_proof = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "browser-capture-origin-mismatches",
            "--raw-id",
            raw_id,
            "--apply",
            "--receipt",
            "repair.jsonl",
        ],
    )
    assert missing_proof.exit_code == 2
    assert "--proof-digest" in missing_proof.output


def test_legacy_browser_native_id_repair_cli_is_bounded_and_requires_receipt_proof(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    del cli_workspace
    raw_id = "c" * 64
    dry_run = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "legacy-browser-capture-missing-native-id",
            "--raw-id",
            raw_id,
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert dry_run.exit_code == 0
    assert json.loads(dry_run.output)["ineligible_count"] == 1
    missing_receipt = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "legacy-browser-capture-missing-native-id",
            "--raw-id",
            raw_id,
            "--apply",
        ],
    )
    assert missing_receipt.exit_code == 2
    assert "--receipt" in missing_receipt.output


def test_browser_canonical_authority_conflicts_cli_is_read_only_by_default(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    del cli_workspace
    raw_id = "d" * 64
    dry_run = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "browser-canonical-authority-conflicts",
            "--raw-id",
            raw_id,
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert dry_run.exit_code == 0
    payload = json.loads(dry_run.output)
    assert payload["requested_count"] == 1
    assert "assertion_ids" not in payload

    plain = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "browser-canonical-authority-conflicts", "--raw-id", raw_id],
        catch_exceptions=False,
    )
    assert plain.exit_code == 0
    assert "Blocker:" not in plain.output


def test_browser_canonical_authority_conflicts_cli_record_calls_the_recording_path(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """``--record`` routes through ``record_browser_canonical_authority_conflict_blockers``.

    Exercises the CLI adapter's ``--record`` branch (polylogue-hleq NIT): the
    JSON payload gains an ``assertion_ids`` key (absent without ``--record``,
    per the sibling read-only test) and the plain-output loop over
    ``assertion_ids`` runs without error. A raw id with no resolvable session
    (this test's fixture) is exactly the shape
    ``record_browser_canonical_authority_conflict_blockers`` itself declines
    to persist a blocker for (no ``session_id`` to target), so this proves the
    CLI calls the recording function and surfaces its real (empty) result
    rather than fabricating one; the deep persistence/idempotency/judged-row-
    protection behavior of the recording function itself is covered at the
    storage layer in ``test_browser_capture_origin_repair.py``.
    """
    archive_root = cli_workspace["archive_root"]
    raw_id = "e" * 64
    dry_run = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "browser-canonical-authority-conflicts",
            "--raw-id",
            raw_id,
            "--record",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert dry_run.exit_code == 0
    payload = json.loads(dry_run.output)
    assert payload["assertion_ids"] == []
    with sqlite3.connect(archive_root / "user.db") as user_conn:
        count = user_conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'blocker'").fetchone()[0]
        assert count == 0

    plain = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "browser-canonical-authority-conflicts", "--raw-id", raw_id, "--record"],
        catch_exceptions=False,
    )
    assert plain.exit_code == 0
    assert "Blocker:" not in plain.output


def test_duplicate_raw_identity_cli_dry_run_is_bounded_and_requires_receipt_proof(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    del cli_workspace
    stale_raw_id = "f" * 64
    canonical_raw_id = "0" * 64
    dry_run = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "duplicate-raw-identity",
            "--pair",
            f"{stale_raw_id}:{canonical_raw_id}",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert dry_run.exit_code == 0
    payload = json.loads(dry_run.output)
    assert payload["mode"] == "dry-run"
    assert payload["requested_count"] == 1
    assert payload["ineligible_count"] == 1
    assert payload["items"][0]["stale_raw_id"] == stale_raw_id
    assert payload["items"][0]["canonical_raw_id"] == canonical_raw_id

    malformed = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "duplicate-raw-identity", "--pair", "not-a-pair"],
    )
    assert malformed.exit_code == 2
    assert "STALE_RAW_ID:CANONICAL_RAW_ID" in malformed.output

    missing_receipt = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "duplicate-raw-identity",
            "--pair",
            f"{stale_raw_id}:{canonical_raw_id}",
            "--apply",
        ],
    )
    assert missing_receipt.exit_code == 2
    assert "--receipt" in missing_receipt.output
    missing_proof = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "duplicate-raw-identity",
            "--pair",
            f"{stale_raw_id}:{canonical_raw_id}",
            "--apply",
            "--receipt",
            "repair.jsonl",
        ],
    )
    assert missing_proof.exit_code == 2
    assert "--proof-digest" in missing_proof.output


def test_archive_read_cli_lists_archive_sessions(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    sqlite3.connect(cli_workspace["archive_root"] / "index.db").close()

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def list_summaries(self, *, limit: int, origin: str | None) -> list[ArchiveSessionSummary]:
            assert limit == 2
            assert origin == "codex-session"
            return [
                ArchiveSessionSummary(
                    session_id="codex-session:native-1",
                    native_id="native-1",
                    origin="codex-session",
                    provider=Provider.CODEX,
                    title="Copied",
                    created_at="2026-01-02T03:04:05Z",
                    updated_at="2026-01-02T03:04:06Z",
                    message_count=3,
                    word_count=9,
                    tags=("archive",),
                )
            ]

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "archive-read",
            "--origin",
            "codex-session",
            "--limit",
            "2",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "list"
    assert payload["sessions"] == [
        {
            "created_at": "2026-01-02T03:04:05Z",
            "message_count": 3,
            "native_id": "native-1",
            "origin": "codex-session",
            "session_id": "codex-session:native-1",
            "tags": ["archive"],
            "title": "Copied",
            "updated_at": "2026-01-02T03:04:06Z",
            "word_count": 9,
        }
    ]


def test_archive_read_cli_searches_archive_blocks(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    sqlite3.connect(cli_workspace["archive_root"] / "index.db").close()

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def search_summaries(self, query: str, *, limit: int, origin: str | None) -> list[ArchiveSessionSearchHit]:
            assert query == "needle"
            assert limit == 5
            assert origin is None
            return [
                ArchiveSessionSearchHit(
                    rank=1,
                    session_id="codex-session:native-1",
                    block_id="codex-session:native-1:m1:0",
                    message_id="codex-session:native-1:m1",
                    origin="codex-session",
                    provider=Provider.CODEX,
                    title="Copied",
                    snippet="[needle]",
                )
            ]

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "archive-read",
            "--query",
            "needle",
            "--limit",
            "5",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "search"
    assert payload["hits"][0]["block_id"] == "codex-session:native-1:m1:0"
    assert payload["hits"][0]["snippet"] == "[needle]"


@pytest.mark.parametrize(
    "selection_args",
    [
        [],
        ["--only-missing"],
        ["--raw-id", "raw-a", "--raw-id", "raw-b"],
    ],
    ids=["all", "only-missing", "explicit"],
)
def test_rebuild_index_source_replay_expands_every_execution_selection_to_authority_cohorts(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    selection_args: list[str],
) -> None:
    monkeypatch.setattr("polylogue.cli.commands.maintenance._rebuild_index._count_source_raw_sessions", lambda _root: 4)
    monkeypatch.setattr(
        "polylogue.cli.commands.maintenance._rebuild_index._all_index_rebuild_raw_ids",
        lambda _root: ["raw-parent", "raw-child"],
    )
    monkeypatch.setattr(
        "polylogue.cli.commands.maintenance._rebuild_index._missing_index_raw_ids",
        lambda _root: ["raw-parent", "raw-child"],
    )

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "rebuild-index",
            *selection_args,
            *(["--no-promote"] if selection_args else []),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Classified:" in result.output
    assert "Replayed:" in result.output
    with sqlite3.connect(cli_workspace["archive_root"] / "ops.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM ingest_attempts WHERE phase = 'rebuild-index'").fetchone() == (0,)


def test_rebuild_index_force_write_option_is_retired(cli_runner: CliRunner) -> None:
    help_result = cli_runner.invoke(cli, ["ops", "maintenance", "rebuild-index", "--help"])
    result = cli_runner.invoke(cli, ["ops", "maintenance", "rebuild-index", "--force-write"])

    assert help_result.exit_code == 0
    assert "--force-write" not in help_result.output
    assert result.exit_code == 2
    assert "No such option" in result.output
    assert "--force-write" in result.output


@pytest.mark.parametrize("selection_args", [["--only-missing"], ["--raw-id", "raw-a"]])
def test_partial_rebuild_requires_no_promote_before_archive_mutation(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, selection_args: list[str]
) -> None:
    index_path = cli_workspace["archive_root"] / "index.db"
    inode_before = index_path.stat().st_ino
    generations_before = tuple(cli_workspace["archive_root"].glob(".index-generations/*"))

    result = cli_runner.invoke(cli, ["--plain", "ops", "maintenance", "rebuild-index", *selection_args])

    assert result.exit_code == 2
    assert "partial rebuild selections require --no-promote" in result.output
    assert index_path.stat().st_ino == inode_before
    assert tuple(cli_workspace["archive_root"].glob(".index-generations/*")) == generations_before


def test_all_index_rebuild_raw_ids_uses_source_acquisition_order(
    cli_workspace: dict[str, Path],
) -> None:
    source_db = cli_workspace["archive_root"] / "source.db"
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        for raw_id, acquired_at_ms in (
            ("raw-child", 30),
            ("raw-parent", 10),
            ("raw-sibling-b", 20),
            ("raw-sibling-a", 20),
        ):
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash,
                    blob_size, acquired_at_ms, validation_status
                )
                VALUES (?, 'codex-session', ?, ?, 0, randomblob(32), 1, ?, 'passed')
                """,
                (raw_id, raw_id, f"/tmp/{raw_id}.jsonl", acquired_at_ms),
            )

    assert maintenance_rebuild_index._all_index_rebuild_raw_ids(cli_workspace["archive_root"]) == [
        "raw-parent",
        "raw-sibling-a",
        "raw-sibling-b",
        "raw-child",
    ]


def test_rebuild_index_helper_returns_typed_empty_replay_receipt(tmp_path: Path) -> None:
    config = Config(
        archive_root=tmp_path,
        render_root=tmp_path / "render",
        sources=[],
        db_path=tmp_path / "index.db",
    )

    result = asyncio.run(
        rebuild_index_from_source(
            config,
            raw_ids=["raw-a", "raw-b"],
            raw_batch_size=7,
            ingest_workers=1,
            materialize=True,
            progress_callback=None,
        )
    )

    assert result == {
        "scanned_raw_count": 0,
        "classified_full_count": 0,
        "replayed_logical_source_count": 0,
        "quarantined_raw_count": 0,
        "adoption_deferred_raw_count": 0,
        "authority_selection_expanded": True,
    }


def test_rebuild_index_explicit_raw_ids_remain_inspectable_in_plan_mode(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "polylogue.cli.commands.maintenance._rebuild_index._count_source_raw_sessions", lambda _root: 10
    )

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "rebuild-index",
            "--raw-id",
            "raw-a",
            "--raw-id",
            "raw-b",
            "--raw-id",
            "raw-a",
            "--plan",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["raw_session_count"] == 10
    assert payload["selected_raw_count"] == 2
    assert payload["raw_id_count"] == 3
    assert payload["skipped_by_blob_limit_count"] == 0
    assert payload["status"] == "ok"


def test_rebuild_index_filters_selected_rows_by_blob_size(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = cli_workspace["archive_root"]
    source_db = archive_root / "source.db"
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        rows = [
            ("raw-small", 1 * 1024 * 1024, 2),
            ("raw-large", 3 * 1024 * 1024, 1),
        ]
        for raw_id, blob_size, acquired_at_ms in rows:
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash,
                    blob_size, acquired_at_ms, validation_status
                )
                VALUES (?, 'codex-session', ?, ?, 0, randomblob(32), ?, ?, 'passed')
                """,
                (raw_id, raw_id, f"/tmp/{raw_id}.jsonl", blob_size, acquired_at_ms),
            )

    monkeypatch.setattr("polylogue.cli.commands.maintenance._rebuild_index._count_source_raw_sessions", lambda _root: 2)
    monkeypatch.setattr(
        "polylogue.cli.commands.maintenance._rebuild_index._missing_index_raw_ids",
        lambda _root: ["raw-large", "raw-small"],
    )
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "rebuild-index",
            "--only-missing",
            "--max-blob-mb",
            "2",
            "--plan",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["selected_raw_count"] == 1
    assert payload["totals"]["blob_bytes"] == 1 * 1024 * 1024
    assert payload["skipped_by_blob_limit_count"] == 1
    assert payload["max_blob_mb"] == 2.0
    assert [row["raw_id"] for row in payload["top_rows"]] == ["raw-small"]


def test_rebuild_index_plan_reports_weighted_top_rows(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    archive_root = cli_workspace["archive_root"]
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    with sqlite3.connect(source_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        for raw_id, native_id, source_path, source_index, blob_size, acquired_at_ms in (
            ("raw-small", "small", "/tmp/raw-small.jsonl", 0, 1_000, 1),
            ("raw-large", "large", "/tmp/raw-large.jsonl", 0, 5_000, 2),
            ("raw-large-2", "large-2", "/tmp/raw-large.jsonl", 1, 3_000, 3),
        ):
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash,
                    blob_size, acquired_at_ms, validation_status
                )
                VALUES (?, 'codex-session', ?, ?, ?, randomblob(32), ?, ?, 'passed')
                """,
                (raw_id, native_id, source_path, source_index, blob_size, acquired_at_ms),
            )
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, message_count, content_hash)
            VALUES ('large', 'codex-session', 'raw-large', 42, randomblob(32))
            """
        )
        session_id = conn.execute("SELECT session_id FROM sessions WHERE raw_id = 'raw-large'").fetchone()[0]
        conn.execute(
            """
            INSERT INTO session_events (session_id, position, event_type, summary)
            VALUES (?, 0, 'capture_gap', 'gap')
            """,
            (session_id,),
        )

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "rebuild-index",
            "--plan",
            "--plan-limit",
            "1",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["raw_session_count"] == 3
    assert payload["selected_raw_count"] == 3
    assert payload["replay_order"] == "acquired_at_ms_asc_raw_id_asc"
    assert payload["risk_order"] == "blob_size_desc"
    assert payload["cost_basis"]["primary"] == "source.db raw_sessions.blob_size"
    assert payload["totals"]["blob_bytes"] == 9_000
    assert payload["totals"]["materialized_messages"] == 42
    assert payload["totals"]["materialized_session_events"] == 1
    assert [row["raw_id"] for row in payload["top_rows"]] == ["raw-large"]
    assert payload["top_rows"][0]["materialized_messages"] == 42
    assert payload["top_groups"] == [
        {
            "origin": "codex-session",
            "native_id": "large",
            "source_path": "/tmp/raw-large.jsonl",
            "row_count": 2,
            "blob_bytes": 8_000,
            "first_acquired_at_ms": 2,
            "last_acquired_at_ms": 3,
            "materialized_sessions": 1,
            "materialized_messages": 42,
            "materialized_session_events": 1,
        }
    ]
