from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.commands.maintenance import _rebuild_index as maintenance_rebuild_index
from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.core.json import json_document
from polylogue.maintenance.replay import rebuild_index_from_source
from polylogue.sources.revision_backfill import census_historical_revision_evidence
from polylogue.storage.blob_gc import read_gc_history
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.raw_authority import RawReplayPlan, record_raw_authority_census
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSearchHit, ArchiveSessionSummary, ArchiveStore
from polylogue.storage.sqlite.archive_tiers.archive_init import (
    ArchiveInitResult,
    ArchiveTierInitResult,
)
from polylogue.storage.sqlite.archive_tiers.archive_plan import ArchiveInitAction, ArchiveInitPlan
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt

_ARCHIVE_TIERS = tuple(spec.filename for spec in ARCHIVE_TIER_SPECS.values())


def test_raw_authority_census_cli_resolves_receipt_handle(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    root = cli_workspace["archive_root"]
    receipt = record_raw_authority_census(
        root,
        (),
        selected_plan_ids=set(),
        executable_plan_ids=set(),
        mode="dry_run",
        quiescent=True,
        scope={"source_family": "codex"},
        residual={},
    )

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "raw-authority-census",
            receipt.query_handle,
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["query_handle"] == receipt.query_handle
    assert payload["census"]["census_id"] == receipt.census_id


def test_raw_authority_cli_bounds_oversized_plan_and_resolves_detail(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    root = cli_workspace["archive_root"]
    raw_ids = tuple(f"raw-{index:05d}" for index in range(2_000))
    plan = RawReplayPlan(
        "raw-replay:cli-oversized",
        "b" * 64,
        raw_ids,
        ("codex:oversized",),
        json_document({"raw_ids": list(raw_ids)}),
        json_document({"raw_ids": list(raw_ids)}),
        json_document({"raw_ids": list(raw_ids)}),
    )
    receipt = record_raw_authority_census(
        root,
        (plan,),
        selected_plan_ids=set(),
        executable_plan_ids={plan.plan_id},
        mode="dry_run",
        quiescent=True,
        scope={"test": "cli-oversized"},
        residual={},
    )

    census_result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "raw-authority-census",
            receipt.query_handle,
            "--limit",
            "1",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert census_result.exit_code == 0
    assert len(census_result.output) < 8_000
    census_payload = json.loads(census_result.stdout)
    item = census_payload["plans"][0]
    assert item["plan"]["input_raw_count"] == 2_000
    assert "raw-01999" not in census_result.output

    detail_result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "raw-authority-detail",
            item["detail_query_handle"],
            "--chunk-chars",
            "256",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert detail_result.exit_code == 0
    detail_payload = json.loads(detail_result.stdout)
    assert len(detail_payload["chunk"]) <= 256
    assert detail_payload["next_query_handle"] is not None


def _seed_raw_authority_blocker(
    archive_root: Path,
    *,
    blocker_id: str,
    plan_id: str = "raw-replay:cli-test-plan",
    census_id: str = "raw-authority-census:cli-test",
    frontier: bool = False,
    judgment_assertion_id: str | None = None,
    reason: str = "immutable source/index preconditions changed after the census",
) -> None:
    """Directly seed one real, unresolved ``raw_authority_blockers`` row.

    Mirrors the full production shape (census -> plan -> blocker, plus one
    real ``raw_sessions`` row so a non-frontier resolution can genuinely
    replan it) with hand-built minimal rows rather than driving the full
    census/reject-stale workflow (see
    ``tests/unit/storage/test_raw_authority_ledger.py`` for that heavier
    path) -- sufficient to exercise ``BlockerResolveActuator.prepare``'s real
    read against ``source.db`` and, for non-frontier blockers,
    ``resolve_raw_authority_blocker``'s real replan.

    ``frontier`` alone seeds a ``frontier_obligation``-kind blocker (missing
    bytes / unresolved provenance / corrupt); pass ``judgment_assertion_id``
    too for a genuine ``frontier_judgment`` blocker, matching how
    ``_reconcile_frontier_obligations`` only writes that key for
    ``CONFLICTING_AUTHORITY_NEEDS_JUDGMENT`` plans.
    """
    raw_id = f"raw-{blocker_id}"
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        payload = (
            b'{"type":"session_meta","payload":{"id":"' + blocker_id.encode() + b'"}}\n'
            b'{"type":"response_item","payload":{"type":"message","id":"m-1",'
            b'"role":"user","content":[{"type":"input_text","text":"hi"}]}}\n'
        )
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path=f"{blocker_id}.jsonl",
            acquired_at_ms=1000,
            raw_id=raw_id,
        )

    witness_schema = "polylogue.raw-authority-frontier-plan.v1" if frontier else "polylogue.raw-authority-plan.v1"
    observed_json = json.dumps({"judgment_assertion_id": judgment_assertion_id}) if judgment_assertion_id else "{}"
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        next_sequence_no = int(
            conn.execute("SELECT COALESCE(MAX(sequence_no), 0) + 1 FROM raw_authority_censuses").fetchone()[0]
        )
        conn.execute(
            """
            INSERT INTO raw_authority_censuses (
                census_id, sequence_no, scope_json, residual_json, parser_fingerprint,
                mode, lifecycle_status, quiescent, inventory_digest, residual_digest,
                plan_count, post_inventory_digest, post_residual_json, post_residual_digest,
                post_plan_count, postflight_at_ms, executable_plan_count, residual_plan_count,
                predecessor_census_id, fixed_point, created_at_ms, completed_at_ms
            ) VALUES (?, ?, '{}', '{}', 'cli-test-fp', 'apply', 'completed', 1, ?, ?, 1,
                      ?, '{}', ?, 0, 1000, 1, 0, NULL, 0, 1000, 1000)
            """,
            (census_id, next_sequence_no, "a" * 64, "b" * 64, "c" * 64, "d" * 64),
        )
        conn.execute(
            """
            INSERT INTO raw_authority_plans (
                plan_id, input_digest, input_raw_ids_json, logical_keys_json,
                authority_witness_json, source_preconditions_json, index_preconditions_json,
                created_at_ms
            ) VALUES (?, ?, ?, '[]', ?, '{}', '{}', 1000)
            """,
            (plan_id, "e" * 64, json.dumps([raw_id]), json.dumps({"schema": witness_schema})),
        )
        conn.execute(
            """
            INSERT INTO raw_authority_blockers (
                blocker_id, plan_id, census_id, reason, expected_json, observed_json, created_at_ms
            ) VALUES (?, ?, ?, ?, '{}', ?, 1000)
            """,
            (blocker_id, plan_id, census_id, reason, observed_json),
        )
        conn.commit()


def test_raw_authority_blocker_resolution_cli_requires_confirmation(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    root = cli_workspace["archive_root"]
    _seed_raw_authority_blocker(root, blocker_id="blocker-1")
    base = [
        "--plain",
        "ops",
        "maintenance",
        "raw-authority-blocker-resolve",
        "--blocker-id",
        "blocker-1",
        "--reason",
        "reviewed current evidence",
    ]
    refused = cli_runner.invoke(cli, base)
    accepted = cli_runner.invoke(cli, [*base, "--yes"], catch_exceptions=False)

    assert refused.exit_code != 0
    assert "without --yes" in refused.output
    assert accepted.exit_code == 0
    assert "Resolved blocker-1" in accepted.output
    with sqlite3.connect(root / "source.db") as conn:
        row = conn.execute(
            "SELECT resolved_at_ms, resolution FROM raw_authority_blockers WHERE blocker_id = ?", ("blocker-1",)
        ).fetchone()
    assert row is not None
    assert row[0] is not None
    assert "reviewed current evidence" in str(row[1])


def test_raw_authority_blocker_resolution_cli_refuses_unknown_blocker(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """Anti-vacuity: an unknown/already-resolved blocker id must not silently succeed."""
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "raw-authority-blocker-resolve",
            "--blocker-id",
            "does-not-exist",
            "--reason",
            "reviewed current evidence",
            "--yes",
        ],
    )
    assert result.exit_code != 0
    assert "not found or already resolved" in result.output


def test_raw_authority_blockers_cli_lists_unresolved_and_classifies_kind(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    root = cli_workspace["archive_root"]
    _seed_raw_authority_blocker(root, blocker_id="blocker-stale", plan_id="raw-replay:stale-plan")
    _seed_raw_authority_blocker(
        root,
        blocker_id="blocker-frontier",
        plan_id="raw-replay:frontier-plan",
        census_id="raw-authority-census:frontier-test",
        frontier=True,
        judgment_assertion_id="judgment:frontier-conflict",
        reason="conflicting canonical authority",
    )
    _seed_raw_authority_blocker(
        root,
        blocker_id="blocker-obligation",
        plan_id="raw-replay:obligation-plan",
        census_id="raw-authority-census:obligation-test",
        frontier=True,
        reason="missing bytes require reacquisition",
    )

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "raw-authority-blockers", "--output-format", "json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    by_id = {row["blocker_id"]: row for row in payload["blockers"]}
    assert by_id["blocker-stale"]["kind"] == "stale_plan"
    assert by_id["blocker-frontier"]["kind"] == "frontier_judgment"
    assert by_id["blocker-obligation"]["kind"] == "frontier_obligation"
    assert payload["total_count"] == 3
    assert payload["truncated"] is False

    # Resolving one blocker removes it from the unresolved listing.
    resolve = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "raw-authority-blocker-resolve",
            "--blocker-id",
            "blocker-stale",
            "--reason",
            "acknowledged",
            "--yes",
        ],
        catch_exceptions=False,
    )
    assert resolve.exit_code == 0

    after = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "raw-authority-blockers", "--output-format", "json"],
        catch_exceptions=False,
    )
    after_payload = json.loads(after.stdout)
    remaining = {row["blocker_id"] for row in after_payload["blockers"]}
    assert remaining == {"blocker-frontier", "blocker-obligation"}
    assert after_payload["total_count"] == 2


def test_raw_authority_blockers_cli_paginates_past_the_limit(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """Anti-vacuity for the Codex-flagged hard cap (PR #3258): --limit below
    the total unresolved count must still expose every blocker via
    --offset, with truncated/next_offset telling the operator to page."""
    root = cli_workspace["archive_root"]
    for index in range(3):
        _seed_raw_authority_blocker(
            root,
            blocker_id=f"blocker-page-{index}",
            plan_id=f"raw-replay:page-plan-{index}",
            census_id=f"raw-authority-census:page-{index}",
        )

    first = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "raw-authority-blockers",
            "--limit",
            "2",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    first_payload = json.loads(first.stdout)
    assert first_payload["returned_count"] == 2
    assert first_payload["total_count"] == 3
    assert first_payload["truncated"] is True
    next_offset = first_payload["next_offset"]
    assert next_offset == 2

    second = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "raw-authority-blockers",
            "--limit",
            "2",
            "--offset",
            str(next_offset),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    second_payload = json.loads(second.stdout)
    assert second_payload["returned_count"] == 1
    assert second_payload["truncated"] is False

    seen_ids = {row["blocker_id"] for row in first_payload["blockers"]} | {
        row["blocker_id"] for row in second_payload["blockers"]
    }
    assert seen_ids == {"blocker-page-0", "blocker-page-1", "blocker-page-2"}

    # Plain-mode output surfaces the truncation notice too.
    plain_first = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "raw-authority-blockers", "--limit", "2"],
        catch_exceptions=False,
    )
    assert "Truncated: pass --offset 2" in plain_first.output


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
    shutil.rmtree(
        archive_root / ".maintenance-state" / "durable-change-trains",
        ignore_errors=True,
    )


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
    _refresh_fresh_bootstrap_marker(path.parent)


def _refresh_fresh_bootstrap_marker(archive_root: Path) -> None:
    """Rebind a fixture bootstrap receipt after deliberate durable-tier edits."""
    marker = archive_root / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    assert marker.is_file(), f"fixture must carry a fresh bootstrap marker: {marker}"
    from polylogue.storage.sqlite.durable_change_train import _record_fresh_durable_bootstrap

    marker.unlink()
    _record_fresh_durable_bootstrap(archive_root)


def _freeze_rebuild_fixture_source(archive_root: Path, *, expected_raws: int) -> None:
    """Census fixture raws, then record their explicit single-revision decision."""
    census = census_historical_revision_evidence(archive_root)
    assert census.scanned == expected_raws
    assert census.classified_full == expected_raws
    with sqlite3.connect(archive_root / "source.db") as source:
        source.execute(
            """
            UPDATE raw_sessions
            SET revision_authority = 'byte_proven',
                revision_kind = 'full',
                source_revision = raw_id,
                baseline_raw_id = raw_id,
                predecessor_raw_id = NULL,
                acquisition_generation = 0
            """
        )
        source.commit()


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
    payload = json.loads(result.stdout)
    assert payload["ready"] is True
    assert {tier["tier"]: tier["action"] for tier in payload["tiers"]} == {
        spec.tier.value: "create" for spec in ARCHIVE_TIER_SPECS.values()
    }
    assert {Path(tier["path"]).name for tier in payload["tiers"]} == set(_ARCHIVE_TIERS)
    assert {tier["tier"]: tier["durability"] for tier in payload["tiers"]} == {
        spec.tier.value: spec.durability for spec in ARCHIVE_TIER_SPECS.values()
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
    payload = json.loads(result.stdout)
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
    payload = json.loads(result.stdout)
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
    payload = json.loads(result.stdout)
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
    rows = [json.loads(line) for line in result.stdout.splitlines()]
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
    assert result.stdout == f"Exported 1 assertions to {out_path}\n"
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
    payload = json.loads(result.stdout)
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


def test_blob_gc_cli_has_no_mutate_flag(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """Read-only by design (automagic-invariants, polylogue-gd6v/4jsk/cfvvt): daemon
    convergence (``periodic_blob_gc_check``) already reclaims eligible blobs
    automatically in bounded batches, so a manual apply path would be a
    redundant, doctrine-forbidden break-glass surface -- the same treatment
    PR #3286 applied to ``embedding-orphan-reconcile`` in the same change.
    """
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-gc", "--yes"],
    )

    # See test_embedding_orphan_reconcile_cli_has_no_mutate_flag: Click's
    # CliRunner surfaces a rejected/unknown option as SystemExit(2), so exit
    # code 2 plus the option name in the rejection message is the stable
    # contract to assert on.
    assert result.exit_code == 2
    assert "--yes" in result.output


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
    payload = json.loads(inspected.stdout)
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
    payload = json.loads(abandoned.stdout)
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
    payload = json.loads(result.stdout)
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
    payload = json.loads(result.stdout)
    assert payload["mode"] == "attachment_acquisition_debt"
    assert payload["mutates"] is False
    assert payload["ok"] is True
    assert payload["total_attachments"] == 0
    assert payload["acquired_missing_blob_count"] == 0
    assert payload["acquired_reachable_count"] == 0
    assert payload["acquired_unreachable_count"] == 0


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
    assert "Unfetched:           0 (honest floor, not missing blobs)" in result.output
    assert "Acquired missing:    0 (genuine debt)" in result.output
    assert "Status:              ok" in result.output


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
    payload = json.loads(result.stdout)
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
        ["--plain", "ops", "maintenance", "blob-reference-replace-from-source"],
    )

    assert result.exit_code != 0
    assert "--manifest-file" in result.output


def test_blob_reference_replace_from_source_preview_cli_does_not_require_manifest(
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
            "blob-reference-replace-from-source-preview",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "blob_reference_replace_from_source"
    assert payload["mutates"] is False
    assert payload["writes_manifest"] is False
    assert payload["candidate_rows"] == 1
    assert payload["replaced_rows"] == 0
    with sqlite3.connect(cli_workspace["archive_root"] / "source.db") as conn:
        refs = conn.execute("SELECT blob_hash FROM blob_refs WHERE source_path = ?", (str(source),)).fetchall()
    assert refs, "preview must not mutate blob_refs"


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
            "--manifest-file",
            str(manifest),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "blob_reference_replace_from_source"
    assert payload["mutates"] is True
    assert payload["writes_manifest"] is True
    assert payload["candidate_rows"] == 1
    assert payload["replaced_rows"] == 1
    assert payload["skipped_error"] == 0
    manifest_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["old_blob_hash"] != manifest_rows[0]["new_blob_hash"]


def test_blob_reference_prune_orphans_preview_cli_keeps_refs(
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
            "blob-reference-prune-orphans-preview",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
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
            "--quarantine-file",
            str(quarantine_file),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
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
    from polylogue.core.enums import BlockType, MaterialOrigin, Origin
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
                    embedding_input_hash=hashlib.sha256(orphan_message_id.encode("utf-8")).digest(),
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
    index_db = archive_root / "index.db"
    (archive_root / ".index-active-pointer").write_text(str(index_db.resolve()), encoding="utf-8")
    generations = archive_root / ".index-generations" / "gen-current"
    generations.mkdir(parents=True, exist_ok=True)
    (generations / "generation.json").write_text(
        json.dumps(
            {
                "generation_id": "gen-current",
                "owner_id": "test",
                "archive_root": str(archive_root),
                "index_path": str(index_db.resolve()),
                "state": "active",
                "created_at_ms": 1_700_000_000_000,
                "source_snapshot": "source-at-rebuild-start",
            }
        ),
        encoding="utf-8",
    )
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
    payload = json.loads(result.stdout)
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
    assert "Would remove:  1 message ref(s), 1 status row(s)" in result.output
    assert "Removed:" not in result.output
    with sqlite3.connect(cli_workspace["archive_root"] / "embeddings.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1


def test_embedding_orphan_reconcile_cli_has_no_mutate_flag(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """Read-only by design (automagic-invariants, polylogue-gd6v/4jsk): daemon
    convergence already reconciles this backlog automatically, so a manual
    apply path would be a redundant, doctrine-forbidden break-glass surface.
    """
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "embedding-orphan-reconcile", "--yes"],
    )

    # Click's CliRunner surfaces a rejected/unknown option as SystemExit(2)
    # (its own UsageError is caught and converted before invoke() returns),
    # so exit code 2 plus the option name in the rejection message is the
    # stable, public contract to assert on -- not Click's internal
    # exception wording, which isn't guaranteed across versions.
    assert result.exit_code == 2
    assert "--yes" in result.output


def test_archive_init_cli_is_dry_run_without_yes(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    _stage_uninitialized_archive(cli_workspace)
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "archive-init", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
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
    payload = json.loads(result.stdout)
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
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["tier"] == "user"
    assert payload["from_version"] == 3
    assert payload["to_version"] == USER_SCHEMA_VERSION
    assert payload["applied_versions"] == list(range(4, USER_SCHEMA_VERSION + 1))
    assert payload["backup_receipt"] == str(manifest.with_name("verification-receipt.json"))
    with sqlite3.connect(user_db) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").fetchone()
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='context_deliveries'"
        ).fetchone()


def test_migrate_tier_cli_executes_and_persists_a_future_change_train(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite import migration_runner
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
    from polylogue.storage.sqlite.migration_runner import (
        DurableChangeRider,
        DurableRuntimeConsumer,
        declare_durable_change_train,
        durable_change_train_to_payload,
        durable_migration_claim_for_sql,
    )

    package_root = tmp_path / "fixture_migrations_cli"
    source_package = package_root / "source"
    source_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (source_package / "__init__.py").write_text("", encoding="utf-8")
    sql = "-- migration-safety: additive-no-backup\nCREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;\n"
    (source_package / "002_future_items.sql").write_text(sql, encoding="utf-8")
    claim = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        "002_future_items.sql",
        sql,
        owner_ref="owner:cli-source",
    )
    rider = DurableChangeRider(
        rider_id="rider:cli",
        owner_ref="owner:cli-rider",
        schema_objects=("table:future_items",),
        runtime_consumers=(
            DurableRuntimeConsumer(
                "bootstrap",
                "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_database",
                "proof:bootstrap",
                ("write",),
            ),
            DurableRuntimeConsumer(
                "daemon-health",
                "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_tier",
                "proof:daemon-health",
                ("read",),
            ),
        ),
        behavior_proof_refs=("proof:bootstrap", "proof:daemon-health"),
    )
    declared = declare_durable_change_train(
        train_id="train:source:cli-v2",
        tier=ArchiveTier.SOURCE,
        current_version=1,
        target_version=2,
        slot=2,
        owner_ref="owner:cli-source",
        migration=claim,
        riders=(rider,),
        declared_at_ms=1,
    )
    (source_package / "002.train.json").write_text(
        json.dumps(durable_change_train_to_payload(declared)), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(migration_runner, "_migration_package", lambda _tier: "fixture_migrations_cli.source")
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: "fixture_migrations_cli.source",
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train.DURABLE_MIGRATION_ADOPTION_FLOORS",
        {ArchiveTier.SOURCE: 1, ArchiveTier.USER: 1},
    )
    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[ArchiveTier.SOURCE] = 2
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    monkeypatch.setattr(bootstrap, "ARCHIVE_VERSION_BY_TIER", versions)
    ddl = dict(ARCHIVE_DDL_BY_TIER)
    ddl[ArchiveTier.SOURCE] = (
        "CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT; "
        "CREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;"
    )
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    source_db = cli_workspace["archive_root"] / "source.db"
    source_db.unlink()
    with sqlite3.connect(source_db) as conn:
        conn.execute("CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    _refresh_fresh_bootstrap_marker(cli_workspace["archive_root"])

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "migrate-tier", "source", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["train_state"] == "released"
    assert payload["applied_versions"] == [2]
    manifest = Path(payload["train_manifest"])
    assert manifest.exists()
    with sqlite3.connect(source_db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone() == (2,)
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='future_items'").fetchone() == ("future_items",)


def test_migrate_tier_cli_exposes_forward_version_receipt(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.cli.commands.maintenance import _migrate_tier
    from polylogue.storage.sqlite.durable_change_train import (
        DurableChangeTrainExecution,
        DurableForwardVersionReceipt,
    )

    source_db = cli_workspace["archive_root"] / "source.db"
    if not source_db.exists():
        with sqlite3.connect(source_db) as conn:
            conn.execute("PRAGMA user_version = 3")
            conn.commit()
    receipt = DurableForwardVersionReceipt(
        tier=ArchiveTier.SOURCE,
        historical_train_id="train:source:v2",
        historical_target_version=2,
        current_target_version=3,
        observed_live_version=3,
        historical_schema_inventory_sha256="a" * 64,
        archive_identity_digest="b" * 64,
    )
    monkeypatch.setattr(
        _migrate_tier,
        "execute_durable_change_train",
        lambda *_args, **_kwargs: DurableChangeTrainExecution(
            train=None,
            manifest_path=None,
            migration_result=None,
            forward_version_receipt=receipt,
        ),
    )

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "migrate-tier", "source", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["forward_version_receipt"] == {
        "archive_identity_digest": "b" * 64,
        "current_target_version": 3,
        "historical_schema_inventory_sha256": "a" * 64,
        "historical_target_version": 2,
        "historical_train_id": "train:source:v2",
        "observed_live_version": 3,
        "tier": "source",
    }

    plain = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "migrate-tier", "source"],
        catch_exceptions=False,
    )
    assert plain.exit_code == 0, plain.output
    assert "historical train train:source:v2 is admitted at live schema v3" in plain.output
    assert "(target v3)" in plain.output


def test_migrate_tier_cli_refuses_live_daemon_before_sql(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    user_db = cli_workspace["archive_root"] / "user.db"
    _create_user_v3(user_db)
    daemon = subprocess.Popen(
        ["bash", "-c", "exec -a polylogued python3 -c 'import time; time.sleep(30)'"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        (cli_workspace["archive_root"] / "daemon.pid").write_text(f"{daemon.pid}\n", encoding="utf-8")
        time.sleep(0.1)
        result = cli_runner.invoke(
            cli,
            ["--plain", "ops", "maintenance", "migrate-tier", "user", "--output-format", "json"],
            catch_exceptions=False,
        )
    finally:
        daemon.terminate()
        daemon.wait(timeout=5)

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "daemon to be stopped" in payload["error"]
    with sqlite3.connect(user_db) as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3


def test_migrate_tier_cli_uses_shared_stable_archive_lock(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation

    user_db = cli_workspace["archive_root"] / "user.db"
    _create_user_v3(user_db)
    with OwnedArchiveLocation.acquire(
        ArchiveLocation.resolve(cli_workspace["archive_root"]),
        owner_id="test:daemon-owner",
    ):
        result = cli_runner.invoke(
            cli,
            ["--plain", "ops", "maintenance", "migrate-tier", "user", "--output-format", "json"],
            catch_exceptions=False,
        )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "archive location already owned" in payload["error"]
    with sqlite3.connect(user_db) as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3


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
    assert "successful backup verification receipt" in json.loads(result.stdout)["error"]
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
    assert "tier artifact hash mismatch" in json.loads(result.stdout)["error"]
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
    payload = json.loads(result.stdout)
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


def test_cursor_authority_reconcile_cli_exposes_only_scoped_inputs(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "cursor-authority-reconcile", "--help"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "--source-path-file" in result.output
    assert "--output-plan" in result.output
    assert "--plan" in result.output
    assert "--backup-manifest" in result.output
    assert "--receipt" in result.output
    assert "--apply" in result.output
    assert "--force" not in result.output
    assert "--bypass" not in result.output


def test_cursor_authority_reconcile_cli_accepts_verified_backup_directory(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.maintenance import cursor_authority_reconcile

    backup = tmp_path / "verified-backup"
    backup.mkdir()
    (backup / "manifest.json").write_text("{}", encoding="utf-8")
    observed: dict[str, Path] = {}

    def fake_apply(*, plan_path: Path, backup_manifest: Path, receipt: Path) -> dict[str, object]:
        observed["backup"] = backup_manifest
        return {"verdict": "failed"}

    monkeypatch.setattr(cursor_authority_reconcile, "apply_reconciliation", fake_apply)
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "cursor-authority-reconcile",
            "--apply",
            "--plan",
            str(tmp_path / "plan.json"),
            "--backup-manifest",
            str(backup),
            "--receipt",
            str(tmp_path / "receipt.json"),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert observed["backup"] == backup


@pytest.mark.parametrize(
    "args",
    [
        ["--apply", "--backup-manifest", "backup", "--receipt", "receipt"],
        ["--apply", "--plan", "plan", "--receipt", "receipt"],
        ["--apply", "--plan", "plan", "--backup-manifest", "backup"],
        ["--source-path-file", "source", "--output-plan", "plan", "--plan", "existing"],
        ["--source-path-file", "source", "--output-plan", "plan", "--receipt", "receipt"],
        ["--plan", "plan", "--backup-manifest", "backup", "--receipt", "receipt"],
    ],
)
def test_cursor_authority_reconcile_cli_rejects_mixed_or_missing_mode_options(
    cli_runner: CliRunner,
    args: list[str],
) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "cursor-authority-reconcile", *args],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    assert "requires" in result.output or "accepts only" in result.output


def test_raw_authority_frontier_cli_replaces_incident_specific_commands(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "raw-authority-frontier",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["accepted_head_count"] == 0
    assert payload["plan_count"] == 0
    assert payload["executable_plan_count"] == 0
    assert payload["state_counts"] == {}
    assert payload["query_handle"].startswith("polylogue://raw-authority-census/")

    help_result = cli_runner.invoke(cli, ["--plain", "ops", "maintenance", "--help"])
    assert help_result.exit_code == 0
    assert "raw-authority-frontier" in help_result.output
    for removed in (
        "missing-raw-blob-cursors",
        "quarantined-accepted-raws",
        "browser-capture-origin-mismatches",
        "legacy-browser-capture-missing-native-id",
        "browser-canonical-authority-conflicts",
        "duplicate-raw-identity",
    ):
        assert removed not in help_result.output

    apply_without_confirmation = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "raw-authority-frontier",
            "--apply-plan",
            "raw-authority-frontier:" + "a" * 64,
            "--preview-census",
            payload["census_id"],
        ],
    )
    assert apply_without_confirmation.exit_code == 1
    assert "without --yes" in apply_without_confirmation.output


def test_raw_authority_frontier_cli_refuses_durable_census_while_daemon_runs(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """A census reconciles durable obligations, so it needs writer exclusion."""
    with patch("polylogue.maintenance.offline_guard.running_daemon_pid", return_value=123):
        result = cli_runner.invoke(
            cli,
            ["--plain", "ops", "maintenance", "raw-authority-frontier", "--output-format", "json"],
        )

    assert result.exit_code == 1
    assert "Refusing offline maintenance while polylogued PID 123 is running" in result.output


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
    payload = json.loads(result.stdout)
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
    payload = json.loads(result.stdout)
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
    receipt_path = write_valid_rebuild_receipt(
        cli_workspace["archive_root"], cli_workspace["archive_root"].parent / "schema-inference-gate-receipt.json"
    )
    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(receipt_path))
    monkeypatch.setattr("polylogue.maintenance.rebuild_index.count_source_raw_sessions", lambda _root: 4)
    monkeypatch.setattr(
        "polylogue.maintenance.rebuild_index.all_index_rebuild_raw_ids",
        lambda _root: ["raw-parent", "raw-child"],
    )
    monkeypatch.setattr(
        "polylogue.maintenance.rebuild_index.missing_index_raw_ids",
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


def test_rebuild_index_preflight_reports_durable_schema_currency(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    root = cli_workspace["archive_root"]
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute("DROP INDEX idx_raw_failure_disposition_receipts_disposed_at")
        conn.execute("DROP TABLE raw_failure_disposition_receipts")
        conn.execute("PRAGMA user_version = 28")

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "rebuild-index", "--preflight", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["kind"] == "rebuild-schema-currency"
    assert payload["status"] == "blocked"
    assert [tier["tier"] for tier in payload["tiers"]] == ["audit", "source", "user"]
    assert payload["blocking_tiers"][0]["tier"] == "source"
    assert payload["blocking_tiers"][0]["actual_user_version"] == 28
    assert payload["blocking_tiers"][0]["expected_user_version"] == 29
    assert "migrate or deploy before rebuilding" in result.stderr


def test_migrate_tier_cli_initializes_only_an_absent_durable_tier(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    _stage_uninitialized_archive(cli_workspace)
    blob_root = cli_workspace["archive_root"] / "blob"
    blob_root.mkdir()
    assert blob_root.is_dir()
    assert not any(blob_root.iterdir())
    audit_db = cli_workspace["archive_root"] / "audit.db"

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["tier"] == "audit"
    assert payload["initialized"] is True
    assert payload["from_version"] == 0
    assert payload["to_version"] == 1
    with sqlite3.connect(audit_db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone() == (1,)
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_migrate_tier_cli_missing_initialization_refuses_an_existing_tier(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    audit_db = cli_workspace["archive_root"] / "audit.db"
    before = audit_db.read_bytes()

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "already exists; refusing missing-tier initialization" in json.loads(result.stdout)["error"]
    assert audit_db.read_bytes() == before


def test_migrate_tier_cli_missing_initialization_loses_publish_race_without_replacement(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage_uninitialized_archive(cli_workspace)
    audit_db = cli_workspace["archive_root"] / "audit.db"
    raced_bytes = b"concurrent durable owner\n"
    real_link = os.link

    def create_target_before_publish(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        assert dst_dir_fd is not None
        target_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=dst_dir_fd)
        try:
            os.write(target_fd, raced_bytes)
        finally:
            os.close(target_fd)
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr("polylogue.operations.durable_change_train.os.link", create_target_before_publish)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "appeared during initialization; refusing to replace it" in json.loads(result.stdout)["error"]
    assert audit_db.read_bytes() == raced_bytes
    assert not list(audit_db.parent.glob(".audit.db.initialize-*.tmp"))


def test_migrate_tier_cli_rejects_archive_directory_swap_before_publication(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage_uninitialized_archive(cli_workspace)
    root = cli_workspace["archive_root"]
    moved_root = root.with_name("archive-moved")
    swapped = False

    def swap_after_archive_ownership(_root: Path) -> str:
        nonlocal swapped
        if not swapped:
            root.rename(moved_root)
            root.mkdir()
            swapped = True
        return "proof:daemon-stopped"

    monkeypatch.setattr(
        "polylogue.cli.commands.maintenance._migrate_tier._require_stopped_daemon",
        swap_after_archive_ownership,
    )

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "changed during validation" in json.loads(result.stdout)["error"]
    assert not (root / "audit.db").exists()
    assert not (moved_root / "audit.db").exists()


def test_migrate_tier_cli_wraps_non_collision_publication_error(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage_uninitialized_archive(cli_workspace)
    audit_db = cli_workspace["archive_root"] / "audit.db"

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("cross-device link")

    monkeypatch.setattr("polylogue.operations.durable_change_train.os.link", fail_link)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    error = json.loads(result.stdout)["error"]
    assert f"cannot initialize missing audit tier: anonymous durable publication failed: {audit_db}" in error
    assert not audit_db.exists()


@pytest.mark.parametrize("failure_stage", ["image_fsync", "directory_open", "directory_fsync"])
def test_migrate_tier_cli_cleans_up_after_publication_failure(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """The real publication route leaves no owned target after a failure."""
    _stage_uninitialized_archive(cli_workspace)
    root = cli_workspace["archive_root"]
    audit_db = root / "audit.db"
    module_name = "polylogue.operations.durable_change_train"

    real_fsync = os.fsync
    fsync_calls = 0
    directory_fsyncs = 0

    def fail_fsync(descriptor: int) -> None:
        nonlocal directory_fsyncs, fsync_calls
        fsync_calls += 1
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        if is_directory:
            directory_fsyncs += 1
        if failure_stage == "image_fsync" and not is_directory and fsync_calls == 2:
            raise OSError(f"{failure_stage} failed")
        if failure_stage == "directory_fsync" and is_directory:
            raise OSError(f"{failure_stage} failed")
        real_fsync(descriptor)

    monkeypatch.setattr(f"{module_name}.os.fsync", fail_fsync)

    real_open = os.open
    dup_failure_armed = False

    from polylogue.cli.commands.maintenance import _migrate_tier

    real_require_stopped_daemon = _migrate_tier._require_stopped_daemon

    def arm_dup_failure(path: Path) -> str:
        nonlocal dup_failure_armed
        dup_failure_armed = True
        return real_require_stopped_daemon(path)

    monkeypatch.setattr(_migrate_tier, "_require_stopped_daemon", arm_dup_failure)

    def fail_directory_open(
        file: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if (
            failure_stage == "directory_open"
            and dup_failure_armed
            and Path(file) == root
            and flags & getattr(os, "O_DIRECTORY", 0)
            and flags & getattr(os, "O_TMPFILE", 0) != getattr(os, "O_TMPFILE", 0)
        ):
            raise OSError("directory open failed")
        return real_open(file, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(f"{module_name}.os.open", fail_directory_open)

    real_dup = os.dup

    def fail_directory_dup(descriptor: int) -> int:
        if failure_stage == "directory_open" and dup_failure_armed:
            raise OSError("directory open failed")
        return real_dup(descriptor)

    monkeypatch.setattr(f"{module_name}.os.dup", fail_directory_dup)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert f"cannot publish audit tier at {audit_db}" in json.loads(result.stdout)["error"]
    if failure_stage == "directory_fsync":
        assert audit_db.exists()
    else:
        assert not audit_db.exists()


def test_migrate_tier_cli_serializes_cleanup_fsync_uncertainty(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cleanup fsync fault is durable-recovery uncertainty, not a lost note."""
    _stage_uninitialized_archive(cli_workspace)
    root = cli_workspace["archive_root"]
    audit_db = root / "audit.db"
    real_stat = os.stat
    published_stat_calls = 0

    def fail_published_stat(
        file: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal published_stat_calls
        if file == "audit.db" and dir_fd is not None:
            published_stat_calls += 1
            if published_stat_calls == 2:
                raise OSError("published identity fault")
        return real_stat(file, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    real_fsync = os.fsync

    def fail_cleanup_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("cleanup fsync fault")
        real_fsync(descriptor)

    monkeypatch.setattr("polylogue.operations.durable_change_train.os.stat", fail_published_stat)
    monkeypatch.setattr("polylogue.operations.durable_change_train.os.fsync", fail_cleanup_fsync)
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["durable_recovery"] == {
        "code": "cleanup_not_atomic",
        "detail": (
            f"published durable tier remains after publication failure; cleanup deferred because no conditional "
            f"inode removal is available: {audit_db}"
        ),
        "state": "uncertain",
        "target": str(audit_db),
    }
    assert audit_db.exists()


def test_migrate_tier_cli_serializes_target_absent_cleanup(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vanished publication target is reported distinctly from cleanup uncertainty."""
    _stage_uninitialized_archive(cli_workspace)
    root = cli_workspace["archive_root"]
    audit_db = root / "audit.db"
    real_stat = os.stat
    published_stat_calls = 0

    def remove_target_before_cleanup_stat(
        file: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal published_stat_calls
        if file == "audit.db" and dir_fd is not None:
            published_stat_calls += 1
            if published_stat_calls == 3:
                audit_db.unlink()
        return real_stat(file, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    real_fsync = os.fsync
    directory_fsyncs = 0

    def fail_after_publish(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 1:
                raise OSError("publication fsync fault")
        real_fsync(descriptor)

    monkeypatch.setattr("polylogue.operations.durable_change_train.os.stat", remove_target_before_cleanup_stat)
    monkeypatch.setattr("polylogue.operations.durable_change_train.os.fsync", fail_after_publish)
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["durable_recovery"] == {
        "code": None,
        "detail": None,
        "state": "target_absent",
        "target": str(audit_db),
    }


def test_migrate_tier_cli_preserves_replacement_during_checked_leaf_cleanup(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cleanup mutation swaps in a foreign leaf before the checked unlink."""
    _stage_uninitialized_archive(cli_workspace)
    root = cli_workspace["archive_root"]
    audit_db = root / "audit.db"
    foreign = root / "foreign-audit.db"
    real_stat = os.stat
    published_stat_calls = 0

    def replace_target_before_checked_unlink(
        file: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal published_stat_calls
        if file == "audit.db" and dir_fd is not None:
            published_stat_calls += 1
            if published_stat_calls == 3:
                audit_db.unlink()
                foreign.write_bytes(b"foreign target")
                foreign.rename(audit_db)
        return real_stat(file, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    real_fsync = os.fsync
    directory_fsyncs = 0

    def fail_after_publish(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 1:
                raise OSError("publication fsync fault")
        real_fsync(descriptor)

    monkeypatch.setattr("polylogue.operations.durable_change_train.os.stat", replace_target_before_checked_unlink)
    monkeypatch.setattr("polylogue.operations.durable_change_train.os.fsync", fail_after_publish)
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["durable_recovery"]["code"] == "leaf_replaced"
    assert audit_db.read_bytes() == b"foreign target"


def test_migrate_tier_cli_preserves_replacement_after_cleanup_final_check(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cleanup rename must not unlink a leaf swapped after its final check."""
    _stage_uninitialized_archive(cli_workspace)
    root = cli_workspace["archive_root"]
    audit_db = root / "audit.db"
    rename_calls = 0

    def reject_unsafe_cleanup_rename(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal rename_calls
        rename_calls += 1
        raise AssertionError(f"unsafe cleanup rename attempted: {source} -> {destination}")

    real_fsync = os.fsync
    directory_fsyncs = 0

    def fail_after_publish(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 1:
                raise OSError("publication fsync fault")
        real_fsync(descriptor)

    monkeypatch.setattr("polylogue.operations.durable_change_train.os.rename", reject_unsafe_cleanup_rename)
    monkeypatch.setattr("polylogue.operations.durable_change_train.os.fsync", fail_after_publish)
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["durable_recovery"]["code"] == "cleanup_not_atomic"
    assert audit_db.exists()
    assert rename_calls == 0


def test_migrate_tier_cli_serializes_cleanup_inspection_uncertainty(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup inspection failure remains typed while the publication error survives."""
    _stage_uninitialized_archive(cli_workspace)
    root = cli_workspace["archive_root"]
    audit_db = root / "audit.db"
    real_stat = os.stat
    target_stat_calls = 0

    def fail_cleanup_inspection(
        file: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal target_stat_calls
        if file == "audit.db" and dir_fd is not None:
            target_stat_calls += 1
            if target_stat_calls == 3:
                raise OSError("cleanup inspection fault")
        return real_stat(file, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    real_fsync = os.fsync
    directory_fsyncs = 0

    def fail_after_publish(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 1:
                raise OSError("publication fsync fault")
        real_fsync(descriptor)

    monkeypatch.setattr("polylogue.operations.durable_change_train.os.stat", fail_cleanup_inspection)
    monkeypatch.setattr("polylogue.operations.durable_change_train.os.fsync", fail_after_publish)
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "cannot publish audit tier" in payload["error"]
    assert payload["durable_recovery"]["code"] == "leaf_inspection_failed"
    assert "could not inspect published durable tier" in payload["durable_recovery"]["detail"]
    assert audit_db.exists()


def test_migrate_tier_cli_fails_closed_when_anonymous_publication_is_unavailable(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage_uninitialized_archive(cli_workspace)
    audit_db = cli_workspace["archive_root"] / "audit.db"
    monkeypatch.setattr(os, "O_TMPFILE", 0, raising=False)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "filesystem does not support O_TMPFILE" in json.loads(result.stdout)["error"]
    assert not audit_db.exists()
    assert not list(audit_db.parent.glob(".audit.db.initialize-*.tmp"))


@pytest.mark.parametrize(
    ("missing_name", "sibling_name"),
    [("source.db", "user.db"), ("user.db", "source.db"), ("audit.db", "source.db")],
)
def test_migrate_tier_cli_refuses_to_initialize_a_tier_in_an_established_archive(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    missing_name: str,
    sibling_name: str,
) -> None:
    root = cli_workspace["archive_root"]
    missing = root / missing_name
    missing.unlink()
    before = missing.exists()

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            missing.stem,
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "established archive" in json.loads(result.stdout)["error"]
    assert missing.exists() is before
    assert (root / sibling_name).exists()


@pytest.mark.parametrize("missing_name", ["source.db", "user.db"])
@pytest.mark.parametrize(
    "retained_evidence",
    [
        "index.db",
        ".index-generations",
        ".index-rebuild-transactions",
        "source-continuity-pending",
        "source-continuity-refreshes",
    ],
)
def test_migrate_tier_cli_missing_initialization_refuses_retained_archive_evidence(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, missing_name: str, retained_evidence: str
) -> None:
    _stage_uninitialized_archive(cli_workspace)
    root = cli_workspace["archive_root"]
    missing_path = root / missing_name
    retained_path = root / retained_evidence
    if retained_evidence == "index.db":
        retained_path.touch()
    elif retained_evidence in {"source-continuity-pending", "source-continuity-refreshes"}:
        retained_path = root / ".maintenance-state" / retained_evidence
        retained_path.mkdir(parents=True)
        (retained_path / "intent.json").write_text("{}", encoding="utf-8")
    else:
        retained_path.mkdir(parents=True)
        (retained_path / "retained.json").write_text("{}", encoding="utf-8")

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            missing_path.stem,
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "established archive" in json.loads(result.stdout)["error"]
    assert retained_path.exists()
    assert not missing_path.exists()


@pytest.mark.parametrize("blob_state", ["nonempty-directory", "regular-file"])
def test_migrate_tier_cli_missing_initialization_refuses_retained_blob_evidence(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, blob_state: str
) -> None:
    _stage_uninitialized_archive(cli_workspace)
    blob_root = cli_workspace["archive_root"] / "blob"
    blob_root.mkdir()
    if blob_state == "nonempty-directory":
        (blob_root / "retained-entry").write_bytes(b"retained")
    else:
        blob_root.rmdir()
        blob_root.write_bytes(b"malformed blob store")

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "established archive" in json.loads(result.stdout)["error"]
    assert not (cli_workspace["archive_root"] / "audit.db").exists()


@pytest.mark.parametrize("marker_name", [".bootstrap", ".bootstrap.pending"])
def test_migrate_tier_cli_missing_initialization_refuses_bootstrap_markers(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, marker_name: str
) -> None:
    _stage_uninitialized_archive(cli_workspace)
    marker_root = cli_workspace["archive_root"] / ".maintenance-state" / "durable-change-trains"
    marker_root.mkdir(parents=True)
    (marker_root / marker_name).write_text("marker", encoding="utf-8")

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    error = json.loads(result.stdout)["error"]
    assert "established archive" in error
    assert str(marker_root / marker_name) in error
    assert not (cli_workspace["archive_root"] / "audit.db").exists()


def test_migrate_tier_cli_missing_initialization_refuses_blob_inspection_failure(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage_uninitialized_archive(cli_workspace)
    root = cli_workspace["archive_root"]
    blob_root = root / "blob"
    blob_root.mkdir()
    blob_identity = blob_root.stat()
    real_listdir = os.listdir

    def fail_blob_inspection(candidate: int | os.PathLike[str] | str) -> list[str] | Iterator[Path]:
        if isinstance(candidate, int):
            candidate_metadata = os.fstat(candidate)
            if (candidate_metadata.st_dev, candidate_metadata.st_ino) == (blob_identity.st_dev, blob_identity.st_ino):
                raise OSError("blob inspection failed")
        return real_listdir(candidate)

    monkeypatch.setattr("polylogue.operations.durable_change_train.os.listdir", fail_blob_inspection)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "cannot inspect retained blob path" in json.loads(result.stdout)["error"]
    assert not (root / "audit.db").exists()


def test_migrate_tier_cli_missing_initialization_refuses_marker_inspection_failure(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage_uninitialized_archive(cli_workspace)
    root = cli_workspace["archive_root"]
    marker_root = root / ".maintenance-state" / "durable-change-trains"
    marker_root.mkdir(parents=True)
    real_stat = os.stat

    def fail_marker_inspection(
        candidate: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if candidate == ".maintenance-state/durable-change-trains" and dir_fd is not None:
            raise OSError("marker inspection failed")
        return real_stat(candidate, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr("polylogue.operations.durable_change_train.os.stat", fail_marker_inspection)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "cannot inspect durable change-train adoption marker" in json.loads(result.stdout)["error"]
    assert not (root / "audit.db").exists()


def test_migrate_tier_cli_missing_initialization_refuses_dangling_active_pointer(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    _stage_uninitialized_archive(cli_workspace)
    root = cli_workspace["archive_root"]
    pointer = root / ".index-active-pointer"
    pointer.symlink_to(root / ".index-generations" / "missing" / "index.db")

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    error = json.loads(result.stdout)["error"]
    assert "established archive" in error
    assert str(pointer) in error
    assert not (root / "audit.db").exists()


def test_migrate_tier_cli_missing_initialization_refuses_malformed_train_marker(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    _stage_uninitialized_archive(cli_workspace)
    root = cli_workspace["archive_root"]
    marker = root / ".maintenance-state" / "durable-change-trains"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("not a marker directory", encoding="utf-8")

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "migrate-tier",
            "audit",
            "--initialize-missing",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    error = json.loads(result.stdout)["error"]
    assert "established archive" in error
    assert str(marker) in error
    assert not (root / "audit.db").exists()


def test_rebuild_index_empty_source_still_runs_the_schema_currency_guard(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    root = cli_workspace["archive_root"]
    with sqlite3.connect(root / "audit.db") as conn:
        expected = int(conn.execute("PRAGMA user_version").fetchone()[0])
        conn.execute(f"PRAGMA user_version = {expected + 1}")

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "rebuild-index", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "audit.db" in result.stderr
    assert not (root / ".index-generations").exists()


def test_rebuild_index_empty_source_preserves_plain_receipt_output_without_schema_receipt(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    """The real empty receipt must render without replay-only counter keys.

    Mutation: removing the status branch reaches the production counter
    formatter and raises KeyError before this exact plain output is emitted.
    """
    root = cli_workspace["archive_root"]

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "rebuild-index"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert result.stdout == f"Archive root: {root}\nNo source.db raw_sessions rows found.\n"
    assert not (root / ".index-generations").exists()


def test_rebuild_index_rejects_daemon_schema_preflight_combination(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "rebuild-index", "--preflight", "--daemon"],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    assert "--preflight cannot be combined with --daemon" in result.output


def test_rebuild_index_daemon_path_posts_the_real_selection_request(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    receipt_path = write_valid_rebuild_receipt(
        cli_workspace["archive_root"], cli_workspace["archive_root"].parent / "schema-inference-gate-receipt.json"
    )

    class Response:
        def read(self) -> bytes:
            return json.dumps(
                {
                    "archive_root": str(cli_workspace["archive_root"]),
                    "classified_full_count": 2,
                    "replayed_logical_source_count": 1,
                    "quarantined_raw_count": 0,
                }
            ).encode()

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_urlopen(request: object, *, timeout: int) -> Response:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["body"] = json.loads(request.data)  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(maintenance_rebuild_index, "urlopen", fake_urlopen)
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "rebuild-index",
            "--daemon",
            "--daemon-url",
            "http://127.0.0.1:9876",
            "--schema-inference-receipt",
            str(receipt_path),
            "--raw-batch-size",
            "17",
            "--pass-byte-budget-mb",
            "12.5",
            "--pass-deadline-seconds",
            "45",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert captured == {
        "url": "http://127.0.0.1:9876/api/maintenance/rebuild-index",
        "body": {
            "only_missing": False,
            "raw_ids": [],
            "max_blob_mb": None,
            "promote": True,
            "operation_id": None,
            "schema_inference_receipt_path": str(receipt_path),
            "raw_batch_size": 17,
            "pass_byte_budget_mb": 12.5,
            "pass_deadline_seconds": 45.0,
        },
        "timeout": 600,
    }
    assert "Classified:" in result.output


def test_rebuild_index_daemon_resolves_relative_schema_receipt_before_post(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Relative receipt references are resolved before daemon serialization.

    Anti-vacuity: removing CLI-side resolution sends the relative filename in
    the captured production daemon payload.
    """
    root = cli_workspace["archive_root"]
    absolute_receipt = write_valid_rebuild_receipt(root, tmp_path / "relative-receipt.json")
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    class Response:
        def read(self) -> bytes:
            return json.dumps(
                {
                    "archive_root": str(root),
                    "classified_full_count": 0,
                    "replayed_logical_source_count": 0,
                    "quarantined_raw_count": 0,
                }
            ).encode()

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_urlopen(request: object, *, timeout: int) -> Response:
        captured["body"] = json.loads(request.data)  # type: ignore[attr-defined]
        return Response()

    monkeypatch.setattr(maintenance_rebuild_index, "urlopen", fake_urlopen)
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "rebuild-index",
            "--daemon",
            "--schema-inference-receipt",
            absolute_receipt.name,
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["schema_inference_receipt_path"] == str(absolute_receipt.resolve())


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


def test_rebuild_index_full_source_resumes_one_candidate_until_terminal_promotion(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bounded pass retains its generation; only the terminal resume promotes it."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = cli_workspace["archive_root"]
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for native_id, acquired_at_ms in (("first", 1), ("second", 2)):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=(
                    f'{{"type":"session_meta","payload":{{"id":"{native_id}"}}}}\n'
                    f'{{"type":"response_item","payload":{{"type":"message","role":"user",'
                    f'"content":[{{"type":"input_text","text":"{native_id}"}}]}}}}\n'
                ).encode(),
                source_path=f"{native_id}.jsonl",
                acquired_at_ms=acquired_at_ms,
            )
    with sqlite3.connect(root / "source.db") as source:
        source.execute(
            """
            UPDATE raw_sessions
            SET logical_source_key = CASE
                    WHEN source_path = 'first.jsonl' THEN 'codex:first'
                    ELSE 'codex:second'
                END,
                revision_kind = 'full',
                source_revision = raw_id,
                baseline_raw_id = raw_id,
                acquisition_generation = 0,
                revision_authority = 'byte_proven'
            """
        )
        source.commit()
    _freeze_rebuild_fixture_source(root, expected_raws=2)
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "schema-inference-gate-receipt.json")
    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(receipt_path))

    first = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "rebuild-index", "--raw-batch-size", "1", "--output-format", "json"],
        catch_exceptions=False,
    )
    assert first.exit_code == 0, first.output
    # This pass now also replays a raw page through the shared
    # revision-backfill machinery, which logs "backfill stage timings" to
    # stderr on every call (see the sibling terminal-promotion test for the
    # full rationale); `.stdout` is the actual `--output-format json`
    # contract surface, `.output` is Click 8.4's always-mixed stream.
    first_payload = json.loads(first.stdout)
    operation_id = first_payload["transaction"]["operation_id"]
    generation_path = Path(first_payload["generation"]["index_path"])
    assert first_payload["status"] == "paused"
    assert first_payload["transaction"]["processed_raw_count"] == 1
    assert generation_path.exists()
    assert root.joinpath("index.db").resolve() != generation_path.resolve()

    terminal = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "rebuild-index",
            "--operation-id",
            operation_id,
            "--raw-batch-size",
            "1",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert terminal.exit_code == 0
    # Promotion logs `rebuild_terminal_stage_complete` structlog events to
    # stderr per invocation (one per terminal stage) -- correct per the
    # stdout=results/stderr=diagnostics channel-separation contract
    # (test_stdout_stderr_split.py), but `.output` is Click 8.4's always-
    # mixed stream (mix_stderr was removed; `.output` "mixes stdout and
    # stderr, in the order they were written"). `.stdout` is the actual
    # `--output-format json` contract surface.
    terminal_payload = json.loads(terminal.stdout)
    assert terminal_payload["status"] == "replayed"
    assert terminal_payload["transaction"]["status"] == "promoted"
    assert root.joinpath("index.db").resolve() == generation_path.resolve()
    with sqlite3.connect(root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (2,)


def test_rebuild_index_persists_durable_pass_receipt_alongside_transaction(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every rebuild pass receipt survives on disk, not only on the CLI's stdout.

    Reproduces the fix for a live incident (polylogue-k8kj): two rebuild page
    receipts were lost because the CLI writes the receipt JSON only to
    stdout, and the invoking shell's pipe died while an orphaned rebuild
    process kept working. Each pass must also be durably persisted under the
    transaction directory so a lost pipe never means a lost receipt.
    """
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = cli_workspace["archive_root"]
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for native_id, acquired_at_ms in (("first", 1), ("second", 2)):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=(
                    f'{{"type":"session_meta","payload":{{"id":"{native_id}"}}}}\n'
                    f'{{"type":"response_item","payload":{{"type":"message","role":"user",'
                    f'"content":[{{"type":"input_text","text":"{native_id}"}}]}}}}\n'
                ).encode(),
                source_path=f"{native_id}.jsonl",
                acquired_at_ms=acquired_at_ms,
            )
    with sqlite3.connect(root / "source.db") as source:
        source.execute(
            """
            UPDATE raw_sessions
            SET logical_source_key = CASE
                    WHEN source_path = 'first.jsonl' THEN 'codex:first'
                    ELSE 'codex:second'
                END,
                revision_kind = 'full',
                source_revision = raw_id,
                baseline_raw_id = raw_id,
                acquisition_generation = 0,
                revision_authority = 'byte_proven'
            """
        )
        source.commit()
    _freeze_rebuild_fixture_source(root, expected_raws=2)
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "schema-inference-gate-receipt.json")
    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(receipt_path))

    first = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "rebuild-index", "--raw-batch-size", "1", "--output-format", "json"],
        catch_exceptions=False,
    )
    assert first.exit_code == 0, first.output
    # This pass now also replays a raw page through the shared
    # revision-backfill machinery, which logs "backfill stage timings" to
    # stderr on every call (see the sibling terminal-promotion test for the
    # full rationale); `.stdout` is the actual `--output-format json`
    # contract surface, `.output` is Click 8.4's always-mixed stream.
    first_payload = json.loads(first.stdout)
    operation_id = first_payload["transaction"]["operation_id"]
    assert first_payload["status"] == "paused"

    receipts_dir = root / ".index-rebuild-transactions" / f"{operation_id}.receipts"
    receipt_files = sorted(receipts_dir.glob("pass-*.json"))
    assert len(receipt_files) == 1
    persisted = json.loads(receipt_files[0].read_text(encoding="utf-8"))
    assert persisted["status"] == "paused"
    assert persisted["transaction"]["operation_id"] == operation_id

    terminal = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "rebuild-index",
            "--operation-id",
            operation_id,
            "--raw-batch-size",
            "1",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert terminal.exit_code == 0
    # Promotion logs to stderr (see the sibling terminal-promotion test for
    # the full rationale); `.stdout` is the actual `--output-format json`
    # contract surface, `.output` is Click 8.4's always-mixed stream.
    terminal_payload = json.loads(terminal.stdout)
    assert terminal_payload["status"] == "replayed"

    receipt_files = sorted(receipts_dir.glob("pass-*.json"))
    assert len(receipt_files) == 2
    final_persisted = json.loads(receipt_files[-1].read_text(encoding="utf-8"))
    assert final_persisted["status"] == "replayed"


def test_rebuild_index_byte_budget_defers_then_reaches_terminal_ready_candidate(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real CLI replays every raw over passes; byte budgeting never filters archive data."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = cli_workspace["archive_root"]
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for native_id, acquired_at_ms, padding in (("large", 1, "x" * 1_024), ("later", 2, "y")):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=(
                    f'{{"type":"session_meta","payload":{{"id":"{native_id}"}}}}\n'
                    f'{{"type":"response_item","payload":{{"type":"message","role":"user",'
                    f'"content":[{{"type":"input_text","text":"{padding}"}}]}}}}\n'
                ).encode(),
                source_path=f"{native_id}.jsonl",
                acquired_at_ms=acquired_at_ms,
            )
    _freeze_rebuild_fixture_source(root, expected_raws=2)
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "schema-inference-gate-receipt.json")
    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(receipt_path))
    first = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "rebuild-index",
            "--pass-byte-budget-mb",
            "0.0001",
            "--no-promote",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert first.exit_code == 0, first.output
    # This pass now also replays a raw page through the shared
    # revision-backfill machinery, which logs "backfill stage timings" to
    # stderr on every call (see the sibling terminal-promotion test for the
    # full rationale); `.stdout` is the actual `--output-format json`
    # contract surface, `.output` is Click 8.4's always-mixed stream.
    first_payload = json.loads(first.stdout)
    assert first_payload["status"] == "deferred"
    operation_id = first_payload["transaction"]["operation_id"]
    terminal = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "rebuild-index",
            "--operation-id",
            operation_id,
            "--no-promote",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert terminal.exit_code == 0
    # Terminal-stage-complete events (session_insights/bulk_build.*/fts_parity/
    # readiness) log to stderr even with --no-promote; `.stdout` is the
    # actual `--output-format json` contract surface, `.output` is Click
    # 8.4's always-mixed stream.
    payload = json.loads(terminal.stdout)
    assert payload["transaction"]["status"] == "ready"
    assert payload["generation"]["state"] == "inactive"
    with sqlite3.connect(Path(payload["generation"]["index_path"])) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (2,)


def test_rebuild_index_source_snapshot_drift_fails_before_candidate_creation(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt/source mismatch stops the route before candidate creation."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = cli_workspace["archive_root"]
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=(
                b'{"type":"session_meta","payload":{"id":"drift"}}\n'
                b'{"type":"response_item","payload":{"type":"message","role":"user",'
                b'"content":[{"type":"input_text","text":"drift"}]}}\n'
            ),
            source_path="drift.jsonl",
            acquired_at_ms=1,
        )
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "schema-inference-gate-receipt.json")
    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(receipt_path))
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"drift-after-receipt"}}\n',
            source_path="drift-after-receipt.jsonl",
            acquired_at_ms=2,
        )
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "rebuild-index", "--output-format", "json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    assert "schema-inference preflight gate failed" in result.output
    assert not list((root / ".index-rebuild-transactions").glob("*.json"))
    assert not list((root / ".index-generations").glob("gen-*"))
    assert not root.joinpath("index.db").is_symlink()


def test_rebuild_index_deadline_defers_postflight_until_resume(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deadline expiry preserves the replayed candidate instead of permitting early promotion."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = cli_workspace["archive_root"]
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=(
                b'{"type":"session_meta","payload":{"id":"deadline"}}\n'
                b'{"type":"response_item","payload":{"type":"message","role":"user",'
                b'"content":[{"type":"input_text","text":"deadline"}]}}\n'
            ),
            source_path="deadline.jsonl",
            acquired_at_ms=1,
        )
    _freeze_rebuild_fixture_source(root, expected_raws=1)
    receipt_path = write_valid_rebuild_receipt(root, root.parent / "schema-inference-gate-receipt.json")
    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(receipt_path))
    # `monkeypatch.setattr("polylogue.maintenance.rebuild_index.time.time", ...)`
    # patches the *stdlib* `time` module's `time` attribute (modules are
    # process-wide singletons), not a private copy scoped to rebuild_index.py.
    # Use a monotonically-advancing fake clock rather than pinning an exact
    # call count before/between the deadline reads this test exercises.
    #
    # polylogue-uhgm: the deadline is now ALSO checked between replay
    # cohorts (not only once, post-hoc, after the whole page replayed), so
    # an ever-advancing clock left active across BOTH invocations would trip
    # the resumed pass's very first between-cohorts check too and never let
    # it promote. Scope the fake clock to just the first (interrupted)
    # invocation with `monkeypatch.context()`; the resumed invocation runs
    # on the real clock against the transaction's durable 1s budget, which
    # is ample for this single-raw fixture.
    fake_clock = itertools.count(100.0, 50.0)
    with pytest.MonkeyPatch.context() as clock_patch:
        clock_patch.setattr(
            "polylogue.maintenance.rebuild_index.time.time",
            lambda: next(fake_clock),
        )
        first = cli_runner.invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "rebuild-index",
                "--pass-deadline-seconds",
                "1",
                "--output-format",
                "json",
            ],
            catch_exceptions=False,
        )
    assert first.exit_code == 0, first.output
    # This pass replays a raw page through the shared revision-backfill
    # machinery, which logs "backfill stage timings" to stderr on every
    # call; `.stdout` is the actual `--output-format json` contract
    # surface, `.output` is Click 8.4's always-mixed stream.
    payload = json.loads(first.stdout)
    assert payload["status"] == "deferred"
    assert payload["transaction"]["status"] == "deferred"
    # polylogue-uhgm: the deadline fired before the sole raw in this page
    # was ever replayed (not merely observed too late afterward), so this
    # pass recorded zero forward progress -- the whole point of the fix.
    assert payload["transaction"]["processed_raw_count"] == 0
    assert payload["transaction"]["last_raw_id"] is None
    assert not root.joinpath("index.db").is_symlink()
    resumed = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "rebuild-index",
            "--operation-id",
            payload["transaction"]["operation_id"],
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert resumed.exit_code == 0
    # This resume promotes, logging terminal-stage-complete events to stderr;
    # `.stdout` is the actual `--output-format json` contract surface,
    # `.output` is Click 8.4's always-mixed stream.
    assert json.loads(resumed.stdout)["transaction"]["status"] == "promoted"


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

    # rebuild_index_from_source now also reports parse_s/apply_s/stage_timings_s
    # (the bulk-build pragma-profile timing instrumentation for owned inactive
    # generations). Those are wall-clock floats, not part of this helper's
    # typed-empty-receipt contract, so they are checked for shape/presence
    # separately rather than folded into the exact-count comparison below.
    timing_keys = {"parse_s", "apply_s", "stage_timings_s"}
    assert timing_keys.issubset(result)
    assert isinstance(result["parse_s"], float)
    assert isinstance(result["apply_s"], float)
    assert isinstance(result["stage_timings_s"], dict)
    assert {key: value for key, value in result.items() if key not in timing_keys} == {
        "scanned_raw_count": 0,
        "classified_full_count": 0,
        "replayed_logical_source_count": 0,
        "quarantined_raw_count": 0,
        "adoption_deferred_raw_count": 0,
        "authority_selection_expanded": True,
        "scheduled_raw_count": 2,
        "raw_batch_size": 7,
        "ingest_workers": 1,
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
    payload = json.loads(result.stdout)
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
    payload = json.loads(result.stdout)
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
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["raw_session_count"] == 3
    assert payload["selected_raw_count"] == 3
    assert payload["replay_order"] == "blob_hash_asc_raw_id_asc"
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
