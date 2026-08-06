"""Real-route tests for cursor-authority reconciliation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.maintenance import cursor_authority_reconcile as reconcile
from polylogue.sources.live.batch import CursorAuthorityBlockedError, scoped_cursor_authority_authorization
from polylogue.sources.live.metrics import LiveBatchMetrics


def _private_path_file(path: Path, source: Path) -> None:
    path.write_text(f"{source}\n", encoding="utf-8")
    path.chmod(0o600)


def test_dry_run_plan_is_deterministic_and_does_not_store_private_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.sources.test_live_watcher import _live_archive_snapshot, _seed_live_cursor_authority_case

    _processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    monkeypatch.setattr(reconcile, "ARCHIVE_ROOT", tmp_path)
    path_file = tmp_path / "selected-path"
    _private_path_file(path_file, source_path)
    before = _live_archive_snapshot(tmp_path)

    first = reconcile.build_reconciliation_plan(source_path_file=path_file, output_plan=tmp_path / "plan-1.json")
    second = reconcile.build_reconciliation_plan(source_path_file=path_file, output_plan=tmp_path / "plan-2.json")

    assert first | {"observed_at_ms": None, "plan_digest": None} == second | {
        "observed_at_ms": None,
        "plan_digest": None,
    }
    assert isinstance(first["observed_at_ms"], int)
    assert first["format"] == reconcile.PLAN_FORMAT
    assert str(source_path) not in json.dumps(first, sort_keys=True)
    assert _live_archive_snapshot(tmp_path) == before
    watcher.stop()


@pytest.mark.asyncio
async def test_apply_authorization_invokes_normal_full_ingest_route(
    tmp_path: Path,
) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path, force_full_fallback=True)
    projection = reconcile._projection_for(tmp_path)
    sample = projection.cursor_ahead_samples[0]
    processor._append_plan = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("append planning called"))  # type: ignore[method-assign]
    with scoped_cursor_authority_authorization(
        source_path_digest=reconcile.cursor_authority_path_digest(source_path),
        cursor_byte_offset=sample.cursor_byte_offset,
        accepted_frontier=sample.accepted_frontier,
        plan_digest="test-plan",
        force_full_ingest=True,
    ):
        metrics = await processor.ingest_files([source_path], emit_event=False)

    assert metrics.full_file_count == 1
    assert metrics.succeeded_file_count == 1
    watcher.stop()


@pytest.mark.asyncio
async def test_scoped_authorization_rejects_a_different_path_without_mutation(tmp_path: Path) -> None:
    from tests.unit.sources.test_live_watcher import _live_archive_snapshot, _seed_live_cursor_authority_case

    processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    other_path = source_path.with_name("other.jsonl")
    other_path.write_bytes(source_path.read_bytes())
    projection = reconcile._projection_for(tmp_path)
    sample = projection.cursor_ahead_samples[0]
    before = _live_archive_snapshot(tmp_path)

    with scoped_cursor_authority_authorization(
        source_path_digest=reconcile.cursor_authority_path_digest(source_path),
        cursor_byte_offset=sample.cursor_byte_offset,
        accepted_frontier=sample.accepted_frontier,
        plan_digest="test-plan",
    ):
        with pytest.raises(CursorAuthorityBlockedError, match="selected path"):
            await processor.ingest_files([other_path], emit_event=False)

    assert _live_archive_snapshot(tmp_path) == before
    watcher.stop()


def test_plan_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    _processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    monkeypatch.setattr(reconcile, "ARCHIVE_ROOT", tmp_path)
    path_file = tmp_path / "selected-path"
    _private_path_file(path_file, source_path)
    output = tmp_path / "plan.json"
    reconcile.build_reconciliation_plan(source_path_file=path_file, output_plan=output)

    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="already exists"):
        reconcile.build_reconciliation_plan(source_path_file=path_file, output_plan=output)
    watcher.stop()


def test_private_path_file_requires_exact_permissions_and_absolute_path(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    path_file = tmp_path / "selected-path"
    path_file.write_text(str(source) + "\n", encoding="utf-8")
    path_file.chmod(0o644)

    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="mode 0600"):
        reconcile._read_private_source_path(path_file)

    path_file.chmod(0o600)
    path_file.write_text("relative.jsonl\n", encoding="utf-8")
    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="absolute"):
        reconcile._read_private_source_path(path_file)


def test_plan_digest_tampering_is_rejected(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"format": reconcile.PLAN_FORMAT, "status": "planned", "plan_digest": "wrong"}),
        encoding="utf-8",
    )

    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="digest mismatch"):
        reconcile._load_plan(plan_path)


@pytest.mark.asyncio
async def test_scoped_authorization_is_single_use(tmp_path: Path) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    projection = reconcile._projection_for(tmp_path)
    sample = projection.cursor_ahead_samples[0]
    authorization = scoped_cursor_authority_authorization(
        source_path_digest=reconcile.cursor_authority_path_digest(source_path),
        cursor_byte_offset=sample.cursor_byte_offset,
        accepted_frontier=sample.accepted_frontier,
        plan_digest="test-plan",
    )
    with authorization:
        processor.require_cursor_authority([source_path])
        with pytest.raises(CursorAuthorityBlockedError, match="already consumed"):
            processor.require_cursor_authority([source_path])
    watcher.stop()


def test_planner_preserves_incomparable_population(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    _processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    projection = replace(reconcile._projection_for(tmp_path), cursor_authority_gap_count=727)
    monkeypatch.setattr(reconcile, "_projection_for", lambda root: projection)
    plan = reconcile._build_plan(tmp_path, source_path)
    before_projection = plan["before_projection"]
    assert isinstance(before_projection, dict)
    assert before_projection["cursor_authority_gap_count"] == 727
    watcher.stop()


def test_planner_refuses_multiple_true_ahead_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    _processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    projection = replace(reconcile._projection_for(tmp_path), cursor_ahead_count=2)
    monkeypatch.setattr(reconcile, "_projection_for", lambda root: projection)

    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="multiple"):
        reconcile._build_plan(tmp_path, source_path)
    watcher.stop()


def test_planner_refuses_unavailable_healthy_sibling_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    _processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    projection = replace(reconcile._projection_for(tmp_path), missing_source_raw_status="unknown")
    monkeypatch.setattr(reconcile, "_projection_for", lambda root: projection)

    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="sibling"):
        reconcile._build_plan(tmp_path, source_path)
    watcher.stop()


def test_wal_effective_snapshot_matches_sqlite_backup(tmp_path: Path) -> None:
    live = tmp_path / "live.db"
    backup = tmp_path / "backup.db"
    with sqlite3.connect(live) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE values_table (value TEXT)")
        conn.execute("INSERT INTO values_table VALUES ('wal-frame')")
        conn.commit()
        live_snapshot = reconcile._sqlite_snapshot(live)
        with sqlite3.connect(backup) as backup_conn:
            conn.backup(backup_conn)
            backup_conn.commit()
    assert live_snapshot == reconcile._sqlite_snapshot(backup)


def test_planner_rejects_source_mutation_during_prefix_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    _processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    observations = iter(((1, 2, 3, 4, 5), (1, 2, 3, 4, 6)))
    monkeypatch.setattr(reconcile, "_stat_observation", lambda path: next(observations))

    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="mutated"):
        reconcile._build_plan(tmp_path, source_path)
    watcher.stop()


@pytest.mark.asyncio
async def test_scoped_authorization_rejects_changed_cursor_frontier(tmp_path: Path) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    processor, watcher, cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    projection = reconcile._projection_for(tmp_path)
    sample = projection.cursor_ahead_samples[0]
    with scoped_cursor_authority_authorization(
        source_path_digest=reconcile.cursor_authority_path_digest(source_path),
        cursor_byte_offset=sample.cursor_byte_offset,
        accepted_frontier=sample.accepted_frontier,
        plan_digest="test-plan",
    ):
        with sqlite3.connect(cursor._db_path) as conn:
            conn.execute(
                "UPDATE ingest_cursor SET byte_offset = byte_offset + 1 WHERE source_path = ?",
                (str(source_path),),
            )
        with pytest.raises(CursorAuthorityBlockedError, match="frontier binding"):
            processor.require_cursor_authority([source_path])
    watcher.stop()


@pytest.mark.asyncio
async def test_scoped_authorization_cannot_turn_a_healthy_archive_into_an_exception(
    tmp_path: Path,
) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path, exact_frontier=True)
    with scoped_cursor_authority_authorization(
        source_path_digest=reconcile.cursor_authority_path_digest(source_path),
        cursor_byte_offset=0,
        accepted_frontier=0,
        plan_digest="test-plan",
    ):
        with pytest.raises(CursorAuthorityBlockedError, match="no planned violation"):
            processor.require_cursor_authority([source_path])
    watcher.stop()


def test_backup_validation_requires_blob_rollback_evidence(tmp_path: Path) -> None:
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "manifest.json").write_text(
        json.dumps(
            {
                "profile": "full_evidence",
                "included_tiers": ["source.db", "index.db", "ops.db", "audit.db"],
            }
        ),
        encoding="utf-8",
    )
    (backup / "verification-receipt.json").write_text(json.dumps({"verdict": "success"}), encoding="utf-8")

    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="blob rollback"):
        reconcile._validate_backup(backup, {})


def test_private_projection_redacts_paths_and_preserves_missing_sample_branches(tmp_path: Path) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    _processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    projection = reconcile._projection_for(tmp_path)
    private = reconcile._private_projection(projection)
    samples = private["cursor_ahead_samples"]
    assert isinstance(samples, list)
    sample = samples[0]
    assert isinstance(sample, dict)
    original = projection.cursor_ahead_samples[0]
    assert sample["source_path"] == reconcile.cursor_authority_path_digest(source_path)
    assert sample["logical_source_key"] == reconcile._identity_digest(original.logical_source_key)
    watcher.stop()


def test_recovery_attempt_requires_a_later_completed_observation(tmp_path: Path) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    _processor, watcher, cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    with sqlite3.connect(cursor._db_path) as conn:
        conn.execute(
            "INSERT INTO ingest_attempts "
            "(attempt_id, status, started_at_ms, finished_at_ms, source_paths_json) "
            "VALUES (?, 'completed', ?, ?, ?)",
            ("attempt-old", 100, 150, json.dumps([str(source_path)])),
        )
        conn.execute(
            "INSERT INTO ingest_attempts "
            "(attempt_id, status, started_at_ms, finished_at_ms, source_paths_json) "
            "VALUES (?, 'completed', ?, ?, ?)",
            ("attempt-new", 200, 250, json.dumps([str(source_path)])),
        )
    plan_digest = "plan-digest"
    path_digest = reconcile.cursor_authority_path_digest(source_path)
    with sqlite3.connect(cursor._db_path) as conn:
        conn.execute(
            "INSERT INTO daemon_stage_events "
            "(event_id, attempt_id, stage, status, observed_at_ms, payload_json) VALUES (?, ?, 'planning', 'completed', ?, ?)",
            (
                "event-new-plan",
                "attempt-new",
                210,
                json.dumps(
                    {
                        "cursor_authority_plan_digest": plan_digest,
                        "cursor_authority_path_digest": path_digest,
                    }
                ),
            ),
        )
    assert (
        reconcile._find_recovery_attempt(
            tmp_path, source_path, 150, plan_digest="unrelated-plan", path_digest=path_digest
        )
        is None
    )
    assert (
        reconcile._find_recovery_attempt(tmp_path, source_path, 250, plan_digest=plan_digest, path_digest=path_digest)
        is None
    )
    observed = reconcile._find_recovery_attempt(
        tmp_path, source_path, 150, plan_digest=plan_digest, path_digest=path_digest
    )
    assert observed is not None
    assert observed["attempt_id"] == "attempt-new"
    watcher.stop()


def test_backup_validation_rehashes_and_rejects_mismatched_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "blob").mkdir()
    (backup / "blob-inventory.json").write_text("{}", encoding="utf-8")
    tiers: dict[str, dict[str, object]] = {}
    for tier in ("source", "index", "ops", "audit"):
        path = backup / f"{tier}.db"
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE marker (value TEXT)")
        tiers[tier] = reconcile._sqlite_snapshot(path)
    plan = {"tier_fingerprints": tiers}
    manifest = {
        "profile": "full_evidence",
        "included_tiers": [f"{tier}.db" for tier in tiers],
        "tier_source_fingerprints": {f"{tier}.db": value for tier, value in tiers.items()},
    }
    (backup / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (backup / "verification-receipt.json").write_text(
        json.dumps(
            {
                "verdict": "success",
                "verification": {
                    "source_blobs_resolved": True,
                    "index_attachment_blobs_resolved": True,
                    "blob_inventory_exact": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(reconcile, "ARCHIVE_ROOT", tmp_path)
    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="attestation"):
        reconcile._validate_backup(backup, plan)
    monkeypatch.setattr(reconcile, "verify_verification_receipt", lambda *args, **kwargs: None)
    assert reconcile._validate_backup(backup, plan)["root"]["basename"] == backup.name
    (backup / "audit.db").unlink()
    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="tier is missing"):
        reconcile._validate_backup(backup, plan)
    with sqlite3.connect(backup / "audit.db") as conn:
        conn.execute("CREATE TABLE marker (value TEXT)")
    with sqlite3.connect(backup / "source.db") as conn:
        conn.execute("INSERT INTO marker VALUES ('tampered')")
    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="image does not match"):
        reconcile._validate_backup(backup, plan)


def _deferred_metrics(source_path: Path) -> LiveBatchMetrics:
    return LiveBatchMetrics(
        queued_file_count=1,
        needed_file_count=1,
        skipped_file_count=0,
        succeeded_file_count=0,
        failed_file_count=1,
        source_group_count=1,
        input_bytes=1,
        source_payload_read_bytes=1,
        cursor_fingerprint_read_bytes=1,
        ingest_worker_count_max=1,
        append_file_count=0,
        full_file_count=1,
        archive_bytes_before=1,
        archive_bytes_after=1,
        archive_write_bytes_delta=0,
        parse_time_s=0.0,
        convergence_time_s=0.0,
        total_time_s=0.0,
        failed_paths=[str(source_path)],
    )


def test_typed_deferred_apply_receipt_is_metric_backed_and_does_not_claim_cursor_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    plan = reconcile._build_plan(tmp_path, source_path)
    projection = reconcile._projection_for(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(reconcile, "ARCHIVE_ROOT", tmp_path)
    monkeypatch.setattr(reconcile, "_require_daemon_stopped", lambda root: None)
    monkeypatch.setattr(reconcile, "_validate_backup", lambda manifest, plan: {"root": "backup"})
    monkeypatch.setattr(reconcile, "_find_path_by_digest", lambda root, digest: source_path)
    monkeypatch.setattr(reconcile, "_build_plan", lambda root, path: plan)

    async def deferred_ingest(
        root: Path, path: Path, plan_payload: dict[str, object]
    ) -> tuple[LiveBatchMetrics, dict[str, object]]:
        return _deferred_metrics(path), {
            "attempt_id": "attempt-deferred",
            "outcome_code": "transient_error",
            "retryable": True,
        }

    monkeypatch.setattr(reconcile, "_normal_ingest", deferred_ingest)
    monkeypatch.setattr(reconcile, "_projection_for", lambda root: projection)
    monkeypatch.setattr(reconcile, "_tier_snapshots", lambda root: {})
    monkeypatch.setattr(reconcile, "_quick_checks", lambda root: {})

    result = reconcile.apply_reconciliation(
        plan_path=plan_path,
        backup_manifest=tmp_path / "backup",
        receipt=receipt_path,
    )

    assert result["verdict"] == "typed_deferred"
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["failed_paths"] == [
        {"path_digest": reconcile.cursor_authority_path_digest(source_path), "basename": source_path.name}
    ]
    changed_rows = result["changed_rows"]
    assert isinstance(changed_rows, dict)
    assert changed_rows["cursor"] is None
    assert result["ingest_attempt_observation"] == "performed"
    watcher.stop()


def test_observed_recovery_receipt_does_not_claim_local_cursor_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    _processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    plan = reconcile._build_plan(tmp_path, source_path)
    projection = replace(
        reconcile._projection_for(tmp_path),
        overall_status="healthy",
        cursor_ahead_status="healthy",
        cursor_ahead_count=0,
        cursor_ahead_samples=(),
    )
    receipt_path = tmp_path / "receipt.json"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(reconcile, "ARCHIVE_ROOT", tmp_path)
    monkeypatch.setattr(reconcile, "_require_daemon_stopped", lambda root: None)
    monkeypatch.setattr(reconcile, "_validate_backup", lambda manifest, plan: {"root": "backup"})
    monkeypatch.setattr(reconcile, "_find_path_by_digest", lambda root, digest: source_path)
    monkeypatch.setattr(reconcile, "_build_plan", lambda root, path: None)
    monkeypatch.setattr(
        reconcile,
        "_find_recovery_attempt",
        lambda root, path, observed, **kwargs: {
            "attempt_id": "external-attempt",
            "outcome_code": "success",
            "retryable": False,
        },
    )
    monkeypatch.setattr(reconcile, "_projection_for", lambda root: projection)
    monkeypatch.setattr(reconcile, "_tier_snapshots", lambda root: {})
    monkeypatch.setattr(reconcile, "_quick_checks", lambda root: {})

    result = reconcile.apply_reconciliation(
        plan_path=plan_path,
        backup_manifest=tmp_path / "backup",
        receipt=receipt_path,
    )

    assert result["verdict"] == "reconciled"
    assert result["ingest_attempt_observation"] == "observed"
    changed_rows = result["changed_rows"]
    assert isinstance(changed_rows, dict)
    assert changed_rows["cursor"] is None
    assert result["metrics"] is None
    watcher.stop()


def test_unexpected_post_ingest_failure_writes_typed_audit_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    _processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    plan = reconcile._build_plan(tmp_path, source_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    projection = replace(reconcile._projection_for(tmp_path), broken_head_count=1)
    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setattr(reconcile, "ARCHIVE_ROOT", tmp_path)
    monkeypatch.setattr(reconcile, "_require_daemon_stopped", lambda root: None)
    monkeypatch.setattr(reconcile, "_validate_backup", lambda manifest, plan: {"root": "backup"})
    monkeypatch.setattr(reconcile, "_find_path_by_digest", lambda root, digest: source_path)
    monkeypatch.setattr(reconcile, "_build_plan", lambda root, path: plan)

    async def ingest(
        root: Path, path: Path, plan_payload: dict[str, object]
    ) -> tuple[LiveBatchMetrics, dict[str, object]]:
        return _deferred_metrics(path), {
            "attempt_id": "attempt-failed",
            "outcome_code": "legacy_unknown",
            "retryable": None,
        }

    monkeypatch.setattr(reconcile, "_normal_ingest", ingest)
    monkeypatch.setattr(reconcile, "_projection_for", lambda root: projection)
    monkeypatch.setattr(reconcile, "_tier_snapshots", lambda root: {})
    monkeypatch.setattr(reconcile, "_quick_checks", lambda root: {})

    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="raw-frontier worsening"):
        reconcile.apply_reconciliation(plan_path=plan_path, backup_manifest=tmp_path / "backup", receipt=receipt_path)

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "failed"
    assert payload["metrics"]["failed_paths"] == [
        {"path_digest": reconcile.cursor_authority_path_digest(source_path), "basename": source_path.name}
    ]
    assert payload["evidence"]["raw_frontier_worsening"] is True
    watcher.stop()


def test_receipt_integrity_evidence_is_required_for_success_but_failure_is_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {"plan_digest": "plan", "active_index": None}

    def unavailable(root: Path) -> dict[str, dict[str, object]]:
        raise reconcile.CursorAuthorityReconciliationError("state unavailable")

    monkeypatch.setattr(reconcile, "_tier_snapshots", unavailable)
    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="state unavailable"):
        reconcile._receipt_payload(
            plan=plan,
            backup={},
            root=tmp_path,
            verdict="reconciled",
            before_projection={},
            after_projection=None,
            metrics=None,
            attempt_id="attempt",
            attempt_observation="performed",
            evidence={},
            tolerate_state_errors=False,
        )
    failed = reconcile._receipt_payload(
        plan=plan,
        backup={},
        root=tmp_path,
        verdict="failed",
        before_projection={},
        after_projection=None,
        metrics=None,
        attempt_id="attempt",
        attempt_observation="performed",
        evidence={},
        tolerate_state_errors=True,
    )
    assert failed["tier_fingerprints"] is None
    assert failed["quick_check"] is None


def test_head_details_reports_missing_cursor_as_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.sources.test_live_watcher import _seed_live_cursor_authority_case

    _processor, watcher, _cursor, source_path = _seed_live_cursor_authority_case(tmp_path)
    projection = reconcile._projection_for(tmp_path)
    monkeypatch.setattr(reconcile, "_cursor_rows", lambda root: [])
    with pytest.raises(reconcile.CursorAuthorityReconciliationError, match="unique current cursor row"):
        reconcile._head_details(tmp_path, source_path, projection)
    watcher.stop()
