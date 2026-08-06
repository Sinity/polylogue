"""Real-route tests for cursor-authority reconciliation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.maintenance import cursor_authority_reconcile as reconcile
from polylogue.sources.live.batch import CursorAuthorityBlockedError, scoped_cursor_authority_authorization


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

    assert first == second
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
    with scoped_cursor_authority_authorization(
        source_path_digest=reconcile.cursor_authority_path_digest(source_path),
        cursor_byte_offset=sample.cursor_byte_offset,
        accepted_frontier=sample.accepted_frontier,
        plan_digest="test-plan",
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
