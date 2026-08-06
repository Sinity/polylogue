"""Real-route tests for cursor-authority reconciliation."""

from __future__ import annotations

import json
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


def test_plan_contains_no_direct_cursor_reset_or_global_bypass() -> None:
    source = Path(reconcile.__file__).read_text(encoding="utf-8")
    assert "cursor.reset" not in source
    assert "force_cursor" not in source
    assert "bypass" not in source.lower()
