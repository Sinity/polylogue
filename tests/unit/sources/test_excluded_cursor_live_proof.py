"""Executable proof for excluded-cursor revival and retry-state honesty."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import LiveWatcher, WatchSource
from tests.infra.excluded_cursor_live_proof import run_excluded_cursor_live_proof, verify_receipt


def test_candidate_fixture_proves_all_cursor_outcomes_and_is_immutable(tmp_path: Path) -> None:
    archive_root = tmp_path / "candidate-archive"
    receipt_path = tmp_path / "proof.json"

    receipt = run_excluded_cursor_live_proof(archive_root, receipt_path)
    checked = verify_receipt(receipt_path)

    assert checked == receipt
    assert set(cast(dict[str, Any], receipt["outcomes"])) == {"indexed", "still_excluded", "deferred_partial"}
    assert receipt["outcomes"] == {
        "indexed": True,
        "still_excluded": True,
        "deferred_partial": True,
    }
    deferred_partial = next(case for case in receipt["cases"] if case["case_id"] == "deferred-partial")
    assert deferred_partial["failure_evidence"] == {
        "artifact_kind": "deferred_hot_jsonl_capture",
        "support_status": "partial_decode",
        "parse_error_present": False,
    }
    assert receipt["execution"] == {
        "mode": "candidate_fixture",
        "live_census": "not_run",
        "live_residual": "Historical excluded population and current live file states were not accessed.",
        "partial_tail_frontier_residual": "The deferred-partial candidate has no accepted byte head, so its readiness gate was injected for this case only.",
        "residual_successor": "polylogue-excluded-cursor-live-proof",
    }
    assert receipt["production_route"]["intake"] == (
        "FairIntakeDispatcher.run_once -> FileIntakeAdapter.admit_page -> "
        "LiveWatcher.select_ingest_candidates -> page ingest"
    )
    assert receipt["anti_vacuity"] == {
        "indexed_authority": "byte_proven_source_raw_and_revision_head",
        "indexed_session_count_before": 0,
        "indexed_session_count": 1,
        "deferred_partial_artifact": "deferred_hot_jsonl_capture",
        "unchanged_excluded_attempt_present": False,
    }


def test_parser_fingerprint_revival_calls_real_actuator_and_excludes_unchanged_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "codex"
    source_root.mkdir()
    path = source_root / "excluded.jsonl"
    path.write_text("payload\n", encoding="utf-8")
    cursor = CursorStore(tmp_path / "ops.db")
    stat = path.stat()
    cursor.set(
        path,
        stat.st_size,
        byte_offset=stat.st_size,
        last_complete_newline=stat.st_size,
        parser_fingerprint="old-parser",
        content_fingerprint="payload-hash",
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        failure_count=5,
        excluded=True,
    )
    watcher = LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=source_root),),
        cursor=cursor,
    )
    try:
        actuator = Mock(wraps=cursor.revive_replaced_exclusion)
        monkeypatch.setattr(cursor, "revive_replaced_exclusion", actuator)
        monkeypatch.setattr(live_watcher, "_PARSER_FINGERPRINT", "new-parser")

        # polylogue-6q16u: the parser change schedules a fresh attempt, which
        # is what ``_needs_work`` reporting True means. The quarantine itself
        # survives the notice and is lifted by the cursor write of an ingest
        # that retained something, so a path that fails again stays dark and a
        # never-admitted path never presents a stale byte offset to the
        # raw-frontier cursor map as committed authority.
        assert watcher._needs_work(path)
        actuator.assert_not_called()
        still_quarantined = cursor.get_record(path)
        assert still_quarantined is not None
        assert still_quarantined.excluded
        assert cursor.list_retry_records() == []
    finally:
        watcher.stop()

    unchanged_root = tmp_path / "unchanged"
    unchanged_root.mkdir()
    unchanged_cursor = CursorStore(unchanged_root / "ops.db")
    assert unchanged_cursor._db_path != cursor._db_path
    assert unchanged_cursor._ops_db_path != cursor._ops_db_path
    unchanged_cursor.set(
        path,
        stat.st_size,
        byte_offset=stat.st_size,
        last_complete_newline=stat.st_size,
        parser_fingerprint="new-parser",
        content_fingerprint="payload-hash",
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        failure_count=5,
        excluded=True,
    )
    unchanged_watcher = LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=source_root),),
        cursor=unchanged_cursor,
    )
    try:
        assert not unchanged_watcher._needs_work(path)
        assert unchanged_cursor.list_excluded() == [str(path)]
        assert unchanged_cursor.list_retry_records() == []
    finally:
        unchanged_watcher.stop()


def test_receipt_with_wrong_self_hash_is_rejected(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    body = {"schema": "test", "receipt_sha256": "placeholder"}
    receipt_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(AssertionError, match="hash mismatch"):
        verify_receipt(receipt_path)


def test_full_retry_invalidation_clears_a_stale_exclusion(tmp_path: Path) -> None:
    """An admitted-but-stale handoff must stay visible to the next scan.

    ``_invalidate_cursor_for_full_retry`` records the NEWEST filesystem
    observation. Anti-vacuity: carry ``excluded`` forward and the watcher's
    exclusion branch sees an unchanged identity with a matching parser
    fingerprint and reports no work for this path forever, so the current
    bytes are never acquired and nothing lands in retry state.
    """
    from polylogue.sources.live.batch import LiveBatchProcessor

    source_root = tmp_path / "codex"
    source_root.mkdir()
    path = source_root / "replaced.jsonl"
    path.write_text('{"a":1}\n', encoding="utf-8")
    cursor = CursorStore(tmp_path / "ops.db")
    stale = path.stat()
    cursor.set(
        path,
        stale.st_size,
        byte_offset=stale.st_size,
        last_complete_newline=stale.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="payload-hash",
        source_name="codex",
        st_dev=stale.st_dev,
        st_ino=stale.st_ino,
        mtime_ns=stale.st_mtime_ns,
        excluded=True,
    )
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=source_root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    path.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
    current = path.stat()
    processor._invalidate_cursor_for_full_retry(path, source_name="codex", stat=current)

    invalidated = cursor.get_record(path)
    assert invalidated is not None
    assert invalidated.excluded is False

    watcher = LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=source_root),),
        cursor=cursor,
    )
    try:
        assert watcher._needs_work(path)
    finally:
        watcher.stop()
