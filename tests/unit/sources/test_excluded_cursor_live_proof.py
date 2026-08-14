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
    assert set(cast(dict[str, Any], receipt["outcomes"])) == {"indexed", "still_excluded", "typed_terminal"}
    assert receipt["outcomes"] == {
        "indexed": True,
        "still_excluded": True,
        "typed_terminal": True,
    }
    typed_terminal = next(case for case in receipt["cases"] if case["case_id"] == "typed-terminal")
    assert typed_terminal["terminal_evidence"]["parse_error_present"] is True
    assert receipt["execution"] == {
        "mode": "candidate_fixture",
        "live_census": "not_run",
        "live_residual": "Historical excluded population and current live file states were not accessed.",
        "terminal_frontier_residual": "The typed-terminal candidate has no accepted byte head, so its readiness gate was injected for this case only.",
        "residual_successor": "polylogue-excluded-cursor-live-proof",
    }
    assert receipt["production_route"]["catch_up"] == (
        "LiveWatcher._catch_up -> _scan_catch_up_candidates -> _catch_up_candidates -> "
        "_plan_catch_up -> coordinated chunk ingest"
    )
    assert receipt["anti_vacuity"] == {
        "indexed_authority": "byte_proven_source_raw_and_revision_head",
        "indexed_session_count_before": 0,
        "indexed_session_count": 1,
        "typed_terminal_artifact": "terminal_corrupt_input",
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

        assert watcher._needs_work(path)
        actuator.assert_called_once()
        revived = cursor.get_record(path)
        assert revived is not None
        assert not revived.excluded
        assert revived.failure_count == 0
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
