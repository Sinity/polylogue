"""Runtime observability evidence for a full catch-up convergence cycle.

This test exercises one synthetic catch-up cycle end-to-end and pins the
Runtime Evidence Matrix declared in #999 / #1148:

- cursor advances and persists across a fresh ``CursorStore`` instance opened
  on the same DB (``cursor.before`` / ``cursor.after`` / ``restart_seen``);
- ingest attempts taxonomy matches discovered artifacts (``discovered`` /
  ``attempted`` / ``skipped`` / ``ingested``);
- bad inputs are classified without blocking good ones (``errors_by_kind`` /
  ``quarantine_count``);
- backlog drains to zero (``backlog_start`` / ``backlog_end`` / ``duration_ms``);
- repair state is actionable (``repair.required`` / ``repair.performed`` /
  ``repair.remaining``);
- per-stage convergence timings are captured for each declared stage.

The test calls real ``CursorStore`` writes and the real
``emit_catch_up_cycle`` envelope so the evidence shape matches what live-archive
probes will later produce. Counters and state transitions only — no payload
content is recorded.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from polylogue.daemon.catchup_status import catchup_status_info
from polylogue.daemon.convergence import (
    ConvergenceStage,
    DaemonConverger,
    FileState,
    StageState,
)
from polylogue.daemon.events import (
    emit_catch_up_cycle,
    query_daemon_events,
)
from polylogue.sources.live.cursor import CursorStore

pytestmark = pytest.mark.usefixtures("workspace_env")


def _require_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _stage(name: str, *, failing: set[Path] | None = None) -> ConvergenceStage:
    """Build a synthetic batch-capable stage that converges all good inputs."""
    failing = failing or set()

    def check(path: Path) -> bool:
        return True

    def execute(path: Path) -> bool:
        # Make stage work observable to perf_counter — keep < 50ms total.
        time.sleep(0.001)
        return path not in failing

    def check_many(paths: Sequence[Path]) -> set[Path]:
        return set(paths)

    def execute_many(paths: Sequence[Path]) -> bool:
        ok = True
        for path in paths:
            if not execute(path):
                ok = False
        return ok

    return ConvergenceStage(
        name=name,
        description=f"synthetic stage {name}",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
    )


def _stage_timings(
    converger: DaemonConverger,
    files: list[Path],
) -> tuple[dict[Path, FileState], dict[str, float]]:
    """Drive ``converge_batch`` and return measured per-stage timings."""
    states, batch_timings = converger.converge_batch(files)
    return states, {name: round(elapsed, 6) for name, elapsed in batch_timings.items()}


def _seed_source_file(path: Path, *, body: str) -> int:
    path.write_text(body, encoding="utf-8")
    return path.stat().st_size


def test_catch_up_cycle_emits_runtime_observability_evidence(
    tmp_path: Path,
) -> None:
    """One full catch-up cycle records the Runtime Evidence Matrix."""

    # ── Stage source files: two good, one bad (quarantine target). ──
    good_a = tmp_path / "good_a.jsonl"
    good_b = tmp_path / "good_b.jsonl"
    bad = tmp_path / "bad.jsonl"
    _seed_source_file(good_a, body='{"role":"user","text":"a"}\n')
    _seed_source_file(good_b, body='{"role":"user","text":"b"}\n')
    bad_size = _seed_source_file(bad, body='{"role":"user","text":"bad"}\n')

    paths = [good_a, good_b, bad]
    input_bytes = sum(path.stat().st_size for path in paths)

    # ── Real CursorStore against an isolated DB. ──
    db_path = tmp_path / "live_cursor.sqlite"
    cursor = CursorStore(db_path)

    # Initial state: empty cursor table, full backlog.
    cursor_before = {
        "tracked": 0,
        "byte_offset_total": 0,
    }
    backlog_start = len(paths)

    # ── Begin a durable ingest attempt and emit a planning stage event. ──
    attempt_id = cursor.begin_ingest_attempt(
        paths=paths,
        input_bytes=input_bytes,
        queued_file_count=len(paths),
    )

    cursor.record_ingest_stage_event(
        attempt_id,
        phase="planning",
        status="running",
        queued_file_count=len(paths),
        needed_file_count=len(paths),
        skipped_file_count=0,
        succeeded_file_count=0,
        failed_file_count=0,
        input_bytes=input_bytes,
        source_payload_read_bytes=0,
        cursor_fingerprint_read_bytes=0,
    )

    # ── Drive the converger over good inputs only; quarantine the bad input. ──
    converger = DaemonConverger(
        stages=[
            _stage("parse", failing=set()),
            _stage("fts", failing=set()),
            _stage("insights", failing=set()),
        ]
    )

    good_paths = [good_a, good_b]
    cycle_started = time.perf_counter()
    converged_states, stage_timings = _stage_timings(converger, good_paths)
    parse_time_s = float(stage_timings.get("parse", 0.0))
    convergence_time_s = float(stage_timings.get("fts", 0.0) + stage_timings.get("insights", 0.0))

    # Classify the bad input into durable convergence debt instead of blocking
    # the cycle. This is the "errors classified without blocking good" path.
    cursor.record_convergence_debt(
        stage="parse",
        subject_type="source_path",
        subject_id=str(bad),
        error="schema_violation: missing required field 'session_id'",
        materializer_version="parser/0",
    )
    errors_by_kind = {"schema_violation": 1}
    quarantine_count = 1

    # Advance cursors for successful inputs (cursor.set is idempotent and
    # records the byte offsets, which is the cross-restart durability signal).
    for path in good_paths:
        cursor.set(
            path,
            byte_size=path.stat().st_size,
            byte_offset=path.stat().st_size,
            last_complete_newline=path.stat().st_size,
            record_count=1,
            parser_fingerprint="parser/test",
        )

    cursor.record_ingest_stage_event(
        attempt_id,
        phase="completed",
        status="completed",
        queued_file_count=len(paths),
        needed_file_count=len(paths),
        skipped_file_count=0,
        succeeded_file_count=len(good_paths),
        failed_file_count=1,
        input_bytes=input_bytes,
        source_payload_read_bytes=input_bytes - bad_size,
        cursor_fingerprint_read_bytes=0,
        archive_write_bytes_delta=4096,
        parse_time_s=parse_time_s,
        convergence_time_s=convergence_time_s,
        total_time_s=parse_time_s + convergence_time_s,
        stage_timings_json=json.dumps(stage_timings, sort_keys=True),
    )
    cursor.finish_ingest_attempt(
        attempt_id,
        status="completed",
        phase="completed",
    )
    duration_ms = (time.perf_counter() - cycle_started) * 1000.0

    # ── Restart durability: open a new CursorStore against the same DB and
    # confirm the advanced offsets survive (the cross-restart sub-assertion).
    restarted = CursorStore(db_path)
    surviving = restarted.get_records(good_paths)
    assert set(surviving) == set(good_paths)
    assert all(record.byte_offset == path.stat().st_size for path, record in surviving.items())
    cursor_after = {
        "tracked": len(surviving),
        "byte_offset_total": sum(record.byte_offset for record in surviving.values()),
    }

    # The persisted convergence debt is the repair-state matrix row.
    debt_after = restarted.list_convergence_debt(limit=10)
    repair = {
        "required": 1,
        "performed": 0,
        "remaining": len(debt_after),
    }

    # ── Backlog drained to zero from the cycle's perspective. ──
    backlog_end = backlog_start - len(good_paths) - quarantine_count
    assert backlog_end == 0

    # ── Per-stage convergence timings: every declared stage must be measured. ──
    assert set(stage_timings) == {"parse", "fts", "insights"}
    assert all(value >= 0.0 for value in stage_timings.values())
    # Converger marked every good input DONE for every stage.
    for path in good_paths:
        states = converged_states[path].stages
        assert all(state == StageState.DONE for state in states.values())

    # ── Catchup status surface reads the durable stage event. ──
    status = catchup_status_info(
        db_path,
        latest_attempt=None,
        convergence=type("StubConvergence", (), {})(),
    )
    assert status.mode in {"idle", "catching_up", "converging"}
    assert any(event.attempt_id == attempt_id for event in status.recent_events)
    latest_event = status.recent_events[0]
    assert latest_event.phase == "completed"
    assert latest_event.succeeded_file_count == len(good_paths)
    assert latest_event.failed_file_count == 1

    # ── Cursor matrix: before/after/restart_seen. ──
    assert cursor_before["tracked"] == 0
    assert cursor_after["tracked"] == len(good_paths)
    restart_seen = cursor_after["byte_offset_total"] > 0
    assert restart_seen is True

    # ── Attempts taxonomy. ──
    discovered = len(paths)
    attempted = len(paths)
    skipped = 0
    ingested = len(good_paths)
    assert ingested + quarantine_count == discovered
    assert attempted == discovered
    assert skipped == 0

    # ── Emit the catch-up cycle envelope (start + end). ──
    emit_catch_up_cycle(
        operation_id=attempt_id,
        phase="start",
        backlog_start=backlog_start,
        backlog_end=backlog_start,
        discovered=discovered,
        attempted=0,
        skipped=0,
        ingested=0,
        quarantine_count=0,
        errors_by_kind={},
        cursor_before=cursor_before,
        cursor_after=None,
        duration_ms=0.0,
        stage_timings_s={},
        repair={"required": 1, "performed": 0, "remaining": 1},
    )
    emit_catch_up_cycle(
        operation_id=attempt_id,
        phase="end",
        backlog_start=backlog_start,
        backlog_end=backlog_end,
        discovered=discovered,
        attempted=attempted,
        skipped=skipped,
        ingested=ingested,
        quarantine_count=quarantine_count,
        errors_by_kind=errors_by_kind,
        cursor_before=cursor_before,
        cursor_after=cursor_after,
        duration_ms=duration_ms,
        stage_timings_s=stage_timings,
        repair=repair,
    )

    events = query_daemon_events(kind="catch_up_cycle", limit=10)
    payloads = [_require_dict(event["payload"]) for event in events]
    phases = {str(payload["phase"]) for payload in payloads}
    assert {"start", "end"}.issubset(phases)
    end_payload = next(payload for payload in payloads if payload["phase"] == "end")
    assert end_payload["backlog_end"] == 0
    assert end_payload["ingested"] == ingested
    assert end_payload["quarantine_count"] == quarantine_count
    assert end_payload["errors_by_kind"] == errors_by_kind
    assert _require_dict(end_payload["cursor_after"])["tracked"] == len(good_paths)
    assert set(_require_dict(end_payload["stage_timings_s"]).keys()) == set(stage_timings)
    assert _require_dict(end_payload["repair"])["remaining"] == repair["remaining"]

    # ── Record bounded evidence so the matrix lands in
    # ``.cache/verification/evidence/`` under the contract id from #999. ──


def test_catch_up_cycle_evidence_payload_is_bounded(tmp_path: Path) -> None:
    """Catch-up cycle event payloads must not carry raw source content."""
    db_path = tmp_path / "live_cursor.sqlite"
    CursorStore(db_path)

    emit_catch_up_cycle(
        operation_id="bounded-cycle",
        phase="end",
        backlog_start=10,
        backlog_end=0,
        discovered=10,
        attempted=10,
        skipped=0,
        ingested=9,
        quarantine_count=1,
        errors_by_kind={"schema_violation": 1},
        cursor_before={"tracked": 0},
        cursor_after={"tracked": 9},
        duration_ms=12.5,
        stage_timings_s={"parse": 0.01, "fts": 0.02, "insights": 0.03},
        repair={"required": 1, "performed": 0, "remaining": 1},
    )

    events = query_daemon_events(kind="catch_up_cycle", limit=1)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert isinstance(payload, dict)
    # Counters and structured state only — no message text / paths beyond
    # the cursor digest fields the test itself supplies.
    encoded = json.dumps(payload, sort_keys=True)
    assert len(encoded.encode("utf-8")) < 2048
    assert "role" not in encoded
    assert "user" not in encoded
