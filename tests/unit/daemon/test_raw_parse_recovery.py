"""Regression coverage for polylogue-61jg: interrupted ingest requeue.

Before this fix, ``CursorStore._mark_interrupted_ops_attempts`` stamped a
dangling ``running`` ``ingest_attempts`` row ``interrupted`` on the next
daemon start and stopped there -- nothing registered the affected source
path as retryable ``convergence_debt``, so a raw row acquired but never
validated/parsed waited for an accidental future touch instead of being
requeued (2026-07-31 acquisition-completeness audit, F-07/F-08).

These tests pin:
1. An interrupted ingest attempt registers ``raw_parse_recovery``
   convergence debt for every source path it covered (including the legacy
   single-``source_path`` fallback for rows without ``source_paths_json``).
2. ``make_raw_parse_recovery_stage``'s ``check``/``execute`` correctly
   detects and drains a stuck raw row (acquired, never parsed, or parsed but
   not indexed, with no materialized session) for a given source path.
3. End to end: a simulated daemon kill mid-batch (SIGKILL -> reopen
   ``CursorStore``) demonstrably resumes parsing on next start once the
   registered debt is drained by the daemon's convergence loop -- the
   bead's AC3.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider, ValidationStatus
from polylogue.core.errors import RawCASFrontierError
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.daemon.convergence import DaemonConverger, StageState
from polylogue.daemon.convergence_stages import make_raw_parse_recovery_stage
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.archive_identity import archive_file_set_root
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.raw_retention import RawFrontierBlockedPaths
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.archive_templates import bootstrap_archive_root

_CHATGPT_CONVERSATION = {
    "id": "conv-stuck",
    "title": "stuck conversation",
    "create_time": 1,
    "current_node": "message-1",
    "mapping": {
        "message-1": {
            "id": "message-1",
            "parent": None,
            "children": [],
            "message": {
                "id": "message-1",
                "author": {"role": "user"},
                "create_time": 1,
                "content": {"content_type": "text", "parts": ["stuck raw content"]},
            },
        }
    },
}


def _write_stuck_raw(archive_root: Path, *, source_path: str, native_id: str = "conv-stuck") -> str:
    """Write a raw row with real conversational content that is never parsed.

    Mirrors an ingest attempt that acquired bytes but was interrupted before
    validation/parse ever ran: ``parsed_at_ms``/``validated_at_ms`` stay NULL
    and no index session exists for it. ``native_id`` distinguishes payloads:
    the archive is content-addressed, so two identical bodies are one raw row.
    """
    import json

    conversation = {**_CHATGPT_CONVERSATION, "id": native_id}
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        return archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps([conversation]).encode(),
            source_path=source_path,
            acquired_at_ms=1,
        )


def _sessions_for_raw(archive_root: Path, raw_id: str) -> list[tuple[object, ...]]:
    conn = sqlite3.connect(archive_root / "index.db")
    try:
        return list(conn.execute("SELECT native_id, raw_id FROM sessions WHERE raw_id = ?", (raw_id,)))
    finally:
        conn.close()


def test_interrupted_ingest_attempt_registers_raw_parse_recovery_debt(tmp_path: Path) -> None:
    db = tmp_path / "live.sqlite"
    store = CursorStore(db)
    src = tmp_path / "conv.json"
    src.write_text("x")
    store.begin_ingest_attempt(paths=[src], input_bytes=1, queued_file_count=1)

    # Simulate SIGKILL: the attempt never finished. Reopening CursorStore is
    # the daemon-restart recovery path.
    reopened = CursorStore(db)

    debts = reopened.list_convergence_debt(limit=50)
    matching = [d for d in debts if d.stage == "raw_parse_recovery" and d.subject_id == str(src)]
    assert len(matching) == 1, f"expected exactly one raw_parse_recovery debt row for {src}, got {debts}"
    assert matching[0].subject_type == "source_path"


def test_interrupted_attempt_without_source_paths_json_falls_back_to_source_path(tmp_path: Path) -> None:
    """Legacy rows populate only ``source_path``, not the JSON list -- still requeued."""
    db = tmp_path / "live.sqlite"
    store = CursorStore(db)
    src = tmp_path / "legacy.jsonl"
    src.write_text("x")
    attempt_id = store.begin_ingest_attempt(paths=[src], input_bytes=1, queued_file_count=1)

    ops_db = db.with_name("ops.db")
    conn = sqlite3.connect(ops_db)
    try:
        conn.execute(
            "UPDATE ingest_attempts SET source_paths_json = '[]', source_path = ? WHERE attempt_id = ?",
            (str(src), attempt_id),
        )
        conn.commit()
    finally:
        conn.close()

    reopened = CursorStore(db)
    debts = reopened.list_convergence_debt(limit=50)
    matching = [d for d in debts if d.stage == "raw_parse_recovery" and d.subject_id == str(src)]
    assert len(matching) == 1


def test_raw_parse_recovery_stage_drains_a_stuck_raw_row(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    path = tmp_path / "stuck.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert stage.check(path) is True, "stuck raw row was not detected as pending backlog"

    result = stage.execute(path)
    assert bool(result) is True

    assert stage.check(path) is False, "raw row should be fully materialized after execute"

    rows = _sessions_for_raw(tmp_path, raw_id)
    assert len(rows) == 1
    assert rows[0][0] == "conv-stuck"


def test_single_observation_recovery_survives_output_loss_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A minimal pass must publish, including after losing output and ops hints.

    Anti-vacuity: using all inspection capacity before publication strands the
    first raw forever; falling back to the legacy scanner fails explicitly.
    """
    bootstrap_archive_root(tmp_path)
    source = tmp_path / "source"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(source / "first.json"))
    monkeypatch.setattr("polylogue.daemon.convergence_stages._RAW_PARSE_RECOVERY_BATCH_LIMIT", 1)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy raw candidate scanner was called")

    monkeypatch.setattr("polylogue.storage.raw_convergence._raw_materialization_candidate_ids", forbidden)
    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")
    assert stage.execute(source) is True
    assert _sessions_for_raw(tmp_path, raw_id) == [("conv-stuck", raw_id)]

    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM sessions WHERE raw_id = ?", (raw_id,))
        conn.commit()
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("DELETE FROM convergence_debt")
        conn.commit()
    later_id = _write_stuck_raw(tmp_path, source_path=str(source / "later.json"), native_id="later")
    restarted = make_raw_parse_recovery_stage(tmp_path / "index.db")
    for _ in range(4):
        if restarted.execute(source):
            break
    assert restarted.check(source) is False
    assert _sessions_for_raw(tmp_path, raw_id) == [("conv-stuck", raw_id)]
    assert _sessions_for_raw(tmp_path, later_id) == [("later", later_id)]


def test_raw_parse_recovery_stage_drains_a_parsed_but_unindexed_raw(tmp_path: Path) -> None:
    """A completed parse must still be retried when index projection was lost."""
    initialize_active_archive_root(tmp_path)
    path = tmp_path / "parsed-before-index.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            UPDATE raw_sessions
            SET parsed_at_ms = 1000, validated_at_ms = 1000,
                validation_status = 'passed', parse_error = NULL
            WHERE raw_id = ?
            """,
            (raw_id,),
        )
        conn.commit()

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    # Anti-vacuity: the old probe only considered parsed_at_ms IS NULL rows,
    # so this stranded raw was incorrectly declared converged.
    assert stage.check(path) is True
    assert stage.execute(path) is True
    assert stage.check(path) is False
    assert _sessions_for_raw(tmp_path, raw_id) == [("conv-stuck", raw_id)]


@pytest.mark.parametrize("refusal_shape", ["unattributed", "own_path"])
def test_raw_parse_recovery_stage_blocks_unproven_cursor_authority(
    refusal_shape: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_archive_root(tmp_path)
    path = tmp_path / "stuck.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))
    refusal = (
        RawFrontierBlockedPaths(frozenset(), "unknown cursor authority")
        if refusal_shape == "unattributed"
        else RawFrontierBlockedPaths(frozenset({str(path)}), None)
    )
    monkeypatch.setattr("polylogue.readiness.capability.raw_frontier_source_selection_refusal", lambda _root: refusal)

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert stage.execute(path) is False
    assert stage.check(path) is True
    assert _sessions_for_raw(tmp_path, raw_id) == []


def test_raw_parse_recovery_stage_proceeds_when_another_path_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another path's broken authority does not stall this path's recovery.

    Anti-vacuity: treating any attributed refusal as a global block (the
    pre-split gate) returns ``False`` here and leaves the raw unmaterialized.
    """
    bootstrap_archive_root(tmp_path)
    path = tmp_path / "stuck.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))
    monkeypatch.setattr(
        "polylogue.readiness.capability.raw_frontier_source_selection_refusal",
        lambda _root: RawFrontierBlockedPaths(frozenset({str(tmp_path / "other-broken.jsonl")}), None),
    )

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert bool(stage.execute(path)) is True
    assert stage.check(path) is False
    assert len(_sessions_for_raw(tmp_path, raw_id)) == 1


def test_raw_parse_recovery_stage_is_false_means_pending() -> None:
    assert make_raw_parse_recovery_stage(Path("/nonexistent/index.db")).false_means_pending is True


def test_raw_parse_recovery_missing_source_is_no_backlog(tmp_path: Path) -> None:
    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")
    converger = DaemonConverger(stages=(stage,))

    states, _timings = converger.converge_batch([tmp_path / "missing.json"])

    state = states[tmp_path / "missing.json"]
    assert state.stages["raw_parse_recovery"] is StageState.DONE
    assert state.converged is True
    assert state.error_count == 0


def test_raw_parse_recovery_missing_source_tier_is_retryable(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    (tmp_path / "source.db").rename(tmp_path / "source.db.unavailable")
    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")
    converger = DaemonConverger(stages=(stage,))

    states, _timings = converger.converge_batch([tmp_path / "missing-source.json"])

    state = states[tmp_path / "missing-source.json"]
    assert state.stages["raw_parse_recovery"] is StageState.FAILED
    assert state.converged is False
    assert state.error_count == 1


def test_raw_parse_recovery_no_qualifying_rows_is_done(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")
    converger = DaemonConverger(stages=(stage,))

    states, _timings = converger.converge_batch([tmp_path / "untracked.json"])

    state = states[tmp_path / "untracked.json"]
    assert state.stages["raw_parse_recovery"] is StageState.DONE
    assert state.converged is True
    assert state.error_count == 0


def test_raw_parse_recovery_uses_active_index_pointer(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    path = tmp_path / "pointer.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))

    active_index = tmp_path / "generations" / "active" / "index.db"
    active_index.parent.mkdir(parents=True)
    shutil.copy2(tmp_path / "index.db", active_index)
    with sqlite3.connect(active_index) as conn:
        origin = conn.execute("SELECT origin FROM main.sessions LIMIT 1").fetchone()
        if origin is None:
            with sqlite3.connect(tmp_path / "source.db") as source_conn:
                origin = source_conn.execute("SELECT origin FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()
        assert origin is not None
        conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, content_hash) VALUES (?, ?, ?, zeroblob(32))",
            ("conv-stuck", origin[0], "stale-index-only-raw"),
        )
        conn.commit()

    (tmp_path / ".index-active-pointer").write_text(f"{active_index}\n", encoding="utf-8")
    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert archive_file_set_root(archive_root=tmp_path, db_path=active_index) == tmp_path
    assert stage.check(path) is True


def test_raw_parse_recovery_retries_authorized_parse_failure(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    path = tmp_path / "retryable.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET parse_error = ?, parsed_at_ms = NULL WHERE raw_id = ?",
            ("OperationalError: database is locked", raw_id),
        )
        conn.commit()

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert stage.check(path) is True


@pytest.mark.parametrize(
    "evidence_kind",
    [
        RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER,
        RawFailureEvidenceKind.DEFERRED_CODEX_CAS_FRONTIER,
    ],
)
def test_raw_parse_recovery_stage_drains_typed_cas_frontier_failure(
    tmp_path: Path, evidence_kind: RawFailureEvidenceKind
) -> None:
    """A stopped daemon requeues canonical and historical CAS retry authority."""
    bootstrap_archive_root(tmp_path)
    path = tmp_path / "cas-frontier.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        if evidence_kind is RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER:
            archive.mark_raw_parse_failed(
                raw_id,
                provider=Provider.CHATGPT,
                error=RawCASFrontierError("frontier changed while daemon stopped"),
            )
        else:
            archive.record_raw_failure_evidence(
                raw_id,
                provider=Provider.CHATGPT,
                source_path=str(path),
                source_index=0,
                acquired_at_ms=1,
                kind=evidence_kind,
            )
            archive.mark_raw_parse_failed(
                raw_id,
                provider=Provider.CHATGPT,
                error=ValueError("historical CAS frontier failure"),
                preserve_existing_failure_evidence=True,
            )

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert stage.check(path) is True
    assert stage.execute(path) is True
    assert stage.check(path) is False
    assert _sessions_for_raw(tmp_path, raw_id) == [("conv-stuck", raw_id)]


def test_raw_parse_recovery_skips_validation_failed_cas_frontier_failure(tmp_path: Path) -> None:
    """A failed validation cannot keep CAS recovery debt pending forever."""
    bootstrap_archive_root(tmp_path)
    path = tmp_path / "validation-failed-cas-frontier.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.mark_raw_parse_failed(
            raw_id,
            provider=Provider.CHATGPT,
            error=RawCASFrontierError("frontier changed after validation failed"),
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET validation_status = 'failed' WHERE raw_id = ?", (raw_id,))
        assert conn.total_changes == 1
        conn.commit()

    assert make_raw_parse_recovery_stage(tmp_path / "index.db").check(path) is False


def test_raw_parse_recovery_drains_previously_parsed_cas_frontier_failure(tmp_path: Path) -> None:
    """A stale validation failure does not suppress newer parse authority."""
    bootstrap_archive_root(tmp_path)
    path = tmp_path / "previously-parsed-cas-frontier.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.mark_raw_parse_succeeded(raw_id, provider=Provider.CHATGPT)
        archive.mark_raw_parse_failed(
            raw_id,
            provider=Provider.CHATGPT,
            error=RawCASFrontierError("frontier changed after parsing completed"),
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        parsed_at_ms = int(
            conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
        )
        conn.execute(
            "UPDATE raw_sessions SET validation_status = 'failed', validated_at_ms = ? WHERE raw_id = ?",
            (parsed_at_ms - 1, raw_id),
        )
        assert conn.total_changes == 1
        conn.commit()

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert stage.check(path) is True
    assert stage.execute(path) is True
    assert stage.check(path) is False
    assert _sessions_for_raw(tmp_path, raw_id) == [("conv-stuck", raw_id)]


def test_raw_parse_recovery_uses_monotonic_parse_state_after_failed_validation(tmp_path: Path) -> None:
    """The probe and repair route agree when a later parse supersedes validation."""
    bootstrap_archive_root(tmp_path)
    path = tmp_path / "monotonic-validation-recovery.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.finalize_raw_parse_state(
            raw_id,
            state=RawSessionStateUpdate(
                parsed_at="1970-01-01T00:00:00.001Z",
                validation_status=ValidationStatus.FAILED,
                validation_error="older validation failure",
            ),
        )
        archive.mark_raw_parse_failed(
            raw_id,
            provider=Provider.CHATGPT,
            error=RawCASFrontierError("retry after the later parser state"),
        )

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert stage.check(path) is True
    assert stage.execute(path) is True
    assert stage.check(path) is False
    assert _sessions_for_raw(tmp_path, raw_id) == [("conv-stuck", raw_id)]


def test_raw_parse_recovery_skips_current_validation_failure_after_prior_parse(tmp_path: Path) -> None:
    """A current validation failure cannot leave CAS recovery permanently pending."""
    bootstrap_archive_root(tmp_path)
    path = tmp_path / "current-validation-failed-cas-frontier.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.mark_raw_parse_succeeded(raw_id, provider=Provider.CHATGPT)
        archive.mark_raw_parse_failed(
            raw_id,
            provider=Provider.CHATGPT,
            error=RawCASFrontierError("frontier changed before current validation failure"),
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        parsed_at_ms = int(
            conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
        )
        conn.execute(
            "UPDATE raw_sessions SET validation_status = 'failed', validated_at_ms = ? WHERE raw_id = ?",
            (parsed_at_ms, raw_id),
        )
        conn.commit()

    assert make_raw_parse_recovery_stage(tmp_path / "index.db").check(path) is False


def test_raw_parse_recovery_source_open_failure_is_failed_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap_archive_root(tmp_path)
    source_path = tmp_path / "unavailable.json"
    calls = 0

    def fail_connect(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("source tier unavailable")

    monkeypatch.setattr("polylogue.daemon.convergence_stages.sqlite3.connect", fail_connect)

    converger = DaemonConverger(stages=(make_raw_parse_recovery_stage(tmp_path / "index.db"),))
    states, _timings = converger.converge_batch([source_path])

    state = states[source_path]
    assert state.stages["raw_parse_recovery"] is StageState.FAILED
    assert state.converged is False
    assert state.error_count == 1

    states, _timings = converger.converge_batch([source_path])
    assert states[source_path].stages["raw_parse_recovery"] is StageState.FAILED
    assert states[source_path].error_count == 2
    assert calls == 2


@pytest.mark.parametrize("failure", ["attach", "query"])
def test_raw_parse_recovery_sqlite_probe_failure_is_failed_and_closes_connection(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap_archive_root(tmp_path)
    source_path = tmp_path / f"{failure}-failed.json"

    class FailingConnection:
        closed = False

        def execute(self, sql: str, *_args: object, **_kwargs: object) -> object:
            if sql.lstrip().upper().startswith("ATTACH"):
                if failure == "attach":
                    raise sqlite3.OperationalError("attach failed")
                return self
            raise sqlite3.OperationalError("query failed")

        def fetchall(self) -> list[tuple[str]]:
            raise AssertionError("failed probe should not fetch rows")

        def close(self) -> None:
            self.closed = True

    connection = FailingConnection()
    monkeypatch.setattr("polylogue.daemon.convergence_stages.sqlite3.connect", lambda *_args, **_kwargs: connection)

    converger = DaemonConverger(stages=(make_raw_parse_recovery_stage(tmp_path / "index.db"),))
    states, _timings = converger.converge_batch([source_path])

    state = states[source_path]
    assert state.stages["raw_parse_recovery"] is StageState.FAILED
    assert state.converged is False
    assert state.error_count == 1
    assert connection.closed is True


class _PlanRecordingConnection:
    """Records the query plan of every ``raw_sessions`` statement the probe runs."""

    def __init__(self, conn: sqlite3.Connection, plans: list[list[str]]) -> None:
        self._conn = conn
        self._plans = plans

    @property
    def row_factory(self) -> Any:
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, factory: Any) -> None:
        self._conn.row_factory = factory

    def execute(self, sql: str, parameters: Sequence[str] = ()) -> sqlite3.Cursor:
        if "raw_sessions" in sql:
            self._plans.append([row[3] for row in self._conn.execute("EXPLAIN QUERY PLAN " + sql, parameters)])
        return self._conn.execute(sql, parameters)

    def close(self) -> None:
        self._conn.close()


def _record_probe_plans(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    plans: list[list[str]] = []
    real_connect = sqlite3.connect
    monkeypatch.setattr(
        "polylogue.daemon.convergence_stages.sqlite3.connect",
        lambda *args, **kwargs: _PlanRecordingConnection(real_connect(*args, **kwargs), plans),
    )
    return plans


def test_raw_parse_recovery_probe_seeks_the_source_path_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pending probe seeks ``source_path``; it never scans ``raw_sessions``.

    A scan here costs one full ``raw_sessions`` pass per ingested file, so a
    cold rebuild pays a term quadratic in archive size. Anti-vacuity: the
    ``source_path = ? OR source_path LIKE ?`` filter this replaces plans as
    ``SCAN r`` (and, with the equality dropped, as a covering-index scan),
    both of which this assertion rejects.
    """
    bootstrap_archive_root(tmp_path)
    path = tmp_path / "seek.json"
    _write_stuck_raw(tmp_path, source_path=str(path))
    plans = _record_probe_plans(monkeypatch)

    assert make_raw_parse_recovery_stage(tmp_path / "index.db").check(path) is True

    assert len(plans) <= 3, f"expected bounded discovery, membership inspection and source lookup, got {plans}"
    steps = [step for plan in plans for step in plan]
    assert not [step for step in steps if re.match(r"^SCAN r\b", step)], steps
    assert [step for step in steps if "SEARCH r USING INDEX idx_raw_sessions_source_path" in step], steps


def test_raw_parse_recovery_probes_a_whole_batch_in_one_statement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One converger chunk costs one probe, not one probe per file.

    Anti-vacuity: without ``check_many``/``execute_many`` the converger falls
    back to the per-path ``check`` and this records four statements.
    """
    bootstrap_archive_root(tmp_path)
    paths = [tmp_path / f"batched-{index}.json" for index in range(4)]
    plans = _record_probe_plans(monkeypatch)

    converger = DaemonConverger(stages=(make_raw_parse_recovery_stage(tmp_path / "index.db"),))
    states, _timings = converger.converge_batch(paths)

    assert len(plans) == 1, f"expected one batched probe, got {len(plans)}"
    for path in paths:
        assert states[path].stages["raw_parse_recovery"] is StageState.DONE
        assert states[path].error_count == 0


def test_raw_parse_recovery_matches_descendants_of_a_directory_root(tmp_path: Path) -> None:
    """Debt registered for a directory root still finds the raws beneath it."""
    bootstrap_archive_root(tmp_path)
    nested = tmp_path / "inbox" / "conv.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(nested))

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert stage.check(tmp_path / "inbox") is True
    assert bool(stage.execute(tmp_path / "inbox")) is True
    assert len(_sessions_for_raw(tmp_path, raw_id)) == 1


@pytest.mark.parametrize(
    ("root_name", "stuck_dir"),
    [
        # ``LIKE '<root>/%'`` reads ``%`` in the root as a wildcard, so the
        # pre-fix filter claimed every sibling sharing the literal prefix.
        ("100%", "1000"),
        # Default ``LIKE`` folds ASCII case; filesystem paths do not.
        ("Case", "case"),
    ],
)
def test_raw_parse_recovery_scope_is_a_literal_case_sensitive_prefix(
    tmp_path: Path, root_name: str, stuck_dir: str
) -> None:
    """A root's scope is its own descendants, not everything ``LIKE`` accepts."""
    bootstrap_archive_root(tmp_path)
    stuck = tmp_path / stuck_dir / "conv.json"
    _write_stuck_raw(tmp_path, source_path=str(stuck))

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert stage.check(tmp_path / root_name) is False
    assert stage.check(tmp_path / stuck_dir) is True


def test_raw_parse_recovery_drains_several_paths_in_one_batch(tmp_path: Path) -> None:
    """``execute_many`` repairs every pending path the chunk carries."""
    bootstrap_archive_root(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_raw = _write_stuck_raw(tmp_path, source_path=str(first), native_id="conv-first")
    second_raw = _write_stuck_raw(tmp_path, source_path=str(second), native_id="conv-second")
    assert first_raw != second_raw

    converger = DaemonConverger(stages=(make_raw_parse_recovery_stage(tmp_path / "index.db"),))
    states, _timings = converger.converge_batch([first, second])

    assert states[first].converged is True
    assert states[second].converged is True
    assert len(_sessions_for_raw(tmp_path, first_raw)) == 1
    assert len(_sessions_for_raw(tmp_path, second_raw)) == 1


def test_daemon_restart_resumes_parsing_of_an_interrupted_batch(tmp_path: Path) -> None:
    """End-to-end AC3: a daemon kill mid-batch demonstrably resumes on restart.

    Models: attempt begins for a source path -> raw bytes for that path are
    durably acquired -> the daemon dies before validation/parse ever runs ->
    a fresh ``CursorStore`` open (the restart) registers retryable debt ->
    the daemon's own convergence loop (``DaemonConverger`` with the
    ``raw_parse_recovery`` stage) drains that debt and the session is
    materialized, without any operator-triggered manual reprocess.
    """
    bootstrap_archive_root(tmp_path)
    source_path = tmp_path / "batch.json"
    source_path.write_text("placeholder")
    raw_id = _write_stuck_raw(tmp_path, source_path=str(source_path))

    live_db = tmp_path / "live.sqlite"
    store = CursorStore(live_db, ops_db_path=tmp_path / "ops.db")
    store.begin_ingest_attempt(paths=[source_path], input_bytes=1, queued_file_count=1)

    # Simulate the crash + restart.
    restarted_store = CursorStore(live_db, ops_db_path=tmp_path / "ops.db")
    debts = restarted_store.list_convergence_debt(limit=50)
    pending_paths = [Path(d.subject_id) for d in debts if d.stage == "raw_parse_recovery"]
    assert source_path in pending_paths

    converger = DaemonConverger(stages=(make_raw_parse_recovery_stage(tmp_path / "index.db"),))
    states, _timings = converger.converge_batch(pending_paths)
    assert states[source_path].converged is True

    rows = _sessions_for_raw(tmp_path, raw_id)
    assert len(rows) == 1
    assert rows[0][0] == "conv-stuck"


def test_restart_rewinds_cursor_that_outran_unparsed_raw(tmp_path: Path) -> None:
    """An interrupted full admission cannot leave its cursor at file end.

    The source row is durable before parse/index work starts.  Reopening the
    cursor store after a simulated kill must rewind the incomplete observation
    while retaining the recovery debt that drives raw materialization.
    """
    bootstrap_archive_root(tmp_path)
    source_path = tmp_path / "cursor-ahead.json"
    source_path.write_text("placeholder")
    _write_stuck_raw(tmp_path, source_path=str(source_path))

    live_db = tmp_path / "live.sqlite"
    store = CursorStore(live_db, ops_db_path=tmp_path / "ops.db")
    store.set(
        source_path,
        source_path.stat().st_size,
        byte_offset=source_path.stat().st_size,
        last_complete_newline=source_path.stat().st_size,
        parser_fingerprint="test-parser",
        content_fingerprint="claimed-complete",
        tail_hash="claimed-complete",
    )
    store.begin_ingest_attempt(paths=[source_path], input_bytes=source_path.stat().st_size, queued_file_count=1)

    restarted_store = CursorStore(live_db, ops_db_path=tmp_path / "ops.db")

    cursor = restarted_store.get_record(source_path)
    assert cursor is not None
    assert cursor.byte_offset == 0
    assert cursor.last_complete_newline == 0
    assert cursor.content_fingerprint is None
    assert cursor.tail_hash is None
    assert any(
        debt.stage == "raw_parse_recovery" and debt.subject_id == str(source_path)
        for debt in restarted_store.list_convergence_debt(limit=50)
    )


__all__: list[str] = []


def _write_decided_unresolved_raw(archive_root: Path, *, source_path: str) -> str:
    """Write a raw whose membership arbitration concluded ``ambiguous``.

    ``apply_raw_membership_classification`` is the production arbiter: it
    records the verdict and quarantines the raw in one call. The result never
    parses and never reaches the index, which is exactly the shape that reads
    as pending recovery work forever.
    """
    from polylogue.archive.session_revision_membership import MembershipClassification
    from polylogue.pipeline.ids import session_revision_projection
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession

    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="decided-unresolved",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="ambiguous content")],
    )
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"native_id":"decided-unresolved"}\n',
            source_path=source_path,
            acquired_at_ms=1,
        )
        archive.replace_raw_membership_census(
            raw_id,
            [session],
            parser_fingerprint="test-parser",
            censused_at_ms=1,
        )
        archive.apply_raw_membership_classification(
            "codex-session:decided-unresolved",
            MembershipClassification((), (), (raw_id,)),
            {raw_id: session},
            {raw_id: session_revision_projection(session)},
            acquired_at_ms=2,
        )
    return raw_id


def test_raw_parse_recovery_terminates_on_a_decided_unresolved_membership(tmp_path: Path) -> None:
    """A decided-ambiguous verdict drains its debt instead of retrying forever.

    polylogue-plbsn/polylogue-i03t8: the raw carries no ``parse_error`` and no
    session, so the pending probe counted it on every pass while
    ``converge_raw_materialization`` reported it converged and quarantined --
    ``execute`` returned False forever and the ``raw_parse_recovery`` debt the
    interrupted-attempt sweep registered for the path was never resolved.

    Anti-vacuity: dropping the ``decided_unresolved_membership_sql`` clause
    from ``_raw_parse_recovery_pending_count`` makes ``check`` stay True and
    ``execute`` return False here, which is the reported defect.
    """
    bootstrap_archive_root(tmp_path)
    path = tmp_path / "decided-unresolved.jsonl"
    raw_id = _write_decided_unresolved_raw(tmp_path, source_path=str(path))

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT revision_authority, parsed_at_ms, parse_error FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone() == ("quarantined", None, None)
        assert conn.execute("SELECT decision FROM raw_session_memberships WHERE raw_id = ?", (raw_id,)).fetchone() == (
            "ambiguous",
        )

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db", archive_root=tmp_path)
    assert stage.check(path) is False
    assert bool(stage.execute(path)) is True
    assert stage.check(path) is False


def test_raw_parse_recovery_still_pending_while_arbitration_has_not_run(tmp_path: Path) -> None:
    """A censused-but-unarbitrated raw stays pending: only a verdict is terminal.

    Pins the narrow scope of the exclusion -- ``decision IS NULL`` is the
    conveyor hand-off state, not a decided outcome, so recovery must still
    drive it.
    """
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession

    bootstrap_archive_root(tmp_path)
    path = tmp_path / "pending-arbitration.jsonl"
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="pending-arbitration",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="pending content")],
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"native_id":"pending-arbitration"}\n',
            source_path=str(path),
            acquired_at_ms=1,
        )
        archive.replace_raw_membership_census(
            raw_id,
            [session],
            parser_fingerprint="test-parser",
            censused_at_ms=1,
        )

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db", archive_root=tmp_path)
    assert stage.check(path) is True
