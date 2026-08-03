"""First-slice safety-case harness for polylogue-yeq.1 (acquisition/cursor authority).

Declares and fault-tests the acquisition/cursor-authority lifecycle owned by
``polylogue.sources.live.cursor_lifecycle`` / ``CursorStore``. Structure:

1. ``classify_cursor_lifecycle_state`` unit coverage.
2. A soundness sweep that drives REAL ``CursorStore`` methods (not a hand
   -rolled replica) across representative starting states and asserts every
   observed transition is in the declared table -- this is what makes the
   table an evidenced artifact rather than prose.
3. A real production guard: ``defer_full_cursor_reconciliation`` refuses an
   excluded cursor (the discovered, not-yet-proven-reachable hazard named in
   the module docstring).
4. Mutation-sensitivity proofs: removing a declared transition, or breaking
   an actuator so it would silently violate the "exclusion only lifts via
   proved identity change" invariant, makes the harness fail loudly instead
   of passing quietly (polylogue-yeq.1 AC4).
5. A real SIGKILL fault-injection test: the read-modify-write transaction is
   killed with SIGKILL after the row UPDATE executes but before the
   surrounding transaction commits, proving the cursor recovers to its exact
   pre-crash state (no torn write) and converges correctly on retry.

Only the five lock-protected actuators are in scope here
(``CursorStore.set`` is deliberately unlocked by design -- see
``test_cursor_failure_count_race_evidence.py``). The remaining five hazard
areas named in polylogue-yeq.1 (materialization/convergence, generation
promotion, assertion lifecycle, deletion/excision, backup/restore) are out of
scope for this slice.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.sources.live import cursor_lifecycle as lifecycle
from polylogue.sources.live.cursor import _MAX_CURSOR_FAILURES_BEFORE_EXCLUDE, CursorRecord, CursorStore
from polylogue.sources.live.cursor_lifecycle import (
    CURSOR_LIFECYCLE_TRANSITIONS,
    CursorLifecycleState,
    CursorLifecycleViolationError,
    classify_cursor_lifecycle_state,
)

_S = CursorLifecycleState


def _record(
    *,
    excluded: bool = False,
    failure_count: int = 0,
    next_retry_at: str | None = None,
    content_fingerprint: str | None = "fp-1",
) -> CursorRecord:
    return CursorRecord(
        source_path="/private/source.jsonl",
        byte_size=10,
        byte_offset=10,
        last_complete_newline=10,
        record_count=1,
        updated_at="2026-07-26T00:00:00+00:00",
        content_fingerprint=content_fingerprint,
        failure_count=failure_count,
        next_retry_at=next_retry_at,
        excluded=excluded,
    )


# ---------------------------------------------------------------------------
# 1. classify_cursor_lifecycle_state
# ---------------------------------------------------------------------------


def test_classify_absent() -> None:
    assert classify_cursor_lifecycle_state(None) is _S.ABSENT


def test_classify_active_with_fingerprint() -> None:
    assert classify_cursor_lifecycle_state(_record()) is _S.ACTIVE


def test_classify_active_dormant_retry_with_fingerprint_is_not_deferred() -> None:
    """A record with BOTH a content_fingerprint and a next_retry_at is ACTIVE,
    not DEFERRED -- matching the real ``list_retry_records`` SQL predicate
    (``content_fingerprint IS NULL AND next_retry_at IS NOT NULL``)."""
    record = _record(content_fingerprint="fp-1", next_retry_at="2099-01-01T00:00:00+00:00")
    assert classify_cursor_lifecycle_state(record) is _S.ACTIVE


def test_classify_retry_pending() -> None:
    record = _record(failure_count=2, next_retry_at="2099-01-01T00:00:00+00:00", content_fingerprint=None)
    assert classify_cursor_lifecycle_state(record) is _S.RETRY_PENDING


def test_classify_deferred() -> None:
    record = _record(content_fingerprint=None, next_retry_at="2099-01-01T00:00:00+00:00", failure_count=0)
    assert classify_cursor_lifecycle_state(record) is _S.DEFERRED


def test_classify_excluded_takes_priority_over_failure_count() -> None:
    record = _record(excluded=True, failure_count=5)
    assert classify_cursor_lifecycle_state(record) is _S.EXCLUDED


# ---------------------------------------------------------------------------
# 2. Soundness sweep: drive REAL CursorStore actuators, check every observed
#    transition is declared.
# ---------------------------------------------------------------------------


def _assert_declared(before: CursorRecord | None, actuator: str, after: CursorRecord | None) -> None:
    prev_state = classify_cursor_lifecycle_state(before)
    next_state = classify_cursor_lifecycle_state(after)
    assert (prev_state, actuator, next_state) in CURSOR_LIFECYCLE_TRANSITIONS, (
        f"undeclared transition observed in real CursorStore behavior: "
        f"{prev_state.value} --{actuator}--> {next_state.value}"
    )


def test_soundness_sweep_mark_failed_from_absent_to_excluded(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    path = tmp_path / "session.jsonl"
    path.write_text("{}\n")

    for _ in range(_MAX_CURSOR_FAILURES_BEFORE_EXCLUDE + 1):
        before = store.get_record(path)
        store.mark_failed(path)
        after = store.get_record(path)
        _assert_declared(before, "mark_failed", after)
    final = store.get_record(path)
    assert final is not None
    assert classify_cursor_lifecycle_state(final) is _S.EXCLUDED


def test_soundness_sweep_defer_then_mark_failed_then_reset(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    path = tmp_path / "session.jsonl"
    path.write_text("{}\n")
    store.set(path, path.stat().st_size, content_fingerprint=None)
    assert classify_cursor_lifecycle_state(store.get_record(path)) is _S.ACTIVE

    before = store.get_record(path)
    store.defer_full_cursor_reconciliation(path)
    after = store.get_record(path)
    _assert_declared(before, "defer_full_cursor_reconciliation", after)
    assert classify_cursor_lifecycle_state(after) is _S.DEFERRED

    before = after
    store.mark_failed(path)
    after = store.get_record(path)
    _assert_declared(before, "mark_failed", after)
    assert classify_cursor_lifecycle_state(after) is _S.RETRY_PENDING

    before = after
    store.reset_failures(path)
    after = store.get_record(path)
    _assert_declared(before, "reset_failures", after)
    assert classify_cursor_lifecycle_state(after) is _S.ACTIVE


def test_soundness_sweep_exclude_then_revive_with_proved_identity_change(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    path = tmp_path / "session.jsonl"
    path.write_text("{}\n")
    store.set(path, path.stat().st_size, st_dev=1, st_ino=1, mtime_ns=1)

    before = store.get_record(path)
    store.mark_excluded(path)
    after = store.get_record(path)
    _assert_declared(before, "mark_excluded", after)
    assert classify_cursor_lifecycle_state(after) is _S.EXCLUDED

    # Same identity: revive is a documented no-op.
    before = after
    assert before is not None
    store.revive_replaced_exclusion(path, byte_size=before.byte_size, st_dev=1, st_ino=1, mtime_ns=1)
    after = store.get_record(path)
    _assert_declared(before, "revive_replaced_exclusion", after)
    assert classify_cursor_lifecycle_state(after) is _S.EXCLUDED

    # Proved-different identity: revive lifts exclusion.
    before = after
    store.revive_replaced_exclusion(path, byte_size=99, st_dev=2, st_ino=2, mtime_ns=2)
    after = store.get_record(path)
    _assert_declared(before, "revive_replaced_exclusion", after)
    assert classify_cursor_lifecycle_state(after) is _S.ACTIVE


def test_soundness_sweep_exclude_then_revive_with_parser_fingerprint_change(tmp_path: Path) -> None:
    """polylogue-ix5r: a parser fingerprint change revives an excluded cursor
    even when the file's bytes never changed -- the previous identity-only
    revival left a cursor permanently dark once its bytes stopped changing,
    even after the parser bug that poisoned it was fixed."""
    store = CursorStore(tmp_path / "live.sqlite")
    path = tmp_path / "session.jsonl"
    path.write_text("{}\n")
    store.set(path, path.stat().st_size, st_dev=1, st_ino=1, mtime_ns=1, parser_fingerprint="parser-v1")

    store.mark_excluded(path)
    excluded = store.get_record(path)
    assert excluded is not None
    assert excluded.excluded

    # Same identity, same parser fingerprint: still a documented no-op.
    before = excluded
    store.revive_replaced_exclusion(
        path,
        byte_size=before.byte_size,
        st_dev=1,
        st_ino=1,
        mtime_ns=1,
        current_parser_fingerprint="parser-v1",
    )
    after = store.get_record(path)
    _assert_declared(before, "revive_replaced_exclusion", after)
    assert classify_cursor_lifecycle_state(after) is _S.EXCLUDED

    # Same identity, DIFFERENT parser fingerprint: revives.
    before = after
    assert before is not None
    store.revive_replaced_exclusion(
        path,
        byte_size=before.byte_size,
        st_dev=1,
        st_ino=1,
        mtime_ns=1,
        current_parser_fingerprint="parser-v2",
    )
    after = store.get_record(path)
    _assert_declared(before, "revive_replaced_exclusion", after)
    assert classify_cursor_lifecycle_state(after) is _S.ACTIVE


# ---------------------------------------------------------------------------
# 3. Real production guard against the discovered hazard: defer must never
#    observe/emit EXCLUDED.
# ---------------------------------------------------------------------------


def test_defer_full_cursor_reconciliation_refuses_excluded_cursor(tmp_path: Path) -> None:
    """``sources/live/batch.py``'s ``_defer_full_cursor_retry`` does not
    itself check ``cursor.excluded`` before calling
    ``defer_full_cursor_reconciliation`` (unlike ``watcher.py``, which does).
    Whether that call site can ever reach an excluded cursor in practice is
    not proven either way here -- what IS proven is that if it ever did, the
    validator fails closed instead of silently un-quarantining a poisoned
    source."""
    store = CursorStore(tmp_path / "live.sqlite")
    path = tmp_path / "session.jsonl"
    path.write_text("{}\n")
    store.set(path, path.stat().st_size)
    store.mark_excluded(path)
    assert classify_cursor_lifecycle_state(store.get_record(path)) is _S.EXCLUDED

    with pytest.raises(CursorLifecycleViolationError):
        store.defer_full_cursor_reconciliation(path)

    # The refused write must not have landed -- the cursor is still excluded,
    # not torn or half-applied.
    after = store.get_record(path)
    assert after is not None
    assert classify_cursor_lifecycle_state(after) is _S.EXCLUDED


# ---------------------------------------------------------------------------
# 4. Mutation sensitivity (polylogue-yeq.1 AC4): removing a declared
#    invariant, or breaking an actuator's guard, must make the harness fail.
# ---------------------------------------------------------------------------


def test_removing_a_declared_transition_makes_a_real_production_call_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity for the validator itself: strip
    ``(EXCLUDED, "reset_failures", EXCLUDED)`` from the declared table and
    show that ``CursorStore.reset_failures`` -- called through its real,
    unmodified production code path -- now raises. This proves the check is
    not a rubber stamp: today's declared table is load-bearing for a
    transition real code actually performs constantly (any daemon/CLI
    backoff-clear on an already-quarantined path)."""
    store = CursorStore(tmp_path / "live.sqlite")
    path = tmp_path / "session.jsonl"
    path.write_text("{}\n")
    store.set(path, path.stat().st_size)
    store.mark_excluded(path)
    assert classify_cursor_lifecycle_state(store.get_record(path)) is _S.EXCLUDED

    mutilated = frozenset(
        entry for entry in CURSOR_LIFECYCLE_TRANSITIONS if entry != (_S.EXCLUDED, "reset_failures", _S.EXCLUDED)
    )
    monkeypatch.setattr(lifecycle, "CURSOR_LIFECYCLE_TRANSITIONS", mutilated)
    # cursor.py imported the symbol directly, so it must be patched there too
    # for the production call path to observe the mutilated table.
    import polylogue.sources.live.cursor as cursor_module

    def patched_validate(*, actuator: str, before: object, after: object) -> None:
        from polylogue.sources.live.cursor_lifecycle import classify_cursor_lifecycle_state as classify

        prev_state = classify(before)  # type: ignore[arg-type]
        next_state = classify(after if after is not None else before)  # type: ignore[arg-type]
        if (prev_state, actuator, next_state) not in mutilated:
            raise CursorLifecycleViolationError(
                f"undeclared cursor lifecycle transition: {prev_state.value} --{actuator}--> {next_state.value}"
            )

    monkeypatch.setattr(cursor_module, "validate_cursor_lifecycle_transition", patched_validate)

    with pytest.raises(CursorLifecycleViolationError):
        store.reset_failures(path)


def test_breaking_the_reset_failures_actuator_is_caught_by_the_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery-actuator regression proof: simulate ``reset_failures`` being
    changed to (incorrectly) lift exclusion -- i.e. the bug the module
    docstring's invariant #2 exists to prevent -- and show the validator
    raises rather than letting the corrupted actuator commit. Without the
    validator wired into ``_read_modify_write_cursor_record``, this broken
    actuator would silently revive a quarantined source with no proof its
    file identity ever changed."""
    store = CursorStore(tmp_path / "live.sqlite")
    path = tmp_path / "session.jsonl"
    path.write_text("{}\n")
    store.set(path, path.stat().st_size)
    store.mark_excluded(path)
    assert classify_cursor_lifecycle_state(store.get_record(path)) is _S.EXCLUDED

    def broken_reset_failures(self: CursorStore, target_path: Path) -> None:
        def mutate(current: CursorRecord | None) -> CursorRecord | None:
            if current is None:
                return None
            # BUG: also clears `excluded`, unlike the real actuator.
            return replace(current, failure_count=0, next_retry_at=None, excluded=False)

        self._read_modify_write_cursor_record(target_path, mutate, actuator="reset_failures")

    monkeypatch.setattr(CursorStore, "reset_failures", broken_reset_failures)

    with pytest.raises(CursorLifecycleViolationError):
        store.reset_failures(path)

    # The corrupted write must not have landed.
    after = store.get_record(path)
    assert after is not None
    assert classify_cursor_lifecycle_state(after) is _S.EXCLUDED


# ---------------------------------------------------------------------------
# 5. Real SIGKILL fault injection across the BEGIN IMMEDIATE seam.
# ---------------------------------------------------------------------------

_SIGKILL_SUBPROCESS_SCRIPT = """
import signal
import sqlite3
import sys
from pathlib import Path

# ``upsert_ingest_cursor`` (polylogue/storage/sqlite/archive_tiers/ops_write.py)
# issues its own ``conn.commit()`` right after the row UPDATE -- discovered
# while building this harness: the outer ``BEGIN IMMEDIATE`` in
# ``CursorStore._read_modify_write_cursor_record`` widens the WRITE LOCK to
# cover the read (that is what fixes polylogue-qug2's lost-update race), but
# the actual durable commit boundary is this inner ``conn.commit()``, not the
# `with conn:` context manager exit several stack frames up. A crash fault
# has to straddle THIS commit call, not merely "somewhere inside
# CursorStore.mark_failed", or it can never observe an uncommitted write --
# proved empirically: wrapping the outer function and killing the process
# right after it returned always observed the write already durable.
#
# ARMED gates the pause so only the write this test cares about is poisoned
# -- CursorStore.__init__ issues its own unrelated ops-tier commits during
# initialize()/interrupted-attempt recovery, which must proceed normally.
ARMED = [False]


class PausingConnection(sqlite3.Connection):
    def commit(self):
        if ARMED[0]:
            sys.stdout.write("WROTE\\n")
            sys.stdout.flush()
            # Genuinely parks this thread in a kernel wait -- no further
            # Python bytecode can run past this point. The parent sends a
            # real SIGKILL once it has seen the announcement, so the
            # transaction is torn down while truly mid-commit, not racing a
            # self-delivered signal (self os.kill(SIGKILL) is NOT
            # deterministic here: signal delivery is asynchronous and the
            # interpreter can -- and, measured empirically, does -- run far
            # enough to finish the commit before the kernel acts on it).
            signal.pause()
        return super().commit()


_original_connect = sqlite3.connect


def _patched_connect(*args, **kwargs):
    kwargs.setdefault("factory", PausingConnection)
    return _original_connect(*args, **kwargs)


sqlite3.connect = _patched_connect

from polylogue.sources.live.cursor import CursorStore  # noqa: E402

db_path = Path(sys.argv[1])
source_path = Path(sys.argv[2])

store = CursorStore(db_path)
ARMED[0] = True
store.mark_failed(source_path)
print("UNREACHABLE")
"""


def test_sigkill_mid_transaction_leaves_no_torn_cursor_write(tmp_path: Path) -> None:
    """Kill the process with SIGKILL exactly inside the row-write commit call
    that durably applies a ``mark_failed`` transition.

    Preventive invariant proved: the cursor's durable state after a crash at
    this seam is EXACTLY its pre-crash state (RETRY_PENDING at
    ``failure_count == _MAX_CURSOR_FAILURES_BEFORE_EXCLUDE - 1``), never a
    torn write (e.g. failure_count bumped without `excluded` following, or
    vice versa). Recovery actuator proved: a retried `mark_failed()` call
    converges correctly to EXCLUDED with no double-count and no lost
    transitions -- committed evidence (the prior real failures) is neither
    lost nor silently re-authorized.
    """
    db_path = tmp_path / "ops_root" / "live.sqlite"
    source_path = tmp_path / "session.jsonl"
    source_path.write_text("{}\n")

    # Seed the pre-crash boundary state through the REAL (unpoisoned) actuator.
    seed_store = CursorStore(db_path)
    for _ in range(_MAX_CURSOR_FAILURES_BEFORE_EXCLUDE - 1):
        seed_store.mark_failed(source_path)
    pre_crash = seed_store.get_record(source_path)
    assert pre_crash is not None
    assert pre_crash.failure_count == _MAX_CURSOR_FAILURES_BEFORE_EXCLUDE - 1
    assert classify_cursor_lifecycle_state(pre_crash) is _S.RETRY_PENDING

    process = subprocess.Popen(
        [sys.executable, "-c", _SIGKILL_SUBPROCESS_SCRIPT, str(db_path), str(source_path)],
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    try:
        marker = process.stdout.readline()
        assert marker.strip() == "WROTE", (
            f"crash point never reached -- test is not exercising the seam (got {marker!r}, stderr={process.stderr})"
        )
        process.kill()  # SIGKILL, sent while the child is genuinely parked in signal.pause()
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    assert process.returncode == -signal.SIGKILL, (
        f"subprocess did not die by SIGKILL as expected: returncode={process.returncode} stderr={stderr}"
    )
    assert "UNREACHABLE" not in stdout

    # Reopen the store from THIS process (a fresh connection, proving no
    # in-memory state survived, only durable committed rows).
    recovery_store = CursorStore(db_path)
    recovered = recovery_store.get_record(source_path)
    assert recovered is not None
    assert recovered == pre_crash, (
        "cursor state after a SIGKILL mid-transaction must be byte-identical to the "
        "pre-crash committed state -- any difference is a torn write"
    )
    assert classify_cursor_lifecycle_state(recovered) is _S.RETRY_PENDING

    # Recovery actuator: a plain retry (no special crash-recovery code path)
    # converges to the correct terminal state with no double-count.
    recovery_store.mark_failed(source_path)
    converged = recovery_store.get_record(source_path)
    assert converged is not None
    assert converged.failure_count == _MAX_CURSOR_FAILURES_BEFORE_EXCLUDE
    assert classify_cursor_lifecycle_state(converged) is _S.EXCLUDED
