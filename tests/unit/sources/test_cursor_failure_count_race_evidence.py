"""Race evidence + fix proof for polylogue-9e5.4.2 / polylogue-qug2.

``CursorStore.mark_failed`` / ``.mark_excluded`` / ``.reset_failures`` used to
do an unlocked read-then-write pair: ``get_record(path)`` opened and closed
one ``ops.db`` connection/transaction, then the caller-computed new state was
written through a *second*, independent connection/transaction via
``set(...)``. No lock spanned the pair, so two actors sharing the same
``ops.db`` (e.g. the live daemon watcher tailing a source file while an
operator's CLI import/reprocess batch parses the same directory) could both
read the same stale ``failure_count`` before either committed its increment.

The fix (``CursorStore._read_modify_write_cursor_record``) scopes the read
and the write to one connection/transaction, opened with ``BEGIN IMMEDIATE``
so the write lock is held from the read onward -- a second concurrent caller
genuinely blocks (SQLite's busy-timeout wait) until the first commits, then
reads the already-updated value. This can only be proven with REAL
concurrent callers of ``mark_failed`` itself: a hand-sequenced
get_record()/set() replay (as this file used before the fix) bypasses the
fix entirely, since it never goes through the locked path.
"""

from __future__ import annotations

import threading
from pathlib import Path

from polylogue.sources.live.cursor import CursorStore


def test_mark_failed_accumulates_correctly_under_real_concurrent_callers(tmp_path: Path) -> None:
    """Fix proof: two real threads both calling ``mark_failed`` on the same
    path must not lose either increment (polylogue-qug2)."""
    store = CursorStore(tmp_path / "live.sqlite")
    path = tmp_path / "session.jsonl"
    path.write_text("{}\n")

    # Seed a baseline cursor row with two prior real failures already
    # recorded, mirroring what mark_failed would have produced after two
    # earlier (non-racing) parse failures.
    store.set(path, path.stat().st_size, failure_count=2)
    baseline = store.get_record(path)
    assert baseline is not None
    assert baseline.failure_count == 2

    # Actor A (the live daemon watcher) and Actor B (a CLI reprocess batch
    # touching the same source path) each hit a real parse failure on the
    # SAME path at roughly the same time -- real threads, not a hand-ordered
    # call sequence, so the test actually exercises BEGIN IMMEDIATE's
    # blocking behavior rather than simulating it.
    start = threading.Barrier(2)

    def call_mark_failed() -> None:
        start.wait()
        store.mark_failed(path)

    threads = [threading.Thread(target=call_mark_failed) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    assert all(not t.is_alive() for t in threads), "mark_failed deadlocked under concurrent callers"

    final = store.get_record(path)
    assert final is not None
    # Two real failures occurred (baseline->A, baseline->B); a correct
    # accounting shows failure_count == 4. Before the fix, this reproduced
    # as 3 -- one increment silently lost to the unlocked read-modify-write.
    assert final.failure_count == 4, (
        f"lost update: expected both concurrent mark_failed() calls to be counted (4), got {final.failure_count}"
    )


def test_get_record_then_set_directly_is_still_racy_by_design(tmp_path: Path) -> None:
    """The LOW-LEVEL get_record()/set() primitives remain individually
    unlocked -- only production callers (mark_failed/mark_excluded/
    reset_failures) were moved onto the locked read-modify-write path. A
    caller that manually replays get_record() then set() with a stale read
    bypasses that fix by construction; this is expected, not a regression,
    and documents why no code in this repo should do that directly."""
    store = CursorStore(tmp_path / "live.sqlite")
    path = tmp_path / "session.jsonl"
    path.write_text("{}\n")
    store.set(path, path.stat().st_size, failure_count=2)

    actor_a_read = store.get_record(path)
    actor_b_read = store.get_record(path)
    assert actor_a_read is not None
    assert actor_b_read is not None
    assert actor_a_read.failure_count == 2
    assert actor_b_read.failure_count == 2  # both saw the same stale value

    store.set(
        path,
        actor_a_read.byte_size,
        failure_count=actor_a_read.failure_count + 1,
        allow_backward=True,
    )
    store.set(
        path,
        actor_b_read.byte_size,
        failure_count=actor_b_read.failure_count + 1,
        allow_backward=True,
    )

    final = store.get_record(path)
    assert final is not None
    assert final.failure_count == 3, (
        "the low-level get_record/set primitives should still be individually unlocked -- "
        "if this now shows 4, set() itself became lock-safe and this test (and its docstring "
        "warning against direct get_record+set replay) should be revisited"
    )
