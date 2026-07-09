"""Race evidence for polylogue-9e5.4.2 (get->modify->put audit, polylogue-9e5.4).

``CursorStore.mark_failed`` / ``.mark_excluded`` / ``.reset_failures`` each do
an unlocked read-then-write pair: ``get_record(path)`` opens and closes one
``ops.db`` connection/transaction, then the caller-computed new state is
written through a *second*, independent connection/transaction via
``set(...)``. No lock spans the pair (``best_effort_cursor_write`` only
retries on transient ``database is locked`` errors — it is not a cross-call
mutex), so two actors sharing the same ``ops.db`` (e.g. the live daemon
watcher tailing a source file while an operator's CLI import/reprocess batch
parses the same directory) can both read the same stale ``failure_count``
before either commits its increment.

This test is evidence for the bug bead, not a fix: it drives the exact
two-actor interleaving by explicit call order (deterministic, no real
threads needed — call order *is* the race) using the real production
``CursorStore.get_record``/``.set`` methods, and asserts the literal lost
update. It intentionally reproduces the current (buggy) behavior; a fix
landing under the follow-up bead should update this assertion.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.sources.live.cursor import CursorStore


def test_mark_failed_lost_update_when_two_actors_read_before_either_writes(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    path = tmp_path / "session.jsonl"
    path.write_text("{}\n")

    # Seed a baseline cursor row with two prior real failures already
    # recorded, mirroring what ``mark_failed`` would have produced after two
    # earlier (non-racing) parse failures.
    store.set(path, path.stat().st_size, failure_count=2)
    baseline = store.get_record(path)
    assert baseline is not None
    assert baseline.failure_count == 2

    # --- Actor A (e.g. the live daemon watcher) and Actor B (e.g. a CLI
    # reprocess batch touching the same source path) each independently hit
    # a real parse failure on the SAME path at roughly the same time. Both
    # read the cursor record on their own connection/transaction *before*
    # either has written its increment back — this is exactly what
    # ``mark_failed`` does internally, just with the read/write pair from
    # two callers interleaved instead of one caller's pair running alone.
    actor_a_read = store.get_record(path)
    actor_b_read = store.get_record(path)
    assert actor_a_read is not None
    assert actor_b_read is not None
    assert actor_a_read.failure_count == 2
    assert actor_b_read.failure_count == 2  # both saw the same stale value

    # Actor A commits its increment first (mirrors the ``set()`` call inside
    # ``mark_failed`` using its own stale read).
    store.set(
        path,
        actor_a_read.byte_size,
        byte_offset=actor_a_read.byte_offset,
        last_complete_newline=actor_a_read.last_complete_newline,
        record_count=actor_a_read.record_count,
        parser_fingerprint=actor_a_read.parser_fingerprint,
        content_fingerprint=actor_a_read.content_fingerprint,
        tail_hash=actor_a_read.tail_hash,
        source_name=actor_a_read.source_name,
        st_dev=actor_a_read.st_dev,
        st_ino=actor_a_read.st_ino,
        mtime_ns=actor_a_read.mtime_ns,
        failure_count=actor_a_read.failure_count + 1,
        next_retry_at=None,
        excluded=bool(actor_a_read.excluded),
        allow_backward=True,
    )
    after_a = store.get_record(path)
    assert after_a is not None
    assert after_a.failure_count == 3

    # Actor B commits its increment second, computed from its OWN (now
    # stale) read taken before Actor A's write landed. The full-row upsert
    # (``ON CONFLICT(source_path) DO UPDATE SET failure_count = excluded...``)
    # has no idea a second real failure has occurred in between.
    store.set(
        path,
        actor_b_read.byte_size,
        byte_offset=actor_b_read.byte_offset,
        last_complete_newline=actor_b_read.last_complete_newline,
        record_count=actor_b_read.record_count,
        parser_fingerprint=actor_b_read.parser_fingerprint,
        content_fingerprint=actor_b_read.content_fingerprint,
        tail_hash=actor_b_read.tail_hash,
        source_name=actor_b_read.source_name,
        st_dev=actor_b_read.st_dev,
        st_ino=actor_b_read.st_ino,
        mtime_ns=actor_b_read.mtime_ns,
        failure_count=actor_b_read.failure_count + 1,
        next_retry_at=None,
        excluded=bool(actor_b_read.excluded),
        allow_backward=True,
    )

    final = store.get_record(path)
    assert final is not None
    # Two real failures occurred (baseline->A, baseline->B), so a correct
    # accounting would show failure_count == 4. The unlocked read-modify-write
    # loses Actor A's increment: Actor B's write replaces the whole row using
    # its own stale read, landing on 3.
    assert final.failure_count == 3, (
        "lost update reproduced: expected the racing writer to clobber the other's "
        "increment (3), got a count that suggests the race no longer reproduces"
    )
