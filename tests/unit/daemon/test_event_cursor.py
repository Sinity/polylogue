"""Cursor completeness of the ``daemon_events`` replay ledger.

``_cursor_refusal_reason`` decides whether a resuming subscriber's cursor is
still honourable from ``MIN(id)`` alone. Two things must hold for that single
number to be a sound completeness proof:

- retention may only remove an ``id`` prefix, never an interior row; and
- the range check and the page read must observe the same snapshot.

Break either and the daemon answers a resumable subscriber with an ``OK`` page
that is silently missing events -- the exact failure the ``AGED_OUT`` refusal
exists to make visible.

Anti-vacuity, executed both ways:

- restore ``DELETE FROM daemon_events WHERE ts_ms < ?`` and the two
  out-of-order tests go red, while ``test_expired_prefix_is_removed`` and
  ``test_expired_ledger_is_emptied`` stay green, so a blanket no-op cannot
  pass either;
- drop the ``BEGIN``/``rollback`` pair in ``query_events_since`` and
  ``test_prune_between_range_and_page_unseen`` goes red.

The retention fixtures carry a deliberate ``ts_ms``/``id`` skew. On a
uniformly increasing ledger the old and new predicates agree exactly, and no
assertion over such a fixture could separate them.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from polylogue.daemon import events as events_mod
from polylogue.daemon.events import DaemonEventRetention, EventCursorStatus


@pytest.fixture
def ledger(workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the event ledger at an isolated ops database."""
    path = workspace_env["archive_root"] / "event_ledger.db"
    monkeypatch.setattr(events_mod, "_events_db_path", lambda: path)
    return path


def _seed(path: Path, rows: list[tuple[int, str]]) -> None:
    """Write exact ``(ts_ms, kind)`` rows into the production ledger schema."""
    events_mod.emit_daemon_event("bootstrap", payload={})
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM daemon_events")
        conn.executemany(
            "INSERT INTO daemon_events (id, ts_ms, kind, operation_id, payload_json) VALUES (?, ?, ?, NULL, '{}')",
            [(index + 1, ts_ms, kind) for index, (ts_ms, kind) in enumerate(rows)],
        )
        conn.commit()


def _ids(path: Path) -> list[int]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return [int(row[0]) for row in conn.execute("SELECT id FROM daemon_events ORDER BY id")]


def _prune(path: Path, retention: DaemonEventRetention, *, now_ms: int) -> int:
    with sqlite3.connect(path) as conn:
        removed = events_mod.prune_daemon_events(conn, retention, now_ms=now_ms)
        conn.commit()
    return removed


#: id 1 at T, id 2 at T-100s, id 3 at T+1s -- valid, non-monotonic timestamps.
_SKEWED = [(1_000_000, "first"), (900_000, "out-of-order"), (1_001_000, "third")]


def test_out_of_order_row_survives_age_prune(ledger: Path) -> None:
    _seed(ledger, _SKEWED)
    removed = _prune(ledger, DaemonEventRetention(max_age_ms=50_000), now_ms=1_001_000)
    assert removed == 0
    assert _ids(ledger) == [1, 2, 3]


def test_cursor_page_keeps_every_event(ledger: Path) -> None:
    _seed(ledger, _SKEWED)
    _prune(ledger, DaemonEventRetention(max_age_ms=50_000), now_ms=1_001_000)
    page = events_mod.query_events_since(1)
    assert page.status is EventCursorStatus.OK
    assert [event["kind"] for event in page.events] == ["out-of-order", "third"]


def test_expired_prefix_is_removed(ledger: Path) -> None:
    """Opposite direction: age retention must still trim a genuine prefix."""
    _seed(ledger, [(100_000, "expired"), (200_000, "expired"), (1_000_000, "kept")])
    removed = _prune(ledger, DaemonEventRetention(max_age_ms=50_000), now_ms=1_000_000)
    assert removed == 2
    assert _ids(ledger) == [3]


def test_expired_ledger_is_emptied(ledger: Path) -> None:
    _seed(ledger, [(100_000, "expired"), (200_000, "expired")])
    removed = _prune(ledger, DaemonEventRetention(max_age_ms=50_000), now_ms=9_000_000)
    assert removed == 2
    assert _ids(ledger) == []


def test_prune_between_range_and_page_unseen(ledger: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(ledger, [(1_000, "a"), (2_000, "b"), (3_000, "c")])
    real_range = events_mod._retained_range

    def prune_after_range(conn: sqlite3.Connection) -> tuple[int | None, int]:
        result = real_range(conn)

        # A separate thread: SQLite will not let this reader's own connection
        # write while its read transaction is open, so the interleave has to
        # arrive from outside.
        def commit_prune() -> None:
            with sqlite3.connect(ledger, timeout=5.0) as writer:
                writer.execute("DELETE FROM daemon_events WHERE id <= 2")
                writer.commit()

        thread = threading.Thread(target=commit_prune)
        thread.start()
        thread.join(timeout=10.0)
        assert not thread.is_alive()
        return result

    monkeypatch.setattr(events_mod, "_retained_range", prune_after_range)
    page = events_mod.query_events_since(0)
    assert page.status is EventCursorStatus.OK
    assert [event["kind"] for event in page.events] == ["a", "b", "c"]
    # The prune really did commit; the reader simply did not observe it.
    assert _ids(ledger) == [3]
