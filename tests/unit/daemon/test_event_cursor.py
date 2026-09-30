"""Retention and cursor completeness of the ``daemon_events`` replay ledger.

Retention has no row count and no age (polylogue-20d.13.6): the owning process
removes a row once every live subscriber has read it and it is either a
granular topic frame or a record superseded by a newer row of its kind. The
removal is not a prefix -- the newest row of each record kind survives inside
the removed range -- so the watermark, not ``MIN(id)``, says where complete
history starts.

Anti-vacuity, executed both ways:

- make ``prune_through`` ignore live cursors and
  ``test_rows_a_live_subscriber_has_not_read_are_kept`` goes red;
- keep only the rows above the lowest live cursor regardless of kind and
  ``test_newest_record_of_each_kind_survives`` goes red;
- decide the cursor from ``MIN(id)`` instead of the watermark and
  ``test_a_cursor_below_the_watermark_is_refused_despite_interior_rows`` goes
  red (an interior newest-of-kind row makes ``MIN(id)`` look complete);
- drop the ``BEGIN``/``rollback`` pair in ``query_events_since`` and
  ``test_prune_between_range_and_page_unseen`` goes red.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from polylogue.daemon import events as events_mod
from polylogue.daemon.events import (
    EVENT_MESSAGE_APPENDED,
    EVENT_SESSION_UPDATED,
    EventCursorStatus,
    EventSubscriberRegistry,
)


@pytest.fixture
def ledger(workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the event ledger at an isolated ops database."""
    path = workspace_env["archive_root"] / "event_ledger.db"
    monkeypatch.setattr(events_mod, "_events_db_path", lambda: path)
    return path


def _seed(path: Path, kinds: list[str]) -> None:
    """Write rows of the given kinds, ids 1..n, into the production ledger schema."""
    events_mod.emit_daemon_event("bootstrap", payload={})
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM daemon_events")
        conn.executemany(
            "INSERT INTO daemon_events (id, ts_ms, kind, operation_id, payload_json) VALUES (?, ?, ?, NULL, '{}')",
            [(index + 1, 1_000 + index, kind) for index, kind in enumerate(kinds)],
        )
        conn.commit()


def _ids(path: Path) -> list[int]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return [int(row[0]) for row in conn.execute("SELECT id FROM daemon_events ORDER BY id")]


def _prune(path: Path, registry: EventSubscriberRegistry) -> int:
    with sqlite3.connect(path) as conn:
        removed = events_mod.prune_daemon_events(conn, subscribers=registry)
        conn.commit()
    return removed


def test_a_process_that_does_not_own_the_ledger_never_prunes(ledger: Path) -> None:
    """Another process cannot see the daemon's live subscribers, so it only appends."""
    _seed(ledger, [EVENT_MESSAGE_APPENDED, EVENT_MESSAGE_APPENDED, "ingestion_batch", "ingestion_batch"])
    assert _prune(ledger, EventSubscriberRegistry()) == 0
    assert _ids(ledger) == [1, 2, 3, 4]


def test_newest_record_of_each_kind_survives(ledger: Path) -> None:
    """With no live subscriber, topic frames go and each record kind keeps its newest row."""
    _seed(
        ledger,
        [
            "ingestion_batch",
            EVENT_MESSAGE_APPENDED,
            "judgment-automation",
            "ingestion_batch",
            EVENT_SESSION_UPDATED,
            "ingestion_batch",
        ],
    )
    registry = EventSubscriberRegistry()
    with registry.owning():
        removed = _prune(ledger, registry)
    assert removed == 4
    assert _ids(ledger) == [3, 6]
    assert events_mod.get_last_ingestion_batch() is not None
    assert events_mod.get_latest_daemon_event("judgment-automation") is not None


def test_rows_a_live_subscriber_has_not_read_are_kept(ledger: Path) -> None:
    _seed(ledger, [EVENT_MESSAGE_APPENDED] * 6)
    registry = EventSubscriberRegistry()
    with registry.owning():
        slow = registry.subscribe(2)
        fast = registry.subscribe(5)
        assert _prune(ledger, registry) == 2
        assert _ids(ledger) == [3, 4, 5, 6]
        slow.advance(4)
        assert _prune(ledger, registry) == 2
        assert _ids(ledger) == [5, 6]
        slow.close()
        fast.close()
        assert _prune(ledger, registry) == 2
    assert _ids(ledger) == []


def test_a_live_subscriber_reads_every_row_after_its_cursor(ledger: Path) -> None:
    """Pruning while a stream is open never shortens that stream's next page."""
    _seed(ledger, [EVENT_MESSAGE_APPENDED] * 4)
    registry = EventSubscriberRegistry()
    with registry.owning(), registry.subscribe(1):
        _prune(ledger, registry)
        page = events_mod.query_events_since(1)
    assert page.status is EventCursorStatus.OK
    assert [event["id"] for event in page.events] == [2, 3, 4]


def test_a_cursor_below_the_watermark_is_refused_despite_interior_rows(ledger: Path) -> None:
    """The newest ``ingestion_batch`` (id 1) survives at the bottom of the removed range.

    ``MIN(id)`` is then 1 and a resume from 1 would look complete, while the
    frames 2..4 it never read are gone.
    """
    _seed(ledger, ["ingestion_batch", EVENT_MESSAGE_APPENDED, EVENT_MESSAGE_APPENDED, EVENT_MESSAGE_APPENDED])
    registry = EventSubscriberRegistry()
    with registry.owning():
        _prune(ledger, registry)
    assert _ids(ledger) == [1]

    page = events_mod.query_events_since(1)
    assert page.status is EventCursorStatus.AGED_OUT
    assert page.resync is not None
    assert page.resync["payload"]["reason"] == events_mod.RESYNC_CURSOR_AGED_OUT  # type: ignore[index]
    assert page.latest_id == 4

    # A subscriber that has read through the watermark is complete.
    assert events_mod.query_events_since(4).status is EventCursorStatus.OK
    # A cursor ahead of every id the ledger ever held is a reset tier.
    reset = events_mod.query_events_since(9)
    assert reset.resync is not None
    assert reset.resync["payload"]["reason"] == events_mod.RESYNC_LEDGER_RESET  # type: ignore[index]


def test_the_production_emit_prunes_in_the_owning_process(ledger: Path) -> None:
    """The enforcement point runs inside every emit, against the process registry."""
    with events_mod.EVENT_SUBSCRIBERS.owning():
        for index in range(5):
            events_mod.emit_message_appended(session_id=f"s-{index}", source_name="codex", appended_count=1)
        events_mod.emit_daemon_event("ingestion_batch", payload={"n": 1})
        events_mod.emit_daemon_event("ingestion_batch", payload={"n": 2})
    kinds = [event["kind"] for event in events_mod.iter_daemon_events(limit=100)]
    assert kinds == ["ingestion_batch"]


def test_prune_between_range_and_page_unseen(ledger: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(ledger, ["a", "b", "c"])
    real_range = events_mod._retained_range

    def prune_after_range(conn: sqlite3.Connection) -> tuple[int | None, int, int]:
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


def test_capture_health_history_survives_supersession(ledger: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``capture-health`` lists history, so a newer report does not supersede an older one.

    polylogue-ntvf6: record-kind supersession kept only the newest
    ``browser_capture_health`` row, and ``polylogued browser capture-health``
    showed one event after several captures. Anti-vacuity: drop the kind from
    the independent history insert and the command loses reports.
    """
    import json

    from click.testing import CliRunner

    from polylogue.daemon.browser_capture import capture_health_command

    registry = EventSubscriberRegistry()
    monkeypatch.setattr(events_mod, "EVENT_SUBSCRIBERS", registry)
    with registry.owning():
        for index in range(3):
            events_mod.emit_daemon_event(
                events_mod.CAPTURE_HEALTH_EVENT_KIND,
                operation_id="extension-1",
                payload={"event": "gap", "provider": "chatgpt", "provider_session_id": f"session-{index}"},
            )
        events_mod.emit_daemon_event("ingestion_batch", payload={})
        events_mod.emit_daemon_event("ingestion_batch", payload={})

    result = CliRunner().invoke(capture_health_command, ["--format", "json"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    listed = json.loads(result.output)["events"]
    assert len(listed) == 3
    assert {event["payload"]["provider_session_id"] for event in listed} == {"session-0", "session-1", "session-2"}
    # An ordinary record kind is still superseded down to its newest row.
    with sqlite3.connect(f"file:{ledger}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM daemon_events WHERE kind = 'ingestion_batch'").fetchone()[0] == 1
        assert (
            conn.execute("SELECT COUNT(*) FROM daemon_events WHERE kind = 'browser_capture_health'").fetchone()[0] == 0
        )
        assert conn.execute("SELECT COUNT(*) FROM capture_health_history").fetchone()[0] == 3


def test_capture_history_snapshot_pages_exclude_new_reports(ledger: Path) -> None:
    """An append between real page reads cannot enter or duplicate the old snapshot."""
    ids = [
        events_mod.emit_daemon_event(events_mod.CAPTURE_HEALTH_EVENT_KIND, payload={"event": "capture_gap"})
        for _ in range(5)
    ]
    page = events_mod.capture_health_page(page_size=2)
    seen = [event["id"] for event in page["events"]]
    appended = events_mod.emit_daemon_event(events_mod.CAPTURE_HEALTH_EVENT_KIND, payload={"event": "capture_gap"})
    while page["next_cursor"] is not None:
        page = events_mod.capture_health_page(page_size=2, cursor=page["next_cursor"])
        seen.extend(event["id"] for event in page["events"])
    assert seen == list(reversed(ids))
    assert appended not in seen


def test_capture_history_refuses_cursor_after_recreated_ops(ledger: Path) -> None:
    """A fresh ops tier reusing ids must not silently continue a lost snapshot."""
    for _ in range(3):
        events_mod.emit_daemon_event(events_mod.CAPTURE_HEALTH_EVENT_KIND)
    cursor = events_mod.capture_health_page(page_size=1)["next_cursor"]
    assert cursor is not None
    # Close all production readers before simulating the disposable tier's replacement.
    ledger.unlink()
    for suffix in ("-wal", "-shm"):
        ledger.with_name(ledger.name + suffix).unlink(missing_ok=True)
    for _ in range(3):
        events_mod.emit_daemon_event(events_mod.CAPTURE_HEALTH_EVENT_KIND)
    with pytest.raises(events_mod.CaptureHistoryCursorError, match="history_cursor_reset"):
        events_mod.capture_health_page(page_size=1, cursor=cursor)


def test_capture_report_id_survives_pruning_and_idempotent_replay(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The emitter returns its report id, never a subsequent ledger MAX."""
    registry = EventSubscriberRegistry()
    monkeypatch.setattr(events_mod, "EVENT_SUBSCRIBERS", registry)
    with registry.owning():
        event_id = events_mod.emit_daemon_event(
            events_mod.CAPTURE_HEALTH_EVENT_KIND, idempotency_key="report-1", payload={"event": "capture_gap"}
        )
        other_id = events_mod.emit_daemon_event("other")
        replay_id = events_mod.emit_daemon_event(
            events_mod.CAPTURE_HEALTH_EVENT_KIND, idempotency_key="report-1", payload={"event": "capture_gap"}
        )
    assert event_id == replay_id < other_id
    page = events_mod.capture_health_page()
    assert [row["id"] for row in page["events"]] == [event_id]
    with sqlite3.connect(ledger) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM daemon_events WHERE kind = ?", (events_mod.CAPTURE_HEALTH_EVENT_KIND,)
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("cursor", ["not-base64", "e30=", "WzEsICJrIiwgMF0=", "WzEsICJrIiwgMl0="])
def test_capture_history_rejects_invalid_continuation(ledger: Path, cursor: str) -> None:
    with pytest.raises(events_mod.CaptureHistoryCursorError, match="invalid_history_cursor"):
        events_mod.capture_health_page(cursor=cursor)


def test_capture_history_cli_streams_all_pages(ledger: Path) -> None:
    import json

    from click.testing import CliRunner

    from polylogue.daemon.browser_capture import capture_health_command

    for _ in range(205):
        events_mod.emit_daemon_event(events_mod.CAPTURE_HEALTH_EVENT_KIND, payload={"event": "capture_gap"})
    result = CliRunner().invoke(capture_health_command, ["--limit", "-1", "--format", "json"], catch_exceptions=False)
    assert result.exit_code == 0
    document = json.loads(result.output)
    assert len(document["events"]) == 205
    assert document["next_cursor"] is None
    first = json.loads(
        CliRunner().invoke(capture_health_command, ["--limit", "2", "--format", "json"], catch_exceptions=False).output
    )
    second = json.loads(
        CliRunner()
        .invoke(
            capture_health_command,
            ["--limit", "2", "--cursor", first["next_cursor"], "--format", "json"],
            catch_exceptions=False,
        )
        .output
    )
    assert [row["id"] for row in first["events"] + second["events"]] == [205, 204, 203, 202]


def test_health_history_insert_failure_rolls_back_resume_frame(ledger: Path) -> None:
    """History and resume announcement cannot commit independently."""
    events_mod.emit_daemon_event("bootstrap")
    with sqlite3.connect(ledger) as conn:
        conn.execute(
            "CREATE TRIGGER refuse_health_history BEFORE INSERT ON capture_health_history BEGIN SELECT RAISE(ABORT, 'synthetic_history_failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="synthetic_history_failure"):
        events_mod.emit_daemon_event(events_mod.CAPTURE_HEALTH_EVENT_KIND)
    with sqlite3.connect(ledger) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM daemon_events WHERE kind = ?", (events_mod.CAPTURE_HEALTH_EVENT_KIND,)
            ).fetchone()[0]
            == 0
        )
        assert conn.execute("SELECT COUNT(*) FROM capture_health_history").fetchone()[0] == 0


def test_negative_event_limit_is_a_stream(ledger: Path) -> None:
    """Full ledger traversal stays an iterator rather than a lifetime-sized list."""
    from collections.abc import Iterator
    from contextlib import closing

    _seed(ledger, ["record"] * 205)
    with closing(events_mod.iter_daemon_events(limit=-1)) as events:
        assert isinstance(events, Iterator)
        assert next(events)["id"] == 205
        assert sum(1 for _ in events) == 204


def test_capture_history_cli_refuses_foreign_identity_without_writes(ledger: Path) -> None:
    from click.testing import CliRunner

    from polylogue.daemon.browser_capture import capture_health_command

    events_mod.emit_daemon_event(events_mod.CAPTURE_HEALTH_EVENT_KIND)
    with sqlite3.connect(ledger) as conn:
        conn.execute("UPDATE schema_identity SET identity = 'foreign' WHERE tier = 'ops'")
    before = ledger.read_bytes()
    result = CliRunner().invoke(capture_health_command, ["--format", "json"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "schema_skew" in result.output
    assert ledger.read_bytes() == before
