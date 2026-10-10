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
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.daemon import events as events_mod
from polylogue.daemon.events import (
    EVENT_MESSAGE_APPENDED,
    EVENT_SESSION_UPDATED,
    EventCursorStatus,
    EventSubscriberRegistry,
)


def _cursor(position: int) -> str:
    from polylogue.daemon.events import get_latest_event_cursor

    current = get_latest_event_cursor()
    assert current is not None
    return current.rsplit(":", 1)[0] + f":{position}"


@pytest.fixture
def ledger(workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the event ledger at an isolated ops database."""
    path = workspace_env["archive_root"] / "event_ledger.db"
    monkeypatch.setattr(events_mod, "_events_db_path", lambda: path)
    return path


def _seed(path: Path, kinds: list[str]) -> None:
    """Write rows of the given kinds, ids 1..n, into the production ledger schema."""
    events_mod.emit_daemon_event("bootstrap", payload={})
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("DELETE FROM daemon_events")
        conn.executemany(
            "INSERT INTO daemon_events (id, ts_ms, kind, operation_id, payload_json) VALUES (?, ?, ?, NULL, '{}')",
            [(index + 1, 1_000 + index, kind) for index, kind in enumerate(kinds)],
        )
        conn.commit()


def _ids(path: Path) -> list[int]:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        return [int(row[0]) for row in conn.execute("SELECT id FROM daemon_events ORDER BY id")]


def _prune(path: Path, registry: EventSubscriberRegistry) -> int:
    with closing(sqlite3.connect(path)) as conn, conn:
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
        slow = registry.subscribe(_cursor(2))
        fast = registry.subscribe(_cursor(5))
        assert _prune(ledger, registry) == 2
        assert _ids(ledger) == [3, 4, 5, 6]
        slow.advance(_cursor(4))
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
    with registry.owning(), registry.subscribe(_cursor(1)):
        _prune(ledger, registry)
        page = events_mod.query_events_since(_cursor(1))
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

    page = events_mod.query_events_since(_cursor(1))
    assert page.status is EventCursorStatus.AGED_OUT
    assert page.resync is not None
    assert page.resync["payload"]["reason"] == events_mod.RESYNC_CURSOR_AGED_OUT  # type: ignore[index]
    assert page.latest_id == 4

    # A subscriber that has read through the watermark is complete.
    assert events_mod.query_events_since(_cursor(4)).status is EventCursorStatus.OK
    # A cursor ahead of every id the ledger ever held is a reset tier.
    reset = events_mod.query_events_since(_cursor(9))
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
            with closing(sqlite3.connect(ledger, timeout=5.0)) as writer, writer:
                writer.execute("DELETE FROM daemon_events WHERE id <= 2")
                writer.commit()

        thread = threading.Thread(target=commit_prune)
        thread.start()
        thread.join(timeout=10.0)
        assert not thread.is_alive()
        return result

    monkeypatch.setattr(events_mod, "_retained_range", prune_after_range)
    page = events_mod.query_events_since(None)
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
    with pytest.raises(events_mod.CaptureHistoryStorageError) as failed:
        events_mod.emit_daemon_event(events_mod.CAPTURE_HEALTH_EVENT_KIND)
    assert failed.value.is_transient is False
    assert isinstance(failed.value.__cause__, sqlite3.IntegrityError)
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


def test_health_page_handles_sqlite_integer_domain_edges(ledger: Path) -> None:
    event_id = events_mod.emit_daemon_event(events_mod.CAPTURE_HEALTH_EVENT_KIND)
    maximum_id = 2**63 - 1
    with sqlite3.connect(ledger) as conn:
        conn.execute("UPDATE capture_health_history SET id = ? WHERE id = ?", (maximum_id, event_id))
    page = events_mod.capture_health_page(page_size=2**100)
    assert [row["id"] for row in page["events"]] == [maximum_id]
    assert page["next_cursor"] is None
    import base64
    import json

    cursor = base64.urlsafe_b64encode(json.dumps([2**100, "synthetic", 1]).encode()).decode()
    with pytest.raises(events_mod.CaptureHistoryCursorError) as refused:
        events_mod.capture_health_page(cursor=cursor)
    assert str(refused.value) == "invalid_history_cursor"


def test_oversized_health_page_request_preserves_full_traversal(ledger: Path) -> None:
    ids = [events_mod.emit_daemon_event(events_mod.CAPTURE_HEALTH_EVENT_KIND) for _ in range(205)]
    page = events_mod.capture_health_page(page_size=2**100)
    assert len(page["events"]) == 100
    seen = [row["id"] for row in page["events"]]
    while page["next_cursor"] is not None:
        page = events_mod.capture_health_page(page_size=2**100, cursor=page["next_cursor"])
        assert len(page["events"]) <= 100
        seen.extend(row["id"] for row in page["events"])
    assert seen == list(reversed(ids))


@pytest.mark.parametrize("failure", ["storage", "reset", "cancel"])
def test_capture_history_json_later_failure_publishes_nothing_and_cleans_scratch(
    ledger: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import os
    import stat
    import tempfile
    from typing import IO

    from click.testing import CliRunner

    from polylogue.daemon import browser_capture as command_mod

    for _ in range(205):
        events_mod.emit_daemon_event(events_mod.CAPTURE_HEALTH_EVENT_KIND)
    opened: list[IO[str]] = []
    original_temporary_file = tempfile.TemporaryFile

    def private_file(*args: object, **kwargs: object) -> IO[str]:
        handle = original_temporary_file(mode="w+", encoding="utf-8", dir=tmp_path)
        assert stat.S_IMODE(os.fstat(handle.fileno()).st_mode) == 0o600
        opened.append(handle)
        return handle

    monkeypatch.setattr(tempfile, "TemporaryFile", private_file)
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    original_open = open_readonly_connection
    reads = 0
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    def failing_later_read(path: str | Path, *, tier: ArchiveTier | None = None) -> sqlite3.Connection:
        nonlocal reads
        reads += 1
        if reads == 2:
            if failure == "storage":
                raise sqlite3.OperationalError("synthetic later read lock")
            if failure == "cancel":
                raise KeyboardInterrupt
            ledger.unlink()
            for suffix in ("-wal", "-shm"):
                ledger.with_name(ledger.name + suffix).unlink(missing_ok=True)
            events_mod.emit_daemon_event(events_mod.CAPTURE_HEALTH_EVENT_KIND)
        return original_open(path, tier=tier)

    monkeypatch.setattr(events_mod, "open_readonly_connection", failing_later_read)
    result = CliRunner().invoke(command_mod.capture_health_command, ["--limit", "-1", "--format", "json"])
    assert result.exit_code != 0
    assert result.stdout == ""
    assert reads == 2
    assert len(opened) == 1 and opened[0].closed
    if failure != "cancel":
        assert ("capture_history_unavailable" if failure == "storage" else "history_cursor_reset") in result.stderr


@pytest.mark.parametrize("replacement_count", [0, 2, 4, 6])
def test_replacement_lifetime_refuses_reused_event_positions(
    ledger: Path, tmp_path: Path, replacement_count: int
) -> None:
    """Equal/larger replacement IDs are not evidence of continuity."""
    import os

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    _seed(ledger, ["old"] * 4)
    original = events_mod.query_events_since(None)
    cursor = str(original.events[-1]["cursor"])
    events_mod.emit_daemon_event("old-append")
    continued = events_mod.query_events_since(cursor)
    assert [event["kind"] for event in continued.events] == ["old-append"]
    assert continued.resync is None
    replacement = tmp_path / "replacement.db"
    initialize_archive_database(replacement, ArchiveTier.OPS)
    with closing(sqlite3.connect(replacement)) as conn, conn:
        conn.executemany(
            "INSERT INTO daemon_events(ts_ms,kind,payload_json) VALUES (1000,'new','{}')",
            [()] * replacement_count,
        )
    os.replace(replacement, ledger)
    page = events_mod.query_events_since(cursor)
    assert page.status is EventCursorStatus.AGED_OUT
    assert page.events == ()
    assert page.resync is not None
    payload = page.resync["payload"]
    assert isinstance(payload, dict)
    assert payload["reason"] == "ledger_reset"
    assert page.latest_cursor != cursor
    assert len(events_mod.query_events_since(None).events) == replacement_count
    assert events_mod.query_events_since(page.latest_cursor).status is EventCursorStatus.OK


def test_old_lifetime_subscriber_cannot_prune_a_replacement(ledger: Path, tmp_path: Path) -> None:
    import os

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    _seed(ledger, [EVENT_MESSAGE_APPENDED] * 4)
    old_cursor = _cursor(4)
    replacement = tmp_path / "replacement.db"
    initialize_archive_database(replacement, ArchiveTier.OPS)
    with closing(sqlite3.connect(replacement)) as conn, conn:
        conn.executemany(
            "INSERT INTO daemon_events(ts_ms,kind,payload_json) VALUES (1000,?,'{}')",
            [(EVENT_MESSAGE_APPENDED,)] * 6,
        )
    os.replace(replacement, ledger)
    registry = EventSubscriberRegistry()
    with registry.owning(), registry.subscribe(old_cursor) as subscription:
        assert _prune(ledger, registry) == 0
        assert _ids(ledger) == [1, 2, 3, 4, 5, 6]
        subscription.advance(_cursor(4))
        assert _prune(ledger, registry) == 4


@pytest.mark.uses_real_clock("production HTTP listener and SSE physical completion")
def test_real_http_replacement_resync_and_native_cursor_reconnect(ledger: Path, tmp_path: Path) -> None:
    import json
    import os
    from http.client import HTTPConnection
    from threading import Thread
    from urllib.parse import urlencode

    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    _seed(ledger, ["old"] * 4)
    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    server.auth_token = ""
    server.api_host = "127.0.0.1"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
        with closing(HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)) as client:
            client.request("GET", path, headers=headers or {})
            response = client.getresponse()
            return response.status, response.read()

    try:
        status, body = request("/api/events?poll=1")
        assert status == 200
        initial = json.loads(body)
        cursor = initial["last_event_id"]
        assert cursor == initial["events"][-1]["cursor"]
        for count in (2, 4, 6):
            replacement = tmp_path / f"replacement-{count}.db"
            initialize_archive_database(replacement, ArchiveTier.OPS)
            with closing(sqlite3.connect(replacement)) as conn, conn:
                conn.executemany(
                    "INSERT INTO daemon_events(ts_ms,kind,payload_json) VALUES (1000,'new','{}')",
                    [()] * count,
                )
            os.replace(replacement, ledger)
            status, body = request("/api/events?" + urlencode({"poll": 1, "since": cursor}))
            page = json.loads(body)
            assert status == 200 and page["outcome"] == "degraded"
            assert page["resync_reason"] == "ledger_reset"
            fresh_status, fresh_body = request("/api/events?poll=1")
            assert fresh_status == 200
            assert [event["id"] for event in json.loads(fresh_body)["events"]] == list(range(1, count + 1))
            status, stream = request("/api/events?max_seconds=1", {"Last-Event-ID": cursor})
            assert status == 200 and b"event: snapshot\n" in stream
            assert b'"ledger_reset"' in stream
            assert ("id: " + page["last_event_id"]).encode() in stream
            status, body = request("/api/events?" + urlencode({"poll": 1, "since": page["last_event_id"]}))
            assert status == 200 and json.loads(body)["events"] == []
        status, body = request("/api/events?poll=1&since=4")
        assert status == 400 and json.loads(body)["error"] == "invalid_event_cursor"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        assert not thread.is_alive()


def test_resumed_empty_ledger_cursor_detects_later_pruning(ledger: Path) -> None:
    _seed(ledger, [])
    cursor = _cursor(0)
    with events_mod.EVENT_SUBSCRIBERS.owning():
        events_mod.emit_message_appended(session_id="neutral", source_name="codex", appended_count=1)
    page = events_mod.query_events_since(cursor)
    assert page.status is EventCursorStatus.AGED_OUT
    assert page.resync is not None
    payload = page.resync["payload"]
    assert isinstance(payload, dict)
    assert payload["reason"] == "cursor_aged_out"
    assert events_mod.query_events_since(None).status is EventCursorStatus.OK


def test_bootstrap_clone_has_its_own_lifetime_and_reopen_preserves_it(ledger: Path, tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    _seed(ledger, ["neutral"])
    original = _cursor(1)
    initialize_archive_database(ledger, ArchiveTier.OPS)
    assert _cursor(1) == original
    copied = tmp_path / "cloned.db"
    initialize_archive_database(copied, ArchiveTier.OPS)
    with closing(sqlite3.connect(copied)) as conn:
        lifetime = conn.execute("SELECT lifetime FROM daemon_event_retention WHERE ledger='daemon_events'").fetchone()[
            0
        ]
    assert lifetime != original.partition(":")[0]
