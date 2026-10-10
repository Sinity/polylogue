"""Daemon event ledger backed by archive ops state."""

from __future__ import annotations

import base64
import json
import sqlite3
import threading
import uuid
from collections.abc import Generator, Iterator
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, TypedDict

from polylogue.core.errors import DatabaseError
from polylogue.operations.judgment_scheduler import (
    ArchiveJudgmentSchedulerReceipt,
    record_judgment_scheduler_receipt,
)
from polylogue.paths import archive_root
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_daemon_connection, open_readonly_connection

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _events_db_path() -> Path:
    """Return the path to the daemon events SQLite database."""
    return archive_root() / "ops.db"


#: Ops-tier admission runs once per (process, path):
#: repeated tier admission used to put schema work inside every emitter.
#: Startup reconverges a stale disposable tier under exclusive ownership;
#: emitters admit the current identity and never patch an existing schema.
_CONVERGED_EVENT_DBS: set[Path] = set()


def _ensure_events_db(path: Path | None = None) -> sqlite3.Connection:
    """Open and initialize the daemon events database for an emitter."""
    path = _events_db_path() if path is None else path
    if path not in _CONVERGED_EVENT_DBS or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        initialize_archive_database(path, ArchiveTier.OPS)
        _CONVERGED_EVENT_DBS.add(path)
    # The ops tier's DDL owns ``daemon_events`` and its idempotency index; an
    # emitter no longer restates or alters it (polylogue-l91i8).
    return open_daemon_connection(path, archive_root=path.parent)


def _open_events_reader(path: Path | None = None) -> sqlite3.Connection | None:
    """Open the existing event ledger read-only, or return ``None``.

    Status, polling, and SSE reads must not turn observation into an ops-tier
    write. A missing file or a pre-event-schema ops database therefore has the
    same documented empty-ledger result without directory creation, tier
    initialization, DDL, or write-profile pragmas.
    """
    path = _events_db_path() if path is None else path
    if not path.is_file():
        return None
    try:
        # The event ledger is diagnostic evidence; read it even when the tier's
        # schema version is not the runtime's.
        conn = open_readonly_connection(path, validate_schema=False)
    except sqlite3.OperationalError:
        if not path.is_file():
            return None
        raise
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'daemon_events' LIMIT 1"
        ).fetchone()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(daemon_event_retention)")}
    except BaseException:
        conn.close()
        raise
    if exists is None or "lifetime" not in columns:
        conn.close()
        return None
    return conn


def current_epoch_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _iso_from_ms(value: object) -> str:
    if isinstance(value, int):
        resolved = value
    elif isinstance(value, str | bytes | bytearray):
        resolved = int(value)
    else:
        resolved = int(str(value))
    return datetime.fromtimestamp(resolved / 1000, tz=UTC).isoformat()


def parse_event_cursor(cursor: str | None) -> tuple[str | None, int]:
    """Decode the opaque ledger lifetime and position; absence starts a new walk."""
    if cursor is None:
        return None, 0
    if not isinstance(cursor, str):
        raise ValueError("invalid_event_cursor")
    lifetime, separator, position = cursor.partition(":")
    if (
        not separator
        or len(lifetime) != 32
        or any(character not in "0123456789abcdef" for character in lifetime)
        or not position.isascii()
        or not position.isdecimal()
        or len(position) > 19
    ):
        raise ValueError("invalid_event_cursor")
    ordinal = int(position)
    if str(ordinal) != position or ordinal > 2**63 - 1:
        raise ValueError("invalid_event_cursor")
    return lifetime, ordinal


def _ledger_lifetime(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT lifetime FROM daemon_event_retention WHERE ledger = ?", (_LEDGER_NAME,)).fetchone()
    if row is None:
        raise DatabaseError("event ledger has no lifetime identity")
    return str(row[0])


class EventSubscription:
    """One live subscriber's position in the ledger, held while its stream is open."""

    __slots__ = ("_registry", "_token")

    def __init__(self, registry: EventSubscriberRegistry, token: int) -> None:
        self._registry = registry
        self._token = token

    def advance(self, cursor: str | None) -> None:
        """Record that the subscriber has now read every row through ``cursor``."""
        self._registry._advance(self._token, cursor)

    def close(self) -> None:
        self._registry._close(self._token)

    def __enter__(self) -> EventSubscription:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class EventSubscriberRegistry:
    """The live subscribers of the ``daemon_events`` ledger, and who may prune it.

    The ledger is a resume buffer, so what it must keep is decided by who is
    reading it, not by a row count or an age (polylogue-20d.13.6). A subscriber
    is *live* while its SSE stream is open; only the daemon's HTTP server
    serves those streams, so the daemon process is the one place that knows
    every live cursor. It claims the ledger with :meth:`owning`, and only the
    owning process prunes: an emitter in any other process cannot see the
    daemon's subscribers and appends without pruning.

    A subscriber that disconnects is not tracked afterwards. When it resumes
    with a ``Last-Event-ID`` below what retention removed, it gets the typed
    ``aged_out`` resync and refetches current state, which the rows it missed
    only ever announced.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cursors: dict[int, tuple[str | None, int]] = {}
        self._next_token = 0
        self._owners = 0

    @contextmanager
    def owning(self) -> Iterator[None]:
        """Make this process the ledger's pruning owner for the block's lifetime."""
        with self._lock:
            self._owners += 1
        try:
            yield
        finally:
            with self._lock:
                self._owners -= 1

    def subscribe(self, cursor: str | None) -> EventSubscription:
        """Register a live subscriber that has read every row through ``cursor``."""
        with self._lock:
            token = self._next_token
            self._next_token += 1
            self._cursors[token] = parse_event_cursor(cursor)
        return EventSubscription(self, token)

    def _advance(self, token: int, cursor: str | None) -> None:
        with self._lock:
            if token in self._cursors:
                self._cursors[token] = parse_event_cursor(cursor)

    def _close(self, token: int) -> None:
        with self._lock:
            self._cursors.pop(token, None)

    def prune_through(self, latest_id: int, lifetime: str) -> int | None:
        """Return the highest id every live subscriber has read, or ``None`` when this process may not prune.

        With no live subscriber every row is read, so the answer is ``latest_id``.
        """
        with self._lock:
            if self._owners <= 0:
                return None
            if not self._cursors:
                return latest_id
            return min(
                latest_id, min(position if owner == lifetime else 0 for owner, position in self._cursors.values())
            )


EVENT_SUBSCRIBERS = EventSubscriberRegistry()
"""The process's ledger subscriber registry; the daemon owns it for its lifetime."""

_LEDGER_NAME = "daemon_events"


def _pruned_through(conn: sqlite3.Connection) -> int:
    """The highest id retention has removed, 0 when it has removed nothing."""
    row = conn.execute(
        "SELECT pruned_through_id FROM daemon_event_retention WHERE ledger = ?", (_LEDGER_NAME,)
    ).fetchone()
    return 0 if row is None else int(row[0])


def prune_daemon_events(
    conn: sqlite3.Connection,
    *,
    subscribers: EventSubscriberRegistry | None = None,
) -> int:
    """Remove every row no live subscriber and no reader still needs; return how many.

    The single enforcement point, run inside every emit's transaction. A row
    at or below the lowest live subscriber cursor is removed when it is either

    - a granular topic frame (:data:`GRANULAR_EVENT_KINDS`): it announces a
      change whose state is committed in the tier its spec names, so a
      subscriber that missed it loses nothing a resync does not return; or
    - a record superseded by a newer row of the same kind: the in-process
      readers of record kinds (status's last ingestion batch, the judgment
      scheduler's latest receipt) read the newest row of the kind, and the
      judgment receipts also have their typed table. Capture-health frames
      announce reports in the independently retained history table and are
      pruned after consumption, just like granular frames.

    So the ledger holds what live subscribers have not read, the newest row
    of each record kind; nothing else grows with
    time or event volume. The
    highest removed id is kept as the ledger's watermark: removal is not a
    prefix any more, and :func:`query_events_since` refuses a cursor below the
    watermark rather than trusting ``MIN(id)``.
    """
    registry = EVENT_SUBSCRIBERS if subscribers is None else subscribers
    latest_row = conn.execute("SELECT MAX(id) FROM daemon_events").fetchone()
    if latest_row is None or latest_row[0] is None:
        return 0
    through = registry.prune_through(int(latest_row[0]), _ledger_lifetime(conn))
    if through is None or through <= 0:
        return 0
    granular = sorted(GRANULAR_EVENT_KINDS | {CAPTURE_HEALTH_EVENT_KIND})
    granular_placeholders = ",".join("?" for _ in granular)
    removed = 0
    watermark = 0
    for row in conn.execute(
        f"""
            DELETE FROM daemon_events
            WHERE id <= ?
              AND (
                kind IN ({granular_placeholders})
                OR (
                  id < (SELECT MAX(newer.id) FROM daemon_events AS newer WHERE newer.kind = daemon_events.kind)
                )
              )
            RETURNING id
            """,
        (through, *granular),
    ):
        removed += 1
        watermark = max(watermark, int(row[0]))
    if removed:
        conn.execute(
            """
            INSERT INTO daemon_event_retention (ledger, pruned_through_id) VALUES (?, ?)
            ON CONFLICT(ledger) DO UPDATE SET
                pruned_through_id = MAX(pruned_through_id, excluded.pruned_through_id)
            """,
            (_LEDGER_NAME, watermark),
        )
    return removed


@dataclass(frozen=True, slots=True)
class DaemonEventRecord:
    """One event row for :func:`emit_daemon_events`."""

    kind: str
    payload: dict[str, object]
    operation_id: str | None = None
    idempotency_key: str | None = None


def emit_daemon_events(
    records: Sequence[DaemonEventRecord],
    *,
    archive_root_path: Path | None = None,
    observed_at_ms: int | None = None,
) -> None:
    """Append several events to the ledger in one connection and one commit.

    A live batch announces its summary plus one event per touched session.
    Emitting those one at a time opened the ledger, ran its DDL, pruned and
    committed once per event: two per ingested session, on the writer. The
    rows, their order and the retention rule are the same; they land together.
    """
    if not records:
        return
    conn = _ensure_events_db() if archive_root_path is None else _ensure_events_db(archive_root_path / "ops.db")
    try:
        ts_ms = current_epoch_ms() if observed_at_ms is None else observed_at_ms
        for record in records:
            _insert_daemon_event(conn, record.kind, ts_ms, record.operation_id, record.idempotency_key, record.payload)
        prune_daemon_events(conn)
        conn.commit()
    finally:
        conn.close()


class CaptureHistoryStorageError(DatabaseError):
    """A failed history transaction/read, never a fabricated empty page."""

    def __init__(self, cause: sqlite3.Error | OSError) -> None:
        error_code = getattr(cause, "sqlite_errorcode", None)
        transient_codes = {
            sqlite3.SQLITE_BUSY,
            sqlite3.SQLITE_LOCKED,
            sqlite3.SQLITE_IOERR,
            sqlite3.SQLITE_CANTOPEN,
            sqlite3.SQLITE_FULL,
            sqlite3.SQLITE_PROTOCOL,
            sqlite3.SQLITE_READONLY,
        }
        self.is_transient = isinstance(cause, OSError) or (
            isinstance(cause, sqlite3.OperationalError) if error_code is None else error_code & 0xFF in transient_codes
        )
        self.code = "capture_history_unavailable" if self.is_transient else "capture_history_storage_failed"
        self.http_status_code = (
            HTTPStatus.SERVICE_UNAVAILABLE if self.is_transient else HTTPStatus.INTERNAL_SERVER_ERROR
        )
        super().__init__(self.code)


@contextmanager
def _capture_history_storage(*, enabled: bool = True) -> Iterator[None]:
    """Classify storage faults at the history owner and preserve the original cause."""
    try:
        yield
    except (sqlite3.Error, OSError) as exc:
        if not enabled:
            raise
        raise CaptureHistoryStorageError(exc) from exc


def _insert_daemon_event(
    conn: sqlite3.Connection,
    kind: str,
    ts_ms: int,
    operation_id: str | None,
    idempotency_key: str | None,
    payload: dict[str, object],
) -> int:
    with _capture_history_storage(enabled=kind == CAPTURE_HEALTH_EVENT_KIND):
        # Health history outlives consumed resume frames, including replay deduplication.
        if kind == CAPTURE_HEALTH_EVENT_KIND and idempotency_key is not None:
            prior = conn.execute(
                "SELECT id FROM capture_health_history WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if prior is not None:
                return int(prior[0])
        payload_json = json.dumps(payload)
        inserted = conn.execute(
            "INSERT INTO daemon_events (ts_ms, kind, operation_id, idempotency_key, payload_json) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(kind, idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING RETURNING id",
            (ts_ms, kind, operation_id, idempotency_key, payload_json),
        ).fetchone()
        if inserted is None:
            prior = conn.execute(
                "SELECT id FROM daemon_events WHERE kind = ? AND idempotency_key = ?", (kind, idempotency_key)
            ).fetchone()
            assert prior is not None
            return int(prior[0])
        event_id = int(inserted[0])
        if kind == CAPTURE_HEALTH_EVENT_KIND:
            conn.execute(
                "INSERT INTO capture_health_history (id, report_key, ts_ms, operation_id, idempotency_key, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, uuid.uuid4().hex, ts_ms, operation_id, idempotency_key, payload_json),
            )
        return event_id


def emit_daemon_event(
    kind: str,
    *,
    operation_id: str | None = None,
    idempotency_key: str | None = None,
    payload: dict[str, object] | None = None,
    archive_root_path: Path | None = None,
    observed_at_ms: int | None = None,
) -> int:
    """Atomically emit a resume event and its history report; return its actual id."""
    with _capture_history_storage(enabled=kind == CAPTURE_HEALTH_EVENT_KIND):
        conn = _ensure_events_db() if archive_root_path is None else _ensure_events_db(archive_root_path / "ops.db")
        try:
            if kind == "judgment-automation":
                receipt_payload = payload or {}
                typed_operation_id = operation_id or f"judgment-automation:{uuid.uuid4().hex}"
                counters = {
                    name: receipt_payload.get(name, 0)
                    for name in ("considered", "accepted", "rejected", "escalated", "idempotent", "failed")
                }
                with suppress(ValueError):
                    record_judgment_scheduler_receipt(
                        conn,
                        ArchiveJudgmentSchedulerReceipt(
                            operation_id=typed_operation_id,
                            observed_at_ms=current_epoch_ms() if observed_at_ms is None else observed_at_ms,
                            status=str(receipt_payload.get("status", "")),
                            reason=str(receipt_payload.get("reason", "")),
                            retryable=receipt_payload.get("retryable", False),  # type: ignore[arg-type]
                            retry_route=str(receipt_payload.get("retry_route", "")),
                            batch_limit=receipt_payload.get("batch_limit", 0),  # type: ignore[arg-type]
                            considered=counters["considered"],  # type: ignore[arg-type]
                            accepted=counters["accepted"],  # type: ignore[arg-type]
                            rejected=counters["rejected"],  # type: ignore[arg-type]
                            escalated=counters["escalated"],  # type: ignore[arg-type]
                            idempotent=counters["idempotent"],  # type: ignore[arg-type]
                            failed=counters["failed"],  # type: ignore[arg-type]
                            receipt_persistence_degraded=receipt_payload.get("receipt_persistence_degraded", False),  # type: ignore[arg-type]
                            receipt_persistence_recovered=receipt_payload.get("receipt_persistence_recovered", False),  # type: ignore[arg-type]
                        ),
                    )
            event_id = _insert_daemon_event(
                conn,
                kind,
                current_epoch_ms() if observed_at_ms is None else observed_at_ms,
                operation_id,
                idempotency_key,
                payload or {},
            )
            prune_daemon_events(conn)
            conn.commit()
            return event_id
        finally:
            conn.close()


def get_latest_daemon_event(
    kind: str,
    *,
    operation_id: str | None = None,
    archive_root_path: Path | None = None,
) -> dict[str, object] | None:
    """Return the latest structured event for a kind and optional operation."""

    conn = _open_events_reader() if archive_root_path is None else _open_events_reader(archive_root_path / "ops.db")
    if conn is None:
        return None
    try:
        if operation_id is None:
            row = conn.execute(
                """
                SELECT id, ts_ms, kind, operation_id, payload_json
                FROM daemon_events
                WHERE kind = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (kind,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT id, ts_ms, kind, operation_id, payload_json
                FROM daemon_events
                WHERE kind = ? AND operation_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (kind, operation_id),
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[4])
        except (TypeError, ValueError):
            payload = {}
        return {
            "id": row[0],
            "ts_ms": row[1],
            "kind": row[2],
            "operation_id": row[3],
            "payload": payload if isinstance(payload, dict) else {},
        }
    finally:
        conn.close()


def iter_daemon_events(
    *,
    kind: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> Generator[dict[str, object], None, None]:
    """Stream recent resume events; close the iterator if stopping before exhaustion.

    Negative SQL limits preserve full-history traversal without materialization.
    Capture reports use their independent history page owner instead.
    """
    conn = _open_events_reader()
    if conn is None:
        return
    try:
        if kind:
            rows = conn.execute(
                "SELECT id, ts_ms, kind, operation_id, payload_json FROM daemon_events WHERE kind = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (kind, limit, offset),
            )
        else:
            rows = conn.execute(
                "SELECT id, ts_ms, kind, operation_id, payload_json FROM daemon_events ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        for row in rows:
            yield {
                "id": row[0],
                "ts": _iso_from_ms(row[1]),
                "kind": row[2],
                "operation_id": row[3],
                "payload": json.loads(row[4]),
            }
    finally:
        conn.close()


CAPTURE_HISTORY_PAGE_ROWS = 100
"""Maximum reports in one page; continuations preserve the complete history."""


class CaptureHealthPage(TypedDict):
    events: list[dict[str, object]]
    next_cursor: str | None


class CaptureHistoryCursorError(ValueError):
    """A malformed continuation or a snapshot lost with the disposable ops tier."""


def capture_health_page(*, page_size: int = CAPTURE_HISTORY_PAGE_ROWS, cursor: str | None = None) -> CaptureHealthPage:
    """Read one newest-first keyset page from an immutable history snapshot.

    The anchor's random report key distinguishes a replaced ops tier even when
    SQLite has reused every numeric id. New reports never enter an old snapshot.
    """
    if page_size <= 0:
        raise CaptureHistoryCursorError("invalid_history_page_size")
    anchor: int | None = None
    before: int | None = None
    report_key: str | None = None
    if cursor is not None:
        try:
            decoded = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True))
            anchor, report_key, before = decoded
            if (
                type(anchor) is not int
                or type(before) is not int
                or not isinstance(report_key, str)
                or not (0 < before <= anchor <= 2**63 - 1)
            ):
                raise ValueError
        except (ValueError, TypeError, UnicodeError):
            raise CaptureHistoryCursorError("invalid_history_cursor") from None
    with _capture_history_storage():
        path = _events_db_path()
        try:
            path.stat()
        except FileNotFoundError:
            if cursor is not None:
                raise CaptureHistoryCursorError("history_cursor_reset") from None
            return {"events": [], "next_cursor": None}
        conn = open_readonly_connection(path, tier=ArchiveTier.OPS)
        try:
            conn.execute("BEGIN")
            if anchor is None:
                head = conn.execute(
                    "SELECT id, report_key FROM capture_health_history ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if head is None:
                    return {"events": [], "next_cursor": None}
                anchor, report_key = int(head[0]), str(head[1])
                before = None
            elif (
                conn.execute(
                    "SELECT 1 FROM capture_health_history WHERE id = ? AND report_key = ?", (anchor, report_key)
                ).fetchone()
                is None
            ):
                raise CaptureHistoryCursorError("history_cursor_reset")
            rows = conn.execute(
                "SELECT id, ts_ms, operation_id, payload_json FROM capture_health_history WHERE id <= ? ORDER BY id DESC LIMIT ?",
                (anchor if before is None else min(anchor, before - 1), min(page_size, CAPTURE_HISTORY_PAGE_ROWS)),
            )
            events = [
                {
                    "id": row[0],
                    "ts": _iso_from_ms(row[1]),
                    "kind": CAPTURE_HEALTH_EVENT_KIND,
                    "operation_id": row[2],
                    "payload": json.loads(row[3]),
                }
                for row in rows
            ]
            next_cursor = None
            if events:
                last_id = events[-1]["id"]
                if (
                    conn.execute("SELECT 1 FROM capture_health_history WHERE id < ? LIMIT 1", (last_id,)).fetchone()
                    is not None
                ):
                    next_cursor = base64.urlsafe_b64encode(json.dumps([anchor, report_key, last_id]).encode()).decode()
            return {"events": events, "next_cursor": next_cursor}
        finally:
            try:
                conn.rollback()
            finally:
                conn.close()


class EventCursorStatus(str, Enum):
    """Whether a resuming subscriber's cursor can still be honoured."""

    OK = "ok"
    """The requested cursor is inside the retained range; ``events`` is the
    complete answer up to ``limit``."""

    AGED_OUT = "aged-out"
    """The requested cursor names history the ledger no longer retains. The
    answer is a resync envelope, never a short or empty page: an empty page
    reads to a subscriber as "nothing happened", which is precisely the silent
    loss this refusal exists to prevent."""


RESYNC_CURSOR_AGED_OUT = "cursor_aged_out"
"""Rows between the cursor and the retained minimum were pruned."""

RESYNC_LEDGER_RESET = "ledger_reset"
"""The cursor is ahead of every retained row: the disposable ops tier holding
the ledger was reset or replaced under the subscriber."""


def build_snapshot_envelope(
    *,
    event_id: int,
    ts: str,
    event_count: int,
    first_event_id: int | None,
    last_event_id: int | None,
    kind_counts: dict[str, int],
    resync_reason: str | None = None,
    requested_since: str | None = None,
    cursor: str | None = None,
) -> dict[str, object]:
    """Build the ``snapshot`` envelope shared by coalescing and resync.

    Backpressure coalescing and an aged-out cursor both tell a subscriber the
    same thing -- "stop animating row deltas and refetch your materialized
    view" -- so they use one envelope shape rather than two signals the client
    has to learn separately. ``resync_reason`` is what distinguishes them.
    """
    payload: dict[str, object] = {
        "event_count": event_count,
        "first_event_id": first_event_id,
        "last_event_id": last_event_id,
        "kind_counts": kind_counts,
        "coalesced": True,
    }
    if resync_reason is not None:
        payload["resync"] = True
        payload["reason"] = resync_reason
        payload["requested_since"] = requested_since
    return {
        "id": event_id,
        "cursor": cursor,
        "ts": ts,
        "kind": "snapshot",
        "operation_id": None,
        "payload": payload,
    }


@dataclass(frozen=True, slots=True)
class DaemonEventPage:
    """One answer to a cursor-scoped ledger read, with its cursor verdict.

    ``status`` is decided once, here, so a second reader cannot bypass the
    retained-minimum check by querying the table directly through this module.
    """

    status: EventCursorStatus
    events: tuple[dict[str, object], ...]
    retained_min_id: int | None
    """Lowest id still in the ledger, or ``None`` when the ledger holds no rows."""
    latest_id: int
    """The newest numeric position the ledger has held."""
    latest_cursor: str | None
    """Lifetime-bound high-water consistent with a resync; None for an absent ledger."""
    resync: dict[str, object] | None = None
    """Snapshot-shaped envelope present exactly when ``status`` is ``AGED_OUT``."""

    def __post_init__(self) -> None:
        if (self.status is EventCursorStatus.AGED_OUT) != (self.resync is not None):
            raise ValueError("an aged-out event page carries a resync envelope and an ok page carries none")
        if self.status is EventCursorStatus.AGED_OUT and self.events:
            raise ValueError("an aged-out event page must not also deliver a partial row page")


def _retained_range(conn: sqlite3.Connection) -> tuple[int | None, int, int]:
    """Return ``(MIN(id), high-water id, pruned-through watermark)`` in one snapshot.

    The high-water id is the newest id the ledger has ever held: the newest row,
    or the watermark when retention removed the newest rows too.
    """
    row = conn.execute("SELECT MIN(id), COALESCE(MAX(id), 0) FROM daemon_events").fetchone()
    pruned_through = _pruned_through(conn)
    if row is None:
        return None, pruned_through, pruned_through
    retained_min = None if row[0] is None else int(row[0])
    return retained_min, max(int(row[1]), pruned_through), pruned_through


def _cursor_refusal_reason(last_id: int, pruned_through: int, high_water: int) -> str | None:
    """Return why ``last_id`` cannot be honoured, or ``None`` when it can.

    Retention removes only ids at or below the watermark, so a subscriber that
    has read through the watermark has every later row; one below it has
    missed at least the watermark row itself.
    """
    if last_id > high_water:
        # Ahead of every id this ledger ever held: the disposable ops tier was
        # reset or replaced under the subscriber.
        return RESYNC_LEDGER_RESET
    if last_id < pruned_through:
        return RESYNC_CURSOR_AGED_OUT
    return None


def query_events_since(
    cursor: str | None,
    *,
    kinds: Sequence[str] | None = None,
    limit: int = 200,
) -> DaemonEventPage:
    """Return the page of daemon events after the opaque ``cursor``, oldest-first.

    Used by the live SSE stream and ETag polling fallback in the web reader.
    ``kinds`` restricts to a whitelist (empty/None means all kinds).

    A cursor below the retention watermark is refused with
    :attr:`EventCursorStatus.AGED_OUT` and a resync envelope rather than the
    silently short page ``WHERE id > ?`` would otherwise produce.

    The watermark check and the page read observe **one** snapshot. In
    autocommit each statement takes its own, so a writer pruning between them
    lets a cursor pass the watermark check and then read a page whose rows
    were deleted in the gap -- delivered as a short ``OK`` page, which is
    exactly the silent event loss the refusal exists to prevent. ``BEGIN`` is
    deferred, so the snapshot is taken by the first read below and released by
    the ``ROLLBACK`` in ``finally``; the connection is ``query_only``, so this
    read transaction can never become a write.
    """
    cursor_lifetime, last_id = parse_event_cursor(cursor)
    conn = _open_events_reader()
    if conn is None:
        if cursor is None:
            return DaemonEventPage(
                status=EventCursorStatus.OK, events=(), retained_min_id=None, latest_id=0, latest_cursor=None
            )
        resync = build_snapshot_envelope(
            event_id=0,
            ts=_iso_from_ms(current_epoch_ms()),
            event_count=0,
            first_event_id=None,
            last_event_id=None,
            kind_counts={},
            resync_reason="ledger_reset",
            requested_since=cursor,
        )
        return DaemonEventPage(
            status=EventCursorStatus.AGED_OUT,
            events=(),
            retained_min_id=None,
            latest_id=0,
            latest_cursor=None,
            resync=resync,
        )
    try:
        conn.execute("BEGIN")
        retained_min, latest, pruned_through = _retained_range(conn)
        lifetime = _ledger_lifetime(conn)
        latest_cursor = f"{lifetime}:{latest}"
        refusal = (
            RESYNC_LEDGER_RESET
            if cursor_lifetime is not None and cursor_lifetime != lifetime
            else (_cursor_refusal_reason(last_id, pruned_through, latest) if cursor is not None else None)
        )
        if refusal is not None:
            kind_counts = {
                str(row[0]): int(row[1])
                for row in conn.execute("SELECT kind, COUNT(*) FROM daemon_events GROUP BY kind")
            }
            retained_count = sum(kind_counts.values())
            resync = build_snapshot_envelope(
                event_id=latest,
                ts=_iso_from_ms(current_epoch_ms()),
                event_count=retained_count,
                first_event_id=retained_min,
                last_event_id=latest if latest else None,
                kind_counts=kind_counts,
                resync_reason=refusal,
                requested_since=cursor,
            )
            resync["cursor"] = latest_cursor
            return DaemonEventPage(
                status=EventCursorStatus.AGED_OUT,
                events=(),
                retained_min_id=retained_min,
                latest_id=latest,
                latest_cursor=latest_cursor,
                resync=resync,
            )
        kinds_tuple = tuple(kinds or ())
        if kinds_tuple:
            placeholders = ",".join("?" for _ in kinds_tuple)
            sql = (
                f"SELECT id, ts_ms, kind, operation_id, payload_json "
                f"FROM daemon_events WHERE id > ? AND kind IN ({placeholders}) "
                f"ORDER BY id ASC LIMIT ?"
            )
            params: tuple[object, ...] = (last_id, *kinds_tuple, limit)
        else:
            sql = (
                "SELECT id, ts_ms, kind, operation_id, payload_json "
                "FROM daemon_events WHERE id > ? ORDER BY id ASC LIMIT ?"
            )
            params = (last_id, limit)
        rows = conn.execute(sql, params).fetchall()
        return DaemonEventPage(
            status=EventCursorStatus.OK,
            events=tuple(
                {
                    "id": row[0],
                    "cursor": f"{lifetime}:{row[0]}",
                    "ts": _iso_from_ms(row[1]),
                    "kind": row[2],
                    "operation_id": row[3],
                    "payload": json.loads(row[4]),
                }
                for row in rows
            ),
            retained_min_id=retained_min,
            latest_id=latest,
            latest_cursor=latest_cursor,
        )
    finally:
        conn.rollback()
        conn.close()


def get_latest_event_cursor() -> str | None:
    """Read the lifetime-bound high-water in one snapshot without hydrating events."""
    conn = _open_events_reader()
    if conn is None:
        return None
    try:
        conn.execute("BEGIN")
        row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM daemon_events").fetchone()
        assert row is not None
        latest = max(int(row[0]), _pruned_through(conn))
        return f"{_ledger_lifetime(conn)}:{latest}"
    finally:
        conn.rollback()
        conn.close()


def get_last_ingestion_batch() -> dict[str, object] | None:
    """Return the most recent ingestion_batch event, if any."""
    with closing(iter_daemon_events(kind="ingestion_batch", limit=1)) as events:
        return next(events, None)


def get_recent_operations(limit: int = 10) -> Iterator[dict[str, object]]:
    """Return recent daemon operations."""
    return iter_daemon_events(kind="operation", limit=limit)


# --------------------------------------------------------------------------
# Granular event kinds (#1204)
# --------------------------------------------------------------------------
#
# These constants name the per-topic SSE events the reader subscribes to.
# Older opaque kinds (``ingestion_batch``/``ingest``/``reset``/``operation``)
# remain on the wire for backwards compatibility with existing consumers
# (status views, polling fallback). The granular kinds below split the
# realtime channel so the reader can subscribe selectively by view and
# animate just-appended rows without rerendering the full list.
#
# ``insight.updated`` / ``progress.update`` / ``progress.complete`` were
# retired here (polylogue-20d.13): grepping the whole codebase found no
# production caller for ``emit_insight_updated``/``emit_progress_update``/
# ``emit_progress_complete`` -- the only callers were their own unit tests,
# and the docstring's claimed consumer (``status --convergence --watch``)
# does not exist in the CLI. Per this bead's AC ("every advertised topic has
# a declared spec and production emitter, or is removed"), an advertised
# topic with no real producer is a completeness defect, not a feature to
# preserve. Wiring real embedding-catchup/insight-rebuild progress into SSE
# remains a legitimate future bead; it should introduce these kinds fresh
# from a real call site rather than resurrect the unwired scaffolding.

EVENT_SESSION_APPENDED = "session.appended"
EVENT_SESSION_UPDATED = "session.updated"
EVENT_MESSAGE_APPENDED = "message.appended"


class EventProducerPhase(str, Enum):
    """When, relative to the archive transaction, a topic's frame is published."""

    POST_COMMIT = "post-commit"
    """Published only after the archive transaction the event describes committed."""

    PRE_COMMIT = "pre-commit"
    """Published while the described transaction is still open -- a consumer that
    fetches the referenced object may observe pre-transaction state."""


class EventAudience(str, Enum):
    """Who a topic's frames may be delivered to."""

    LOOPBACK_READER = "loopback-reader"
    """Any subscriber already authorized for the daemon's loopback read API; the
    frame carries refs and counters only, never archive content."""


@dataclass(frozen=True, slots=True)
class EventSpec:
    """The declared contract for one advertised SSE topic (polylogue-20d.13.5).

    A topic without an entry here is a topic a subscriber cannot reason about:
    it has no statement of which object the event refers to, which cursor
    resumes it, which wire frame carries it, whether the described write has
    committed, what the payload may contain, or who may see it. The registry
    exists so that advertising a topic and declaring its contract are the same
    act -- :data:`GRANULAR_EVENT_KINDS` is derived from :data:`EVENT_SPECS`
    rather than restated beside it.
    """

    kind: str
    """Stable event id; also the value of the ledger's ``kind`` column."""

    summary: str
    """What a subscriber learns from one frame of this topic."""

    object_ref: str
    """Payload key naming the archive object the event refers to."""

    source_ref: str | None
    """Payload key naming the acquisition source, when the topic carries one."""

    archive_tier: ArchiveTier
    """Tier whose committed state ``object_ref``/``source_ref`` resolve against."""

    ledger_tier: ArchiveTier
    """Tier holding the replay ledger this topic's cursor indexes."""

    cursor_field: str
    """Ledger position combined with its lifetime in the public resume cursor."""

    frame: str
    """SSE ``event:`` frame name written on the wire for this topic."""

    producer_phase: EventProducerPhase
    payload_projection: tuple[str, ...]
    """Exhaustive set of keys the topic's payload may carry. An emitter that
    adds a key without declaring it here is publishing an undeclared projection."""

    required_payload_fields: tuple[str, ...]
    """Keys every frame of this topic carries, whatever their value."""

    audience: EventAudience
    emitter: str
    """Dotted path of the production function that publishes this topic. A topic
    whose only producer lives under ``tests/`` is advertised but never emitted --
    the regression that retired ``insight.updated``/``progress.*``."""

    def __post_init__(self) -> None:
        for field_name in ("kind", "summary", "object_ref", "cursor_field", "frame", "emitter"):
            if not getattr(self, field_name):
                raise ValueError(f"event spec {self.kind or '<unnamed>'} requires a non-empty {field_name}")
        if self.frame != self.kind:
            # ``_write_sse_event`` writes the ledger ``kind`` as the SSE frame
            # name, so a spec whose declared frame differs from its kind would
            # describe a wire shape the daemon never produces.
            raise ValueError(f"event spec {self.kind} declares frame {self.frame!r}, but the wire frame is the kind")
        if self.object_ref not in self.payload_projection:
            raise ValueError(f"event spec {self.kind} declares object_ref {self.object_ref!r} outside its projection")
        if self.source_ref is not None and self.source_ref not in self.payload_projection:
            raise ValueError(f"event spec {self.kind} declares source_ref {self.source_ref!r} outside its projection")
        undeclared_required = set(self.required_payload_fields) - set(self.payload_projection)
        if undeclared_required:
            raise ValueError(
                f"event spec {self.kind} requires payload fields outside its projection: {sorted(undeclared_required)}"
            )
        if self.object_ref not in self.required_payload_fields:
            raise ValueError(f"event spec {self.kind} must always carry its object ref {self.object_ref!r}")
        if not self.emitter.startswith("polylogue."):
            raise ValueError(f"event spec {self.kind} names a non-production emitter: {self.emitter}")

    def to_payload(self) -> dict[str, object]:
        """Return the spec as a JSON-ready declaration."""
        return {
            "kind": self.kind,
            "summary": self.summary,
            "object_ref": self.object_ref,
            "source_ref": self.source_ref,
            "archive_tier": self.archive_tier.value,
            "ledger_tier": self.ledger_tier.value,
            "cursor_field": self.cursor_field,
            "frame": self.frame,
            "producer_phase": self.producer_phase.value,
            "payload_projection": list(self.payload_projection),
            "required_payload_fields": list(self.required_payload_fields),
            "audience": self.audience.value,
            "emitter": self.emitter,
        }


_EVENT_SPECS: tuple[EventSpec, ...] = (
    EventSpec(
        kind=EVENT_SESSION_APPENDED,
        summary="A session was materialized into the archive by a full-parse ingest route.",
        object_ref="session_id",
        source_ref="source_name",
        archive_tier=ArchiveTier.INDEX,
        ledger_tier=ArchiveTier.OPS,
        cursor_field="id",
        frame=EVENT_SESSION_APPENDED,
        producer_phase=EventProducerPhase.POST_COMMIT,
        payload_projection=(
            "session_id",
            "source_name",
            "succeeded_file_count",
            "failed_file_count",
            "source_paths",
        ),
        required_payload_fields=("session_id", "source_name", "succeeded_file_count", "failed_file_count"),
        audience=EventAudience.LOOPBACK_READER,
        emitter="polylogue.daemon.events.emit_session_appended",
    ),
    EventSpec(
        kind=EVENT_SESSION_UPDATED,
        summary="An already-archived session grew through the live-ingest append route.",
        object_ref="session_id",
        source_ref="source_name",
        archive_tier=ArchiveTier.INDEX,
        ledger_tier=ArchiveTier.OPS,
        cursor_field="id",
        frame=EVENT_SESSION_UPDATED,
        producer_phase=EventProducerPhase.POST_COMMIT,
        payload_projection=("session_id", "source_name", "appended_count"),
        required_payload_fields=("session_id", "source_name", "appended_count"),
        audience=EventAudience.LOOPBACK_READER,
        emitter="polylogue.daemon.events.emit_session_updated",
    ),
    EventSpec(
        kind=EVENT_MESSAGE_APPENDED,
        summary="Messages were appended to a session, for live-tail consumers of that session.",
        object_ref="session_id",
        source_ref="source_name",
        archive_tier=ArchiveTier.INDEX,
        ledger_tier=ArchiveTier.OPS,
        cursor_field="id",
        frame=EVENT_MESSAGE_APPENDED,
        producer_phase=EventProducerPhase.POST_COMMIT,
        payload_projection=("session_id", "source_name", "appended_count", "source_path"),
        required_payload_fields=("session_id", "source_name", "appended_count"),
        audience=EventAudience.LOOPBACK_READER,
        emitter="polylogue.daemon.events.emit_message_appended",
    ),
)

EVENT_SPECS: Mapping[str, EventSpec] = MappingProxyType({spec.kind: spec for spec in _EVENT_SPECS})
"""Per-topic contracts, keyed by advertised kind."""

if len(EVENT_SPECS) != len(_EVENT_SPECS):  # pragma: no cover - construction-time guard
    raise ValueError("duplicate event spec kind in EVENT_SPECS")

#: The advertised granular SSE topics. Derived from :data:`EVENT_SPECS` so a
#: topic cannot be advertised without declaring its contract first.
GRANULAR_EVENT_KINDS: frozenset[str] = frozenset(EVENT_SPECS)

#: Extension-reported browser capture health (polylogue-3v1).
CAPTURE_HEALTH_EVENT_KIND = "browser_capture_health"


def event_spec(kind: str) -> EventSpec:
    """Return the declared contract for ``kind``.

    Raises ``KeyError`` for a kind with no declared contract -- including the
    opaque legacy kinds (``ingestion_batch``/``ingest``/``reset``/
    ``operation``), which are not advertised granular topics.
    """
    return EVENT_SPECS[kind]


def session_appended_event(
    *,
    source_name: str | None,
    succeeded_file_count: int,
    failed_file_count: int = 0,
    source_paths: Sequence[str] | None = None,
    session_id: str | None = None,
) -> DaemonEventRecord:
    """A ``session.appended`` event for a newly-materialized session.

    ``session_id`` is the real archive identity of the session this event
    describes (polylogue-20d.13) -- when known, callers should always pass
    it so consumers can scope refresh/animation to the exact session rather
    than treating every event as "refresh whatever is open". ``None`` is
    reserved for legacy/aggregate callers that genuinely cannot attribute a
    single session (e.g. pre-#1204 opaque batch summaries).
    """
    payload: dict[str, object] = {
        "source_name": source_name,
        "succeeded_file_count": int(succeeded_file_count),
        "failed_file_count": int(failed_file_count),
        "session_id": session_id,
    }
    if source_paths is not None:
        payload["source_paths"] = list(source_paths)
    return DaemonEventRecord(EVENT_SESSION_APPENDED, payload, operation_id=session_id)


def session_updated_event(
    *,
    session_id: str,
    source_name: str | None = None,
    appended_count: int = 0,
) -> DaemonEventRecord:
    """A ``session.updated`` event when an existing session grows.

    Distinct from :func:`session_appended_event`: the live-ingest append
    route only ever grows a file whose session already exists (a
    cursor-tracked prior observation), so every real producer of this event
    is describing a mutation of a session the reader may already have open
    -- exactly the identity the description this bead started from called
    unscoped (polylogue-20d.13).
    """
    payload: dict[str, object] = {
        "session_id": session_id,
        "source_name": source_name,
        "appended_count": int(appended_count),
    }
    return DaemonEventRecord(EVENT_SESSION_UPDATED, payload, operation_id=session_id)


def message_appended_event(
    *,
    session_id: str | None,
    source_name: str | None = None,
    appended_count: int = 0,
    source_path: str | None = None,
) -> DaemonEventRecord:
    """A ``message.appended`` event for live-tail consumers.

    The reader subscribes to this topic only for the currently-open
    session; subscription is encoded via ``?kinds=message.appended``
    plus filtering by ``session_id`` on the client.
    """
    payload: dict[str, object] = {
        "session_id": session_id,
        "source_name": source_name,
        "appended_count": int(appended_count),
    }
    if source_path is not None:
        payload["source_path"] = source_path
    return DaemonEventRecord(EVENT_MESSAGE_APPENDED, payload)


def emit_session_appended(
    *,
    source_name: str | None,
    succeeded_file_count: int,
    failed_file_count: int = 0,
    source_paths: Sequence[str] | None = None,
    session_id: str | None = None,
) -> None:
    """Emit one ``session.appended`` event; see :func:`session_appended_event`."""
    emit_daemon_events(
        [
            session_appended_event(
                source_name=source_name,
                succeeded_file_count=succeeded_file_count,
                failed_file_count=failed_file_count,
                source_paths=source_paths,
                session_id=session_id,
            )
        ]
    )


def emit_session_updated(
    *,
    session_id: str,
    source_name: str | None = None,
    appended_count: int = 0,
) -> None:
    """Emit one ``session.updated`` event; see :func:`session_updated_event`."""
    emit_daemon_events(
        [session_updated_event(session_id=session_id, source_name=source_name, appended_count=appended_count)]
    )


def emit_message_appended(
    *,
    session_id: str | None,
    source_name: str | None = None,
    appended_count: int = 0,
    source_path: str | None = None,
) -> None:
    """Emit one ``message.appended`` event; see :func:`message_appended_event`."""
    emit_daemon_events(
        [
            message_appended_event(
                session_id=session_id,
                source_name=source_name,
                appended_count=appended_count,
                source_path=source_path,
            )
        ]
    )


def get_daemon_event_counts() -> dict[str, int]:
    """Return event counts by kind."""
    conn = _open_events_reader()
    if conn is None:
        return {}
    try:
        rows = conn.execute("SELECT kind, COUNT(*) FROM daemon_events GROUP BY kind ORDER BY COUNT(*) DESC").fetchall()
        return {row[0]: row[1] for row in rows}
    finally:
        conn.close()


__all__ = [
    "EVENT_MESSAGE_APPENDED",
    "EVENT_SESSION_APPENDED",
    "EVENT_SESSION_UPDATED",
    "EVENT_SPECS",
    "GRANULAR_EVENT_KINDS",
    "RESYNC_CURSOR_AGED_OUT",
    "RESYNC_LEDGER_RESET",
    "DaemonEventPage",
    "EVENT_SUBSCRIBERS",
    "EventAudience",
    "EventSubscriberRegistry",
    "EventSubscription",
    "EventCursorStatus",
    "EventProducerPhase",
    "EventSpec",
    "build_snapshot_envelope",
    "event_spec",
    "prune_daemon_events",
    "emit_session_appended",
    "emit_session_updated",
    "emit_daemon_event",
    "get_latest_daemon_event",
    "emit_message_appended",
    "get_daemon_event_counts",
    "get_last_ingestion_batch",
    "get_latest_event_cursor",
    "get_recent_operations",
    "current_epoch_ms",
    "iter_daemon_events",
    "capture_health_page",
    "CAPTURE_HISTORY_PAGE_ROWS",
    "CaptureHistoryCursorError",
    "CaptureHistoryStorageError",
    "query_events_since",
]
