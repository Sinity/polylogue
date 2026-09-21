"""Daemon event ledger backed by archive ops state."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

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

_DAEMON_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS daemon_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    kind TEXT NOT NULL,
    operation_id TEXT,
    idempotency_key TEXT,
    payload_json TEXT NOT NULL
 ) STRICT;
"""


def _events_db_path() -> Path:
    """Return the path to the daemon events SQLite database."""
    return archive_root() / "ops.db"


#: Ops-tier convergence runs once per (process, path): the multi-statement
#: tier initialization can wait out SQLite lock timeouts statement by
#: statement, and running it on EVERY emit put that aggregate wait inside the
#: write-coordinator's shutdown window -- a SIGTERM'd daemon then exceeded its
#: 15s exit deadline stuck in tier DDL (polylogue-b9oi8). First emit still
#: converges the tier, so benign-DDL convergence-on-open is preserved.
_CONVERGED_EVENT_DBS: set[Path] = set()


def _ensure_events_db(path: Path | None = None) -> sqlite3.Connection:
    """Open and initialize the daemon events database for an emitter."""
    path = _events_db_path() if path is None else path
    if path not in _CONVERGED_EVENT_DBS or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        initialize_archive_database(path, ArchiveTier.OPS)
        _CONVERGED_EVENT_DBS.add(path)
    conn = open_daemon_connection(path)
    conn.executescript(_DAEMON_EVENTS_DDL)
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(daemon_events)")}
    if "idempotency_key" not in columns:
        conn.execute("ALTER TABLE daemon_events ADD COLUMN idempotency_key TEXT")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_daemon_events_idempotency "
        "ON daemon_events(kind, idempotency_key) WHERE idempotency_key IS NOT NULL"
    )
    conn.commit()
    return conn


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
    except BaseException:
        conn.close()
        raise
    if exists is None:
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


@dataclass(frozen=True, slots=True)
class DaemonEventRetention:
    """Declared bound on the ``daemon_events`` replay ledger.

    ``daemon_events`` lives in the disposable ops tier but nothing in
    ``polylogue/`` ever deleted from it, so a long-running daemon emitting one
    ``message.appended`` per live append grew the table without limit
    (polylogue-20d.13.6).

    Both bounds default to ``None``: **no retention value is declared here on
    purpose.** The parent bead supplies none, no measurement of the real
    emission rate exists, and a number invented to make the enforcement point
    look satisfied would be a fake bound. :func:`prune_daemon_events` is the
    named enforcement point and runs on every emit; until an operator or a
    live-daemon measurement supplies a bound it prunes nothing and
    :attr:`is_bounded` reports the ledger as unbounded rather than pretending
    otherwise.
    """

    max_rows: int | None = None
    max_age_ms: int | None = None

    def __post_init__(self) -> None:
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("daemon event retention max_rows must be positive when declared")
        if self.max_age_ms is not None and self.max_age_ms <= 0:
            raise ValueError("daemon event retention max_age_ms must be positive when declared")

    @property
    def is_bounded(self) -> bool:
        """True only when at least one bound has actually been declared."""
        return self.max_rows is not None or self.max_age_ms is not None


_UNBOUNDED_RETENTION = DaemonEventRetention()
_RETENTION: DaemonEventRetention = _UNBOUNDED_RETENTION


def daemon_event_retention() -> DaemonEventRetention:
    """Return the retention bound currently enforced on ``daemon_events``."""
    return _RETENTION


def set_daemon_event_retention(retention: DaemonEventRetention) -> DaemonEventRetention:
    """Install ``retention`` as the enforced bound and return the previous one."""
    global _RETENTION
    previous = _RETENTION
    _RETENTION = retention
    return previous


def prune_daemon_events(
    conn: sqlite3.Connection,
    retention: DaemonEventRetention | None = None,
    *,
    now_ms: int | None = None,
) -> int:
    """Enforce ``retention`` on ``daemon_events`` and return the rows removed.

    The single enforcement point for the ledger bound. Called on every emit so
    a resuming subscriber's cursor and the retained range are trimmed by the
    same writer, inside the emit transaction.
    """
    resolved = daemon_event_retention() if retention is None else retention
    if not resolved.is_bounded:
        return 0
    removed = 0
    if resolved.max_age_ms is not None:
        horizon = (current_epoch_ms() if now_ms is None else now_ms) - resolved.max_age_ms
        removed += conn.execute("DELETE FROM daemon_events WHERE ts_ms < ?", (horizon,)).rowcount
    if resolved.max_rows is not None:
        row_count = int(conn.execute("SELECT COUNT(*) FROM daemon_events").fetchone()[0])
        excess = row_count - resolved.max_rows
        if excess > 0:
            removed += conn.execute(
                "DELETE FROM daemon_events WHERE id IN (SELECT id FROM daemon_events ORDER BY id ASC LIMIT ?)",
                (excess,),
            ).rowcount
    return removed


def emit_daemon_event(
    kind: str,
    *,
    operation_id: str | None = None,
    idempotency_key: str | None = None,
    payload: dict[str, object] | None = None,
    archive_root_path: Path | None = None,
    observed_at_ms: int | None = None,
) -> None:
    """Emit a daemon event to the event ledger."""
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
        conn.execute(
            "INSERT INTO daemon_events (ts_ms, kind, operation_id, idempotency_key, payload_json) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(kind, idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING",
            (
                current_epoch_ms() if observed_at_ms is None else observed_at_ms,
                kind,
                operation_id,
                idempotency_key,
                json.dumps(payload or {}),
            ),
        )
        prune_daemon_events(conn, now_ms=observed_at_ms)
        conn.commit()
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


def query_daemon_events(
    *,
    kind: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> Sequence[dict[str, object]]:
    """Query recent daemon events."""
    conn = _open_events_reader()
    if conn is None:
        return []
    try:
        if kind:
            rows = conn.execute(
                "SELECT id, ts_ms, kind, operation_id, payload_json FROM daemon_events WHERE kind = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (kind, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, ts_ms, kind, operation_id, payload_json FROM daemon_events ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "id": row[0],
                    "ts": _iso_from_ms(row[1]),
                    "kind": row[2],
                    "operation_id": row[3],
                    "payload": json.loads(row[4]),
                }
            )
        return result
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
    requested_since: int | None = None,
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
    resync: dict[str, object] | None = None
    """Snapshot-shaped envelope present exactly when ``status`` is ``AGED_OUT``."""

    def __post_init__(self) -> None:
        if (self.status is EventCursorStatus.AGED_OUT) != (self.resync is not None):
            raise ValueError("an aged-out event page carries a resync envelope and an ok page carries none")
        if self.status is EventCursorStatus.AGED_OUT and self.events:
            raise ValueError("an aged-out event page must not also deliver a partial row page")


def _retained_range(conn: sqlite3.Connection) -> tuple[int | None, int]:
    row = conn.execute("SELECT MIN(id), COALESCE(MAX(id), 0) FROM daemon_events").fetchone()
    if row is None:
        return None, 0
    return (None if row[0] is None else int(row[0])), int(row[1])


def _cursor_refusal_reason(last_id: int, retained_min: int | None, latest: int) -> str | None:
    """Return why ``last_id`` cannot be honoured, or ``None`` when it can."""
    if last_id <= 0:
        return None
    if retained_min is None:
        # The table exists but holds nothing, while the subscriber claims to
        # have already seen row ``last_id``: its history is gone.
        return RESYNC_LEDGER_RESET
    if last_id + 1 < retained_min:
        return RESYNC_CURSOR_AGED_OUT
    if last_id > latest:
        return RESYNC_LEDGER_RESET
    return None


def query_events_since(
    last_id: int,
    *,
    kinds: Sequence[str] | None = None,
    limit: int = 200,
) -> DaemonEventPage:
    """Return the page of daemon events after ``last_id``, oldest-first.

    Used by the live SSE stream and ETag polling fallback in the web reader.
    ``kinds`` restricts to a whitelist (empty/None means all kinds).

    A cursor below the retained minimum is refused with
    :attr:`EventCursorStatus.AGED_OUT` and a resync envelope rather than the
    silently short page ``WHERE id > ?`` would otherwise produce.
    """
    conn = _open_events_reader()
    if conn is None:
        # No ledger file, or an ops database predating the event schema: there
        # is no retained range to compare a cursor against, so this stays the
        # documented empty-ledger result rather than a fabricated refusal.
        return DaemonEventPage(status=EventCursorStatus.OK, events=(), retained_min_id=None, latest_id=0)
    try:
        retained_min, latest = _retained_range(conn)
        refusal = _cursor_refusal_reason(last_id, retained_min, latest)
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
                requested_since=last_id,
            )
            return DaemonEventPage(
                status=EventCursorStatus.AGED_OUT,
                events=(),
                retained_min_id=retained_min,
                latest_id=latest,
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
                    "ts": _iso_from_ms(row[1]),
                    "kind": row[2],
                    "operation_id": row[3],
                    "payload": json.loads(row[4]),
                }
                for row in rows
            ),
            retained_min_id=retained_min,
            latest_id=latest,
        )
    finally:
        conn.close()


def get_latest_event_id() -> int:
    """Return the id of the most recent daemon event, or 0 if none exist."""
    conn = _open_events_reader()
    if conn is None:
        return 0
    try:
        row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM daemon_events").fetchone()
        return int(row[0]) if row is not None else 0
    finally:
        conn.close()


def get_last_ingestion_batch() -> dict[str, object] | None:
    """Return the most recent ingestion_batch event, if any."""
    events = query_daemon_events(kind="ingestion_batch", limit=1)
    if events:
        return events[0]
    return None


def get_recent_operations(limit: int = 10) -> Sequence[dict[str, object]]:
    """Return recent daemon operations."""
    return query_daemon_events(kind="operation", limit=limit)


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
    """Ledger column carrying the monotonic resume cursor (``Last-Event-ID``)."""

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


def event_spec(kind: str) -> EventSpec:
    """Return the declared contract for ``kind``.

    Raises ``KeyError`` for a kind with no declared contract -- including the
    opaque legacy kinds (``ingestion_batch``/``ingest``/``reset``/
    ``operation``), which are not advertised granular topics.
    """
    return EVENT_SPECS[kind]


def emit_session_appended(
    *,
    source_name: str | None,
    succeeded_file_count: int,
    failed_file_count: int = 0,
    source_paths: Sequence[str] | None = None,
    session_id: str | None = None,
) -> None:
    """Emit a ``session.appended`` event for a newly-materialized session.

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
    emit_daemon_event(EVENT_SESSION_APPENDED, operation_id=session_id, payload=payload)


def emit_session_updated(
    *,
    session_id: str,
    source_name: str | None = None,
    appended_count: int = 0,
) -> None:
    """Emit a ``session.updated`` event when an existing session grows.

    Distinct from :func:`emit_session_appended`: the live-ingest append
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
    emit_daemon_event(EVENT_SESSION_UPDATED, operation_id=session_id, payload=payload)


def emit_message_appended(
    *,
    session_id: str | None,
    source_name: str | None = None,
    appended_count: int = 0,
    source_path: str | None = None,
) -> None:
    """Emit a ``message.appended`` event for live-tail consumers.

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
    emit_daemon_event(EVENT_MESSAGE_APPENDED, payload=payload)


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
    "DaemonEventRetention",
    "EventAudience",
    "EventCursorStatus",
    "EventProducerPhase",
    "EventSpec",
    "build_snapshot_envelope",
    "daemon_event_retention",
    "event_spec",
    "prune_daemon_events",
    "set_daemon_event_retention",
    "emit_session_appended",
    "emit_session_updated",
    "emit_daemon_event",
    "get_latest_daemon_event",
    "emit_message_appended",
    "get_daemon_event_counts",
    "get_last_ingestion_batch",
    "get_latest_event_id",
    "get_recent_operations",
    "current_epoch_ms",
    "query_daemon_events",
    "query_events_since",
]
