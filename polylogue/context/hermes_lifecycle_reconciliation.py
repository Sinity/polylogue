"""Render a Hermes session's drained lifecycle-event stream visible (fs1.7).

``hermes_lifecycle.reconcile_lifecycle_events`` was unit-tested but, until
this module, never reachable from any production surface: nothing read the
durable ``raw_hook_events`` spool (fs1.7) or the ingested session snapshot
(fs1.2/``hermes_state``) and fed them through it. That meant fs1.7's own AC
-- "an incomplete event stream is reconciled visibly against the session
snapshot" -- was not actually true at the product level (flagged by review).

This module is the read-side join that makes it true, mirroring the pattern
``context.hermes_delivery_correlation`` already established for fs1.11: a
read-only bridge over two durable tiers (``source.db`` lifecycle events,
``index.db`` ingested message snapshot), wired onto the real API facade
(``api.archive.PolylogueArchiveMixin.reconcile_hermes_session_lifecycle``)
rather than a second parallel mechanism.

Session-id join: lifecycle events carry the *raw* Hermes session id
(``sources.hooks`` enqueues whatever the producer sent as ``session_id``,
unqualified). The ingested conversational session's ``native_id`` is
profile-qualified (``<raw_id>@profile-<key>``, see
``hermes_state._qualified_session_id``). This module resolves both shapes so
a caller holding only the raw id gets every message the snapshot retains,
regardless of which profile ingested it.
"""

from __future__ import annotations

import sqlite3

from polylogue.core.enums import Origin
from polylogue.sources.parsers.hermes_lifecycle import (
    HermesLifecycleEvent,
    HermesLifecycleReconciliation,
    reconcile_lifecycle_events,
)
from polylogue.storage.sqlite.archive_tiers.source_write import list_hook_events


def _snapshot_message_ids(index_conn: sqlite3.Connection, hermes_session_native_id: str) -> frozenset[str]:
    """Return every message ``native_id`` the ingested Hermes snapshot retains.

    Matches both the raw and profile-qualified forms of the session's
    ``native_id`` (see module docstring) -- a caller only ever holds the raw
    Hermes session id, never the internal qualifier.
    """
    rows = index_conn.execute(
        """
        SELECT m.native_id
        FROM messages m
        JOIN sessions s ON s.session_id = m.session_id
        WHERE s.origin = ?
          AND (s.native_id = ? OR s.native_id LIKE ? || '@profile-%')
          AND m.native_id IS NOT NULL
        """,
        (Origin.HERMES_SESSION.value, hermes_session_native_id, hermes_session_native_id),
    ).fetchall()
    return frozenset(str(row[0]) for row in rows if row[0] is not None)


def _lifecycle_events_for(source_conn: sqlite3.Connection, hermes_session_native_id: str) -> list[HermesLifecycleEvent]:
    events: list[HermesLifecycleEvent] = []
    for hook_event in list_hook_events(
        source_conn, origin=Origin.HERMES_SESSION, session_native_id=hermes_session_native_id
    ):
        # ``hook_event.payload`` is the full spooled envelope (event_id/
        # event_type/session_id/timestamp/provider/payload/observed_at_ms,
        # see ``sources.hooks._persist_record``); the producer's own event
        # body is nested one level deeper under its own "payload" key --
        # same unwrap ``context.hermes_delivery_correlation`` performs.
        inner_payload = hook_event.payload.get("payload")
        events.append(
            HermesLifecycleEvent(
                event_id=hook_event.hook_event_id,
                event_type=hook_event.event_type,
                session_native_id=hook_event.session_native_id or hermes_session_native_id,
                observed_at_ms=hook_event.observed_at_ms,
                payload=inner_payload if isinstance(inner_payload, dict) else {},
            )
        )
    return events


def reconcile_hermes_session_lifecycle(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    *,
    hermes_session_native_id: str,
) -> HermesLifecycleReconciliation:
    """Reconcile one Hermes session's drained lifecycle-event stream.

    ``source_conn`` reads the durable spool (``raw_hook_events``,
    source.db); ``index_conn`` reads the ingested conversational snapshot
    (``messages``, index.db) so unpaired-event and unknown-message-reference
    gaps are computed against real archive state, not an empty default. A
    session with zero drained events still returns a well-formed
    (``total_events=0``, ``complete``) report rather than raising -- "no
    events observed yet" is itself visible information, not an error.
    """
    events = _lifecycle_events_for(source_conn, hermes_session_native_id)
    snapshot_message_ids = _snapshot_message_ids(index_conn, hermes_session_native_id)
    return reconcile_lifecycle_events(
        hermes_session_native_id,
        events,
        snapshot_message_ids=snapshot_message_ids,
    )


__all__ = ["reconcile_hermes_session_lifecycle"]
