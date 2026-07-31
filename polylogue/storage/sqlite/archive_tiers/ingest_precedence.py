"""Shared archive write precedence helpers.

Writer module: index.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from typing import Literal

from polylogue.core.timestamps import parse_timestamp
from polylogue.sources.parsers.base_models import ParsedSessionEvent

BrowserCapturePrecedence = Literal["default", "replace", "skip"]


def should_skip_stale_replace(
    *,
    incoming_freshness_ms: int | None,
    existing_updated_at_ms: int | None,
) -> bool:
    """Return whether an incoming full-replace write is strictly staler than what is stored.

    This is the ONE freshness-tie policy for whether an incoming session
    write should be skipped as stale, consolidated from three previously
    independent copies (polylogue-t83e): ``write_parsed_session_to_archive``
    in ``archive_tiers/write.py``, the daemon batch-write path in
    ``pipeline/services/ingest_batch/_core.py``, and
    ``revision_governance.py``'s raw-parsed write path. Each call site keeps
    its own surrounding guard conditions (``force_write``/``force_replace``,
    browser-capture precedence, append-only, revision-authority membership,
    ``source_index`` gating) — those decide *whether this check applies at
    all*, not the comparison itself.

    Deliberately a strict ``<``, not ``<=``: a genuine tie (same
    ``updated_at_ms``) still replaces, so a replay that legitimately carries
    identical freshness but different/corrected content is not silently
    dropped.

    This is a *timestamp* tie-break only, deliberately narrow: it is not
    where "which raw is the real content" gets decided when two raws
    resolve to the same ``session_id`` and one is a strict content subset of
    the other (e.g. an appended-to Claude Code transcript re-acquired twice
    at different completeness). That is decided upstream, before this
    function's timestamp comparison is ever reached, by
    ``archive/session_revision_membership.py``'s content-only set relation
    (``equal``/``a_contains_b``/``b_contains_a``/``conflict`` — polylogue-aggz,
    landed in #3401/#3405) via ``raw_session_memberships``/
    ``raw_revision_heads``: a cohort with an accepted head is written from
    that head, never from this per-write timestamp comparison. This function
    only governs the fallback for raws revision governance never classified
    into a cohort (single-raw sessions, or a governance table not yet
    populated for older data) — see polylogue-t83e for a live example where
    stale pre-#3401 ``raw_session_memberships`` rows briefly left two raws
    of one session_id -- a local Claude Code transcript and an older byte-
    prefix copy of the same file the operator had uploaded into an AI Studio
    conversation -- without an accepted head, and this function's ordinary
    timestamp comparison (not content-subset awareness) decided the
    outcome. Recomputing revision membership under current code (any
    ``rebuild_index_from_source`` replay, which calls
    ``backfill_historical_revision_evidence``) resolves that case correctly
    upstream of this function.
    """
    return (
        existing_updated_at_ms is not None
        and incoming_freshness_ms is not None
        and incoming_freshness_ms < existing_updated_at_ms
    )


def browser_capture_precedence(
    *,
    existing_is_dom_fallback: bool,
    incoming_is_dom_fallback: bool,
    existing_has_native_payload: bool,
    incoming_has_native_payload: bool,
    stored_message_count: int,
    incoming_message_count: int,
) -> BrowserCapturePrecedence:
    """Resolve browser-source ownership before provider timestamp freshness."""
    existing_is_browser_capture = existing_is_dom_fallback or existing_has_native_payload
    incoming_is_browser_capture = incoming_is_dom_fallback or incoming_has_native_payload

    # A genuine, non-browser-capture arrival (a direct/native provider export,
    # or any other real re-ingest) always outranks content that was only ever
    # established by a browser capture (dom-fallback or native-payload). A
    # browser capture exists to backfill a session before its paired direct
    # export shows up, never to shadow the export once it arrives. Falling
    # through to provider-timestamp freshness for this combination is order
    # dependent: whichever material is ingested first sets the session's
    # ``updated_at_ms``, and a capture's own capture timestamp is frequently
    # *later* than the underlying conversation's real ``updated_at`` (the
    # capture can happen well after the last message was sent). A
    # single-writer replay that always processes the direct export first
    # never observes this; a live daemon ingesting a batch of newly staged
    # files with no ordering guarantee across them can process the capture
    # first and then permanently skip the real export as "stale"
    # (polylogue-z1c6).
    if existing_is_browser_capture and not incoming_is_browser_capture:
        return "replace"

    # The mirror of the rule above: a genuine, already-established
    # non-browser-capture session must never be shadowed by an incoming
    # browser capture either, regardless of message count. Without this,
    # ``incoming_owns_browser_merge``'s native-payload leg below (message
    # count alone) would let a native browser capture that happens to carry
    # at least as many messages as the currently-stored real export replace
    # that export -- an order-dependent regression symmetric to the one the
    # first rule fixes: whichever material a live daemon processes first
    # would win, instead of the direct export always winning
    # (polylogue-z1c6 review follow-up).
    if incoming_is_browser_capture and not existing_is_browser_capture:
        return "skip"

    lower_precedence_fallback = incoming_is_dom_fallback and not existing_is_dom_fallback
    lower_precedence_export = (
        existing_has_native_payload
        and not incoming_has_native_payload
        and incoming_message_count <= stored_message_count
    )
    strictly_less_complete = incoming_message_count < stored_message_count and not (
        existing_is_dom_fallback and not incoming_is_dom_fallback
    )
    if lower_precedence_fallback or lower_precedence_export or strictly_less_complete:
        return "skip"

    incoming_owns_browser_merge = (
        (existing_is_dom_fallback and not incoming_is_dom_fallback)
        or (
            incoming_has_native_payload
            and not existing_has_native_payload
            and incoming_message_count >= stored_message_count
        )
        or (
            existing_has_native_payload
            and not incoming_has_native_payload
            and incoming_message_count > stored_message_count
        )
    )
    return "replace" if incoming_owns_browser_merge else "default"


def stored_message_count(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row[0] or 0) if row is not None else 0


def session_has_parser_ingest_flag(conn: sqlite3.Connection, session_id: str, flag: str | Sequence[str]) -> bool:
    """Return whether *session_id* carries any of the given parser ingest flag(s).

    Accepts either a single flag or a sequence -- native browser-capture
    payloads are tagged with either ``NATIVE_BROWSER_CAPTURE_INGEST_FLAG`` or
    ``COMPACT_BROWSER_CAPTURE_INGEST_FLAG`` depending on capture shape
    (``polylogue/sources/parsers/browser_capture.py``), and callers checking
    "is this session native-payload-flagged" must check both or silently
    misclassify compact captures as plain, non-browser-capture content
    (polylogue-z1c6 review follow-up).
    """
    flags = (flag,) if isinstance(flag, str) else tuple(flag)
    if not flags:
        return False
    placeholders = ",".join("?" for _ in flags)
    row = conn.execute(
        f"""
        SELECT 1
        FROM session_tags
        WHERE session_id = ?
          AND tag IN ({placeholders})
          AND tag_source = 'auto'
          AND method = 'parser'
        LIMIT 1
        """,
        (session_id, *flags),
    ).fetchone()
    return row is not None


def record_capture_gap_event(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    existing_raw_id: str,
    incoming_raw_id: str,
    stored_message_count: int,
    incoming_message_count: int,
) -> None:
    row = conn.execute(
        """
        SELECT MAX(position) + 1
        FROM (
            SELECT position FROM session_events WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_agent_policies WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_provider_usage_events WHERE session_id = ?
        )
        """,
        (session_id, session_id, session_id),
    ).fetchone()
    position = int(row[0] or 0) if row is not None else 0
    summary = (
        "Skipped lower-precedence DOM browser-capture fallback "
        f"{incoming_raw_id!r}; existing raw {existing_raw_id!r} has "
        f"{stored_message_count} message(s), incoming fallback has {incoming_message_count}."
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO session_events (
            session_id, source_message_id, source_message_provider_id,
            position, event_type, summary, occurred_at_ms
        ) VALUES (?, NULL, NULL, ?, 'capture_gap', ?, NULL)
        """,
        (session_id, position, summary),
    )


def record_source_outage_events(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    events: Sequence[ParsedSessionEvent],
) -> int:
    """Persist source-outage telemetry a capture declares about itself even
    when its message content loses the browser-capture precedence merge and
    is otherwise discarded.

    A capture that loses the content merge can still be telling the truth
    about when it was not observing the page: that claim does not depend on
    whether its transcript content wins. Mirrors ``record_capture_gap_event``:
    a lightweight write survives the skip path instead of the whole incoming
    session's evidence being silently dropped alongside its discarded
    messages.
    """
    outage_events = [event for event in events if event.event_type == "source_outage"]
    if not outage_events:
        return 0
    row = conn.execute(
        """
        SELECT MAX(position) + 1
        FROM (
            SELECT position FROM session_events WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_agent_policies WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_provider_usage_events WHERE session_id = ?
        )
        """,
        (session_id, session_id, session_id),
    ).fetchone()
    position = int(row[0] or 0) if row is not None else 0
    for event in outage_events:
        summary = str(event.payload.get("summary") or "")
        occurred_at_ms: int | None = None
        if event.timestamp:
            parsed_timestamp = parse_timestamp(event.timestamp)
            if parsed_timestamp is not None:
                occurred_at_ms = int(parsed_timestamp.timestamp() * 1000)
        conn.execute(
            """
            INSERT OR REPLACE INTO session_events (
                session_id, source_message_id, source_message_provider_id,
                position, event_type, summary, payload_json, occurred_at_ms
            ) VALUES (?, NULL, NULL, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                position,
                event.event_type,
                summary,
                json.dumps(event.payload, sort_keys=True, ensure_ascii=False),
                occurred_at_ms,
            ),
        )
        position += 1
    return len(outage_events)


__all__ = [
    "BrowserCapturePrecedence",
    "browser_capture_precedence",
    "record_capture_gap_event",
    "record_source_outage_events",
    "session_has_parser_ingest_flag",
    "stored_message_count",
]
