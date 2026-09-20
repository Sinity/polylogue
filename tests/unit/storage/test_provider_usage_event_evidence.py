"""Provider usage events keep the evidence they already hold (polylogue-1pzmq)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_rows, write_parsed_session_to_archive


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _write(conn: sqlite3.Connection, session: ParsedSession) -> str:
    return write_parsed_session_to_archive(
        conn,
        session,
        content_hash=str(session_content_hash(session)),
        prepared=prepare_session_rows(session),
    )


def _usage_rows(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM session_provider_usage_events WHERE session_id = ? ORDER BY position",
            (session_id,),
        )
    )


def test_finish_reason_survives_all_zero_token_fields(tmp_path: Path) -> None:
    """A Drive chunk reporting only ``finishReason`` keeps a typed usage row.

    ``token_count`` is in ``_SESSION_EVENTS_REDUNDANT_TYPES``, so this row is
    the only place the event lands; the old "some token field is a non-zero
    int" evidence rule deleted it and took the finish reason with it.

    Anti-vacuity: restore that numeric-only rule, or drop the ``finish_reason``
    column, and this is red.
    """
    session = ParsedSession(
        source_name=Provider.GEMINI,
        provider_session_id="finish-reason-only",
        messages=[ParsedMessage(provider_message_id="chunk-1", role=Role.ASSISTANT, text="blocked")],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                source_message_provider_id="chunk-1",
                payload={"type": "token_count", "finish_reason": "SAFETY"},
            )
        ],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        session_id = _write(conn, session)
        rows = _usage_rows(conn, session_id)
        assert len(rows) == 1
        assert rows[0]["finish_reason"] == "SAFETY"
        assert rows[0]["source_message_resolution"] == "resolved"
        assert rows[0]["last_input_tokens"] == 0
    finally:
        conn.close()


def test_stop_reason_survives_an_all_zero_usage_report(tmp_path: Path) -> None:
    """A Claude turn reporting ``stop_reason`` beside zero usage keeps its row.

    ``message_usage`` is redundant-set, so the typed row is the only landing
    place for the reason. The old evidence rule saw four zero token fields and
    deleted the row.

    Anti-vacuity: restore the numeric-only rule (or drop the ``finish_reason``
    column) and both asserts go red.
    """
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="zero-usage-with-stop-reason",
        messages=[ParsedMessage(provider_message_id="a1", role=Role.ASSISTANT, text="cut off")],
        session_events=[
            ParsedSessionEvent(
                event_type="message_usage",
                source_message_provider_id="a1",
                payload={
                    "type": "message_usage",
                    "semantics": "per_message",
                    "last_token_usage": {"input_tokens": 0, "output_tokens": 0},
                    "stop_reason": "max_tokens",
                },
            )
        ],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        session_id = _write(conn, session)
        rows = _usage_rows(conn, session_id)
        assert len(rows) == 1
        assert rows[0]["finish_reason"] == "max_tokens"
    finally:
        conn.close()


def test_payload_with_no_storable_fact_still_writes_no_row(tmp_path: Path) -> None:
    """The evidence gate still exists: zero counters and no reason write nothing.

    This also pins the deliberate limit: a zero-valued counter is not admitted
    on its own, because the parsers coerce unreported counters to zero before
    the writer sees them (polylogue-664l's billing-only Hermes rows depend on
    exactly this).

    Anti-vacuity: make the writer unconditional, or admit any present usage
    mapping, and this is red.
    """
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="framing-only-usage",
        messages=[ParsedMessage(provider_message_id="a1", role=Role.ASSISTANT, text="body")],
        session_events=[
            ParsedSessionEvent(
                event_type="message_usage",
                source_message_provider_id="a1",
                payload={
                    "type": "message_usage",
                    "semantics": "per_message",
                    "last_token_usage": {"input_tokens": 0, "output_tokens": 0},
                },
            )
        ],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        session_id = _write(conn, session)
        assert _usage_rows(conn, session_id) == []
    finally:
        conn.close()


def test_duplicate_provider_message_id_keeps_usage_as_ambiguous(tmp_path: Path) -> None:
    """Usage attached to a duplicated provider id stays, marked ambiguous.

    Two messages declare the same ``provider_message_id``, so the writer gives
    both content-derived ids and excludes the id from its native-id map. The
    usage event naming that id used to be dropped entirely.

    Anti-vacuity: restore the ``source_message_id is not None`` admission guard
    and both asserts go red (no row at all).
    """
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="duplicate-provider-ids",
        messages=[
            ParsedMessage(provider_message_id="dup", role=Role.ASSISTANT, text="first"),
            ParsedMessage(provider_message_id="dup", role=Role.ASSISTANT, text="second"),
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="message_usage",
                source_message_provider_id="dup",
                payload={
                    "type": "message_usage",
                    "semantics": "per_message",
                    "last_token_usage": {"input_tokens": 7, "output_tokens": 11},
                },
            )
        ],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        session_id = _write(conn, session)
        rows = _usage_rows(conn, session_id)
        assert len(rows) == 1
        assert rows[0]["source_message_id"] is None
        assert rows[0]["source_message_provider_id"] == "dup"
        assert rows[0]["source_message_resolution"] == "ambiguous"
        assert (rows[0]["last_input_tokens"], rows[0]["last_output_tokens"]) == (7, 11)
    finally:
        conn.close()


def test_unresolvable_provider_message_id_is_recorded_as_unresolved(tmp_path: Path) -> None:
    """An id naming no message in this session is kept as unresolved, not dropped.

    Anti-vacuity: restore the admission guard and the row disappears.
    """
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="unresolved-provider-id",
        messages=[ParsedMessage(provider_message_id="a1", role=Role.ASSISTANT, text="body")],
        session_events=[
            ParsedSessionEvent(
                event_type="message_usage",
                source_message_provider_id="missing",
                payload={
                    "type": "message_usage",
                    "semantics": "per_message",
                    "last_token_usage": {"input_tokens": 3},
                },
            )
        ],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        session_id = _write(conn, session)
        rows = _usage_rows(conn, session_id)
        assert len(rows) == 1
        assert rows[0]["source_message_id"] is None
        assert rows[0]["source_message_provider_id"] == "missing"
        assert rows[0]["source_message_resolution"] == "unresolved"
    finally:
        conn.close()


def test_session_grain_usage_event_is_not_reported_as_a_failed_attribution(tmp_path: Path) -> None:
    """An event that claims no message id is ``session``, not ``unresolved``.

    Anti-vacuity: collapse the four-state vocabulary to resolved/unresolved and
    this is red.
    """
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="session-grain-usage",
        messages=[ParsedMessage(provider_message_id="a1", role=Role.ASSISTANT, text="body")],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                payload={"type": "token_count", "total_token_usage": {"input_tokens": 50}},
            )
        ],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        session_id = _write(conn, session)
        rows = _usage_rows(conn, session_id)
        assert len(rows) == 1
        assert rows[0]["source_message_provider_id"] is None
        assert rows[0]["source_message_resolution"] == "session"
    finally:
        conn.close()
