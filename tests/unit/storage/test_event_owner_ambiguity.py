"""Duplicated provider message IDs: usage stays typed-ambiguous, other events refuse.

A usage event has a typed ``source_message_resolution`` and is recorded as
``ambiguous`` (polylogue-1pzmq). A session event without such a slot cannot
record which occurrence it belongs to, so the writer refuses it rather than
storing an owner-less row indistinguishable from a session-level event.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.core.message_owner import MessageOwnerAmbiguityError
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_rows
from tests.infra.index_writer import write_fixture_index_session


def _session(event: ParsedSessionEvent) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="duplicate-owner-evidence",
        messages=[
            ParsedMessage(provider_message_id="dup", role=Role.ASSISTANT, text="first"),
            ParsedMessage(provider_message_id="dup", role=Role.ASSISTANT, text="second"),
        ],
        session_events=[event],
    )


def _write(path: Path, session: ParsedSession) -> sqlite3.Connection:
    conn = connect_measured(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    write_fixture_index_session(
        conn,
        session,
        content_hash=str(session_content_hash(session)),
        prepared_rows=prepare_session_rows(session),
    )
    return conn


def test_usage_on_duplicate_provider_id_is_recorded_ambiguous(tmp_path: Path) -> None:
    usage = ParsedSessionEvent(
        event_type="message_usage",
        source_message_provider_id="dup",
        payload={"type": "message_usage", "semantics": "per_message", "last_token_usage": {"input_tokens": 5}},
    )
    conn = _write(tmp_path / "index.db", _session(usage))
    try:
        rows = list(conn.execute("SELECT * FROM session_provider_usage_events"))
        assert [(row["source_message_id"], row["source_message_resolution"]) for row in rows] == [(None, "ambiguous")]
    finally:
        conn.close()


def test_owner_bearing_event_on_duplicate_provider_id_refuses(tmp_path: Path) -> None:
    event = ParsedSessionEvent(
        event_type="model_configuration",
        source_message_provider_id="dup",
        payload={"model": "synthetic-model"},
    )
    with pytest.raises(MessageOwnerAmbiguityError):
        _write(tmp_path / "index.db", _session(event)).close()
