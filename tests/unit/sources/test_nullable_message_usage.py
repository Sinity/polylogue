"""Message-grain usage keeps absent counters distinct from measured zero."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Origin, Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers import codex, drive, local_agent
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.hydrators import message_from_record
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_rows, write_parsed_session_to_archive
from polylogue.storage.sqlite.queries.mappers_archive import bind_message_row_mapper
from polylogue.surfaces.payloads import message_render_envelope_from_domain


def test_supported_json_usage_extractors_preserve_missing_and_zero() -> None:
    assert codex._token_usage({}) == {
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
    }
    assert codex._token_usage({"usage": {"input_tokens": 0, "output_tokens": 0}}) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
    }

    assert drive._usage_fields({}, role=Role.USER) == {"input_tokens": None, "output_tokens": None}
    assert drive._usage_fields({"tokenCount": 0}, role=Role.USER) == {"input_tokens": 0, "output_tokens": 0}

    assert local_agent._token_usage_fields({"usage": {}})["input_tokens"] is None
    assert local_agent._token_usage_fields({"usage": {"input": 0}})["input_tokens"] == 0


def test_unknown_and_measured_zero_survive_prepared_write_and_public_message(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="nullable-message-usage",
        messages=[
            ParsedMessage(provider_message_id="unknown", role=Role.ASSISTANT, text="unknown"),
            ParsedMessage(
                provider_message_id="zero",
                role=Role.ASSISTANT,
                text="zero",
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            ParsedMessage(
                provider_message_id="known",
                role=Role.ASSISTANT,
                text="known",
                input_tokens=10,
                output_tokens=20,
                cache_read_tokens=3,
                cache_write_tokens=4,
            ),
        ],
    )
    path = tmp_path / "index.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        session_id = write_parsed_session_to_archive(
            conn,
            session,
            content_hash=str(session_content_hash(session)),
            prepared=prepare_session_rows(session),
        )
        rows = conn.execute(
            "SELECT message_id, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens "
            "FROM messages WHERE session_id = ? ORDER BY position",
            (session_id,),
        ).fetchall()
        assert [tuple(row[1:]) for row in rows] == [
            (None, None, None, None),
            (0, 0, 0, 0),
            (10, 20, 3, 4),
        ]

        from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import MESSAGES_SPEC

        cursor = conn.execute(
            f"SELECT {MESSAGES_SPEC.record_select_column_names('m')} FROM messages AS m "
            "JOIN sessions AS s ON s.session_id = m.session_id ORDER BY m.position"
        )
        decode = bind_message_row_mapper(tuple(column[0] for column in cursor.description or ()))
        records = [decode(row) for row in cursor.fetchall()]
        domains = [message_from_record(record, attachments=[], origin=Origin.CODEX_SESSION) for record in records]
        envelopes = [message_render_envelope_from_domain(message, session_id=session_id) for message in domains]
        assert domains[0].input_tokens is None
        assert domains[1].input_tokens == 0
        assert envelopes[0].input_tokens is None
        assert envelopes[1].input_tokens == 0
    finally:
        conn.close()
