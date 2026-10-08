"""Message-grain usage keeps absent counters distinct from measured zero."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.core.enums import Origin, Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers import antigravity, browser_capture, chatgpt, codex, drive, grok, local_agent
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.hydrators import message_from_record
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_rows
from polylogue.storage.sqlite.queries.mappers_archive import bind_message_row_mapper
from polylogue.surfaces.payloads import message_render_envelope_from_domain
from tests.infra.index_writer import close_fixture_index_connection, write_fixture_index_session


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
    # The Index writer admits only connections from its measured creator.
    conn = connect_measured(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        session_id = write_fixture_index_session(
            conn,
            session,
            content_hash=str(session_content_hash(session)),
            prepared_rows=prepare_session_rows(session),
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
        close_fixture_index_connection(conn)


def _chatgpt_payload() -> dict[str, object]:
    return {
        "title": "counter-less export",
        "create_time": 1.0,
        "mapping": {
            "a": {
                "id": "a",
                "parent": None,
                "children": ["b"],
                "message": {
                    "id": "a",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hi"]},
                    "create_time": 1.0,
                },
            },
            "b": {
                "id": "b",
                "parent": "a",
                "children": [],
                "message": {
                    "id": "b",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["hello"]},
                    "create_time": 2.0,
                },
            },
        },
    }


def _grok_payload() -> dict[str, object]:
    return {
        "conversation": {"title": "counter-less chat", "create_time": "2026-01-01T00:00:00Z"},
        "responses": [
            {"sender": "user", "message": "hello", "create_time": "2026-01-01T00:00:01Z"},
            {"sender": "grok", "message": "hi there", "create_time": "2026-01-01T00:00:02Z"},
        ],
    }


def _browser_capture_payload() -> dict[str, object]:
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {
            "source_url": "https://chatgpt.com/c/conv-123",
            "page_title": "ChatGPT - Work plan",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-dom-v1",
        },
        "session": {
            "provider": "chatgpt",
            "provider_session_id": "conv-123",
            "title": "Work plan",
            "turns": [
                {"provider_turn_id": "u1", "role": "user", "text": "Draft"},
                {"provider_turn_id": "a1", "role": "assistant", "text": "Here"},
            ],
        },
    }


def test_counter_less_parsers_leave_every_message_counter_unknown() -> None:
    """Parsers whose providers report no per-message usage must emit ``None``.

    These four never touch the usage fields, so the value is decided entirely
    by ``ParsedMessage``'s defaults. Anti-vacuity: flipping any of those four
    defaults (or a parser writing ``0`` for an unreported counter) turns
    every tuple below into ``(0, 0, 0, 0)`` and makes this red -- which is
    the exact false measured-zero polylogue-qgyuj removed from the storage
    and read paths.
    """
    sessions = {
        "chatgpt": chatgpt.parse(_chatgpt_payload(), "conv"),
        "grok": grok.parse_conversation(_grok_payload(), "grok-1"),
        "browser_capture": browser_capture.parse(_browser_capture_payload(), "conv-123"),
        "antigravity": antigravity.parse_markdown_export(
            "### User Input\n\nhello\n\n### Assistant Response\n\nhi",
            antigravity.AntigravitySessionSummary(cascade_id="cascade"),
        ),
    }
    for name, session in sessions.items():
        assert session.messages, f"{name} fixture produced no messages"
        for message in session.messages:
            assert (
                message.input_tokens,
                message.output_tokens,
                message.cache_read_tokens,
                message.cache_write_tokens,
            ) == (None, None, None, None), f"{name} coerced an unreported counter"


def test_public_message_model_defaults_usage_to_unknown() -> None:
    """A ``Message`` built without usage reports unknown, never measured zero.

    Anti-vacuity: restoring ``= 0`` on the four ``Message`` counters (the
    surviving zero-coercion polylogue-qgyuj's storage/read work left behind)
    makes both halves of this red.
    """
    message = Message(id="s:1", role=Role.ASSISTANT, text="no usage reported")
    assert message.input_tokens is None
    assert message.output_tokens is None
    assert message.cache_read_tokens is None
    assert message.cache_write_tokens is None
    envelope = message_render_envelope_from_domain(message, session_id="s")
    assert envelope.input_tokens is None
    assert envelope.cache_write_tokens is None


def _usage(message: object) -> tuple[object, ...]:
    return (
        getattr(message, "model_name", None),
        getattr(message, "input_tokens", None),
        getattr(message, "output_tokens", None),
        getattr(message, "cache_read_tokens", None),
        getattr(message, "cache_write_tokens", None),
    )


@pytest.mark.asyncio
async def test_composed_and_bounded_message_reads_carry_stored_usage(workspace_env: dict[str, Path]) -> None:
    """``get_session`` and the bounded message page report the stored counters.

    The storage tier keeps ``NULL`` apart from a measured zero, but both
    production read projections used to omit the usage columns, so every
    hydrated message reported unknown usage and no model even when the
    provider measured it. Anti-vacuity: drop the usage fields from
    ``ArchiveMessageRow`` or ``ArchiveMessageQueryRow`` (or their hydration
    dispositions) and the ``known`` and ``zero`` rows read back as
    ``(None, None, None, None, None)``.
    """
    from polylogue.api import Polylogue
    from polylogue.archive.hydration import archive_message_query_row_to_domain
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_templates import run_off_event_loop
    from tests.infra.live_ingest import write_session_sync
    from tests.infra.storage_records import db_setup

    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="usage-through-reads",
        messages=[
            ParsedMessage(provider_message_id="unknown", role=Role.ASSISTANT, text="unknown", position=0),
            ParsedMessage(
                provider_message_id="zero",
                role=Role.ASSISTANT,
                text="zero",
                position=1,
                model_name="claude-opus-5",
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            ParsedMessage(
                provider_message_id="known",
                role=Role.ASSISTANT,
                text="known",
                position=2,
                model_name="claude-opus-5",
                input_tokens=10,
                output_tokens=20,
                cache_read_tokens=3,
                cache_write_tokens=None,
            ),
        ],
    )
    archive_root = workspace_env["archive_root"]
    # Seeding takes a synchronous write lease, which may not block this test's loop.
    session_id = run_off_event_loop(lambda: write_session_sync(db_setup(workspace_env), session))
    expected = [
        (None, None, None, None, None),
        ("claude-opus-5", 0, 0, 0, 0),
        ("claude-opus-5", 10, 20, 3, None),
    ]

    api = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    composed = await api.get_session(session_id)
    assert composed is not None
    assert [_usage(message) for message in composed.messages.to_list()] == expected
    envelopes = [message_render_envelope_from_domain(m, session_id=session_id) for m in composed.messages.to_list()]
    assert [_usage(envelope) for envelope in envelopes] == expected

    with ArchiveStore(archive_root, initialize=False, read_only=True) as archive:
        page_rows = archive.query_session_messages((session_id,), limit=50, offset=0)
    assert [_usage(archive_message_query_row_to_domain(row)) for row in page_rows] == expected
