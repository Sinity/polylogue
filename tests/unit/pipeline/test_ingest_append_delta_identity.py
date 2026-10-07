"""Append identity laws at the current archive writer boundary.

The former ``ingest_batch._append_delta_payload`` preparation carrier is
retired. Live append admission now records literal Raw revisions and the
resident owner publishes their derived message rows. Its current route is
covered by ``test_live_batch_support.py``; this focused writer law keeps the
identity case that can be isolated here: a new provider-native message with
the same prose as an earlier message remains a distinct append row.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from tests.infra.index_writer import fixture_index_connection, write_fixture_index_session


def _session(message_id: str, text: str) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="append-delta-identity",
        title="Append delta identity",
        messages=[
            ParsedMessage(
                provider_message_id=message_id,
                role=Role.USER,
                text=text,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
            )
        ],
    )


def test_append_keeps_a_new_native_id_even_when_content_repeats(tmp_path: Path) -> None:
    """Content equality does not turn a distinct native message into replay."""
    with fixture_index_connection(tmp_path / "index.db") as conn:
        initial = _session("m0", "same earlier prompt")
        session_id = write_fixture_index_session(conn, initial)
        appended = _session("m1", "same earlier prompt")

        write_fixture_index_session(conn, appended, merge_append=True)
        rows = conn.execute(
            """SELECT m.native_id, b.text
               FROM messages AS m JOIN blocks AS b ON b.message_id = m.message_id
               WHERE m.session_id = ? AND b.block_type = 'text'
               ORDER BY m.position, b.position""",
            (session_id,),
        ).fetchall()

    assert [tuple(row) for row in rows] == [
        ("m0", "same earlier prompt"),
        ("m1", "same earlier prompt"),
    ]
