"""The canonical append writer advances raw evidence without adding messages."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from polylogue.core.enums import Provider, Role
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import ArchiveWriteOutcome
from tests.infra.archive_templates import run_archive_fixture_write
from tests.infra.index_writer import fixture_index_connection, write_fixture_index_session


def test_duplicate_only_append_refreshes_raw_link_without_adding_messages(tmp_path: Path) -> None:
    """A committed append raw can advance the session link with no message delta."""
    archive_root = tmp_path

    def acquire(payload: bytes, *, name: str, acquired_at_ms: int) -> str:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=name,
                canonical_source_path=name,
                acquired_at_ms=acquired_at_ms,
            )

    original_payload = (
        b'{"type":"session_meta","payload":{"id":"stable-session"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m1",'
        b'"role":"user","content":[{"type":"input_text","text":"stable prompt"}]}}\n'
    )
    duplicate_payload = original_payload + original_payload.splitlines(keepends=True)[1]
    raw_before = asyncio.run(
        run_archive_fixture_write(
            archive_root,
            lambda: acquire(original_payload, name="stable-session.jsonl", acquired_at_ms=1),
        )
    )
    raw_after = asyncio.run(
        run_archive_fixture_write(
            archive_root,
            lambda: acquire(duplicate_payload, name="stable-session.jsonl", acquired_at_ms=2),
        )
    )
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="stable-session",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="stable prompt")],
    )

    with fixture_index_connection(archive_root / "index.db") as conn:
        initial_outcome: list[ArchiveWriteOutcome] = []
        session_id = write_fixture_index_session(
            conn,
            session,
            raw_id=raw_before,
            write_outcome=initial_outcome,
        )
        append_outcome: list[ArchiveWriteOutcome] = []
        write_fixture_index_session(
            conn,
            session,
            raw_id=raw_after,
            merge_append=True,
            write_outcome=append_outcome,
        )
        conn.commit()
        row = conn.execute(
            "SELECT raw_id, message_count FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        message_rows = conn.execute(
            "SELECT native_id, role FROM messages WHERE session_id = ? ORDER BY position",
            (session_id,),
        ).fetchall()

    with sqlite3.connect(archive_root / "source.db") as conn:
        raw_count = conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]

    assert initial_outcome[-1].wrote is True
    assert append_outcome[-1] == ArchiveWriteOutcome(session_id=session_id, wrote=False)
    assert row is not None
    assert row[0] == raw_after != raw_before
    assert row[1] == 1
    assert [(message[0], message[1]) for message in message_rows] == [("m1", "user")]
    assert raw_count == 2
