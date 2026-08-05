from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.parsers import antigravity
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    read_archive_session_envelope,
    search_archive_blocks,
    write_parsed_session_to_archive,
)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _load_drive_payload(filename: str) -> JSONDocument:
    fixture_path = Path(__file__).resolve().parent.parent.parent / "data" / "gemini_chunked_prompt" / filename
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(JSONDocument, payload)


def _antigravity_payload() -> JSONDocument:
    summary = antigravity.AntigravitySessionSummary(
        cascade_id="anti-fixture-001",
        title="Fixture antigravity",
        workspace_name="research",
        snippet="fixture run",
        last_modified_time="2026-01-02T10:15:00+00:00",
    )
    return antigravity.markdown_export_payload(
        summary,
        """### User Input

check indexing

### Planner Response
index write-back complete
""",
    )


def test_archive_tiers_writer_materializes_drive_payload(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    [session] = parse_payload(Provider.DRIVE, _load_drive_payload("text_only_prompt.json"), "drive-fixture")

    session_id = write_parsed_session_to_archive(conn, session)
    envelope = read_archive_session_envelope(conn, session_id)

    assert envelope.origin == "aistudio-drive"
    assert envelope.native_id == "gem-text-only"
    assert envelope.title == "Capital trivia"
    message_ids = [message.message_id for message in envelope.messages]
    assert envelope.active_leaf_message_id == message_ids[-1]
    assert message_ids == [f"aistudio-drive:gem-text-only:{position}.0" for position in range(4)]
    assert [message.role for message in envelope.messages] == ["user", "assistant", "user", "assistant"]
    assert [message.is_active_leaf for message in envelope.messages] == [False, False, False, True]
    assert search_archive_blocks(conn, "Paris") == [f"{message_ids[1]}:0"]


def test_archive_tiers_writer_materializes_antigravity_payload(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    [session] = parse_payload(Provider.ANTIGRAVITY, _antigravity_payload(), "ag-fixture")

    session_id = write_parsed_session_to_archive(conn, session)
    envelope = read_archive_session_envelope(conn, session_id)

    assert envelope.origin == "antigravity-session"
    assert envelope.native_id == "anti-fixture-001"
    message_ids = [message.message_id for message in envelope.messages]
    assert len(message_ids) == len(set(message_ids)) == 2
    assert all(message_id.startswith("antigravity-session:anti-fixture-001:synthetic-") for message_id in message_ids)
    assert [message.role for message in envelope.messages] == ["user", "assistant"]
    assert envelope.messages[1].is_active_leaf is True
    assert search_archive_blocks(conn, "complete") == [f"{message_ids[1]}:0"]
