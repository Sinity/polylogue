from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider, ToolOutcome
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    read_archive_session_envelope,
    write_parsed_session_to_archive,
)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _session(provider: Provider, result: ParsedContentBlock | None, *, tool_id: str = "call-1") -> ParsedSession:
    use = ParsedMessage(
        provider_message_id="use",
        role=Role.ASSISTANT,
        blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, tool_id=tool_id, tool_name="run")],
    )
    messages = [use]
    if result is not None:
        messages.append(ParsedMessage(provider_message_id="result", role=Role.TOOL, blocks=[result]))
    return ParsedSession(source_name=provider, provider_session_id="outcome", messages=messages)


@pytest.mark.parametrize("provider", list(Provider))
def test_structured_outcome_round_trips_for_each_provider_wire(provider: Provider, tmp_path: Path) -> None:
    conn = _connect(tmp_path / f"{provider.value}.db")
    try:
        session_id = write_parsed_session_to_archive(
            conn,
            _session(
                provider,
                ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id="call-1", text="done", is_error=False),
            ),
        )
        row = conn.execute(
            """
            SELECT b.block_type, b.tool_outcome
            FROM blocks b JOIN messages m ON m.message_id = b.message_id
            WHERE b.session_id = ? ORDER BY m.position, b.position
            """,
            (session_id,),
        ).fetchall()
        assert [(item["block_type"], item["tool_outcome"]) for item in row] == [
            ("tool_use", ToolOutcome.OK.value),
            ("tool_result", ToolOutcome.OK.value),
        ]
        envelope = read_archive_session_envelope(conn, session_id)
        assert envelope.messages[1].blocks[0].tool_outcome is ToolOutcome.OK
    finally:
        conn.close()


def test_sidecar_execution_evidence_derives_result_outcome(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "sidecar.db")
    try:
        session_id = write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="sidecar-outcome",
                messages=[
                    ParsedMessage(
                        provider_message_id="use",
                        role=Role.ASSISTANT,
                        blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, tool_id="call-1")],
                    ),
                    ParsedMessage(
                        provider_message_id="result",
                        role=Role.TOOL,
                        blocks=[ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id="call-1", text="failed")],
                    ),
                ],
                session_events=[
                    ParsedSessionEvent(
                        event_type="claude_tool_execution_result",
                        payload={"tool_use_id": "call-1", "exit_code": 2},
                    )
                ],
            ),
        )
        outcomes = conn.execute(
            """
            SELECT b.tool_outcome, b.tool_result_exit_code
            FROM blocks b JOIN messages m ON m.message_id = b.message_id
            WHERE b.session_id = ? ORDER BY m.position, b.position
            """,
            (session_id,),
        ).fetchall()
        assert [(row["tool_outcome"], row["tool_result_exit_code"]) for row in outcomes] == [
            (ToolOutcome.ERROR.value, None),
            (ToolOutcome.ERROR.value, 2),
        ]
    finally:
        conn.close()


def test_unpaired_tool_use_is_no_result(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "no-result.db")
    try:
        session_id = write_parsed_session_to_archive(conn, _session(Provider.CODEX, None))
        assert conn.execute("SELECT tool_outcome FROM blocks WHERE session_id = ?", (session_id,)).fetchone()[0] == (
            ToolOutcome.NO_RESULT.value
        )
    finally:
        conn.close()


def test_result_without_structural_evidence_refuses_write(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "refused.db")
    try:
        with pytest.raises(ValueError, match="chatgpt-export.*unsupported tool_result block shape"):
            write_parsed_session_to_archive(
                conn,
                _session(
                    Provider.CHATGPT,
                    ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id="call-1", text="looks successful"),
                ),
            )
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    finally:
        conn.close()
