from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider, ToolOutcome, ToolResultUnknownReason
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.sources.parsers.claude.code_parser import parse_code
from polylogue.sources.parsers.codex import parse as parse_codex
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


@pytest.mark.parametrize("reason", [ToolResultUnknownReason.NOT_REPORTED, ToolResultUnknownReason.DISTRUSTED])
def test_declared_unknown_outcome_is_admitted_and_preserves_reason(
    reason: ToolResultUnknownReason, tmp_path: Path
) -> None:
    conn = _connect(tmp_path / f"unknown-{reason.value}.db")
    try:
        session_id = write_parsed_session_to_archive(
            conn,
            _session(
                Provider.CLAUDE_CODE,
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT,
                    tool_id="call-1",
                    text="provider omitted a trusted verdict",
                    outcome_unknown_reason=reason.value,
                ),
            ),
        )
        row = conn.execute(
            """
            SELECT block_type, tool_outcome, tool_result_is_error,
                   tool_result_exit_code, tool_result_outcome_unknown_reason
            FROM blocks WHERE session_id = ? ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        by_type = {item["block_type"]: item for item in row}
        assert {block_type: item["tool_outcome"] for block_type, item in by_type.items()} == {
            "tool_use": ToolOutcome.UNKNOWN.value,
            "tool_result": ToolOutcome.UNKNOWN.value,
        }
        result_row = by_type["tool_result"]
        assert result_row["tool_result_is_error"] is None
        assert result_row["tool_result_exit_code"] is None
        assert result_row["tool_result_outcome_unknown_reason"] == reason.value
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("provider", "parse_session", "payload"),
    [
        (
            Provider.CLAUDE_CODE,
            lambda payload: parse_code(payload, "claude-parser-unknown"),
            [
                {
                    "type": "assistant",
                    "uuid": "assistant-1",
                    "sessionId": "claude-parser-unknown",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "call-1", "name": "Bash", "input": {}}],
                    },
                },
                {
                    "type": "user",
                    "uuid": "user-1",
                    "sessionId": "claude-parser-unknown",
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "plain output"}],
                    },
                },
            ],
        ),
        (
            Provider.CODEX,
            lambda payload: parse_codex(payload, "codex-parser-unknown"),
            [
                {"type": "session_meta", "payload": {"id": "codex-parser-unknown"}},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "Bash",
                        "arguments": "{}",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": "plain output",
                    },
                },
            ],
        ),
    ],
    ids=["claude-code", "codex"],
)
def test_real_parser_unknown_shape_is_admitted_by_writer(
    provider: Provider,
    parse_session: Callable[[list[dict[str, object]]], ParsedSession],
    payload: list[dict[str, object]],
    tmp_path: Path,
) -> None:
    """Parser-produced no-verdict results are admitted without inventing success."""
    conn = _connect(tmp_path / f"parser-{provider.value}.db")
    try:
        session = parse_session(payload)
        result_blocks = [
            block for message in session.messages for block in message.blocks if block.type is BlockType.TOOL_RESULT
        ]
        assert len(result_blocks) == 1
        assert result_blocks[0].outcome_unknown_reason == ToolResultUnknownReason.NOT_REPORTED.value

        session_id = write_parsed_session_to_archive(conn, session)
        row = conn.execute(
            """
            SELECT tool_outcome, tool_result_is_error, tool_result_exit_code,
                   tool_result_outcome_unknown_reason
            FROM blocks WHERE session_id = ? AND block_type = 'tool_result'
            """,
            (session_id,),
        ).fetchone()
        assert row is not None
        assert tuple(row) == (
            ToolOutcome.UNKNOWN.value,
            None,
            None,
            ToolResultUnknownReason.NOT_REPORTED.value,
        )
    finally:
        conn.close()


@pytest.mark.parametrize("provider", [Provider.CLAUDE_CODE, Provider.CODEX, Provider.HERMES])
def test_declared_origin_without_verdict_is_admitted_as_unknown(provider: Provider, tmp_path: Path) -> None:
    """Declared origins preserve a missing provider verdict as typed unknown.

    Anti-vacuity: removing the origin declaration makes this normalized
    no-verdict result refuse at the writer seam.
    """
    conn = _connect(tmp_path / f"declared-{provider.value}.db")
    try:
        session_id = write_parsed_session_to_archive(
            conn,
            _session(
                provider,
                ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id="call-1", text="provider omitted a verdict"),
            ),
        )
        row = conn.execute(
            "SELECT tool_outcome, tool_result_is_error, tool_result_outcome_unknown_reason "
            "FROM blocks WHERE session_id = ? AND block_type = 'tool_result'",
            (session_id,),
        ).fetchone()
        assert tuple(row) == (ToolOutcome.UNKNOWN.value, None, ToolResultUnknownReason.NOT_REPORTED.value)
    finally:
        conn.close()


@pytest.mark.parametrize("content_type", ["execution_output", "computer_output", "citable_code_output"])
def test_chatgpt_declared_unknown_output_is_admitted_without_inventing_success(
    content_type: str, tmp_path: Path
) -> None:
    """Recognized ChatGPT result shapes with no status become explicit unknown.

    Anti-vacuity: removing the writer's content-type normalization makes this
    supported parser output raise instead of storing ``unknown``.
    """
    conn = _connect(tmp_path / f"chatgpt-{content_type}.db")
    try:
        session_id = write_parsed_session_to_archive(
            conn,
            _session(
                Provider.CHATGPT,
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT,
                    tool_id="call-1",
                    text="provider has not reported a terminal status",
                    metadata={"content_type": content_type},
                ),
            ),
        )
        row = conn.execute(
            "SELECT tool_outcome, tool_result_is_error, tool_result_outcome_unknown_reason "
            "FROM blocks WHERE session_id = ? AND block_type = 'tool_result'",
            (session_id,),
        ).fetchone()
        assert tuple(row) == (ToolOutcome.UNKNOWN.value, None, ToolResultUnknownReason.NOT_REPORTED.value)
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


def test_merge_selects_unknown_verdict_as_one_atomic_legacy_projection(tmp_path: Path) -> None:
    """A newer unknown result cannot inherit an older success flag.

    Anti-vacuity: removing correlated-field normalization from the merge
    leaves ``tool_outcome='unknown'`` paired with ``is_error=0``.
    """
    conn = _connect(tmp_path / "merged-verdict.db")
    try:
        first = _session(
            Provider.CLAUDE_CODE,
            ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id="call-1", text="done", is_error=False),
        )
        second = _session(
            Provider.CLAUDE_CODE,
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id="call-1",
                text="done",
                outcome_unknown_reason=ToolResultUnknownReason.NOT_REPORTED.value,
            ),
        )
        session_id = write_parsed_session_to_archive(conn, first)
        write_parsed_session_to_archive(conn, second)
        row = conn.execute(
            "SELECT tool_outcome, tool_result_is_error, tool_result_outcome_unknown_reason "
            "FROM blocks WHERE session_id = ? AND block_type = 'tool_result'",
            (session_id,),
        ).fetchone()
        assert tuple(row) == (ToolOutcome.UNKNOWN.value, None, ToolResultUnknownReason.NOT_REPORTED.value)
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("first_exit_code", "second_is_error"),
    [(2, False), (0, True)],
)
def test_merge_known_verdict_clears_conflicting_legacy_exit_code(
    first_exit_code: int, second_is_error: bool, tmp_path: Path
) -> None:
    conn = _connect(tmp_path / f"merged-exit-{first_exit_code}.db")
    try:
        first = _session(
            Provider.CLAUDE_CODE,
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id="call-1",
                text="first",
                is_error=first_exit_code != 0,
                exit_code=first_exit_code,
            ),
        )
        second = _session(
            Provider.CLAUDE_CODE,
            ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id="call-1", text="second", is_error=second_is_error),
        )
        session_id = write_parsed_session_to_archive(conn, first)
        write_parsed_session_to_archive(conn, second)
        row = conn.execute(
            "SELECT tool_outcome, tool_result_is_error, tool_result_exit_code FROM blocks WHERE session_id = ? AND block_type = 'tool_result'",
            (session_id,),
        ).fetchone()
        assert tuple(row) == (
            ToolOutcome.ERROR.value if second_is_error else ToolOutcome.OK.value,
            int(second_is_error),
            None,
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
