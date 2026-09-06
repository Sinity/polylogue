"""The candidate census counts the four shapes the outcome contract forbids.

Anti-vacuity: each defect is injected by writing the forbidden row directly,
past the producer boundary and the writer seam that normally refuse it. A
census that only counted rows, or read ``tool_outcome`` without cross-checking
the reason and the public projection, reports zero for every one of them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from devtools.tool_outcome_census import UNPAIRED_CONSTRUCT, compute_tool_outcome_census
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider, ToolOutcome, ToolResultUnknownReason
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _session(result: ParsedContentBlock, *, session_id: str = "census-1") -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=session_id,
        messages=[
            ParsedMessage(
                provider_message_id="use",
                role=Role.ASSISTANT,
                blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, tool_id="call-1", tool_name="Bash")],
            ),
            ParsedMessage(provider_message_id="result", role=Role.TOOL, blocks=[result]),
        ],
    )


def _write_clean(conn: sqlite3.Connection) -> None:
    write_parsed_session_to_archive(
        conn,
        _session(
            ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id="call-1", text="done", is_error=False),
            session_id="census-known",
        ),
    )
    write_parsed_session_to_archive(
        conn,
        _session(
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id="call-1",
                text="no verdict",
                outcome_unknown_reason=ToolResultUnknownReason.NOT_REPORTED.value,
            ),
            session_id="census-unknown",
        ),
    )


def _corrupt_result_row(conn: sqlite3.Connection, **columns: object) -> None:
    """Set columns on the single tool_result row of the unknown-outcome session."""
    assignments = ", ".join(f"{name} = ?" for name in columns)
    conn.execute(
        f"UPDATE blocks SET {assignments} WHERE block_type = 'tool_result' "
        "AND session_id = 'claude-code-session:census-unknown'",
        tuple(columns.values()),
    )


def test_a_conforming_archive_reports_no_defects(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "clean.db")
    try:
        _write_clean(conn)
        census = compute_tool_outcome_census(conn)
        assert census.total_tool_results == 2
        assert census.defect_counts == {
            "unknown_without_reason": 0,
            "known_with_reason": 0,
            "unsupported_without_owner": 0,
            "public_projection_disagreement": 0,
        }
        assert census.is_clean
        constructs = {key[1] for key in census.by_classification}
        assert constructs == {"Bash"} and UNPAIRED_CONSTRUCT not in constructs
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("columns", "defect"),
    [
        ({"tool_result_outcome_unknown_reason": None}, "unknown_without_reason"),
        (
            {"tool_outcome": None, "tool_result_outcome_unknown_reason": None},
            "unknown_without_reason",
        ),
        (
            {"tool_outcome": ToolOutcome.OK.value, "tool_result_is_error": 0},
            "known_with_reason",
        ),
        (
            {"tool_result_outcome_unknown_reason": ToolResultUnknownReason.SOURCE_TRUNCATED.value},
            "unsupported_without_owner",
        ),
    ],
    ids=["unknown-without-reason", "null-outcome", "known-with-reason", "reason-with-no-owner"],
)
def test_each_forbidden_shape_is_counted(columns: dict[str, object], defect: str, tmp_path: Path) -> None:
    conn = _connect(tmp_path / f"{defect}.db")
    try:
        _write_clean(conn)
        conn.execute("PRAGMA ignore_check_constraints = ON")
        _corrupt_result_row(conn, **columns)
        census = compute_tool_outcome_census(conn)
        assert census.defect_counts[defect] == 1, census.defect_counts
        assert not census.is_clean
    finally:
        conn.close()


def test_a_public_projection_that_contradicts_the_block_is_counted(tmp_path: Path) -> None:
    """``actions.result_state`` follows the invocation; a divergent result row is a defect."""
    conn = _connect(tmp_path / "projection.db")
    try:
        _write_clean(conn)
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            "UPDATE action_pairs SET tool_outcome = ? WHERE session_id = 'claude-code-session:census-unknown'",
            (ToolOutcome.OK.value,),
        )
        census = compute_tool_outcome_census(conn)
        assert census.public_projection_disagreement == 1, census.defect_counts
    finally:
        conn.close()
