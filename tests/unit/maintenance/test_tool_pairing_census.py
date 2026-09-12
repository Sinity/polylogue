"""The pairing census classifies every gap it counts, and counts every gap.

Anti-vacuity: a census that classified nothing would still report zero
unclassified rows, so each case asserts the cohort's own class and the
denominators it was derived from. The conservation assertion is red whenever a
call is neither paired nor counted as a gap.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from devtools.tool_pairing_census import (
    CLASS_SOURCE_OMISSION,
    CLASS_SOURCE_TRUNCATED,
    CLASS_UNKNOWN,
    CLASS_UNSUPPORTED_CONSTRUCT,
    COMPLETION_SETTLED,
    COMPLETION_SUPERSEDED,
    POSITION_INTERIOR,
    POSITION_TAIL,
    SOURCE_BYTES_ABSENT,
    SOURCE_PRESENT,
    CensusArgs,
    SourceState,
    _classify_call,
    _classify_result,
    build_report,
)
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from tests.infra.live_ingest import write_session_sync


def _codex_session() -> ParsedSession:
    """A Codex session: one answered call, one code-mode child with no answer, one tail call."""
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="census-codex",
        messages=[
            ParsedMessage(
                provider_message_id="m0",
                role=Role.ASSISTANT,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="shell",
                        tool_id="call_answered",
                        tool_input={"command": "ls"},
                    ),
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        tool_id="call_answered",
                        text="ok",
                        is_error=False,
                    ),
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="exec_command",
                        tool_id="call_script::polylogue-child::1",
                        tool_input={"command": "echo 2"},
                    ),
                ],
            ),
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="shell",
                        tool_id="call_interrupted",
                        tool_input={"command": "sleep"},
                    )
                ],
            ),
        ],
    )


def _census(db_path: Path) -> dict[str, Any]:
    return build_report(
        CensusArgs(
            archive_root=db_path.parent,
            index_db=db_path,
            source_db=db_path.parent / "absent-source.db",
            near_tail_messages=0,
            check_source=False,
            replay_sessions=0,
            json=True,
        )
    )


def test_census_classifies_every_gap_it_counts(test_db: Path) -> None:
    write_session_sync(test_db, _codex_session())
    report = _census(test_db)

    denominator = report["denominator"]
    assert denominator["tool_calls"] == 3
    assert denominator["paired_calls"] == 1
    assert denominator["no_result_calls"] == 2
    assert denominator["physical_results"] == 1
    assert denominator["unmatched_results"] == 0

    verdict = report["verdict"]
    assert verdict["unclassified"] == 0
    assert verdict["unexplained"] == 0
    assert CLASS_UNKNOWN not in report["classification_totals"]

    cohorts = {(row["construct"], row["position"]): (row["classification"], row["count"]) for row in report["calls"]}
    assert cohorts[("codex_code_mode_child", POSITION_INTERIOR)] == (CLASS_UNSUPPORTED_CONSTRUCT, 1)
    assert cohorts[("provider_native_call", POSITION_TAIL)][1] == 1


def test_census_conservation_covers_every_origin(test_db: Path) -> None:
    write_session_sync(test_db, _codex_session())
    report = _census(test_db)
    conservation = {row["origin"]: row for row in report["conservation"]}
    codex = conservation["codex-session"]
    assert codex["calls"] == codex["paired"] + codex["no_result"]
    assert codex["unexplained"] == 0


def test_census_counts_an_unmatched_physical_result(test_db: Path) -> None:
    """A stored answer no call claims is reported, not silently dropped."""
    session = _codex_session()
    extra = ParsedMessage(
        provider_message_id="m2",
        role=Role.USER,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id="call_never_made",
                text="orphan",
                is_error=False,
            )
        ],
    )
    write_session_sync(test_db, session.model_copy(update={"messages": [*session.messages, extra]}))

    report = _census(test_db)
    assert report["denominator"]["unmatched_results"] == 1
    orphans = [row for row in report["results"] if row["count"]]
    assert orphans and orphans[0]["classification"] == CLASS_SOURCE_OMISSION


def test_census_reports_its_query_plan_and_runtime(test_db: Path) -> None:
    """The report carries the plan it ran, so an unindexed regression is visible."""
    write_session_sync(test_db, _codex_session())
    report = _census(test_db)
    plans = {item["query"]: item for item in report["plan"]}
    assert {"calls_by_identity", "results_by_identity"} <= set(plans)
    assert all(isinstance(item["seconds"], float) for item in plans.values())
    result_plan = " ".join(plans["results_by_identity"]["steps"])
    assert "idx_blocks_tool_result_outcome" in result_plan


def test_census_reads_an_index_whose_derived_schema_predates_the_code(test_db: Path) -> None:
    """A stale derived tier is the normal case: report its shape, do not refuse it."""
    write_session_sync(test_db, _codex_session())
    conn = sqlite3.connect(str(test_db))
    try:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(action_pairs)").fetchall()}
    finally:
        conn.close()
    report = _census(test_db)
    assert report["derived_schema"]["action_pairs_has_tool_outcome"] == ("tool_outcome" in columns)


def test_unowned_result_with_surviving_source_stays_unknown() -> None:
    """A physical result without a declared owner must not borrow one."""
    source = SourceState(SOURCE_PRESENT, COMPLETION_SETTLED)
    assert _classify_result(owner_present=False, source=source) == CLASS_UNKNOWN


def test_superseded_source_is_truncated_before_position_heuristics() -> None:
    """An acquisition known to trail its source is not guessed as interrupted."""
    source = SourceState(SOURCE_PRESENT, COMPLETION_SUPERSEDED)
    assert _classify_call(rule=None, position=POSITION_TAIL, source=source) == CLASS_SOURCE_TRUNCATED
    assert _classify_result(owner_present=False, source=source) == CLASS_SOURCE_TRUNCATED
    assert _classify_result(owner_present=False, source=SourceState(SOURCE_BYTES_ABSENT, COMPLETION_SETTLED)) == (
        CLASS_SOURCE_TRUNCATED
    )
