"""stale_context reads the stored replaced range (polylogue-4ts.5).

A compaction boundary records the message range it replaced in
``session_events.boundary_start_position``/``boundary_end_position``. The
detector reports those messages, so a reader lands on the content the run
stopped being able to see rather than on the session as a whole.

Anti-vacuity: ``test_compaction_finding_disappears_without_the_stored_range``
nulls the two columns on the same archive. A detector that inferred the range
from message shape, or that flagged the session merely because a compaction
event exists, stays green there and is red only here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.analysis.pathology import PathologyFinding
from polylogue.api import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

_SESSION_ID = "codex-session:compaction-pathology"
# Positions 0..2 are replaced; position 3 is the summary that stands in for them.
_REPLACED_NATIVE_IDS = ("m0", "m1", "m2")


def _message(native_id: str, role: Role, text: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=native_id,
        role=role,
        text=text,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _seed(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="compaction-pathology",
        title="compaction pathology",
        messages=[
            _message("m0", Role.USER, "first ask"),
            _message("m1", Role.ASSISTANT, "first answer"),
            _message("m2", Role.USER, "second ask"),
            _message("m3", Role.SYSTEM, "compaction summary"),
            _message("m4", Role.USER, "post-compaction ask"),
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="compaction",
                source_message_provider_id="m3",
                boundary_start_position=0,
                boundary_end_position=2,
                boundary_message_position=3,
                payload={"type": "compaction"},
            )
        ],
    )
    write_parsed_session_to_archive(conn, session)
    conn.commit()
    conn.close()


def _clear_boundary_range(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE session_events SET boundary_start_position = NULL, boundary_end_position = NULL "
        "WHERE event_type = 'compaction'"
    )
    conn.commit()
    conn.close()


async def _stale_context_findings(archive_root: Path, db_path: Path) -> list[PathologyFinding]:
    polylogue = Polylogue(archive_root=archive_root, db_path=db_path)
    try:
        report = await polylogue.pathology_report()
    finally:
        await polylogue.close()
    return [finding for finding in report.findings if finding.kind == "stale_context"]


@pytest.mark.asyncio
async def test_compaction_finding_names_exactly_the_replaced_messages(
    workspace_env: dict[str, Path],
) -> None:
    db_path = workspace_env["archive_root"] / "index.db"
    _seed(db_path)

    findings = await _stale_context_findings(workspace_env["archive_root"], db_path)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.session_id == _SESSION_ID
    assert "0-2" in finding.detail
    assert "3 message(s)" in finding.detail
    assert [ref.message_id for ref in finding.evidence_refs] == [
        f"{_SESSION_ID}:n:{native_id}" for native_id in _REPLACED_NATIVE_IDS
    ]


@pytest.mark.asyncio
async def test_compaction_finding_disappears_without_the_stored_range(
    workspace_env: dict[str, Path],
) -> None:
    db_path = workspace_env["archive_root"] / "index.db"
    _seed(db_path)
    _clear_boundary_range(db_path)

    assert await _stale_context_findings(workspace_env["archive_root"], db_path) == []
