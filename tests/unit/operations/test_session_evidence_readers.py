"""The moved evidence views answer with the facade's own rows (polylogue-r3cuz).

``file-edits``, ``agent-policies`` and ``web-content`` moved from opening the
archive in this process through ``Polylogue`` onto the declared ``session.read``
evidence kinds.  Both CLI legs (direct and daemon) now run the *same*
operation, so a daemon/direct parity test cannot see a field the move dropped
or renamed -- both legs would drop it together.

These tests compare the operation's evidence body against the Python API
reader the view used before the move, field for field and in order.  That is
the comparison the move has to survive: the row vocabulary is the contract,
not merely "some rows came back".

Anti-vacuity: rename, drop or reorder any key in
``operations/session_evidence.py`` -- for instance projecting
``structured_patch_json`` instead of ``structured_patch``, or ordering file
edits by ``observed_at_ms`` instead of ``(message_id, tool_use_block_id)`` --
and the corresponding comparison fails.  The non-empty assertions keep two
empty relations from agreeing trivially.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.enums import BlockType, Provider, Role
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedFileEdit, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_NATIVE_ID = "evidence-readers"
_SESSION_ID = f"claude-code-session:{_NATIVE_ID}"


def _seed(archive_root: Path) -> None:
    with ArchiveStore(archive_root) as archive_db:
        archive_db.write_raw_and_parsed(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id=_NATIVE_ID,
                title="Evidence readers",
                messages=[
                    ParsedMessage(
                        provider_message_id=f"m{index}",
                        role=Role.ASSISTANT,
                        position=index * 2,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Edit",
                                tool_id=f"edit-{index}",
                                tool_input={"file_path": f"/tmp/file-{index}.py"},
                            )
                        ],
                    )
                    for index in range(2)
                ]
                + [
                    ParsedMessage(
                        provider_message_id=f"r{index}",
                        role=Role.USER,
                        position=index * 2 + 1,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                outcome_unknown_reason="not_reported",
                                tool_id=f"edit-{index}",
                                text="applied",
                                file_edit=ParsedFileEdit(
                                    file_path=f"/tmp/file-{index}.py",
                                    structured_patch=[
                                        {
                                            "oldStart": 1,
                                            "oldLines": 1,
                                            "newStart": 1,
                                            "newLines": 2,
                                            "lines": [f"+line {index}"],
                                        }
                                    ],
                                    original_file=f"before {index}\n",
                                    old_string=f"old {index}",
                                    new_string=f"new {index}",
                                    replace_all=bool(index),
                                    user_modified=not index,
                                ),
                            )
                        ],
                    )
                    for index in range(2)
                ],
            ),
            payload=b'{"raw": "claude payload"}',
            source_path="/tmp/evidence-readers.jsonl",
            acquired_at_ms=1735689600000,
        )


@pytest.fixture
def seeded_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    _seed(archive_root)
    return archive_root


async def test_file_edit_rows_match_the_facade_reader_they_replaced(seeded_root: Path) -> None:
    from polylogue.api import Polylogue
    from polylogue.operations.session_evidence import read_file_edits_evidence

    archive = Polylogue(archive_root=seeded_root)
    try:
        facade_rows = await archive.get_file_edits(_SESSION_ID)
    finally:
        await archive.close()

    with ArchiveStore(seeded_root) as store:
        evidence = read_file_edits_evidence(store, _SESSION_ID)

    assert facade_rows, "the fixture must actually record file edits, or this comparison is vacuous"
    assert evidence["file_edits"] == facade_rows
    assert evidence["total"] == len(facade_rows)
    assert evidence["session_id"] == _SESSION_ID


async def test_agent_policy_rows_match_the_facade_reader_they_replaced(seeded_root: Path) -> None:
    """An empty relation still has to agree in shape, not merely in emptiness."""

    from polylogue.api import Polylogue
    from polylogue.operations.session_evidence import read_agent_policies_evidence

    archive = Polylogue(archive_root=seeded_root)
    try:
        facade_rows = await archive.get_agent_policies(_SESSION_ID)
    finally:
        await archive.close()

    with ArchiveStore(seeded_root) as store:
        evidence = read_agent_policies_evidence(store, _SESSION_ID)

    assert facade_rows is not None
    assert evidence["agent_policies"] == facade_rows
    assert evidence["total"] == len(facade_rows)


async def test_web_content_rows_match_the_facade_reader_they_replaced(seeded_root: Path) -> None:
    from polylogue.api import Polylogue
    from polylogue.operations.session_evidence import read_web_content_constructs_evidence

    archive = Polylogue(archive_root=seeded_root)
    try:
        facade_rows = await archive.get_web_content_constructs(_SESSION_ID)
    finally:
        await archive.close()

    with ArchiveStore(seeded_root) as store:
        evidence = read_web_content_constructs_evidence(store, _SESSION_ID)

    assert facade_rows is not None
    assert evidence["web_content_constructs"] == facade_rows
    assert evidence["total"] == len(facade_rows)
