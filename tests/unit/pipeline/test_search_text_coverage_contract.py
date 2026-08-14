"""Executable contract for the documented search-text coverage boundary."""

from __future__ import annotations

from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.archive.query.expression import compile_expression, parse_unit_source_expression
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_WRITE_TOKEN = "write-body-needle"
_EDIT_OLD_TOKEN = "edit-old-body-needle"
_EDIT_TOKEN = "edit-body-needle"


def test_action_body_lookup_paths_execute(tmp_path: Path) -> None:
    """Production action predicates find write/edit bodies excluded from FTS."""
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="search-text-coverage-contract",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Write",
                        tool_id="write-body",
                        tool_input={"file_path": "/tmp/plan.md", "content": _WRITE_TOKEN},
                    ),
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Edit",
                        tool_id="edit-body",
                        tool_input={
                            "file_path": "/tmp/module.py",
                            "old_string": _EDIT_OLD_TOKEN,
                            "new_string": _EDIT_TOKEN,
                        },
                    ),
                ],
            )
        ],
    )

    with ArchiveStore(tmp_path / "archive") as archive:
        session_id = archive.write_parsed(session)

        terminal_source = parse_unit_source_expression(f'actions where tool:write AND text:"{_WRITE_TOKEN}"')
        assert terminal_source is not None
        assert terminal_source.unit == "action"
        terminal_rows = archive.query_actions(terminal_source.predicate, limit=10)
        assert [row.tool_name for row in terminal_rows] == ["Write"]

        for tool, token in (
            ("write", _WRITE_TOKEN),
            ("edit", _EDIT_OLD_TOKEN),
            ("edit", _EDIT_TOKEN),
        ):
            spec = compile_expression(f'exists action(tool:{tool} AND text:"{token}")')
            assert spec.boolean_predicate is not None
            rows = archive.list_summaries(limit=10, boolean_predicate=spec.boolean_predicate)
            assert [row.session_id for row in rows] == [session_id]
