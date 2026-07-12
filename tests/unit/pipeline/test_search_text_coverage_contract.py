"""Executable contract for the documented search-text coverage boundary."""

from __future__ import annotations

import re
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.archive.query.expression import compile_expression
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_REPO_ROOT = Path(__file__).parents[3]
_SEARCH_DOC = _REPO_ROOT / "docs" / "search.md"
_INDEX_DDL = _REPO_ROOT / "polylogue" / "storage" / "sqlite" / "archive_tiers" / "index.py"

_WRITE_TOKEN = "quokka-manifesto-9f3c1a"
_EDIT_TOKEN = "renamed-walrus-descriptor-4e91"


def _coverage_section() -> str:
    document = _SEARCH_DOC.read_text(encoding="utf-8")
    match = re.search(
        r"^## Searchable Content Coverage\n(?P<section>.*?)(?=^## )",
        document,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "docs/search.md must define Searchable Content Coverage"
    return match.group("section")


def _coverage_decision(section: str, source: str) -> str:
    for row in section.splitlines():
        if f"`{source}`" not in row:
            continue
        match = re.search(r"\| (?:\*\*)?(Yes|No)(?:\*\*)? \|", row)
        assert match is not None, f"coverage matrix must declare whether {source} feeds search_text"
        return match.group(1)
    raise AssertionError(f"coverage matrix must declare whether {source} feeds search_text")


def _documented_sql_probe(section: str) -> str:
    match = re.search(
        r"equivalent JSON-aware SQL probe is:\n\n```sql\n(?P<sql>.*?)\n```",
        section,
        flags=re.DOTALL,
    )
    assert match is not None, "coverage docs must contain the executable JSON-aware SQL probe"
    return match.group("sql")


def test_documented_coverage_matrix_matches_live_search_text_ddl() -> None:
    """The matrix is derived from the DDL, not separately asserted prose fragments."""
    section = _coverage_section()
    ddl_source = _INDEX_DDL.read_text(encoding="utf-8")
    match = re.search(
        r"search_text\s+TEXT GENERATED ALWAYS AS \((?P<expr>.*?)\)\s*VIRTUAL,",
        ddl_source,
        flags=re.DOTALL,
    )
    assert match is not None, "blocks.search_text generated-column definition not found"
    expression = match.group("expr")

    for source, ddl_fragment in {
        "blocks.text": "COALESCE(text, '')",
        "blocks.tool_name": "COALESCE(tool_name, '')",
        "tool_input.$.command": "json_extract(tool_input, '$.command')",
        "tool_input.$.file_path": "json_extract(tool_input, '$.file_path')",
        "tool_input.$.path": "json_extract(tool_input, '$.path')",
    }.items():
        assert _coverage_decision(section, source) == "Yes"
        assert ddl_fragment in expression

    for source, ddl_fragment in {
        "tool_input.$.content": "json_extract(tool_input, '$.content')",
        "tool_input.$.old_string": "json_extract(tool_input, '$.old_string')",
        "tool_input.$.new_string": "json_extract(tool_input, '$.new_string')",
    }.items():
        assert _coverage_decision(section, source) == "No"
        assert ddl_fragment not in expression


def test_documented_action_and_sql_body_lookup_paths_execute(tmp_path: Path) -> None:
    """The documented DSL and fenced SQL both find bodies excluded from FTS."""
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
                        tool_input={"file_path": "/tmp/module.py", "new_string": _EDIT_TOKEN},
                    ),
                ],
            )
        ],
    )

    with ArchiveStore(tmp_path / "archive") as archive:
        session_id = archive.write_parsed(session)

        for tool, token in (("write", _WRITE_TOKEN), ("edit", _EDIT_TOKEN)):
            spec = compile_expression(f'exists action(tool:{tool} AND text:"{token}")')
            assert spec.boolean_predicate is not None
            rows = archive.list_summaries(limit=10, boolean_predicate=spec.boolean_predicate)
            assert [row.session_id for row in rows] == [session_id]

        sql_probe = _documented_sql_probe(_coverage_section())
        for token, expected_tool in ((_WRITE_TOKEN, "Write"), (_EDIT_TOKEN, "Edit")):
            sql_rows = archive._conn.execute(sql_probe.replace("needle", token)).fetchall()
            assert [row[2] for row in sql_rows] == [expected_tool]
