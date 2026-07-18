"""Parity between shipped MCP prompt recipes and executable query_units grammar."""

from __future__ import annotations

import re
from collections.abc import Mapping

import pytest

from polylogue.archive.query.expression import parse_unit_source_expression
from polylogue.mcp.server import build_server
from tests.infra.mcp import invoke_surface_async

_QUERY_CALL_RE = re.compile(
    r"query_units\(expression=(?P<quote>['\"])(?P<expression>.*?)(?P=quote)(?P<arguments>[^)]*)\)",
    re.DOTALL,
)
_ARGUMENT_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\s*=")
_PROMPT_CASES: Mapping[str, dict[str, object]] = {
    "decisions_about": {"topic": "cursor-bound-pages"},
    "unacknowledged_failures": {"repo": "polylogue", "since": "7d"},
    "sessions_touching_file": {"path": "polylogue/mcp/server_prompts.py", "repo": "polylogue"},
}


@pytest.mark.asyncio
async def test_shipped_query_units_prompt_recipes_match_parser_and_discovered_schema() -> None:
    server = build_server(role="read")
    query_tool = server._tool_manager._tools["query_units"]
    properties = query_tool.parameters["properties"]
    assert isinstance(properties, dict)
    observed_prompts: set[str] = set()

    for prompt_name, kwargs in _PROMPT_CASES.items():
        prompt = server._prompt_manager._prompts[prompt_name]
        rendered = await invoke_surface_async(prompt.fn, **kwargs)
        assert isinstance(rendered, str)
        matches = list(_QUERY_CALL_RE.finditer(rendered))
        assert matches, f"{prompt_name} must expose at least one query_units recipe"
        observed_prompts.add(prompt_name)
        for match in matches:
            expression = match.group("expression")
            source = parse_unit_source_expression(expression)
            assert source is not None, f"{prompt_name} advertises a non-executable expression: {expression}"
            advertised_arguments = {"expression", *_ARGUMENT_RE.findall(match.group("arguments"))}
            assert advertised_arguments <= set(properties), (
                f"{prompt_name} advertises hidden query_units arguments: "
                f"{sorted(advertised_arguments - set(properties))}"
            )

    assert observed_prompts == set(_PROMPT_CASES)
