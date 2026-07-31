"""Discovery pinning for the MCP prompt surface (polylogue-il50).

Two independent gaps found by a shipped-but-dead audit: six of the seven
``TARGET_PROMPTS`` declarations instructed callers to invoke tool names
retired at the ten-tool cutover (e.g. ``find_resume_candidates``,
``get_session_summary``, ``search``), and five live-registered prompts
(``analyze_errors``, ``summarize_week``, ``extract_code``,
``compare_sessions``, ``extract_patterns``) were absent from
``TARGET_PROMPTS``, leaving every completeness/discovery consumer that reads
the declaration blind to them. ``EXPECTED_PROMPT_NAMES`` in
``tests/infra/mcp.py`` existed but was never referenced by any test, so
neither gap was caught.

This module closes both gaps with declaration-derived pins, mirroring how
``EXPECTED_TOOL_NAMES`` is derived rather than hand-copied:

1. The declared prompt set (``TARGET_PROMPTS`` -> ``EXPECTED_PROMPT_NAMES``)
   must equal the live-registered prompt set on the actual server.
2. Every prompt's rendered instruction text must reference only tool names
   that exist on the live ten-tool dispatcher surface -- catching a prompt
   that regresses back to naming a retired tool, the exact failure shape
   this bead found.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import cast

import pytest

from tests.infra.mcp import EXPECTED_PROMPT_NAMES, EXPECTED_TOOL_NAMES, MCPServerUnderTest, invoke_surface_async

#: Minimal arguments so every prompt renders without error. Prompts with no
#: required parameters use their own defaults (empty dict).
_PROMPT_INVOCATION_ARGS: Mapping[str, dict[str, object]] = {
    "decisions_about": {"topic": "schema migration"},
    "sessions_touching_file": {"path": "polylogue/mcp/server_prompts.py"},
    "compare_sessions": {"id1": "example-origin:session-a", "id2": "example-origin:session-b"},
}

#: A call-like ``name(`` pattern in a prompt's rendered instruction text.
#: Matches the same shape as the parity check in test_prompt_query_parity.py
#: but generalized to every dispatcher tool, not just ``query``.
_CALL_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\(")


def _build_server() -> MCPServerUnderTest:
    from polylogue.mcp.server import build_server

    return cast(MCPServerUnderTest, build_server())


def test_registered_prompts_match_target_prompts() -> None:
    """The live-registered prompt set and the declared set must be identical.

    Fails in either direction: a prompt registered in server_prompts.py
    without a TARGET_PROMPTS entry (the analyze_errors/summarize_week/
    extract_code/compare_sessions/extract_patterns gap), or a declared
    prompt that was never actually registered.
    """
    server = _build_server()
    registered = set(server._prompt_manager._prompts)
    assert registered == EXPECTED_PROMPT_NAMES, (
        f"registered-but-undeclared: {sorted(registered - EXPECTED_PROMPT_NAMES)}; "
        f"declared-but-unregistered: {sorted(EXPECTED_PROMPT_NAMES - registered)}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_name", sorted(EXPECTED_PROMPT_NAMES))
async def test_prompt_instructions_reference_only_live_tools(prompt_name: str) -> None:
    """Every ``name(`` call-like reference in a prompt's rendered text must
    name a tool on the live ten-tool dispatcher surface.

    Direct regression guard for the bead: six declared prompts instructed
    callers to invoke retired names (find_resume_candidates,
    get_resume_brief, agent_coordination_brief, blackboard_list,
    find_abandoned_sessions, get_session_summary, get_postmortem_bundle,
    get_pathologies, list_assertion_claims, search, find_stuck_sessions,
    list_marks, list_annotations, cost_rollups, session_costs,
    provider_usage). A prompt that regresses to any of those again fails
    this test.
    """
    server = _build_server()
    prompt = server._prompt_manager._prompts[prompt_name]
    kwargs = _PROMPT_INVOCATION_ARGS.get(prompt_name, {})
    rendered = await invoke_surface_async(prompt.fn, **kwargs)
    assert isinstance(rendered, str)

    referenced = set(_CALL_RE.findall(rendered))
    # Prompts may reference prose fragments that happen to match the call
    # shape (none currently do); only flag names that look like a tool call
    # AND are not a live tool name.
    unknown = referenced - EXPECTED_TOOL_NAMES
    assert not unknown, (
        f"{prompt_name} references non-tool or retired-tool call-like names {sorted(unknown)}; "
        f"live tools are {sorted(EXPECTED_TOOL_NAMES)}"
    )
