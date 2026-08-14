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

from typing import cast

from tests.infra.mcp import EXPECTED_PROMPT_NAMES, MCPServerUnderTest


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
