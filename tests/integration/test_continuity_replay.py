"""Terminal continuity replay through a real local MCP stdio client/server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.continuity_replay import main, run_live_mcp_replay

FIXTURE = json.loads((Path(__file__).parents[1] / "data" / "continuity" / "incident.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_live_mcp_replay_enumerates_known_answer_after_cold_resume() -> None:
    """The registered stdio server, not a supplied observation, earns pass."""
    result = await run_live_mcp_replay("mcp-query-transaction", FIXTURE)

    assert result["classification"] == "pass"
    assert result["observed_refs"] == result["expected_refs"]
    receipt = result["receipt"]
    assert isinstance(receipt, dict)
    assert receipt["cancelled_after_pages"] == 1
    assert receipt["resumed_with_fresh_server_process"] is True
    transcript = receipt["transcript"]
    assert isinstance(transcript, list)
    assert transcript[0]["query_units_discovered"] is True
    assert transcript[0]["capability_resource_read"] is True
    assert transcript[1]["phase"] == "cancelled-walk"
    assert any(row["phase"] == "cold-resume-page" for row in transcript)


@pytest.mark.asyncio
async def test_live_mcp_replay_rejects_changed_independent_known_answer() -> None:
    """Anti-vacuity: the route cannot bless its own output as the oracle."""
    mutated = {**FIXTURE, "mcp-query-transaction": {"refs": ["session:missing"]}}

    result = await run_live_mcp_replay("mcp-query-transaction", mutated)

    assert result["classification"] == "projection"


def test_live_mcp_command_writes_durable_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "continuity-receipt.json"

    exit_code = main(
        [
            "mcp-query-transaction",
            str(Path(__file__).parents[1] / "data" / "continuity" / "incident.json"),
            "--live-mcp",
            "--receipt",
            str(receipt),
        ]
    )

    written = json.loads(receipt.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert written["classification"] == "pass"
    assert written["receipt"]["transcript"][1]["phase"] == "cancelled-walk"
