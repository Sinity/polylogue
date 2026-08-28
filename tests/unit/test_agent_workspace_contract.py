from __future__ import annotations

from pathlib import Path

import tomllib

ROOT = Path(__file__).parents[2]


def test_editing_agent_contracts_require_an_agentctl_workspace() -> None:
    worker = tomllib.loads((ROOT / ".codex" / "agents" / "worker.toml").read_text(encoding="utf-8"))
    worker_instructions = worker["developer_instructions"]
    lane = (ROOT / ".claude" / "agents" / "lane.md").read_text(encoding="utf-8")
    conventions = (ROOT / ".agent" / "CONVENTIONS.md").read_text(encoding="utf-8")

    for contract in (worker_instructions, lane, conventions):
        assert "AgentCTL-managed workspace" in contract
        assert "workspace ID" in contract
        assert "editing dispatch" in contract
