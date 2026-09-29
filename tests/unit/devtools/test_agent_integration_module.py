from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_home_manager_mcp_capabilities_are_only_individual_opt_ins() -> None:
    """A role alone must not grant any privileged MCP capability."""
    module = (REPO_ROOT / "nix" / "agent-integration-module.nix").read_text(encoding="utf-8")

    assert "mcpRole" not in module, "anti-vacuity: no role-based compatibility option remains"
    assert 'cfg.mcpEnableWrite [ "--enable-write" ]' in module
    assert 'cfg.mcpEnableJudge [ "--enable-judge" ]' in module
    assert 'cfg.mcpEnableMaintenance [ "--enable-maintenance" ]' in module
    assert "roleWrite" not in module
    assert "roleJudge" not in module
    assert "roleMaintenance" not in module


@pytest.mark.parametrize("missing", [None, "Write", "Judge", "Maintenance"])
def test_home_manager_lane_checks_independent_capability_switches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str | None
) -> None:
    """The readiness lane passes the real module and fails when any capability switch is dropped.

    Before the fix it required the removed role-ladder expressions, so it
    rejected the intact module and could not tell a dropped switch apart.
    """
    from devtools import verify_agent_integration

    module = (REPO_ROOT / "nix/agent-integration-module.nix").read_text(encoding="utf-8")
    if missing is not None:
        module = module.replace(f"lib.optionals cfg.mcpEnable{missing}", "lib.optionals false")
    (tmp_path / "nix").mkdir()
    (tmp_path / "nix/agent-integration-module.nix").write_text(module, encoding="utf-8")
    (tmp_path / "flake.nix").write_text("homeManagerModules.agentIntegration", encoding="utf-8")
    monkeypatch.setattr(verify_agent_integration, "REPO_ROOT", tmp_path)
    result = verify_agent_integration._packaging_home_manager_lane()
    assert result.status == ("pass" if missing is None else "fail")
