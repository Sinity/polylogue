from pathlib import Path

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
