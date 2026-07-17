"""Packaged six-tool asset, native hook, and command-surface tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from unittest.mock import patch

from click.testing import CliRunner

from devtools.render_agent_manual import expected_outputs
from devtools.render_agent_manual import main as render_main
from polylogue.agent_integration.assets import ALL_ASSETS, agent_asset_metadata, read_agent_asset, read_agent_json
from polylogue.agent_integration.installer import claude_session_start_payload
from polylogue.agent_integration.manifest import target_surface_is_registered, target_tool_names
from polylogue.agent_integration.spec import (
    CAPABILITY_FAMILIES,
    CLIENTS,
    DEFAULT_READ_TOOLS,
    ORIGIN_MEANINGS,
    PRIVILEGED_TOOLS,
    RECIPES,
    TOOL_CONTRACTS,
)
from polylogue.cli.commands.agent import agent_command


def test_all_packaged_assets_are_nonempty_and_measured() -> None:
    """Mutation: removing an asset or typed family changes the measured package contract."""
    metadata = agent_asset_metadata()
    asset_bytes = cast(dict[str, int], metadata["asset_bytes"])

    assert set(asset_bytes) == set(ALL_ASSETS)
    assert metadata["capability_family_count"] == len(CAPABILITY_FAMILIES) == 6
    assert metadata["recipe_count"] == len(RECIPES) == 4
    assert cast(int, metadata["standing_manual_bytes"]) > 20_000
    assert cast(int, metadata["deep_reference_bytes"]) > 20_000
    assert len(cast(str, metadata["asset_digest"])) == 64


def test_json_assets_match_typed_authority() -> None:
    """Mutation: deleting a typed transaction/recipe/client makes its generated JSON projection diverge."""
    recipes = read_agent_json("recipes.json")
    spec = read_agent_json("integration-spec.json")
    contracts = read_agent_json("tool-contracts.json")
    recipe_rows = cast(list[dict[str, object]], recipes["recipes"])
    family_rows = cast(list[dict[str, object]], spec["capability_families"])
    transaction_rows = cast(list[dict[str, object]], contracts["transactions"])

    assert [row["id"] for row in recipe_rows] == [recipe.id for recipe in RECIPES]
    assert [row["id"] for row in family_rows] == [family.id for family in CAPABILITY_FAMILIES]
    assert spec["clients"] == list(CLIENTS)
    assert spec["origins"] == [{"token": item.token, "meaning": item.meaning} for item in ORIGIN_MEANINGS]
    assert contracts["default_read_tools"] == list(DEFAULT_READ_TOOLS)
    assert contracts["privileged_tools"] == list(PRIVILEGED_TOOLS)
    assert [row["name"] for row in transaction_rows] == [contract.name for contract in TOOL_CONTRACTS]


def test_manual_contains_six_tool_continuation_role_and_origin_contract() -> None:
    """Mutation: restoring 103-tool prose or omitting continuation/citation teaching fails this test."""
    manual = read_agent_asset("standing-manual.md")

    assert "## Cold-start decision route" in manual
    assert "## The six tools" in manual
    assert "same tool with **only** the returned opaque token" in manual
    assert "Never cite a continuation token" in manual
    assert "strict command floor" in manual
    assert "preview-bound confirmation" in manual
    assert "11 tokens" in manual
    for name in DEFAULT_READ_TOOLS:
        assert f"### `{name}`" in manual
    for origin in ORIGIN_MEANINGS:
        assert f"`{origin.token}`" in manual
    for recipe in RECIPES:
        assert f"(`{recipe.id}`)" in manual


def test_claude_session_start_returns_complete_manual() -> None:
    """Production dependency: Claude SessionStart payload injects the packaged manual without a fetch turn."""
    payload = claude_session_start_payload()

    output = cast(dict[str, object], payload["hookSpecificOutput"])
    assert output["hookEventName"] == "SessionStart"
    assert output["additionalContext"] == read_agent_asset("standing-manual.md")


def test_manual_and_hidden_session_start_commands() -> None:
    runner = CliRunner()

    manual = runner.invoke(agent_command, ["manual", "--kind", "standing"])
    hook = runner.invoke(agent_command, ["session-start", "--client", "claude-code"])

    assert manual.exit_code == 0
    assert manual.output == read_agent_asset("standing-manual.md")
    assert hook.exit_code == 0
    assert json.loads(hook.output) == claude_session_start_payload()


def test_install_command_fails_closed_before_target_surface_cutover() -> None:
    """Mutation: removing the cutover guard would install six-tool guidance against the 104-tool runtime."""
    result = CliRunner().invoke(agent_command, ["install", "--client", "codex", "--format", "json"])

    assert result.exit_code == 1
    assert "six-tool agent guidance is staged" in result.output


def test_names_only_cutover_cannot_activate_parameterized_guidance() -> None:
    """Mutation: dropping the live-schema marker gate would activate calls before signature reconciliation."""
    with patch(
        "polylogue.agent_integration.manifest.declared_runtime_tool_names",
        return_value=target_tool_names("read"),
    ):
        assert target_surface_is_registered("read") is False


def test_generated_document_mirrors_match_packaged_assets_and_check_mode() -> None:
    """Mutation: editing a generated manual without the renderer makes --check fail."""
    root = Path(__file__).resolve().parents[3]

    assert (root / "docs" / "agent-manual.md").read_text() == read_agent_asset("standing-manual.md")
    assert (root / "docs" / "agent-integration-reference.md").read_text() == read_agent_asset("deep-reference.md")
    assert all(path.exists() and path.read_text() == expected for path, expected in expected_outputs().items())
    assert render_main(["--check"]) == 0
