"""Role-scoped target/runtime manifest for agent integration."""

from __future__ import annotations

from typing import Any, cast

from polylogue.agent_integration.assets import agent_asset_metadata
from polylogue.agent_integration.spec import (
    CAPABILITY_FAMILIES,
    RECIPES,
    ROLE_ORDER,
    ROLES,
    TARGET_SCHEMA_STATUS,
    TOOL_CONTRACTS,
)
from polylogue.mcp.declarations import TARGET_PROMPTS, TARGET_RESOURCES, MCPRole, declared_tool_names
from polylogue.version import POLYLOGUE_VERSION

_MANUAL_RESOURCES = (
    "polylogue://agent/manual",
    "polylogue://agent/reference",
    "polylogue://agent/manifest/{role}",
)


def target_tool_names(role: MCPRole = "read") -> tuple[str, ...]:
    """Return target transaction names visible at ``role`` in stable order."""

    if role not in ROLES:
        raise ValueError(f"unknown MCP role: {role}")
    return tuple(contract.name for contract in TOOL_CONTRACTS if ROLE_ORDER[role] >= ROLE_ORDER[contract.minimum_role])


def declared_runtime_tool_names(role: MCPRole = "read") -> tuple[str, ...]:
    """Return the compatibility/runtime declaration names visible at ``role``."""

    if role not in ROLES:
        raise ValueError(f"unknown MCP role: {role}")
    return tuple(sorted(declared_tool_names(role)))


def target_tool_names_are_registered(role: MCPRole = "read") -> bool:
    """Return whether the runtime declaration names equal the target role surface."""

    return set(declared_runtime_tool_names(role)) == set(target_tool_names(role))


def target_contract_schemas_are_live_verified() -> bool:
    """Return whether generated contracts have been rebound to final live schemas."""

    return TARGET_SCHEMA_STATUS == "live-verified" and all(
        contract.schema_status == "live-verified" for contract in TOOL_CONTRACTS
    )


def target_surface_is_registered(role: MCPRole = "read") -> bool:
    """Return whether target names and generated schemas have completed cutover.

    Both gates are required so a names-only cutover cannot activate stale
    parameterized calls. Native client installers may still stage the generated
    package for an apply-after-cutover deployment.
    """

    return target_tool_names_are_registered(role) and target_contract_schemas_are_live_verified()


def build_live_manifest(role: MCPRole = "read") -> dict[str, object]:
    """Build an honest role-scoped manifest from executable declarations.

    The declaration registrar validates the actual FastMCP set against these
    names, so this remains dependency-light while preserving runtime authority.
    During the cutover it reports both the current compatibility surface and
    the intended small target surface instead of claiming they are equivalent.
    """

    if role not in ROLES:
        raise ValueError(f"unknown MCP role: {role}")
    runtime_tools = declared_runtime_tool_names(role)
    target_tools = target_tool_names(role)
    runtime_set = set(runtime_tools)
    target_set = set(target_tools)
    names_registered = runtime_set == target_set
    schemas_verified = target_contract_schemas_are_live_verified()
    cutover_ready = names_registered and schemas_verified
    allowed_recipes = [recipe.id for recipe in RECIPES if ROLE_ORDER[role] >= ROLE_ORDER["read"]]
    metadata = agent_asset_metadata()
    return {
        "schema_version": 2,
        "package_version": POLYLOGUE_VERSION,
        "role": role,
        "asset": metadata,
        "schema_status": "live-verified" if cutover_ready else "cutover-parameterized",
        "cutover_ready": cutover_ready,
        "tool_names_registered": names_registered,
        "contract_schemas_verified": schemas_verified,
        "tools": list(runtime_tools),
        "target_tools": list(target_tools),
        "missing_target_tools": sorted(target_set - runtime_set),
        "compatibility_tools_remaining": sorted(runtime_set - target_set),
        "resources": [item.uri_template for item in TARGET_RESOURCES if "{" not in item.uri_template],
        "resource_templates": [item.uri_template for item in TARGET_RESOURCES if "{" in item.uri_template],
        "manual_resources": list(_MANUAL_RESOURCES),
        "prompts": [item.name for item in TARGET_PROMPTS],
        "counts": {
            "runtime_tools": len(runtime_tools),
            "target_tools": len(target_tools),
            "resources": len(TARGET_RESOURCES),
            "prompts": len(TARGET_PROMPTS),
        },
        "capability_families": [family.id for family in CAPABILITY_FAMILIES],
        "available_recipes": allowed_recipes,
    }


def manifest_name_sets(payload: dict[str, object]) -> dict[str, set[str]]:
    """Return typed name sets for executable validation."""

    return {
        key: {str(item) for item in cast(list[Any], payload.get(key, []))}
        for key in ("tools", "target_tools", "resources", "resource_templates", "prompts")
    }


__all__ = [
    "build_live_manifest",
    "declared_runtime_tool_names",
    "manifest_name_sets",
    "target_contract_schemas_are_live_verified",
    "target_surface_is_registered",
    "target_tool_names_are_registered",
    "target_tool_names",
]
