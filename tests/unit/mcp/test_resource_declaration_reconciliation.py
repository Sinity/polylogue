"""TARGET_RESOURCES declares exactly what the server registers.

polylogue-w17k1: the declaration was wrong in both directions at once -- seven
of nine declared resources did not exist (a caller is told it can fetch
something that is not there) and thirteen live registrations were undeclared
(real capability invisible to an agent that reads the declaration). Neither
direction had a test.
"""

from __future__ import annotations

from typing import cast

from polylogue.agent_integration.manifest import build_live_manifest, live_resource_uri_templates
from polylogue.mcp.declarations.registry import TARGET_RESOURCES, UNBUILT_RESOURCE_OBJECT_KINDS
from tests.infra.mcp import EXPECTED_RESOURCE_TEMPLATE_URIS, EXPECTED_RESOURCE_URIS


def test_declared_resources_equal_live_registrations() -> None:
    """Anti-vacuity: add an ``@mcp.resource`` in ``server_resources.py`` without
    a TARGET_RESOURCES entry (or the reverse) and this names the gap."""
    declared = {entry.uri_template for entry in TARGET_RESOURCES}
    live = set(live_resource_uri_templates())
    assert declared - live == set(), "declared but never registered"
    assert live - declared == set(), "registered but never declared"
    assert declared == EXPECTED_RESOURCE_URIS | EXPECTED_RESOURCE_TEMPLATE_URIS


def test_manifest_reports_resource_reconciliation_in_both_directions() -> None:
    manifest = build_live_manifest()
    assert manifest["missing_target_resources"] == []
    assert manifest["undeclared_live_resources"] == []
    counts = cast("dict[str, int]", manifest["counts"])
    assert counts["resources"] == len(TARGET_RESOURCES)


def test_unbuilt_resource_intent_is_recorded_not_deleted() -> None:
    """Narrowing the declaration must not read as "these were never wanted"."""
    declared = {entry.uri_template for entry in TARGET_RESOURCES}
    for kind in UNBUILT_RESOURCE_OBJECT_KINDS:
        assert f"polylogue://{kind}/{{id}}" not in declared
    assert build_live_manifest()["unbuilt_resource_object_kinds"] == list(UNBUILT_RESOURCE_OBJECT_KINDS)
