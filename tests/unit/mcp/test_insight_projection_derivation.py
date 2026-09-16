"""The MCP projection surface is derived from the insight registry.

polylogue-h8l96: ``_INSIGHT_PROJECTIONS`` was a hand-written frozenset of five
tokens, four of which are not registry descriptors at all and one of which
(``tool-episodes``) was the only one of the registry's thirteen descriptors MCP
served. Nothing compared the two, so twelve registered insights had no MCP
surface and a newly registered one would have gained none.
"""

from __future__ import annotations

from polylogue.analysis.registry import INSIGHT_REGISTRY
from polylogue.mcp.server_cutover import (
    _STANDALONE_INSIGHT_REPORTS,
    insight_projections,
    registry_insight_projections,
)


def test_every_registered_insight_has_an_mcp_projection() -> None:
    """Anti-vacuity: re-freeze the projection set to a literal that omits any
    descriptor (or register a new descriptor without touching MCP) and this
    fails naming the missing token."""
    expected = {descriptor.resolved_cli_command_name for descriptor in INSIGHT_REGISTRY.values()}
    assert registry_insight_projections() == expected
    assert expected <= insight_projections()
    assert len(expected) == len(INSIGHT_REGISTRY)


def test_standalone_reports_are_declared_separately_from_the_registry() -> None:
    """The four dispatcher-compiled reports are not registry descriptors, and
    saying so is what keeps the derived half honestly derived."""
    registry_tokens = registry_insight_projections()
    assert _STANDALONE_INSIGHT_REPORTS.isdisjoint(registry_tokens)
    assert insight_projections() == _STANDALONE_INSIGHT_REPORTS | registry_tokens


def test_every_registry_projection_declares_a_query_model() -> None:
    """A descriptor with no query model cannot be fetched generically; the
    dispatcher refuses it in typed form rather than 500ing, but a descriptor
    that reaches production without one is a registration defect."""
    missing = sorted(name for name, descriptor in INSIGHT_REGISTRY.items() if descriptor.query_model is None)
    assert missing == []
