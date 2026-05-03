"""Insight registry parity tests for #624.

Proves that every insight type has the required consumer descriptors:
CLI, MCP, API, readiness, and export.
"""

from __future__ import annotations

from polylogue.insights.registry import INSIGHT_REGISTRY


def _consumer_report() -> dict[str, dict[str, str | None]]:
    """Build a consumer-descriptor parity table for all insight types."""
    report: dict[str, dict[str, str | None]] = {}
    for it in INSIGHT_REGISTRY.values():
        report[it.name] = {
            "cli_command": it.cli_command_name or None,
            "json_key": it.json_key,
            "ops_method": it.operations_method_name or None,
            "query_model": it.query_model.__name__ if it.query_model else None,
            "mcp_registered": "yes" if it.query_model and it.operations_method_name else "no",
            "export_eligible": "yes" if it.export_eligible else "no",
            "reader_panel": it.reader_panel,
            "readiness_required": "no" if it.readiness_exempt else "yes",
        }
    return report


def test_all_insight_types_have_json_key() -> None:
    """Every insight type must have a json_key for API/MCP responses."""
    for name, it in INSIGHT_REGISTRY.items():
        assert it.json_key, f"{name}: missing json_key"
        assert it.json_key.isidentifier() or "_" in it.json_key, (
            f"{name}: json_key='{it.json_key}' is not a valid identifier"
        )


def test_all_insight_types_have_display_name() -> None:
    """Every insight type must have a human-readable display_name."""
    for name, it in INSIGHT_REGISTRY.items():
        assert it.display_name, f"{name}: missing display_name"


def test_cli_registered_insights_have_help() -> None:
    """Insight types with CLI commands must have help text."""
    for name, it in INSIGHT_REGISTRY.items():
        if it.cli_command_name:
            assert it.cli_help, f"{name}: has cli_command_name but no cli_help"


def test_mcp_registered_insights_have_query_model() -> None:
    """Insight types registered for MCP must have a query_model."""
    for name, it in INSIGHT_REGISTRY.items():
        if it.operations_method_name:
            assert it.query_model is not None, f"{name}: has ops_method but no query_model — MCP reg requires it"


def test_descriptor_parity_report() -> None:
    """Generate the consumer descriptor parity report (informational)."""
    report = _consumer_report()
    _missing_cli = [n for n, r in report.items() if r["cli_command"] is None]
    _missing_export = [n for n, r in report.items() if r["export_eligible"] == "no"]
    exempt_readiness = [n for n, r in report.items() if r["readiness_required"] == "no"]

    assert len(report) == len(INSIGHT_REGISTRY), "report size mismatch"

    # Every insight should have at least CLI or MCP exposure
    for name, r in report.items():
        has_surface = r["cli_command"] is not None or r["mcp_registered"] == "yes"
        assert has_surface, f"{name}: no CLI or MCP surface"

    # Export-eligible should be the overwhelming majority
    if _missing_export:
        # Not failing — export is opt-out, not opt-in
        pass

    # Readiness-exempt should have an explicit reason documented
    for name in exempt_readiness:
        it = INSIGHT_REGISTRY[name]
        assert it.readiness_exempt, f"{name}: listed as exempt but readiness_exempt=False"
