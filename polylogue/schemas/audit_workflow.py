"""Schema audit workflow orchestration."""

from __future__ import annotations

from polylogue.lib.outcomes import OutcomeCheck as CheckResult
from polylogue.lib.outcomes import OutcomeStatus
from polylogue.schemas.audit_checks import (
    check_annotation_coverage,
    check_cross_provider_consistency,
    check_privacy_guards,
    check_semantic_roles,
)
from polylogue.schemas.audit_models import AuditReport
from polylogue.schemas.audit_walkers import _load_committed_schema


def audit_provider(provider: str) -> AuditReport:
    """Run all audit checks on a single provider's committed schema."""
    report = AuditReport(provider=provider)

    schema = _load_committed_schema(provider)
    if schema is None:
        report.checks.append(
            CheckResult(
                name="schema_exists",
                status=OutcomeStatus.ERROR,
                summary=f"No committed schema found for {provider}",
            )
        )
        return report

    report.checks.append(
        CheckResult(
            name="schema_exists",
            status=OutcomeStatus.OK,
            summary="Committed schema loaded",
        )
    )
    report.checks.append(check_privacy_guards(schema))
    report.checks.append(check_semantic_roles(schema))
    report.checks.append(check_annotation_coverage(schema))

    return report


def audit_all_providers(providers: list[str] | None = None) -> AuditReport:
    """Run audit checks across all (or specified) providers."""
    from polylogue.schemas.observation import PROVIDERS

    provider_list = providers or list(PROVIDERS.keys())
    report = AuditReport()

    schemas = {}
    for provider in provider_list:
        provider_report = audit_provider(provider)
        report.checks.extend(provider_report.checks)
        schema = _load_committed_schema(provider)
        if schema:
            schemas[provider] = schema

    if len(schemas) >= 2:
        report.checks.append(check_cross_provider_consistency(schemas))

    return report


__all__ = ["audit_all_providers", "audit_provider"]
