"""Schema audit workflow orchestration."""

from __future__ import annotations

from pathlib import Path

from polylogue.core.outcomes import OutcomeCheck as CheckResult
from polylogue.core.outcomes import OutcomeStatus
from polylogue.schemas.audit.checks import (
    check_annotation_coverage,
    check_cross_provider_consistency,
    check_privacy_guards,
    check_schema_drift,
    check_schema_staleness,
    check_semantic_roles,
)
from polylogue.schemas.audit.models import AuditCheck, AuditReport
from polylogue.schemas.audit.walkers import _load_committed_schema
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry


def _scoped(provider: str, check: CheckResult) -> AuditCheck:
    return AuditCheck(
        name=check.name,
        status=check.status,
        summary=check.summary,
        count=check.count,
        details=list(check.details),
        breakdown=dict(check.breakdown),
        provider=provider,
    )


def audit_provider(provider: str, *, db_path: Path | None = None) -> AuditReport:
    """Run all audit checks on a single provider's committed schema."""
    report = AuditReport(provider=provider)

    schema = _load_committed_schema(provider)
    if schema is None:
        report.checks.append(
            AuditCheck(
                name="schema_exists",
                status=OutcomeStatus.ERROR,
                summary=f"No committed schema found for {provider}",
                provider=provider,
            )
        )
        return report

    report.checks.append(
        AuditCheck(
            name="schema_exists",
            status=OutcomeStatus.OK,
            summary="Committed schema loaded",
            provider=provider,
        )
    )
    report.checks.append(_scoped(provider, check_privacy_guards(schema)))
    report.checks.append(_scoped(provider, check_semantic_roles(schema)))
    report.checks.append(_scoped(provider, check_annotation_coverage(schema)))
    report.checks.append(_scoped(provider, check_schema_staleness(schema)))

    if db_path is not None:
        report.checks.append(
            _scoped(
                provider,
                check_schema_drift(schema, db_path=db_path, provider=provider),
            )
        )

    return report


def audit_all_providers(
    providers: list[str] | None = None,
    *,
    db_path: Path | None = None,
) -> AuditReport:
    """Run audit checks across all (or specified) providers."""
    from polylogue.schemas.observation import PROVIDERS

    provider_list = providers or list(PROVIDERS.keys())
    report = AuditReport()

    schemas = {}
    for provider in provider_list:
        provider_report = audit_provider(provider, db_path=db_path)
        report.checks.extend(provider_report.checks)
        schema = _load_committed_schema(provider)
        if schema:
            schemas[provider] = schema

    if len(schemas) >= 2:
        report.checks.append(check_cross_provider_consistency(schemas))

    return report


def audit_schema_bundle_privacy(*, registry: SchemaRegistry | None = None) -> AuditReport:
    """Run the registered privacy predicate over every committed schema element.

    The provider audit intentionally follows the public audit workflow and
    checks each provider's default schema. This gate covers the complete
    committed package bundle, including non-default versions, through the
    runtime registry that resolves those artifacts for production callers.
    """
    bundle_registry = registry or SchemaRegistry(storage_root=SCHEMA_DIR)
    report = AuditReport()
    for provider in bundle_registry.list_providers():
        for version in bundle_registry.list_versions(provider):
            package = bundle_registry.get_package(provider, version=version)
            if package is None:
                continue
            for element in package.elements:
                if element.schema_file is None:
                    continue
                scope = f"{provider}/{version}/{element.element_kind}"
                schema = bundle_registry.get_element_schema(
                    provider,
                    version=version,
                    element_kind=element.element_kind,
                )
                if schema is None:
                    report.checks.append(
                        AuditCheck(
                            name="privacy_guards",
                            status=OutcomeStatus.ERROR,
                            summary="Committed element schema is missing",
                            provider=scope,
                        )
                    )
                    continue
                report.checks.append(_scoped(scope, check_privacy_guards(schema)))
    return report


__all__ = ["audit_all_providers", "audit_provider", "audit_schema_bundle_privacy"]
