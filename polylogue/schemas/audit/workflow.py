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
from polylogue.schemas.packages import SchemaVersionPackage
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
    committed package bundle, including non-default versions. Discovery and
    reads here use the committed tree directly so a missing or stale catalog
    cannot hide an artifact from the audit.
    """
    bundle_registry = registry or SchemaRegistry(storage_root=SCHEMA_DIR)
    report = AuditReport()
    for provider in bundle_registry.list_committed_providers():
        try:
            catalog = bundle_registry.load_committed_catalog(provider)
        except Exception as error:
            catalog = None
            report.checks.append(
                AuditCheck(
                    name="privacy_guards",
                    status=OutcomeStatus.ERROR,
                    summary=f"Committed schema catalog is unreadable: {type(error).__name__}",
                    provider=provider,
                )
            )
        if catalog is None:
            report.checks.append(
                AuditCheck(
                    name="privacy_guards",
                    status=OutcomeStatus.ERROR,
                    summary="Committed schema catalog is missing",
                    provider=provider,
                )
            )

        committed_versions = set(bundle_registry.list_committed_versions(provider))
        catalog_versions = {package.version for package in catalog.packages} if catalog is not None else set()
        versions = sorted(committed_versions | catalog_versions)
        if not versions:
            report.checks.append(
                AuditCheck(
                    name="privacy_guards",
                    status=OutcomeStatus.ERROR,
                    summary="No committed schema versions discovered",
                    provider=provider,
                )
            )
            continue

        audited_artifacts: set[tuple[str, str]] = set()
        for version in versions:
            try:
                package = bundle_registry.load_committed_package(provider, version)
            except Exception as error:
                package = None
                report.checks.append(
                    AuditCheck(
                        name="privacy_guards",
                        status=OutcomeStatus.ERROR,
                        summary=f"Committed schema package is unreadable: {type(error).__name__}",
                        provider=f"{provider}/{version}",
                    )
                )
            catalog_package = catalog.package(version) if catalog is not None else None
            scope = f"{provider}/{version}"
            if package is None:
                report.checks.append(
                    AuditCheck(
                        name="privacy_guards",
                        status=OutcomeStatus.ERROR,
                        summary="Committed schema package is missing",
                        provider=scope,
                    )
                )
            if catalog is not None and catalog_package is None:
                report.checks.append(
                    AuditCheck(
                        name="privacy_guards",
                        status=OutcomeStatus.ERROR,
                        summary="Cataloged schema package is missing",
                        provider=scope,
                    )
                )
            if package is not None and catalog_package is not None:
                package_schema_files = {element.element_kind: element.schema_file for element in package.elements}
                catalog_schema_files = {
                    element.element_kind: element.schema_file for element in catalog_package.elements
                }
                for element_kind in sorted(set(package_schema_files) | set(catalog_schema_files)):
                    package_schema_file = package_schema_files.get(element_kind)
                    catalog_schema_file = catalog_schema_files.get(element_kind)
                    if package_schema_file != catalog_schema_file:
                        report.checks.append(
                            AuditCheck(
                                name="privacy_guards",
                                status=OutcomeStatus.ERROR,
                                summary="Catalog/package schema_file disagreement",
                                details=[f"catalog={catalog_schema_file!r};package={package_schema_file!r}"],
                                provider=f"{scope}/{element_kind}",
                            )
                        )

            scope_check_start = len(report.checks)
            manifests: list[SchemaVersionPackage] = []
            if catalog_package is not None:
                manifests.append(catalog_package)
            if package is not None and package is not catalog_package:
                manifests.append(package)
            for manifest in manifests:
                if not manifest.elements:
                    report.checks.append(
                        AuditCheck(
                            name="privacy_guards",
                            status=OutcomeStatus.ERROR,
                            summary="Committed schema package has no auditable elements",
                            provider=scope,
                        )
                    )
                    continue
                for element in manifest.elements:
                    element_scope = f"{scope}/{element.element_kind}"
                    if element.schema_file is None:
                        if element.supported:
                            report.checks.append(
                                AuditCheck(
                                    name="privacy_guards",
                                    status=OutcomeStatus.ERROR,
                                    summary="Committed element schema file is missing",
                                    provider=element_scope,
                                )
                            )
                        continue
                    artifact_key = (version, element.schema_file)
                    if artifact_key in audited_artifacts:
                        continue
                    audited_artifacts.add(artifact_key)
                    try:
                        schema = bundle_registry.load_committed_schema_file(
                            provider,
                            version,
                            element.schema_file,
                        )
                    except Exception as error:
                        schema = None
                        report.checks.append(
                            AuditCheck(
                                name="privacy_guards",
                                status=OutcomeStatus.ERROR,
                                summary=f"Committed element schema is unreadable: {type(error).__name__}",
                                provider=element_scope,
                            )
                        )
                    if schema is None:
                        if element.supported:
                            report.checks.append(
                                AuditCheck(
                                    name="privacy_guards",
                                    status=OutcomeStatus.ERROR,
                                    summary="Committed element schema is missing",
                                    provider=element_scope,
                                )
                            )
                        continue
                    report.checks.append(_scoped(element_scope, check_privacy_guards(schema)))
            if len(report.checks) == scope_check_start and manifests:
                report.checks.append(
                    AuditCheck(
                        name="privacy_guards",
                        status=OutcomeStatus.OK,
                        summary="No supported schema artifacts require privacy audit",
                        provider=scope,
                    )
                )
    if not report.checks:
        report.checks.append(
            AuditCheck(
                name="privacy_guards",
                status=OutcomeStatus.ERROR,
                summary="No committed schema bundles were discovered",
            )
        )
    return report


__all__ = ["audit_all_providers", "audit_provider", "audit_schema_bundle_privacy"]
