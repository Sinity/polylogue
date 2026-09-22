"""Schema audit workflow orchestration."""

from __future__ import annotations

import re
from pathlib import Path

from polylogue.core.outcomes import OutcomeCheck as CheckResult
from polylogue.core.outcomes import OutcomeStatus
from polylogue.core.schema_subjects import SCHEMA_SUBJECT_BY_TOKEN
from polylogue.schemas.audit.checks import (
    check_annotation_coverage,
    check_cross_provider_consistency,
    check_privacy_guards,
    check_published_paths,
    check_published_vocabulary,
    check_schema_drift,
    check_schema_staleness,
    check_semantic_roles,
)
from polylogue.schemas.audit.models import AuditCheck, AuditReport
from polylogue.schemas.audit.walkers import _load_committed_schema
from polylogue.schemas.packages import SchemaVersionPackage
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry


def _load_committed_package(provider: str) -> SchemaVersionPackage | None:
    """Load the manifest of the provider's default committed package.

    Observation time and sample counts live in the manifest rather than in the
    element schema document, which is a byte-deterministic content projection
    (see :func:`polylogue.schemas.audit.checks.check_schema_staleness`).
    """
    schema_root = Path(__file__).resolve().parent.parent / "providers"
    return SchemaRegistry(storage_root=schema_root).get_package(provider, version="default")


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


def _package_not_required_reason(provider: str) -> str | None:
    """Return the declared reason this subject needs no committed package.

    ``polylogue.core.schema_subjects`` is the authority on which subjects owe a
    package.  A subject declared ``requires_package=False`` with a reason has an
    adjudicated absence, so an audit that reported it as a missing package would
    be contradicting the declaration it is meant to enforce.  A subject that is
    undeclared, or declared as requiring a package, gets no reason here and a
    missing package stays an error.
    """
    subject = SCHEMA_SUBJECT_BY_TOKEN.get(provider)
    if subject is None or subject.requires_package:
        return None
    return subject.package_not_required_reason or "declared as not requiring a committed package"


def audit_provider(provider: str, *, db_path: Path | None = None) -> AuditReport:
    """Run all audit checks on a single provider's committed schema."""
    report = AuditReport(provider=provider)

    schema = _load_committed_schema(provider)
    if schema is None:
        not_required = _package_not_required_reason(provider)
        if not_required is not None:
            report.checks.append(
                AuditCheck(
                    name="schema_exists",
                    status=OutcomeStatus.SKIP,
                    summary=f"No committed package required for {provider}",
                    details=[not_required],
                    provider=provider,
                )
            )
            return report
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
    package = _load_committed_package(provider)
    report.checks.append(_scoped(provider, check_schema_staleness(package.last_seen if package else None)))

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
    sample_counts: dict[str, int | None] = {}
    for provider in provider_list:
        provider_report = audit_provider(provider, db_path=db_path)
        report.checks.extend(provider_report.checks)
        schema = _load_committed_schema(provider)
        if schema:
            schemas[provider] = schema
            package = _load_committed_package(provider)
            sample_counts[provider] = package.sample_count if package else None

    if len(schemas) >= 2:
        report.checks.append(check_cross_provider_consistency(schemas, sample_counts=sample_counts))

    return report


_OPAQUE_SCOPE_IDENTITY = re.compile(r"^scope:[0-9a-f]{64}$")


def _scope_identity_checks(manifest: SchemaVersionPackage, *, scope: str) -> list[AuditCheck]:
    """Refuse a manifest that publishes a readable bundle scope.

    ``bundle_scope_identity()`` exists so a catalog carries routing capability
    without serializing the local path or session identifier it was derived
    from. Nothing validated the field, so a manual promotion (or a generator
    regression) writing the raw scope published it in both ``catalog.json`` and
    ``package.json`` with every gate green.
    """
    checks: list[AuditCheck] = []
    declared: list[tuple[str, str]] = [(scope, value) for value in manifest.bundle_scope_identities]
    for element in manifest.elements:
        declared.extend((f"{scope}/{element.element_kind}", value) for value in element.bundle_scope_identities)
    for site, value in declared:
        if _OPAQUE_SCOPE_IDENTITY.fullmatch(value):
            continue
        checks.append(
            AuditCheck(
                name="privacy_guards",
                status=OutcomeStatus.ERROR,
                summary="Bundle scope identity is not an opaque scope digest",
                details=[f"length={len(value)}"],
                provider=site,
            )
        )
    return checks


def _workload_profile_checks(
    registry: SchemaRegistry,
    manifest: SchemaVersionPackage,
    *,
    provider: str,
    version: str,
    scope: str,
) -> list[AuditCheck]:
    """Audit the workload profile a manifest declares.

    ``SchemaRegistry.get_workload_profile()`` serves this artifact at runtime
    and it is committed alongside the element schemas, but the bundle audit
    inventoried element files only. A profile records observed structural
    values under ``tokens``, so an identifier or address landing there
    published with both required quick gates green.
    """
    name = manifest.workload_profile_file
    if name is None:
        return []
    try:
        profile = registry.load_committed_version_document(provider, version, name)
    except Exception as error:
        return [
            AuditCheck(
                name="privacy_guards",
                status=OutcomeStatus.ERROR,
                summary=f"Declared workload profile is unreadable: {type(error).__name__}",
                provider=scope,
            )
        ]
    if profile is None:
        return [
            AuditCheck(
                name="privacy_guards",
                status=OutcomeStatus.ERROR,
                summary="Declared workload profile is missing",
                provider=scope,
            )
        ]
    return [
        _scoped(scope, check_privacy_guards(profile)),
        _scoped(scope, check_published_paths(profile)),
    ]


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

            audited_workload_profiles: set[str] = set()
            for manifest in manifests:
                report.checks.extend(_scope_identity_checks(manifest, scope=scope))
                if manifest.workload_profile_file not in audited_workload_profiles:
                    audited_workload_profiles.add(manifest.workload_profile_file or "")
                    report.checks.extend(
                        _workload_profile_checks(
                            bundle_registry,
                            manifest,
                            provider=provider,
                            version=version,
                            scope=scope,
                        )
                    )

            declared_artifacts: dict[str, tuple[bool, str]] = {}
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
                    previous = declared_artifacts.get(element.schema_file)
                    declared_artifacts[element.schema_file] = (
                        element.supported or (previous[0] if previous is not None else False),
                        previous[1] if previous is not None else element_scope,
                    )

            committed_schema_files = set(bundle_registry.list_committed_schema_files(provider, version))
            artifact_files = sorted(set(declared_artifacts) | committed_schema_files)
            for schema_file in artifact_files:
                artifact_key = (version, schema_file)
                if artifact_key in audited_artifacts:
                    continue
                audited_artifacts.add(artifact_key)
                declared = declared_artifacts.get(schema_file)
                element_scope = declared[1] if declared is not None else f"{scope}/{schema_file}"
                supported = declared[0] if declared is not None else True
                try:
                    schema = bundle_registry.load_committed_schema_file(provider, version, schema_file)
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
                    if supported:
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
                report.checks.append(_scoped(element_scope, check_published_vocabulary(schema)))
                report.checks.append(_scoped(element_scope, check_published_paths(schema)))
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
