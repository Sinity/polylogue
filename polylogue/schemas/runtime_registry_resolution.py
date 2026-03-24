"""Payload-to-package resolution for runtime schema authority."""

from __future__ import annotations

from typing import Any

from polylogue.schemas.observation import (
    derive_bundle_scope,
    extract_schema_units_from_payload,
    profile_similarity,
    resolve_provider_config,
)
from polylogue.schemas.packages import SchemaElementManifest, SchemaResolution, SchemaVersionPackage
from polylogue.schemas.runtime_registry_support import canonical_schema_provider
from polylogue.schemas.runtime_registry_write import SchemaRegistryWriteMixin


class SchemaRegistryResolutionMixin(SchemaRegistryWriteMixin):
    """Payload resolution against runtime schema catalogs."""

    def resolve_payload(
        self,
        provider: str,
        payload: Any,
        *,
        source_path: str | None = None,
    ) -> SchemaResolution | None:
        provider_token = str(canonical_schema_provider(provider))
        catalog = self.load_package_catalog(provider_token)
        if catalog is None or not catalog.packages:
            return None

        config = resolve_provider_config(provider_token)
        units = extract_schema_units_from_payload(
            payload,
            provider_name=provider_token,
            source_path=source_path,
            raw_id=None,
            observed_at=None,
            config=config,
            max_samples=64,
        )
        if not units:
            default_package = self.get_package(provider_token, version="default")
            if default_package is None:
                return None
            return SchemaResolution(
                provider=provider_token,
                package_version=default_package.version,
                element_kind=default_package.default_element_kind,
                exact_structure_id=None,
                bundle_scope=derive_bundle_scope(provider_token, source_path),
                reason="package_default",
            )

        unit = units[0]
        bundle_scope = unit.bundle_scope or derive_bundle_scope(provider_token, source_path)

        bundle_resolution = self._resolve_by_bundle_scope(catalog.packages, unit.artifact_kind, bundle_scope, provider_token, unit.exact_structure_id)
        if bundle_resolution is not None:
            return bundle_resolution

        exact_resolution = self._resolve_by_exact_structure(catalog.packages, unit.artifact_kind, unit.exact_structure_id, bundle_scope, provider_token)
        if exact_resolution is not None:
            return exact_resolution

        profile_resolution = self._resolve_by_profile_similarity(catalog.packages, unit.artifact_kind, unit.profile_tokens, unit.exact_structure_id, bundle_scope, provider_token)
        if profile_resolution is not None:
            return profile_resolution

        default_package = self.get_package(provider_token, version="default")
        if default_package is None:
            return None
        return SchemaResolution(
            provider=provider_token,
            package_version=default_package.version,
            element_kind=default_package.default_element_kind,
            exact_structure_id=unit.exact_structure_id,
            bundle_scope=bundle_scope,
            reason="package_default",
        )

    def _resolve_by_bundle_scope(
        self,
        packages: list[SchemaVersionPackage],
        artifact_kind: str,
        bundle_scope: str | None,
        provider: str,
        exact_structure_id: str | None,
    ) -> SchemaResolution | None:
        if bundle_scope is None:
            return None
        for package in packages:
            if bundle_scope not in package.bundle_scopes:
                continue
            element = package.element(artifact_kind)
            if element is None:
                continue
            return SchemaResolution(
                provider=provider,
                package_version=package.version,
                element_kind=element.element_kind,
                exact_structure_id=exact_structure_id,
                bundle_scope=bundle_scope,
                reason="bundle_scope",
            )
        return None

    def _resolve_by_exact_structure(
        self,
        packages: list[SchemaVersionPackage],
        artifact_kind: str,
        exact_structure_id: str | None,
        bundle_scope: str | None,
        provider: str,
    ) -> SchemaResolution | None:
        for package in packages:
            element = package.element(artifact_kind)
            if element is None:
                continue
            if exact_structure_id in element.exact_structure_ids:
                return SchemaResolution(
                    provider=provider,
                    package_version=package.version,
                    element_kind=element.element_kind,
                    exact_structure_id=exact_structure_id,
                    bundle_scope=bundle_scope,
                    reason="exact_structure",
                )
        return None

    def _resolve_by_profile_similarity(
        self,
        packages: list[SchemaVersionPackage],
        artifact_kind: str,
        profile_tokens: list[str],
        exact_structure_id: str | None,
        bundle_scope: str | None,
        provider: str,
    ) -> SchemaResolution | None:
        best_match: tuple[float, SchemaVersionPackage, SchemaElementManifest] | None = None
        for package in packages:
            element = package.element(artifact_kind)
            if element is None or not element.profile_tokens:
                continue
            score = profile_similarity(set(element.profile_tokens), set(profile_tokens))
            if best_match is None or score > best_match[0]:
                best_match = (score, package, element)
        if best_match is None or best_match[0] <= 0.0:
            return None
        _score, package, element = best_match
        return SchemaResolution(
            provider=provider,
            package_version=package.version,
            element_kind=element.element_kind,
            exact_structure_id=exact_structure_id,
            bundle_scope=bundle_scope,
            reason="profile_family",
        )

    def match_payload_version(
        self,
        provider: str,
        payload: Any,
        *,
        source_path: str | None = None,
    ) -> str | None:
        resolution = self.resolve_payload(provider, payload, source_path=source_path)
        return resolution.package_version if resolution is not None else None


__all__ = ["SchemaRegistryResolutionMixin"]
