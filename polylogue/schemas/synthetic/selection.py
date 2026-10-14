"""Package-aware provider selection for synthetic generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, TypeAlias

from polylogue.schemas.packages import SchemaVersionPackage
from polylogue.schemas.runtime_registry import SchemaRegistry, canonical_schema_provider
from polylogue.schemas.synthetic.models import SchemaRecord, SyntheticSchemaSelection
from polylogue.schemas.synthetic.wire_formats import (
    PROVIDER_WIRE_ROUTES,
    UnsupportedSyntheticWireRouteError,
)


class _SchemaRegistryLike(Protocol):
    def get_package(
        self,
        provider: str,
        version: str = "default",
    ) -> SchemaVersionPackage | None: ...

    def get_element_schema(
        self,
        provider: str,
        *,
        version: str = "default",
        element_kind: str | None = None,
    ) -> SchemaRecord | None: ...

    def get_schema(self, provider: str, version: str = "default") -> SchemaRecord | None: ...

    def get_workload_profile(self, provider: str, version: str = "default") -> Mapping[str, object] | None: ...

    def list_providers(self) -> list[str]: ...


RegistryFactory: TypeAlias = Callable[[], _SchemaRegistryLike]
CanonicalProviderResolver: TypeAlias = Callable[[str], str]


def _default_registry_factory() -> _SchemaRegistryLike:
    return SchemaRegistry()


def select_synthetic_schema(
    provider: str,
    *,
    version: str = "default",
    element_kind: str | None = None,
    registry_factory: RegistryFactory = _default_registry_factory,
    canonical_provider_resolver: CanonicalProviderResolver = canonical_schema_provider,
) -> SyntheticSchemaSelection:
    canonical_provider = canonical_provider_resolver(provider)
    registry = registry_factory()
    package = registry.get_package(canonical_provider, version=version)
    resolved_element_kind = element_kind

    schema: SchemaRecord | None
    if package is not None:
        if resolved_element_kind is None or resolved_element_kind == "default":
            resolved_element_kind = package.default_element_kind
        if package.element(resolved_element_kind) is None:
            raise ValueError(
                "No element kind "
                f"{resolved_element_kind!r} in package {package.version} for provider "
                f"{canonical_provider}"
            )
        schema = registry.get_element_schema(
            canonical_provider,
            # ``get_package`` resolves aliases such as ``default``.  Reuse
            # that exact package identity for every subsequent load so a
            # registry cannot pair one package's manifest with another
            # package's element schema.
            version=package.version,
            element_kind=resolved_element_kind,
        )
        canonical_version = package.version
    else:
        if element_kind is not None and element_kind != "default":
            raise ValueError(
                f"Element schemas are not available for provider {canonical_provider}; "
                f"cannot request element_kind={element_kind!r}"
            )
        schema = registry.get_schema(canonical_provider, version=version)
        canonical_version = version
        resolved_element_kind = None if element_kind is None else element_kind

    if schema is None:
        raise FileNotFoundError(
            f"No schema for provider {provider} (canonical: {canonical_provider}, "
            f"version={canonical_version}, element_kind={resolved_element_kind})"
        )

    route = PROVIDER_WIRE_ROUTES.get(canonical_provider)
    if route is None:
        raise UnsupportedSyntheticWireRouteError(canonical_provider, "no explicit synthetic wire route")
    if route.status == "unsupported":
        raise UnsupportedSyntheticWireRouteError(canonical_provider, route.reason or "route is unsupported")
    wire_format = route.wire_format
    if wire_format is None:
        raise UnsupportedSyntheticWireRouteError(canonical_provider, "supported route has no wire format")

    workload_profile = registry.get_workload_profile(canonical_provider, canonical_version)
    element_schemas: dict[str, SchemaRecord] = {}
    if resolved_element_kind is not None:
        element_schemas[resolved_element_kind] = schema
    if workload_profile is not None:
        declared = workload_profile.get("elements")
        if not isinstance(declared, dict):
            raise ValueError("package workload has no declared element structures")
        if package is None:
            raise ValueError("a package workload requires its original package manifest")
        for kind in sorted(declared):
            if kind in element_schemas:
                continue
            if package.element(kind) is None:
                raise ValueError("package workload names an element outside its original manifest")
            owned = registry.get_element_schema(canonical_provider, version=canonical_version, element_kind=kind)
            if owned is None:
                raise FileNotFoundError(f"Declared workload element {kind!r} has no original package schema")
            element_schemas[kind] = owned
    return SyntheticSchemaSelection(
        provider=canonical_provider,
        package_version=canonical_version,
        element_kind=resolved_element_kind,
        schema=schema,
        wire_format=wire_format,
        workload_profile=workload_profile,
        element_schemas=element_schemas,
    )


def available_synthetic_providers(
    *,
    registry_factory: RegistryFactory = _default_registry_factory,
) -> list[str]:
    schema_providers = set(registry_factory().list_providers())
    return [
        provider
        for provider, route in PROVIDER_WIRE_ROUTES.items()
        if route.status == "supported" and provider in schema_providers
    ]


__all__ = [
    "SchemaRegistry",
    "available_synthetic_providers",
    "canonical_schema_provider",
    "select_synthetic_schema",
]
