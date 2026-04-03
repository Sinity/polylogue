"""Schema resolution helpers for validator cache keys and schema loading."""

from __future__ import annotations

from typing import Any

from polylogue.schemas.runtime_registry import SchemaRegistry, canonical_schema_provider
from polylogue.types import Provider


def canonical_provider(provider: str | Provider) -> Provider:
    """Normalize provider names to canonical schema provider names."""
    return Provider.from_string(canonical_schema_provider(str(provider)))


def resolve_provider_schema(
    provider: str | Provider,
    *,
    registry_cls: type[SchemaRegistry] = SchemaRegistry,
) -> tuple[Provider, dict[str, Any], tuple[str, str, str]]:
    canonical = canonical_provider(provider)
    registry = registry_cls()
    package = registry.get_package(str(canonical), version="default") if hasattr(registry, "get_package") else None
    package_version = package.version if package is not None else "latest"
    element_kind = package.default_element_kind if package is not None else "default"

    if package is not None and hasattr(registry, "get_element_schema"):
        schema = registry.get_element_schema(
            str(canonical),
            version=package.version,
            element_kind=package.default_element_kind,
        )
    else:
        schema = registry.get_schema(str(canonical), version="latest")
    if schema is None:
        raise FileNotFoundError(f"No schema found for provider: {provider} (canonical: {canonical})")
    return canonical, schema, (str(canonical), package_version, element_kind)


def resolve_payload_schema(
    provider: str | Provider,
    payload: Any,
    *,
    source_path: str | None = None,
    registry_cls: type[SchemaRegistry] = SchemaRegistry,
) -> tuple[Provider, dict[str, Any], tuple[str, str, str]]:
    canonical = canonical_provider(provider)
    registry = registry_cls()
    resolution = (
        registry.resolve_payload(
            str(canonical),
            payload,
            source_path=source_path,
        )
        if hasattr(registry, "resolve_payload")
        else None
    )
    package_version = resolution.package_version if resolution is not None else "latest"
    element_kind = resolution.element_kind if resolution is not None else "default"

    if hasattr(registry, "get_element_schema"):
        schema = registry.get_element_schema(
            str(canonical),
            version=package_version,
            element_kind=None if element_kind == "default" else element_kind,
        )
    else:
        schema = registry.get_schema(str(canonical), version=package_version)
    if schema is None:
        raise FileNotFoundError(
            "No schema found for provider: "
            f"{provider} (canonical: {canonical}, package: {package_version}, element: {element_kind})"
        )
    return canonical, schema, (str(canonical), package_version, element_kind)


def available_providers(*, registry_cls: type[SchemaRegistry] = SchemaRegistry) -> list[str]:
    return registry_cls().list_providers()
