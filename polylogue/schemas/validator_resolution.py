"""Schema resolution helpers for validator cache keys and schema loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.paths import data_home
from polylogue.schemas.json_types import JSONDocument, JSONValue
from polylogue.schemas.runtime_registry import SchemaRegistry, canonical_schema_provider
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.schemas.packages import SchemaResolution


@lru_cache(maxsize=8)
def _shared_registry(storage_root: str) -> SchemaRegistry:
    return SchemaRegistry(storage_root=Path(storage_root))


def _registry_for(registry_cls: type[SchemaRegistry]) -> SchemaRegistry:
    if registry_cls is SchemaRegistry:
        return _shared_registry(str(data_home() / "schemas"))
    return registry_cls()


def reset_registry_cache() -> None:
    """Clear shared runtime-registry instances used by schema validation."""
    _shared_registry.cache_clear()


def canonical_provider(provider: str | Provider) -> Provider:
    """Normalize provider names to canonical schema provider names."""
    return Provider.from_string(canonical_schema_provider(str(provider)))


def resolve_provider_schema(
    provider: str | Provider,
    *,
    registry_cls: type[SchemaRegistry] = SchemaRegistry,
) -> tuple[Provider, JSONDocument, tuple[str, str, str]]:
    canonical = canonical_provider(provider)
    registry = _registry_for(registry_cls)
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
    payload: object,
    *,
    source_path: str | None = None,
    schema_resolution: SchemaResolution | None = None,
    registry_cls: type[SchemaRegistry] = SchemaRegistry,
) -> tuple[Provider, JSONDocument, tuple[str, str, str]]:
    canonical = canonical_provider(provider)
    registry = _registry_for(registry_cls)
    resolution = schema_resolution
    if resolution is None:
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
    return _registry_for(registry_cls).list_providers()
