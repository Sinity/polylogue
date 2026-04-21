"""Schema resolution helpers for validator cache keys and schema loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.lib.json import JSONDocument
from polylogue.paths import data_home
from polylogue.schemas.runtime_registry import SchemaRegistry, canonical_schema_provider
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.schemas.packages import SchemaResolution, SchemaVersionPackage


@lru_cache(maxsize=8)
def _shared_registry(storage_root: str) -> SchemaRegistry:
    return SchemaRegistry(storage_root=Path(storage_root))


def _registry_for(registry_cls: type[SchemaRegistry]) -> SchemaRegistry:
    if registry_cls is SchemaRegistry:
        return _shared_registry(str(data_home() / "schemas"))
    return registry_cls()


def _load_package(
    registry: SchemaRegistry,
    provider: Provider,
    *,
    version: str,
) -> SchemaVersionPackage:
    package = registry.get_package(str(provider), version=version)
    if package is None:
        raise FileNotFoundError(f"No schema found for provider: {provider} (version: {version})")
    return package


def _load_schema(
    registry: SchemaRegistry,
    provider: Provider,
    *,
    package_version: str,
    element_kind: str,
) -> JSONDocument:
    schema = registry.get_element_schema(
        str(provider),
        version=package_version,
        element_kind=element_kind,
    )
    if schema is None:
        raise FileNotFoundError(
            f"No schema found for provider: {provider} (package: {package_version}, element: {element_kind})"
        )
    return schema


def _load_latest_schema(
    registry: object,
    provider: Provider,
) -> JSONDocument | None:
    get_schema = getattr(registry, "get_schema", None)
    if not callable(get_schema):
        return None
    latest_schema = get_schema(str(provider), version="latest")
    if isinstance(latest_schema, dict):
        return latest_schema
    default_schema = get_schema(str(provider), version="default")
    return default_schema if isinstance(default_schema, dict) else None


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
    if not hasattr(registry, "get_package"):
        schema = _load_latest_schema(registry, canonical)
        if schema is None:
            raise FileNotFoundError(f"No schema found for provider: {canonical}")
        return canonical, schema, (str(canonical), "latest", "conversation_document")
    package = _load_package(registry, canonical, version="default")
    package_version = package.version
    element_kind = package.default_element_kind
    schema = _load_schema(
        registry,
        canonical,
        package_version=package_version,
        element_kind=element_kind,
    )
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
        resolution = registry.resolve_payload(
            str(canonical),
            payload,
            source_path=source_path,
        )

    if resolution is None:
        package = _load_package(registry, canonical, version="default")
        package_version = package.version
        element_kind = package.default_element_kind
    else:
        package_version = resolution.package_version
        element_kind = resolution.element_kind

    schema = _load_schema(
        registry,
        canonical,
        package_version=package_version,
        element_kind=element_kind,
    )
    return canonical, schema, (str(canonical), package_version, element_kind)


def available_providers(*, registry_cls: type[SchemaRegistry] = SchemaRegistry) -> list[str]:
    return _registry_for(registry_cls).list_providers()
