"""Schema resolution helpers for validator cache keys and schema loading."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.paths import data_home
from polylogue.schemas.runtime_registry import SchemaRegistry, canonical_schema_provider

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


def _historical_schemas(
    registry: SchemaRegistry,
    provider: Provider,
    *,
    element_kind: str,
) -> Iterator[tuple[str, JSONDocument]]:
    catalog = registry.load_package_catalog(str(provider))
    if catalog is None:
        return
    for package in registry._ranked_packages(catalog):
        element = package.element(element_kind)
        if element is None or (
            package.observation_status != "historical" and element.observation_status != "historical"
        ):
            continue
        schema = _load_schema(
            registry,
            provider,
            package_version=package.version,
            element_kind=element_kind,
        )
        yield package.version, schema


def _choose_retained_schema(
    base_version: str,
    base_schema: JSONDocument,
    historical_schemas: Iterator[tuple[str, JSONDocument]],
    *,
    schema_accepts: Callable[[JSONDocument], bool],
) -> tuple[str, JSONDocument]:
    """Apply the shared retained current-then-historical acceptance order."""
    if schema_accepts(base_schema):
        return base_version, base_schema
    for historical_version, historical_schema in historical_schemas:
        if schema_accepts(historical_schema):
            return historical_version, historical_schema
    return base_version, base_schema


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
    schema_resolution_is_explicit: bool = True,
    registry_cls: type[SchemaRegistry] = SchemaRegistry,
    schema_accepts: Callable[[JSONDocument], bool] | None = None,
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
    if (
        (schema_resolution is not None and schema_resolution_is_explicit)
        or schema_accepts is None
        or schema_accepts(schema)
    ):
        return canonical, schema, (str(canonical), package_version, element_kind)

    for historical_version, historical_schema in _historical_schemas(
        registry,
        canonical,
        element_kind=element_kind,
    ):
        if schema_accepts(historical_schema):
            return canonical, historical_schema, (str(canonical), historical_version, element_kind)
    return canonical, schema, (str(canonical), package_version, element_kind)


def resolve_retained_schema(
    provider: str | Provider,
    payload: object,
    *,
    source_path: str | None = None,
    schema_resolution: SchemaResolution | None = None,
    schema_resolution_is_explicit: bool = False,
    registry: SchemaRegistry,
    schema_store: Callable[[object], JSONDocument] | None = None,
    schema_accepts: Callable[[JSONDocument], bool] | None = None,
) -> tuple[Provider, JSONDocument, tuple[str, str, str], SchemaResolution | None]:
    """Resolve and validate a retained lazy document without materializing it.

    Shape observation and validation borrow the caller's replayable document
    view.  Inferred selection preserves the ordinary registry precedence and
    tries ranked historical packages only after the selected schema rejects
    the complete sample stream.  Each candidate receives a fresh iterator
    from the caller, so a failed package cannot consume evidence needed by a
    later one.
    """
    canonical = canonical_provider(provider)
    resolution = schema_resolution
    if resolution is None:
        observations, _cluster = registry.observe_payload(
            str(canonical),
            cast(JSONValue, payload),
            source_path=source_path,
            schema_store=schema_store,
        )
        resolution = registry.resolve_observation(str(canonical), observations, source_path=source_path)

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
    if (schema_resolution is not None and schema_resolution_is_explicit) or schema_accepts is None:
        key = (str(canonical), package_version, element_kind)
        return canonical, schema, key, _replace_resolution(resolution, canonical, key)
    package_version, schema = _choose_retained_schema(
        package_version,
        schema,
        _historical_schemas(registry, canonical, element_kind=element_kind),
        schema_accepts=schema_accepts,
    )
    key = (str(canonical), package_version, element_kind)
    return canonical, schema, key, _replace_resolution(resolution, canonical, key)


def _replace_resolution(
    resolution: SchemaResolution | None,
    canonical: Provider,
    key: tuple[str, str, str],
) -> SchemaResolution | None:
    if resolution is None:
        return None
    from dataclasses import replace

    return replace(resolution, provider=str(canonical), package_version=key[1], element_kind=key[2])


def available_providers(*, registry_cls: type[SchemaRegistry] = SchemaRegistry) -> list[str]:
    registry = _registry_for(registry_cls)
    providers = registry.list_providers()
    return [str(provider) for provider in providers]
