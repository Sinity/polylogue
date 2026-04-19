"""Package-aware provider selection for synthetic generation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeAlias, cast

from polylogue.schemas.runtime_registry import SchemaRegistry, canonical_schema_provider
from polylogue.schemas.synthetic.models import SchemaRecord, SyntheticSchemaSelection
from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS


class _SchemaVersionPackageLike(Protocol):
    version: str
    default_element_kind: str | None

    def element(self, element_kind: str | None) -> object | None: ...


class _SchemaRegistryLike(Protocol):
    def get_package(self, provider: str, version: str = "default") -> object | None: ...

    def get_element_schema(
        self,
        provider: str,
        *,
        version: str = "default",
        element_kind: str | None = None,
    ) -> object | None: ...

    def get_schema(self, provider: str, version: str = "default") -> object | None: ...

    def list_providers(self) -> list[str]: ...


RegistryFactory: TypeAlias = Callable[[], _SchemaRegistryLike]
CanonicalProviderResolver: TypeAlias = Callable[[str], str]


def _default_registry_factory() -> _SchemaRegistryLike:
    return SchemaRegistry()


def _schema_package(value: object) -> _SchemaVersionPackageLike | None:
    return cast(_SchemaVersionPackageLike, value) if value is not None else None


def _schema_record(value: object) -> SchemaRecord | None:
    return cast(SchemaRecord, value) if isinstance(value, dict) else None


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
    package = _schema_package(registry.get_package(canonical_provider, version=version))
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
        schema = _schema_record(
            registry.get_element_schema(
                canonical_provider,
                version=version,
                element_kind=resolved_element_kind,
            )
        )
        canonical_version = package.version
    else:
        if element_kind is not None and element_kind != "default":
            raise ValueError(
                f"Element schemas are not available for provider {canonical_provider}; "
                f"cannot request element_kind={element_kind!r}"
            )
        schema = _schema_record(registry.get_schema(canonical_provider, version=version))
        canonical_version = version
        resolved_element_kind = None if element_kind is None else element_kind

    if schema is None:
        raise FileNotFoundError(
            f"No schema for provider {provider} (canonical: {canonical_provider}, "
            f"version={canonical_version}, element_kind={resolved_element_kind})"
        )

    wire_format = PROVIDER_WIRE_FORMATS.get(canonical_provider)
    if not wire_format:
        raise ValueError(f"No wire format config for provider: {canonical_provider}")

    return SyntheticSchemaSelection(
        provider=canonical_provider,
        package_version=canonical_version,
        element_kind=resolved_element_kind,
        schema=schema,
        wire_format=wire_format,
    )


def available_synthetic_providers(
    *,
    registry_factory: RegistryFactory = _default_registry_factory,
) -> list[str]:
    schema_providers = set(registry_factory().list_providers())
    return [provider for provider in PROVIDER_WIRE_FORMATS if provider in schema_providers]


__all__ = [
    "SchemaRegistry",
    "available_synthetic_providers",
    "canonical_schema_provider",
    "select_synthetic_schema",
]
