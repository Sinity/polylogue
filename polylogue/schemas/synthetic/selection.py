"""Package-aware provider selection for synthetic generation."""

from __future__ import annotations

from polylogue.schemas.runtime_registry import SchemaRegistry, canonical_schema_provider
from polylogue.schemas.synthetic.models import SyntheticSchemaSelection
from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS


def select_synthetic_schema(
    provider: str,
    *,
    version: str = "default",
    element_kind: str | None = None,
) -> SyntheticSchemaSelection:
    canonical_provider = canonical_schema_provider(provider)
    registry = SchemaRegistry()
    package = registry.get_package(canonical_provider, version=version) if hasattr(registry, "get_package") else None
    resolved_element_kind = element_kind

    schema: dict[str, object] | None
    if package is not None and hasattr(registry, "get_element_schema"):
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
            version=version,
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


def available_synthetic_providers() -> list[str]:
    schema_providers = set(SchemaRegistry().list_providers())
    return [provider for provider in PROVIDER_WIRE_FORMATS if provider in schema_providers]


__all__ = [
    "SchemaRegistry",
    "available_synthetic_providers",
    "canonical_schema_provider",
    "select_synthetic_schema",
]
