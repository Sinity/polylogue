"""JSON Schema validation for provider exports with drift detection."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError:
    jsonschema = None
    Draft202012Validator = None

from polylogue.lib.raw_payload import extract_payload_samples
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.types import Provider

from .validator_resolution import (
    available_providers as _available_providers,
)
from .validator_resolution import (
    canonical_provider as _canonical_provider,
)
from .validator_resolution import (
    resolve_payload_schema,
    resolve_provider_schema,
)

if TYPE_CHECKING:
    from polylogue.schemas.packages import SchemaResolution

ValidationSchema: TypeAlias = dict[str, Any]

# ---------------------------------------------------------------------------
# Validation result model
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of schema validation with drift detection."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    drift_warnings: list[str] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return len(self.drift_warnings) > 0

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            raise ValueError(f"Schema validation failed: {'; '.join(self.errors)}")


# ---------------------------------------------------------------------------
# Validation support helpers
# ---------------------------------------------------------------------------

_RECORD_VALIDATION_PROVIDERS = {Provider.CLAUDE_CODE, Provider.CODEX}

_UUID_KEY_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def collect_validation_samples(
    payload: object,
    *,
    schema: ValidationSchema,
    provider: Provider | None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Extract representative objects from a payload for validation."""
    raw_granularity = schema.get("x-polylogue-sample-granularity")
    granularity: Literal["document", "record"]
    if raw_granularity == "record":
        granularity = "record"
    elif raw_granularity == "document":
        granularity = "document"
    else:
        granularity = "record" if provider in _RECORD_VALIDATION_PROVIDERS else "document"
    return extract_payload_samples(
        payload,
        sample_granularity=granularity,
        max_samples=max_samples,
    )


def format_validation_error(error: Any) -> str:
    path = ".".join(str(part) for part in error.absolute_path) or "root"
    return f"{path}: {error.message}"


def _schema_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def detect_drift(
    data: dict[str, Any],
    schema: Mapping[str, object],
    path: str,
) -> list[str]:
    """Detect fields in data not present in schema (drift)."""
    warnings: list[str] = []
    schema_props_mapping = _schema_mapping(schema.get("properties", {}))
    schema_props = set(schema_props_mapping.keys())
    has_additional = schema.get("additionalProperties", True)

    for key, value in data.items():
        current_path = f"{path}.{key}" if path else key

        if key not in schema_props:
            if has_additional is False:
                warnings.append(f"Unexpected field: {current_path}")
            elif has_additional is True:
                pass
            elif isinstance(has_additional, Mapping):
                dynamic_container = bool(schema.get("x-polylogue-dynamic-keys"))
                if not dynamic_container and not looks_dynamic_key(key):
                    warnings.append(f"Unexpected field: {current_path}")
                if isinstance(value, dict):
                    warnings.extend(detect_drift(value, has_additional, current_path))
        else:
            prop_schema = _schema_mapping(schema_props_mapping.get(key))
            if isinstance(value, dict) and "properties" in prop_schema:
                warnings.extend(detect_drift(value, prop_schema, current_path))
            elif isinstance(value, list) and "items" in prop_schema:
                items_schema = _schema_mapping(prop_schema.get("items"))
                if "properties" in items_schema:
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            warnings.extend(detect_drift(item, items_schema, f"{current_path}[{i}]"))

    return warnings


def looks_dynamic_key(key: str) -> bool:
    """Detect dynamic identifier keys (UUIDs, hashes, generated IDs)."""
    if _UUID_KEY_RE.match(key):
        return True
    if re.match(r"^[0-9a-f]{24,}$", key, re.IGNORECASE):
        return True
    return bool(re.match(r"^(msg|node|conv|item|att)-[0-9a-f-]+$", key, re.IGNORECASE))


class SchemaValidator:
    """Validates data against JSON schemas with drift detection."""

    _cache: dict[tuple[str, str, str, bool], SchemaValidator] = {}

    def __init__(self, schema: ValidationSchema, strict: bool = True, provider: Provider | None = None):
        if jsonschema is None:
            raise ImportError("jsonschema not installed. Run: pip install jsonschema")

        self.schema = schema
        self.strict = strict
        self.provider = provider
        self._validator = Draft202012Validator(schema)

    @classmethod
    def canonical_provider(cls, provider: str | Provider) -> Provider:
        return _canonical_provider(provider)

    @classmethod
    def for_provider(cls, provider: str | Provider, strict: bool = True) -> SchemaValidator:
        canonical, schema, base_key = resolve_provider_schema(provider, registry_cls=SchemaRegistry)
        key = (*base_key, strict)
        cached = cls._cache.get(key)
        if cached is not None:
            return cached
        instance = cls(schema, strict=strict, provider=canonical)
        cls._cache[key] = instance
        return instance

    @classmethod
    def for_payload(
        cls,
        provider: str | Provider,
        payload: object,
        *,
        source_path: str | None = None,
        schema_resolution: SchemaResolution | None = None,
        strict: bool = True,
    ) -> SchemaValidator:
        canonical, schema, base_key = resolve_payload_schema(
            provider,
            payload,
            source_path=source_path,
            schema_resolution=schema_resolution,
            registry_cls=SchemaRegistry,
        )
        key = (*base_key, strict)
        cached = cls._cache.get(key)
        if cached is not None:
            return cached
        instance = cls(schema, strict=strict, provider=canonical)
        cls._cache[key] = instance
        return instance

    @classmethod
    def available_providers(cls) -> list[str]:
        return _available_providers(registry_cls=SchemaRegistry)

    def validate(self, data: object, *, include_drift: bool | None = None) -> ValidationResult:
        errors = [format_validation_error(error) for error in self._validator.iter_errors(data)]
        drift_warnings: list[str] = []
        should_detect_drift = self.strict if include_drift is None else include_drift
        if should_detect_drift and isinstance(data, dict):
            drift_warnings.extend(detect_drift(data, self.schema, ""))
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            drift_warnings=drift_warnings,
        )

    def validation_samples(self, payload: object, *, max_samples: int | None = None) -> list[dict[str, Any]]:
        return collect_validation_samples(
            payload,
            schema=self.schema,
            provider=self.provider,
            max_samples=max_samples,
        )

    def _looks_dynamic_key(self, key: str) -> bool:
        return looks_dynamic_key(key)


def validate_provider_export(
    data: object,
    provider: str | Provider,
    strict: bool = True,
) -> ValidationResult:
    """Convenience function to validate a provider export."""
    validator = SchemaValidator.for_provider(provider, strict=strict)
    return validator.validate(data)
