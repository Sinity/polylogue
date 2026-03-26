"""JSON Schema validation for provider exports with drift detection."""

from __future__ import annotations

from typing import Any

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError:
    jsonschema = None
    Draft202012Validator = None

from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.types import Provider

from .validator_models import ValidationResult
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
from .validator_support import (
    detect_drift,
    format_validation_error,
    looks_dynamic_key,
)
from .validator_support import (
    validation_samples as collect_validation_samples,
)


class SchemaValidator:
    """Validates data against JSON schemas with drift detection."""

    _cache: dict[tuple[str, str, str, bool], SchemaValidator] = {}

    def __init__(self, schema: dict[str, Any], strict: bool = True, provider: Provider | None = None):
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
        payload: Any,
        *,
        source_path: str | None = None,
        strict: bool = True,
    ) -> SchemaValidator:
        canonical, schema, base_key = resolve_payload_schema(
            provider,
            payload,
            source_path=source_path,
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

    def validate(self, data: Any) -> ValidationResult:
        errors = [format_validation_error(error) for error in self._validator.iter_errors(data)]
        drift_warnings: list[str] = []
        if self.strict and isinstance(data, dict):
            drift_warnings.extend(detect_drift(data, self.schema, ""))
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            drift_warnings=drift_warnings,
        )

    def validation_samples(self, payload: Any, *, max_samples: int | None = None) -> list[dict[str, Any]]:
        return collect_validation_samples(
            payload,
            schema=self.schema,
            provider=self.provider,
            max_samples=max_samples,
        )

    def _looks_dynamic_key(self, key: str) -> bool:
        return looks_dynamic_key(key)


def validate_provider_export(
    data: Any,
    provider: str | Provider,
    strict: bool = True,
) -> ValidationResult:
    """Convenience function to validate a provider export."""
    validator = SchemaValidator.for_provider(provider, strict=strict)
    return validator.validate(data)
