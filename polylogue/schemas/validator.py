"""JSON Schema validation for provider exports with drift detection.

This module provides strict validation of provider export formats against
generated schemas, with special handling for drift detection (new/changed fields).

Usage:
    validator = SchemaValidator.for_provider("chatgpt")
    result = validator.validate(export_data)

    if result.is_valid:
        print("Export validates against schema")
    if result.has_drift:
        print(f"Detected drift: {result.drift_warnings}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

try:
    import jsonschema
    from jsonschema import Draft202012Validator, ValidationError
except ImportError:
    jsonschema = None
    Draft202012Validator = None
    ValidationError = Exception

from polylogue.lib.raw_payload import extract_payload_samples
from polylogue.schemas.registry import SchemaRegistry, canonical_schema_provider
from polylogue.types import Provider

_RECORD_VALIDATION_PROVIDERS = {Provider.CLAUDE_CODE, Provider.CODEX}

_UUID_KEY_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

@dataclass
class ValidationResult:
    """Result of schema validation with drift detection."""

    is_valid: bool
    """Whether the data passes schema validation."""

    errors: list[str] = field(default_factory=list)
    """Validation errors (required fields missing, type mismatches)."""

    drift_warnings: list[str] = field(default_factory=list)
    """Drift warnings (new fields, changed structures)."""

    @property
    def has_drift(self) -> bool:
        """Whether drift was detected (valid but changed)."""
        return len(self.drift_warnings) > 0

    def raise_if_invalid(self) -> None:
        """Raise ValueError if validation failed."""
        if not self.is_valid:
            raise ValueError(f"Schema validation failed: {'; '.join(self.errors)}")


class SchemaValidator:
    """Validates data against JSON schemas with drift detection.

    Drift detection identifies changes that don't break the schema but
    indicate the provider format may have changed:
    - New fields not in the schema
    - Fields with unexpected types (when schema allows multiple)
    - Nested structures that differ from samples
    """

    # Class-level cache: avoids re-reading schema files and re-compiling
    # validators for the same provider during a pipeline run.
    _cache: dict[tuple[str, bool], SchemaValidator] = {}

    def __init__(self, schema: dict[str, Any], strict: bool = True, provider: Provider | None = None):
        """Initialize validator with a schema.

        Args:
            schema: JSON Schema dict
            strict: If True, treat unexpected fields as drift warnings
        """
        if jsonschema is None:
            raise ImportError("jsonschema not installed. Run: pip install jsonschema")

        self.schema = schema
        self.strict = strict
        self.provider = provider
        self._validator = Draft202012Validator(schema)

    @classmethod
    def canonical_provider(cls, provider: str | Provider) -> Provider:
        """Normalize provider names to canonical schema provider names."""
        return Provider.from_string(canonical_schema_provider(str(provider)))

    @classmethod
    def for_provider(cls, provider: str | Provider, strict: bool = True) -> SchemaValidator:
        """Create a validator for a specific provider, caching by (provider, strict).

        Args:
            provider: Provider name (chatgpt, claude-ai, claude-code, codex, gemini)
            strict: If True, treat unexpected fields as drift warnings

        Returns:
            SchemaValidator configured for the provider

        Raises:
            FileNotFoundError: If no schema exists for the provider
        """
        canonical_provider = cls.canonical_provider(provider)
        key = (str(canonical_provider), strict)
        cached = cls._cache.get(key)
        if cached is not None:
            return cached

        schema = SchemaRegistry().get_schema(str(canonical_provider), version="latest")
        if schema is None:
            raise FileNotFoundError(f"No schema found for provider: {provider} (canonical: {canonical_provider})")
        instance = cls(schema, strict=strict, provider=canonical_provider)
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
        """Create a validator matched to the most likely schema version."""
        canonical_provider = cls.canonical_provider(provider)
        registry = SchemaRegistry()
        version = registry.match_payload_version(
            str(canonical_provider),
            payload,
            source_path=source_path,
        ) or "latest"
        key = (f"{canonical_provider}:{version}", strict)
        cached = cls._cache.get(key)
        if cached is not None:
            return cached

        schema = registry.get_schema(str(canonical_provider), version=version)
        if schema is None:
            raise FileNotFoundError(
                f"No schema found for provider: {provider} (canonical: {canonical_provider}, version: {version})"
            )
        instance = cls(schema, strict=strict, provider=canonical_provider)
        cls._cache[key] = instance
        return instance

    @classmethod
    def available_providers(cls) -> list[str]:
        """List providers with available schemas."""
        return SchemaRegistry().list_providers()

    def validate(self, data: Any) -> ValidationResult:
        """Validate data against the schema with drift detection.

        Args:
            data: Data to validate (typically a dict from JSON)

        Returns:
            ValidationResult with is_valid, errors, and drift_warnings
        """
        errors: list[str] = []
        drift_warnings: list[str] = []

        # Run schema validation
        for error in self._validator.iter_errors(data):
            errors.append(self._format_error(error))

        # Drift detection (only if data is a dict and strict mode)
        if self.strict and isinstance(data, dict):
            drift_warnings.extend(self._detect_drift(data, self.schema, ""))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            drift_warnings=drift_warnings,
        )

    def validation_samples(self, payload: Any, *, max_samples: int | None = None) -> list[dict[str, Any]]:
        """Extract representative objects from a payload for validation.

        For record-oriented providers (JSONL), this returns a stratified subset of
        record dicts when explicitly bounded. By default it validates all record
        dicts. For document-oriented providers, this returns the top-level payload
        object or all dict documents in a list payload.
        """
        granularity = self.schema.get("x-polylogue-sample-granularity")
        if not isinstance(granularity, str):
            granularity = "record" if self.provider in _RECORD_VALIDATION_PROVIDERS else "document"
        return extract_payload_samples(
            payload,
            sample_granularity=granularity,
            max_samples=max_samples,
        )

    def _format_error(self, error: ValidationError) -> str:
        """Format a validation error for display."""
        path = ".".join(str(p) for p in error.absolute_path) or "root"
        return f"{path}: {error.message}"

    def _detect_drift(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
        path: str,
    ) -> list[str]:
        """Detect fields in data not present in schema (drift)."""
        warnings: list[str] = []

        # Get expected properties from schema
        schema_props = set(schema.get("properties", {}).keys())
        has_additional = schema.get("additionalProperties", True)

        # Check for unexpected fields
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key

            if key not in schema_props:
                if has_additional is False:
                    # Schema explicitly disallows additional properties
                    warnings.append(f"Unexpected field: {current_path}")
                elif has_additional is True:
                    # Schema allows any additional properties - no warning
                    pass
                elif isinstance(has_additional, dict):
                    # Field is allowed by additionalProperties schema but not named -> Drift
                    dynamic_container = bool(schema.get("x-polylogue-dynamic-keys"))
                    if not dynamic_container and not self._looks_dynamic_key(key):
                        warnings.append(f"Unexpected field: {current_path}")
                    # Recurse if value is dict to find nested drift
                    if isinstance(value, dict):
                        warnings.extend(self._detect_drift(value, has_additional, current_path))
            else:
                # Known property - recurse into nested objects
                prop_schema = schema["properties"][key]
                if isinstance(value, dict) and "properties" in prop_schema:
                    warnings.extend(self._detect_drift(value, prop_schema, current_path))
                elif isinstance(value, list) and "items" in prop_schema:
                    items_schema = prop_schema["items"]
                    if isinstance(items_schema, dict) and "properties" in items_schema:
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                warnings.extend(self._detect_drift(item, items_schema, f"{current_path}[{i}]"))

        return warnings

    def _looks_dynamic_key(self, key: str) -> bool:
        """Detect dynamic identifier keys (UUIDs, hashes, generated IDs)."""
        if _UUID_KEY_RE.match(key):
            return True
        if re.match(r"^[0-9a-f]{24,}$", key, re.IGNORECASE):
            return True
        return bool(re.match(r"^(msg|node|conv|item|att)-[0-9a-f-]+$", key, re.IGNORECASE))

def validate_provider_export(
    data: Any,
    provider: str | Provider,
    strict: bool = True,
) -> ValidationResult:
    """Convenience function to validate a provider export.

    Args:
        data: Export data to validate
        provider: Provider name
        strict: If True, detect drift (new/changed fields)

    Returns:
        ValidationResult
    """
    validator = SchemaValidator.for_provider(provider, strict=strict)
    return validator.validate(data)
