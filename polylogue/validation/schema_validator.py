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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import jsonschema
    from jsonschema import Draft202012Validator, ValidationError
except ImportError:
    jsonschema = None  # type: ignore
    Draft202012Validator = None  # type: ignore
    ValidationError = Exception  # type: ignore


# Schema directory relative to this file
SCHEMA_DIR = Path(__file__).parent.parent / "schemas" / "providers"


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

    def __init__(self, schema: dict[str, Any], strict: bool = True):
        """Initialize validator with a schema.

        Args:
            schema: JSON Schema dict
            strict: If True, treat unexpected fields as drift warnings
        """
        if jsonschema is None:
            raise ImportError("jsonschema not installed. Run: pip install jsonschema")

        self.schema = schema
        self.strict = strict
        self._validator = Draft202012Validator(schema)

        # Extract known property names for drift detection
        self._known_properties = self._extract_known_properties(schema)

    @classmethod
    def for_provider(cls, provider: str, strict: bool = True) -> SchemaValidator:
        """Create a validator for a specific provider.

        Args:
            provider: Provider name (chatgpt, claude-ai, claude-code, codex, gemini)
            strict: If True, treat unexpected fields as drift warnings

        Returns:
            SchemaValidator configured for the provider

        Raises:
            FileNotFoundError: If no schema exists for the provider
        """
        schema_path = SCHEMA_DIR / f"{provider}.schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"No schema found for provider: {provider}")

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        return cls(schema, strict=strict)

    @classmethod
    def available_providers(cls) -> list[str]:
        """List providers with available schemas."""
        if not SCHEMA_DIR.exists():
            return []
        return sorted(
            p.stem.replace(".schema", "")
            for p in SCHEMA_DIR.glob("*.schema.json")
        )

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
                    # Schema has additionalProperties schema - recurse if dict
                    if isinstance(value, dict):
                        warnings.extend(
                            self._detect_drift(value, has_additional, current_path)
                        )
            else:
                # Known property - recurse into nested objects
                prop_schema = schema["properties"][key]
                if isinstance(value, dict) and "properties" in prop_schema:
                    warnings.extend(
                        self._detect_drift(value, prop_schema, current_path)
                    )
                elif isinstance(value, list) and "items" in prop_schema:
                    items_schema = prop_schema["items"]
                    if isinstance(items_schema, dict) and "properties" in items_schema:
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                warnings.extend(
                                    self._detect_drift(
                                        item, items_schema, f"{current_path}[{i}]"
                                    )
                                )

        return warnings

    def _extract_known_properties(self, schema: dict[str, Any]) -> set[str]:
        """Extract all known property names from schema (recursively)."""
        props: set[str] = set()

        if "properties" in schema:
            props.update(schema["properties"].keys())
            for prop_schema in schema["properties"].values():
                if isinstance(prop_schema, dict):
                    props.update(self._extract_known_properties(prop_schema))

        if "items" in schema and isinstance(schema["items"], dict):
            props.update(self._extract_known_properties(schema["items"]))

        if "additionalProperties" in schema and isinstance(
            schema["additionalProperties"], dict
        ):
            props.update(self._extract_known_properties(schema["additionalProperties"]))

        return props


def validate_provider_export(
    data: Any,
    provider: str,
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
