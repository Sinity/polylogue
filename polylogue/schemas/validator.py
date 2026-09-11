"""JSON Schema validation for provider exports with drift detection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from re import compile as compile_pattern
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError:
    jsonschema = None
    Draft202012Validator = None

from polylogue.archive.raw_payload import extract_payload_samples
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue, is_json_value, json_document
from polylogue.schemas.field_stats.detection import is_dynamic_key
from polylogue.schemas.runtime_registry import SchemaRegistry

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

ValidationSchema: TypeAlias = Mapping[str, object]
ValidationSample: TypeAlias = JSONDocument


class ValidationErrorLike(Protocol):
    absolute_path: Iterable[object]
    message: str


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


def _schema_type_values(schema: object) -> set[str]:
    if not isinstance(schema, Mapping):
        return set()
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return {schema_type}
    if isinstance(schema_type, list):
        return {item for item in schema_type if isinstance(item, str)}
    return set()


def _schema_allows_type(schema: object, type_name: str) -> bool:
    if not isinstance(schema, Mapping):
        return False
    if type_name in _schema_type_values(schema):
        return True
    for key in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(key)
        if isinstance(branches, list) and any(_schema_allows_type(branch, type_name) for branch in branches):
            return True
    return False


def _schema_branch_for_value(schema: object, value: object) -> object:
    if not isinstance(schema, Mapping):
        return None
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if not isinstance(branches, list):
            continue
        # A union can contain several object (or array) branches.  Pick a
        # branch which accepts the concrete value before falling back to its
        # broad JSON type, otherwise a drift walk can attribute a field to an
        # unrelated sibling branch.
        for branch in branches:
            if isinstance(branch, Mapping) and _schema_accepts_value(branch, value):
                return branch
        for branch in branches:
            if isinstance(value, Mapping) and _schema_allows_type(branch, "object"):
                return branch
            if isinstance(value, list) and _schema_allows_type(branch, "array"):
                return branch
            if value is None and _schema_allows_type(branch, "null"):
                return branch
        for branch in branches:
            if isinstance(branch, Mapping):
                return branch
    return schema


def _schema_accepts_value(schema: Mapping[str, object], value: object) -> bool:
    """Return whether an individual union branch accepts ``value``.

    This is deliberately a branch-selection aid only.  The complete schema
    validator remains the authority for acceptance and error reporting.
    """
    if Draft202012Validator is None:
        return False
    try:
        return bool(Draft202012Validator(schema).is_valid(value))
    except Exception:
        # A referenced or otherwise incomplete branch still has useful type
        # information below; do not let diagnostics mask validation itself.
        return False


def _schema_for_property(schema: object, key: str, value: object) -> object:
    if not isinstance(schema, Mapping):
        return None
    properties = schema.get("properties")
    if isinstance(properties, Mapping) and key in properties:
        return _schema_branch_for_value(properties[key], value)
    pattern_properties = schema.get("patternProperties")
    if isinstance(pattern_properties, Mapping):
        matches: list[Mapping[str, object]] = []
        for pattern, pattern_schema in pattern_properties.items():
            if isinstance(pattern, str):
                try:
                    pattern_match = compile_pattern(pattern).search(key)
                except Exception:
                    pattern_match = None
                if pattern_match:
                    selected = _schema_branch_for_value(pattern_schema, value)
                    if isinstance(selected, Mapping):
                        matches.append(selected)
        if matches:
            return _merge_pattern_observation_schemas(matches)
    additional_properties = schema.get("additionalProperties")
    if isinstance(additional_properties, Mapping):
        return _schema_branch_for_value(additional_properties, value)
    return None


def _merge_pattern_observation_schemas(schemas: Iterable[Mapping[str, object]]) -> ValidationSchema:
    """Union declared observation fields from every matching pattern schema.

    JSON Schema applies *all* matching ``patternProperties`` schemas.  Drift
    observation therefore cannot pick the first match: a field declared by a
    later matching pattern would otherwise be reported as unexpected, with
    the result depending on mapping insertion order.
    """
    property_schemas: dict[str, list[Mapping[str, object]]] = {}
    nested_patterns: dict[str, list[Mapping[str, object]]] = {}
    additional_schemas: list[Mapping[str, object]] = []
    additional_forbidden = False
    dynamic_containers: list[bool] = []
    for schema in schemas:
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for name, property_schema in properties.items():
                if isinstance(name, str) and isinstance(property_schema, Mapping):
                    property_schemas.setdefault(name, []).append(property_schema)
        pattern_properties = schema.get("patternProperties")
        if isinstance(pattern_properties, Mapping):
            for pattern, pattern_schema in pattern_properties.items():
                if isinstance(pattern, str) and isinstance(pattern_schema, Mapping):
                    nested_patterns.setdefault(pattern, []).append(pattern_schema)
        additional = schema.get("additionalProperties", True)
        if additional is False:
            additional_forbidden = True
        elif isinstance(additional, Mapping):
            additional_schemas.append(additional)
        dynamic_containers.append(bool(schema.get("x-polylogue-dynamic-keys")))

    merged: dict[str, object] = {
        "type": "object",
        "properties": {
            name: _merge_pattern_observation_schemas(values) if len(values) > 1 else values[0]
            for name, values in property_schemas.items()
        },
    }
    if nested_patterns:
        merged["patternProperties"] = {
            pattern: _merge_pattern_observation_schemas(values) if len(values) > 1 else values[0]
            for pattern, values in nested_patterns.items()
        }
    if additional_forbidden:
        merged["additionalProperties"] = False
    elif additional_schemas:
        merged["additionalProperties"] = _merge_pattern_observation_schemas(additional_schemas)
    if dynamic_containers and all(dynamic_containers):
        merged["x-polylogue-dynamic-keys"] = True
    return merged


def _has_matching_pattern_property(schema: Mapping[str, object], key: str) -> bool:
    pattern_properties = schema.get("patternProperties")
    if not isinstance(pattern_properties, Mapping):
        return False
    for pattern in pattern_properties:
        if not isinstance(pattern, str):
            continue
        try:
            if compile_pattern(pattern).search(key):
                return True
        except Exception:
            continue
    return False


def _schema_for_items(schema: object, value: object) -> object:
    if not isinstance(schema, Mapping):
        return None
    del value
    items = schema.get("items")
    # Selection belongs to each concrete item.  Selecting against the parent
    # array would choose neither a nullable nor object item branch and break
    # the normalization walk which invokes this helper before it has an item.
    return items if isinstance(items, Mapping) else None


def _normalize_empty_arrays(data: object, schema: object = None) -> object:
    """Coerce empty arrays to null only when the active schema expects null.

    Some provider fields moved between ``null`` and ``[]`` across exports, but
    structural fields such as ChatGPT ``mapping.*.children`` are genuinely
    arrays. Normalization must therefore follow the selected JSON schema instead
    of rewriting every empty list globally.
    """
    schema = _schema_branch_for_value(schema, data)
    if isinstance(data, dict):
        return {
            key: _normalize_empty_arrays(value, _schema_for_property(schema, key, value)) for key, value in data.items()
        }
    if isinstance(data, list):
        if len(data) == 0:
            if _schema_allows_type(schema, "array"):
                return []
            if _schema_allows_type(schema, "null"):
                return None
            return []
        item_schema = _schema_for_items(schema, data)
        return [_normalize_empty_arrays(item, item_schema) for item in data]
    return data


def _sample_payload(value: object) -> ValidationSample | None:
    if not isinstance(value, Mapping):
        return None
    return json_document(dict(value))


def _schema_mapping(value: object) -> ValidationSchema:
    payload = _sample_payload(value)
    return payload if payload is not None else {}


def _validation_samples(
    payload: object,
    *,
    sample_granularity: Literal["document", "record"],
    max_samples: int | None,
) -> list[ValidationSample]:
    if not is_json_value(payload):
        return []
    normalized_payload: JSONValue = payload
    raw_samples = extract_payload_samples(
        normalized_payload,
        sample_granularity=sample_granularity,
        max_samples=max_samples,
    )
    return [sample for raw in raw_samples if (sample := _sample_payload(raw)) is not None]


def collect_validation_samples(
    payload: object,
    *,
    schema: ValidationSchema,
    provider: Provider | None,
    max_samples: int | None = None,
) -> list[ValidationSample]:
    """Extract representative objects from a payload for validation."""
    raw_granularity = schema.get("x-polylogue-sample-granularity")
    granularity: Literal["document", "record"]
    if raw_granularity == "record":
        granularity = "record"
    elif raw_granularity == "document":
        granularity = "document"
    else:
        granularity = "record" if provider in _RECORD_VALIDATION_PROVIDERS else "document"
    return _validation_samples(
        payload,
        sample_granularity=granularity,
        max_samples=max_samples,
    )


def format_validation_error(error: ValidationErrorLike) -> str:
    path = ".".join(str(part) for part in error.absolute_path) or "root"
    return f"{path}: {error.message}"


def detect_drift(
    data: ValidationSample,
    schema: Mapping[str, object],
    path: str,
) -> list[str]:
    """Detect newly observed named fields without changing schema acceptance.

    JSON Schema's default ``additionalProperties`` is permissive.  That is an
    admission rule, not evidence that a provider named field is already
    known: newly named fields are still reported.  Explicit dynamic-key maps
    and pattern properties remain declarations, so their members are not
    reported one-by-one.
    """
    warnings: list[str] = []
    selected_schema = _schema_branch_for_value(schema, data)
    if not isinstance(selected_schema, Mapping):
        return warnings
    schema = selected_schema
    schema_props = _schema_mapping(schema.get("properties", {}))
    has_additional = schema.get("additionalProperties", True)
    dynamic_container = bool(schema.get("x-polylogue-dynamic-keys"))

    for key, value in data.items():
        current_path = f"{path}.{key}" if path else key

        if key not in schema_props:
            property_schema = _schema_for_property(schema, key, value)
            if _has_matching_pattern_property(schema, key):
                warnings.extend(_detect_nested_drift(value, property_schema, current_path))
                continue
            if has_additional is False:
                warnings.append(f"Unexpected field: {current_path}")
            elif has_additional is True:
                if not dynamic_container:
                    warnings.append(f"Unexpected field: {current_path}")
            else:
                additional_schema = _schema_mapping(has_additional)
                if dynamic_container:
                    continue
                if not looks_dynamic_key(key):
                    warnings.append(f"Unexpected field: {current_path}")
                warnings.extend(_detect_nested_drift(value, additional_schema, current_path))
            continue

        warnings.extend(_detect_nested_drift(value, schema_props.get(key), current_path))

    return warnings


def _detect_nested_drift(value: object, schema: object, path: str) -> list[str]:
    """Walk an object or array using the schema branch selected by ``value``."""
    selected_schema = _schema_branch_for_value(schema, value)
    if not isinstance(selected_schema, Mapping):
        return []
    nested_value = _sample_payload(value)
    if nested_value is not None:
        return detect_drift(nested_value, selected_schema, path)
    if not isinstance(value, list):
        return []
    warnings: list[str] = []
    for index, item in enumerate(value):
        warnings.extend(_detect_nested_drift(item, _schema_for_items(selected_schema, item), f"{path}[{index}]"))
    return warnings


def looks_dynamic_key(key: str) -> bool:
    """Apply the inference contract for non-structural property names."""
    return is_dynamic_key(key)


class SchemaValidator:
    """Validates data against JSON schemas with drift detection."""

    _cache: dict[tuple[str, str, str, bool], SchemaValidator] = {}

    def __init__(self, schema: ValidationSchema, strict: bool = True, provider: Provider | None = None):
        if jsonschema is None:
            raise ImportError("jsonschema not installed. Run: pip install jsonschema")

        self.schema = dict(schema)
        self.strict = strict
        self.provider = provider
        self._validator = Draft202012Validator(self.schema)

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
        schema_resolution_is_explicit: bool = True,
        strict: bool = True,
    ) -> SchemaValidator:
        """Select the validator that accepts a payload."""
        return cls.validate_payload(
            provider,
            payload,
            source_path=source_path,
            schema_resolution=schema_resolution,
            schema_resolution_is_explicit=schema_resolution_is_explicit,
            strict=strict,
        ).validator

    @classmethod
    def validate_payload(
        cls,
        provider: str | Provider,
        payload: object,
        *,
        source_path: str | None = None,
        schema_resolution: SchemaResolution | None = None,
        schema_resolution_is_explicit: bool = True,
        strict: bool = True,
        max_samples: int | None = None,
    ) -> PayloadValidation:
        """Select a schema and validate each payload sample for one operation."""
        selected: PayloadValidation | None = None
        initial_probe: tuple[SchemaValidator, tuple[ValidationSample, ...], list[ValidationResult]] | None = None

        def schema_accepts(schema: JSONDocument) -> bool:
            nonlocal selected, initial_probe
            probe = cls(schema, strict=strict, provider=_canonical_provider(provider))
            samples = tuple(probe.validation_samples(payload, max_samples=max_samples))
            results: list[ValidationResult] = []
            if initial_probe is None:
                initial_probe = probe, samples, results
            for sample in samples:
                result = probe.validate(sample, include_drift=True)
                results.append(result)
                if not result.is_valid:
                    return False
            selected = PayloadValidation(
                validator=probe,
                samples=samples,
                results=tuple(results),
                schema_resolution=schema_resolution,
                schema_resolution_is_explicit=schema_resolution_is_explicit,
            )
            return True

        canonical, schema, base_key = resolve_payload_schema(
            provider,
            payload,
            source_path=source_path,
            schema_resolution=schema_resolution,
            schema_resolution_is_explicit=schema_resolution_is_explicit,
            registry_cls=SchemaRegistry,
            schema_accepts=schema_accepts,
        )
        candidate = selected
        if candidate is None and initial_probe is not None:
            probe, initial_samples, initial_results = initial_probe
            initial_results.extend(
                probe.validate(initial_samples[index], include_drift=True)
                for index in range(len(initial_results), len(initial_samples))
            )
            candidate = PayloadValidation(
                validator=probe,
                samples=initial_samples,
                results=tuple(initial_results),
                schema_resolution=schema_resolution,
                schema_resolution_is_explicit=schema_resolution_is_explicit,
            )
        if candidate is not None:
            if schema_resolution is not None:
                return replace(
                    candidate,
                    schema_resolution=replace(
                        schema_resolution,
                        provider=str(canonical),
                        package_version=base_key[1],
                        element_kind=base_key[2],
                    ),
                )
            return candidate

        key = (*base_key, strict)
        validator = cls._cache.get(key)
        if validator is None:
            validator = cls(schema, strict=strict, provider=canonical)
            cls._cache[key] = validator
        samples = tuple(validator.validation_samples(payload, max_samples=max_samples))
        results = tuple(validator.validate(sample, include_drift=True) for sample in samples)
        resolved = (
            replace(
                schema_resolution,
                provider=str(canonical),
                package_version=base_key[1],
                element_kind=base_key[2],
            )
            if schema_resolution is not None
            else None
        )
        return PayloadValidation(
            validator=validator,
            samples=samples,
            results=results,
            schema_resolution=resolved,
            schema_resolution_is_explicit=schema_resolution_is_explicit,
        )

    @classmethod
    def available_providers(cls) -> list[str]:
        return _available_providers(registry_cls=SchemaRegistry)

    def validate(self, data: object, *, include_drift: bool | None = None) -> ValidationResult:
        normalized = _normalize_empty_arrays(data, self.schema)
        errors = [format_validation_error(error) for error in self._validator.iter_errors(normalized)]
        drift_warnings: list[str] = []
        should_detect_drift = self.strict if include_drift is None else include_drift
        sample = _sample_payload(normalized)
        if should_detect_drift and sample is not None:
            drift_warnings.extend(detect_drift(sample, self.schema, ""))
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            drift_warnings=drift_warnings,
        )

    def validation_samples(self, payload: object, *, max_samples: int | None = None) -> list[ValidationSample]:
        return collect_validation_samples(
            payload,
            schema=self.schema,
            provider=self.provider,
            max_samples=max_samples,
        )

    def _looks_dynamic_key(self, key: str) -> bool:
        return looks_dynamic_key(key)


@dataclass(frozen=True, slots=True)
class PayloadValidation:
    """Schema selection and sample verdicts for one payload operation."""

    validator: SchemaValidator
    samples: tuple[object, ...]
    results: tuple[ValidationResult, ...]
    schema_resolution: SchemaResolution | None
    schema_resolution_is_explicit: bool

    @property
    def sample_results(self) -> tuple[tuple[object, ValidationResult], ...]:
        return tuple(zip(self.samples, self.results, strict=True))


def validate_provider_export(
    data: object,
    provider: str | Provider,
    strict: bool = True,
) -> ValidationResult:
    """Convenience function to validate a provider export."""
    validator = SchemaValidator.for_provider(provider, strict=strict)
    return validator.validate(data)


__all__ = [
    "PayloadValidation",
    "SchemaValidator",
    "ValidationResult",
    "collect_validation_samples",
    "detect_drift",
    "format_validation_error",
    "looks_dynamic_key",
    "validate_provider_export",
]
