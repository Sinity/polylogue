"""JSON Schema validation for provider exports with drift detection."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Generator, Iterable, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from re import compile as compile_pattern
from typing import TYPE_CHECKING, BinaryIO, Literal, Protocol, TypeAlias

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError:
    jsonschema = None
    Draft202012Validator = None

from polylogue.archive.raw_payload import extract_payload_samples
from polylogue.core.compute_cancel import check_compute_cancelled, raise_if_operation_cancelled
from polylogue.core.enums import Provider, ValidationMode
from polylogue.core.json import JSONDocument, JSONValue, is_json_value, json_document
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate
from polylogue.schemas.field_stats.detection import is_dynamic_key
from polylogue.schemas.packages import SchemaResolution
from polylogue.schemas.runtime_registry import ProviderSchemaSnapshot, SchemaRegistry

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
    from polylogue.schemas.observation_spill import SpilledKey
    from polylogue.schemas.packages import SchemaResolution
    from polylogue.schemas.retained_validation import RetainedValidationBody, RetainedValidationVerdict

ValidationSchema: TypeAlias = Mapping[str, object]
ValidationSample: TypeAlias = JSONDocument


@dataclass(frozen=True, slots=True)
class _RetainedValidationEvidence:
    recipe: str
    snapshot: ProviderSchemaSnapshot
    path: Path
    input_identity: tuple[int, int, int, int, int]
    signature_directory: Path
    body: RetainedValidationBody
    content_digest: str
    backing_identity: tuple[Path, tuple[int, int, int, int, int]] | None


_ReuseOutcome: TypeAlias = Literal[
    "pending",
    "cleared",
    "disabled",
    "missing",
    "backing_changed",
    "recipe_changed",
    "schema_changed",
    "input_changed",
    "hit",
]


class RetainedValidationReuse:
    """One complete body witness borrowed for an owned input's lifetime.

    The capture owner keeps the input and drift-signature directory alive and
    clears this result before retiring them. The validator certifies physical
    aliases by exact bytes and projects fresh acquired-evidence coordinates;
    current schema and original-input/backing checks still precede every hit.
    """

    def __init__(self) -> None:
        self._entry: _RetainedValidationEvidence | None = None
        self._outcome: _ReuseOutcome = "cleared"

    @property
    def outcome(self) -> _ReuseOutcome:
        """The last call's actual reuse guard outcome, not parser eligibility."""
        return self._outcome

    def clear(self) -> None:
        self._entry = None
        self._outcome = "cleared"


def _validation_input_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _validation_backing_identity(
    verdict: RetainedValidationVerdict | RetainedValidationBody,
) -> tuple[Path, tuple[int, int, int, int, int]] | None:
    observation = verdict.drift_observation
    path = None if observation is None else observation.unseen_key_signature._path
    return None if path is None else (path, _validation_input_identity(path.stat()))


@contextmanager
def _owned_validation_input(
    path: Path, prefix: int | None
) -> Generator[tuple[tuple[int, int, int, int, int], BinaryIO]]:
    from polylogue.schemas.observation_spill import AcceptedPrefixReadError

    check_compute_cancelled()
    with path.open("rb", buffering=0) as source:
        before = _validation_input_identity(os.fstat(source.fileno()))
        if prefix is not None and (prefix < 0 or before[2] < prefix):
            raise AcceptedPrefixReadError("retained JSONL parser prefix exceeds source bytes")
        try:
            yield before, source
        except BaseException as error:
            raise_if_operation_cancelled(error)
            raise
        finally:
            check_compute_cancelled()
            unchanged = False
            try:
                unchanged = (
                    before
                    == _validation_input_identity(os.fstat(source.fileno()))
                    == _validation_input_identity(path.stat())
                )
            except OSError as error:
                raise AcceptedPrefixReadError("retained JSONL validation input changed") from error
            if not unchanged:
                raise AcceptedPrefixReadError("retained JSONL validation input changed")


def _validation_input_digest(source: BinaryIO, byte_length: int) -> str:
    """Certify a physical alias using its pinned exact bytes, never its name."""
    from polylogue.schemas.observation_spill import AcceptedPrefixReadError

    digest = hashlib.sha256()
    copied = 0
    try:
        source.seek(0)
        while chunk := source.read(1024 * 1024):
            check_compute_cancelled()
            copied += len(chunk)
            digest.update(chunk)
    except OSError as error:
        raise AcceptedPrefixReadError("retained validation input could not be read") from error
    if copied != byte_length:
        raise AcceptedPrefixReadError("retained validation input has a physical short read")
    check_compute_cancelled()
    return digest.hexdigest()


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


def _resolve_local_ref(schema: object, root: Mapping[str, object] | None) -> object:
    """Follow a local ``$ref`` chain so the walk sees the declared node.

    A ``{"$ref": "#/$defs/Provenance"}`` fragment is a Mapping with no
    ``properties`` and a permissive default ``additionalProperties``, so the
    drift walk treated every legitimately declared nested field as
    ``Unexpected`` and ingest recorded spurious ``new_field`` observations for
    ordinary browser-capture, Codex and ``allOf``-wrapped payloads. An external
    or unresolvable ref yields ``None``: refusing to observe a node whose
    declarations are not in hand is correct, reporting all of it as drift is
    not.
    """
    seen: set[str] = set()
    siblings: list[Mapping[str, object]] = []
    while isinstance(schema, Mapping) and isinstance(schema.get("$ref"), str):
        pointer = str(schema["$ref"])
        if root is None or not pointer.startswith("#") or pointer in seen:
            return None
        seen.add(pointer)
        sibling = {key: value for key, value in schema.items() if key != "$ref"}
        if sibling:
            siblings.append(sibling)
        target: object = root
        for part in pointer.lstrip("#/").split("/"):
            if not part:
                continue
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(target, Mapping) and part in target:
                target = target[part]
            else:
                return None
        schema = target
    if not siblings:
        return schema
    # Draft 2020-12 evaluates ``$ref`` siblings beside the target, so their
    # declarations join the observation view instead of being discarded.
    # The result stays one node, so item, type and union readers see the
    # target's own keywords.
    resolved: dict[str, object] = {}
    if isinstance(schema, Mapping):
        resolved.update(schema)
    for declarations in siblings:
        for key, value in declarations.items():
            current = resolved.get(key)
            if (
                key in {"properties", "patternProperties"}
                and isinstance(current, Mapping)
                and isinstance(value, Mapping)
            ):
                combined = dict(current)
                for name, declaration in value.items():
                    existing = combined.get(name)
                    combined[name] = (
                        _merge_pattern_observation_schemas([existing, declaration])
                        if isinstance(existing, Mapping) and isinstance(declaration, Mapping)
                        else declaration
                    )
                resolved[key] = combined
            elif key == "x-polylogue-dynamic-keys":
                resolved[key] = current is True or value is True
            else:
                resolved[key] = value
    return resolved


def _schema_branch_for_value(
    schema: object,
    value: object,
    root: Mapping[str, object] | None = None,
    *,
    connection: sqlite3.Connection | None = None,
) -> object:
    schema = _resolve_local_ref(schema, root)
    if not isinstance(schema, Mapping):
        return None
    all_of = schema.get("allOf")
    if isinstance(all_of, list) and isinstance(value, Mapping):
        # ``allOf`` contributes every branch's declarations. Treating the
        # wrapper as one object loses those properties and reports valid,
        # branch-declared names as drift. Validation remains delegated to
        # Draft 2020-12; this merged view is only for observation.
        branches: list[Mapping[str, object]] = []
        base = {key: item for key, item in schema.items() if key != "allOf"}
        if base:
            selected_base = _schema_branch_for_value(base, value, root, connection=connection)
            if isinstance(selected_base, Mapping):
                branches.append(selected_base)
        for branch in all_of:
            selected = _schema_branch_for_value(branch, value, root, connection=connection)
            if isinstance(selected, Mapping):
                branches.append(selected)
        if branches:
            merged = _merge_pattern_observation_schemas(branches)
            # An annotation absent from one conjunct does not negate an
            # explicit dynamic-key declaration on another.
            if any(branch.get("x-polylogue-dynamic-keys") is True for branch in branches):
                return {**merged, "x-polylogue-dynamic-keys": True}
            return merged
    for key in ("anyOf", "oneOf"):
        union_branches = schema.get(key)
        if not isinstance(union_branches, list):
            continue
        # A union can contain several object (or array) branches.  Pick a
        # branch which accepts the concrete value before falling back to its
        # broad JSON type, otherwise a drift walk can attribute a field to an
        # unrelated sibling branch.
        for branch in union_branches:
            resolved = _resolve_local_ref(branch, root)
            if isinstance(resolved, Mapping) and _schema_accepts_value(resolved, value, connection=connection):
                return resolved
        for branch in union_branches:
            resolved = _resolve_local_ref(branch, root)
            if not isinstance(resolved, Mapping):
                continue
            if isinstance(value, Mapping) and _schema_allows_type(resolved, "object"):
                return resolved
            if isinstance(value, list) and _schema_allows_type(resolved, "array"):
                return resolved
            if value is None and _schema_allows_type(resolved, "null"):
                return resolved
        for branch in union_branches:
            resolved = _resolve_local_ref(branch, root)
            if isinstance(resolved, Mapping):
                return resolved
    return schema


def _schema_accepts_value(
    schema: Mapping[str, object], value: object, *, connection: sqlite3.Connection | None = None
) -> bool:
    """Return whether an individual union branch accepts ``value``.

    This is deliberately a branch-selection aid only.  The complete schema
    validator remains the authority for acceptance and error reporting.
    """
    if Draft202012Validator is None:
        return False
    try:
        if connection is not None:
            from polylogue.schemas.retained_validation import _bounded_validator

            return bool(_bounded_validator(schema, connection).is_valid(value))
        return bool(Draft202012Validator(schema).is_valid(value))
    except Exception:
        # A referenced or otherwise incomplete branch still has useful type
        # information below; do not let diagnostics mask validation itself.
        return False


def _schema_for_property(
    schema: object,
    key: str,
    value: object,
    root: Mapping[str, object] | None = None,
    *,
    connection: sqlite3.Connection | None = None,
) -> object:
    if not isinstance(schema, Mapping):
        return None
    properties = schema.get("properties")
    if isinstance(properties, Mapping) and key in properties:
        return _schema_branch_for_value(properties[key], value, root, connection=connection)
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
                    selected = _schema_branch_for_value(pattern_schema, value, root, connection=connection)
                    if isinstance(selected, Mapping):
                        matches.append(selected)
        if matches:
            return _merge_pattern_observation_schemas(matches)
    additional_properties = schema.get("additionalProperties")
    if isinstance(additional_properties, Mapping):
        return _schema_branch_for_value(additional_properties, value, root, connection=connection)
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


def _schema_for_items(schema: object, value: object, root: Mapping[str, object] | None = None) -> object:
    schema = _resolve_local_ref(schema, root)
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
    return json_document(dict(value.items()))


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
    root: Mapping[str, object] | None = None,
) -> list[str]:
    """Detect newly observed named fields without changing schema acceptance.

    JSON Schema's default ``additionalProperties`` is permissive.  That is an
    admission rule, not evidence that a provider named field is already
    known: newly named fields are still reported.  Explicit dynamic-key maps
    and pattern properties remain declarations, so their members are not
    reported one-by-one.
    """
    return [f"Unexpected field: {field_path}" for field_path in _iter_drift_paths(data, schema, path, root)]


@dataclass(frozen=True)
class _DriftPath:
    """A diagnostic path borrowing exact keys from the completed JSON owner."""

    parts: tuple[str | SpilledKey, ...] = ()

    def iter_utf8_chunks(self) -> Generator[bytes, None, None]:
        from polylogue.schemas.observation_spill import SpilledKey

        for part in self.parts:
            if isinstance(part, SpilledKey):
                with closing(part.iter_utf8_chunks()) as chunks:
                    yield from chunks
            else:
                for offset in range(0, len(part), 1024):
                    yield part[offset : offset + 1024].encode("utf-8", "surrogatepass")

    def text(self) -> str:
        return "".join(chunk.decode("utf-8", "surrogatepass") for chunk in self.iter_utf8_chunks())

    def has_text(self) -> bool:
        from polylogue.schemas.observation_spill import SpilledKey

        return any(part.small_name != "" if isinstance(part, SpilledKey) else bool(part) for part in self.parts)

    def field(self, key: str | SpilledKey) -> _DriftPath:
        return _DriftPath(self.parts + ((".",) if self.has_text() else ()) + (key,))

    def item(self, index: int) -> _DriftPath:
        return _DriftPath(self.parts + (f"[{index}]",))


def _drift_keys(data: Mapping[str, object]) -> Generator[str | SpilledKey, None, None]:
    from polylogue.schemas.observation_spill import SpilledObject

    if isinstance(data, SpilledObject):
        with closing(data.key_entries()) as keys:
            for key, _child in keys:
                yield key
    else:
        yield from data


def _drift_value(data: Mapping[str, object], key: str | SpilledKey) -> object:
    from polylogue.schemas.observation_spill import SpilledKey, SpilledObject

    if isinstance(key, SpilledKey) and isinstance(data, SpilledObject):
        return data.value_for_key(key)
    return data[key.read() if isinstance(key, SpilledKey) else key]


def _iter_drift_paths(
    data: Mapping[str, object],
    schema: Mapping[str, object],
    path: str,
    root: Mapping[str, object] | None = None,
    connection: sqlite3.Connection | None = None,
) -> Iterable[str]:
    """Materialize complete paths only for an explicitly selected warning caller."""
    with closing(_iter_drift_key_paths(data, schema, _DriftPath((path,)), root, connection)) as paths:
        for key_path in paths:
            yield key_path.text()


def _iter_drift_key_paths(
    data: Mapping[str, object],
    schema: Mapping[str, object],
    path: _DriftPath,
    root: Mapping[str, object] | None = None,
    connection: sqlite3.Connection | None = None,
) -> Generator[_DriftPath, None, None]:
    """Retain exact unexpected names without retrieving unknown values or names."""
    from polylogue.schemas.observation_spill import SpilledKey

    root = root if root is not None else schema
    selected_schema = _schema_branch_for_value(schema, data, root, connection=connection)
    if not isinstance(selected_schema, Mapping):
        return
    properties_value = selected_schema.get("properties", {})
    properties = properties_value if isinstance(properties_value, Mapping) else {}
    has_additional = selected_schema.get("additionalProperties", True)
    dynamic_container = bool(selected_schema.get("x-polylogue-dynamic-keys"))
    patterns = selected_schema.get("patternProperties")
    with closing(_drift_keys(data)) as keys:
        for key in keys:
            check_compute_cancelled()
            name = key.small_name if isinstance(key, SpilledKey) else key
            if name is None:
                assert isinstance(key, SpilledKey)
                # Declared literal names can be compared exactly without reading
                # an unknown key. Generic regex evaluation remains a selected
                # full-name demand when the schema actually declares patterns.
                name = next((declared for declared in properties if key.matches(declared)), None)
                if name is None and isinstance(patterns, Mapping) and patterns:
                    name = key.read()
            current_path = path.field(key)
            if name not in properties:
                if name is not None and _has_matching_pattern_property(selected_schema, name):
                    value = _drift_value(data, key)
                    property_schema = _schema_for_property(selected_schema, name, value, root, connection=connection)
                    yield from _iter_nested_drift_key_paths(value, property_schema, current_path, root, connection)
                elif has_additional is False:
                    yield current_path
                elif has_additional is True:
                    if not dynamic_container:
                        yield current_path
                else:
                    if not dynamic_container and name is not None and not looks_dynamic_key(name):
                        yield current_path
                    yield from _iter_nested_drift_key_paths(
                        _drift_value(data, key), has_additional, current_path, root, connection
                    )
                continue
            yield from _iter_nested_drift_key_paths(
                _drift_value(data, key), properties.get(name), current_path, root, connection
            )


def _iter_nested_drift_key_paths(
    value: object,
    schema: object,
    path: _DriftPath,
    root: Mapping[str, object],
    connection: sqlite3.Connection | None,
) -> Generator[_DriftPath, None, None]:
    selected_schema = _schema_branch_for_value(schema, value, root, connection=connection)
    if not isinstance(selected_schema, Mapping):
        return
    if isinstance(value, Mapping):
        yield from _iter_drift_key_paths(value, selected_schema, path, root, connection)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_compute_cancelled()
            yield from _iter_nested_drift_key_paths(
                item, _schema_for_items(selected_schema, item, root), path.item(index), root, connection
            )


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
        from polylogue.schemas.retained_validation import _bounded_validator, _ensure_reducer_tables, _normalized
        from polylogue.storage.sqlite.connection_profile import scratch_connection_context

        with scratch_connection_context(prefix="polylogue-schema-validation-", filename="validation.sqlite") as conn:
            _ensure_reducer_tables(conn)
            normalized = _normalized(data, self.schema, self.schema, conn)
            validator = _bounded_validator(self.schema, conn)
            errors = [format_validation_error(error) for error in validator.iter_errors(normalized)]
            should_detect_drift = self.strict if include_drift is None else include_drift
            sample = _sample_payload(normalized)
            drift_warnings = (
                [f"Unexpected field: {path}" for path in _iter_drift_paths(sample, self.schema, "", self.schema, conn)]
                if should_detect_drift and sample is not None
                else []
            )
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


def validate_retained_document(
    provider: str | Provider,
    path: Path,
    *,
    mode: ValidationMode,
    raw_id: str,
    revision_sha256: str,
    evidence_id: str,
    source_path: str | None = None,
    jsonl: bool = False,
    accepted_prefix_size: int | None = None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = None,
    schema_resolution: SchemaResolution | None = None,
    schema_resolution_is_explicit: bool = False,
    registry: SchemaRegistry | None = None,
    signature_directory: Path,
    reuse: RetainedValidationReuse | None = None,
) -> RetainedValidationVerdict:
    """Validate a retained source revision with a compact spill-backed verdict.

    The retained ingest route uses this entrypoint while the raw revision and
    its original evidence remain owned.  Importing lazily keeps the ordinary
    in-memory validation surface independent of the retained storage adapter.
    """
    from contextlib import AbstractContextManager, nullcontext

    from polylogue.schemas.retained_validation import validate_retained_document as validate

    if reuse is not None:
        reuse._outcome = "pending"
    active_registry = registry
    snapshot: AbstractContextManager[ProviderSchemaSnapshot | None] = nullcontext()
    if ValidationMode.from_string(mode) is not ValidationMode.OFF:
        active_registry = registry or SchemaRegistry()
        snapshot = active_registry.current_provider_snapshot(provider)

    def perform() -> RetainedValidationVerdict:
        return validate(
            provider,
            path,
            mode=mode,
            raw_id=raw_id,
            revision_sha256=revision_sha256,
            evidence_id=evidence_id,
            source_path=source_path,
            jsonl=jsonl,
            accepted_prefix_size=accepted_prefix_size,
            captured_zip_coordinate=captured_zip_coordinate,
            schema_resolution=schema_resolution,
            schema_resolution_is_explicit=schema_resolution_is_explicit,
            registry=active_registry,
            signature_directory=signature_directory,
        )

    with snapshot as snapshot_identity:
        if snapshot_identity is not None:
            active_registry = snapshot_identity.reader()
        if reuse is None or ValidationMode.from_string(mode) is ValidationMode.OFF:
            if reuse is not None:
                reuse._outcome = "disabled"
            from polylogue.schemas.retained_validation import RetainedValidationBody

            return RetainedValidationBody.from_verdict(perform()).bind(
                raw_id=raw_id, revision_sha256=revision_sha256, evidence_id=evidence_id, source_path=source_path
            )
        from polylogue.schemas.retained_validation import _retained_validation_productive_identity

        recipe = _retained_validation_productive_identity(
            provider,
            path,
            mode=mode,
            raw_id="",
            revision_sha256="",
            evidence_id="",
            source_path=source_path,
            jsonl=jsonl,
            accepted_prefix_size=accepted_prefix_size,
            captured_zip_coordinate=captured_zip_coordinate,
            schema_resolution=schema_resolution,
            schema_resolution_is_explicit=schema_resolution_is_explicit,
            registry=active_registry,
            signature_directory=signature_directory,
        )
        from polylogue.schemas.retained_validation import RetainedValidationBody

        with _owned_validation_input(path, accepted_prefix_size) as (input_identity, source):
            entry = reuse._entry
            try:
                backing_current = entry is not None and entry.backing_identity == _validation_backing_identity(
                    entry.body
                )
            except OSError:
                backing_current = False
            try:
                original_current = (
                    entry is not None
                    and entry.signature_directory.is_dir()
                    and entry.input_identity == _validation_input_identity(entry.path.stat())
                )
            except OSError:
                original_current = False
            content_digest: str | None = None
            hit = False
            body: RetainedValidationBody | None = None
            outcome: _ReuseOutcome
            if entry is None:
                outcome = "missing"
            elif not backing_current:
                outcome = "backing_changed"
            elif entry.recipe != recipe:
                outcome = "recipe_changed"
            elif entry.snapshot != snapshot_identity:
                outcome = "schema_changed"
            elif (
                not original_current
                or (entry.path == path and entry.input_identity != input_identity)
                or (
                    entry.path != path
                    and (
                        input_identity[2] != entry.input_identity[2]
                        or (content_digest := _validation_input_digest(source, input_identity[2]))
                        != entry.content_digest
                    )
                )
            ):
                outcome = "input_changed"
            else:
                hit = True
                body = entry.body
            if not hit:
                reuse.clear()
                reuse._outcome = outcome
                body = RetainedValidationBody.from_verdict(perform())
                if content_digest is None:
                    content_digest = _validation_input_digest(source, input_identity[2])
        assert body is not None
        if hit:
            assert entry is not None
            from polylogue.schemas.observation_spill import AcceptedPrefixReadError

            try:
                current = (
                    reuse._entry is entry
                    and entry.signature_directory.is_dir()
                    and entry.input_identity == _validation_input_identity(entry.path.stat())
                    and entry.backing_identity == _validation_backing_identity(entry.body)
                )
            except OSError:
                current = False
            if not current:
                raise AcceptedPrefixReadError("retained validation evidence retired during binding")
            reuse._outcome = "hit"
            return body.bind(
                raw_id=raw_id, revision_sha256=revision_sha256, evidence_id=evidence_id, source_path=source_path
            )
        assert snapshot_identity is not None and content_digest is not None
        reuse._entry = _RetainedValidationEvidence(
            recipe,
            snapshot_identity,
            path,
            input_identity,
            signature_directory,
            body,
            content_digest,
            _validation_backing_identity(body),
        )
        return body.bind(
            raw_id=raw_id, revision_sha256=revision_sha256, evidence_id=evidence_id, source_path=source_path
        )


__all__ = [
    "PayloadValidation",
    "RetainedValidationVerdict",
    "RetainedValidationReuse",
    "SchemaValidator",
    "ValidationResult",
    "collect_validation_samples",
    "detect_drift",
    "format_validation_error",
    "looks_dynamic_key",
    "validate_retained_document",
    "validate_provider_export",
]


def __getattr__(name: str) -> object:
    if name == "RetainedValidationVerdict":
        from polylogue.schemas.retained_validation import RetainedValidationVerdict

        return RetainedValidationVerdict
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
