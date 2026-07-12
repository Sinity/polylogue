"""Annotation schema registry: declared shape for external-agent labeling.

An :class:`AnnotationSchema` turns a raw JSON blob into an analytical
variable: it declares the value shape (typed fields, not an open dict), the
:class:`~polylogue.core.refs.ObjectRefKind` grains it is allowed to target, an
abstention convention, and whether evidence refs are required before a row is
accepted as a candidate. Without this, imported labels are queryable blobs
with no construct-validity metadata (see polylogue-rxdo.7 and the related
measure-registry discipline, polylogue-9l5.7).

This module owns *declaration and validation only* -- it has no storage
dependency. :mod:`polylogue.annotations.write` is the thin write-path adapter
that validates one row against a schema and then upserts it through the
existing single assertion-write chokepoint
(:func:`polylogue.storage.sqlite.archive_tiers.user_write.upsert_assertion`).

The registry includes the production delegation-discourse v1 definition and
each schema can be lowered to one canonical definition JSON/fingerprint for
durable storage. Batch provenance is declared in
:mod:`polylogue.annotations.batch`; JSONL/CLI/MCP import remains a leaf
surface owned by polylogue-rxdo.7.2.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal, cast, get_args

from polylogue.core.json import loads as json_loads
from polylogue.core.refs import ObjectRef, ObjectRefKind

AnnotationFieldType = Literal["string", "integer", "number", "boolean", "enum"]
AnnotationEvidencePolicy = Literal["none", "optional", "required"]
AnnotationSchemaStatus = Literal["draft", "active", "deprecated"]
ANNOTATION_SCHEMA_DEFINITION_FORMAT = "polylogue.annotation-schema/v1"

_OBJECT_REF_KINDS: frozenset[str] = frozenset(get_args(ObjectRefKind))
_ANNOTATION_FIELD_TYPES: frozenset[str] = frozenset(get_args(AnnotationFieldType))
_ANNOTATION_EVIDENCE_POLICIES: frozenset[str] = frozenset(get_args(AnnotationEvidencePolicy))
_ANNOTATION_SCHEMA_STATUSES: frozenset[str] = frozenset(get_args(AnnotationSchemaStatus))
_SCHEMA_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class AnnotationSchemaError(ValueError):
    """Raised for a malformed schema declaration (registration-time, not row-time)."""


def _require_exact_keys(value: Mapping[str, object], expected: frozenset[str], *, context: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AnnotationSchemaError(f"{context} keys differ from canonical shape: missing={missing}, extra={extra}")


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise AnnotationSchemaError(f"{context} must be a string")
    return value


def _nfc_string(value: object, *, context: str) -> str:
    """Return one detached NFC-normalized declaration string."""

    return unicodedata.normalize("NFC", _require_string(value, context=context))


def _nfc_string_tuple(value: object, *, context: str) -> tuple[str, ...]:
    """Snapshot and normalize a declaration sequence, rejecting collisions."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AnnotationSchemaError(f"{context} must be a sequence of strings")
    raw_values = tuple(value)
    normalized = tuple(_nfc_string(item, context=f"{context} item") for item in raw_values)
    if len(normalized) != len(set(normalized)):
        raise AnnotationSchemaError(f"{context} contains duplicate values after NFC normalization")
    return normalized


def _require_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise AnnotationSchemaError(f"{context} must be a boolean")
    return value


def _require_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnnotationSchemaError(f"{context} must be an integer")
    return value


def _require_optional_number(value: object, *, context: str) -> int | float | None:
    if value is None:
        return None
    if not _is_finite_json_number(value):
        raise AnnotationSchemaError(f"{context} must be a finite number or null")
    return cast(int | float, value)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _nfc_json_value(value: object) -> object:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_nfc_json_value(item) for item in value]
    if isinstance(value, dict):
        return {unicodedata.normalize("NFC", str(key)): _nfc_json_value(item) for key, item in value.items()}
    return value


def _is_finite_json_number(value: object) -> bool:
    """Return whether *value* is a finite JSON number without coercing huge ints."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, int):
        return True
    return isfinite(value)


@dataclass(frozen=True, slots=True)
class AnnotationField:
    """One typed field in an annotation schema's value shape."""

    name: str
    value_type: AnnotationFieldType
    required: bool = True
    description: str = ""
    enum_values: tuple[str, ...] = ()
    minimum: int | float | None = None
    maximum: int | float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nfc_string(self.name, context="annotation field name"))
        object.__setattr__(
            self,
            "value_type",
            cast(
                AnnotationFieldType,
                _nfc_string(self.value_type, context=f"annotation field {self.name!r} value_type"),
            ),
        )
        object.__setattr__(
            self,
            "description",
            _nfc_string(self.description, context=f"annotation field {self.name!r} description"),
        )
        object.__setattr__(
            self,
            "enum_values",
            _nfc_string_tuple(self.enum_values, context=f"annotation field {self.name!r} enum_values"),
        )
        object.__setattr__(
            self,
            "required",
            _require_bool(self.required, context=f"annotation field {self.name!r} required"),
        )
        if self.value_type not in _ANNOTATION_FIELD_TYPES:
            raise AnnotationSchemaError(
                f"annotation field {self.name!r} declares unknown value_type {self.value_type!r}"
            )
        if not _FIELD_NAME_RE.fullmatch(self.name):
            raise AnnotationSchemaError(f"annotation field name {self.name!r} must match {_FIELD_NAME_RE.pattern!r}")
        if self.value_type == "enum" and not self.enum_values:
            raise AnnotationSchemaError(f"annotation field {self.name!r} is type 'enum' but declares no enum_values")
        if self.value_type != "enum" and self.enum_values:
            raise AnnotationSchemaError(f"annotation field {self.name!r} declares enum_values but is not type 'enum'")
        if self.value_type not in {"integer", "number"} and (self.minimum is not None or self.maximum is not None):
            raise AnnotationSchemaError(
                f"annotation field {self.name!r} declares minimum/maximum but is not a numeric type"
            )
        for bound_name, bound in (("minimum", self.minimum), ("maximum", self.maximum)):
            if bound is None:
                continue
            if not _is_finite_json_number(bound):
                raise AnnotationSchemaError(
                    f"annotation field {self.name!r} {bound_name} must be a finite number, got {bound!r}"
                )
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise AnnotationSchemaError(f"annotation field {self.name!r} has minimum > maximum")

    def validate_value(self, value: object) -> str | None:
        """Return an error message for *value*, or ``None`` if it satisfies this field."""

        if self.value_type == "string":
            if not isinstance(value, str):
                return f"field {self.name!r} must be a string, got {type(value).__name__}"
            return None
        if self.value_type == "boolean":
            if not isinstance(value, bool):
                return f"field {self.name!r} must be a boolean, got {type(value).__name__}"
            return None
        if self.value_type == "enum":
            if not isinstance(value, str) or value not in self.enum_values:
                return f"field {self.name!r} must be one of {sorted(self.enum_values)}, got {value!r}"
            return None
        # integer / number: bool is a JSON-illegal surprise (bool is an int subclass in
        # Python) -- reject it explicitly so a stray `true` never silently reads as 1.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"field {self.name!r} must be a {self.value_type}, got {type(value).__name__}"
        if self.value_type == "integer" and not isinstance(value, int):
            return f"field {self.name!r} must be an integer, got {type(value).__name__}"
        if not _is_finite_json_number(value):
            return f"field {self.name!r} must be a finite JSON number, got {value!r}"
        if self.minimum is not None and value < self.minimum:
            return f"field {self.name!r} value {value} is below minimum {self.minimum}"
        if self.maximum is not None and value > self.maximum:
            return f"field {self.name!r} value {value} is above maximum {self.maximum}"
        return None

    def definition_document(self) -> dict[str, object]:
        """Return the complete JSON definition used by schema fingerprints."""

        return {
            "description": self.description,
            "enum_values": list(self.enum_values),
            "maximum": self.maximum,
            "minimum": self.minimum,
            "name": self.name,
            "required": self.required,
            "value_type": self.value_type,
        }

    @classmethod
    def from_definition_document(cls, value: object) -> AnnotationField:
        """Decode one canonical field definition without coercing malformed types."""

        if not isinstance(value, Mapping):
            raise AnnotationSchemaError("annotation field definition must be a JSON object")
        document = {str(key): item for key, item in value.items()}
        _require_exact_keys(
            document,
            frozenset({"description", "enum_values", "maximum", "minimum", "name", "required", "value_type"}),
            context="annotation field definition",
        )
        raw_enum_values = document["enum_values"]
        if not isinstance(raw_enum_values, list) or not all(isinstance(item, str) for item in raw_enum_values):
            raise AnnotationSchemaError("annotation field enum_values must be an array of strings")
        value_type = _require_string(document["value_type"], context="annotation field value_type")
        return cls(
            name=_require_string(document["name"], context="annotation field name"),
            value_type=cast(AnnotationFieldType, value_type),
            required=_require_bool(document["required"], context="annotation field required"),
            description=_require_string(document["description"], context="annotation field description"),
            enum_values=tuple(raw_enum_values),
            minimum=_require_optional_number(document["minimum"], context="annotation field minimum"),
            maximum=_require_optional_number(document["maximum"], context="annotation field maximum"),
        )


@dataclass(frozen=True, slots=True)
class AnnotationSchema:
    """A versioned, declared shape for one family of imported labels.

    ``abstain_field`` names a boolean field (declared in ``fields``,
    ``required=False``) that, when set ``true`` on a row, exempts every other
    *required* field from presence-checking -- the labeler is recording
    "I cannot label this" rather than a partial label. Non-abstain-field
    values present on an abstained row are still type-checked.
    """

    schema_id: str
    version: int
    title: str
    fields: tuple[AnnotationField, ...]
    target_ref_kinds: tuple[ObjectRefKind, ...]
    description: str = ""
    abstain_field: str | None = None
    evidence_policy: AnnotationEvidencePolicy = "required"
    status: AnnotationSchemaStatus = "draft"

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_id", _nfc_string(self.schema_id, context="annotation schema schema_id"))
        object.__setattr__(
            self,
            "title",
            _nfc_string(self.title, context=f"annotation schema {self.schema_id!r} title"),
        )
        object.__setattr__(
            self,
            "description",
            _nfc_string(self.description, context=f"annotation schema {self.schema_id!r} description"),
        )
        if self.abstain_field is not None:
            object.__setattr__(
                self,
                "abstain_field",
                _nfc_string(
                    self.abstain_field,
                    context=f"annotation schema {self.schema_id!r} abstain_field",
                ),
            )
        object.__setattr__(
            self,
            "evidence_policy",
            cast(
                AnnotationEvidencePolicy,
                _nfc_string(
                    self.evidence_policy,
                    context=f"annotation schema {self.schema_id!r} evidence_policy",
                ),
            ),
        )
        object.__setattr__(
            self,
            "status",
            cast(
                AnnotationSchemaStatus,
                _nfc_string(self.status, context=f"annotation schema {self.schema_id!r} status"),
            ),
        )
        raw_fields = cast(object, self.fields)
        if isinstance(raw_fields, (str, bytes)) or not isinstance(raw_fields, Sequence):
            raise AnnotationSchemaError(f"schema {self.schema_id!r} fields must be a sequence")
        fields = tuple(raw_fields)
        if not all(isinstance(entry, AnnotationField) for entry in fields):
            raise AnnotationSchemaError(f"schema {self.schema_id!r} fields must contain AnnotationField values")
        object.__setattr__(self, "fields", cast(tuple[AnnotationField, ...], fields))
        object.__setattr__(
            self,
            "target_ref_kinds",
            cast(
                tuple[ObjectRefKind, ...],
                _nfc_string_tuple(
                    self.target_ref_kinds,
                    context=f"schema {self.schema_id!r} target_ref_kinds",
                ),
            ),
        )
        if not _SCHEMA_ID_RE.fullmatch(self.schema_id):
            raise AnnotationSchemaError(f"schema_id {self.schema_id!r} must match {_SCHEMA_ID_RE.pattern!r}")
        if not _is_positive_int(self.version):
            raise AnnotationSchemaError(f"schema {self.schema_id!r} version must be >= 1, got {self.version}")
        if self.evidence_policy not in _ANNOTATION_EVIDENCE_POLICIES:
            raise AnnotationSchemaError(
                f"schema {self.schema_id!r} declares unknown evidence_policy {self.evidence_policy!r}"
            )
        if self.status not in _ANNOTATION_SCHEMA_STATUSES:
            raise AnnotationSchemaError(f"schema {self.schema_id!r} declares unknown status {self.status!r}")
        if not self.fields:
            raise AnnotationSchemaError(f"schema {self.schema_id!r} declares no fields")
        field_names = [entry.name for entry in self.fields]
        if len(field_names) != len(set(field_names)):
            raise AnnotationSchemaError(f"schema {self.schema_id!r} declares duplicate field names: {field_names}")
        if not self.target_ref_kinds:
            raise AnnotationSchemaError(f"schema {self.schema_id!r} declares no target_ref_kinds")
        unknown_kinds = [kind for kind in self.target_ref_kinds if kind not in _OBJECT_REF_KINDS]
        if unknown_kinds:
            raise AnnotationSchemaError(f"schema {self.schema_id!r} declares unknown target_ref_kinds: {unknown_kinds}")
        if self.abstain_field is not None:
            abstain_field_def = self._field_by_name(self.abstain_field)
            if abstain_field_def is None:
                raise AnnotationSchemaError(
                    f"schema {self.schema_id!r} abstain_field {self.abstain_field!r} is not a declared field"
                )
            if abstain_field_def.value_type != "boolean":
                raise AnnotationSchemaError(
                    f"schema {self.schema_id!r} abstain_field {self.abstain_field!r} must be type 'boolean'"
                )
            if abstain_field_def.required:
                raise AnnotationSchemaError(
                    f"schema {self.schema_id!r} abstain_field {self.abstain_field!r} must not be required "
                    "(a row that omits it is a non-abstaining label)"
                )

    @property
    def qualified_id(self) -> str:
        """Return the ``schema_id@vN`` identity used for provenance stamps."""

        return f"{self.schema_id}@v{self.version}"

    def _field_by_name(self, name: str) -> AnnotationField | None:
        for entry in self.fields:
            if entry.name == name:
                return entry
        return None

    def accepts_target_kind(self, target_ref: str) -> bool:
        """Return whether *target_ref* resolves to a kind this schema can label."""

        try:
            parsed = ObjectRef.parse(target_ref)
        except ValueError:
            return False
        return parsed.kind in self.target_ref_kinds

    def definition_document(self) -> dict[str, object]:
        """Return the complete construct definition in stable JSON-compatible form."""

        return {
            "abstain_field": self.abstain_field,
            "description": self.description,
            "evidence_policy": self.evidence_policy,
            "fields": [entry.definition_document() for entry in self.fields],
            "format": ANNOTATION_SCHEMA_DEFINITION_FORMAT,
            "schema_id": self.schema_id,
            "status": self.status,
            "target_ref_kinds": list(self.target_ref_kinds),
            "title": self.title,
            "version": self.version,
        }

    def canonical_definition_json(self) -> str:
        """Return the one byte-stable JSON representation used for durable identity."""

        return json.dumps(
            _nfc_json_value(self.definition_document()),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def definition_fingerprint(self) -> str:
        """Return SHA-256 over :meth:`canonical_definition_json`."""

        return hashlib.sha256(self.canonical_definition_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_definition_document(cls, value: object) -> AnnotationSchema:
        """Decode a complete persisted construct definition, rejecting shape drift."""

        if not isinstance(value, Mapping):
            raise AnnotationSchemaError("annotation schema definition must be a JSON object")
        document = {str(key): item for key, item in value.items()}
        _require_exact_keys(
            document,
            frozenset(
                {
                    "abstain_field",
                    "description",
                    "evidence_policy",
                    "fields",
                    "format",
                    "schema_id",
                    "status",
                    "target_ref_kinds",
                    "title",
                    "version",
                }
            ),
            context="annotation schema definition",
        )
        raw_fields = document["fields"]
        if not isinstance(raw_fields, list):
            raise AnnotationSchemaError("annotation schema fields must be an array")
        raw_target_kinds = document["target_ref_kinds"]
        if not isinstance(raw_target_kinds, list) or not all(isinstance(item, str) for item in raw_target_kinds):
            raise AnnotationSchemaError("annotation schema target_ref_kinds must be an array of strings")
        abstain_field = document["abstain_field"]
        if abstain_field is not None and not isinstance(abstain_field, str):
            raise AnnotationSchemaError("annotation schema abstain_field must be a string or null")
        if document["format"] != ANNOTATION_SCHEMA_DEFINITION_FORMAT:
            raise AnnotationSchemaError(
                f"annotation schema definition format must be {ANNOTATION_SCHEMA_DEFINITION_FORMAT!r}"
            )
        return cls(
            schema_id=_require_string(document["schema_id"], context="annotation schema schema_id"),
            version=_require_int(document["version"], context="annotation schema version"),
            title=_require_string(document["title"], context="annotation schema title"),
            fields=tuple(AnnotationField.from_definition_document(entry) for entry in raw_fields),
            target_ref_kinds=cast(tuple[ObjectRefKind, ...], tuple(raw_target_kinds)),
            description=_require_string(document["description"], context="annotation schema description"),
            abstain_field=abstain_field,
            evidence_policy=cast(
                AnnotationEvidencePolicy,
                _require_string(document["evidence_policy"], context="annotation schema evidence_policy"),
            ),
            status=cast(
                AnnotationSchemaStatus,
                _require_string(document["status"], context="annotation schema status"),
            ),
        )

    @classmethod
    def from_canonical_definition_json(cls, value: str) -> AnnotationSchema:
        """Decode canonical definition JSON and reject non-canonical encodings."""

        try:
            parsed = json_loads(value)
        except ValueError as exc:
            raise AnnotationSchemaError("annotation schema definition is not valid finite JSON") from exc
        schema = cls.from_definition_document(parsed)
        if schema.canonical_definition_json() != value:
            raise AnnotationSchemaError("annotation schema definition JSON is not canonical")
        return schema


def validate_annotation_value(schema: AnnotationSchema, value: Mapping[str, object]) -> list[str]:
    """Validate one label payload against *schema*. Returns a list of error strings (empty = valid)."""

    errors: list[str] = []
    declared_names = {entry.name for entry in schema.fields}
    unknown_keys = sorted(set(value.keys()) - declared_names)
    if unknown_keys:
        errors.append(f"value has undeclared fields: {unknown_keys}")

    abstained = bool(schema.abstain_field and value.get(schema.abstain_field) is True)

    for field_def in schema.fields:
        present = field_def.name in value
        if not present:
            if field_def.required and not (abstained and field_def.name != schema.abstain_field):
                errors.append(f"missing required field {field_def.name!r}")
            continue
        field_error = field_def.validate_value(value[field_def.name])
        if field_error is not None:
            errors.append(field_error)

    return errors


def validate_annotation_row(
    schema: AnnotationSchema,
    *,
    target_ref: str,
    value: Mapping[str, object],
    evidence_refs: Sequence[str] = (),
) -> list[str]:
    """Validate one full annotation row: target grain, value shape, evidence policy."""

    errors: list[str] = []
    if not schema.accepts_target_kind(target_ref):
        errors.append(
            f"target_ref {target_ref!r} does not resolve to one of this schema's target_ref_kinds "
            f"{list(schema.target_ref_kinds)}"
        )
    errors.extend(validate_annotation_value(schema, value))
    if schema.evidence_policy == "required" and not evidence_refs:
        errors.append("schema requires evidence_refs and none were provided")
    return errors


class AnnotationSchemaRegistry:
    """Mutable registry of declared annotation schemas, keyed by ``schema_id@vN``.

    Not a hidden global by construction: :data:`ANNOTATION_SCHEMA_REGISTRY` is
    the conventional default instance for production registration, but tests
    and future import/MCP surfaces may build their own instance.
    """

    def __init__(self) -> None:
        self._schemas: dict[str, AnnotationSchema] = {}

    def register(self, schema: AnnotationSchema) -> AnnotationSchema:
        """Register *schema*. Canonical retries return the existing definition."""

        existing = self._schemas.get(schema.qualified_id)
        if existing is not None:
            if (
                existing.definition_fingerprint != schema.definition_fingerprint
                or existing.canonical_definition_json() != schema.canonical_definition_json()
            ):
                raise AnnotationSchemaError(
                    f"annotation schema {schema.qualified_id!r} is already registered with a different definition"
                )
            return existing
        self._schemas[schema.qualified_id] = schema
        return schema

    def get(self, schema_id: str, version: int | None = None) -> AnnotationSchema:
        """Return a registered schema, defaulting to the highest registered version."""

        if version is not None:
            key = f"{schema_id}@v{version}"
            try:
                return self._schemas[key]
            except KeyError as exc:
                raise KeyError(f"no registered annotation schema {key!r}") from exc
        candidates = [schema for schema in self._schemas.values() if schema.schema_id == schema_id]
        if not candidates:
            raise KeyError(f"no registered annotation schema with schema_id {schema_id!r}")
        return max(candidates, key=lambda schema: schema.version)

    def require_active(self, schema: AnnotationSchema) -> AnnotationSchema:
        """Return the identical registered active schema or fail closed.

        Annotation writes accept a schema object so callers can retain typed
        declarations, but the object is not authority by itself. Requiring an
        exact registry match prevents two definitions from reusing one
        ``schema_id@vN`` provenance stamp, and requiring ``active`` prevents
        draft/deprecated constructs from producing analytical rows.
        """

        try:
            registered = self.get(schema.schema_id, schema.version)
        except KeyError as exc:
            raise AnnotationSchemaError(
                f"annotation schema {schema.qualified_id!r} must be registered before writing"
            ) from exc
        if (
            registered.definition_fingerprint != schema.definition_fingerprint
            or registered.canonical_definition_json() != schema.canonical_definition_json()
        ):
            raise AnnotationSchemaError(
                f"annotation schema {schema.qualified_id!r} does not match its registered definition"
            )
        if registered.status != "active":
            raise AnnotationSchemaError(
                f"annotation schema {schema.qualified_id!r} is {registered.status!r}, not 'active'"
            )
        return registered

    def list(self) -> tuple[AnnotationSchema, ...]:
        """Return every registered schema, sorted by ``(schema_id, version)``."""

        return tuple(sorted(self._schemas.values(), key=lambda schema: (schema.schema_id, schema.version)))


ANNOTATION_SCHEMA_REGISTRY = AnnotationSchemaRegistry()


def register_annotation_schema(
    schema: AnnotationSchema, *, registry: AnnotationSchemaRegistry = ANNOTATION_SCHEMA_REGISTRY
) -> AnnotationSchema:
    """Register *schema* against *registry* (the module default unless overridden)."""

    return registry.register(schema)


def get_annotation_schema(
    schema_id: str, version: int | None = None, *, registry: AnnotationSchemaRegistry = ANNOTATION_SCHEMA_REGISTRY
) -> AnnotationSchema:
    """Look up a registered schema (module default registry unless overridden)."""

    return registry.get(schema_id, version)


def list_annotation_schemas(
    *, registry: AnnotationSchemaRegistry = ANNOTATION_SCHEMA_REGISTRY
) -> tuple[AnnotationSchema, ...]:
    """List every registered schema (module default registry unless overridden)."""

    return registry.list()


DELEGATION_DISCOURSE_SCHEMA = register_annotation_schema(
    AnnotationSchema(
        schema_id="delegation.discourse",
        version=1,
        title="Delegation discourse",
        description="Evidence-backed discourse role and applicability for one delegation attempt.",
        fields=(
            AnnotationField(
                name="directive_mode",
                value_type="enum",
                description="How the work order frames the requested action.",
                enum_values=("imperative", "collaborative", "goal_delegation", "question", "mixed", "not_observed"),
            ),
            AnnotationField(
                name="prohibitions",
                value_type="enum",
                description="How explicitly the work order constrains forbidden actions.",
                enum_values=("none", "implicit", "explicit", "multiple"),
            ),
            AnnotationField(
                name="autonomy",
                value_type="enum",
                description="How much execution discretion the delegate receives.",
                enum_values=("low", "bounded", "high", "unspecified"),
            ),
            AnnotationField(
                name="output_contract",
                value_type="enum",
                description="How specifically the expected result shape is declared.",
                enum_values=("unspecified", "informal", "structured", "machine_readable"),
            ),
            AnnotationField(
                name="scope_control",
                value_type="enum",
                description="How the work order bounds the implementation surface.",
                enum_values=("open", "bounded", "owned_paths", "owned_and_avoid_paths"),
            ),
            AnnotationField(
                name="verification_demand",
                value_type="enum",
                description="The strongest explicit verification obligation.",
                enum_values=("none", "self_check", "focused_tests", "broad_gate"),
            ),
            AnnotationField(
                name="checkpoint_escalation",
                value_type="enum",
                description="Whether checkpoints or escalation conditions are specified.",
                enum_values=("none", "checkpoint", "escalation", "both"),
            ),
            AnnotationField(
                name="relational_frame",
                value_type="enum",
                description="The interpersonal frame expressed by the work order.",
                enum_values=("directive", "collaborative", "advisory", "evaluative", "mixed"),
            ),
            AnnotationField(
                name="rationale_visibility",
                value_type="enum",
                description="How much rationale for constraints and choices is exposed.",
                enum_values=("none", "partial", "explicit"),
            ),
            AnnotationField(
                name="applicable",
                value_type="boolean",
                description="Whether the discourse construct applies to this delegation.",
            ),
            AnnotationField(
                name="confidence",
                value_type="number",
                description="Label confidence on the closed interval from zero to one.",
                minimum=0.0,
                maximum=1.0,
            ),
            AnnotationField(
                name="abstain",
                value_type="boolean",
                required=False,
                description="True when available evidence is insufficient to label this delegation.",
            ),
            AnnotationField(
                name="rationale",
                value_type="string",
                required=False,
                description="Concise evidence-grounded rationale for the label or abstention.",
            ),
        ),
        target_ref_kinds=("delegation",),
        abstain_field="abstain",
        evidence_policy="required",
        status="active",
    )
)


__all__ = [
    "ANNOTATION_SCHEMA_REGISTRY",
    "ANNOTATION_SCHEMA_DEFINITION_FORMAT",
    "DELEGATION_DISCOURSE_SCHEMA",
    "AnnotationEvidencePolicy",
    "AnnotationField",
    "AnnotationFieldType",
    "AnnotationSchema",
    "AnnotationSchemaError",
    "AnnotationSchemaRegistry",
    "AnnotationSchemaStatus",
    "get_annotation_schema",
    "list_annotation_schemas",
    "register_annotation_schema",
    "validate_annotation_row",
    "validate_annotation_value",
]
