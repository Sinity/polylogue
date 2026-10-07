"""Spill-backed schema validation for retained source revisions.

This module is the compact-result path used while preparing a retained raw
revision.  The caller owns one :class:`StreamedJSONDocument` context; all
sample scans, package fallbacks and drift reductions replay that same lazy
view before it is closed.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Iterator, KeysView, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, SupportsIndex, cast, overload

from jsonschema import Draft202012Validator, ValidationError, validators

from polylogue.archive.raw_payload.sampling_buckets import is_record_candidate
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.core.json import JSONDocument
from polylogue.core.provider_identity import normalize_provider_token
from polylogue.core.sources import origin_from_provider
from polylogue.schemas.drift_sentinel import (
    FIELD_CHANGED,
    KNOWN_FIELD_UNREAD,
    NEW_FIELD,
    UNSEEN_SHAPE,
    DriftClassification,
    SchemaDriftObservation,
)
from polylogue.schemas.packages import SchemaResolution
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.schemas.schema_parser_coverage import unread_field_names
from polylogue.schemas.validator_resolution import resolve_retained_schema

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class RetainedValidationVerdict:
    """Bounded validation outcome attached to its retained source evidence."""

    raw_id: str
    revision_sha256: str
    evidence_id: str
    mode: ValidationMode
    status: ValidationStatus
    sample_count: int
    invalid_count: int
    error_count: int
    drift_count: int
    first_diagnostic: str | None
    schema_resolution: SchemaResolution | None
    drift_observation: SchemaDriftObservation | None
    strict_refusal: bool


_DRIFT_STRENGTH: dict[DriftClassification, int] = {
    FIELD_CHANGED: 4,
    UNSEEN_SHAPE: 3,
    KNOWN_FIELD_UNREAD: 2,
    NEW_FIELD: 1,
}


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
    schema_resolution: SchemaResolution | None = None,
    schema_resolution_is_explicit: bool = False,
    registry: SchemaRegistry | None = None,
) -> RetainedValidationVerdict:
    """Decode and validate a complete retained JSON document using disk spill.

    Validation mode controls schema validation only.  Decoding remains the
    caller's independent admission concern, and drift never changes whether
    the source revision can be published.
    """
    mode = ValidationMode.from_string(mode)
    if mode is ValidationMode.OFF:
        return _verdict(
            raw_id,
            revision_sha256,
            evidence_id,
            mode,
            ValidationStatus.SKIPPED,
            schema_resolution=schema_resolution,
        )

    from polylogue.schemas.observation_spill import StreamedJSONDocument

    document = StreamedJSONDocument(path, jsonl=jsonl)
    with document as payload:
        spill = _active_spill(document)
        active_registry = registry or SchemaRegistry()
        resolved: SchemaResolution | None = schema_resolution
        from polylogue.schemas.validator_resolution import canonical_provider

        canonical = canonical_provider(provider)
        selected_schema: Mapping[str, object] | None = None
        schema_key: tuple[str, str, str] | None = None
        try:
            canonical, selected, schema_key, resolved = resolve_retained_schema(
                provider,
                payload,
                source_path=source_path,
                schema_resolution=schema_resolution,
                schema_resolution_is_explicit=schema_resolution_is_explicit,
                registry=active_registry,
                schema_store=spill.store_schema,
                schema_accepts=lambda candidate: _schema_accepts_document(
                    payload,
                    candidate,
                    canonical,
                    spill.connection,
                ),
            )
            selected_schema = selected
        except (FileNotFoundError, ImportError):
            return _verdict(
                raw_id,
                revision_sha256,
                evidence_id,
                mode,
                ValidationStatus.SKIPPED,
                schema_resolution=resolved,
            )

        assert selected_schema is not None
        sample_count = 0
        invalid_count = 0
        error_count = 0
        drift_count = 0
        first_diagnostic: str | None = None
        strongest: SchemaDriftObservation | None = None
        for sample in _validation_samples(payload, selected_schema, canonical):
            check_compute_cancelled()
            sample_count += 1
            validator = _bounded_validator(selected_schema, spill.connection)
            normalized = _normalized(sample, selected_schema, selected_schema, spill.connection)
            sample_errors = 0
            for error in validator.iter_errors(normalized):
                check_compute_cancelled()
                sample_errors += 1
                error_count += 1
                if first_diagnostic is None:
                    first_diagnostic = _diagnostic(error)
            valid = sample_errors == 0
            if not valid:
                invalid_count += 1

            drift_count += _collect_drift_paths(
                sample,
                selected_schema,
                spill.connection,
                sample_count,
            )
            sample_drift = _reduce_sample_drift(
                sample,
                selected_schema,
                canonical,
                resolved,
                spill.connection,
                sample_count,
                raw_id=raw_id,
                native_id_example=source_path or raw_id,
                is_valid=valid,
            )
            strongest = _stronger_drift(strongest, sample_drift)

        status = ValidationStatus.FAILED if mode is ValidationMode.STRICT and invalid_count else ValidationStatus.PASSED
        diagnostic = (
            f"Schema validation failed: {first_diagnostic}"
            if status is ValidationStatus.FAILED and first_diagnostic is not None
            else first_diagnostic
        )
        return RetainedValidationVerdict(
            raw_id=raw_id,
            revision_sha256=revision_sha256,
            evidence_id=evidence_id,
            mode=mode,
            status=status,
            sample_count=sample_count,
            invalid_count=invalid_count,
            error_count=error_count,
            drift_count=drift_count,
            first_diagnostic=diagnostic,
            schema_resolution=resolved,
            drift_observation=strongest,
            strict_refusal=status is ValidationStatus.FAILED,
        )


def _verdict(
    raw_id: str,
    revision_sha256: str,
    evidence_id: str,
    mode: ValidationMode,
    status: ValidationStatus,
    *,
    schema_resolution: SchemaResolution | None = None,
) -> RetainedValidationVerdict:
    return RetainedValidationVerdict(
        raw_id=raw_id,
        revision_sha256=revision_sha256,
        evidence_id=evidence_id,
        mode=mode,
        status=status,
        sample_count=0,
        invalid_count=0,
        error_count=0,
        drift_count=0,
        first_diagnostic=None,
        schema_resolution=schema_resolution,
        drift_observation=None,
        strict_refusal=False,
    )


def _active_spill(document: object) -> Any:
    # The root lazy object intentionally carries the same owner connection as
    # child objects.  The public context exposes it through the document, so
    # callers can use a single private database for tree and reducers.
    connection = getattr(document, "connection", None)
    if connection is None:
        raise RuntimeError("retained schema document has no live spill connection")

    class SpillAccess:
        def __init__(self, document: object, conn: sqlite3.Connection) -> None:
            self.document = document
            self.connection = conn

        def store_schema(self, value: object) -> JSONDocument:
            store = getattr(self.document, "store_schema", None)
            if store is None:
                raise RuntimeError("retained schema document does not expose schema spill storage")
            return cast(JSONDocument, store(value))

    return SpillAccess(document, cast(sqlite3.Connection, connection))


def _validation_samples(
    payload: object,
    schema: Mapping[str, object],
    provider: Provider,
) -> Iterator[Mapping[str, object]]:
    granularity = schema.get("x-polylogue-sample-granularity")
    if granularity not in {"record", "document"}:
        granularity = "record" if provider in {Provider.CLAUDE_CODE, Provider.CODEX} else "document"
    if isinstance(payload, Mapping):
        if granularity == "document" or is_record_candidate(cast(JSONDocument, payload)):
            yield payload
        return
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for value in payload:
            check_compute_cancelled()
            if isinstance(value, Mapping) and (
                granularity == "document" or is_record_candidate(cast(JSONDocument, value))
            ):
                yield value


def _diagnostic(error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path) or "root"
    keyword = str(error.validator or "schema")
    return f"{path}: {keyword} validation failed"


def _schema_accepts_document(
    payload: object,
    schema: Mapping[str, object],
    provider: Provider,
    connection: sqlite3.Connection,
) -> bool:
    for sample in _validation_samples(payload, schema, provider):
        check_compute_cancelled()
        validator = _bounded_validator(schema, connection)
        if next(validator.iter_errors(_normalized(sample, schema, schema, connection)), None) is not None:
            return False
    return True


def _normalized(
    value: object,
    schema: object,
    root: Mapping[str, object],
    connection: sqlite3.Connection,
) -> object:
    from polylogue.schemas.validator import (
        _schema_allows_type,
        _schema_branch_for_value,
    )

    selected = _schema_branch_for_value(schema, value, root, connection=connection)
    if isinstance(value, dict):
        return _NormalizedObject(value, selected, root, connection)
    if isinstance(value, list):
        if not value and not _schema_allows_type(selected, "array") and _schema_allows_type(selected, "null"):
            return None
        return _NormalizedArray(value, selected, root, connection)
    return value


class _NormalizedObject(dict[str, object]):
    def __init__(
        self, value: dict[str, object], schema: object, root: Mapping[str, object], connection: sqlite3.Connection
    ) -> None:
        dict.__init__(self)
        self._value = value
        self._schema = schema
        self._root = root
        self._connection = connection

    def __iter__(self) -> Iterator[str]:
        for key in self._value:
            check_compute_cancelled()
            yield key

    def __len__(self) -> int:
        return len(self._value)

    def __contains__(self, key: object) -> bool:
        return key in self._value

    def keys(self) -> KeysView[str]:  # type: ignore[override]
        return KeysView(self)

    def sorted_keys(self) -> Iterator[str]:
        source = getattr(self._value, "sorted_keys", None)
        keys = source() if source is not None else iter(sorted(self._value))
        for key in keys:
            check_compute_cancelled()
            yield key

    def __getitem__(self, key: str) -> object:
        check_compute_cancelled()
        from polylogue.schemas.validator import _schema_for_property

        value = self._value[key]
        child_schema = _schema_for_property(self._schema, key, value, self._root, connection=self._connection)
        return _normalized(value, child_schema, self._root, self._connection)

    def get(self, key: str, default: object = None) -> object:
        try:
            return self[key]
        except KeyError:
            return default

    def items(self) -> Iterator[tuple[str, object]]:  # type: ignore[override]
        for key in self._value:
            check_compute_cancelled()
            yield key, self[key]

    def values(self) -> Iterator[object]:  # type: ignore[override]
        for key in self._value:
            check_compute_cancelled()
            yield self[key]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Mapping)
            and len(self) == len(other)
            and all(key in other and self[key] == other[key] for key in self)
        )

    def __ne__(self, other: object) -> bool:
        return not self == other


class _NormalizedArray(list[object]):
    def __init__(
        self, value: list[object], schema: object, root: Mapping[str, object], connection: sqlite3.Connection
    ) -> None:
        list.__init__(self)
        self._value = value
        self._schema = schema
        self._root = root
        self._connection = connection

    def __len__(self) -> int:
        return len(self._value)

    def __iter__(self) -> Iterator[object]:
        for index in range(len(self)):
            check_compute_cancelled()
            yield self[index]

    @overload
    def __getitem__(self, index: SupportsIndex) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> list[object]: ...

    def __getitem__(self, index: SupportsIndex | slice) -> object:
        check_compute_cancelled()
        from polylogue.schemas.validator import _schema_for_items

        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        value = self._value[int(index)]
        return _normalized(value, _schema_for_items(self._schema, value, self._root), self._root, self._connection)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Sequence)
            and not isinstance(other, (str, bytes))
            and len(self) == len(other)
            and all(left == right for left, right in zip(self, other, strict=True))
        )

    def __ne__(self, other: object) -> bool:
        return not self == other


def _bounded_validator(schema: Mapping[str, object], connection: sqlite3.Connection) -> Any:

    def any_of(
        validator: Any, schemas: Sequence[object], instance: object, schema_node: object
    ) -> Iterator[ValidationError]:
        accepted = False
        for branch in schemas:
            check_compute_cancelled()
            if validator.evolve(schema=branch).is_valid(instance):
                accepted = True
                break
        if not accepted:
            yield ValidationError("No anyOf branch accepted the value", instance=instance, schema=schema_node)

    def one_of(
        validator: Any, schemas: Sequence[object], instance: object, schema_node: object
    ) -> Iterator[ValidationError]:
        accepted = 0
        for branch in schemas:
            check_compute_cancelled()
            if validator.evolve(schema=branch).is_valid(instance):
                accepted += 1
                if accepted > 1:
                    break
        if accepted != 1:
            yield ValidationError("oneOf requires exactly one accepting branch", instance=instance, schema=schema_node)

    def additional_properties(
        validator: Any, additional: object, instance: object, schema_node: object
    ) -> Iterator[ValidationError]:
        if not validator.is_type(instance, "object") or not isinstance(instance, Mapping):
            return
        properties = schema_node.get("properties", {}) if isinstance(schema_node, Mapping) else {}
        patterns = schema_node.get("patternProperties", {}) if isinstance(schema_node, Mapping) else {}
        compiled = [(re.compile(pattern), subschema) for pattern, subschema in patterns.items()]
        for key, value in instance.items():
            check_compute_cancelled()
            if key in properties:
                continue
            matching = [subschema for pattern, subschema in compiled if pattern.search(key)]
            if matching:
                continue
            if additional is False:
                error = ValidationError("additional property is not allowed", instance=instance, schema=schema_node)
                error.path.append(key)
                yield error
            else:
                yield from validator.descend(value, additional, path=key)

    def unique_items(
        validator: Any, enabled: object, instance: object, schema_node: object
    ) -> Iterator[ValidationError]:
        if enabled is not True or not validator.is_type(instance, "array") or not isinstance(instance, Sequence):
            return
        scope = _new_scope(connection)
        try:
            for index, item in enumerate(instance):
                check_compute_cancelled()
                digest = _json_equality_digest(item)
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO retained_unique(scope,digest) VALUES (?,?)", (scope, digest)
                )
                if cursor.rowcount == 0:
                    error = ValidationError("array items are not unique", instance=instance, schema=schema_node)
                    error.path.append(index)
                    yield error
                    break
        finally:
            connection.execute("DELETE FROM retained_unique WHERE scope=?", (scope,))
            connection.execute("DELETE FROM retained_scope WHERE id=?", (scope,))

    def unevaluated_properties(
        validator: Any, unevaluated: object, instance: object, schema_node: object
    ) -> Iterator[ValidationError]:
        if not validator.is_type(instance, "object") or not isinstance(instance, Mapping):
            return
        scope = _new_scope(connection)
        try:
            for key in _evaluated_property_keys(validator, instance, schema_node):
                connection.execute("INSERT OR IGNORE INTO retained_eval_props VALUES (?,?)", (scope, key))
            for key, value in instance.items():
                check_compute_cancelled()
                found = connection.execute(
                    "SELECT 1 FROM retained_eval_props WHERE scope=? AND property=?", (scope, key)
                ).fetchone()
                if found is None:
                    yield from validator.descend(value, unevaluated, path=key)
        finally:
            connection.execute("DELETE FROM retained_eval_props WHERE scope=?", (scope,))
            connection.execute("DELETE FROM retained_scope WHERE id=?", (scope,))

    def unevaluated_items(
        validator: Any, unevaluated: object, instance: object, schema_node: object
    ) -> Iterator[ValidationError]:
        if not validator.is_type(instance, "array") or not isinstance(instance, Sequence):
            return
        scope = _new_scope(connection)
        try:
            for index in _evaluated_item_indexes(validator, instance, schema_node):
                connection.execute("INSERT OR IGNORE INTO retained_eval_items VALUES (?,?)", (scope, index))
            for index, value in enumerate(instance):
                check_compute_cancelled()
                found = connection.execute(
                    "SELECT 1 FROM retained_eval_items WHERE scope=? AND item_index=?", (scope, index)
                ).fetchone()
                if found is None:
                    yield from validator.descend(value, unevaluated, path=index)
        finally:
            connection.execute("DELETE FROM retained_eval_items WHERE scope=?", (scope,))
            connection.execute("DELETE FROM retained_scope WHERE id=?", (scope,))

    custom = validators.extend(
        Draft202012Validator,
        validators={
            "anyOf": any_of,
            "oneOf": one_of,
            "additionalProperties": additional_properties,
            "uniqueItems": unique_items,
            "unevaluatedProperties": unevaluated_properties,
            "unevaluatedItems": unevaluated_items,
        },
    )
    _ensure_reducer_tables(connection)
    return custom(schema)


def _ensure_reducer_tables(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE IF NOT EXISTS retained_scope(id INTEGER PRIMARY KEY)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retained_unique(scope INTEGER NOT NULL,digest BLOB NOT NULL,PRIMARY KEY(scope,digest)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retained_drift(sample INTEGER NOT NULL,path TEXT NOT NULL,PRIMARY KEY(sample,path)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retained_unread(sample INTEGER NOT NULL,path TEXT NOT NULL,PRIMARY KEY(sample,path)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retained_eval_props(scope INTEGER NOT NULL,property TEXT NOT NULL,PRIMARY KEY(scope,property)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retained_eval_items(scope INTEGER NOT NULL,item_index INTEGER NOT NULL,PRIMARY KEY(scope,item_index)) WITHOUT ROWID"
    )


def _new_scope(connection: sqlite3.Connection) -> int:
    lastrowid = connection.execute("INSERT INTO retained_scope DEFAULT VALUES").lastrowid
    assert lastrowid is not None
    return int(lastrowid)


def _evaluated_property_keys(validator: Any, instance: Mapping[str, object], schema: object) -> Iterator[str]:
    if validator.is_type(schema, "boolean") or not isinstance(schema, Mapping):
        return
    for ref_key in ("$ref", "$dynamicRef"):
        ref = schema.get(ref_key)
        if ref is not None:
            resolved = validator._resolver.lookup(ref)
            yield from _evaluated_property_keys(
                validator.evolve(schema=resolved.contents, _resolver=resolved.resolver), instance, resolved.contents
            )
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for key in properties:
            check_compute_cancelled()
            if key in instance:
                yield key
    for keyword in ("additionalProperties", "unevaluatedProperties"):
        sub = schema.get(keyword)
        if sub is None:
            continue
        for key, value in instance.items():
            check_compute_cancelled()
            if next(validator.descend(value, sub), None) is None:
                yield key
    patterns = schema.get("patternProperties")
    if isinstance(patterns, Mapping):
        compiled = tuple(re.compile(pattern) for pattern in patterns if isinstance(pattern, str))
        for key in instance:
            check_compute_cancelled()
            if any(pattern.search(key) for pattern in compiled):
                yield key
    dependent = schema.get("dependentSchemas")
    if isinstance(dependent, Mapping):
        for key, sub in dependent.items():
            check_compute_cancelled()
            if key in instance:
                yield from _evaluated_property_keys(validator, instance, sub)
    for keyword in ("allOf", "oneOf", "anyOf"):
        for sub in schema.get(keyword, ()):
            check_compute_cancelled()
            evolved = validator.evolve(schema=sub)
            if evolved.is_valid(instance):
                yield from _evaluated_property_keys(evolved, instance, sub)
    condition = schema.get("if")
    if condition is not None:
        if validator.evolve(schema=condition).is_valid(instance):
            yield from _evaluated_property_keys(validator, instance, condition)
            if "then" in schema:
                yield from _evaluated_property_keys(validator, instance, schema["then"])
        elif "else" in schema:
            yield from _evaluated_property_keys(validator, instance, schema["else"])


def _evaluated_item_indexes(validator: Any, instance: Sequence[object], schema: object) -> Iterator[int]:
    if validator.is_type(schema, "boolean") or not isinstance(schema, Mapping):
        return
    if "items" in schema:
        yield from range(len(instance))
        return
    for ref_key in ("$ref", "$dynamicRef"):
        ref = schema.get(ref_key)
        if ref is not None:
            resolved = validator._resolver.lookup(ref)
            yield from _evaluated_item_indexes(
                validator.evolve(schema=resolved.contents, _resolver=resolved.resolver), instance, resolved.contents
            )
    prefix = schema.get("prefixItems")
    if isinstance(prefix, Sequence) and not isinstance(prefix, (str, bytes)):
        for index in range(min(len(prefix), len(instance))):
            check_compute_cancelled()
            yield index
    condition = schema.get("if")
    if condition is not None:
        if validator.evolve(schema=condition).is_valid(instance):
            yield from _evaluated_item_indexes(validator, instance, condition)
            if "then" in schema:
                yield from _evaluated_item_indexes(validator, instance, schema["then"])
        elif "else" in schema:
            yield from _evaluated_item_indexes(validator, instance, schema["else"])
    for keyword in ("contains", "unevaluatedItems"):
        sub = schema.get(keyword)
        if sub is not None:
            for index, value in enumerate(instance):
                check_compute_cancelled()
                if validator.evolve(schema=sub).is_valid(value):
                    yield index
    for keyword in ("allOf", "oneOf", "anyOf"):
        for sub in schema.get(keyword, ()):
            check_compute_cancelled()
            evolved = validator.evolve(schema=sub)
            if evolved.is_valid(instance):
                yield from _evaluated_item_indexes(evolved, instance, sub)


def _json_equality_digest(value: object) -> bytes:
    digest = hashlib.sha256()

    def visit(node: object) -> None:
        check_compute_cancelled()
        if node is None:
            digest.update(b"null\0")
        elif isinstance(node, bool):
            digest.update(b"bool\1" if node else b"bool\0")
        elif isinstance(node, (int, float, Decimal)):
            digest.update(b"number\0")
            number = Decimal(str(node)).normalize()
            digest.update(str(number).encode("ascii"))
            digest.update(b"\0")
        elif isinstance(node, str):
            digest.update(b"string\0")
            data = node.encode("utf-8", "surrogatepass")
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
        elif isinstance(node, Mapping):
            digest.update(b"object\0")
            keys = node.sorted_keys() if hasattr(node, "sorted_keys") else iter(sorted(node))
            for key in keys:
                check_compute_cancelled()
                visit(str(key))
                visit(node[key])
            digest.update(b"end-object\0")
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            digest.update(b"array\0")
            for item in node:
                check_compute_cancelled()
                visit(item)
            digest.update(b"end-array\0")
        else:
            digest.update(type(node).__qualname__.encode("utf-8"))

    visit(value)
    return digest.digest()


def _collect_drift_paths(
    sample: Mapping[str, object],
    schema: Mapping[str, object],
    connection: sqlite3.Connection,
    sample_index: int,
) -> int:
    from polylogue.schemas.validator import _iter_drift_paths

    _ensure_reducer_tables(connection)
    count = 0
    connection.execute("DELETE FROM retained_drift WHERE sample=?", (sample_index,))
    for path in _iter_drift_paths(sample, schema, "", schema, connection):
        check_compute_cancelled()
        count += 1
        connection.execute("INSERT OR IGNORE INTO retained_drift VALUES (?,?)", (sample_index, path))
    return count


def _reduce_sample_drift(
    sample: Mapping[str, object],
    schema: Mapping[str, object],
    provider: Provider,
    resolution: SchemaResolution | None,
    connection: sqlite3.Connection,
    sample_index: int,
    *,
    raw_id: str,
    native_id_example: str,
    is_valid: bool,
) -> SchemaDriftObservation | None:
    if resolution is None:
        return None
    path_count = int(
        connection.execute("SELECT COUNT(*) FROM retained_drift WHERE sample=?", (sample_index,)).fetchone()[0]
    )
    path_signature = str(
        connection.execute(
            "SELECT group_concat(path, ',') FROM (SELECT path FROM retained_drift WHERE sample=? ORDER BY path)",
            (sample_index,),
        ).fetchone()[0]
        or ""
    )
    unread_count = 0
    unread_signature = ""
    if not path_count and is_valid:
        unread_names = unread_field_names(normalize_provider_token(str(provider)))
        connection.execute("DELETE FROM retained_unread WHERE sample=?", (sample_index,))
        for key in sample:
            check_compute_cancelled()
            if str(key) in unread_names:
                connection.execute("INSERT OR IGNORE INTO retained_unread VALUES (?,?)", (sample_index, str(key)))
        unread_count = int(
            connection.execute("SELECT COUNT(*) FROM retained_unread WHERE sample=?", (sample_index,)).fetchone()[0]
        )
        unread_signature = str(
            connection.execute(
                "SELECT group_concat(path, ',') FROM (SELECT path FROM retained_unread WHERE sample=? ORDER BY path)",
                (sample_index,),
            ).fetchone()[0]
            or ""
        )
    classification: DriftClassification | None
    if not is_valid:
        classification = FIELD_CHANGED
    elif resolution.reason == "package_default":
        classification = UNSEEN_SHAPE
    elif path_count:
        classification = NEW_FIELD
    elif unread_count:
        classification = KNOWN_FIELD_UNREAD
    else:
        classification = None
    if classification is None:
        return None
    signature = unread_signature if classification == KNOWN_FIELD_UNREAD else path_signature
    return SchemaDriftObservation(
        origin=str(origin_from_provider(provider)),
        element_kind=resolution.element_kind,
        classification=classification,
        unseen_key_signature=signature,
        native_id_example=native_id_example,
        raw_id=raw_id,
    )


def _stronger_drift(
    current: SchemaDriftObservation | None,
    candidate: SchemaDriftObservation | None,
) -> SchemaDriftObservation | None:
    if current is None:
        return candidate
    if candidate is None:
        return current
    left = _DRIFT_STRENGTH[current.classification]
    right = _DRIFT_STRENGTH[candidate.classification]
    if right != left:
        return candidate if right > left else current
    return min(current, candidate, key=lambda item: item.unseen_key_signature)
