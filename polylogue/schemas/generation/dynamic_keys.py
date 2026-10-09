"""Dynamic-key collapse helpers for schema generation."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass

try:
    from genson import SchemaBuilder

    GENSON_AVAILABLE = True
except ImportError:
    SchemaBuilder = None
    GENSON_AVAILABLE = False

from polylogue.core.json import JSONDocument, JSONValue, json_document, json_document_list
from polylogue.schemas.field_stats.detection import is_dynamic_key, should_collapse_observed_keys

_STRUCTURAL_DEDUP_WINDOW = 1_024
_COMPOSITE_KEYWORDS = ("anyOf", "oneOf", "allOf")
_EXACT_STRUCTURE_WITNESS_CAP = 1_024


@dataclass(frozen=True)
class StructureWitnessRetention:
    """The bounded, monotonic witness set published for one package element."""

    exact_structure_ids: tuple[str, ...]
    omitted_current_witness_count: int


def canonicalize_structure_schema(schema: Mapping[str, object]) -> JSONDocument:
    """Normalize schema-only unordered ``required`` arrays for structural IDs.

    JSON Schema treats ``required`` as a set.  Its order is nevertheless
    inherited from the source object's key order, including below array items
    and dynamic-map value schemas.  Other arrays have schema-defined order and
    must remain untouched.
    """

    def canonicalize(value: object, *, parent_key: str | None = None) -> JSONValue:
        if isinstance(value, Mapping):
            return {str(key): canonicalize(child, parent_key=str(key)) for key, child in value.items()}
        if isinstance(value, list):
            values = [canonicalize(item) for item in value]
            if parent_key == "required" and all(isinstance(item, str) for item in values):
                ordered_values: list[JSONValue] = []
                ordered_values.extend(sorted(item for item in values if isinstance(item, str)))
                return ordered_values
            return values
        if value is None or isinstance(value, str | int | float | bool):
            return value
        raise TypeError(f"Unsupported structural schema value: {type(value).__name__}")

    return json_document(canonicalize(schema))


def _structure_schema_parts(value: object, *, canonical: bool, parent_key: str | None = None) -> Iterator[str]:
    from polylogue.schemas.shape_fingerprint import ordered_keys

    if isinstance(value, Mapping):
        yield "{"
        for index, key in enumerate(ordered_keys(value)):
            if index:
                yield ","
            yield json.dumps(key, ensure_ascii=False) + ":"
            yield from _structure_schema_parts(value[key], canonical=canonical, parent_key=key)
        yield "}"
    elif isinstance(value, list):
        items = (
            sorted(value)
            if canonical and parent_key == "required" and all(isinstance(item, str) for item in value)
            else value
        )
        yield "["
        for index, child in enumerate(items):
            if index:
                yield ","
            yield from _structure_schema_parts(child, canonical=canonical)
        yield "]"
    else:
        yield json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _structure_schema_hash(schema: Mapping[str, object], *, canonical: bool) -> str:
    digest = hashlib.sha256()
    for part in _structure_schema_parts(schema, canonical=canonical):
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def legacy_structure_schema_digest(schema: Mapping[str, object]) -> str:
    """Return the shipped required-order digest, streaming the schema tree."""
    return _structure_schema_hash(schema, canonical=False)


def structure_schema_digest(schema: Mapping[str, object]) -> str:
    """Return the canonical required-order digest, streaming the schema tree."""
    return _structure_schema_hash(schema, canonical=True)


def is_source_structure_witness(value: str) -> bool:
    """Source evidence uses full SHA-256 IDs; archive cluster IDs are shorter."""
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def retain_exact_structure_witnesses(
    prior_witnesses: Iterable[str], current_witnesses: Iterable[str]
) -> StructureWitnessRetention:
    """Preserve all archive IDs and bound source witnesses to 1024.

    Published witnesses remain authoritative when their source budget is full.
    """

    prior = set(prior_witnesses)
    unseen_current = set(current_witnesses) - prior
    source_ids = sorted(value for value in unseen_current if is_source_structure_witness(value))
    archive_ids = unseen_current - set(source_ids)
    capacity = max(0, _EXACT_STRUCTURE_WITNESS_CAP - sum(map(is_source_structure_witness, prior)))
    admitted = source_ids[:capacity]
    return StructureWitnessRetention(
        exact_structure_ids=tuple(sorted(prior | archive_ids | set(admitted))),
        omitted_current_witness_count=len(source_ids) - len(admitted),
    )


def merge_schemas(schemas: Iterable[JSONDocument]) -> JSONDocument:
    """Merge multiple schemas into one using genson when available."""
    if not GENSON_AVAILABLE:
        return next(iter(schemas), {})
    builder = SchemaBuilder()
    observed = False
    for schema in schemas:
        builder.add_schema(schema)
        observed = True
    return json_document(builder.to_schema()) if observed else {}


def _schema_types(schema: Mapping[str, object]) -> set[str]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return {schema_type}
    if isinstance(schema_type, list):
        return {item for item in schema_type if isinstance(item, str)}
    return set()


def _required_names(schema: Mapping[str, object]) -> set[str]:
    required = schema.get("required")
    return {item for item in required if isinstance(item, str)} if isinstance(required, list) else set()


def _schema_object(value: JSONValue) -> JSONDocument:
    """Read an object node from an already JSON-typed schema."""
    return value if isinstance(value, dict) else {}


def _flatten_composite_branches(schema: JSONDocument) -> JSONDocument:
    """Fold ``anyOf``/``oneOf``/``allOf`` branches into the schema body.

    The observed-structure merge is a structural union that reads ``type``,
    ``properties``, ``items`` and ``additionalProperties``.  Composite
    keywords were invisible to it, so merging a nullable object -- which the
    generator emits as ``anyOf: [null, object]`` -- against a plain object
    silently discarded every property inside the branches.  Folding the
    branches in first makes the union monotonic: merging can no longer remove
    a name the accumulator already carried.
    """
    branches: list[JSONDocument] = []
    for keyword in _COMPOSITE_KEYWORDS:
        raw = schema.get(keyword)
        if isinstance(raw, list):
            branches.extend(_schema_object(branch) for branch in raw)
    if not branches:
        return schema
    merged = {key: value for key, value in schema.items() if key not in _COMPOSITE_KEYWORDS}
    for branch in branches:
        merged = _merge_observed_structure_pair(merged, branch)
    return merged


def _merge_observed_structure_pair(
    left: JSONDocument, right: JSONDocument, *, store: Callable[[object], JSONDocument] | None = None
) -> JSONDocument:
    if not left:
        return right
    if not right:
        return left
    left = _flatten_composite_branches(left)
    right = _flatten_composite_branches(right)
    if not left:
        return right
    if not right:
        return left

    merged: JSONDocument = {}
    schema_types = sorted(_schema_types(left) | _schema_types(right))
    if len(schema_types) == 1:
        merged["type"] = schema_types[0]
    elif schema_types:
        type_values: list[JSONValue] = list(schema_types)
        merged["type"] = type_values

    left_properties = _schema_object(left.get("properties"))
    right_properties = _schema_object(right.get("properties"))
    property_names = sorted(set(left_properties) | set(right_properties))
    properties: JSONDocument = {}
    for name in property_names:
        left_schema = _schema_object(left_properties.get(name))
        right_schema = _schema_object(right_properties.get(name))
        properties[name] = _merge_observed_structure_pair(left_schema, right_schema, store=store)

    required = sorted(_required_names(left) & _required_names(right))
    left_additional = _schema_object(left.get("additionalProperties"))
    right_additional = _schema_object(right.get("additionalProperties"))
    additional = _merge_observed_structure_pair(left_additional, right_additional, store=store)

    already_high_cardinality = (
        left.get("x-polylogue-high-cardinality-keys") is True or right.get("x-polylogue-high-cardinality-keys") is True
    )
    if already_high_cardinality:
        # A collapsed object stays collapsed: a key first seen in a later
        # sample joins the value schema. Only the names a marked side already
        # retained as finite siblings of an explicit map remain properties.
        retained_names: set[str] = set()
        for side in (left, right):
            if side.get("x-polylogue-high-cardinality-keys") is True:
                retained_names.update(_schema_object(side.get("properties")))
        fresh = [schema for name, schema in properties.items() if name not in retained_names]
        if fresh:
            additional = merge_observed_structure_schemas([additional, *map(_schema_object, fresh)], store=store)
            properties = {name: schema for name, schema in properties.items() if name in retained_names}
            required = [name for name in required if name in retained_names]
    elif properties and should_collapse_observed_keys(properties.keys()):
        # Beside an explicit map only the individually dynamic names fold
        # into it; the static partition stays as finite sibling properties.
        # Without a map every name folds.
        collapsed_names = {name for name in properties if is_dynamic_key(name)} if additional else set(properties)
        if collapsed_names:
            additional = merge_observed_structure_schemas(
                [additional, *(_schema_object(properties[name]) for name in sorted(collapsed_names))], store=store
            )
            properties = {name: schema for name, schema in properties.items() if name not in collapsed_names}
            required = [name for name in required if name not in collapsed_names]
        merged["x-polylogue-high-cardinality-keys"] = True
        merged["x-polylogue-dynamic-keys"] = True

    if properties:
        merged["properties"] = properties
    if required:
        required_values: list[JSONValue] = list(required)
        merged["required"] = required_values

    left_items = _schema_object(left.get("items"))
    right_items = _schema_object(right.get("items"))
    items = _merge_observed_structure_pair(left_items, right_items, store=store)
    if items:
        merged["items"] = items
    if additional:
        merged["additionalProperties"] = additional
        merged["x-polylogue-dynamic-keys"] = True

    for marker in ("x-polylogue-dynamic-keys", "x-polylogue-high-cardinality-keys"):
        if left.get(marker) is True or right.get(marker) is True:
            merged[marker] = True
    return store(merged) if store is not None else merged


def merge_observed_structure_schemas(
    schemas: Iterable[JSONDocument], *, store: Callable[[object], JSONDocument] | None = None
) -> JSONDocument:
    """Incrementally merge structural schemas without retaining property history."""
    merged: JSONDocument = {}
    seen_identities: set[bytes] = set()
    identity_order: deque[bytes] = deque()
    for schema in schemas:
        identity = bytes.fromhex(legacy_structure_schema_digest(schema))
        if identity in seen_identities:
            continue
        if len(identity_order) >= _STRUCTURAL_DEDUP_WINDOW:
            seen_identities.remove(identity_order.popleft())
        seen_identities.add(identity)
        identity_order.append(identity)
        merged = _merge_observed_structure_pair(merged, schema, store=store)
    return merged


def dynamic_object_paths(schema: Mapping[str, object], path: str = "$") -> set[str]:
    """Return schema paths whose observed object keys were collapsed."""
    paths: set[str] = set()
    if schema.get("x-polylogue-dynamic-keys") is True:
        paths.add(path)
    for name, child in json_document(schema.get("properties")).items():
        paths.update(dynamic_object_paths(json_document(child), f"{path}.{name}"))
    items = json_document(schema.get("items"))
    if items:
        paths.update(dynamic_object_paths(items, f"{path}[*]"))
    additional = json_document(schema.get("additionalProperties"))
    if additional:
        paths.update(dynamic_object_paths(additional, f"{path}.*"))
    return paths


def observed_structure_schema(
    value: object, *, field_name: str | None = None, store: Callable[[object], JSONDocument] | None = None
) -> JSONDocument:
    """Build a bounded structural schema before Genson sees dynamic keys.

    Genson creates one node per object property.  Feeding it provider payloads
    directly therefore retains unbounded ID-, path-, and content-shaped keys
    until the later schema-collapse pass.  This recursive projection preserves
    every observed value shape while merging dynamic-key values incrementally
    into ``additionalProperties``.
    """
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        array_schema: JSONDocument = {"type": "array"}
        items = merge_observed_structure_schemas(
            (
                observed_structure_schema(item, store=store)
                for item in (value.structure_values() if hasattr(value, "structure_values") else value)
            ),
            store=store,
        )
        if items:
            array_schema["items"] = items
        return store(array_schema) if store is not None else array_schema
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported schema observation value: {type(value).__name__}")

    from polylogue.schemas.observation_spill import SpilledObject

    def entries() -> Iterator[tuple[str | None, JSONValue]]:
        if isinstance(value, SpilledObject):
            with closing(value.structure_key_items()) as children:
                for key, child in children:
                    yield key.small_name, child
        else:
            for key, child in value.items():
                yield str(key), child

    filename_map = field_name == "trackedFileBackups"
    collapse_all = filename_map or (
        value.collapse_observed_keys()
        if isinstance(value, SpilledObject)
        else should_collapse_observed_keys(value.keys())
    )
    properties: JSONDocument = {}
    required: list[JSONValue] = []
    if not collapse_all:
        with closing(entries()) as children:
            for key, child in children:
                if key is None or is_dynamic_key(key):
                    continue
                properties[key] = observed_structure_schema(child, field_name=key, store=store)
                required.append(key)

    object_schema: JSONDocument = {"type": "object"}
    if properties:
        object_schema["properties"] = properties
        object_schema["required"] = required
    dynamic_values = merge_observed_structure_schemas(
        (
            observed_structure_schema(child, field_name=key, store=store)
            for key, child in entries()
            if collapse_all or key is None or is_dynamic_key(key)
        ),
        store=store,
    )
    if dynamic_values or filename_map:
        object_schema["additionalProperties"] = dynamic_values
        object_schema["x-polylogue-dynamic-keys"] = True
        if collapse_all and not filename_map:
            object_schema["x-polylogue-high-cardinality-keys"] = True
    return store(object_schema) if store is not None else object_schema


def collapse_dynamic_keys(schema: JSONDocument) -> JSONDocument:
    """Collapse dynamic key properties into additionalProperties."""
    properties = json_document(schema.get("properties"))
    if properties:
        static_props: JSONDocument = {}
        dynamic_schemas: list[JSONDocument] = []
        key_names = list(properties.keys())

        for key, value in properties.items():
            collapsed_value = collapse_dynamic_keys(json_document(value))
            if is_dynamic_key(key):
                dynamic_schemas.append(collapsed_value)
            else:
                static_props[key] = collapsed_value

        if should_collapse_observed_keys(key_names):
            schema["x-polylogue-high-cardinality-keys"] = True
            if not dynamic_schemas:
                # Only fold the individually-static properties away when the
                # per-key `is_dynamic_key` pass didn't already produce an
                # `additionalProperties` model on its own -- e.g. a wide,
                # uniformly-shaped map (256 "ordinary-key-N" names) where no
                # single key looks dynamic but the aggregate shape still
                # calls for one. When at least one key already collapsed
                # (a content-bearing key, or enough identifier-ish keys to
                # cross the dynamic-ratio threshold), that already produced
                # a correct additionalProperties block; sweeping the
                # remaining, individually-static keys into it too would
                # discard real structure `should_collapse_observed_keys`'s
                # own docstring says to preserve ("collapsing a mostly-
                # static map because one id appeared would lose real
                # structure") -- and disclosure-risk keys are never lost
                # either way, since `is_dynamic_key` already routed them.
                dynamic_schemas.extend(json_document(value) for value in static_props.values() if json_document(value))
                static_props = {}

        if dynamic_schemas:
            schema["properties"] = static_props
            schema["additionalProperties"] = merge_schemas(dynamic_schemas)
            schema["x-polylogue-dynamic-keys"] = True
            required = schema.get("required")
            if isinstance(required, list):
                schema["required"] = [item for item in required if isinstance(item, str) and item in static_props]
        else:
            schema["properties"] = static_props

    items = json_document(schema.get("items"))
    if items:
        schema["items"] = collapse_dynamic_keys(items)

    for keyword in ("anyOf", "oneOf", "allOf"):
        variants = json_document_list(schema.get(keyword))
        if variants:
            schema[keyword] = [collapse_dynamic_keys(item) for item in variants]

    additional_properties = json_document(schema.get("additionalProperties"))
    if additional_properties:
        schema["additionalProperties"] = collapse_dynamic_keys(additional_properties)

    return schema


__all__ = [
    "GENSON_AVAILABLE",
    "SchemaBuilder",
    "StructureWitnessRetention",
    "canonicalize_structure_schema",
    "collapse_dynamic_keys",
    "dynamic_object_paths",
    "is_source_structure_witness",
    "legacy_structure_schema_digest",
    "merge_schemas",
    "merge_observed_structure_schemas",
    "observed_structure_schema",
    "retain_exact_structure_witnesses",
    "structure_schema_digest",
]
