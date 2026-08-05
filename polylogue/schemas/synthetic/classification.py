"""Canonical executable-support classification for persisted schemas."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from polylogue.schemas.synthetic.models import SchemaRecord

ConstructSupportState = Literal["supported", "unsupported"]


@dataclass(frozen=True, order=True)
class ConstructSupport:
    """Support verdict for one schema keyword or runtime annotation."""

    construct: str
    state: ConstructSupportState

    def to_payload(self) -> dict[str, str]:
        return {"construct": self.construct, "state": self.state}


_SUPPORTED_SCHEMA_CONSTRUCTS = frozenset(
    {
        "$anchor",
        "$comment",
        "$id",
        "$schema",
        "additionalProperties",
        "anyOf",
        "default",
        "deprecated",
        "description",
        "examples",
        "items",
        "oneOf",
        "properties",
        "readOnly",
        "title",
        "type",
        "writeOnly",
    }
)
_STANDARD_SCHEMA_KEYWORDS = frozenset(
    {
        "$anchor",
        "$comment",
        "$defs",
        "$dynamicAnchor",
        "$dynamicRef",
        "$id",
        "$recursiveAnchor",
        "$recursiveRef",
        "$ref",
        "$schema",
        "$vocabulary",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "contains",
        "contentEncoding",
        "contentMediaType",
        "contentSchema",
        "default",
        "definitions",
        "dependentRequired",
        "dependentSchemas",
        "dependencies",
        "deprecated",
        "description",
        "else",
        "enum",
        "examples",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "formatAssertion",
        "if",
        "items",
        "maxContains",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minContains",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "multipleOf",
        "not",
        "oneOf",
        "pattern",
        "patternProperties",
        "prefixItems",
        "properties",
        "propertyNames",
        "readOnly",
        "required",
        "then",
        "title",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
        "writeOnly",
    }
)
_SCHEMA_MAPPING_KEYWORDS = frozenset(
    {
        "$defs",
        "additionalProperties",
        "contentSchema",
        "contains",
        "dependentSchemas",
        "dependencies",
        "else",
        "if",
        "items",
        "not",
        "patternProperties",
        "properties",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SCHEMA_ARRAY_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SUPPORTED_SYNTHETIC_ANNOTATIONS = frozenset(
    {
        "x-polylogue-array-lengths",
        "x-polylogue-foreign-keys",
        "x-polylogue-format",
        "x-polylogue-frequency",
        "x-polylogue-multiline",
        "x-polylogue-mutually-exclusive",
        "x-polylogue-observed-distribution",
        "x-polylogue-range",
        "x-polylogue-semantic-role",
        "x-polylogue-string-lengths",
        "x-polylogue-time-deltas",
        "x-polylogue-values",
    }
)
_SUPPORTED_FORMAT_VALUES = frozenset(
    {"uuid4", "uuid", "hex-id", "iso8601", "unix-epoch", "unix-epoch-str", "url", "email", "mime-type", "base64"}
)
_SUPPORTED_SEMANTIC_ROLE_VALUES = frozenset({"message_role", "message_body", "message_timestamp", "session_title"})
_PERSISTED_SCHEMA_METADATA_ANNOTATIONS = frozenset(
    {
        "x-polylogue-anchor-profile-family-id",
        "x-polylogue-artifact-kind",
        "x-polylogue-element-bundle-scope-count",
        "x-polylogue-element-first-seen",
        "x-polylogue-element-kind",
        "x-polylogue-element-last-seen",
        "x-polylogue-evidence",
        "x-polylogue-evidence-confidence",
        "x-polylogue-exact-structure-ids",
        "x-polylogue-generated-at",
        "x-polylogue-generator",
        "x-polylogue-high-cardinality-keys",
        "x-polylogue-observed-artifact-count",
        "x-polylogue-package-profile-family-ids",
        "x-polylogue-package-version",
        "x-polylogue-profile-family-ids",
        "x-polylogue-profile-tokens",
        "x-polylogue-promoted-at",
        "x-polylogue-registered-at",
        "x-polylogue-sample-count",
        "x-polylogue-sample-granularity",
        "x-polylogue-score",
        "x-polylogue-version",
    }
)
_SCHEMA_PATH_SEGMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def _number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(float(value)) else None
    except (OverflowError, ValueError):
        return None


def _valid_pair(value: object, *, integral: bool = False, nonnegative: bool = False) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    first, second = (_number(item) for item in value)
    if first is None or second is None:
        return False
    return not (
        (integral and (not isinstance(first, int) or not isinstance(second, int)))
        or (nonnegative and (first < 0 or second < 0))
        or first > second
    )


def _relation_records(value: object, required: tuple[str, ...]) -> list[dict[str, object]] | None:
    if not isinstance(value, list) or not value:
        return None
    records: list[dict[str, object]] = []
    for record in value:
        if not isinstance(record, dict) or not all(
            isinstance(record.get(field), str) and record[field].strip() for field in required
        ):
            return None
        records.append(record)
    return records


def _histogram_bucket_is_safe(index: int, log_base: int | float) -> bool:
    try:
        sampled_value = math.expm1((abs(float(index)) - 0.5) * math.log(float(log_base)))
    except (OverflowError, ValueError):
        return False
    return math.isfinite(sampled_value)


def _valid_observed_distribution(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    for distribution in value.values():
        if not isinstance(distribution, Mapping):
            return False
        histogram = distribution.get("histogram")
        log_base = _number(distribution.get("log_base"))
        if not isinstance(histogram, list) or not histogram or log_base is None or log_base <= 1:
            return False
        if not all(
            isinstance(bucket, list)
            and len(bucket) == 2
            and isinstance(bucket[0], int)
            and not isinstance(bucket[0], bool)
            and isinstance(bucket[1], int)
            and not isinstance(bucket[1], bool)
            and bucket[1] > 0
            and _histogram_bucket_is_safe(bucket[0], float(log_base))
            for bucket in histogram
        ):
            return False
        values = tuple(_number(distribution.get(key)) for key in ("p0", "p50", "p90", "p95", "p99", "p100"))
        present = tuple(item for item in values if item is not None)
        minimum = _number(distribution.get("min"))
        maximum = _number(distribution.get("max"))
        if minimum is not None and maximum is not None and minimum > maximum:
            return False
        if any(left > right for left, right in zip(present, present[1:], strict=False)):
            return False
        if any(
            (minimum is not None and item < minimum) or (maximum is not None and item > maximum) for item in present
        ):
            return False
        for key in ("min", "max", "mean", "p0", "p50", "p90", "p95", "p99", "p100", "stddev"):
            if key in distribution and _number(distribution[key]) is None:
                return False
        stddev = _number(distribution.get("stddev"))
        if stddev is not None and stddev < 0:
            return False
    return True


def _schema_nodes_at_path(schema: SchemaRecord, path: object) -> tuple[SchemaRecord, ...]:
    if not isinstance(path, str) or not path.startswith("$"):
        return ()
    if path == "$":
        return (schema,)
    if not path.startswith("$."):
        return ()
    nodes: tuple[SchemaRecord, ...] = (schema,)
    for raw_segment in path[2:].split("."):
        wants_items = raw_segment.endswith("[*]")
        segment = raw_segment[:-3] if wants_items else raw_segment
        if not _SCHEMA_PATH_SEGMENT.fullmatch(segment):
            return ()
        next_nodes: list[SchemaRecord] = []
        for node in nodes:
            variants = [node]
            for branch_key in ("anyOf", "oneOf"):
                branches = node.get(branch_key)
                if isinstance(branches, list):
                    variants.extend(branch for branch in branches if isinstance(branch, dict))
            for variant in variants:
                properties = variant.get("properties")
                child = properties.get(segment) if isinstance(properties, dict) else None
                if not isinstance(child, dict):
                    continue
                if not wants_items:
                    next_nodes.append(child)
                    continue
                child_variants = [child]
                for branch_key in ("anyOf", "oneOf"):
                    branches = child.get(branch_key)
                    if isinstance(branches, list):
                        child_variants.extend(branch for branch in branches if isinstance(branch, dict))
                next_nodes.extend(
                    item_schema
                    for child_variant in child_variants
                    if isinstance(item_schema := child_variant.get("items"), dict)
                )
        nodes = tuple(next_nodes)
        if not nodes:
            return ()
    return nodes


def _schema_types(node: Mapping[str, object]) -> set[str]:
    schema_type = node.get("type")
    if isinstance(schema_type, str):
        return {schema_type}
    if isinstance(schema_type, list):
        return {item for item in schema_type if isinstance(item, str)}
    types: set[str] = set()
    for keyword in ("anyOf", "oneOf"):
        variants = node.get(keyword)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, Mapping):
                    types.update(_schema_types(variant))
    return types


def _schema_type(node: Mapping[str, object]) -> str | None:
    schema_type = node.get("type")
    if isinstance(schema_type, str):
        return schema_type
    if isinstance(schema_type, list):
        types = [item for item in schema_type if isinstance(item, str) and item != "null"]
        return types[0] if len(types) == 1 else None
    return None


def _schema_nodes_have_type(nodes: tuple[SchemaRecord, ...], allowed: set[str]) -> bool:
    return bool(nodes) and all(bool(types := _schema_types(node)) and types <= allowed for node in nodes)


def _annotation_supported(key: str, value: object, root_schema: SchemaRecord) -> bool:
    if key not in _SUPPORTED_SYNTHETIC_ANNOTATIONS:
        return False
    if key == "x-polylogue-format":
        return isinstance(value, str) and value in _SUPPORTED_FORMAT_VALUES
    if key == "x-polylogue-semantic-role":
        return isinstance(value, str) and value in _SUPPORTED_SEMANTIC_ROLE_VALUES
    if key == "x-polylogue-frequency":
        frequency = _number(value)
        return frequency is not None and 0 <= frequency <= 1
    if key == "x-polylogue-multiline":
        return isinstance(value, bool)
    if key == "x-polylogue-values":
        return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item for item in value)
    if key == "x-polylogue-range":
        return _valid_pair(value)
    if key == "x-polylogue-array-lengths":
        return _valid_pair(value, integral=True, nonnegative=True)
    if key == "x-polylogue-observed-distribution":
        return _valid_observed_distribution(value)
    required = {
        "x-polylogue-foreign-keys": ("source", "target"),
        "x-polylogue-time-deltas": ("field_a", "field_b"),
        "x-polylogue-mutually-exclusive": ("parent",),
        "x-polylogue-string-lengths": ("path",),
    }.get(key)
    if required is None:
        return False
    records = _relation_records(value, required)
    if records is None:
        return False
    if key == "x-polylogue-foreign-keys":
        return all(
            _schema_nodes_at_path(root_schema, record["source"])
            and _schema_nodes_at_path(root_schema, record["target"])
            for record in records
        )
    if key == "x-polylogue-time-deltas":
        return all(
            _schema_nodes_have_type(
                _schema_nodes_at_path(root_schema, record["field_a"]), {"string", "number", "integer"}
            )
            and _schema_nodes_have_type(
                _schema_nodes_at_path(root_schema, record["field_b"]), {"string", "number", "integer"}
            )
            and _valid_pair([record.get("min_delta"), record.get("max_delta")])
            and _number(record.get("avg_delta")) is not None
            for record in records
        )
    if key == "x-polylogue-mutually-exclusive":
        return all(
            isinstance(fields := record.get("fields"), list)
            and len(fields) >= 2
            and all(
                isinstance(field, str)
                and _schema_nodes_have_type(
                    _schema_nodes_at_path(root_schema, f"{record['parent']}.{field}"),
                    {"string", "number", "integer", "boolean", "array", "object", "null"},
                )
                for field in fields
            )
            for record in records
        )
    return all(
        _schema_nodes_at_path(root_schema, record["path"])
        and any("string" in _schema_types(node) for node in _schema_nodes_at_path(root_schema, record["path"]))
        and _valid_pair([record.get("min"), record.get("max")], integral=True, nonnegative=True)
        and _number(record.get("avg")) is not None
        and (stddev := _number(record.get("stddev"))) is not None
        and stddev >= 0
        for record in records
    )


def _annotation_is_supported_at_node(
    key: str, value: object, node: Mapping[str, object], path: str, root_schema: SchemaRecord
) -> bool:
    schema_type = _schema_type(node)
    schema_types = _schema_types(node)
    union_path = ".anyOf[" in path or ".oneOf[" in path
    if key in {
        "x-polylogue-foreign-keys",
        "x-polylogue-time-deltas",
        "x-polylogue-mutually-exclusive",
        "x-polylogue-string-lengths",
    }:
        if key in {"x-polylogue-foreign-keys", "x-polylogue-time-deltas"}:
            return False
        return path == "$" and _annotation_supported(key, value, root_schema)
    if key == "x-polylogue-frequency":
        return path != "$" and _annotation_supported(key, value, root_schema)
    if key == "x-polylogue-array-lengths":
        return _annotation_supported(key, value, root_schema) and ("array" in schema_types or union_path)
    if key == "x-polylogue-observed-distribution":
        if schema_type not in {"array", "number", "integer"} or not isinstance(value, Mapping):
            return False
        expected_distribution = "array_length" if schema_type == "array" else "numeric"
        return expected_distribution in value and _annotation_supported(key, value, root_schema)
    if key == "x-polylogue-range":
        return schema_type in {"number", "integer"} and _annotation_supported(key, value, root_schema)
    if key in {"x-polylogue-format", "x-polylogue-values", "x-polylogue-multiline"}:
        if key == "x-polylogue-format" and schema_type in {"number", "integer"}:
            return value == "unix-epoch"
        return _annotation_supported(key, value, root_schema) and (
            schema_type == "string" or "string" in schema_types or union_path
        )
    if key == "x-polylogue-semantic-role":
        role = value if isinstance(value, str) else None
        if role == "message_timestamp":
            return bool(schema_types & {"string", "number", "integer"}) and _annotation_supported(
                key, value, root_schema
            )
        if role == "message_container":
            return "object" in schema_types
        return schema_type == "string" and _annotation_supported(key, value, root_schema)
    return key in _PERSISTED_SCHEMA_METADATA_ANNOTATIONS


def classify_schema_constructs(schema: object) -> tuple[ConstructSupport, ...]:
    """Classify exactly what the production synthetic runtime can consume."""

    found: dict[str, ConstructSupportState] = {}

    def record(key: str, state: ConstructSupportState) -> None:
        if found.get(key) == "unsupported":
            return
        found[key] = state

    def visit(node: object, path: str = "$") -> None:
        if not isinstance(node, Mapping):
            return
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            if key.startswith("x-"):
                if key not in _PERSISTED_SCHEMA_METADATA_ANNOTATIONS:
                    record(
                        key,
                        "supported"
                        if _annotation_is_supported_at_node(
                            key, value, node, path, schema if isinstance(schema, dict) else {}
                        )
                        else "unsupported",
                    )
                continue
            record(key, "supported" if key in _SUPPORTED_SCHEMA_CONSTRUCTS else "unsupported")
            if key in _SCHEMA_MAPPING_KEYWORDS and isinstance(value, Mapping):
                if key in {"$defs", "dependentSchemas", "dependencies", "patternProperties", "properties"}:
                    for child_name, child in value.items():
                        child_path = f"{path}.{child_name}" if key == "properties" else f"{path}.{key}.{child_name}"
                        visit(child, child_path)
                else:
                    visit(value, f"{path}.{key}")
            elif key in _SCHEMA_ARRAY_KEYWORDS and isinstance(value, Sequence) and not isinstance(value, str):
                for index, child in enumerate(value):
                    visit(child, f"{path}.{key}[{index}]")
            elif key == "items":
                if isinstance(value, Sequence) and not isinstance(value, str):
                    for index, child in enumerate(value):
                        visit(child, f"{path}.items[{index}]")
                else:
                    visit(value, f"{path}.items")
            elif key == "additionalProperties" and isinstance(value, Mapping):
                visit(value, f"{path}.additionalProperties")

    visit(schema)
    return tuple(ConstructSupport(construct, found[construct]) for construct in sorted(found))


def unsupported_schema_constructs(schema: object) -> tuple[str, ...]:
    """Return the canonical unsupported construct details for one schema."""

    return tuple(item.construct for item in classify_schema_constructs(schema) if item.state == "unsupported")


__all__ = ["ConstructSupport", "ConstructSupportState", "classify_schema_constructs", "unsupported_schema_constructs"]
