"""Canonical executable-support classification for persisted schemas."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from polylogue.schemas.json_walk import require_finite_json
from polylogue.schemas.synthetic.models import SchemaRecord
from polylogue.schemas.synthetic.observed_values import (
    format_text_shape_is_representable,
    measured_string_formats,
    observed_length_support,
)

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
        "prefixItems",
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
        "x-polylogue-time-deltas",
        "x-polylogue-numeric-order",
        "x-polylogue-values",
    }
)
_SUPPORTED_FORMAT_VALUES = frozenset(
    {
        "uuid4",
        "uuid",
        "hex-id",
        "iso8601",
        "unix-epoch",
        "unix-epoch-str",
        "url",
        "email",
        "mime-type",
        "base64",
        "persisted-tool-output",
    }
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
        "x-polylogue-authored-fixture",
        "x-polylogue-authored-record-branch",
        "x-polylogue-evidence",
        "x-polylogue-evidence-confidence",
        "x-polylogue-exact-structure-ids",
        "x-polylogue-field-first-seen",
        "x-polylogue-field-last-seen",
        "x-polylogue-generated-at",
        "x-polylogue-generator",
        "x-polylogue-high-cardinality-keys",
        "x-polylogue-observation-status",
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
        "x-polylogue-statistics-status",
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


def _valid_histogram(distribution: Mapping[str, object]) -> bool:
    """A log-bucketed histogram distribution, if it carries one, is well formed."""
    histogram = distribution.get("histogram")
    if histogram is None:
        return True
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
    if any((minimum is not None and item < minimum) or (maximum is not None and item > maximum) for item in present):
        return False
    for key in ("min", "max", "mean", "p0", "p50", "p90", "p95", "p99", "p100", "stddev"):
        if key in distribution and _number(distribution[key]) is None:
            return False
    stddev = _number(distribution.get("stddev"))
    return stddev is None or stddev >= 0


#: Distribution kinds the runtime samples, and the schema type that consumes each.
_HISTOGRAM_DISTRIBUTIONS = frozenset({"numeric", "array_length", "string_length", "newline_count", "object_fanout"})
_COUNT_DISTRIBUTIONS = frozenset({"type_counts", "boolean_counts", "co_occurring_fields"})
_CONSUMED_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "array": ("array_length",),
    "boolean": ("boolean_counts",),
    "integer": ("numeric",),
    "number": ("numeric",),
    "object": ("object_fanout",),
    "string": ("string_length", "newline_count", "categorical"),
}


def _valid_counts(value: object) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in value.values()
    )


def normalized_observed_type_counts(observed: Mapping[str, object]) -> dict[str, int] | None:
    """Include a separately declared null count only when the type map omits it."""
    types = observed.get("type_counts", {})
    if not isinstance(types, Mapping) or not _valid_counts(types):
        return None
    counts = {str(kind): count for kind, count in types.items() if isinstance(count, int)}
    nulls = observed.get("null_observations")
    if "null" not in counts and isinstance(nulls, int) and not isinstance(nulls, bool) and nulls >= 0:
        counts["null"] = nulls
    return counts


def cooccurrence_observation_count(observed: Mapping[str, object]) -> int | None:
    """Match the normalized occurrence counts, or an explicit document-only measurement."""
    if "type_counts" in observed:
        counts = normalized_observed_type_counts(observed)
        return sum(counts.values()) if counts is not None else None
    documents = observed.get("encountered_documents")
    return documents if isinstance(documents, int) and not isinstance(documents, bool) and documents >= 0 else None


def _valid_categorical(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    buckets = value.get("bucket_histogram")
    count = value.get("bucket_count")
    return (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        and isinstance(buckets, list)
        and all(
            isinstance(bucket, list)
            and len(bucket) == 2
            and all(isinstance(part, int) and not isinstance(part, bool) for part in bucket)
            and 0 <= bucket[0] < count
            and bucket[1] > 0
            for bucket in buckets
        )
    )


def _valid_observed_distribution(value: object) -> bool:
    """Every distribution kind the runtime reads is well formed; other entries are inference metadata."""
    if not isinstance(value, Mapping) or not value:
        return False
    for name, distribution in value.items():
        if name in _HISTOGRAM_DISTRIBUTIONS:
            if not isinstance(distribution, Mapping) or not _valid_histogram(distribution):
                return False
            if name != "numeric" and any(
                isinstance(item, (int, float)) and item < 0
                for key in ("min", "max")
                if (item := distribution.get(key)) is not None
            ):
                return False
        elif name in _COUNT_DISTRIBUTIONS:
            if not _valid_counts(distribution):
                return False
        elif name == "categorical" and not _valid_categorical(distribution):
            return False
    together = value.get("co_occurring_fields")
    observations = cooccurrence_observation_count(value)
    return not (
        isinstance(together, Mapping)
        and observations is not None
        and any(isinstance(count, int) and count > observations for count in together.values())
    )


def _sampleable_distribution(value: Mapping[str, object], name: str) -> bool:
    distribution = value.get(name)
    if name == "categorical":
        return _valid_categorical(distribution)
    if name in _COUNT_DISTRIBUTIONS:
        return _valid_counts(distribution) and bool(distribution)
    return isinstance(distribution, Mapping) and isinstance(distribution.get("histogram"), list)


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
    types: set[str] = set()
    pending = [node]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        schema_type = current.get("type")
        if isinstance(schema_type, str):
            types.add(schema_type)
        elif isinstance(schema_type, list):
            types.update(item for item in schema_type if isinstance(item, str))
        else:
            for keyword in ("anyOf", "oneOf"):
                variants = current.get(keyword)
                if isinstance(variants, list):
                    pending.extend(variant for variant in variants if isinstance(variant, Mapping))
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
    if key == "x-polylogue-numeric-order":
        properties = node.get("properties")
        if not isinstance(properties, dict) or not isinstance(value, list) or not value:
            return False
        seen: set[str] = set()
        for row in value:
            if not isinstance(row, dict):
                return False
            lower, upper = row.get("lower"), row.get("upper")
            if (
                not isinstance(lower, str)
                or not isinstance(upper, str)
                or (lower, upper)
                not in {("cached_input_tokens", "input_tokens"), ("reasoning_output_tokens", "output_tokens")}
                or lower in seen
                or upper in seen
            ):
                return False
            if any(
                not isinstance(properties.get(name), dict) or properties[name].get("type") not in {"integer", "number"}
                for name in (lower, upper)
            ):
                return False
            seen.update((lower, upper))
        return True
    union_path = ".anyOf[" in path or ".oneOf[" in path
    if key in {
        "x-polylogue-foreign-keys",
        "x-polylogue-time-deltas",
        "x-polylogue-mutually-exclusive",
    }:
        if key in {"x-polylogue-foreign-keys", "x-polylogue-time-deltas"}:
            return False
        return path == "$" and _annotation_supported(key, value, root_schema)
    if key == "x-polylogue-frequency":
        # A root/alias operand is already present. At properties this controls
        # presence; at union variants the same valid declaration is a weight.
        return _annotation_supported(key, value, root_schema)
    if key == "x-polylogue-array-lengths":
        return _annotation_supported(key, value, root_schema) and ("array" in schema_types or union_path)
    if key == "x-polylogue-observed-distribution":
        if not isinstance(value, Mapping) or not _annotation_supported(key, value, root_schema):
            return False
        length_distribution = value.get("string_length")
        try:
            lengths = observed_length_support(length_distribution) if isinstance(length_distribution, Mapping) else {}
            newline_distribution = value.get("newline_count")
            newlines = (
                observed_length_support(newline_distribution) if isinstance(newline_distribution, Mapping) else {}
            )
        except ValueError:
            return False
        if lengths and newlines and not any(count <= length for length in lengths for count in newlines):
            return False
        categorical = value.get("categorical")
        declared_values = node.get("x-polylogue-values")
        vocabulary_authority = (isinstance(declared_values, list) and bool(declared_values)) or node.get(
            "x-polylogue-semantic-role"
        ) == "message_role"
        if "string_formats" in value:
            try:
                partitions = measured_string_formats(node)
                for fmt, shape in partitions.items():
                    length_shape = shape.get("string_length")
                    newline_shape = shape.get("newline_count")
                    if not isinstance(length_shape, Mapping) or not isinstance(newline_shape, Mapping):
                        return False
                    lengths_for_format = observed_length_support(length_shape)
                    newlines_for_format = observed_length_support(newline_shape)
                    categories = shape.get("categories")
                    shape_rows = categories.values() if isinstance(categories, dict) else (shape,)
                    for partition in shape_rows:
                        if not isinstance(partition, Mapping):
                            return False
                        partition_lengths = partition.get("string_length")
                        partition_newlines = partition.get("newline_count")
                        if not isinstance(partition_lengths, Mapping) or not isinstance(partition_newlines, Mapping):
                            return False
                        lengths_for_format = observed_length_support(partition_lengths)
                        newlines_for_format = observed_length_support(partition_newlines) or {0: 1}
                        if (
                            not lengths_for_format
                            or not all(
                                any(
                                    format_text_shape_is_representable(fmt, length, count)
                                    for count in newlines_for_format
                                )
                                for length in lengths_for_format
                            )
                            or not all(
                                any(
                                    format_text_shape_is_representable(fmt, length, count)
                                    for length in lengths_for_format
                                )
                                for count in newlines_for_format
                            )
                        ):
                            return False
            except (TypeError, ValueError):
                return False
            # Conditional evidence owns format/shape compatibility; the
            # unchanged marginal is intentionally allowed to mix formats.
            return bool(partitions) and "string" in schema_types
        effective_format = node.get("x-polylogue-format")
        if (
            effective_format is None
            and node.get("x-polylogue-semantic-role") == "message_timestamp"
            and "string" in schema_types
        ):
            effective_format = "iso8601"
        if not vocabulary_authority and lengths == {0: sum(lengths.values())} and isinstance(categorical, Mapping):
            histogram = categorical.get("bucket_histogram")
            if isinstance(histogram, list) and len({row[0] for row in histogram if isinstance(row, list) and row}) > 1:
                return False
        if (
            not vocabulary_authority
            and lengths
            and not all(
                any(
                    format_text_shape_is_representable(effective_format, length, count)
                    for count in (newlines or {0: 1})
                )
                for length in lengths
            )
        ):
            return False
        if (
            not vocabulary_authority
            and newlines
            and not all(
                any(format_text_shape_is_representable(effective_format, length, count) for length in lengths)
                for count in newlines
            )
        ):
            return False
        consumed = {name for kind in schema_types for name in _CONSUMED_DISTRIBUTIONS.get(kind, ())}
        return (
            any(_sampleable_distribution(value, name) for name in consumed)
            or ((len(schema_types) > 1 or schema_types == {"null"}) and _sampleable_distribution(value, "type_counts"))
            or _sampleable_distribution(value, "co_occurring_fields")
        )
    if key == "x-polylogue-range":
        return (
            bool(schema_types - {"null"})
            and schema_types - {"null"} <= {"number", "integer"}
            and _annotation_supported(key, value, root_schema)
        )
    if key in {"x-polylogue-format", "x-polylogue-values", "x-polylogue-multiline"}:
        if key == "x-polylogue-format" and schema_types - {"null"} and schema_types - {"null"} <= {"number", "integer"}:
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
        return "string" in schema_types and _annotation_supported(key, value, root_schema)
    return key in _PERSISTED_SCHEMA_METADATA_ANNOTATIONS


def classify_schema_constructs(schema: object, *, _descend: bool = True) -> tuple[ConstructSupport, ...]:
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
                        pending.append((child, child_path))
                else:
                    pending.append((value, f"{path}.{key}"))
            elif key in _SCHEMA_ARRAY_KEYWORDS and isinstance(value, Sequence) and not isinstance(value, str):
                for index, child in enumerate(value):
                    pending.append((child, f"{path}.{key}[{index}]"))
            elif key == "items":
                if isinstance(value, Sequence) and not isinstance(value, str):
                    for index, child in enumerate(value):
                        pending.append((child, f"{path}.items[{index}]"))
                else:
                    pending.append((value, f"{path}.items"))
            elif key == "additionalProperties" and isinstance(value, Mapping):
                pending.append((value, f"{path}.additionalProperties"))

    if _descend:
        require_finite_json(schema)
    pending: list[tuple[object, str]] = [(schema, "$")]
    while pending:
        node, path = pending.pop()
        visit(node, path)
        if not _descend:
            break
    return tuple(ConstructSupport(construct, found[construct]) for construct in sorted(found))


def unsupported_schema_constructs(schema: object) -> tuple[str, ...]:
    """Return the canonical unsupported construct details for one schema."""

    return tuple(item.construct for item in classify_schema_constructs(schema) if item.state == "unsupported")


__all__ = [
    "ConstructSupport",
    "ConstructSupportState",
    "classify_schema_constructs",
    "cooccurrence_observation_count",
    "normalized_observed_type_counts",
    "unsupported_schema_constructs",
]
