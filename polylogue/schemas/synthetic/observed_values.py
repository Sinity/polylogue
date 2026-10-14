"""The common interpretation of inference histogram buckets."""

from __future__ import annotations

import math
from collections.abc import Generator, Mapping

from polylogue.core.tool_result_pointer import PERSISTED_OUTPUT_PREFIX, persisted_output_shape
from polylogue.schemas.field_stats.detection import FORMAT_PATTERNS, _detect_string_format
from polylogue.schemas.field_stats.distributions import CATEGORICAL_BUCKET_COUNT, _bucket_index
from polylogue.schemas.field_stats.models import PointerShapeStats
from polylogue.schemas.json_walk import require_finite_json
from polylogue.schemas.synthetic.array_value import GeneratedArray, walk_generated_values
from polylogue.schemas.synthetic.models import GeneratedRecord, SchemaRecord, SchemaValue


def histogram_value(distribution: Mapping[str, object], index: int) -> float | None:
    log_base = distribution.get("log_base")
    if not isinstance(log_base, (int, float)) or isinstance(log_base, bool) or log_base <= 1:
        return None
    value = 0.0 if index == 0 else math.expm1((abs(index) - 0.5) * math.log(float(log_base)))
    if index < 0:
        value = -value
    minimum, maximum = distribution.get("min"), distribution.get("max")
    if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        value = max(float(minimum), value)
    if isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
        value = min(float(maximum), value)
    return value


def observed_length_support(distribution: Mapping[str, object]) -> dict[int, int]:
    histogram = distribution.get("histogram")
    if not isinstance(histogram, list):
        return {}
    result: dict[int, int] = {}
    for item in histogram:
        if not isinstance(item, list) or len(item) != 2:
            continue
        index, count = item
        if not isinstance(index, int) or not isinstance(count, int) or count <= 0:
            continue
        value = histogram_value(distribution, index)
        if value is not None:
            if value < 0:
                raise ValueError("observed length or count is negative")
            length = int(round(value))
            result[length] = result.get(length, 0) + count
    return result


def format_length_is_representable(fmt: object, length: int) -> bool:
    if fmt == "persisted-tool-output":
        return length >= len(PERSISTED_OUTPUT_PREFIX) + 1
    if fmt in ("uuid", "uuid4"):
        return length == 36
    if fmt == "hex-id":
        return length >= 24
    if fmt == "iso8601":
        return length >= 19
    if fmt in ("unix-epoch", "unix-epoch-str"):
        return length == 10 or length >= 12
    if fmt == "base64":
        return length >= 40
    minimum = {"url": 8, "email": 5, "mime-type": 3}.get(fmt) if isinstance(fmt, str) else None
    return minimum is None or length >= minimum


def format_text_shape_is_representable(fmt: object, length: int, newlines: int) -> bool:
    """Use the original collector pattern's prefix or terminal-newline shape."""
    if not 0 <= newlines <= length:
        return False
    if fmt == "persisted-tool-output":
        return length - newlines >= len(PERSISTED_OUTPUT_PREFIX) + 1
    if fmt in {"url", "iso8601"}:
        return format_length_is_representable(fmt, length - newlines)
    if fmt:
        return newlines <= 1 and format_length_is_representable(fmt, length - newlines)
    return True


def _format_accepts(fmt: str, value: str) -> bool:
    # Format annotations come from these same declared patterns. A prefix
    # classifier is not a stricter datetime/base64 decoder, and Python's $
    # anchor permits one final newline. Alias proof preserves that contract.
    lookup = "unix-epoch-str" if fmt == "unix-epoch" else fmt
    return any(pattern.search(value) is not None for name, pattern in FORMAT_PATTERNS if name == lookup)


def _format_record(value: object) -> SchemaRecord:
    return value if isinstance(value, dict) else {}


class _ProjectedStringShape(dict[str, SchemaValue]):
    """Transient draw support borrowing its original measured coverage."""

    original: SchemaRecord

    def __init__(self, original: SchemaRecord, values: SchemaRecord) -> None:
        super().__init__(values)
        self.original = original.original if isinstance(original, _ProjectedStringShape) else original


def measured_character_shapes(shape: Mapping[str, object], count: int) -> dict[str, SchemaRecord]:
    """Validate the collector's fixed, anonymous character partitions."""
    if isinstance(shape, _ProjectedStringShape):
        original_count = _format_record(shape.original.get("string_length")).get("count")
        if type(original_count) is not int:
            raise ValueError("projected character shape has no original observation count")
        original = measured_character_shapes(shape.original, original_count)
        rows = _format_record(shape.get("character_shapes"))
        if not rows or not set(rows) <= set(original):
            raise ValueError("projected character draw leaves its original support")
        result: dict[str, SchemaRecord] = {}
        for name, value in rows.items():
            record = _format_record(value)
            draw_count = _format_record(record.get("string_length")).get("count")
            if type(draw_count) is not int or draw_count <= 0:
                raise ValueError("projected character draw has no supported mass")
            if record.get("newline_count") != original[name].get("newline_count"):
                raise ValueError("projected character draw changed original newline evidence")
            result[name] = record
        return result
    rows = _format_record(shape.get("character_shapes"))
    if not rows or not set(rows) <= {"ascii", "non_ascii"}:
        raise ValueError("measured string character evidence requires complete reinference")
    total = 0
    for row in rows.values():
        record = _format_record(row)
        n = _format_record(record.get("string_length")).get("count")
        if type(n) is not int or n <= 0 or _format_record(record.get("newline_count")).get("count") != n:
            raise ValueError("measured string character counts disagree")
        total += n
    if total != count:
        raise ValueError("measured string character partitions do not cover the field")
    return {name: _format_record(row) for name, row in rows.items()}


def measured_string_formats(schema: Mapping[str, object]) -> dict[str, SchemaRecord]:
    """Validate collector-owned format/category partitions for every consumer."""
    observed = _format_record(schema.get("x-polylogue-observed-distribution"))
    if "string_formats" not in observed:
        length_count = _format_record(observed.get("string_length")).get("count")
        if type(length_count) is int and length_count > 0:
            raise ValueError("measured string formats require complete reinference")
        return {}
    shapes = _format_record(observed["string_formats"])
    if not shapes:
        raise ValueError("measured string format partitions are empty")
    counts: dict[str, int] = {}
    for fmt, value in shapes.items():
        if fmt and fmt not in {name for name, _pattern in FORMAT_PATTERNS}:
            raise ValueError("measured string format partition names an unknown format")
        shape = _format_record(value)
        length = _format_record(shape.get("string_length"))
        count = length.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("measured string format partition has no observations")
        if any(
            type(_format_record(shape.get(name)).get("count")) is not int
            or _format_record(shape.get(name)).get("count") != count
            for name in ("newline_count", "categorical")
        ):
            raise ValueError("measured string format partition counts disagree")
        measured_character_shapes(shape, count)
        categories = _format_record(shape.get("categories"))
        categorical = _format_record(shape.get("categorical"))
        histogram = categorical.get("bucket_histogram")
        if not isinstance(histogram, list):
            raise ValueError("measured string format category histogram is missing")
        expected: dict[str, int] = {}
        for row in histogram:
            if (
                not isinstance(row, list)
                or len(row) != 2
                or type(row[0]) is not int
                or not 0 <= row[0] < CATEGORICAL_BUCKET_COUNT
                or type(row[1]) is not int
                or row[1] <= 0
                or str(row[0]) in expected
            ):
                raise ValueError("measured string format category histogram is malformed")
            expected[str(row[0])] = row[1]
        if set(categories) != set(expected) or sum(expected.values()) != count:
            raise ValueError("measured string format categories require complete reinference")
        for bucket, category_count in expected.items():
            category = _format_record(categories[bucket])
            measured_character_shapes(category, category_count)
            if any(
                type(_format_record(category.get(name)).get("count")) is not int
                or _format_record(category.get(name)).get("count") != category_count
                for name in ("string_length", "newline_count")
            ):
                raise ValueError("measured string format category counts disagree")
        if fmt == "persisted-tool-output":
            for bucket, category_count in expected.items():
                joint = PointerShapeStats.from_payload(_format_record(categories[bucket]).get("pointer_shapes"))
                if sum(count for _point, count in joint.bins.values()) != category_count:
                    raise ValueError("pointer shape joint requires complete reinference")
        counts[fmt] = count
    if type(_format_record(observed.get("string_length")).get("count")) is not int or sum(
        counts.values()
    ) != _format_record(observed.get("string_length")).get("count"):
        raise ValueError("measured string format partitions do not cover the field")
    return {fmt: _format_record(value) for fmt, value in shapes.items()}


def _observed(schema: Mapping[str, object], name: str) -> Mapping[str, object]:
    observed = schema.get("x-polylogue-observed-distribution")
    result = observed.get(name) if isinstance(observed, Mapping) else None
    return result if isinstance(result, Mapping) else {}


def _existing_value_frame(
    schema: Mapping[str, object], value: object, *, string_partition_proven: bool = False
) -> Generator[tuple[Mapping[str, object], object], bool, bool]:
    if (
        not string_partition_proven
        and isinstance(value, str)
        and not schema.get("x-polylogue-values")
        and schema.get("x-polylogue-semantic-role") != "message_role"
    ):
        partitions = measured_string_formats(schema)
        if partitions:
            partition_format = _detect_string_format(value) or ""
            shape = partitions.get(partition_format)
            if shape is None:
                return False
            categories = _format_record(shape.get("categories"))
            pointer_shape = persisted_output_shape(value) if partition_format == "persisted-tool-output" else None
            matching = [
                category
                for category in categories.values()
                if (
                    pointer_shape is not None
                    and tuple(_bucket_index(float(item)) for item in pointer_shape)
                    in PointerShapeStats.from_payload(_format_record(category).get("pointer_shapes")).bins
                )
                or (
                    pointer_shape is None
                    and len(value)
                    in observed_length_support(_format_record(_format_record(category).get("string_length")))
                    and value.count("\n")
                    in observed_length_support(_format_record(_format_record(category).get("newline_count")))
                )
            ]
            if not matching:
                return False
            observed = _format_record(schema.get("x-polylogue-observed-distribution"))
            conditioned = {
                **schema,
                "x-polylogue-observed-distribution": {**observed, **shape, **_format_record(matching[0])},
            }
            if partition_format:
                conditioned["x-polylogue-format"] = partition_format
            else:
                conditioned.pop("x-polylogue-format", None)
                if conditioned.get("x-polylogue-semantic-role") == "message_timestamp":
                    conditioned.pop("x-polylogue-semantic-role")
            # The original partition has been proven; evaluate this exact shape
            # without re-entering the partition selection on the next frame.
            conditioned_observed = _format_record(conditioned["x-polylogue-observed-distribution"])
            conditioned_observed.pop("string_formats", None)
            if pointer_shape is not None:
                conditioned_observed.pop("string_length", None)
                conditioned_observed.pop("newline_count", None)
            return (yield from _existing_value_frame(conditioned, value, string_partition_proven=True))
    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if keyword in schema and (not isinstance(variants, list) or not variants):
            raise ValueError("alias schema union is unsupported")
        if isinstance(variants, list) and variants:
            successes = 0
            for variant in variants:
                if not isinstance(variant, Mapping):
                    raise ValueError("alias schema union contains an unsupported branch")
                inherited = {key: item for key, item in schema.items() if key.startswith("x-polylogue-")}
                accepted = yield ({**inherited, **variant}, value)
                successes += int(accepted)
            if (keyword == "anyOf" and successes == 0) or (keyword == "oneOf" and successes != 1):
                return False
    declared = schema.get("type")
    kinds = declared if isinstance(declared, list) else [declared] if isinstance(declared, str) else []
    if "type" in schema and (
        not kinds
        or any(kind not in {"null", "boolean", "string", "array", "object", "integer", "number"} for kind in kinds)
    ):
        raise ValueError("alias schema type is unsupported")
    kind = (
        "null"
        if value is None
        else "boolean"
        if isinstance(value, bool)
        else "string"
        if isinstance(value, str)
        else "array"
        if isinstance(value, list | GeneratedArray)
        else "object"
        if isinstance(value, dict | GeneratedRecord)
        else "integer"
        if isinstance(value, int)
        else "number"
        if isinstance(value, float)
        else "unsupported"
    )
    if kind == "unsupported" or (kinds and kind not in kinds and not (kind == "integer" and "number" in kinds)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    _require_active_alias_support(schema, kind)
    labels = schema.get("x-polylogue-values")
    if isinstance(value, str) and isinstance(labels, list) and labels:
        return value in labels
    if isinstance(value, str):
        fmt = schema.get("x-polylogue-format")
        if fmt is None and schema.get("x-polylogue-semantic-role") == "message_timestamp":
            fmt = "iso8601"
        if isinstance(fmt, str) and not _format_accepts(fmt, value):
            return False
        # Fixed role vocabularies take precedence over unrelated observed
        # lengths, exactly as the canonical scalar owner does.
        if schema.get("x-polylogue-semantic-role") != "message_role":
            lengths = observed_length_support(_observed(schema, "string_length"))
            newlines = observed_length_support(_observed(schema, "newline_count"))
            if (lengths and len(value) not in lengths) or (newlines and value.count("\n") not in newlines):
                return False
    elif isinstance(value, list | GeneratedArray):
        lengths = observed_length_support(_observed(schema, "array_length"))
        fallback = schema.get("x-polylogue-array-lengths")
        if lengths and len(value) not in lengths:
            return False
        if (
            not lengths
            and isinstance(fallback, list)
            and len(fallback) >= 2
            and not fallback[0] <= len(value) <= fallback[1]
        ):
            return False
        prefix = schema.get("prefixItems", [])
        if not isinstance(prefix, list) or any(not isinstance(item, Mapping) for item in prefix):
            raise ValueError("alias array prefix schemas are unsupported")
        items = schema.get("items")
        for ordinal, item in enumerate(value):
            child = prefix[ordinal] if ordinal < len(prefix) else items
            if child is False:
                return False
            if isinstance(child, Mapping):
                if not (yield (child, item)):
                    return False
            elif child is None or child is True:
                if not (yield ({}, item)):
                    return False
            else:
                raise ValueError("alias schema items are not supported")
        ordering = _observed(items, "ordered_numeric_pairs") if isinstance(items, Mapping) else {}
        if (
            not prefix
            and ordering.get("nondecreasing_rate") == 1
            and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
            and value != sorted(value)
        ):
            return False
    elif isinstance(value, dict | GeneratedRecord):
        properties = schema.get("properties")
        if "properties" in schema and not isinstance(properties, Mapping):
            raise ValueError("alias schema properties are unsupported")
        properties = properties if isinstance(properties, Mapping) else {}
        if any(not isinstance(key, str) for key in value):
            return False
        additional = schema.get("additionalProperties")
        if isinstance(additional, Mapping) and not properties:
            fanout = observed_length_support(_observed(schema, "object_fanout"))
            if fanout and len(value) not in fanout:
                return False
        for key, item in value.items():
            child = properties.get(key, additional)
            if child is False:
                return False
            if isinstance(child, Mapping) and not (yield (child, item)):
                return False
            if (child is None or child is True) and not (yield ({}, item)):
                return False
            if child is not None and child is not True and not isinstance(child, Mapping):
                raise ValueError("alias schema property is not supported")
    elif isinstance(value, bool):
        counts = _observed(schema, "boolean_counts")
        count = counts.get("true" if value else "false")
        if any(isinstance(item, int) and item > 0 for item in counts.values()) and not (
            isinstance(count, int) and count > 0
        ):
            return False
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        distribution = _observed(schema, "numeric")
        histogram = distribution.get("histogram")
        support: list[int | float] = []
        if isinstance(histogram, list):
            for item in histogram:
                if (
                    isinstance(item, list)
                    and len(item) == 2
                    and isinstance(item[0], int)
                    and isinstance(item[1], int)
                    and item[1] > 0
                ):
                    sampled = histogram_value(distribution, item[0])
                    if sampled is not None:
                        support.append(int(sampled) if kind == "integer" and "integer" in kinds else sampled)
        if support and value not in support:
            return False
        fallback = schema.get("x-polylogue-range")
        if (
            not support
            and isinstance(fallback, list)
            and len(fallback) >= 2
            and not fallback[0] <= value <= fallback[1]
        ):
            return False
    return True


_ALIAS_TYPED_KEYWORDS = {
    "object": frozenset(
        {
            "properties",
            "additionalProperties",
            "required",
            "patternProperties",
            "dependentRequired",
            "dependentSchemas",
            "dependencies",
            "minProperties",
            "maxProperties",
            "propertyNames",
            "unevaluatedProperties",
            "x-polylogue-foreign-keys",
            "x-polylogue-time-deltas",
            "x-polylogue-mutually-exclusive",
            "x-polylogue-numeric-order",
        }
    ),
    "array": frozenset(
        {
            "items",
            "prefixItems",
            "contains",
            "minContains",
            "maxContains",
            "minItems",
            "maxItems",
            "uniqueItems",
            "unevaluatedItems",
            "x-polylogue-array-lengths",
        }
    ),
    "string": frozenset(
        {
            "pattern",
            "format",
            "minLength",
            "maxLength",
            "contentEncoding",
            "contentMediaType",
            "contentSchema",
            "x-polylogue-ref",
            "x-polylogue-values",
            "x-polylogue-multiline",
        }
    ),
    "number": frozenset(
        {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "x-polylogue-range"}
    ),
}


def _require_active_alias_support(schema: Mapping[str, object], kind: str) -> None:
    """Check the current selected value, not inactive descendant schemas.

    This does not grant relational custody or generation support. The complete
    schema was validated before entering the iterative value proof; the same
    classifier now checks only this applicable node. Child frames check their
    own actual values and union branches keep their own applicability.
    """
    from polylogue.schemas.synthetic.classification import classify_schema_constructs

    selected_kind = "number" if kind == "integer" else kind
    inactive = set().union(*(keys for declared, keys in _ALIAS_TYPED_KEYWORDS.items() if declared != selected_kind))
    # String and numeric format annotations describe no null/object/array draw.
    if kind not in {"string", "number", "integer"}:
        inactive.update({"x-polylogue-format", "x-polylogue-semantic-role"})
    active = {key: item for key, item in schema.items() if key not in inactive}
    unsupported = tuple(
        row.construct for row in classify_schema_constructs(active, _descend=False) if row.state == "unsupported"
    )
    if unsupported:
        raise ValueError(f"alias reuse cannot prove unsupported constructs: {unsupported}")
    if any(
        key in active
        for key in (
            "x-polylogue-foreign-keys",
            "x-polylogue-time-deltas",
            "x-polylogue-mutually-exclusive",
            "x-polylogue-numeric-order",
        )
    ):
        raise ValueError("alias reuse requires unavailable relational solver custody")


def existing_value_is_supported(schema: Mapping[str, object], value: object) -> bool:
    """Prove deterministic alias reuse under the canonical support policy.

    Density marginals do not identify a joint alias distribution. The existing
    owner samples it; the missing alias is conditioned on that sampled value.
    Relations need their own solver custody and cannot be proven here.
    """
    require_finite_json(schema)
    for _ in walk_generated_values(value):
        pass
    frames = [_existing_value_frame(schema, value)]
    result: bool | None = None
    try:
        while frames:
            try:
                child_schema, child_value = next(frames[-1]) if result is None else frames[-1].send(result)
            except StopIteration as completed:
                frames.pop()
                result = completed.value
            else:
                frames.append(_existing_value_frame(child_schema, child_value))
                result = None
        return bool(result)
    finally:
        for frame in reversed(frames):
            frame.close()


def neutral_numeric_range(schema: Mapping[str, object]) -> tuple[float, float]:
    """The original scalar sampler's public range when magnitudes are absent.

    This is authored neutral support, never a projection of private observations.
    The relation owner may condition it just as it conditions an explicit range.
    Semantic timestamp proposals remain owned by their separate semantic draw.
    """
    declared = schema.get("x-polylogue-range")
    if isinstance(declared, list) and len(declared) >= 2:
        return float(declared[0]), float(declared[1])
    if schema.get("x-polylogue-format") == "unix-epoch":
        return 1670000000.0, 1760000000.0
    return 0.0, 1000.0
