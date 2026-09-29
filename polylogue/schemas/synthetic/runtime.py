"""Recursive schema-to-data generation helpers for synthetic corpora."""

from __future__ import annotations

import json
import math
import random
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, TypeAlias

from polylogue.archive.raw_payload.decode import JSONValue
from polylogue.schemas.synthetic.build_wire_formats import normalize_browser_capture_attachments, validate_wire_payload
from polylogue.schemas.synthetic.conservation import COVERAGE_EXTRA_KEY
from polylogue.schemas.synthetic.models import SchemaRecord, SchemaValue
from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator, _text_for_role
from polylogue.schemas.synthetic.wire_formats import WireFormat

if TYPE_CHECKING:
    from polylogue.schemas.synthetic.relations import RelationConstraintSolver

SyntheticRecord: TypeAlias = dict[str, JSONValue]

# The receipt reads this registry and the recursive generator gates dispatch on
# it. Removing a construct handler therefore changes both generated behavior
# and the support receipt instead of becoming a silent ``None`` fallback.
SCHEMA_CONSTRUCT_HANDLERS: dict[str, str] = {
    "anyOf": "schema-union",
    "array": "_generate_array",
    "boolean": "boolean",
    "integer": "_generate_number",
    "null": "null",
    "number": "_generate_number",
    "object": "_generate_object",
    "oneOf": "schema-union",
    "string": "_generate_string",
}


class _SyntheticRuntimeContext(Protocol):
    provider: str
    wire_format: WireFormat
    _semantic_gen: SemanticValueGenerator | None
    _relation_solver: RelationConstraintSolver
    _active_profile_tokens: tuple[str, ...]
    _active_record_bucket: tuple[str, str] | None
    _categorical_pools: dict[str, list[tuple[str, int]]]
    _coverage_branch_choices: dict[str, int]
    _coverage_type_choices: dict[str, str]
    _coverage_null_paths: set[str]
    _coverage_witness_mode: bool

    def _generate_from_schema(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        skip_keys: set[str] | None = None,
        depth: int = 0,
        max_depth: int = 6,
        path: str = "$",
    ) -> JSONValue: ...

    def _generate_object(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        skip_keys: set[str] | None = None,
        depth: int = 0,
        max_depth: int = 6,
        path: str = "$",
    ) -> SyntheticRecord: ...

    def _generate_array(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        depth: int = 0,
        max_depth: int = 6,
        path: str = "$",
    ) -> list[JSONValue]: ...

    def _generate_string(self, schema: SchemaRecord, rng: random.Random, *, path: str = "$") -> str: ...

    def _generate_number(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        is_int: bool = False,
    ) -> float | int: ...


def _schema_record(value: SchemaValue | object) -> SchemaRecord:
    return value if isinstance(value, dict) else {}


def _schema_records(value: SchemaValue | object) -> list[SchemaRecord]:
    if not isinstance(value, list):
        return []
    return [record for item in value if (record := _schema_record(item))]


def _schema_type(self: _SyntheticRuntimeContext, schema: SchemaRecord, rng: random.Random, path: str) -> str | None:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return schema_type
    if isinstance(schema_type, list):
        selected_type = self._coverage_type_choices.get(path) if self._coverage_witness_mode else None
        if selected_type in schema_type:
            return selected_type
        if self._coverage_witness_mode and path in self._coverage_null_paths and "null" in schema_type:
            return "null"
        declared = [item for item in schema_type if isinstance(item, str)]
        observed = _observed_type_weights(schema)
        weighted = [(item, observed.get(item, 0)) for item in declared]
        if any(weight for _, weight in weighted):
            return rng.choices([item for item, _ in weighted], weights=[weight for _, weight in weighted], k=1)[0]
        non_null = [item for item in declared if item != "null"]
        return rng.choice(non_null) if non_null else "null"
    return None


def _coerce_float(value: SchemaValue, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return default


def _coerce_int(value: SchemaValue, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return default


def _observed_distribution(schema: SchemaRecord, name: str) -> SchemaRecord:
    observed = _schema_record(schema.get("x-polylogue-observed-distribution"))
    return _schema_record(observed.get(name))


def _sample_observed_distribution(
    schema: SchemaRecord,
    name: str,
    rng: random.Random,
) -> float | None:
    distribution = _observed_distribution(schema, name)
    histogram = distribution.get("histogram")
    log_base = distribution.get("log_base")
    if not isinstance(histogram, list) or not isinstance(log_base, (int, float)) or log_base <= 1:
        return None
    buckets: list[tuple[int, int]] = []
    for item in histogram:
        if not isinstance(item, list) or len(item) != 2:
            continue
        index, count = item
        if isinstance(index, int) and isinstance(count, int) and count > 0:
            buckets.append((index, count))
    if not buckets:
        return None
    index = rng.choices([item[0] for item in buckets], weights=[item[1] for item in buckets], k=1)[0]
    if index == 0:
        value = 0.0
    else:
        magnitude = math.expm1((abs(index) - 0.5) * math.log(float(log_base)))
        value = magnitude if index > 0 else -magnitude
    minimum = distribution.get("min")
    maximum = distribution.get("max")
    if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        value = max(float(minimum), value)
    if isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
        value = min(float(maximum), value)
    return value


def _count_weights(value: SchemaValue | object) -> dict[str, int]:
    record = _schema_record(value)
    return {
        str(key): count
        for key, count in record.items()
        if isinstance(count, int) and not isinstance(count, bool) and count > 0
    }


def _observed_type_weights(schema: SchemaRecord) -> dict[str, int]:
    """Observed type mix at this node, with null observations as the ``null`` weight."""
    observed = _schema_record(schema.get("x-polylogue-observed-distribution"))
    weights = _count_weights(observed.get("type_counts"))
    nulls = observed.get("null_observations")
    if isinstance(nulls, int) and not isinstance(nulls, bool) and nulls > 0:
        weights["null"] = weights.get("null", 0) + nulls
    return weights


def _categorical_buckets(schema: SchemaRecord) -> list[tuple[int, int]] | None:
    """Hashed value buckets for a field whose observed values repeat.

    Inference keeps only hashed buckets, never raw values. A field is treated
    as categorical when its values repeat: fewer occupied buckets than half the
    bucket space, and at least twice as many observations as occupied buckets.
    Otherwise it is free text and gets a fresh value per occurrence.
    """
    categorical = _observed_distribution(schema, "categorical")
    histogram = categorical.get("bucket_histogram")
    bucket_count = categorical.get("bucket_count")
    if not isinstance(histogram, list) or not isinstance(bucket_count, int) or bucket_count <= 0:
        return None
    buckets = [
        (item[0], item[1])
        for item in histogram
        if isinstance(item, list)
        and len(item) == 2
        and isinstance(item[0], int)
        and isinstance(item[1], int)
        and item[1] > 0
    ]
    observations = sum(count for _, count in buckets)
    if not buckets or len(buckets) * 2 > bucket_count or observations < 2 * len(buckets):
        return None
    return buckets


def _free_text(rng: random.Random, length: int, newlines: int) -> str:
    """Synthetic text of exactly ``length`` characters with ``newlines`` line breaks."""
    if length <= 0:
        return ""
    words: list[str] = []
    size = 0
    while size < length:
        word = _WORDS[rng.randrange(len(_WORDS))]
        words.append(word)
        size += len(word) + 1
    text = list(" ".join(words)[:length])
    breaks = min(max(newlines, 0), length)
    for position in sorted(rng.sample(range(length), breaks)) if breaks else ():
        text[position] = "\n"
    return "".join(text)


_WORDS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "signal",
    "vector",
    "module",
    "record",
    "stream",
    "packet",
    "window",
    "buffer",
    "anchor",
    "lattice",
    "cursor",
    "ledger",
    "branch",
    "kernel",
    "shard",
    "frame",
)


def _generate_from_schema(
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    skip_keys: set[str] | None = None,
    depth: int = 0,
    max_depth: int = 6,
    path: str = "$",
) -> JSONValue:
    if depth > max_depth or not schema:
        return None

    semantic_role = schema.get("x-polylogue-semantic-role")
    if (
        isinstance(semantic_role, str)
        and self._semantic_gen is not None
        and not (self._coverage_witness_mode and path in self._coverage_type_choices)
    ):
        handled, value = self._semantic_gen.try_generate(schema)
        if handled:
            return value

    for keyword in ("anyOf", "oneOf"):
        raw_variants = schema.get(keyword)
        variants = [
            (index, variant)
            for index, item in enumerate(raw_variants if isinstance(raw_variants, list) else [])
            if (variant := _schema_record(item))
        ]
        if variants:
            if keyword not in SCHEMA_CONSTRUCT_HANDLERS:
                return None
            selected_index = self._coverage_branch_choices.get(path) if self._coverage_witness_mode else None
            if selected_index is not None:
                selected = next((variant for index, variant in variants if index == selected_index), None)
                if selected is not None:
                    return self._generate_from_schema(
                        selected,
                        rng,
                        skip_keys=skip_keys,
                        depth=depth,
                        max_depth=max_depth,
                        path=(f"{path}.{keyword}[{selected_index}]" if self._coverage_witness_mode else path),
                    )

            non_null = [item for item in variants if item[1].get("type") != "null"]
            candidates = non_null if non_null else variants

            # Weight by x-polylogue-frequency when available on variants.
            # Falls back to uniform when no variant carries the annotation
            # (backward-compatible with un-annotated schemas).
            weights: list[float] = []
            for _, variant in candidates:
                f = variant.get("x-polylogue-frequency")
                weights.append(float(f) if isinstance(f, (int, float)) and f > 0 else 0.0)
            if sum(weights) > 0:
                # Normalize so random.choices treats them as relative weights
                total = sum(weights)
                weights = [w / total for w in weights]
                chosen = rng.choices(candidates, weights=weights, k=1)[0]
            else:
                chosen = rng.choice(candidates)
            return self._generate_from_schema(
                chosen[1],
                rng,
                skip_keys=skip_keys,
                depth=depth,
                max_depth=max_depth,
                path=(f"{path}.{keyword}[{chosen[0]}]" if self._coverage_witness_mode else path),
            )

    schema_type = _schema_type(self, schema, rng, path)
    if schema_type is not None and schema_type not in SCHEMA_CONSTRUCT_HANDLERS:
        return None
    freq_value = schema.get("x-polylogue-frequency")
    freq = float(freq_value) if isinstance(freq_value, (int, float)) else 1.0
    if depth > 0 and freq < 1.0 and rng.random() > freq:
        return None

    match schema_type:
        case "object":
            return self._generate_object(
                schema,
                rng,
                skip_keys=skip_keys,
                depth=depth,
                max_depth=max_depth,
                path=path,
            )
        case "string":
            value = self._generate_string(schema, rng, path=path)
            fmt = schema.get("x-polylogue-format")
            if fmt in {"uuid4", "uuid", "hex-id"}:
                self._relation_solver.register_generated_id(path, value)
            return value
        case "number":
            return self._generate_number(schema, rng, is_int=False)
        case "integer":
            return self._generate_number(schema, rng, is_int=True)
        case "array":
            return self._generate_array(schema, rng, depth=depth, max_depth=max_depth, path=path)
        case "boolean":
            counts = _count_weights(_observed_distribution(schema, "boolean_counts"))
            true_count, false_count = counts.get("true", 0), counts.get("false", 0)
            if true_count + false_count:
                return rng.random() < true_count / (true_count + false_count)
            return rng.choice([True, False])
        case "null":
            return None
        case _:
            if "object" not in SCHEMA_CONSTRUCT_HANDLERS:
                return None
            if "properties" in schema:
                return self._generate_object(
                    schema,
                    rng,
                    skip_keys=skip_keys,
                    depth=depth,
                    max_depth=max_depth,
                    path=path,
                )
            return None


def _generate_object(
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    skip_keys: set[str] | None = None,
    depth: int = 0,
    max_depth: int = 6,
    path: str = "$",
) -> SyntheticRecord:
    obj: SyntheticRecord = {}
    properties = _schema_record(schema.get("properties"))
    candidate_keys = set(properties.keys())
    selected_root_fields: set[str] = set()
    if path == "$" and self._active_profile_tokens:
        if self._active_record_bucket is None:
            selected_root_fields = {
                token.removeprefix("field:")
                for token in self._active_profile_tokens
                if token.startswith("field:") and token.count(":") == 1
            }
        else:
            discriminator, bucket_value = self._active_record_bucket
            prefix = f"field:{discriminator}:{bucket_value}:"
            selected_root_fields = {
                token.removeprefix(prefix)
                for token in self._active_profile_tokens
                if token.startswith(prefix) and token.count(":") == 3
            }
        if selected_root_fields:
            candidate_keys &= selected_root_fields
    if skip_keys:
        candidate_keys -= skip_keys

    if self._relation_solver.mutual_exclusions and not self._coverage_witness_mode:
        candidate_keys = self._relation_solver.filter_mutually_exclusive(path, candidate_keys, rng)

    for prop_name, prop_value in properties.items():
        if prop_name not in candidate_keys:
            continue
        prop_schema = _schema_record(prop_value)
        if not prop_schema:
            continue

        freq_value = prop_schema.get("x-polylogue-frequency")
        freq = float(freq_value) if isinstance(freq_value, (int, float)) else 1.0
        if prop_name not in selected_root_fields and freq < 1.0:
            freq = _conditional_presence(properties, prop_name, obj, freq)
            if rng.random() > freq:
                continue
        if prop_name in selected_root_fields and freq < 1.0:
            prop_schema = {**prop_schema, "x-polylogue-frequency": 1.0}

        child_path = f"{path}.properties.{prop_name}" if self._coverage_witness_mode else f"{path}.{prop_name}"
        ref = self._relation_solver.resolve_foreign_key(child_path, rng)
        if ref is not None:
            obj[prop_name] = ref
            continue

        value = self._generate_from_schema(
            prop_schema,
            rng,
            depth=depth + 1,
            max_depth=max_depth,
            path=child_path,
        )
        if value is not None or (self._coverage_witness_mode and _schema_allows_null(prop_schema)):
            obj[prop_name] = value

    additional_schema = _schema_record(schema.get("additionalProperties"))
    if not self._coverage_witness_mode and additional_schema and not properties:
        # A dynamic-key map (model ids, file paths, tool names as keys): draw
        # the observed key count and fill each value from the value schema.
        fanout = _sampled_length(schema, "object_fanout", rng)
        for index in range(fanout if fanout is not None else 0):
            key = f"key-{index:04d}"
            value = self._generate_from_schema(
                additional_schema,
                rng,
                depth=depth + 1,
                max_depth=max_depth,
                path=f"{path}.*",
            )
            if value is not None or _schema_allows_null(additional_schema):
                obj[key] = value
    if self._coverage_witness_mode and additional_schema:
        extra_name = COVERAGE_EXTRA_KEY
        while extra_name in properties:
            extra_name = f"_{extra_name}"
        extra_value = self._generate_from_schema(
            additional_schema,
            rng,
            depth=depth + 1,
            max_depth=max_depth,
            path=f"{path}.additionalProperties.*",
        )
        if extra_value is not None or _schema_allows_null(additional_schema):
            obj[extra_name] = extra_value

    return obj


def _conditional_presence(
    properties: SchemaRecord,
    prop_name: str,
    present: SyntheticRecord,
    marginal: float,
) -> float:
    """P(field present | fields already generated), from observed co-occurrence.

    Each already-present sibling records how many of its documents also carried
    ``prop_name``. The most specific sibling (fewest documents) decides; with no
    co-occurrence evidence the marginal frequency stands.
    """
    best: tuple[int, float] | None = None
    for sibling in present:
        sibling_schema = _schema_record(properties.get(sibling))
        observed = _schema_record(sibling_schema.get("x-polylogue-observed-distribution"))
        together = _schema_record(observed.get("co_occurring_fields")).get(prop_name, 0)
        documents = observed.get("encountered_documents")
        if not isinstance(documents, int) or isinstance(documents, bool) or documents <= 0:
            continue
        together_count = together if isinstance(together, int) and not isinstance(together, bool) else 0
        if best is None or documents < best[0]:
            best = (documents, together_count / documents)
    return marginal if best is None else best[1]


def _schema_allows_null(schema: SchemaRecord) -> bool:
    schema_type = schema.get("type")
    if schema_type == "null":
        return True
    if isinstance(schema_type, list) and "null" in schema_type:
        return True
    return any(
        _schema_allows_null(variant)
        for keyword in ("anyOf", "oneOf")
        for variant in _schema_records(schema.get(keyword))
    )


def _sampled_length(schema: SchemaRecord, name: str, rng: random.Random) -> int | None:
    sampled = _sample_observed_distribution(schema, name, rng)
    return None if sampled is None else max(0, int(round(sampled)))


def _generate_string(
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    path: str = "$",
) -> str:
    values = schema.get("x-polylogue-values")
    if isinstance(values, list) and values:
        return str(rng.choice(values))

    match schema.get("x-polylogue-format"):
        case "uuid4" | "uuid":
            return str(uuid.UUID(int=rng.getrandbits(128), version=4))
        case "hex-id":
            return rng.randbytes(12).hex()
        case "iso8601":
            ts = rng.uniform(1670000000, 1760000000)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        case "unix-epoch" | "unix-epoch-str":
            return str(rng.uniform(1670000000, 1760000000))
        case "url":
            return f"https://example.com/{rng.randint(1000, 9999)}"
        case "email":
            return f"user{rng.randint(1, 999)}@example.com"
        case "mime-type":
            return rng.choice(["text/plain", "application/json", "text/html"])
        case "base64":
            return rng.randbytes(24).hex()

    buckets = _categorical_buckets(schema)
    if buckets is not None:
        # One stable synthetic value per observed hashed bucket, drawn with the
        # observed bucket frequencies, so repeated values repeat as they did.
        pool = self._categorical_pools.get(path)
        if pool is None:
            pool = []
            for bucket, count in buckets:
                value_rng = random.Random(f"{self.provider}|{path}|{bucket}")
                length = _sampled_length(schema, "string_length", value_rng)
                token = f"{path.rsplit('.', 1)[-1].strip('[*]') or 'value'}-{bucket:02x}"
                pool.append((token if length is None else _fit(token, length, value_rng), count))
            self._categorical_pools[path] = pool
        return rng.choices([value for value, _ in pool], weights=[count for _, count in pool], k=1)[0]

    length = _sampled_length(schema, "string_length", rng)
    newlines = _sampled_length(schema, "newline_count", rng) or 0
    if length is not None:
        return _free_text(rng, length, newlines)
    if schema.get("x-polylogue-multiline"):
        return _text_for_role(rng, "assistant")

    return f"synthetic-{rng.randint(0, 99999)}"


def _fit(token: str, length: int, rng: random.Random) -> str:
    if len(token) >= length:
        return token[:length]
    return token + "-" + _free_text(rng, length - len(token) - 1, 0) if length - len(token) > 1 else token + "-"


def _generate_number(
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    is_int: bool = False,
) -> float | int:
    observed = _sample_observed_distribution(schema, "numeric", rng)
    range_value = schema.get("x-polylogue-range")
    if observed is not None:
        value = observed
    elif isinstance(range_value, list) and len(range_value) >= 2:
        lo = _coerce_float(range_value[0])
        hi = _coerce_float(range_value[1])
        value = rng.uniform(lo, hi)
    elif schema.get("x-polylogue-format") == "unix-epoch":
        value = rng.uniform(1670000000, 1760000000)
    else:
        value = rng.uniform(0, 1000)
    return int(value) if is_int else value


def _generate_array(
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    depth: int = 0,
    max_depth: int = 6,
    path: str = "$",
) -> list[JSONValue]:
    item_schema = _schema_record(schema.get("items"))
    observed_length = _sample_observed_distribution(schema, "array_length", rng)
    lengths = schema.get("x-polylogue-array-lengths")
    if observed_length is not None:
        n_items = max(0, int(round(observed_length)))
    elif isinstance(lengths, list) and len(lengths) >= 2:
        lo = _coerce_int(lengths[0])
        hi = _coerce_int(lengths[1])
        bounded_lo = max(0, lo)
        bounded_hi = max(hi, bounded_lo)
        n_items = rng.randint(bounded_lo, bounded_hi)
    else:
        n_items = rng.randint(1, 3)
    if self._coverage_witness_mode:
        n_items = 1

    item_type = item_schema.get("type")
    item_allows_null = item_type == "null" or (
        isinstance(item_type, list) and any(item == "null" for item in item_type if isinstance(item, str))
    )
    if not item_allows_null:
        item_allows_null = any(variant.get("type") == "null" for variant in _schema_records(item_schema.get("anyOf")))
    if not item_allows_null:
        item_allows_null = any(variant.get("type") == "null" for variant in _schema_records(item_schema.get("oneOf")))

    item_path = f"{path}.items[*]" if self._coverage_witness_mode else f"{path}[*]"
    items = [
        self._generate_from_schema(
            item_schema,
            rng,
            depth=depth + 1,
            max_depth=max_depth,
            path=item_path,
        )
        for _ in range(n_items)
    ]
    if not item_allows_null:
        items = [value for value in items if value is not None]
    ordered = _observed_distribution(item_schema, "ordered_numeric_pairs")
    rate = ordered.get("nondecreasing_rate")
    if (
        isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and items
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in items)
        and rng.random() < rate
    ):
        numeric = [value for value in items if isinstance(value, (int, float))]
        items = [*sorted(numeric)]
    return items


def _serialize(self: _SyntheticRuntimeContext, data: JSONValue) -> bytes:
    normalize_browser_capture_attachments(data)
    validate_wire_payload(self.provider, data)
    if self.wire_format.encoding == "jsonl":
        if not isinstance(data, list):
            raise ValueError("JSONL wire format requires a list payload")
        lines = [json.dumps(record, separators=(",", ":")) for record in data]
        return ("\n".join(lines) + "\n").encode("utf-8")
    return json.dumps(data, indent=2).encode("utf-8")


__all__ = [
    "SCHEMA_CONSTRUCT_HANDLERS",
    "_generate_array",
    "_generate_from_schema",
    "_generate_number",
    "_generate_object",
    "_generate_string",
    "_serialize",
]
