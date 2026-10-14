"""Recursive schema-to-data generation helpers for synthetic corpora."""

from __future__ import annotations

import base64
import hashlib
import json
import random
import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from itertools import chain
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

from polylogue.core.tool_result_pointer import (
    PERSISTED_OUTPUT_POINTER_PATTERN,
    PERSISTED_OUTPUT_PREFIX,
    pointer_width_relation,
)
from polylogue.schemas.field_stats.distributions import CATEGORICAL_BUCKET_COUNT, _bucket_index, categorical_bucket
from polylogue.schemas.field_stats.models import PointerShapeStats
from polylogue.schemas.synthetic.array_value import GeneratedArray, walk_generated_values
from polylogue.schemas.synthetic.build_wire_formats import normalize_browser_capture_attachments, validate_wire_payload
from polylogue.schemas.synthetic.categorical_values import _strings_of_shape, categorical_codebook
from polylogue.schemas.synthetic.classification import (
    cooccurrence_observation_count,
    declared_required_properties,
    normalized_observed_type_counts,
)
from polylogue.schemas.synthetic.conservation import COVERAGE_EXTRA_KEY
from polylogue.schemas.synthetic.models import GeneratedRecord, GeneratedString, SchemaRecord, SchemaValue
from polylogue.schemas.synthetic.models import GeneratedValue as JSONValue
from polylogue.schemas.synthetic.observed_values import (
    _ProjectedStringShape,
    format_length_is_representable,
    format_text_shape_is_representable,
    histogram_value,
    measured_character_shapes,
    measured_string_formats,
    neutral_numeric_range,
    observed_length_support,
)
from polylogue.schemas.synthetic.record_profiles import ConditionalContentKinds, PublicRecordPlan
from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator, _text_for_role
from polylogue.schemas.synthetic.text_value import JoinedText, SampledText, joined_text_value, sampled_text_value
from polylogue.schemas.synthetic.wire_formats import WireFormat

if TYPE_CHECKING:
    from polylogue.schemas.synthetic.models import SyntheticGenerationState
    from polylogue.schemas.synthetic.relations import RelationConstraintSolver

SyntheticRecord: TypeAlias = dict[str, JSONValue] | GeneratedRecord

# The receipt reads this registry and the recursive generator gates dispatch on
# it. Removing a construct handler therefore changes both generated behavior
# and the support receipt instead of becoming a silent ``None`` fallback.
SCHEMA_CONSTRUCT_HANDLERS: dict[str, str] = {
    "anyOf": "schema-union",
    "array": "_walk_array",
    "boolean": "boolean",
    "integer": "_generate_number",
    "null": "null",
    "number": "_generate_number",
    "object": "_walk_object",
    "oneOf": "schema-union",
    "prefixItems": "_walk_array",
    "string": "_generate_string",
}


class _SyntheticRuntimeContext(Protocol):
    provider: str
    wire_format: WireFormat
    _semantic_gen: SemanticValueGenerator | None
    _generation_state: SyntheticGenerationState
    _relation_solver: RelationConstraintSolver
    _public_record_plan: PublicRecordPlan | None
    _categorical_pools: dict[str, tuple[str, list[tuple[GeneratedString, float]]]]
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
        path: str = "$",
    ) -> JSONValue: ...

    def _generate_string(
        self, schema: SchemaRecord, rng: random.Random, *, path: str = "$"
    ) -> str | SampledText | JoinedText: ...

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


def _annotated_variant(parent: SchemaRecord, variant: SchemaRecord) -> SchemaRecord:
    return {**{key: value for key, value in parent.items() if key.startswith("x-polylogue-")}, **variant}


def _schema_type(self: _SyntheticRuntimeContext, schema: SchemaRecord, rng: random.Random, path: str) -> str | None:
    schema_type = schema.get("type")
    plan = None if self._coverage_witness_mode else self._public_record_plan
    if isinstance(schema_type, str):
        if plan is not None and not plan.compatible(schema, path):
            raise ValueError("selected public record has no compatible original field type")
        return schema_type
    if isinstance(schema_type, list):
        selected_type = self._coverage_type_choices.get(path) if self._coverage_witness_mode else None
        if selected_type in schema_type:
            return selected_type
        if self._coverage_witness_mode and path in self._coverage_null_paths and "null" in schema_type:
            return "null"
        declared = [item for item in schema_type if isinstance(item, str)]
        if plan is not None:
            # The selected form owns required parents and scalar/result types.
            # Condition the existing type draw on that same branch proof;
            # merged nullable fields cannot erase an already selected message.
            declared = [item for item in declared if plan.compatible({**schema, "type": item}, path)]
            if not declared:
                raise ValueError("selected public record has no compatible original field type")
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
    return histogram_value(distribution, index)


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
    return normalized_observed_type_counts(observed) or {}


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
    counts: dict[int, int] = {}
    for bucket, count in buckets:
        counts[bucket] = counts.get(bucket, 0) + count
    buckets = sorted(counts.items())
    observations = sum(count for _, count in buckets)
    if not buckets or len(buckets) * 2 > bucket_count or observations < 2 * len(buckets):
        return None
    return buckets


def _free_text(
    rng: random.Random, length: int, newlines: int, *, prefix: str = "", character_class: str | None = None
) -> str | SampledText | JoinedText:
    """Spell the sampled shape through a bounded, replayable text recipe."""
    if character_class == "ascii" and not prefix.isascii():
        raise ValueError("neutral proposal contradicts original ASCII shape")
    if character_class == "non_ascii":
        if length <= newlines:
            raise ValueError("non-ASCII shape has no non-newline coordinate")
        return joined_text_value("é", _free_text(rng, length - 1, newlines, prefix=prefix))
    pool = prefix.replace("\n", " ") + " " + " ".join(rng.choice(_WORDS) for _ in range(32))
    return sampled_text_value(
        SampledText(
            length,
            pool,
            offset=0 if prefix else rng.randrange(len(pool)),
            newlines=newlines,
            newline_phase=rng.randrange(length) if length else 0,
        )
    )


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


_GenerationRequest: TypeAlias = tuple[SchemaRecord, str, set[str] | None]
_GenerationFrame: TypeAlias = Generator[_GenerationRequest, JSONValue, JSONValue]


def _generate_from_schema(
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    skip_keys: set[str] | None = None,
    path: str = "$",
) -> JSONValue:
    """Walk finite schema nodes with explicit, cancellable traversal frames."""
    self._relation_solver.fk_graph.schema_coordinate(schema)
    frames = [_walk_node(self, schema, rng, skip_keys=skip_keys, path=path)]
    result: JSONValue = None
    try:
        while frames:
            try:
                child_schema, child_path, child_skip = frames[-1].send(result)
            except StopIteration as completed:
                frames.pop()
                result = completed.value
            else:
                frames.append(_walk_node(self, child_schema, rng, path=child_path, skip_keys=child_skip))
                result = None
        return result
    finally:
        for frame in reversed(frames):
            frame.close()


def _walk_node(
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    skip_keys: set[str] | None = None,
    path: str = "$",
) -> _GenerationFrame:
    plan = None if self._coverage_witness_mode else self._public_record_plan
    if plan is not None and plan.requires_field(path) and not plan.compatible(schema, path):
        raise ValueError(
            "measured public record conflicts with selected field support "
            f"at {path} (kind={plan.kind}, form={plan.form}); re-infer the package"
        )
    if not schema and (plan is None or not plan.requires_field(path)):
        return None

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
                    return (
                        yield (
                            _annotated_variant(schema, selected),
                            f"{path}.{keyword}[{selected_index}]" if self._coverage_witness_mode else path,
                            skip_keys,
                        )
                    )

            candidates = [
                (index, variant) for index, variant in variants if plan is None or plan.compatible(variant, path)
            ]
            if not candidates:
                raise ValueError("measured public record has no supported union branch; re-infer the package")

            # Weight by x-polylogue-frequency when available on variants.
            # Falls back to uniform when no variant carries the annotation
            # (backward-compatible with un-annotated schemas).
            weights: list[float] = []
            observed_types = _observed_type_weights(schema)
            for _, variant in candidates:
                f = variant.get("x-polylogue-frequency")
                variant_type = variant.get("type")
                type_weight = observed_types.get(variant_type, 0) if isinstance(variant_type, str) else 0
                weights.append(float(f) if isinstance(f, (int, float)) and f > 0 else float(type_weight))
            if sum(weights) > 0:
                # Normalize so random.choices treats them as relative weights
                total = sum(weights)
                weights = [w / total for w in weights]
                chosen = rng.choices(candidates, weights=weights, k=1)[0]
            else:
                chosen = rng.choice(candidates)
            return (
                yield (
                    _annotated_variant(schema, chosen[1]),
                    f"{path}.{keyword}[{chosen[0]}]" if self._coverage_witness_mode else path,
                    skip_keys,
                )
            )

    if plan is not None and path in plan.scalars:
        return plan.scalars[path]
    if plan is not None and path in plan.graph_values:
        return plan.graph_values[path]
    if plan is not None and (tool_value := plan.tool_value(path, schema)) is not None:
        plan.observe_tool_field(path, tool_value, schema)
        return tool_value
    schema_type = (
        plan.content_type if plan is not None and path == plan.content_path else _schema_type(self, schema, rng, path)
    )
    if schema_type is not None and schema_type not in SCHEMA_CONSTRUCT_HANDLERS:
        return None
    # Field presence is decided once, by the owning object (co-occurrence
    # aware); a union variant's frequency is its selection weight above.
    match schema_type:
        case "object":
            return (
                yield from _walk_object(
                    self,
                    schema,
                    rng,
                    skip_keys=skip_keys,
                    path=path,
                )
            )
        case "string":
            value = self._generate_string(schema, rng, path=path)
            if plan is not None:
                plan.observe_tool_field(path, value, schema)
            fmt = schema.get("x-polylogue-format")
            if fmt in {"uuid4", "uuid", "hex-id"}:
                self._relation_solver.register_generated_id(path, value)
            return value
        case "number":
            return self._generate_number(schema, rng, is_int=False)
        case "integer":
            return self._generate_number(schema, rng, is_int=True)
        case "array":
            return (yield from _walk_array(self, schema, rng, path=path))
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
                return (
                    yield from _walk_object(
                        self,
                        schema,
                        rng,
                        skip_keys=skip_keys,
                        path=path,
                    )
                )
            return None


def _walk_object(
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    skip_keys: set[str] | None = None,
    path: str = "$",
) -> _GenerationFrame:
    obj = self._relation_solver.fk_graph.new_record(schema, path=path)
    properties = _schema_record(schema.get("properties"))
    schema_required = declared_required_properties(schema)
    candidate_keys = set(properties.keys())
    plan = None if self._coverage_witness_mode else self._public_record_plan
    if plan is not None:
        candidate_keys |= plan.child_fields(path)
    if plan is not None:
        if any(path + "." + key in plan.absent for key in schema_required):
            raise ValueError("required field conflicts with the selected original wire form")
        candidate_keys = {key for key in candidate_keys if path + "." + key not in plan.absent}
    if plan is not None and plan.mode == "absent":
        candidate_keys = {key for key in candidate_keys if path + "." + key != plan.content_path}
    if skip_keys:
        candidate_keys -= skip_keys

    if self._relation_solver.mutual_exclusions and not self._coverage_witness_mode:
        required = schema_required | {
            key for key in candidate_keys if plan is not None and plan.requires_field(path + "." + key)
        }
        candidate_keys = self._relation_solver.filter_mutually_exclusive(path, candidate_keys, rng, required=required)

    ordered_keys = list(chain(properties, sorted(candidate_keys - properties.keys())))
    if plan is not None:
        # A measured pointer consumes the original ID width. Generate that
        # dependency first even when wire properties list content before ID.
        identities = {
            identity
            for content, (identity, _bucket, _relation) in plan.tool_pointer_widths.items()
            if content.rsplit(".", 1)[0] == path
        }
        ordered_keys.sort(key=lambda key: path + "." + key not in identities)
    orders = self._relation_solver.numeric_orders(schema) if self.provider == "codex" else ()
    selected_numeric_fields: set[str] | None = None
    if orders:
        # Presence remains the original schema/co-occurrence decision. Make it
        # once before either value draw so an absent sibling adds no constraint.
        selected_numeric_fields = set()
        presence: dict[str, JSONValue] = {}
        for name in ordered_keys:
            if name not in candidate_keys:
                continue
            field_schema = _schema_record(properties.get(name, schema.get("additionalProperties")))
            forced = name in schema_required or plan is not None and plan.requires_field(path + "." + name)
            if not field_schema and not forced:
                continue
            frequency = field_schema.get("x-polylogue-frequency")
            frequency = float(frequency) if isinstance(frequency, int | float) else 1.0
            if not forced and frequency < 1.0:
                frequency = _conditional_presence(properties, name, presence, frequency)
                if rng.random() > frequency:
                    continue
            selected_numeric_fields.add(name)
            presence[name] = None
        upper_fields = {order.upper for order in orders}
        ordered_keys.sort(key=lambda name: name not in upper_fields)
    for prop_name in ordered_keys:
        if prop_name not in candidate_keys:
            continue
        prop_schema = _schema_record(properties.get(prop_name, schema.get("additionalProperties")))
        forced = prop_name in schema_required or plan is not None and plan.requires_field(path + "." + prop_name)
        if not prop_schema and not forced:
            continue

        freq_value = prop_schema.get("x-polylogue-frequency")
        freq = float(freq_value) if isinstance(freq_value, (int, float)) else 1.0
        if selected_numeric_fields is not None:
            if prop_name not in selected_numeric_fields:
                continue
            prop_schema = self._relation_solver.condition_numeric_field(
                prop_name,
                prop_schema,
                properties=properties,
                present=selected_numeric_fields,
                values=obj,
                orders=orders,
            )
        elif not forced and freq < 1.0:
            freq = _conditional_presence(properties, prop_name, obj, freq)
            if rng.random() > freq:
                continue

        child_path = f"{path}.properties.{prop_name}" if self._coverage_witness_mode else f"{path}.{prop_name}"
        ref = (
            None
            if plan is not None and (child_path in plan.tool_fields or child_path in plan.string_exclusions)
            else self._relation_solver.resolve_foreign_key(child_path, rng)
        )
        if ref is not None:
            obj[prop_name] = ref
            continue

        value = yield (prop_schema, child_path, None)
        if value is not None or _schema_allows_null(prop_schema):
            obj[prop_name] = value

    additional_value = schema.get("additionalProperties")
    additional_schema = _schema_record(additional_value)
    additional_enabled = isinstance(additional_value, dict) or additional_value is True
    if not self._coverage_witness_mode and additional_enabled and not properties:
        # A dynamic-key map (model ids, file paths, tool names as keys): draw
        # the observed key count and fill each value from the value schema.
        fanout = _sampled_length(schema, "object_fanout", rng)
        for index in range(fanout if fanout is not None else 0):
            key = f"key-{index:04d}"
            value = yield (additional_schema, f"{path}.*", None)
            if value is not None or _schema_allows_null(additional_schema):
                obj[key] = value
    if self._coverage_witness_mode and additional_enabled:
        extra_name = COVERAGE_EXTRA_KEY
        while extra_name in properties:
            extra_name = f"_{extra_name}"
        extra_value = yield (additional_schema, f"{path}.additionalProperties.*", None)
        if extra_value is not None or _schema_allows_null(additional_schema):
            obj[extra_name] = extra_value

    if (
        plan is not None
        and self.provider == "claude-code"
        and plan.scalars.get(path + ".type") == "tool_result"
        and isinstance(tool_id := obj.get("tool_use_id"), str | SampledText | JoinedText)
        and len(tool_id)
    ):
        self._relation_solver.stage_tool_result_reference(obj)
    return obj


def _conditional_presence(
    properties: SchemaRecord,
    prop_name: str,
    present: SyntheticRecord,
    marginal: float,
) -> float:
    """P(field present | fields already generated), from observed co-occurrence.

    Each already-present sibling records how many of its occurrences also carried
    ``prop_name``. The most specific sibling (fewest occurrences) decides; with no
    co-occurrence evidence the marginal frequency stands.
    """
    best: tuple[int, float] | None = None
    for sibling in present:
        sibling_schema = _schema_record(properties.get(sibling))
        observed = _schema_record(sibling_schema.get("x-polylogue-observed-distribution"))
        together = _schema_record(observed.get("co_occurring_fields")).get(prop_name)
        # Public maps may be absent or partial; only an explicit measured count
        # can override the marginal (including an explicit measured zero).
        if not isinstance(together, int) or isinstance(together, bool) or together < 0:
            continue
        observations = cooccurrence_observation_count(observed)
        if observations is None:
            continue
        if together > observations:
            raise ValueError("co-occurrence count exceeds its observation count")
        if observations == 0:
            continue
        if best is None or observations < best[0]:
            best = (observations, together / observations)
    return marginal if best is None else best[1]


def _schema_allows_null(schema: SchemaRecord) -> bool:
    if not schema:
        return True
    pending = [schema]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        schema_type = node.get("type")
        if schema_type == "null" or (isinstance(schema_type, list) and "null" in schema_type):
            return True
        pending.extend(variant for keyword in ("anyOf", "oneOf") for variant in _schema_records(node.get(keyword)))
    return False


def _sampled_length(schema: SchemaRecord, name: str, rng: random.Random) -> int | None:
    sampled = _sample_observed_distribution(schema, name, rng)
    if sampled is not None and sampled < 0:
        raise ValueError("observed length or count is negative")
    return None if sampled is None else int(round(sampled))


def _sample_text_shape(schema: SchemaRecord, rng: random.Random, *, minimum_plain: int = 0) -> tuple[int | None, int]:
    lengths = observed_length_support(_observed_distribution(schema, "string_length"))
    newlines = observed_length_support(_observed_distribution(schema, "newline_count")) or {0: 1}
    if not lengths:
        count = rng.choices(list(newlines), weights=list(newlines.values()), k=1)[0]
        return None, count
    candidates = [length for length in lengths if any(count + minimum_plain <= length for count in newlines)]
    if not candidates:
        raise ValueError("observed string length and newline counts have no representable combination")
    length = rng.choices(candidates, weights=[lengths[item] for item in candidates], k=1)[0]
    counts = [count for count in newlines if count + minimum_plain <= length]
    count = rng.choices(counts, weights=[newlines[item] for item in counts], k=1)[0]
    return length, count


def _generate_string(
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    path: str = "$",
    _format_body: bool = False,
) -> str | SampledText | JoinedText:
    plan = None if self._coverage_witness_mode else self._public_record_plan
    excluded = plan.string_exclusions.get(path, frozenset()) if plan is not None else frozenset()
    agent_prefix = plan.agent_identity_prefix if plan is not None and path == "$.toolUseResult.agentId" else None
    fresh_tool = plan is not None and path in plan.tool_fields and plan.tool_fields[path][1] == "new"

    def accepted_identity(value: GeneratedString) -> bool:
        if agent_prefix is not None:
            if not isinstance(value, str) or value.startswith("agent-") != agent_prefix:
                return False
            assert plan is not None
            return plan.owner.agent_identity_is_new(value)
        return not fresh_tool or (plan is not None and plan.owner.tool_identity_is_new(value))

    width = plan.tool_widths.get(path) if plan is not None else None
    pointer_width = plan.tool_pointer_widths.get(path) if plan is not None else None
    geometry = (
        (plan.tool_widths[pointer_width[0]], pointer_width[1], pointer_width[2])
        if pointer_width is not None and plan is not None
        else None
    )
    values = schema.get("x-polylogue-values")
    if isinstance(values, list) and values:
        labels = [str(value) for value in values if str(value) not in excluded]
        if agent_prefix is not None:
            labels = [value for value in labels if accepted_identity(value)]
        if width is not None:
            labels = [value for value in labels if len(value) == width]
        if geometry is not None:
            from polylogue.core.tool_result_pointer import persisted_output_shape

            labels = [
                value
                for value in labels
                if (point := persisted_output_shape(value)) is not None
                and point[5] == geometry[1]
                and pointer_width_relation(geometry[0], point[5]) == geometry[2]
            ]
        plan = None if self._coverage_witness_mode else self._public_record_plan
        if plan is not None and path in plan.tool_pointer_fields:
            pointer = plan.tool_pointer_fields[path]
            labels = [
                label for label in labels if (PERSISTED_OUTPUT_POINTER_PATTERN.search(label) is not None) == pointer
            ]
        if plan is not None and path in plan.tool_fields and plan.tool_fields[path][1] == "new":
            labels = [label for label in labels if label and plan.owner.tool_identity_is_new(label)]
        if not labels:
            raise ValueError("selected relation has no supported original field value")
        categorical = _observed_distribution(schema, "categorical")
        if categorical.get("bucket_count") == CATEGORICAL_BUCKET_COUNT:
            histogram = categorical.get("bucket_histogram")
            counts = (
                {
                    row[0]: row[1]
                    for row in histogram
                    if isinstance(row, list)
                    and len(row) == 2
                    and isinstance(row[0], int)
                    and isinstance(row[1], int)
                    and row[1] > 0
                }
                if isinstance(histogram, list)
                else {}
            )
            label_buckets = [categorical_bucket(label) for label in labels]
            multiplicity = {bucket: label_buckets.count(bucket) for bucket in set(label_buckets)}
            weights = [counts.get(bucket, 0) / multiplicity[bucket] for bucket in label_buckets]
            if any(weights):
                return rng.choices(labels, weights=weights, k=1)[0]
        return rng.choice(labels)

    plan = None if self._coverage_witness_mode else self._public_record_plan
    selected_pointer = plan.tool_pointer_fields.get(path) if plan is not None else None
    if not _format_body:
        schema = _condition_string_format(schema, rng, pointer=selected_pointer, width=width, geometry=geometry)
    semantic_role = schema.get("x-polylogue-semantic-role")
    if plan is not None and (path in plan.tool_fields or excluded) and semantic_role == "message_role":
        # The original compiled tool coordinate is an identity, even when a
        # low-confidence role heuristic selected this leaf during inference.
        # Its measured field shape and relation width remain the authorities.
        schema = {key: value for key, value in schema.items() if key != "x-polylogue-semantic-role"}
        semantic_role = None
    if (
        semantic_role == "message_timestamp"
        and "x-polylogue-format" not in schema
        and "string_formats" not in _schema_record(schema.get("x-polylogue-observed-distribution"))
    ):
        schema = {**schema, "x-polylogue-format": "iso8601"}
    if semantic_role == "message_role" and self._semantic_gen is not None:
        # Hashed categories cannot recover a provider's role vocabulary.
        # The parser-owned role cycle is the semantic value authority.
        return self._semantic_gen.current_role

    character_class_value = _schema_record(schema.get("x-polylogue-observed-distribution")).get("character_class")
    character_class = character_class_value if isinstance(character_class_value, str) else None
    fmt = schema.get("x-polylogue-format")
    if character_class == "non_ascii" and fmt in {"uuid", "uuid4", "hex-id", "base64"}:
        raise ValueError("character evidence cannot represent original ASCII wire format")
    if not _format_body and isinstance(fmt, str) and fmt != "persisted-tool-output":
        lengths = observed_length_support(_observed_distribution(schema, "string_length"))
        newline_weights = observed_length_support(_observed_distribution(schema, "newline_count")) or {0: 1}
        if lengths:
            eligible = {
                length: [count for count in newline_weights if format_text_shape_is_representable(fmt, length, count)]
                for length in lengths
            }
            eligible = {length: counts for length, counts in eligible.items() if counts}
            if not eligible:
                raise ValueError(f"observed shape cannot represent original wire format {fmt!r} at {path}")
            length = rng.choices(list(eligible), weights=[lengths[item] for item in eligible], k=1)[0]
            newline_count = rng.choices(
                eligible[length], weights=[newline_weights[item] for item in eligible[length]], k=1
            )[0]
            observed = _schema_record(schema.get("x-polylogue-observed-distribution"))
            body_length = length - newline_count
            # Condition the same already-selected format/category support on
            # its drawn shape. The spelling draws no second marginal or format.
            body_distribution: SchemaRecord = {
                "min": body_length,
                "max": body_length,
                "count": 1,
                "histogram": [[_bucket_index(float(body_length)), 1]],
                "log_base": 1.1,
            }
            newline_distribution: SchemaRecord = {
                "min": 0,
                "max": 0,
                "count": 1,
                "histogram": [[0, 1]],
                "log_base": 1.1,
            }
            conditioned: SchemaRecord = {
                **schema,
                "x-polylogue-observed-distribution": {
                    **{key: value for key, value in observed.items() if key != "string_formats"},
                    "string_length": body_distribution,
                    "newline_count": newline_distribution,
                },
            }
            body = _generate_string(self, conditioned, rng, path=path, _format_body=True)
            return (
                body
                if not newline_count
                else joined_text_value(body, SampledText(newline_count, "x", newlines=newline_count))
            )

    match schema.get("x-polylogue-format"):
        case "persisted-tool-output":
            # This category's numeric joint was drawn before spelling. Unlike
            # independent length/path marginals, every representative is an
            # original feasible coordinate tuple.
            rows = _schema_record(schema.get("x-polylogue-observed-distribution")).get("pointer_shapes")
            if not isinstance(rows, list) or not rows:
                raise ValueError("persisted pointer width joint is missing; re-infer the package")
            joint = PointerShapeStats.from_payload(rows)
            keys = sorted(joint.bins)
            selected_key = rng.choices(keys, weights=[joint.bins[key][1] for key in keys], k=1)[0]
            selected = rng.choice(joint.extrema[selected_key])
            length, newlines, start, prefix_newlines, width, basename = selected
            directory_width = width - basename
            directory: GeneratedString = (
                "" if directory_width == 0 else joined_text_value(SampledText(directory_width - 1, "x"), "/")
            )
            filename: GeneratedString = (
                sampled_text_value(
                    SampledText(
                        basename, "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(min(basename, 256)))
                    )
                )
                if basename
                else ""
            )
            if excluded and basename:
                first = _identity_character(
                    "abcdefghijklmnopqrstuvwxyz", excluded, start + directory_width, length, rng
                )
                filename = joined_text_value(first, SampledText(basename - 1, "x"))
            if plan is not None and pointer_width is not None:
                identity = plan.tool_fields[pointer_width[0]][3]
                if identity is None:
                    raise RuntimeError("pointer field lost its original identity dependency")
                if plan.tool_pointer_relations[path] == "own-direct":
                    suffix_width = basename - len(identity)
                    if suffix_width < 0:
                        raise ValueError("original direct pointer width cannot represent its identity")
                    filename = (
                        identity
                        if suffix_width == 0
                        else joined_text_value(identity, ".", SampledText(suffix_width - 1, "x"))
                    )
                elif isinstance(filename, str) and isinstance(identity, str) and filename.rsplit(".", 1)[0] == identity:
                    filename = "y" * basename if identity != "y" * basename else "z" * basename
            if not basename:
                directory = SampledText(width, "/")
            whitespace = SampledText(start - len(PERSISTED_OUTPUT_PREFIX), " ", newlines=prefix_newlines)
            tail_length = length - start - width
            tail_newlines = newlines - prefix_newlines
            if tail_length == 0:
                tail: GeneratedString = ""
            else:
                separator = "\n" if tail_newlines else " "
                tail = joined_text_value(
                    separator, SampledText(tail_length - 1, "x", newlines=max(0, tail_newlines - 1))
                )
            return joined_text_value(PERSISTED_OUTPUT_PREFIX, whitespace, directory, filename, tail)
        case "uuid4" | "uuid":
            first = _identity_character("0123456789abcdef", excluded, 0, 36, rng) if excluded else ""
            uuid_bits = (rng.getrandbits(124) | (int(first, 16) << 124)) if excluded else rng.getrandbits(128)
            return _fixed_format_value(schema, str(uuid.UUID(int=uuid_bits, version=4)), rng)
        case "hex-id":
            length = _format_length(schema, rng, "hex-id", 24)
            if excluded:
                first = _identity_character("0123456789abcdef", excluded, 0, length, rng)
                return joined_text_value(first, SampledText(length - 1, rng.randbytes(2048).hex()))
            if length <= 1 << 20:
                return rng.randbytes((length + 1) // 2).hex()[:length]
            return SampledText(length, rng.randbytes(2048).hex())
        case "iso8601":
            iso_width = _format_length(schema, rng, "iso8601", 32) if excluded else None
            timestamp = (
                self._semantic_gen.base_ts + self._semantic_gen.turn_index * 60
                if semantic_role == "message_timestamp" and self._semantic_gen is not None
                else rng.uniform(1670000000, 1760000000)
            )
            moment = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            if excluded:
                assert iso_width is not None
                digit = _identity_character("0123456789", excluded, 3, iso_width, rng)
                moment = moment.replace(year=2020 + int(digit), day=min(moment.day, 28))
            iso_value = _iso8601_like(
                moment, iso_width if iso_width is not None else _format_length(schema, rng, "iso8601", 32)
            )
            return _numeric_format_spelling(iso_value, character_class)
        case "unix-epoch" | "unix-epoch-str":
            length = _format_length(schema, rng, "unix-epoch-str", 17)
            timestamp = (
                self._semantic_gen.base_ts + self._semantic_gen.turn_index * 60
                if semantic_role == "message_timestamp" and self._semantic_gen is not None
                else rng.uniform(1670000000, 1760000000)
            )
            integer = str(int(timestamp))
            if excluded:
                integer = integer[:9] + _identity_character("0123456789", excluded, 9, length, rng)
            epoch_value = (
                integer
                if length == 10
                else integer + "." + f"{timestamp % 1:.6f}"[2:].ljust(length - 11, "0")[: length - 11]
            )
            return _numeric_format_spelling(epoch_value, character_class)
        case "url":
            length = _format_length(schema, rng, "url", 24)
            alphabet = "éöž" if character_class == "non_ascii" else "abcdefghijklmnopqrstuvwxyz"
            host = _identity_character(alphabet, excluded, 7, length, rng) if excluded else alphabet[0]
            return (
                "http://" + host
                if length == 8
                else joined_text_value("http://" + host + "/", SampledText(length - 9, "a"))
            )
        case "email":
            length = _format_length(schema, rng, "email", 18)
            alphabet = "éöž" if character_class == "non_ascii" else "abcdefghijklmnopqrstuvwxyz"
            letter = _identity_character(alphabet, excluded, 0, length, rng) if excluded else alphabet[0]
            return joined_text_value(SampledText(length - 4, letter), "@b.c")
        case "mime-type":
            length = _format_length(schema, rng, "mime-type", 10)
            alphabet = "ıſ" if character_class == "non_ascii" else "abcdefghijklmnopqrstuvwxyz"
            letter = _identity_character(alphabet, excluded, 0, length, rng) if excluded else alphabet[0]
            return joined_text_value(letter + "/", SampledText(length - 2, "b"))
        case "base64":
            length = _format_length(schema, rng, "base64", 48)
            if excluded:
                if not length:
                    raise ValueError("independent identity has no nonempty base64 shape")
                alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
                first = _identity_character(alphabet, excluded, 0, length, rng)
                # Each quartet encodes three bytes. Select its first six bits
                # before spelling, preserving valid base64 and the drawn width.
                return joined_text_value(first, "AAA", SampledText(length - 4, "A"))
            if length <= 1 << 20:
                return base64.b64encode(rng.randbytes((length + 3) // 4 * 3)).decode("ascii")[:length]
            return SampledText(length, base64.b64encode(rng.randbytes(3072)).decode("ascii"))

    buckets = _categorical_buckets(schema)
    if buckets is not None:
        # One stable synthetic value per observed hashed bucket, drawn with the
        # observed bucket frequencies, so repeated values repeat as they did.
        annotations = {
            name: dict(_observed_distribution(schema, name))
            for name in ("categorical", "string_length", "newline_count")
        }
        categories = _schema_record(_schema_record(schema.get("x-polylogue-observed-distribution")).get("categories"))
        annotations["categories"] = categories
        annotations["character_shapes"] = dict(_observed_distribution(schema, "character_shapes"))
        for annotation in annotations.values():
            for key in ("histogram", "bucket_histogram"):
                rows = annotation.get(key)
                if isinstance(rows, list):
                    annotation[key] = sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))
        fingerprint = hashlib.sha256(
            json.dumps(
                {"annotations": annotations, "excluded": sorted(excluded)} if excluded else annotations,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cached = self._categorical_pools.get(path)
        pool = (
            cached[1]
            if cached is not None and cached[0] == fingerprint and agent_prefix is None and not fresh_tool
            else None
        )
        if pool is None:
            lengths = observed_length_support(_observed_distribution(schema, "string_length")) or {1: 1}
            pool = categorical_codebook(
                buckets,
                lengths,
                seed=f"{self.provider}|{path}|{fingerprint}",
                newlines=observed_length_support(_observed_distribution(schema, "newline_count")),
                category_shapes={
                    int(bucket): (
                        observed_length_support(_schema_record(_schema_record(value).get("string_length"))),
                        observed_length_support(_schema_record(_schema_record(value).get("newline_count"))),
                    )
                    for bucket, value in categories.items()
                }
                if categories
                else None,
                category_character_shapes={
                    int(bucket): {
                        name: (
                            observed_length_support(_schema_record(row.get("string_length"))),
                            observed_length_support(_schema_record(row.get("newline_count"))),
                            cast(int, _schema_record(row.get("string_length"))["count"]),
                        )
                        for name, row in measured_character_shapes(
                            _schema_record(value),
                            cast(int, _schema_record(_schema_record(value).get("string_length"))["count"]),
                        ).items()
                    }
                    for bucket, value in categories.items()
                }
                if categories
                else None,
                excluded=excluded,
                prefix="agent-" if agent_prefix else "",
                accepts=accepted_identity if agent_prefix is not None or fresh_tool else None,
            )
            # Replace this path entry rather than retaining every historical
            # branch/distribution version in the corpus.
            self._categorical_pools[path] = (fingerprint, pool)
        return rng.choices([value for value, _ in pool], weights=[count for _, count in pool], k=1)[0]

    schema = _condition_character_shape(schema, rng)
    character_class_value = _schema_record(schema.get("x-polylogue-observed-distribution")).get("character_class")
    character_class = character_class_value if isinstance(character_class_value, str) else None
    text_length, newlines = _sample_text_shape(schema, rng, minimum_plain=int(character_class == "non_ascii"))
    if (excluded or agent_prefix is not None or fresh_tool) and text_length is None:
        text_length = max(32, newlines + 1)
    if (excluded or agent_prefix is not None or fresh_tool) and text_length is not None:
        # Anonymous spelling is already owned by the category sampler. Borrow
        # its exact shape domain, excluding aliases before choosing a value.
        from polylogue.schemas.synthetic.text_value import text_equals

        if plan is not None and (fresh_tool or agent_prefix is not None):
            value = plan.owner.fresh_shape_identity(
                "agent" if agent_prefix is not None else "tool",
                text_length,
                newlines,
                prefix="agent-" if agent_prefix else "",
                excluded=excluded,
                character_class=character_class,
            )
        else:
            value = next(
                (
                    value
                    for value in _strings_of_shape(
                        text_length, newlines, prefix="agent-" if agent_prefix else "", character_class=character_class
                    )
                    if not any(text_equals(value, item) for item in excluded) and accepted_identity(value)
                ),
                None,
            )
        if value is None:
            raise ValueError("independent identity has no supported original text shape")
        return value
    proposal: str | None = None
    if self._semantic_gen is not None:
        handled, semantic_value = self._semantic_gen.try_generate(schema)
        if handled and isinstance(semantic_value, str):
            proposal = semantic_value
    if text_length is None and _observed_distribution(schema, "newline_count"):
        text_length = max(len(proposal) if proposal is not None else 32, newlines)
    if text_length is not None:
        return _free_text(rng, text_length, newlines, prefix=proposal or "", character_class=character_class)
    if proposal is not None:
        return proposal
    if schema.get("x-polylogue-multiline"):
        return _text_for_role(rng, "assistant")

    return f"synthetic-{rng.randint(0, 99999)}"


def _identity_character(alphabet: str, excluded: frozenset[str], position: int, length: int, rng: random.Random) -> str:
    """Condition neutral spelling before allocation; no field is redrawn."""
    forbidden = {value[position] for value in excluded if len(value) == length and position < len(value)}
    supported = [character for character in alphabet if character not in forbidden]
    if not supported:
        raise ValueError("independent identity has exhausted its declared spelling domain")
    return rng.choice(supported)


def _condition_width_partition(
    shape: SchemaRecord, *, width: int | None, geometry: tuple[int, int, str] | None
) -> SchemaRecord | None:
    """Restrict the existing field partition to its original relation draw.

    Count weights remain measured. No new spelling or length sampler is used;
    the conditional relation supplies the exact supported ID width and filters
    original pointer witnesses before the field's category/shape draw.
    """
    result: SchemaRecord = _ProjectedStringShape(shape, dict(shape))
    if "character_shapes" in shape:
        original_count = _schema_record(shape.get("string_length")).get("count")
        if type(original_count) is not int:
            raise ValueError("original character shape has no count")
        measured_character_shapes(shape, original_count)
    if width is not None:
        lengths = _schema_record(shape.get("string_length"))
        rows = lengths.get("histogram")
        if not isinstance(rows, list):
            return None
        matching = [
            row for row in rows if isinstance(row, list) and len(row) == 2 and row[0] == _bucket_index(float(width))
        ]
        if not matching:
            return None
        result["string_length"] = {
            **lengths,
            "min": width,
            "max": width,
            "histogram": cast(SchemaValue, matching),
            "count": sum(row[1] for row in matching if isinstance(row[1], int)),
        }
    if geometry is not None and not isinstance(shape.get("categories"), dict):
        payload = shape.get("pointer_shapes")
        if not isinstance(payload, list):
            return None
        joint = PointerShapeStats.from_payload(payload)
        point_rows: list[SchemaValue] = []
        for key in sorted(joint.bins):
            points = [
                point
                for point in joint.extrema[key]
                if point[5] == geometry[1] and pointer_width_relation(geometry[0], point[5]) == geometry[2]
            ]
            if points:
                point_rows.append(
                    {"count": joint.bins[key][1], "points": cast(SchemaValue, [list(point) for point in points])}
                )
        if not point_rows:
            return None
        result["pointer_shapes"] = point_rows
    categories = shape.get("categories")
    if isinstance(categories, dict):
        selected = {
            bucket: filtered
            for bucket, category in categories.items()
            if isinstance(category, dict)
            and (filtered := _condition_width_partition(category, width=width, geometry=geometry)) is not None
        }
        if not selected:
            return None
        result["categories"] = cast(SchemaValue, selected)
        categorical = _schema_record(shape.get("categorical"))
        histogram = categorical.get("bucket_histogram")
        if isinstance(histogram, list):
            result["categorical"] = {
                **categorical,
                "bucket_histogram": [row for row in histogram if isinstance(row, list) and str(row[0]) in selected],
            }
    character_shapes = shape.get("character_shapes")
    if isinstance(character_shapes, dict):
        selected_characters = {
            name: filtered
            for name, row in character_shapes.items()
            if isinstance(row, dict)
            and (filtered := _condition_width_partition(row, width=width, geometry=geometry)) is not None
        }
        if not selected_characters:
            return None
        result["character_shapes"] = cast(SchemaValue, selected_characters)
    return result


def _condition_character_shape(schema: SchemaRecord, rng: random.Random) -> SchemaRecord:
    observed = _schema_record(schema.get("x-polylogue-observed-distribution"))
    if "character_shapes" not in observed or "character_class" in observed:
        return schema
    count = _schema_record(observed.get("string_length")).get("count")
    if type(count) is not int:
        raise ValueError("original character shape has no observation count")
    shapes = measured_character_shapes(observed, count)
    names = sorted(shapes)
    name = rng.choices(
        names, weights=[cast(int, _schema_record(shapes[item].get("string_length"))["count"]) for item in names], k=1
    )[0]
    return {
        **schema,
        "x-polylogue-observed-distribution": {**observed, **shapes[name], "character_class": name},
    }


def _condition_string_format(
    schema: SchemaRecord,
    rng: random.Random,
    *,
    pointer: bool | None = None,
    width: int | None = None,
    geometry: tuple[int, int, str] | None = None,
) -> SchemaRecord:
    """Choose a measured format partition before drawing its string shape.

    A dominant-format annotation is not a joint distribution. In particular,
    a UUID and a non-UUID identifier may share a field and a coarse length
    bucket. Only collector-owned conditional sketches can separate those
    observations; the marginal histogram remains intact in the schema.
    """
    observed = _schema_record(schema.get("x-polylogue-observed-distribution"))
    if "string_formats" not in observed:
        if width is None:
            return schema
        conditioned = _condition_width_partition(observed, width=width, geometry=None)
        if conditioned is None:
            raise ValueError("original tool ID has no measured field width support")
        return {**schema, "x-polylogue-observed-distribution": conditioned}
    shapes = measured_string_formats(schema)
    if width is not None or geometry is not None:
        shapes = {
            fmt: filtered
            for fmt, shape in shapes.items()
            if (filtered := _condition_width_partition(shape, width=width, geometry=geometry)) is not None
        }
        if not shapes:
            raise ValueError("original tool/pointer width joint has no field support")
    if pointer is not None:
        shapes = {fmt: shape for fmt, shape in shapes.items() if (fmt == "persisted-tool-output") == pointer}
        if not shapes:
            raise ValueError("selected result has no measured string-format support")
    counts: dict[str, int] = {}
    for fmt, shape in shapes.items():
        count = _schema_record(shape.get("string_length"))["count"]
        assert isinstance(count, int) and not isinstance(count, bool)
        counts[fmt] = count
    fmt = rng.choices(sorted(counts), weights=[counts[key] for key in sorted(counts)], k=1)[0]
    shape = _schema_record(shapes[fmt])
    if fmt:
        # A formatted value still belongs to an observed anonymous category.
        # Its conditional numeric shape keeps the measured length/newline
        # joint; drawing the two format-wide marginals can invent a pair.
        categorical = _schema_record(shape.get("categorical"))
        histogram = categorical.get("bucket_histogram")
        assert isinstance(histogram, list)
        weights: dict[str, int] = {}
        for row in histogram:
            assert isinstance(row, list) and len(row) == 2
            assert isinstance(row[0], int) and isinstance(row[1], int)
            weights[str(row[0])] = row[1]
        bucket = rng.choices(sorted(weights), weights=[weights[key] for key in sorted(weights)], k=1)[0]
        category_shape = _schema_record(_schema_record(shape.get("categories")).get(bucket))
        shape = _ProjectedStringShape(category_shape, {**shape, **category_shape})
    conditioned = {**schema, "x-polylogue-observed-distribution": _ProjectedStringShape(shape, {**observed, **shape})}
    if fmt:
        conditioned["x-polylogue-format"] = fmt
    else:
        conditioned.pop("x-polylogue-format", None)
    return _condition_character_shape(conditioned, rng) if fmt else conditioned


def _numeric_format_spelling(value: str, character_class: str | None) -> str:
    """Spell the same drawn numeric format with neutral Unicode digits."""
    if character_class != "non_ascii":
        return value
    return value.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))


def _iso8601_like(moment: datetime, length: int | None) -> str:
    """The ISO 8601 spelling whose length matches the observed one (``Z`` and millis vary by source)."""
    if length is None:
        return moment.isoformat()
    if not format_length_is_representable("iso8601", length):
        raise ValueError("observed string length cannot represent the ISO 8601 timestamp spelling")
    base = moment.strftime("%Y-%m-%dT%H:%M:%S")
    if length == 19:
        return base
    if length == 20:
        return base + "Z"
    if length == 21:
        return base + "." + f"{moment.microsecond:06d}"[0]
    return base + "." + f"{moment.microsecond:06d}".ljust(length - 21, "0")[: length - 21] + "Z"


def _format_length(schema: SchemaRecord, rng: random.Random, fmt: str, default: int) -> int:
    if any(observed_length_support(_observed_distribution(schema, "newline_count"))):
        raise ValueError("observed newline count contradicts the required wire format")
    length = _sampled_length(schema, "string_length", rng)
    if length is None:
        return default
    if not format_length_is_representable(fmt, length):
        raise ValueError("observed string length contradicts the required wire format")
    return length


def _fixed_format_value(schema: SchemaRecord, value: str, rng: random.Random) -> str:
    length = _format_length(schema, rng, str(schema.get("x-polylogue-format")), len(value))
    if length != len(value):
        raise ValueError("observed string length contradicts the required wire format")
    return value


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
    elif self._semantic_gen is not None and schema.get("x-polylogue-semantic-role") == "message_timestamp":
        _, proposal = self._semantic_gen.try_generate(schema)
        value = float(proposal) if isinstance(proposal, (int, float)) else self._semantic_gen.base_ts
    else:
        value = rng.uniform(*neutral_numeric_range(schema))
    return int(value) if is_int else value


def _walk_array(
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    path: str = "$",
) -> _GenerationFrame:
    item_schema = _schema_record(schema.get("items"))
    plan = None if self._coverage_witness_mode else self._public_record_plan
    conditioned = plan is not None and path == plan.content_path
    length_schema: SchemaRecord = (
        {"x-polylogue-observed-distribution": {"array_length": plan.content["array_block_count"]}}
        if conditioned and plan is not None
        else schema
    )
    observed_length = _sampled_length(length_schema, "array_length", rng)
    if conditioned and observed_length is None:
        raise ValueError("conditional public block count is missing; re-infer the package")
    lengths = schema.get("x-polylogue-array-lengths")
    if observed_length is not None:
        n_items = observed_length
    elif isinstance(lengths, list) and len(lengths) >= 2:
        lo = _coerce_int(lengths[0])
        hi = _coerce_int(lengths[1])
        if lo < 0 or hi < lo:
            raise ValueError("declared array length range is inconsistent")
        n_items = rng.randint(lo, hi)
    else:
        n_items = rng.randint(1, 3)
    if self._coverage_witness_mode:
        n_items = 1

    prefix = schema.get("prefixItems")
    if prefix is not None and (not isinstance(prefix, list) or any(not isinstance(item, dict) for item in prefix)):
        raise ValueError("declared array prefix schemas must be objects")
    prefix_schemas = prefix if isinstance(prefix, list) else []
    if schema.get("items") is False and n_items > len(prefix_schemas):
        raise ValueError("positive observed array length conflicts with forbidden items")
    item_allows_null = _schema_allows_null(item_schema)

    item_path = f"{path}.items[*]" if self._coverage_witness_mode else f"{path}[*]"
    owner = self._relation_solver.fk_graph
    items = owner.new_array()
    all_numeric = True
    grammar = (
        iter(ConditionalContentKinds(self.provider, plan.kind, plan.content, n_items, owner).sample(rng))
        if conditioned and plan is not None
        else None
    )
    discriminator: str | None = None
    try:
        for ordinal in range(n_items):
            original_item_schema = (
                _schema_record(prefix_schemas[ordinal]) if ordinal < len(prefix_schemas) else item_schema
            )
            item_allows_null = _schema_allows_null(original_item_schema)
            if grammar is not None and plan is not None:
                if discriminator is not None:
                    plan.scalars.pop(discriminator, None)
                plan.clear_tool_operand(item_path)
                discriminator = plan.bind_content_kind(next(grammar), item_path, rng)
            value = yield (original_item_schema, item_path, None)
            if value is None and not item_allows_null:
                if conditioned:
                    raise ValueError("conditional public block was not represented by its selected schema")
                continue
            all_numeric = all_numeric and isinstance(value, (int, float)) and not isinstance(value, bool)
            owner.append_array_item(items, value)
    finally:
        if grammar is not None:
            grammar.close()
        if plan is not None and discriminator is not None:
            plan.scalars.pop(discriminator, None)
            plan.clear_tool_operand(item_path)
    ordered = _observed_distribution(item_schema, "ordered_numeric_pairs")
    rate = ordered.get("nondecreasing_rate")
    if (
        isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and len(items)
        and not prefix_schemas
        and all_numeric
        and rng.random() < rate
    ):
        return owner.sorted_numeric_array(items)
    return items


def _json_frame(value: JSONValue, level: int, *, pretty: bool) -> Generator[str | tuple[JSONValue, int], None, None]:
    if isinstance(value, dict | GeneratedRecord):
        yield "{"
        for index, (key, child) in enumerate(value.items()):
            if index:
                yield ","
            if pretty:
                yield "\n" + "  " * (level + 1)
            yield json.dumps(key) + (": " if pretty else ":")
            yield child, level + 1
        if value and pretty:
            yield "\n" + "  " * level
        yield "}"
    elif isinstance(value, list | GeneratedArray):
        yield "["
        for index, child in enumerate(value):
            if index:
                yield ","
            if pretty:
                yield "\n" + "  " * (level + 1)
            yield child, level + 1
        if value and pretty:
            yield "\n" + "  " * level
        yield "]"
    elif isinstance(value, SampledText | JoinedText):
        yield '"'
        for piece in value.pieces():
            yield json.dumps(piece, ensure_ascii=True)[1:-1]
        yield '"'
    else:
        yield json.dumps(value, allow_nan=False)


def _json_chunks(value: JSONValue, *, pretty: bool) -> Generator[str, None, None]:
    """Encode through lazy container frames rather than a native depth limit."""
    frames = [_json_frame(value, 0, pretty=pretty)]
    try:
        while frames:
            try:
                piece = next(frames[-1])
            except StopIteration:
                frames.pop()
                continue
            if isinstance(piece, str):
                yield piece
            else:
                child, level = piece
                frames.append(_json_frame(child, level, pretty=pretty))
    finally:
        for frame in reversed(frames):
            frame.close()


def iter_serialized_chunks(self: _SyntheticRuntimeContext, data: JSONValue) -> Generator[bytes, None, None]:
    """The serializer owns both explicit fixture collection and streamed writes."""
    for _ in walk_generated_values(data):
        pass
    normalize_browser_capture_attachments(data)
    validate_wire_payload(self.provider, data)
    if self.wire_format.encoding == "jsonl":
        if not isinstance(data, list | GeneratedArray):
            raise ValueError("JSONL wire format requires a list payload")
        for record in data:
            for piece in _json_chunks(record, pretty=False):
                yield piece.encode("utf-8")
            yield b"\n"
    else:
        for piece in _json_chunks(data, pretty=True):
            yield piece.encode("utf-8")


def _serialize(self: _SyntheticRuntimeContext, data: JSONValue) -> bytes:
    return b"".join(iter_serialized_chunks(self, data))


__all__ = [
    "SCHEMA_CONSTRUCT_HANDLERS",
    "_generate_from_schema",
    "_generate_number",
    "_generate_string",
    "_serialize",
    "iter_serialized_chunks",
]
