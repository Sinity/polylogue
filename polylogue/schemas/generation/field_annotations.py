"""Field-stat-driven schema annotation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import overload

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.field_stats.models import StringCharacterStats
from polylogue.schemas.field_stats.stats import FieldStats
from polylogue.schemas.json_walk import require_finite_json
from polylogue.schemas.privacy import (
    PUBLISHABLE_VOCABULARY_ROLES,
    _is_content_field,
    _is_safe_enum_value,
)
from polylogue.schemas.privacy_config import SchemaPrivacyConfig

_ENUM_MIN_COUNT = 2
_ENUM_MIN_FREQ = 0.03


def _enum_values(
    field_stats: FieldStats,
    *,
    path: str,
    privacy_config: SchemaPrivacyConfig | None,
) -> list[JSONValue]:
    """Every observed value of an enum-like leaf that clears the declared filters.

    The population is bounded twice already: the caller publishes only from a
    slot in ``PUBLISHABLE_VOCABULARY_ROLES``, and only when
    ``FieldStats.is_enum_like`` holds -- ``len(observed_values) <=
    ENUM_MAX_CARDINALITY``. Inside those bounds the list is exhaustive.

    A third bound, a flat ceiling of 20 output entries, used to cut it in
    silence. Measured on a ``message_role`` slot carrying 25 distinct tokens
    each observed 20 times -- every one past ``_ENUM_MIN_COUNT`` and
    ``_ENUM_MIN_FREQ``, none privacy-filtered -- the annotation recorded 20 and
    dropped five, with nothing in ``loss_inventory`` saying so. A published
    slot is a *declared protocol vocabulary*: the member list is the answer,
    and a partial one is indistinguishable from a complete one to
    :func:`polylogue.sources.origin_specs.schema_observed_leaf_values`, which
    reads it as every value the committed schema recorded.

    Privacy filters stay for every value. Adapter-declared public tokens retain
    every observation; other allowed values retain count/frequency selection.
    A rare public record branch must remain realizable under this vocabulary.
    """
    from polylogue.schemas.record_structure import published_kind_tokens

    public_tokens = published_kind_tokens()
    total = max(field_stats.value_count, 1)
    min_count = _ENUM_MIN_COUNT if total >= 20 else 1
    min_freq = _ENUM_MIN_FREQ if total >= 50 else 0.0
    values: list[JSONValue] = []
    for value, count in field_stats.observed_values.most_common():
        if not _is_safe_enum_value(value, path=path, config=privacy_config):
            continue
        # A measured public discriminator is executable field support even
        # when its branch is rare. Selection filters cannot turn the same
        # original record grammar into an impossible schema vocabulary.
        if value not in public_tokens:
            if count < min_count:
                continue
            if min_freq and (count / total) < min_freq:
                continue
        values.append(value)
    # A retained safe counter is a projection, not the complete vocabulary.
    # x-polylogue-values constrains string draws only; null/numeric branches
    # retain their separate type support. Never turn a filtered subset into
    # the field's exhaustive string domain.
    string_count = field_stats.type_counts.get("string", field_stats.value_count if not field_stats.type_counts else 0)
    if sum(field_stats.observed_values[value] for value in values if isinstance(value, str)) != string_count:
        return []
    return values


def _schema_node(value: JSONValue | object) -> JSONDocument | None:
    return value if isinstance(value, dict) else None


def _character_shape_payload(shapes: Mapping[str, StringCharacterStats], count: int) -> JSONDocument:
    if not shapes or not set(shapes) <= {"ascii", "non_ascii"}:
        raise ValueError("string character evidence requires complete reinference")
    if sum(shape.length.count for shape in shapes.values()) != count or any(
        shape.length.count <= 0 or shape.newlines.count != shape.length.count for shape in shapes.values()
    ):
        raise ValueError("string character evidence counts disagree")
    return {
        name: {
            "string_length": shape.length.to_payload(),
            "newline_count": shape.newlines.to_payload(),
            **({"pointer_shapes": shape.pointer_shapes.to_payload()} if shape.pointer_shapes.bins else {}),
        }
        for name, shape in sorted(shapes.items())
    }


def _observed_distribution_payload(field_stats: FieldStats) -> JSONDocument:
    payload: JSONDocument = {
        "documents": field_stats.total_samples,
        "encountered_documents": field_stats.document_encountered_count,
        "non_null_documents": field_stats.document_non_null_count,
        "null_observations": field_stats.null_count,
        "missing_documents": max(0, field_stats.total_samples - field_stats.document_encountered_count),
        "type_counts": dict(sorted(field_stats.type_counts.items())),
    }
    distributions = {
        "string_length": field_stats.string_length_distribution,
        "newline_count": field_stats.newline_distribution,
        "array_length": field_stats.array_length_distribution,
        "object_fanout": field_stats.object_fanout_distribution,
    }
    numeric = field_stats.numeric_distribution
    if numeric.count or numeric.non_finite_count:
        payload["numeric"] = {"count": numeric.count, "non_finite_count": numeric.non_finite_count}
    for name, distribution in distributions.items():
        if distribution.count or distribution.non_finite_count:
            payload[name] = distribution.to_payload()
    if field_stats.ordered_pair_count:
        payload["ordered_numeric_pairs"] = {
            "count": field_stats.ordered_pair_count,
            "nondecreasing_count": field_stats.ordered_increasing_pair_count,
            "nondecreasing_rate": field_stats.monotonicity_score,
        }
    if field_stats.co_occurring_fields:
        payload["co_occurring_fields"] = dict(sorted(field_stats.co_occurring_fields.items()))
    if field_stats.categorical_distribution.count:
        payload["categorical"] = field_stats.categorical_distribution.to_payload()
    if field_stats.object_key_distribution.count:
        payload["object_keys"] = field_stats.object_key_distribution.to_payload()
    if field_stats.co_occurrence_distribution.count:
        payload["co_occurrence_hashes"] = field_stats.co_occurrence_distribution.to_payload()
    if field_stats.boolean_counts:
        payload["boolean_counts"] = dict(sorted(field_stats.boolean_counts.items()))
    if field_stats.detected_formats:
        payload["format_counts"] = dict(sorted(field_stats.detected_formats.items()))
    if field_stats.string_formats:
        if (
            sum(shape.length.count for shape in field_stats.string_formats.values())
            != field_stats.string_length_distribution.count
        ):
            raise ValueError("incomplete string format evidence requires reinference")
        for shape in field_stats.string_formats.values():
            if (
                set(shape.categories) != set(shape.categorical.buckets)
                or sum(category.length.count for category in shape.categories.values()) != shape.length.count
            ):
                raise ValueError("incomplete string category evidence requires reinference")
            for bucket, category in shape.categories.items():
                if (
                    category.length.count != shape.categorical.buckets[bucket]
                    or category.newlines.count != category.length.count
                ):
                    raise ValueError("string category evidence counts disagree")
        for fmt, shape in field_stats.string_formats.items():
            if fmt == "persisted-tool-output" and any(
                sum(count for _point, count in category.pointer_shapes.bins.values()) != category.length.count
                for category in shape.categories.values()
            ):
                raise ValueError("pointer shape joint requires complete reinference")
        payload["string_formats"] = {
            fmt: {
                "string_length": shape.length.to_payload(),
                "newline_count": shape.newlines.to_payload(),
                "categorical": shape.categorical.to_payload(),
                "character_shapes": _character_shape_payload(shape.character_shapes, shape.length.count),
                "categories": {
                    str(bucket): {
                        "string_length": category.length.to_payload(),
                        "newline_count": category.newlines.to_payload(),
                        "character_shapes": _character_shape_payload(category.character_shapes, category.length.count),
                        **(
                            {"pointer_shapes": category.pointer_shapes.to_payload()}
                            if fmt == "persisted-tool-output"
                            else {}
                        ),
                    }
                    for bucket, category in sorted(shape.categories.items())
                },
            }
            for fmt, shape in sorted(field_stats.string_formats.items())
        }
    if field_stats.truncated_evidence:
        payload["loss_inventory"] = dict(sorted(field_stats.truncated_evidence.items()))
    return payload


def _set_frequency(schema: JSONDocument, frequency: float | None) -> None:
    schema.pop("x-polylogue-frequency", None)
    if frequency is not None and 0.0 < frequency < 0.95:
        schema["x-polylogue-frequency"] = frequency


def _apply_property_presence(schema: JSONDocument, occurrences: int, containers: int) -> None:
    if containers <= 0 or occurrences < 0 or occurrences > containers:
        raise ValueError("observed property count contradicts its containing-object count")
    observed = _schema_node(schema.get("x-polylogue-observed-distribution"))
    if observed is None:
        raise ValueError("observed property has no original occurrence evidence")
    observed["parent_object_observations"] = containers
    _set_frequency(schema, occurrences / containers)


def _annotate_node(
    schema: JSONValue,
    stats: Mapping[str, FieldStats],
    path: str = "$",
    *,
    property_owner_path: str | None = None,
    alias_branch: bool = False,
    privacy_config: SchemaPrivacyConfig | None = None,
) -> JSONValue:
    """Apply x-polylogue-* annotations to a schema from collected field stats."""
    schema_node = _schema_node(schema)
    if schema_node is None:
        return schema

    field_stats = stats.get(path)
    if field_stats:
        observed = _observed_distribution_payload(field_stats)
        schema_node["x-polylogue-observed-distribution"] = observed
        _set_frequency(schema_node, None if alias_branch else field_stats.document_frequency)
        if property_owner_path is not None:
            owner = stats.get(property_owner_path)
            if owner is None:
                raise ValueError("observed property has no original containing-object evidence")
            # The object is already selected. Root-document frequency would
            # thin its population again and lose repeated array/map occurrences.
            _apply_property_presence(schema_node, sum(field_stats.type_counts.values()), owner.type_counts["object"])
        # Union branches keep their observed type counts, not the enclosing
        # property's repeated presence rate, as the original selection weights.

        if field_stats.field_first_seen:
            schema_node["x-polylogue-field-first-seen"] = field_stats.field_first_seen
        if field_stats.field_last_seen:
            schema_node["x-polylogue-field-last-seen"] = field_stats.field_last_seen

        fmt = field_stats.dominant_format
        if fmt:
            schema_node["x-polylogue-format"] = fmt

        id_formats = {"uuid4", "uuid", "hex-id", "base64"}
        # A member list is published only from a slot this repository has
        # declared a protocol vocabulary (PUBLISHABLE_VOCABULARY_ROLES).  Every
        # other enum-like field keeps its type, frequency and distribution and
        # publishes no observed member, because a committed package is public
        # and no value-shape heuristic can tell a wire constant from a
        # recurring private token.
        sem_role = schema_node.get("x-polylogue-semantic-role")
        if (
            isinstance(sem_role, str)
            and sem_role in PUBLISHABLE_VOCABULARY_ROLES
            and fmt not in id_formats
            and not _is_content_field(path)
        ):
            schema_node.pop("x-polylogue-values", None)
        if (
            isinstance(sem_role, str)
            and sem_role in PUBLISHABLE_VOCABULARY_ROLES
            and field_stats.is_enum_like
            and field_stats.observed_values
            and fmt not in id_formats
            and not _is_content_field(path)
        ):
            enum_values = _enum_values(
                field_stats,
                path=path,
                privacy_config=privacy_config,
            )
            if enum_values:
                schema_node["x-polylogue-values"] = enum_values

        if field_stats.array_length_distribution.count:
            minimum = field_stats.array_length_distribution.minimum
            maximum = field_stats.array_length_distribution.maximum
            if minimum is not None and maximum is not None:
                schema_node["x-polylogue-array-lengths"] = [int(minimum), int(maximum)]

        if (
            field_stats.is_multiline
            and field_stats.value_count
            and field_stats.is_multiline / field_stats.value_count > 0.3
        ):
            schema_node["x-polylogue-multiline"] = True

        if field_stats.ref_target:
            schema_node["x-polylogue-ref"] = field_stats.ref_target

    return schema_node


def _schema_children(node: JSONDocument, path: str) -> list[tuple[JSONDocument, str, str | None, bool]]:
    children: list[tuple[JSONDocument, str, str | None, bool]] = []
    properties = _schema_node(node.get("properties"))
    if properties is not None:
        for name, value in properties.items():
            child = _schema_node(value)
            if child is not None:
                children.append((child, f"{path}.{name}", path, False))
    for keyword, suffix in (("additionalProperties", ".*"), ("items", "[*]")):
        child = _schema_node(node.get(keyword))
        if child is not None:
            children.append((child, path + suffix, None, False))
    for keyword in ("anyOf", "oneOf", "allOf"):
        values = node.get(keyword)
        if isinstance(values, list):
            for value in values:
                child = _schema_node(value)
                if child is not None:
                    children.append((child, path, None, True))
    return children


def reannotate_property_presence(schema: JSONDocument) -> JSONDocument:
    """Publish corrected presence from retained exact original type counts.

    This producer pass changes only presence annotations. It keeps original
    distributions, privacy decisions and schema support intact, and refuses a
    property whose containing-object measurement is unavailable.
    """
    require_finite_json(schema)
    pending: list[tuple[JSONDocument, JSONDocument | None, bool, str]] = [(schema, None, False, "$")]
    while pending:
        node, owner, alias, path = pending.pop()
        if alias:
            _set_frequency(node, None)
        if owner is not None:
            observed = _schema_node(node.get("x-polylogue-observed-distribution"))
            parent = _schema_node(owner.get("x-polylogue-observed-distribution"))
            counts = _schema_node(observed.get("type_counts")) if observed is not None else None
            parent_counts = _schema_node(parent.get("type_counts")) if parent is not None else None
            if (
                counts is None
                or parent_counts is None
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values())
            ):
                raise ValueError("retained property occurrence evidence is unavailable")
            containers = parent_counts.get("object")
            if not isinstance(containers, int) or isinstance(containers, bool):
                raise ValueError("retained containing-object occurrence evidence is unavailable")
            occurrences = sum(value for value in counts.values() if isinstance(value, int))
            _apply_property_presence(node, occurrences, containers)
        pending.extend(
            (child, node if owner_path is not None else None, child_alias, child_path)
            for child, child_path, owner_path, child_alias in reversed(_schema_children(node, path))
        )
    return schema


@overload
def annotate_schema(
    schema: JSONDocument,
    stats: Mapping[str, FieldStats],
    path: str = "$",
    *,
    privacy_config: SchemaPrivacyConfig | None = None,
) -> JSONDocument: ...


@overload
def annotate_schema(
    schema: JSONValue,
    stats: Mapping[str, FieldStats],
    path: str = "$",
    *,
    privacy_config: SchemaPrivacyConfig | None = None,
) -> JSONValue: ...


def annotate_schema(
    schema: JSONValue,
    stats: Mapping[str, FieldStats],
    path: str = "$",
    *,
    privacy_config: SchemaPrivacyConfig | None = None,
) -> JSONValue:
    """Apply observed annotations without limiting valid schema depth."""
    require_finite_json(schema)
    node = _schema_node(schema)
    pending: list[tuple[JSONDocument, str, str | None, bool]] = [(node, path, None, False)] if node is not None else []
    while pending:
        current, current_path, property_owner_path, alias_branch = pending.pop()
        _annotate_node(
            current,
            stats,
            current_path,
            property_owner_path=property_owner_path,
            alias_branch=alias_branch,
            privacy_config=privacy_config,
        )
        pending.extend(reversed(_schema_children(current, current_path)))
    return schema


@overload
def remove_nested_required(schema: JSONDocument, depth: int = 0) -> JSONDocument: ...


@overload
def remove_nested_required(schema: JSONValue, depth: int = 0) -> JSONValue: ...


def remove_nested_required(schema: JSONValue, depth: int = 0) -> JSONValue:
    """Remove nested required declarations from generated schemas."""
    require_finite_json(schema)
    node = _schema_node(schema)
    pending = [(node, depth)] if node is not None else []
    while pending:
        current, current_depth = pending.pop()
        if current_depth > 0:
            current.pop("required", None)
        pending.extend((child, current_depth + 1) for child, _path, _owner, _alias in _schema_children(current, "$"))
    return schema


__all__ = [
    "annotate_schema",
    "remove_nested_required",
    "reannotate_property_presence",
]
