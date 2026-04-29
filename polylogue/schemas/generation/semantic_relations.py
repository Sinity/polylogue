"""Semantic-role and relational annotation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import overload

from polylogue.lib.json import JSONDocument, JSONDocumentList, JSONValue, json_document, require_json_value
from polylogue.schemas.field_stats.stats import FieldStats
from polylogue.schemas.inference.relational.inference import infer_relations
from polylogue.schemas.inference.relational.models import RelationalAnnotations
from polylogue.schemas.inference.semantic.inference import infer_semantic_roles, select_best_roles


def _schema_node(value: JSONValue | object) -> JSONDocument | None:
    return value if isinstance(value, dict) else None


def _json_value(value: object) -> JSONValue:
    return require_json_value(value, context="semantic relation annotation")


def _relation_payloads(
    relations: RelationalAnnotations,
) -> tuple[
    JSONDocumentList,
    JSONDocumentList,
    JSONDocumentList,
    JSONDocumentList,
]:
    return (
        [
            {
                "source": relation.source_path,
                "target": relation.target_path,
                "match_ratio": round(relation.match_ratio, 3),
            }
            for relation in relations.foreign_keys
        ],
        [
            {
                "field_a": relation.field_a,
                "field_b": relation.field_b,
                "min_delta": round(relation.min_delta, 1),
                "max_delta": round(relation.max_delta, 1),
                "avg_delta": round(relation.avg_delta, 1),
            }
            for relation in relations.time_deltas
        ],
        [
            {
                "parent": relation.parent_path,
                "fields": _json_value(sorted(relation.field_names)),
            }
            for relation in relations.mutual_exclusions
        ],
        [
            {
                "path": relation.path,
                "min": relation.min_length,
                "max": relation.max_length,
                "avg": round(relation.avg_length, 1),
                "stddev": round(relation.stddev, 1),
            }
            for relation in relations.string_lengths
        ],
    )


@overload
def annotate_semantic_and_relational(
    schema: JSONDocument,
    field_stats: Mapping[str, FieldStats],
    *,
    artifact_kind: str | None = None,
) -> JSONDocument: ...


@overload
def annotate_semantic_and_relational(
    schema: JSONValue,
    field_stats: Mapping[str, FieldStats],
    *,
    artifact_kind: str | None = None,
) -> JSONValue: ...


def annotate_semantic_and_relational(
    schema: JSONValue,
    field_stats: Mapping[str, FieldStats],
    *,
    artifact_kind: str | None = None,
) -> JSONValue:
    """Attach semantic-role and relational annotations to a schema."""
    schema_node = _schema_node(schema)
    if schema_node is None:
        return schema
    stats_by_path = dict(field_stats)
    candidates = infer_semantic_roles(stats_by_path, artifact_kind=artifact_kind)
    best_roles = select_best_roles(candidates)
    role_by_path: dict[str, tuple[str, float, JSONDocument]] = {}
    for role, candidate in best_roles.items():
        role_by_path[candidate.path] = (role, candidate.confidence, json_document(candidate.evidence))

    _attach_semantic_roles(schema_node, role_by_path)
    relations = infer_relations(stats_by_path)
    _attach_relational_annotations(schema_node, relations)
    return schema_node


def _attach_semantic_roles(
    schema: JSONDocument,
    role_by_path: Mapping[str, tuple[str, float, JSONDocument]],
    *,
    path: str = "$",
) -> None:
    if path in role_by_path:
        role, confidence, evidence = role_by_path[path]
        schema["x-polylogue-semantic-role"] = role
        schema["x-polylogue-score"] = round(confidence, 3)
        schema["x-polylogue-evidence"] = evidence

    properties = _schema_node(schema.get("properties"))
    if properties is not None:
        for prop_name, prop_schema in properties.items():
            child_schema = _schema_node(prop_schema)
            if child_schema is not None:
                _attach_semantic_roles(child_schema, role_by_path, path=f"{path}.{prop_name}")
    additional_properties = _schema_node(schema.get("additionalProperties"))
    if additional_properties is not None:
        _attach_semantic_roles(additional_properties, role_by_path, path=f"{path}.*")
    items = _schema_node(schema.get("items"))
    if items is not None:
        _attach_semantic_roles(items, role_by_path, path=f"{path}[*]")
    for keyword in ("anyOf", "oneOf", "allOf"):
        keyword_items = schema.get(keyword)
        if isinstance(keyword_items, Sequence) and not isinstance(keyword_items, (str, bytes, bytearray)):
            for item in keyword_items:
                child_schema = _schema_node(item)
                if child_schema is not None:
                    _attach_semantic_roles(child_schema, role_by_path, path=path)


def _attach_relational_annotations(schema: JSONDocument, relations: RelationalAnnotations) -> None:
    foreign_keys, time_deltas, mutual_exclusions, string_lengths = _relation_payloads(relations)
    if foreign_keys:
        schema["x-polylogue-foreign-keys"] = _json_value(foreign_keys)
    if time_deltas:
        schema["x-polylogue-time-deltas"] = _json_value(time_deltas)
    if mutual_exclusions:
        schema["x-polylogue-mutually-exclusive"] = _json_value(mutual_exclusions)
    if string_lengths:
        schema["x-polylogue-string-lengths"] = _json_value(string_lengths)


__all__ = ["annotate_semantic_and_relational"]
