"""Semantic-role and relational annotation helpers."""

from __future__ import annotations

from typing import Any

from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.relational_inference import infer_relations
from polylogue.schemas.semantic_inference import infer_semantic_roles, select_best_roles


def annotate_semantic_and_relational(
    schema: dict[str, Any],
    field_stats: dict[str, FieldStats],
    *,
    artifact_kind: str | None = None,
) -> dict[str, Any]:
    """Attach semantic-role and relational annotations to a schema."""
    candidates = infer_semantic_roles(field_stats, artifact_kind=artifact_kind)
    best_roles = select_best_roles(candidates)
    role_by_path: dict[str, tuple[str, float, dict[str, Any]]] = {}
    for role, candidate in best_roles.items():
        role_by_path[candidate.path] = (role, candidate.confidence, candidate.evidence)

    _attach_semantic_roles(schema, role_by_path)
    relations = infer_relations(field_stats)
    _attach_relational_annotations(schema, relations)
    return schema


def _attach_semantic_roles(
    schema: dict[str, Any],
    role_by_path: dict[str, tuple[str, float, dict[str, Any]]],
    *,
    path: str = "$",
) -> None:
    if not isinstance(schema, dict):
        return
    if path in role_by_path:
        role, confidence, evidence = role_by_path[path]
        schema["x-polylogue-semantic-role"] = role
        schema["x-polylogue-score"] = round(confidence, 3)
        schema["x-polylogue-evidence"] = evidence

    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            _attach_semantic_roles(prop_schema, role_by_path, path=f"{path}.{prop_name}")
    if isinstance(schema.get("additionalProperties"), dict):
        _attach_semantic_roles(schema["additionalProperties"], role_by_path, path=f"{path}.*")
    if isinstance(schema.get("items"), dict):
        _attach_semantic_roles(schema["items"], role_by_path, path=f"{path}[*]")
    for keyword in ("anyOf", "oneOf", "allOf"):
        for item in schema.get(keyword, []):
            _attach_semantic_roles(item, role_by_path, path=path)


def _attach_relational_annotations(schema: dict[str, Any], relations) -> None:
    if relations.foreign_keys:
        schema["x-polylogue-foreign-keys"] = [
            {
                "source": relation.source_path,
                "target": relation.target_path,
                "match_ratio": round(relation.match_ratio, 3),
            }
            for relation in relations.foreign_keys
        ]

    if relations.time_deltas:
        schema["x-polylogue-time-deltas"] = [
            {
                "field_a": relation.field_a,
                "field_b": relation.field_b,
                "min_delta": round(relation.min_delta, 1),
                "max_delta": round(relation.max_delta, 1),
                "avg_delta": round(relation.avg_delta, 1),
            }
            for relation in relations.time_deltas
        ]

    if relations.mutual_exclusions:
        schema["x-polylogue-mutually-exclusive"] = [
            {
                "parent": relation.parent_path,
                "fields": sorted(relation.field_names),
            }
            for relation in relations.mutual_exclusions
        ]

    if relations.string_lengths:
        schema["x-polylogue-string-lengths"] = [
            {
                "path": relation.path,
                "min": relation.min_length,
                "max": relation.max_length,
                "avg": round(relation.avg_length, 1),
                "stddev": round(relation.stddev, 1),
            }
            for relation in relations.string_lengths
        ]


__all__ = ["annotate_semantic_and_relational"]
