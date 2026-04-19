"""Public root for generation annotation, dynamic-key, and redaction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeAlias

from polylogue.schemas import generation_dynamic_keys as _dynamic_keys
from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.generation_field_annotations import annotate_schema, remove_nested_required
from polylogue.schemas.generation_redaction import _build_redaction_report
from polylogue.schemas.generation_semantic_relations import annotate_semantic_and_relational

SchemaPayload: TypeAlias = dict[str, object]
SchemaCollection: TypeAlias = list[SchemaPayload]
FieldStatsMapping: TypeAlias = Mapping[str, FieldStats]


class PrivacyConfigLike(Protocol):
    level: str


GENSON_AVAILABLE = _dynamic_keys.GENSON_AVAILABLE
SchemaBuilder = _dynamic_keys.SchemaBuilder
collapse_dynamic_keys = _dynamic_keys.collapse_dynamic_keys


def _merge_schemas(schemas: SchemaCollection) -> SchemaPayload:
    return _dynamic_keys.merge_schemas(schemas)


def _annotate_schema(
    schema: SchemaPayload,
    stats: FieldStatsMapping,
    path: str = "$",
    *,
    min_conversation_count: int = 1,
    privacy_config: PrivacyConfigLike | None = None,
) -> SchemaPayload:
    return annotate_schema(
        schema,
        dict(stats),
        path,
        min_conversation_count=min_conversation_count,
        privacy_config=privacy_config,
    )


def _annotate_semantic_and_relational(
    schema: SchemaPayload,
    field_stats: FieldStatsMapping,
    *,
    artifact_kind: str | None = None,
) -> SchemaPayload:
    return annotate_semantic_and_relational(schema, dict(field_stats), artifact_kind=artifact_kind)


def _remove_nested_required(schema: SchemaPayload, depth: int = 0) -> SchemaPayload:
    return remove_nested_required(schema, depth=depth)


__all__ = [
    "GENSON_AVAILABLE",
    "SchemaBuilder",
    "_annotate_schema",
    "_annotate_semantic_and_relational",
    "_build_redaction_report",
    "_merge_schemas",
    "_remove_nested_required",
    "collapse_dynamic_keys",
]
