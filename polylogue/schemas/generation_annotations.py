"""Small public root for schema generation annotation helpers."""

from __future__ import annotations

from polylogue.schemas import generation_dynamic_keys as _dynamic_keys
from polylogue.schemas.generation_field_annotations import (
    annotate_schema,
    remove_nested_required,
)
from polylogue.schemas.generation_semantic_relations import annotate_semantic_and_relational

GENSON_AVAILABLE = _dynamic_keys.GENSON_AVAILABLE
SchemaBuilder = _dynamic_keys.SchemaBuilder
collapse_dynamic_keys = _dynamic_keys.collapse_dynamic_keys


def _merge_schemas(schemas):
    return _dynamic_keys.merge_schemas(schemas)


def _annotate_schema(schema, stats, path: str = "$", *, min_conversation_count: int = 1, privacy_config=None):
    return annotate_schema(
        schema,
        stats,
        path,
        min_conversation_count=min_conversation_count,
        privacy_config=privacy_config,
    )


def _annotate_semantic_and_relational(schema, field_stats):
    return annotate_semantic_and_relational(schema, field_stats)


def _remove_nested_required(schema, depth: int = 0):
    return remove_nested_required(schema, depth=depth)


__all__ = [
    "GENSON_AVAILABLE",
    "_annotate_schema",
    "_annotate_semantic_and_relational",
    "_merge_schemas",
    "_remove_nested_required",
    "collapse_dynamic_keys",
]
