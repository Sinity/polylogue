"""Small public root for generation annotation and redaction helpers."""

from __future__ import annotations

from polylogue.schemas.generation_annotations import (
    GENSON_AVAILABLE,
    SchemaBuilder,
    _annotate_schema,
    _annotate_semantic_and_relational,
    _merge_schemas,
    _remove_nested_required,
    collapse_dynamic_keys,
)
from polylogue.schemas.generation_redaction import _build_redaction_report

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
