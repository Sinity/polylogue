"""Annotation substrate: declared schemas and schema-validated assertion writes.

External-agent labeling only becomes an analytical variable, not a queryable
blob, once the label's shape is declared (polylogue-rxdo.7). This package
holds:

- :mod:`polylogue.annotations.schema` -- ``AnnotationSchema``/``AnnotationField``
  declarations, a registry, and shape validation.
- :mod:`polylogue.annotations.write` -- a single-row write helper that
  validates a label against a schema and writes it as a candidate assertion
  through the existing single write chokepoint.
- :mod:`polylogue.annotations.batch` -- immutable source/actor/model/prompt,
  validation, and count provenance for independent labeling batches.

The JSONL/CLI/MCP import surface that loops over
:func:`polylogue.annotations.write.upsert_annotation_assertion` remains in
``polylogue-rxdo.7.2``; this package owns its durable schema/batch contracts.
"""

from __future__ import annotations

from polylogue.annotations.batch import AnnotationBatch, AnnotationBatchError
from polylogue.annotations.schema import (
    ANNOTATION_SCHEMA_DEFINITION_FORMAT,
    ANNOTATION_SCHEMA_REGISTRY,
    DELEGATION_DISCOURSE_SCHEMA,
    AnnotationEvidencePolicy,
    AnnotationField,
    AnnotationFieldType,
    AnnotationSchema,
    AnnotationSchemaError,
    AnnotationSchemaRegistry,
    AnnotationSchemaStatus,
    get_annotation_schema,
    list_annotation_schemas,
    register_annotation_schema,
    validate_annotation_row,
    validate_annotation_value,
)
from polylogue.annotations.write import (
    AnnotationValidationError,
    assertion_id_for_schema_annotation,
    upsert_annotation_assertion,
)

__all__ = [
    "ANNOTATION_SCHEMA_REGISTRY",
    "ANNOTATION_SCHEMA_DEFINITION_FORMAT",
    "DELEGATION_DISCOURSE_SCHEMA",
    "AnnotationBatch",
    "AnnotationBatchError",
    "AnnotationEvidencePolicy",
    "AnnotationField",
    "AnnotationFieldType",
    "AnnotationSchema",
    "AnnotationSchemaError",
    "AnnotationSchemaRegistry",
    "AnnotationSchemaStatus",
    "AnnotationValidationError",
    "assertion_id_for_schema_annotation",
    "get_annotation_schema",
    "list_annotation_schemas",
    "register_annotation_schema",
    "upsert_annotation_assertion",
    "validate_annotation_row",
    "validate_annotation_value",
]
