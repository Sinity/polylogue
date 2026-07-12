"""Single-row annotation write path: schema-validated labels as assertions.

This is the atomic operation a future batch/JSONL import surface would loop
over (polylogue-rxdo.7 residual scope: ``annotation_batches`` provenance
container, CLI/MCP import surface). It deliberately does not attempt that
surface here -- it validates one label against a declared
:class:`~polylogue.annotations.schema.AnnotationSchema` and writes it through
the existing single assertion-write chokepoint
(:func:`polylogue.storage.sqlite.archive_tiers.user_write.upsert_assertion`),
so it inherits that function's agent-authored candidate-coercion invariant
(polylogue-37t.15) for free: any ``author_kind`` other than ``"user"`` lands
``status=candidate`` with a non-injected context policy, regardless of what
this helper requests.

Scope note: this validates the row's *shape* (schema conformance) and that
evidence refs are well-formed object/evidence refs (enforced transitively by
``upsert_assertion``'s own ref normalization). It does not check that
``target_ref``/``evidence_refs`` resolve to rows that actually exist in the
live archive -- that referential-integrity check belongs to the batch import
surface, which has the archive handle this single-connection helper does not.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping, Sequence

from polylogue.annotations.schema import (
    ANNOTATION_SCHEMA_REGISTRY,
    AnnotationSchema,
    AnnotationSchemaRegistry,
    validate_annotation_row,
)
from polylogue.core.enums import AssertionKind
from polylogue.core.refs import normalize_object_ref_text
from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionEnvelope, upsert_assertion

_SCHEMA_PROVENANCE_KEY = "_schema"


class AnnotationValidationError(ValueError):
    """Raised when a candidate annotation row fails schema/shape validation."""

    def __init__(self, *, schema_id: str, target_ref: str, errors: Sequence[str]) -> None:
        self.schema_id = schema_id
        self.target_ref = target_ref
        self.errors = tuple(errors)
        message = f"annotation row for schema {schema_id!r} target {target_ref!r} failed validation: " + "; ".join(
            self.errors
        )
        super().__init__(message)


def assertion_id_for_schema_annotation(
    *, schema_qualified_id: str, target_ref: str, author_ref: str, row_key: str
) -> str:
    """Return a deterministic assertion id for one schema-validated annotation row.

    Namespaced separately from :func:`polylogue.storage.sqlite.archive_tiers.
    user_write.assertion_id_for_annotation` (the freeform user-note helper) so
    the two annotation concepts can never collide on identity even though
    both currently write ``kind=AssertionKind.ANNOTATION`` rows.
    """

    digest = hashlib.sha256()
    for part in (schema_qualified_id, target_ref, author_ref, row_key):
        digest.update(part.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return f"assertion-annotation-schema:{digest.hexdigest()}"


def upsert_annotation_assertion(
    conn: sqlite3.Connection,
    *,
    schema: AnnotationSchema,
    registry: AnnotationSchemaRegistry = ANNOTATION_SCHEMA_REGISTRY,
    target_ref: str,
    value: Mapping[str, object],
    row_key: str,
    evidence_refs: Sequence[str] = (),
    author_ref: str,
    author_kind: str = "agent",
    confidence: float | None = None,
    body_text: str | None = None,
    now_ms: int | None = None,
) -> ArchiveAssertionEnvelope:
    """Validate one label against *schema* and upsert it as a candidate assertion.

    The schema must match an active entry in *registry*. Raises
    :class:`~polylogue.annotations.schema.AnnotationSchemaError` for missing,
    drifted, draft, or deprecated registrations, and
    :class:`AnnotationValidationError` (writing nothing) when the row fails
    target-grain, field-shape, or evidence-policy validation.
    On success, delegates to ``upsert_assertion`` with
    ``kind=AssertionKind.ANNOTATION``; the resulting row's status/context
    policy is decided by that function's own author-kind chokepoint, not by
    this caller.
    """

    registered_schema = registry.require_active(schema)
    errors = validate_annotation_row(
        registered_schema,
        target_ref=target_ref,
        value=value,
        evidence_refs=evidence_refs,
    )
    if errors:
        raise AnnotationValidationError(
            schema_id=registered_schema.qualified_id,
            target_ref=target_ref,
            errors=errors,
        )

    normalized_target_ref = normalize_object_ref_text(target_ref)
    assertion_id = assertion_id_for_schema_annotation(
        schema_qualified_id=registered_schema.qualified_id,
        target_ref=normalized_target_ref,
        author_ref=author_ref,
        row_key=row_key,
    )
    stamped_value: dict[str, object] = {
        _SCHEMA_PROVENANCE_KEY: registered_schema.qualified_id,
        **dict(value),
    }

    return upsert_assertion(
        conn,
        assertion_id=assertion_id,
        target_ref=normalized_target_ref,
        kind=AssertionKind.ANNOTATION,
        key=row_key,
        value=stamped_value,
        body_text=body_text,
        author_ref=author_ref,
        author_kind=author_kind,
        evidence_refs=evidence_refs,
        confidence=confidence,
        now_ms=now_ms,
    )


__all__ = [
    "AnnotationValidationError",
    "assertion_id_for_schema_annotation",
    "upsert_annotation_assertion",
]
