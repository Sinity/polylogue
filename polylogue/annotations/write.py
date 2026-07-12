"""Single-row annotation write path: schema-validated labels as assertions.

This is the atomic operation a future batch/JSONL import surface loops over.
It accepts durable ``annotation-batch:`` provenance but deliberately does not
attempt the CLI/MCP surface here -- it validates one label against a declared
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
import json
import math
import sqlite3
import unicodedata
from collections.abc import Mapping, Sequence

from polylogue.annotations.schema import (
    ANNOTATION_SCHEMA_REGISTRY,
    AnnotationSchema,
    AnnotationSchemaRegistry,
    validate_annotation_row,
)
from polylogue.core.enums import AssertionKind
from polylogue.core.refs import ObjectRef, normalize_object_ref_text, normalize_public_ref_text
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ArchiveAssertionEnvelope,
    read_assertion_envelope,
    upsert_assertion,
)

_SCHEMA_PROVENANCE_KEY = "_schema"
_BATCH_PROVENANCE_KEY = "_batch"


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


def _nfc_json_value(value: object) -> object:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_nfc_json_value(item) for item in value]
    if isinstance(value, dict):
        return {unicodedata.normalize("NFC", str(key)): _nfc_json_value(item) for key, item in value.items()}
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _nfc_json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _normalized_annotation_confidence(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError("confidence must be a finite float or null")
    return 0.0 if value == 0.0 else value


def _immutable_annotation_inputs(
    *,
    scope_ref: str | None,
    target_ref: str,
    key: str | None,
    value: object,
    body_text: str | None,
    author_ref: str,
    author_kind: str,
    evidence_refs: Sequence[str],
    confidence: float | None,
) -> dict[str, object]:
    return {
        "author_kind": author_kind,
        "author_ref": author_ref,
        "body_text": body_text,
        "confidence": confidence,
        "evidence_refs": list(evidence_refs),
        "key": key,
        "kind": AssertionKind.ANNOTATION.value,
        "scope_ref": scope_ref,
        "target_ref": target_ref,
        "value": value,
    }


def assertion_id_for_schema_annotation(
    *,
    schema_qualified_id: str,
    target_ref: str,
    author_ref: str,
    row_key: str,
    batch_ref: str | None = None,
) -> str:
    """Return a deterministic assertion id for one schema-validated annotation row.

    Namespaced separately from :func:`polylogue.storage.sqlite.archive_tiers.
    user_write.assertion_id_for_annotation` (the freeform user-note helper) so
    the two annotation concepts can never collide on identity even though
    both currently write ``kind=AssertionKind.ANNOTATION`` rows.
    """

    digest = hashlib.sha256()
    identity_parts = [schema_qualified_id, target_ref, author_ref, row_key]
    if batch_ref is not None:
        identity_parts.append(normalize_object_ref_text(batch_ref))
    for part in identity_parts:
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
    batch_ref: str | None = None,
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
    this caller. When ``batch_ref`` is supplied, it must resolve to durable
    provenance for the same schema, target, actor, and declared assertion id.
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
    normalized_author_ref = normalize_object_ref_text(author_ref)
    normalized_evidence_refs = tuple(normalize_public_ref_text(ref) for ref in evidence_refs)
    try:
        normalized_confidence = _normalized_annotation_confidence(confidence)
    except ValueError as exc:
        raise AnnotationValidationError(
            schema_id=registered_schema.qualified_id,
            target_ref=normalized_target_ref,
            errors=(str(exc),),
        ) from exc
    normalized_batch_ref: str | None = None
    durable_batch = None
    if batch_ref is not None:
        from polylogue.storage.sqlite.archive_tiers.user_annotations import (
            read_annotation_batch,
            read_durable_annotation_schema,
        )

        batch_errors: list[str] = []
        try:
            normalized_batch_ref = normalize_object_ref_text(batch_ref)
            parsed_batch_ref = ObjectRef.parse(normalized_batch_ref)
        except ValueError:
            parsed_batch_ref = None
            batch_errors.append("batch_ref must be a valid ObjectRef")
        if parsed_batch_ref is not None and parsed_batch_ref.kind != "annotation-batch":
            batch_errors.append("batch_ref must use the 'annotation-batch' ObjectRef kind")
        durable_batch = (
            read_annotation_batch(conn, parsed_batch_ref.object_id)
            if parsed_batch_ref is not None and not batch_errors
            else None
        )
        if not batch_errors and durable_batch is None:
            batch_errors.append(f"batch_ref {normalized_batch_ref!r} does not resolve to durable batch provenance")
        if durable_batch is not None and durable_batch.qualified_schema_id != registered_schema.qualified_id:
            batch_errors.append(
                f"batch_ref {normalized_batch_ref!r} uses schema {durable_batch.qualified_schema_id!r}, "
                f"not {registered_schema.qualified_id!r}"
            )
        if durable_batch is not None and durable_batch.target_ref != normalized_target_ref:
            batch_errors.append(
                f"batch_ref {normalized_batch_ref!r} targets {durable_batch.target_ref!r}, "
                f"not {normalized_target_ref!r}"
            )
        if durable_batch is not None and durable_batch.actor_ref != normalized_author_ref:
            batch_errors.append(
                f"batch_ref {normalized_batch_ref!r} records actor {durable_batch.actor_ref!r}, "
                f"not {normalized_author_ref!r}"
            )
        durable_schema = (
            read_durable_annotation_schema(conn, durable_batch.schema_id, durable_batch.schema_version)
            if durable_batch is not None
            else None
        )
        if durable_batch is not None and durable_schema is None:
            batch_errors.append(f"batch_ref {normalized_batch_ref!r} does not resolve its durable schema definition")
        if (
            durable_schema is not None
            and durable_schema.definition_json != registered_schema.canonical_definition_json()
        ):
            batch_errors.append(
                f"batch_ref {normalized_batch_ref!r} resolves a durable schema definition that differs from the writer"
            )
        if batch_errors:
            raise AnnotationValidationError(
                schema_id=registered_schema.qualified_id,
                target_ref=normalized_target_ref,
                errors=batch_errors,
            )
    assertion_id = assertion_id_for_schema_annotation(
        schema_qualified_id=registered_schema.qualified_id,
        target_ref=normalized_target_ref,
        author_ref=normalized_author_ref,
        row_key=row_key,
        batch_ref=normalized_batch_ref,
    )
    if durable_batch is not None and f"assertion:{assertion_id}" not in durable_batch.assertion_refs:
        raise AnnotationValidationError(
            schema_id=registered_schema.qualified_id,
            target_ref=normalized_target_ref,
            errors=(f"batch_ref {normalized_batch_ref!r} does not declare assertion:{assertion_id}",),
        )
    stamped_value: dict[str, object] = {
        _SCHEMA_PROVENANCE_KEY: registered_schema.qualified_id,
        **dict(value),
    }
    if normalized_batch_ref is not None:
        stamped_value[_BATCH_PROVENANCE_KEY] = normalized_batch_ref

    candidate_inputs = _immutable_annotation_inputs(
        scope_ref=normalized_batch_ref,
        target_ref=normalized_target_ref,
        key=row_key,
        value=stamped_value,
        body_text=body_text,
        author_ref=normalized_author_ref,
        author_kind=author_kind,
        evidence_refs=normalized_evidence_refs,
        confidence=normalized_confidence,
    )
    try:
        _canonical_json_bytes(candidate_inputs)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise AnnotationValidationError(
            schema_id=registered_schema.qualified_id,
            target_ref=normalized_target_ref,
            errors=("annotation immutable inputs must be finite canonical JSON",),
        ) from exc

    existing = read_assertion_envelope(conn, assertion_id)
    if existing is not None and normalized_batch_ref is not None:
        existing_inputs = _immutable_annotation_inputs(
            scope_ref=existing.scope_ref,
            target_ref=existing.target_ref,
            key=existing.key,
            value=existing.value,
            body_text=existing.body_text,
            author_ref=existing.author_ref or "",
            author_kind=existing.author_kind or "",
            evidence_refs=existing.evidence_refs,
            confidence=existing.confidence,
        )
        try:
            drifted_fields = [
                field
                for field in candidate_inputs
                if _canonical_json_bytes(candidate_inputs[field]) != _canonical_json_bytes(existing_inputs[field])
            ]
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise AnnotationValidationError(
                schema_id=registered_schema.qualified_id,
                target_ref=normalized_target_ref,
                errors=(f"assertion_id {assertion_id!r} has non-canonical stored immutable inputs",),
            ) from exc
        if drifted_fields:
            raise AnnotationValidationError(
                schema_id=registered_schema.qualified_id,
                target_ref=normalized_target_ref,
                errors=(f"assertion_id {assertion_id!r} immutable input drift: {sorted(drifted_fields)}",),
            )
        return existing

    return upsert_assertion(
        conn,
        assertion_id=assertion_id,
        target_ref=normalized_target_ref,
        kind=AssertionKind.ANNOTATION,
        scope_ref=normalized_batch_ref,
        key=row_key,
        value=stamped_value,
        body_text=body_text,
        author_ref=normalized_author_ref,
        author_kind=author_kind,
        evidence_refs=normalized_evidence_refs,
        confidence=normalized_confidence,
        now_ms=now_ms,
    )


__all__ = [
    "AnnotationValidationError",
    "assertion_id_for_schema_annotation",
    "upsert_annotation_assertion",
]
