"""Bounded, provenance-preserving annotation batch import operation."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from polylogue.annotations.batch import AnnotationBatch
from polylogue.annotations.schema import (
    ANNOTATION_SCHEMA_REGISTRY,
    AnnotationSchema,
    AnnotationSchemaRegistry,
    validate_annotation_row,
)
from polylogue.annotations.write import assertion_id_for_schema_annotation, upsert_annotation_assertion
from polylogue.core.json import JSONDocument, require_json_document
from polylogue.core.refs import EvidenceRef, parse_public_ref
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_annotations import (
    persist_annotation_batch,
    persist_annotation_schema,
    read_annotation_batch,
)

if TYPE_CHECKING:
    from polylogue.api import Polylogue

MAX_ANNOTATION_IMPORT_BYTES = 1_048_576
MAX_ANNOTATION_IMPORT_ROWS = 10_000
MAX_ANNOTATION_IMPORT_LINE_BYTES = 65_536
MAX_ANNOTATION_IMPORT_REF_BYTES = 4_096
MAX_ANNOTATION_IMPORT_METADATA_BYTES = 65_536


class AnnotationBatchImportError(ValueError):
    """Raised when a batch envelope cannot be admitted safely."""


class AnnotationImportRow(BaseModel):
    """One JSONL label row under the batch-wide schema and target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_key: str = Field(min_length=1, max_length=256)
    value: dict[str, object]
    evidence_refs: tuple[str, ...]
    body_text: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class AnnotationBatchImportRequest(BaseModel):
    """Complete product-layer request for one bounded JSONL batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    jsonl: str
    batch_id: str = Field(min_length=1, max_length=256)
    schema_id: str = Field(min_length=1, max_length=256)
    schema_version: int = Field(ge=1)
    target_ref: str = Field(min_length=1, max_length=4_096)
    source_result_ref: str = Field(min_length=1, max_length=4_096)
    actor_ref: str = Field(min_length=1, max_length=4_096)
    model_ref: str = Field(min_length=1, max_length=4_096)
    prompt_ref: str = Field(min_length=1, max_length=4_096)
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at_ms: int | None = Field(default=None, ge=0)


class AnnotationImportRowOutcome(BaseModel):
    """Bounded per-line outcome returned by every surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    line: int = Field(ge=1)
    row_key: str | None = None
    status: Literal["imported", "invalid"]
    assertion_ref: str | None = None
    errors: tuple[str, ...] = ()


class AnnotationBatchImportResult(BaseModel):
    """Shared CLI/MCP/Python result for a completed atomic import."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok", "partial"]
    batch_ref: str
    qualified_schema_id: str
    target_ref: str
    total_count: int = Field(ge=0)
    valid_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)
    abstained_count: int = Field(ge=0)
    rows: tuple[AnnotationImportRowOutcome, ...]


RefResolver = Callable[[str], Awaitable[bool]]


def _ref_preview(ref: str) -> str:
    encoded = ref.encode("utf-8")
    if len(encoded) <= 160:
        return repr(ref)
    return repr(encoded[:160].decode("utf-8", errors="replace") + "…")


def _validate_request_bounds(request: AnnotationBatchImportRequest) -> None:
    for name in ("target_ref", "source_result_ref", "actor_ref", "model_ref", "prompt_ref"):
        value = getattr(request, name)
        if len(value.encode("utf-8")) > MAX_ANNOTATION_IMPORT_REF_BYTES:
            raise AnnotationBatchImportError(f"{name} exceeds {MAX_ANNOTATION_IMPORT_REF_BYTES} byte limit")
    metadata_bytes = json.dumps(request.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(metadata_bytes) > MAX_ANNOTATION_IMPORT_METADATA_BYTES:
        raise AnnotationBatchImportError(
            f"annotation metadata exceeds {MAX_ANNOTATION_IMPORT_METADATA_BYTES} byte limit"
        )


def _assertion_confidence(schema: AnnotationSchema, row: AnnotationImportRow, errors: list[str]) -> float | None:
    schema_fields = {field.name for field in schema.fields}
    value_confidence = row.value.get("confidence") if "confidence" in schema_fields else None
    if (
        value_confidence is not None
        and not isinstance(value_confidence, bool)
        and isinstance(value_confidence, (int, float))
    ):
        derived = float(value_confidence)
        if row.confidence is not None and row.confidence != derived:
            errors.append("top-level confidence must equal value.confidence")
        return derived
    return row.confidence


def _failure(line: int, *, row_key: str | None, errors: list[str]) -> AnnotationImportRowOutcome:
    return AnnotationImportRowOutcome(line=line, row_key=row_key, status="invalid", errors=tuple(errors))


def _failure_document(outcome: AnnotationImportRowOutcome) -> JSONDocument:
    return require_json_document(
        {"errors": list(outcome.errors), "line": outcome.line, "row_key": outcome.row_key},
        context="annotation import validation failure",
    )


def _parse_rows(jsonl: str) -> tuple[list[tuple[int, AnnotationImportRow]], list[AnnotationImportRowOutcome]]:
    try:
        payload_bytes = jsonl.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AnnotationBatchImportError("annotation JSONL must be valid UTF-8") from exc
    if len(payload_bytes) > MAX_ANNOTATION_IMPORT_BYTES:
        raise AnnotationBatchImportError(f"annotation JSONL exceeds {MAX_ANNOTATION_IMPORT_BYTES} byte limit")

    parsed: list[tuple[int, AnnotationImportRow]] = []
    failures: list[AnnotationImportRowOutcome] = []
    nonempty_count = 0
    for line_number, raw_line in enumerate(jsonl.splitlines(), start=1):
        if not raw_line.strip():
            continue
        nonempty_count += 1
        if nonempty_count > MAX_ANNOTATION_IMPORT_ROWS:
            raise AnnotationBatchImportError(f"annotation JSONL exceeds {MAX_ANNOTATION_IMPORT_ROWS} row limit")
        if len(raw_line.encode("utf-8")) > MAX_ANNOTATION_IMPORT_LINE_BYTES:
            failures.append(_failure(line_number, row_key=None, errors=["line exceeds byte limit"]))
            continue
        try:
            document = json.loads(raw_line)
            row = AnnotationImportRow.model_validate(document)
            require_json_document(row.value, context="annotation row value")
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            failures.append(_failure(line_number, row_key=None, errors=[str(exc)]))
            continue
        parsed.append((line_number, row))
    if nonempty_count == 0:
        raise AnnotationBatchImportError("annotation JSONL contains no rows")
    return parsed, failures


async def import_annotation_batch(
    poly: Polylogue,
    request: AnnotationBatchImportRequest,
    *,
    resolve_ref: RefResolver | None = None,
    registry: AnnotationSchemaRegistry = ANNOTATION_SCHEMA_REGISTRY,
) -> AnnotationBatchImportResult:
    """Validate live refs, persist provenance, and write candidates atomically."""

    _validate_request_bounds(request)
    try:
        schema = registry.get(request.schema_id, request.schema_version)
    except KeyError as exc:
        raise AnnotationBatchImportError(str(exc)) from exc

    async def default_resolver(ref: str) -> bool:
        return (await poly.resolve_ref(ref)).resolved

    resolver = resolve_ref or default_resolver
    if not await resolver(request.target_ref):
        raise AnnotationBatchImportError(
            f"target_ref {_ref_preview(request.target_ref)} does not resolve in the live archive"
        )

    parsed_rows, outcomes = _parse_rows(request.jsonl)
    valid_rows: list[tuple[int, AnnotationImportRow, str, float | None]] = []
    admitted_row_keys: set[str] = set()
    for line_number, row in parsed_rows:
        errors = validate_annotation_row(
            schema,
            target_ref=request.target_ref,
            value=row.value,
            evidence_refs=row.evidence_refs,
        )
        for evidence_ref_text in row.evidence_refs:
            if resolve_ref is not None:
                resolved = await resolver(evidence_ref_text)
            else:
                try:
                    parsed_ref = parse_public_ref(evidence_ref_text)
                    resolution = await poly.resolve_ref(evidence_ref_text)
                    resolved = resolution.resolved
                    if resolved and isinstance(parsed_ref, EvidenceRef):
                        resolved = f"session:{parsed_ref.session_id}" in resolution.object_refs
                except ValueError:
                    resolved = False
            if not resolved:
                errors.append(f"evidence_ref {_ref_preview(evidence_ref_text)} does not resolve in the live archive")
        confidence = _assertion_confidence(schema, row, errors)
        if row.row_key in admitted_row_keys:
            errors.append(f"duplicate row_key {row.row_key!r} in annotation batch")
        if errors:
            outcomes.append(_failure(line_number, row_key=row.row_key, errors=errors))
            continue
        admitted_row_keys.add(row.row_key)
        assertion_id = assertion_id_for_schema_annotation(
            schema_qualified_id=schema.qualified_id,
            target_ref=request.target_ref,
            author_ref=request.actor_ref,
            row_key=row.row_key,
            batch_ref=f"annotation-batch:{request.batch_id}",
        )
        valid_rows.append((line_number, row, assertion_id, confidence))

    outcomes.sort(key=lambda item: item.line)
    failure_documents = tuple(_failure_document(item) for item in outcomes if item.status == "invalid")
    abstained_count = sum(
        row.value.get(schema.abstain_field) is True for _, row, _, _ in valid_rows if schema.abstain_field is not None
    )
    user_db_path = Path(poly.archive_root) / "user.db"
    initialize_archive_database(user_db_path, ArchiveTier.USER)
    conn = sqlite3.connect(user_db_path)
    conn.row_factory = sqlite3.Row
    imported_outcomes: list[AnnotationImportRowOutcome] = []
    batch: AnnotationBatch
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing_batch = read_annotation_batch(conn, request.batch_id)
        created_at_ms = request.created_at_ms
        if created_at_ms is None:
            created_at_ms = existing_batch.created_at_ms if existing_batch is not None else int(time.time() * 1000)
        batch = AnnotationBatch(
            batch_id=request.batch_id,
            schema_id=schema.schema_id,
            schema_version=schema.version,
            target_ref=request.target_ref,
            source_result_ref=request.source_result_ref,
            actor_ref=request.actor_ref,
            model_ref=request.model_ref,
            prompt_ref=request.prompt_ref,
            total_count=len(valid_rows) + len(failure_documents),
            valid_count=len(valid_rows),
            invalid_count=len(failure_documents),
            abstained_count=abstained_count,
            assertion_refs=tuple(f"assertion:{assertion_id}" for _, _, assertion_id, _ in valid_rows),
            validation_failures=failure_documents,
            metadata=require_json_document(request.metadata, context="annotation import metadata"),
            created_at_ms=created_at_ms,
        )
        persist_annotation_schema(conn, schema, registered_at_ms=batch.created_at_ms)
        persist_annotation_batch(conn, batch)
        for line_number, row, assertion_id, confidence in valid_rows:
            envelope = upsert_annotation_assertion(
                conn,
                schema=schema,
                registry=registry,
                target_ref=request.target_ref,
                value=row.value,
                row_key=row.row_key,
                evidence_refs=row.evidence_refs,
                author_ref=request.actor_ref,
                author_kind="agent",
                confidence=confidence,
                body_text=row.body_text,
                batch_ref=batch.batch_ref,
                now_ms=batch.created_at_ms,
            )
            if envelope.assertion_id != assertion_id:
                raise RuntimeError("annotation assertion identity drifted after batch admission")
            imported_outcomes.append(
                AnnotationImportRowOutcome(
                    line=line_number,
                    row_key=row.row_key,
                    status="imported",
                    assertion_ref=f"assertion:{assertion_id}",
                )
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    all_outcomes = tuple(sorted((*outcomes, *imported_outcomes), key=lambda item: item.line))
    return AnnotationBatchImportResult(
        status="partial" if batch.invalid_count else "ok",
        batch_ref=batch.batch_ref,
        qualified_schema_id=schema.qualified_id,
        target_ref=batch.target_ref,
        total_count=batch.total_count,
        valid_count=batch.valid_count,
        invalid_count=batch.invalid_count,
        abstained_count=batch.abstained_count,
        rows=all_outcomes,
    )


__all__ = [
    "AnnotationBatchImportError",
    "AnnotationBatchImportRequest",
    "AnnotationBatchImportResult",
    "AnnotationImportRow",
    "AnnotationImportRowOutcome",
    "import_annotation_batch",
]
