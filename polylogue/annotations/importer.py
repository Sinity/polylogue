"""Bounded, provenance-preserving annotation batch import operation."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from polylogue.annotations.batch import AnnotationBatch
from polylogue.annotations.import_spill import AnnotationImportSpill
from polylogue.annotations.schema import (
    ANNOTATION_SCHEMA_REGISTRY,
    AnnotationSchema,
    AnnotationSchemaRegistry,
    validate_annotation_row,
)
from polylogue.annotations.write import assertion_id_for_schema_annotation, upsert_annotation_assertion
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.json import JSONDocument, require_json_document
from polylogue.core.refs import EvidenceRef, parse_public_ref
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.mutation_transaction import (
    ConfirmationStrength,
    MutationPlan,
    MutationPrincipal,
    MutationReceipt,
    OperationExecutor,
    RecoveryResolution,
    ReplayHandles,
    build_plan,
    register_recovery_route,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_annotations import (
    annotation_batch_provenance_digest,
    persist_annotation_schema,
    persist_spilled_annotation_batch,
    read_durable_annotation_schema,
)
from polylogue.storage.sqlite.connection_profile import (
    connection_context,
    open_readonly_connection,
    readonly_connection_context,
    scratch_connection_context,
)

if TYPE_CHECKING:
    from polylogue.api import Polylogue


class AnnotationBatchImportError(ValueError):
    """Raised when a batch envelope cannot be admitted safely."""


class AnnotationImportRow(BaseModel):
    """One JSONL label row under the batch-wide schema and target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_key: str = Field(min_length=1)
    value: dict[str, object]
    evidence_refs: tuple[str, ...]
    body_text: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class AnnotationBatchImportRequest(BaseModel):
    """Complete product-layer request for one bounded JSONL batch."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    batch_id: str = Field(min_length=1)
    schema_id: str = Field(min_length=1)
    schema_version: int = Field(ge=1)
    target_ref: str = Field(min_length=1)
    source_result_ref: str = Field(min_length=1)
    actor_ref: str = Field(min_length=1)
    model_ref: str = Field(min_length=1)
    prompt_ref: str = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at_ms: int | None = Field(default=None, ge=0)


class AnnotationImportRowOutcome(BaseModel):
    """Validation failure retained in the original batch authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    line: int = Field(ge=1)
    row_key: str | None = None
    status: Literal["invalid"]
    errors: tuple[str, ...] = ()


class AnnotationBatchImportResult(BaseModel):
    """Committed batch summary; exact evidence is read through ``batch_ref``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok", "partial"]
    batch_ref: str
    qualified_schema_id: str
    target_ref: str
    total_count: int = Field(ge=0)
    valid_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)
    abstained_count: int = Field(ge=0)


RefResolver = Callable[[str], Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class AnnotationBatchImportArgs:
    """Sealed request-owned evidence shared by authorization and apply."""

    user_db_path: Path
    schema: AnnotationSchema
    effective_registry: AnnotationSchemaRegistry
    request: AnnotationBatchImportRequest
    batch: AnnotationImportSpill


def _persist_annotation_batch(args: AnnotationBatchImportArgs) -> None:
    """Commit schema, complete provenance and every assertion together."""
    batch = args.batch
    with connection_context(args.user_db_path, archive_root=args.user_db_path.parent) as conn:
        conn.row_factory = sqlite3.Row
        # Complete JSON cells must satisfy durable CHECKs at INSERT. Their
        # unconstrained staging relation belongs on disk, not in temp memory.
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("BEGIN IMMEDIATE")
        persist_annotation_schema(conn, args.schema, registered_at_ms=int(cast(int, batch.header["created_at_ms"])))
        persist_spilled_annotation_batch(conn, batch)
        with closing(batch.rows()) as rows:
            for row_json, assertion_ref, confidence in rows:
                check_compute_cancelled()
                row = AnnotationImportRow.model_validate_json(row_json)
                envelope = upsert_annotation_assertion(
                    conn,
                    schema=args.schema,
                    registry=args.effective_registry,
                    target_ref=args.request.target_ref,
                    value=row.value,
                    row_key=row.row_key,
                    evidence_refs=row.evidence_refs,
                    author_ref=args.request.actor_ref,
                    author_kind="agent",
                    confidence=confidence,
                    body_text=row.body_text,
                    batch_ref=batch.batch_ref,
                    now_ms=int(cast(int, batch.header["created_at_ms"])),
                )
                if f"assertion:{envelope.assertion_id}" != assertion_ref:
                    raise RuntimeError("annotation assertion identity drifted after batch admission")
        conn.commit()


@dataclass(frozen=True, slots=True)
class AnnotationBatchImportActuator:
    """Executor actuator for the atomic, provenance-bearing annotation import."""

    operation: str = "mutate-import-annotation-batch"
    destructive_class: Literal["reversible"] = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: AnnotationBatchImportArgs) -> MutationPlan:
        batch = args.batch
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(batch.batch_ref,),
            affected_tiers=("user",),
            reversible=True,
            context={
                "batch_id": batch.header["batch_id"],
                "created_at_ms": batch.header["created_at_ms"],
                "invalid_count": batch.header["invalid_count"],
                "provenance_sha256": batch.provenance_digest(),
                "rows_sha256": batch.rows_digest(),
                "schema": args.schema.qualified_id,
                "valid_count": batch.header["valid_count"],
            },
        )

    def apply(self, plan: MutationPlan, args: AnnotationBatchImportArgs) -> MutationReceipt:
        batch = args.batch
        expected_digest = str(plan.context["provenance_sha256"])
        if batch.provenance_digest() != expected_digest:
            raise RuntimeError("annotation batch provenance changed after authorization")
        if batch.rows_digest() != plan.context["rows_sha256"]:
            raise RuntimeError("annotation batch row values changed after authorization")
        _persist_annotation_batch(args)
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied",
            target_refs=plan.target_refs,
            affected_count=int(cast(int, batch.header["valid_count"])) + 1,
            detail=None,
            receipt_ref=batch.batch_ref,
            applied_at=plan.prepared_at,
            domain_receipt={},
        )

    def recover(self, handles: ReplayHandles, plan: MutationPlan) -> RecoveryResolution:
        """Resolve an interrupted import from its one atomic transaction.

        The rows are not in the plan -- only their provenance digest -- so the
        import cannot be re-applied. It does not need to be: the schema, batch
        row and every assertion commit together, so the batch row is present
        exactly when the whole import is.
        """
        with readonly_connection_context(handles.archive_root / "user.db") as conn:
            stored_digest = annotation_batch_provenance_digest(conn, str(plan.context["batch_id"]))
        if stored_digest != plan.context["provenance_sha256"]:
            return RecoveryResolution("absent", "this batch import never committed")
        return RecoveryResolution(
            "complete",
            "the atomic batch import committed",
            MutationReceipt(
                operation=self.operation,
                plan_hash=plan.plan_hash,
                status="applied",
                target_refs=plan.target_refs,
                affected_count=int(cast(int, plan.context["valid_count"])) + 1,
                detail=None,
                receipt_ref=plan.target_refs[0],
                applied_at=plan.prepared_at,
            ),
        )


def _resolve_import_schema(
    user_db_path: Path,
    request: AnnotationBatchImportRequest,
    registry: AnnotationSchemaRegistry,
) -> tuple[AnnotationSchema, AnnotationSchemaRegistry]:
    """Resolve packaged, caller-supplied, or governed archive-local schemas.

    A promoted archive-specific ontology is durable in ``user.db`` rather than
    injected into the process-global registry (which would leak one archive's
    vocabulary into another). The importer builds a one-schema local registry
    when only the durable definition exists, while still rejecting any
    caller/global definition that disagrees with the archive authority.
    """

    try:
        registered = registry.get(request.schema_id, request.schema_version)
    except KeyError:
        registered = None

    conn = open_readonly_connection(user_db_path)
    conn.row_factory = sqlite3.Row
    try:
        durable = read_durable_annotation_schema(conn, request.schema_id, request.schema_version)
    finally:
        conn.close()

    if durable is not None:
        if registered is not None and (
            registered.canonical_definition_json() != durable.definition_json
            or registered.definition_fingerprint != durable.definition_sha256
        ):
            raise AnnotationBatchImportError(
                f"annotation schema {durable.schema.qualified_id!r} disagrees with its durable archive definition"
            )
        if registered is not None:
            return registered, registry
        local_registry = AnnotationSchemaRegistry()
        local_registry.register(durable.schema)
        return durable.schema, local_registry

    if registered is None:
        raise AnnotationBatchImportError(
            f"no registered annotation schema {request.schema_id!r}@v{request.schema_version}"
        )
    return registered, registry


def _ref_preview(ref: str) -> str:
    encoded = ref.encode("utf-8")
    if len(encoded) <= 160:
        return repr(ref)
    return repr(encoded[:160].decode("utf-8", errors="replace") + "…")


def _assertion_confidence(schema: AnnotationSchema, row: AnnotationImportRow, errors: list[str]) -> float | None:
    schema_fields = {field.name for field in schema.fields}
    value_confidence = row.value.get("confidence") if "confidence" in schema_fields else None
    if (
        value_confidence is not None
        and not isinstance(value_confidence, bool)
        and isinstance(value_confidence, (int, float))
    ):
        # The existing extraction maps a schema's confidence number to the
        # assertion probability. Validate that domain before narrowing an
        # arbitrary JSON integer to float; other payload numbers stay exact.
        if not 0 <= value_confidence <= 1:
            errors.append("value.confidence must be a finite probability between 0 and 1")
            return None
        try:
            derived = float(value_confidence)
        except (OverflowError, ValueError):
            errors.append("value.confidence cannot be represented as an assertion probability")
            return None
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


def _parse_rows(input: BinaryIO) -> Iterator[tuple[int, AnnotationImportRow | AnnotationImportRowOutcome]]:
    nonempty_count = 0
    # Binary iteration splits only physical LF. CRLF remains JSON whitespace;
    # U+0085/U+2028/U+2029 inside strings remain ordinary content.
    for line_number, raw_line in enumerate(input, start=1):
        check_compute_cancelled()
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AnnotationBatchImportError("annotation JSONL must be valid UTF-8") from exc
        if not line.strip(" \t\r\n"):
            continue
        nonempty_count += 1
        try:
            row = AnnotationImportRow.model_validate_json(line)
            require_json_document(row.value, context="annotation row value")
        except (ValidationError, TypeError, ValueError) as exc:
            yield line_number, _failure(line_number, row_key=None, errors=[str(exc)])
            continue
        yield line_number, row
    if nonempty_count == 0:
        raise AnnotationBatchImportError("annotation JSONL contains no rows")


async def import_annotation_batch(
    poly: Polylogue,
    request: AnnotationBatchImportRequest,
    *,
    input: BinaryIO,
    resolve_ref: RefResolver | None = None,
    registry: AnnotationSchemaRegistry = ANNOTATION_SCHEMA_REGISTRY,
    before_durable_execution: Callable[[], None] | None = None,
) -> AnnotationBatchImportResult:
    """Validate live refs, persist provenance, and write candidates atomically.

    Each exact reference is resolved once per import; results spill on disk
    rather than reopening the same read for every row. The daemon supplies
    its pinned operation reader as the resolver.

    ``before_durable_execution`` runs after validation and before the first
    audit or ``user.db`` write. A daemon caller fences its acceptance boundary
    there, so a request that was cancelled or timed out while validating
    refuses instead of committing after its caller was told it did not.
    """

    user_db_path = Path(poly.archive_root) / "user.db"
    initialize_archive_database(user_db_path, ArchiveTier.USER)
    schema, effective_registry = _resolve_import_schema(user_db_path, request, registry)

    async def default_resolver(ref: str) -> bool:
        return (await poly.resolve_ref(ref)).resolved

    resolver = resolve_ref or default_resolver
    if not await resolver(request.target_ref):
        raise AnnotationBatchImportError(
            f"target_ref {_ref_preview(request.target_ref)} does not resolve in the live archive"
        )

    created_at_ms = request.created_at_ms
    if created_at_ms is None:
        with readonly_connection_context(user_db_path) as conn:
            stored = conn.execute(
                "SELECT created_at_ms FROM annotation_batches WHERE batch_id=?", (request.batch_id,)
            ).fetchone()
        created_at_ms = int(stored[0]) if stored is not None else int(time.time() * 1000)
    header = AnnotationBatch(
        batch_id=request.batch_id,
        schema_id=schema.schema_id,
        schema_version=schema.version,
        target_ref=request.target_ref,
        source_result_ref=request.source_result_ref,
        actor_ref=request.actor_ref,
        model_ref=request.model_ref,
        prompt_ref=request.prompt_ref,
        total_count=0,
        valid_count=0,
        invalid_count=0,
        abstained_count=0,
        metadata=require_json_document(request.metadata, context="annotation import metadata"),
        created_at_ms=created_at_ms,
    )
    with scratch_connection_context(prefix="annotation-import-", filename="validated.sqlite") as scratch:
        batch = AnnotationImportSpill(scratch, header)
        for line_number, row in _parse_rows(input):
            if isinstance(row, AnnotationImportRowOutcome):
                batch.append_failure(line_number, _failure_document(row))
                continue
            errors = validate_annotation_row(
                schema,
                target_ref=request.target_ref,
                value=row.value,
                evidence_refs=row.evidence_refs,
            )
            for evidence_ref_text in row.evidence_refs:
                resolved = batch.ref_resolution(evidence_ref_text)
                if resolved is None:
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
                    batch.record_ref_resolution(evidence_ref_text, resolved)
                if not resolved:
                    errors.append(
                        f"evidence_ref {_ref_preview(evidence_ref_text)} does not resolve in the live archive"
                    )
            confidence = _assertion_confidence(schema, row, errors)
            if batch.has_row_key(row.row_key):
                errors.append(f"duplicate row_key {row.row_key!r} in annotation batch")
            if errors:
                batch.append_failure(
                    line_number, _failure_document(_failure(line_number, row_key=row.row_key, errors=errors))
                )
                continue
            assertion_id = assertion_id_for_schema_annotation(
                schema_qualified_id=schema.qualified_id,
                target_ref=request.target_ref,
                author_ref=request.actor_ref,
                row_key=row.row_key,
                batch_ref=f"annotation-batch:{request.batch_id}",
            )
            batch.append_row(
                line_number,
                row.row_key,
                row.model_dump_json(),
                f"assertion:{assertion_id}",
                confidence,
                schema.abstain_field is not None and row.value.get(schema.abstain_field) is True,
            )

        batch.seal()
        args = AnnotationBatchImportArgs(user_db_path, schema, effective_registry, request, batch)
        executor = OperationExecutor.for_archive_root(user_db_path.parent)
        actuator = AnnotationBatchImportActuator()
        binding = runtime_operation_binding(actuator)
        principal = MutationPrincipal(
            request.actor_ref, frozenset({"archive.annotation.import_batch"}), "internal", "write"
        )
        if before_durable_execution is not None:
            before_durable_execution()
        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=user_db_path.parent)
        authorization = executor.authorize_bound(binding, preview, principal)
        executor.execute_bound(binding, preview, authorization, args)
        return AnnotationBatchImportResult(
            status="partial" if batch.header["invalid_count"] else "ok",
            batch_ref=batch.batch_ref,
            qualified_schema_id=schema.qualified_id,
            target_ref=str(batch.header["target_ref"]),
            total_count=int(cast(int, batch.header["total_count"])),
            valid_count=int(cast(int, batch.header["valid_count"])),
            invalid_count=int(cast(int, batch.header["invalid_count"])),
            abstained_count=int(cast(int, batch.header["abstained_count"])),
        )


register_recovery_route(AnnotationBatchImportActuator())
