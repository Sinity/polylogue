"""Typed annotation enrichment against exact structural targets."""

from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from polylogue.annotations.schema import ANNOTATION_SCHEMA_REGISTRY, validate_annotation_row
from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.core.json import JSONDocument, require_json_document
from polylogue.core.refs import ObjectRef
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary, ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_annotations import read_durable_annotation_schema
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ArchiveAssertionEnvelope,
    count_assertion_claims,
    list_assertion_claims,
    read_assertion_envelope,
    read_latest_candidate_judgment,
)

AnnotationGroupDimension = Literal["repo", "model", "time", "origin"]
AnnotationJoinDiagnosticCode = Literal["missing_target", "ambiguous_target", "schema_drift", "invalid_value"]
_MAX_JOIN_LIMIT = 1_000
_MAX_DIAGNOSTICS = 100


class StructuralJoinArchive(Protocol):
    """Minimal facade contract consumed by the product operation."""

    @property
    def archive_root(self) -> Path: ...

    async def resolve_ref(self, ref: str) -> Any: ...


class AnnotationStructuralJoinError(ValueError):
    """Raised when a join request cannot be evaluated honestly."""


class AnnotationStructuralJoinRequest(BaseModel):
    """Explicit schema/status selection for one bounded annotation join."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: str = Field(min_length=1, max_length=256)
    schema_version: int = Field(ge=1)
    statuses: tuple[AssertionStatus, ...] = Field(min_length=1)
    target_kind: str | None = Field(default=None, min_length=1, max_length=64)
    group_by: tuple[AnnotationGroupDimension, ...] = ()
    limit: int = Field(default=500, ge=1, le=_MAX_JOIN_LIMIT)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_lifecycle_selection(self) -> AnnotationStructuralJoinRequest:
        if len(set(self.statuses)) != len(self.statuses):
            raise ValueError("annotation join statuses must be unique")
        if AssertionStatus.ACCEPTED in self.statuses and AssertionStatus.ACTIVE in self.statuses:
            raise ValueError("accepted and active cannot be joined together because they represent one label lifecycle")
        return self


class AnnotationJoinDiagnostic(BaseModel):
    """One bounded reason a selected annotation did not join cleanly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: AnnotationJoinDiagnosticCode
    assertion_ref: str
    target_ref: str
    detail: str = Field(max_length=512)


class AnnotationStructuralJoinRow(BaseModel):
    """One label joined to one exact target without collapsing label identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assertion_ref: str
    batch_ref: str | None
    schema_id: str
    schema_version: int
    status: AssertionStatus
    labeler_ref: str | None
    adjudicator_ref: str | None
    source_assertion_ref: str
    judgment_ref: str | None
    judgment_decision: str | None
    judgment_reason: str | None
    supersedes: tuple[str, ...]
    target_ref: str
    value: dict[str, Any]
    evidence_refs: tuple[str, ...]
    structural: dict[str, Any]


class AnnotationStructuralGroup(BaseModel):
    """Deterministic aggregate over successfully joined label rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimensions: dict[str, Any]
    label_count: int = Field(ge=1)
    distinct_target_count: int = Field(ge=1)


class AnnotationStructuralJoinResult(BaseModel):
    """Rows, aggregates, and explicit non-join/fanout accounting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qualified_schema_id: str
    requested_statuses: tuple[AssertionStatus, ...]
    selected_annotation_count: int = Field(ge=0)
    matched_annotation_count: int = Field(ge=0)
    offset: int = Field(ge=0)
    next_offset: int | None = Field(default=None, ge=0)
    selection_truncated: bool
    joined_count: int = Field(ge=0)
    missing_target_count: int = Field(ge=0)
    ambiguous_target_count: int = Field(ge=0)
    schema_drift_count: int = Field(ge=0)
    invalid_value_count: int = Field(ge=0)
    multi_label_target_count: int = Field(ge=0)
    duplicate_label_count: int = Field(ge=0)
    diagnostics_truncated: bool
    diagnostics: tuple[AnnotationJoinDiagnostic, ...]
    rows: tuple[AnnotationStructuralJoinRow, ...]
    groups: tuple[AnnotationStructuralGroup, ...]


def _bounded_detail(detail: str) -> str:
    return detail if len(detail) <= 512 else detail[:511] + "…"


def _schema_stamp(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    stamp = value.get("_schema")
    return stamp if isinstance(stamp, str) else None


def _typed_value(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items() if key not in {"_schema", "_batch"}}


def _batch_ref(value: object, scope_ref: str | None) -> str | None:
    if isinstance(value, dict) and isinstance(value.get("_batch"), str):
        return str(value["_batch"])
    return scope_ref if scope_ref and scope_ref.startswith("annotation-batch:") else None


def _summary_for_ref(
    archive: ArchiveStore,
    session_ref: str | None,
    cache: dict[str, ArchiveSessionSummary | None],
) -> ArchiveSessionSummary | None:
    if session_ref is None:
        return None
    session_id = session_ref.removeprefix("session:")
    if session_id not in cache:
        summaries = archive.list_summaries(session_id=session_id, limit=1)
        cache[session_id] = summaries[0] if summaries else None
    return cache[session_id]


def _structural_document(
    *,
    target_ref: str,
    payload_kind: str | None,
    payload: dict[str, object] | None,
    summary: ArchiveSessionSummary | None,
) -> JSONDocument:
    target_kind = ObjectRef.parse(target_ref).kind
    attempt_raw = payload.get("attempt") if payload else None
    attempt = cast(dict[str, object], attempt_raw) if isinstance(attempt_raw, dict) else None
    model = attempt.get("dispatch_turn_model") if attempt else None
    mapping_state = attempt.get("mapping_state") if attempt else None
    document: dict[str, object] = {
        "target_kind": target_kind,
        "payload_kind": payload_kind,
        "session_id": summary.session_id if summary else None,
        "origin": summary.origin if summary else None,
        "repo": summary.git_repository_url if summary else None,
        "created_at": summary.created_at if summary else None,
        "updated_at": summary.updated_at if summary else None,
        "model": model if isinstance(model, str) else None,
        "mapping_state": mapping_state if isinstance(mapping_state, str) else None,
    }
    return require_json_document(document, context="annotation structural join")


def _group_value(row: AnnotationStructuralJoinRow, dimension: AnnotationGroupDimension) -> object:
    value = row.structural.get(dimension)
    if dimension == "time" and isinstance(row.structural.get("created_at"), str):
        value = str(row.structural["created_at"])[:10]
    return value if value is not None else "unknown"


def _groups(
    rows: list[AnnotationStructuralJoinRow],
    dimensions: tuple[AnnotationGroupDimension, ...],
) -> tuple[AnnotationStructuralGroup, ...]:
    if not dimensions:
        return ()
    grouped: dict[tuple[object, ...], list[AnnotationStructuralJoinRow]] = {}
    for row in rows:
        key = tuple(_group_value(row, dimension) for dimension in dimensions)
        grouped.setdefault(key, []).append(row)
    return tuple(
        AnnotationStructuralGroup(
            dimensions=require_json_document(dict(zip(dimensions, key, strict=True)), context="annotation group"),
            label_count=len(items),
            distinct_target_count=len({item.target_ref for item in items}),
        )
        for key, items in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0]))
    )


async def join_typed_annotations(
    poly: StructuralJoinArchive,
    request: AnnotationStructuralJoinRequest,
) -> AnnotationStructuralJoinResult:
    """Join selected typed labels to exact targets, retaining one row per label."""

    user_db = Path(poly.archive_root) / "user.db"
    if not user_db.exists():
        raise AnnotationStructuralJoinError("annotation user tier is not initialized")
    with closing(sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)) as user_conn:
        user_conn.row_factory = sqlite3.Row
        durable = read_durable_annotation_schema(user_conn, request.schema_id, request.schema_version)
        if durable is None:
            raise AnnotationStructuralJoinError(
                f"annotation schema {request.schema_id!r}@v{request.schema_version} is not registered durably"
            )
        schema = durable.schema
        try:
            active_schema = ANNOTATION_SCHEMA_REGISTRY.get(request.schema_id, request.schema_version)
        except KeyError:
            active_schema = None
        registry_drift = active_schema is not None and active_schema.definition_fingerprint != durable.definition_sha256
        qualified_id = schema.qualified_id
        same_schema_prefix = f"{schema.schema_id}@v"
        assertions = list_assertion_claims(
            user_conn,
            kinds=(AssertionKind.ANNOTATION,),
            statuses=request.statuses,
            annotation_schema_qualified_id=qualified_id,
            annotation_target_kind=request.target_kind,
            limit=request.limit,
            offset=request.offset,
        )
        matched_count = count_assertion_claims(
            user_conn,
            kinds=(AssertionKind.ANNOTATION,),
            statuses=request.statuses,
            annotation_schema_qualified_id=qualified_id,
            annotation_target_kind=request.target_kind,
        )
        drift_count = count_assertion_claims(
            user_conn,
            kinds=(AssertionKind.ANNOTATION,),
            statuses=request.statuses,
            annotation_schema_prefix=same_schema_prefix,
            annotation_schema_excluded_qualified_id=qualified_id,
            annotation_target_kind=request.target_kind,
        )
        drift_rows = list_assertion_claims(
            user_conn,
            kinds=(AssertionKind.ANNOTATION,),
            statuses=request.statuses,
            annotation_schema_prefix=same_schema_prefix,
            annotation_schema_excluded_qualified_id=qualified_id,
            annotation_target_kind=request.target_kind,
            limit=_MAX_DIAGNOSTICS,
        )
        source_candidates: dict[str, ArchiveAssertionEnvelope] = {}
        judgments: dict[str, ArchiveAssertionEnvelope] = {}
        for assertion in assertions:
            for superseded_ref in assertion.supersedes:
                if not superseded_ref.startswith("assertion:"):
                    continue
                source = read_assertion_envelope(user_conn, superseded_ref.removeprefix("assertion:"))
                if source is not None:
                    source_candidates[superseded_ref] = source
            assertion_ref = f"assertion:{assertion.assertion_id}"
            source_ref = next((ref for ref in assertion.supersedes if ref in source_candidates), assertion_ref)
            judgment = read_latest_candidate_judgment(user_conn, source_ref)
            if judgment is not None:
                judgments[source_ref] = judgment

    selection_truncated = request.offset + len(assertions) < matched_count
    selected = list(assertions)
    diagnostics: list[AnnotationJoinDiagnostic] = []
    rows: list[AnnotationStructuralJoinRow] = []
    missing_count = 0
    ambiguous_count = 0
    invalid_count = 0
    summary_cache: dict[str, ArchiveSessionSummary | None] = {}

    def diagnose(code: AnnotationJoinDiagnosticCode, assertion_ref: str, target_ref: str, detail: str) -> None:
        if len(diagnostics) < _MAX_DIAGNOSTICS:
            diagnostics.append(
                AnnotationJoinDiagnostic(
                    code=code,
                    assertion_ref=assertion_ref,
                    target_ref=target_ref,
                    detail=_bounded_detail(detail),
                )
            )

    for drift in drift_rows:
        diagnose(
            "schema_drift",
            f"assertion:{drift.assertion_id}",
            drift.target_ref,
            f"row schema {_schema_stamp(drift.value)!r}; expected {qualified_id!r}",
        )

    with ArchiveStore.open_existing(poly.archive_root) as archive:
        for assertion in selected:
            assertion_ref = f"assertion:{assertion.assertion_id}"
            stamp = _schema_stamp(assertion.value)
            if registry_drift:
                diagnose(
                    "schema_drift",
                    assertion_ref,
                    assertion.target_ref,
                    f"row schema {stamp!r}; expected {qualified_id!r}",
                )
                continue
            value = _typed_value(assertion.value)
            errors = (
                ["annotation value is not a JSON object"]
                if value is None
                else validate_annotation_row(
                    schema,
                    target_ref=assertion.target_ref,
                    value=value,
                    evidence_refs=assertion.evidence_refs,
                )
            )
            if errors:
                invalid_count += 1
                diagnose("invalid_value", assertion_ref, assertion.target_ref, "; ".join(errors))
                continue
            resolution = await poly.resolve_ref(assertion.target_ref)
            if not resolution.resolved:
                missing_count += 1
                diagnose(
                    "missing_target", assertion_ref, assertion.target_ref, "; ".join(resolution.caveats) or "not found"
                )
                continue
            payload = resolution.payload
            attempt_raw = payload.get("attempt") if payload else None
            attempt = cast(dict[str, object], attempt_raw) if isinstance(attempt_raw, dict) else None
            if attempt is not None and attempt.get("mapping_state") == "ambiguous":
                ambiguous_count += 1
                diagnose(
                    "ambiguous_target", assertion_ref, assertion.target_ref, "delegation mapping_state is ambiguous"
                )
            session_ref = next((ref for ref in resolution.object_refs if ref.startswith("session:")), None)
            summary = _summary_for_ref(archive, session_ref, summary_cache)
            structural = _structural_document(
                target_ref=assertion.target_ref,
                payload_kind=resolution.payload_kind,
                payload=payload,
                summary=summary,
            )
            source_ref = next((ref for ref in assertion.supersedes if ref in source_candidates), assertion_ref)
            source = source_candidates.get(source_ref)
            judgment = judgments.get(source_ref)
            judgment_value = judgment.value if judgment is not None and isinstance(judgment.value, dict) else {}
            rows.append(
                AnnotationStructuralJoinRow(
                    assertion_ref=assertion_ref,
                    batch_ref=_batch_ref(assertion.value, assertion.scope_ref),
                    schema_id=schema.schema_id,
                    schema_version=schema.version,
                    status=assertion.status,
                    labeler_ref=source.author_ref if source is not None else assertion.author_ref,
                    adjudicator_ref=judgment.author_ref if judgment is not None else None,
                    source_assertion_ref=source_ref,
                    judgment_ref=None if judgment is None else f"assertion:{judgment.assertion_id}",
                    judgment_decision=(
                        str(judgment_value["decision"]) if isinstance(judgment_value.get("decision"), str) else None
                    ),
                    judgment_reason=judgment.body_text if judgment is not None else None,
                    supersedes=tuple(assertion.supersedes),
                    target_ref=assertion.target_ref,
                    value=require_json_document(value, context="joined annotation value"),
                    evidence_refs=tuple(assertion.evidence_refs),
                    structural=structural,
                )
            )

    target_counts = Counter(row.target_ref for row in rows)
    multi_label_target_count = sum(count > 1 for count in target_counts.values())
    duplicate_label_count = sum(max(count - 1, 0) for count in target_counts.values())
    final_schema_drift_count = drift_count + (matched_count if registry_drift else 0)
    return AnnotationStructuralJoinResult(
        qualified_schema_id=qualified_id,
        requested_statuses=request.statuses,
        selected_annotation_count=len(selected),
        matched_annotation_count=matched_count,
        offset=request.offset,
        next_offset=request.offset + len(selected) if selection_truncated else None,
        selection_truncated=selection_truncated,
        joined_count=len(rows),
        missing_target_count=missing_count,
        ambiguous_target_count=ambiguous_count,
        schema_drift_count=final_schema_drift_count,
        invalid_value_count=invalid_count,
        multi_label_target_count=multi_label_target_count,
        duplicate_label_count=duplicate_label_count,
        diagnostics_truncated=(missing_count + ambiguous_count + final_schema_drift_count + invalid_count)
        > len(diagnostics),
        diagnostics=tuple(diagnostics),
        rows=tuple(rows),
        groups=_groups(rows, request.group_by),
    )


__all__ = [
    "AnnotationJoinDiagnostic",
    "AnnotationStructuralGroup",
    "AnnotationStructuralJoinError",
    "AnnotationStructuralJoinRequest",
    "AnnotationStructuralJoinResult",
    "AnnotationStructuralJoinRow",
    "join_typed_annotations",
]
