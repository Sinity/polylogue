"""Typed annotation enrichment against exact structural targets."""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from polylogue.annotations.join_contracts import (
    AnnotationGroupDimension,
    AnnotationJoinDiagnostic,
    AnnotationJoinDiagnosticCode,
    AnnotationStructuralGroup,
    AnnotationStructuralJoinRequest,
    AnnotationStructuralJoinResult,
    AnnotationStructuralJoinRow,
)
from polylogue.annotations.schema import (
    ANNOTATION_SCHEMA_REGISTRY,
    RETIRED_ANNOTATION_TARGET_KINDS,
    normalize_annotation_target_ref,
    validate_annotation_row,
    validate_annotation_value,
)
from polylogue.core.enums import AssertionKind
from polylogue.core.json import JSONDocument, require_json_document
from polylogue.core.refs import ObjectRef
from polylogue.storage.sqlite.archive_tiers.user_annotations import read_durable_annotation_schema
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ArchiveAssertionEnvelope,
    count_assertion_claims,
    list_assertion_claims,
    read_assertion_envelope,
    read_latest_candidate_judgment,
)


class StructuralJoinArchive(Protocol):
    """Minimal facade contract consumed by the product operation."""

    @property
    def archive_root(self) -> Path: ...

    async def resolve_ref(self, ref: str) -> Any: ...

    async def get_session_summary(self, session_id: str) -> object | None: ...


class AnnotationStructuralJoinError(ValueError):
    """Raised when a join request cannot be evaluated honestly."""


_MAX_DIAGNOSTICS = 100


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


async def _summary_for_ref(
    poly: StructuralJoinArchive,
    session_ref: str | None,
    cache: dict[str, object | None],
) -> object | None:
    if session_ref is None:
        return None
    session_id = session_ref.removeprefix("session:")
    if session_id not in cache:
        cache[session_id] = await poly.get_session_summary(session_id)
    return cache[session_id]


def _structural_document(
    *,
    target_ref: str,
    payload_kind: str | None,
    payload: dict[str, object] | None,
    summary: object | None,
) -> JSONDocument:
    target_kind = ObjectRef.parse(target_ref).kind
    attempt_raw = payload.get("attempt") if payload else None
    attempt = cast(dict[str, object], attempt_raw) if isinstance(attempt_raw, dict) else None
    model = attempt.get("dispatch_turn_model") if attempt else None
    mapping_state = attempt.get("mapping_state") if attempt else None
    session_id = getattr(summary, "session_id", None) if summary is not None else None
    if session_id is None and summary is not None:
        session_id = getattr(summary, "id", None)
    origin = getattr(summary, "origin", None) if summary is not None else None
    repo = getattr(summary, "git_repository_url", None) if summary is not None else None
    created_at = getattr(summary, "created_at", None) if summary is not None else None
    updated_at = getattr(summary, "updated_at", None) if summary is not None else None

    def json_value(value: object) -> object:
        return value.isoformat() if isinstance(value, datetime) else value

    document: dict[str, object] = {
        "target_kind": target_kind,
        "payload_kind": payload_kind,
        "session_id": str(session_id) if session_id is not None else None,
        "origin": str(origin) if origin is not None else None,
        "repo": repo,
        "created_at": json_value(created_at),
        "updated_at": json_value(updated_at),
        "model": model if isinstance(model, str) else None,
        "mapping_state": mapping_state if isinstance(mapping_state, str) else None,
    }
    return require_json_document(document, context="annotation structural join")


def _group_value(row: AnnotationStructuralJoinRow, dimension: AnnotationGroupDimension) -> object:
    value = row.structural.get(dimension)
    if dimension == "time" and isinstance(row.structural.get("created_at"), str):
        value = str(row.structural["created_at"])[:10]
    return value


def _groups(
    rows: list[AnnotationStructuralJoinRow],
    dimensions: tuple[AnnotationGroupDimension, ...],
    checkpoint: Callable[[], None],
) -> tuple[AnnotationStructuralGroup, ...]:
    if not dimensions:
        return ()
    grouped: dict[tuple[object, ...], list[AnnotationStructuralJoinRow]] = {}
    for row in rows:
        checkpoint()
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
    *,
    user_conn: sqlite3.Connection,
    user_schema: str | None,
    checkpoint: Callable[[], None],
) -> AnnotationStructuralJoinResult:
    """Join selected typed labels to exact targets, retaining one row per label."""

    user_conn.row_factory = sqlite3.Row
    checkpoint()
    durable = read_durable_annotation_schema(user_conn, request.schema_id, request.schema_version, schema=user_schema)
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
    as_of_ms = int(datetime.now(UTC).timestamp() * 1000)
    checkpoint()
    assertions = list_assertion_claims(
        user_conn,
        schema=user_schema,
        kinds=(AssertionKind.ANNOTATION,),
        statuses=request.statuses,
        annotation_schema_qualified_id=qualified_id,
        annotation_target_kind=request.target_kind,
        as_of_ms=as_of_ms,
        limit=request.limit,
        offset=request.offset,
    )
    checkpoint()
    matched_count = count_assertion_claims(
        user_conn,
        schema=user_schema,
        kinds=(AssertionKind.ANNOTATION,),
        statuses=request.statuses,
        annotation_schema_qualified_id=qualified_id,
        annotation_target_kind=request.target_kind,
        as_of_ms=as_of_ms,
    )
    checkpoint()
    drift_count = count_assertion_claims(
        user_conn,
        schema=user_schema,
        kinds=(AssertionKind.ANNOTATION,),
        statuses=request.statuses,
        annotation_schema_prefix=same_schema_prefix,
        annotation_schema_excluded_qualified_id=qualified_id,
        annotation_target_kind=request.target_kind,
        as_of_ms=as_of_ms,
    )
    checkpoint()
    drift_rows = list_assertion_claims(
        user_conn,
        schema=user_schema,
        kinds=(AssertionKind.ANNOTATION,),
        statuses=request.statuses,
        annotation_schema_prefix=same_schema_prefix,
        annotation_schema_excluded_qualified_id=qualified_id,
        annotation_target_kind=request.target_kind,
        as_of_ms=as_of_ms,
        limit=_MAX_DIAGNOSTICS,
    )
    source_candidates: dict[str, ArchiveAssertionEnvelope] = {}
    judgments: dict[str, ArchiveAssertionEnvelope] = {}
    for assertion in assertions:
        checkpoint()
        for superseded_ref in assertion.supersedes:
            checkpoint()
            if not superseded_ref.startswith("assertion:"):
                continue
            source = read_assertion_envelope(user_conn, superseded_ref.removeprefix("assertion:"), schema=user_schema)
            if source is not None:
                source_candidates[superseded_ref] = source
        assertion_ref = f"assertion:{assertion.assertion_id}"
        source_ref = next((ref for ref in assertion.supersedes if ref in source_candidates), assertion_ref)
        judgment = read_latest_candidate_judgment(user_conn, source_ref, schema=user_schema)
        if judgment is not None:
            judgments[source_ref] = judgment

    selection_truncated = request.offset + len(assertions) < matched_count
    selected = list(assertions)
    diagnostics: list[AnnotationJoinDiagnostic] = []
    rows: list[AnnotationStructuralJoinRow] = []
    missing_count = 0
    ambiguous_count = 0
    invalid_count = 0
    summary_cache: dict[str, object | None] = {}

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
        checkpoint()
        diagnose(
            "schema_drift",
            f"assertion:{drift.assertion_id}",
            drift.target_ref,
            f"row schema {_schema_stamp(drift.value)!r}; expected {qualified_id!r}",
        )

    for assertion in selected:
        checkpoint()
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
        target_kind = assertion.target_ref.partition(":")[0]
        if target_kind in RETIRED_ANNOTATION_TARGET_KINDS and target_kind in schema.target_ref_kinds:
            try:
                normalize_annotation_target_ref(assertion.target_ref)
            except ValueError:
                pass  # The ordinary validator reports malformed provenance below.
            else:
                errors = (
                    ["annotation value is not a JSON object"]
                    if value is None
                    else validate_annotation_value(schema, value)
                )
                if schema.evidence_policy == "required" and not assertion.evidence_refs:
                    errors.append("schema requires evidence_refs and none were provided")
                if not errors:
                    missing_count += 1
                    diagnose(
                        "missing_target",
                        assertion_ref,
                        assertion.target_ref,
                        f"target kind {target_kind!r} is retired; structural resolution is unsupported",
                    )
                    continue
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
        exact_session = target_kind != "session" or assertion.target_ref in resolution.object_refs
        if not resolution.resolved or not exact_session:
            missing_count += 1
            diagnose(
                "missing_target",
                assertion_ref,
                assertion.target_ref,
                "; ".join(resolution.caveats) or "exact target not found",
            )
            continue
        payload = resolution.payload
        attempt_raw = payload.get("attempt") if payload else None
        attempt = cast(dict[str, object], attempt_raw) if isinstance(attempt_raw, dict) else None
        # polylogue-1vpm.7 retired 'ambiguous' from delegation_facts.mapping_state
        # entirely (a cardinality mismatch is no longer a reachable pairing
        # state) -- this branch is now dead against current data. Left in
        # place (rather than deleted) because AnnotationJoinDiagnosticCode's
        # 'ambiguous_target' member and ambiguous_target_count are a broader,
        # separately-scoped annotation-join contract (Pydantic field, not
        # exclusive to delegation) that a full retirement would need to
        # regenerate schemas for; not part of this bead's write scope.
        if attempt is not None and attempt.get("mapping_state") == "ambiguous":
            ambiguous_count += 1
            diagnose("ambiguous_target", assertion_ref, assertion.target_ref, "delegation mapping_state is ambiguous")
        session_ref = next((ref for ref in resolution.object_refs if ref.startswith("session:")), None)
        summary = await _summary_for_ref(poly, session_ref, summary_cache)
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
        groups=_groups(rows, request.group_by, checkpoint),
    )


__all__ = ["AnnotationStructuralJoinError", "StructuralJoinArchive", "join_typed_annotations"]
