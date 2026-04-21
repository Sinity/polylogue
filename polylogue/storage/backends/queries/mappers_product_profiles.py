"""Row mappers for durable session-profile records."""

from __future__ import annotations

import sqlite3

from polylogue.archive_product_models import (
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
)
from polylogue.storage.backends.queries.mappers_product_legacy import (
    parse_legacy_payload_dict,
    parse_payload_model,
    session_profile_enrichment_from_legacy,
    session_profile_evidence_from_legacy,
    session_profile_inference_from_legacy,
)
from polylogue.storage.backends.queries.mappers_support import (
    _json_text_tuple,
    _parse_json,
    _row_float,
    _row_get,
    _row_int,
    _row_text,
)
from polylogue.storage.store import SessionProfileRecord
from polylogue.types import ConversationId


def _row_to_session_profile_record(row: sqlite3.Row) -> SessionProfileRecord:
    search_text = row["search_text"]
    evidence_search_text = (_row_get(row, "evidence_search_text", "") or "").strip() or search_text
    inference_search_text = (_row_get(row, "inference_search_text", "") or "").strip() or search_text
    enrichment_search_text = (_row_get(row, "enrichment_search_text", "") or "").strip() or inference_search_text
    legacy_payload = parse_legacy_payload_dict(
        row,
        record_id=row["conversation_id"],
    )
    evidence_payload = parse_payload_model(
        row,
        "evidence_payload_json",
        record_id=row["conversation_id"],
        model=SessionEvidencePayload,
    )
    if evidence_payload is None:
        evidence_payload = session_profile_evidence_from_legacy(row, legacy_payload)
    inference_payload = parse_payload_model(
        row,
        "inference_payload_json",
        record_id=row["conversation_id"],
        model=SessionInferencePayload,
    )
    if inference_payload is None:
        inference_payload = session_profile_inference_from_legacy(row, legacy_payload)
    enrichment_payload = parse_payload_model(
        row,
        "enrichment_payload_json",
        record_id=row["conversation_id"],
        model=SessionEnrichmentPayload,
    )
    if enrichment_payload is None:
        enrichment_payload = session_profile_enrichment_from_legacy(
            row,
            legacy_payload,
            inference_payload=inference_payload,
        )
    return SessionProfileRecord(
        conversation_id=ConversationId(row["conversation_id"]),
        materializer_version=int(_row_int(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_text(row, "source_updated_at"),
        source_sort_key=_row_float(row, "source_sort_key"),
        provider_name=row["provider_name"],
        title=_row_text(row, "title"),
        first_message_at=_row_text(row, "first_message_at"),
        last_message_at=_row_text(row, "last_message_at"),
        canonical_session_date=_row_text(row, "canonical_session_date"),
        repo_paths=_json_text_tuple(_parse_json(_row_get(row, "repo_paths_json"))),
        repo_names=_json_text_tuple(_parse_json(_row_get(row, "repo_names_json"))),
        tags=_json_text_tuple(_parse_json(_row_get(row, "tags_json"))),
        auto_tags=_json_text_tuple(_parse_json(_row_get(row, "auto_tags_json"))),
        message_count=int(_row_int(row, "message_count", 0) or 0),
        substantive_count=int(_row_int(row, "substantive_count", 0) or 0),
        attachment_count=int(_row_int(row, "attachment_count", 0) or 0),
        work_event_count=int(_row_int(row, "work_event_count", 0) or 0),
        phase_count=int(_row_int(row, "phase_count", 0) or 0),
        word_count=int(_row_int(row, "word_count", 0) or 0),
        tool_use_count=int(_row_int(row, "tool_use_count", 0) or 0),
        thinking_count=int(_row_int(row, "thinking_count", 0) or 0),
        total_cost_usd=float(_row_float(row, "total_cost_usd", 0.0) or 0.0),
        total_duration_ms=int(_row_int(row, "total_duration_ms", 0) or 0),
        engaged_duration_ms=int(_row_int(row, "engaged_duration_ms", 0) or 0),
        wall_duration_ms=int(_row_int(row, "wall_duration_ms", 0) or 0),
        cost_is_estimated=bool(int(_row_int(row, "cost_is_estimated", 0) or 0)),
        evidence_payload=evidence_payload,
        inference_payload=inference_payload,
        enrichment_payload=enrichment_payload,
        search_text=search_text,
        evidence_search_text=evidence_search_text,
        inference_search_text=inference_search_text,
        enrichment_search_text=enrichment_search_text,
        enrichment_version=int(_row_int(row, "enrichment_version", 1) or 1),
        enrichment_family=_row_text(row, "enrichment_family") or "scored_session_enrichment",
        inference_version=int(_row_int(row, "inference_version", 1) or 1),
        inference_family=_row_text(row, "inference_family") or "heuristic_session_semantics",
    )


__all__ = ["_row_to_session_profile_record"]
