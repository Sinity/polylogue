"""Build and hydrate session-profile storage rows."""

from __future__ import annotations

from polylogue.lib.session_profile import SessionAnalysis, SessionProfile
from polylogue.storage.session_product_profile_payloads import (
    profile_evidence_payload,
    profile_inference_payload,
    session_enrichment_payload,
)
from polylogue.storage.session_product_profile_search import (
    profile_enrichment_search_text,
    profile_evidence_search_text,
    profile_inference_search_text,
    profile_search_text,
)
from polylogue.storage.session_product_row_support import now_iso, primary_work_kind
from polylogue.storage.store import (
    SESSION_ENRICHMENT_FAMILY,
    SESSION_ENRICHMENT_VERSION,
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
    SESSION_PRODUCT_MATERIALIZER_VERSION,
    SessionProfileRecord,
)


def build_session_profile_record(
    profile: SessionProfile,
    *,
    analysis: SessionAnalysis | None = None,
    materialized_at: str | None = None,
) -> SessionProfileRecord:
    built_at = materialized_at or now_iso()
    evidence_payload = profile_evidence_payload(profile)
    inference_payload = profile_inference_payload(profile)
    enrichment_payload = session_enrichment_payload(profile, analysis)
    return SessionProfileRecord(
        conversation_id=profile.conversation_id,
        materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
        materialized_at=built_at,
        source_updated_at=profile.updated_at.isoformat() if profile.updated_at else None,
        source_sort_key=profile.updated_at.timestamp() if profile.updated_at else None,
        provider_name=profile.provider,
        title=profile.title,
        first_message_at=profile.first_message_at.isoformat() if profile.first_message_at else None,
        last_message_at=profile.last_message_at.isoformat() if profile.last_message_at else None,
        primary_work_kind=primary_work_kind(profile),
        repo_paths=profile.repo_paths,
        canonical_projects=profile.canonical_projects,
        tags=profile.tags,
        auto_tags=profile.auto_tags,
        message_count=profile.message_count,
        substantive_count=profile.substantive_count,
        attachment_count=profile.attachment_count,
        work_event_count=len(profile.work_events),
        phase_count=len(profile.phases),
        word_count=profile.word_count,
        tool_use_count=profile.tool_use_count,
        thinking_count=profile.thinking_count,
        total_cost_usd=profile.total_cost_usd,
        total_duration_ms=profile.total_duration_ms,
        engaged_duration_ms=profile.engaged_duration_ms,
        wall_duration_ms=profile.wall_duration_ms,
        cost_is_estimated=profile.cost_is_estimated,
        canonical_session_date=(
            profile.canonical_session_date.isoformat()
            if profile.canonical_session_date
            else None
        ),
        evidence_payload=evidence_payload,
        inference_payload=inference_payload,
        enrichment_payload=enrichment_payload,
        search_text=profile_search_text(profile),
        evidence_search_text=profile_evidence_search_text(profile),
        inference_search_text=profile_inference_search_text(profile),
        enrichment_search_text=profile_enrichment_search_text(profile, enrichment_payload),
        enrichment_version=SESSION_ENRICHMENT_VERSION,
        enrichment_family=SESSION_ENRICHMENT_FAMILY,
        inference_version=SESSION_INFERENCE_VERSION,
        inference_family=SESSION_INFERENCE_FAMILY,
    )


def hydrate_session_profile(record: SessionProfileRecord) -> SessionProfile:
    merged_payload = {
        **record.evidence_payload,
        **record.inference_payload,
        "conversation_id": str(record.conversation_id),
        "provider": record.provider_name,
        "title": record.title,
        "first_message_at": record.first_message_at,
        "last_message_at": record.last_message_at,
        "canonical_session_date": record.canonical_session_date,
        "repo_paths": list(record.repo_paths),
        "canonical_projects": list(record.canonical_projects),
        "tags": list(record.tags),
        "auto_tags": list(record.auto_tags),
        "message_count": record.message_count,
        "substantive_count": record.substantive_count,
        "attachment_count": record.attachment_count,
        "work_event_count": record.work_event_count,
        "phase_count": record.phase_count,
        "word_count": record.word_count,
        "tool_use_count": record.tool_use_count,
        "thinking_count": record.thinking_count,
        "total_cost_usd": record.total_cost_usd,
        "total_duration_ms": record.total_duration_ms,
        "engaged_duration_ms": record.engaged_duration_ms,
        "wall_duration_ms": record.wall_duration_ms,
        "cost_is_estimated": record.cost_is_estimated,
        "primary_work_kind": record.primary_work_kind,
    }
    return SessionProfile.from_dict(merged_payload)
