"""Search-text builders for session-profile products."""

from __future__ import annotations

from polylogue.lib.session_profile import SessionProfile


def profile_evidence_search_text(profile: SessionProfile) -> str:
    parts = [
        profile.provider,
        profile.title or "",
        *profile.repo_paths,
        *profile.cwd_paths,
        *profile.file_paths_touched,
        *profile.tags,
        *profile.branch_names,
        *profile.languages_detected,
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or profile.conversation_id


def profile_inference_search_text(profile: SessionProfile) -> str:
    parts = [
        profile.provider,
        profile.title or "",
        *profile.canonical_projects,
        *profile.auto_tags,
        *(event.summary for event in profile.work_events),
        *(event.kind.value for event in profile.work_events),
        *(phase.kind for phase in profile.phases),
        *(decision.summary for decision in profile.decisions),
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or profile.conversation_id


def profile_search_text(profile: SessionProfile) -> str:
    return " \n".join(
        part
        for part in (
            profile_evidence_search_text(profile),
            profile_inference_search_text(profile),
        )
        if part
    ) or profile.conversation_id


def profile_enrichment_search_text(profile: SessionProfile, enrichment_payload: dict[str, object]) -> str:
    blockers = tuple(str(item) for item in enrichment_payload.get("blockers", []) or [])
    support_signals = tuple(str(item) for item in enrichment_payload.get("support_signals", []) or [])
    parts = [
        profile.provider,
        profile.title or "",
        str(enrichment_payload.get("refined_work_kind") or ""),
        str(enrichment_payload.get("intent_summary") or ""),
        str(enrichment_payload.get("outcome_summary") or ""),
        *profile.canonical_projects,
        *profile.repo_paths,
        *blockers,
        *support_signals,
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or profile.conversation_id
