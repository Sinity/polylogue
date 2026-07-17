"""Session latency profile materialization helpers."""

from __future__ import annotations

import json as _json
from datetime import datetime

from polylogue.archive.models import Session
from polylogue.archive.semantic.timing import SessionLatencyProfileFacts, compute_session_latency_profile
from polylogue.archive.session.session_profile import SessionProfile
from polylogue.core.types import SessionId
from polylogue.storage.runtime import SessionLatencyProfileRecord
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION

from .profiles import now_iso


def _iso_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def build_latency_profile_facts(
    session: Session,
    profile: SessionProfile,
) -> SessionLatencyProfileFacts:
    return compute_session_latency_profile(
        list(session.messages),
        list(session.session_events),
        session_end=session.updated_at or profile.last_message_at,
        tool_call_count_by_category=dict(profile.tool_categories),
    )


def build_session_latency_profile_record(
    session: Session,
    profile: SessionProfile,
    facts: SessionLatencyProfileFacts,
    *,
    materialized_at: str | None = None,
    input_high_water_mark: str | None = None,
    input_high_water_mark_source: str | None = None,
    input_row_count: int = 0,
) -> SessionLatencyProfileRecord:
    built_at = materialized_at or now_iso()
    evidence = {
        **facts.to_dict(),
        "construct_boundary": (
            "agent-response time includes both model output delay and any intervening tool execution; "
            "provider tool latency requires timestamped session-event pairs"
        ),
    }
    search_text = " \n".join(
        part
        for part in (
            str(session.id),
            session.origin.value,
            session.title or "",
            profile.workflow_shape,
            profile.terminal_state,
        )
        if part
    )
    return SessionLatencyProfileRecord(
        session_id=SessionId(str(session.id)),
        materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
        materialized_at=built_at,
        source_updated_at=_iso_datetime(session.updated_at),
        source_sort_key=float(session.updated_at.timestamp()) if session.updated_at is not None else None,
        input_high_water_mark=input_high_water_mark,
        input_high_water_mark_source=input_high_water_mark_source,
        input_row_count=input_row_count,
        source_name=session.origin.value,
        title=session.title,
        first_message_at=_iso_datetime(profile.first_message_at),
        last_message_at=_iso_datetime(profile.last_message_at),
        canonical_session_date=profile.canonical_session_date.isoformat() if profile.canonical_session_date else None,
        median_tool_call_ms=facts.median_tool_call_ms,
        p90_tool_call_ms=facts.p90_tool_call_ms,
        max_tool_call_ms=facts.max_tool_call_ms,
        stuck_tool_count=facts.stuck_tool_count,
        median_agent_response_ms=facts.median_agent_response_ms,
        median_user_response_ms=facts.median_user_response_ms,
        tool_call_count_by_category_json=_json.dumps(facts.tool_call_count_by_category, sort_keys=True),
        evidence_payload_json=_json.dumps(evidence, sort_keys=True),
        search_text=search_text or str(session.id),
    )


__all__ = ["build_latency_profile_facts", "build_session_latency_profile_record"]
