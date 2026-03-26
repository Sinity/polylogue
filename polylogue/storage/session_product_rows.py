"""Small public root for session-product row families."""

from __future__ import annotations

from polylogue.lib.session_profile import build_session_analysis, build_session_profile
from polylogue.storage.session_product_profile_rows import (
    build_session_profile_record,
    hydrate_session_profile,
)
from polylogue.storage.session_product_row_support import now_iso
from polylogue.storage.session_product_thread_rows import (
    build_work_thread_record,
    hydrate_work_thread,
)
from polylogue.storage.session_product_timeline_rows import (
    build_session_phase_records,
    build_session_work_event_records,
    hydrate_session_phase,
    hydrate_work_event,
)


def build_session_product_records(
    conversation,
) -> tuple[object, list[object], list[object]]:
    analysis = build_session_analysis(conversation)
    profile = build_session_profile(conversation, analysis=analysis)
    materialized_at = now_iso()
    return (
        build_session_profile_record(profile, analysis=analysis, materialized_at=materialized_at),
        build_session_work_event_records(profile, materialized_at=materialized_at),
        build_session_phase_records(profile, materialized_at=materialized_at),
    )


__all__ = [
    "build_session_phase_records",
    "build_session_product_records",
    "build_session_profile_record",
    "build_session_work_event_records",
    "build_work_thread_record",
    "hydrate_session_phase",
    "hydrate_session_profile",
    "hydrate_work_event",
    "hydrate_work_thread",
]
