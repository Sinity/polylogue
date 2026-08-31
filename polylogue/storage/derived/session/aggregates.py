"""Read-time grouping helpers for session insight summaries."""

from __future__ import annotations

from polylogue.insights.archive import date_from_iso
from polylogue.storage.runtime import SessionProfileRecord

_PROFILE_BUCKET_DAY_SQL = (
    "COALESCE(sp.canonical_session_date, "
    "date(COALESCE(sp.first_message_at, json_extract(sp.evidence_payload_json, '$.created_at'), "
    "sp.source_updated_at, sp.last_message_at)))"
)


def profile_provider_day(record: SessionProfileRecord | None) -> tuple[str, str] | None:
    """Return the profile's source and canonical day for refresh accounting."""
    if record is None:
        return None
    if record.canonical_session_date:
        return (record.source_name, record.canonical_session_date)
    evidence_created_at = record.evidence_payload.created_at
    day_candidates = [
        record.first_message_at,
        str(evidence_created_at) if evidence_created_at else None,
        record.source_updated_at,
        record.last_message_at,
    ]
    for candidate in day_candidates:
        if not candidate:
            continue
        try:
            return (record.source_name, date_from_iso(str(candidate)[:10]).isoformat())
        except ValueError:
            continue
    return None


__all__ = ["_PROFILE_BUCKET_DAY_SQL", "profile_provider_day"]
