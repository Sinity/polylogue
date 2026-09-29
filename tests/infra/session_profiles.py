"""Real ``session_profiles`` rows, written through the canonical writer.

A hand-written ``INSERT INTO session_profiles (session_id) ...`` leaves the
three stored payloads at their ``'{}'`` defaults, which is a row no production
writer produces and the typed readers refuse (polylogue-77tzg). Tests that
need a profile row build one here instead: the record is a real
``SessionProfileRecord`` and the write is ``replace_session_profile_sync``.

Record fields are passed by name. A field that the evidence or inference
payload also carries (``message_count``, ``repo_names``, ...) is mirrored into
that payload, so the columns and the payload agree the way the materializer
leaves them. ``evidence=``, ``inference=`` and ``enrichment=`` override
individual payload fields.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping

from polylogue.analysis.archive_models import (
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
)
from polylogue.core.types import SessionId
from polylogue.storage.derived.session.records import SessionProfileRecord
from polylogue.storage.derived.session.storage import replace_session_profile_sync

_PAYLOAD_OWNED = frozenset({"evidence_payload", "inference_payload", "enrichment_payload"})


def session_profile_record(
    session_id: str,
    *,
    evidence: Mapping[str, object] | None = None,
    inference: Mapping[str, object] | None = None,
    enrichment: Mapping[str, object] | None = None,
    **fields: object,
) -> SessionProfileRecord:
    """Build a complete profile record for ``session_id``."""
    unknown = set(fields) - (set(SessionProfileRecord.model_fields) - _PAYLOAD_OWNED)
    if unknown:
        raise TypeError(f"not a session profile field: {sorted(unknown)}")
    record_fields = dict(fields)
    search_text = str(record_fields.pop("search_text", None) or session_id)
    evidence_payload = SessionEvidencePayload.model_validate(
        {
            **{key: value for key, value in record_fields.items() if key in SessionEvidencePayload.model_fields},
            **dict(evidence or {}),
        }
    )
    inference_payload = SessionInferencePayload.model_validate(
        {
            **{key: value for key, value in record_fields.items() if key in SessionInferencePayload.model_fields},
            **dict(inference or {}),
        }
    )
    enrichment_payload = SessionEnrichmentPayload.model_validate(dict(enrichment or {}))
    origin, _, _ = session_id.partition(":")
    return SessionProfileRecord.model_validate(
        {
            "logical_session_id": SessionId(session_id),
            "source_name": origin or "fixture",
            "materialized_at": "2026-01-01T00:00:00+00:00",
            "evidence_search_text": search_text,
            "inference_search_text": search_text,
            "enrichment_search_text": search_text,
            **record_fields,
            "session_id": SessionId(session_id),
            "search_text": search_text,
            "evidence_payload": evidence_payload,
            "inference_payload": inference_payload,
            "enrichment_payload": enrichment_payload,
        }
    )


def write_session_profile(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    evidence: Mapping[str, object] | None = None,
    inference: Mapping[str, object] | None = None,
    enrichment: Mapping[str, object] | None = None,
    **fields: object,
) -> SessionProfileRecord:
    """Write one real profile row through the canonical writer and return it."""
    record = session_profile_record(
        session_id,
        evidence=evidence,
        inference=inference,
        enrichment=enrichment,
        **fields,
    )
    replace_session_profile_sync(conn, record)
    return record


__all__ = ["session_profile_record", "write_session_profile"]
