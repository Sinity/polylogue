"""Read-only corpus acceptance measurements.

The measurements compare durable source evidence with the indexed read model
through the same archive tiers used by the production maintenance gate. Each
measure is reported separately so one unresolved corpus defect cannot become a
generic green result.

Revision counts are only approximately comparable across schema generations:
``raw_session_memberships.message_count`` is historical parser evidence while
the current index may represent former messages as ``session_events``. Event
reclassification is reported as an explanation, not silently treated as a
pass for a genuinely missing message.
"""

from __future__ import annotations

import collections
import sqlite3
from typing import Any

DEFAULT_SAMPLE_LIMIT = 10


def audit_absences(
    source: sqlite3.Connection,
    index: sqlite3.Connection,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Find logical documents backed by source evidence but absent from index."""
    present_ids = {str(row[0]) for row in index.execute("SELECT session_id FROM sessions")}
    by_document: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for origin, membership_provider_session_id, decision in source.execute(
        """
        SELECT r.origin, m.provider_session_id, COALESCE(m.decision, '<none>')
        FROM raw_session_memberships AS m
        JOIN raw_sessions AS r USING (raw_id)
        """
    ):
        by_document[(str(origin), str(membership_provider_session_id))].add(str(decision))

    unattributable_sample: list[str] = []
    unattributable = 0
    non_session_artifacts = 0
    for raw_id, origin, logical_source_key, native_id, status in source.execute(
        """
        SELECT r.raw_id, r.origin, r.logical_source_key, r.native_id,
               COALESCE(c.status, '')
        FROM raw_sessions AS r
        LEFT JOIN raw_membership_census AS c USING (raw_id)
        WHERE NOT EXISTS (
            SELECT 1 FROM raw_session_memberships AS m WHERE m.raw_id = r.raw_id
        )
        """
    ):
        if status == "non_session":
            non_session_artifacts += 1
            continue
        provider_session_id: str | None = None
        if logical_source_key and ":" in str(logical_source_key):
            provider_session_id = str(logical_source_key).split(":", 1)[1]
        elif native_id:
            provider_session_id = str(native_id)
        if provider_session_id:
            by_document[(str(origin), provider_session_id)].add("<byte-revision>")
        else:
            unattributable += 1
            if len(unattributable_sample) < sample_limit:
                unattributable_sample.append(str(raw_id))

    absent: collections.Counter[tuple[str, str]] = collections.Counter()
    documents_known_by_origin: collections.Counter[str] = collections.Counter()
    documents_present_by_origin: collections.Counter[str] = collections.Counter()
    samples: dict[str, list[str]] = collections.defaultdict(list)
    for (origin, provider_session_id), decisions in by_document.items():
        documents_known_by_origin[origin] += 1
        if f"{origin}:{provider_session_id}" in present_ids:
            documents_present_by_origin[origin] += 1
            continue
        if decisions == {"<byte-revision>"}:
            cause = "byte-revision-governed"
        elif decisions == {"ambiguous"}:
            cause = "ambiguous-only"
        elif "ambiguous" in decisions:
            cause = "mixed-ambiguous"
        else:
            cause = "settled-yet-absent"
        absent[(origin, cause)] += 1
        if len(samples[cause]) < sample_limit:
            samples[cause].append(f"{origin}:{provider_session_id}")

    return {
        "documents_known": len(by_document),
        "documents_present": len(by_document) - sum(absent.values()),
        "documents_known_by_origin": dict(sorted(documents_known_by_origin.items())),
        "documents_present_by_origin": dict(sorted(documents_present_by_origin.items())),
        "absent_total": sum(absent.values()),
        "raws_without_attributable_identity": unattributable,
        "membershipless_non_session_artifacts_excluded": non_session_artifacts,
        "absent_by_origin_cause": {
            f"{origin}/{cause}": count
            for (origin, cause), count in sorted(absent.items(), key=lambda item: (-item[1], item[0]))
        },
        "samples": dict(samples),
        "unattributable_sample": unattributable_sample,
    }


def audit_attachment_fidelity(index: sqlite3.Connection) -> dict[str, Any]:
    """Report attachment acquisition by origin, upload origin, and status.

    ``unavailable`` is terminal only when the reference retains structured
    provenance explaining where the unavailable bytes came from. An
    unprovenanced terminal status is indistinguishable from a blanket waiver
    and remains an actionable fidelity failure.
    """
    rows = index.execute(
        """
        SELECT s.origin,
               COALESCE(r.upload_origin, '<none>'),
               a.acquisition_status,
               CASE WHEN a.acquisition_status = 'unavailable'
                          AND NOT (
                              NULLIF(TRIM(COALESCE(r.upload_origin, '')), '') IS NOT NULL
                              OR NULLIF(TRIM(COALESCE(r.source_url, '')), '') IS NOT NULL
                              OR EXISTS (
                                  SELECT 1
                                  FROM attachment_native_ids AS n
                                  WHERE n.ref_id = r.ref_id
                              )
                          )
                    THEN 1 ELSE 0 END,
               COUNT(*)
        FROM attachments AS a
        JOIN attachment_refs AS r USING (attachment_id)
        JOIN sessions AS s USING (session_id)
        GROUP BY 1, 2, 3, 4
        """
    ).fetchall()
    breakdown: dict[str, int] = {}
    counts = collections.Counter[str]()
    unprovenanced_unavailable = 0
    for origin, upload_origin, status, unprovenanced, count in rows:
        amount = int(count)
        breakdown[f"{origin}/{upload_origin}/{status}"] = amount
        counts[str(status)] += amount
        if int(unprovenanced):
            unprovenanced_unavailable += amount
    return {
        "refs_acquired": counts["acquired"],
        "refs_unfetched": counts["unfetched"],
        "refs_unavailable": counts["unavailable"],
        "refs_unavailable_without_provenance": unprovenanced_unavailable,
        "refs_not_acquired": counts["unfetched"] + counts["unavailable"],
        "breakdown": breakdown,
    }


def audit_revision_fidelity(
    source: sqlite3.Connection,
    index: sqlite3.Connection,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Find indexed documents smaller than the best recorded revision."""
    best: dict[tuple[str, str], int] = {}
    for origin, provider_session_id, message_count in source.execute(
        """
        SELECT r.origin, m.provider_session_id, m.message_count
        FROM raw_session_memberships AS m
        JOIN raw_sessions AS r USING (raw_id)
        WHERE m.message_count IS NOT NULL
        """
    ):
        key = (str(origin), str(provider_session_id))
        best[key] = max(int(message_count), best.get(key, -1))

    messages = {
        str(row[0]): int(row[1]) for row in index.execute("SELECT session_id, COUNT(*) FROM messages GROUP BY 1")
    }
    events = {
        str(row[0]): int(row[1])
        for row in index.execute(
            """
            SELECT session_id, COUNT(*)
            FROM session_events
            WHERE source_message_id IS NOT NULL
               OR NULLIF(TRIM(COALESCE(source_message_provider_id, '')), '') IS NOT NULL
            GROUP BY 1
            """
        )
    }
    shortfalls: collections.Counter[str] = collections.Counter()
    explained: collections.Counter[str] = collections.Counter()
    worst: list[dict[str, Any]] = []
    for (origin, provider_session_id), best_count in best.items():
        session_id = f"{origin}:{provider_session_id}"
        have_messages = messages.get(session_id)
        if have_messages is None or have_messages >= best_count:
            continue
        have_events = events.get(session_id, 0)
        if have_messages + have_events >= best_count:
            explained[origin] += 1
            continue
        shortfalls[origin] += 1
        worst.append(
            {
                "session_id": session_id,
                "indexed_messages": have_messages,
                "indexed_events": have_events,
                "best_recorded_messages": best_count,
            }
        )
    worst.sort(key=lambda item: item["indexed_messages"] - item["best_recorded_messages"])
    return {
        "unexplained_shortfall": sum(shortfalls.values()),
        "explained_by_event_reclassification": sum(explained.values()),
        "unexplained_by_origin": dict(shortfalls.most_common()),
        "explained_by_origin": dict(explained.most_common()),
        "worst": worst[:sample_limit],
    }


__all__ = [
    "DEFAULT_SAMPLE_LIMIT",
    "audit_absences",
    "audit_attachment_fidelity",
    "audit_revision_fidelity",
]
