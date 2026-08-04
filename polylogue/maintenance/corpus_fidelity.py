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

import argparse
import collections
import json
import sqlite3
from pathlib import Path
from typing import Any

from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

DEFAULT_ROOT = Path("/realm/db/polylogue")
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
    for raw_id, origin, logical_source_key, native_id in source.execute(
        """
        SELECT r.raw_id, r.origin, r.logical_source_key, r.native_id
        FROM raw_sessions AS r
        WHERE NOT EXISTS (
            SELECT 1 FROM raw_session_memberships AS m WHERE m.raw_id = r.raw_id
        )
        """
    ):
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
    samples: dict[str, list[str]] = collections.defaultdict(list)
    for (origin, provider_session_id), decisions in by_document.items():
        if f"{origin}:{provider_session_id}" in present_ids:
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
        "absent_total": sum(absent.values()),
        "raws_without_attributable_identity": unattributable,
        "absent_by_origin_cause": {
            f"{origin}/{cause}": count
            for (origin, cause), count in sorted(absent.items(), key=lambda item: (-item[1], item[0]))
        },
        "samples": dict(samples),
        "unattributable_sample": unattributable_sample,
    }


def audit_attachment_fidelity(index: sqlite3.Connection) -> dict[str, Any]:
    """Report attachment acquisition by origin, upload origin, and status.

    ``unavailable`` is a typed terminal outcome and does not fail the gate.
    Only ``unfetched`` references are actionable fidelity failures.
    """
    rows = index.execute(
        """
        SELECT s.origin, COALESCE(r.upload_origin, '<none>'), a.acquisition_status, COUNT(*)
        FROM attachments AS a
        JOIN attachment_refs AS r USING (attachment_id)
        JOIN sessions AS s USING (session_id)
        GROUP BY 1, 2, 3
        """
    ).fetchall()
    breakdown: dict[str, int] = {}
    counts = collections.Counter[str]()
    for origin, upload_origin, status, count in rows:
        amount = int(count)
        breakdown[f"{origin}/{upload_origin}/{status}"] = amount
        counts[str(status)] += amount
    return {
        "refs_acquired": counts["acquired"],
        "refs_unfetched": counts["unfetched"],
        "refs_unavailable": counts["unavailable"],
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
        str(row[0]): int(row[1]) for row in index.execute("SELECT session_id, COUNT(*) FROM session_events GROUP BY 1")
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


def audit_corpus_fidelity(archive_root: Path, *, sample_limit: int = DEFAULT_SAMPLE_LIMIT) -> dict[str, Any]:
    """Run all corpus measurements against the active production tiers."""
    source_path = archive_root / "source.db"
    index_path = resolve_active_index_path(archive_root)
    with open_readonly_connection(source_path) as source, open_readonly_connection(index_path) as index:
        return {
            "archive_root": str(archive_root),
            "absences": audit_absences(source, index, sample_limit=sample_limit),
            "attachment_fidelity": audit_attachment_fidelity(index),
            "revision_fidelity": audit_revision_fidelity(source, index, sample_limit=sample_limit),
        }


def _failing_measures(report: dict[str, Any]) -> dict[str, int]:
    absence = report["absences"]
    attachment = report["attachment_fidelity"]
    revision = report["revision_fidelity"]
    failures = {
        "absent_documents": int(absence["absent_total"]),
        "raws_without_attributable_identity": int(absence["raws_without_attributable_identity"]),
        "unfetched_attachment_refs": int(attachment["refs_unfetched"]),
        "unexplained_revision_shortfall": int(revision["unexplained_shortfall"]),
    }
    return {key: value for key, value in failures.items() if value}


def format_report(report: dict[str, Any]) -> str:
    """Render the operator-facing report used by the compatibility script."""
    absence = report["absences"]
    attachment = report["attachment_fidelity"]
    revision = report["revision_fidelity"]
    lines = [
        f"archive: {report['archive_root']}",
        f"\nABSENCES  {absence['absent_total']} of {absence['documents_known']} known documents",
    ]
    lines.extend(f"    {count:6d}  {key}" for key, count in absence["absent_by_origin_cause"].items())
    if absence["raws_without_attributable_identity"]:
        lines.append(f"    {absence['raws_without_attributable_identity']:6d}  raws-without-attributable-identity")
    lines.append(
        f"\nATTACHMENT FIDELITY  acquired={attachment['refs_acquired']} not-acquired={attachment['refs_not_acquired']}"
    )
    lines.extend(
        f"    {count:6d}  {key}"
        for key, count in sorted(attachment["breakdown"].items(), key=lambda item: -item[1])[:8]
    )
    lines.append(
        f"\nREVISION FIDELITY  {revision['unexplained_shortfall']} unexplained, "
        f"{revision['explained_by_event_reclassification']} explained by event reclassification"
    )
    lines.extend(
        f"    {count:6d}  {origin}  (unexplained)" for origin, count in revision["unexplained_by_origin"].items()
    )
    failing = _failing_measures(report)
    if failing:
        lines.append("\nFAILING MEASURES")
        lines.extend(f"    {value:8d}  {key}" for key, value in failing.items())
    lines.append(f"\nVERDICT: {'PASS' if not failing else 'FAIL'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    report = audit_corpus_fidelity(args.archive_root)
    failing = _failing_measures(report)
    report["failing_measures"] = failing
    report["verdict"] = "PASS" if not failing else "FAIL"
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(format_report(report))
    return 0 if not failing else 1


__all__ = [
    "DEFAULT_ROOT",
    "DEFAULT_SAMPLE_LIMIT",
    "audit_absences",
    "audit_attachment_fidelity",
    "audit_corpus_fidelity",
    "audit_revision_fidelity",
    "format_report",
    "main",
]
