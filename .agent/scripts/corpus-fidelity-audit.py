#!/usr/bin/env python3
"""Corpus-wide acceptance gate: no absences, maximum fidelity.

The operator's bar for the archive is stronger than "the rebuild finished":
every logical document the archive holds evidence for must be present in the
index, and where several revisions of one document exist, the indexed one must
be the best evidence available.

Neither half is covered by the existing checks. ``verify-archive``'s
``source-index-coverage`` counts superseded revisions as missing work
(polylogue-ey3r), so it cannot reach zero on any real archive and cannot serve
as an acceptance criterion. Nothing at all measures fidelity -- an archive can
report perfect coverage while every attachment it holds bytes for is recorded
``unfetched``, which is exactly the state measured on 2026-07-30.

Read-only. Every connection opens ``mode=ro``; this never mutates the archive.

Usage:
    python3 .agent/scripts/corpus-fidelity-audit.py [--archive-root PATH]
                                                    [--json OUT.json]

Exit status is 1 when either half fails, so it can gate a rebuild.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("/realm/db/polylogue")


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def audit_absences(source: sqlite3.Connection, index: sqlite3.Connection) -> dict[str, Any]:
    """Logical documents the archive has evidence for but does not surface.

    A logical document is one (origin, provider_session_id). It is *absent*
    when no session row carries its id, regardless of why -- an ambiguous
    cohort that refused to arbitrate is just as absent to a reader as one that
    was never ingested, which is the point of the operator's bar.
    """
    present_ids = {row[0] for row in index.execute("SELECT session_id FROM sessions")}
    by_document: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for origin, provider_session_id, decision in source.execute(
        """
        SELECT r.origin, m.provider_session_id, COALESCE(m.decision, '<none>')
        FROM raw_session_memberships AS m
        JOIN raw_sessions AS r USING (raw_id)
        """
    ):
        by_document[(origin, provider_session_id)].add(decision)

    absent: collections.Counter[tuple[str, str]] = collections.Counter()
    samples: dict[str, list[str]] = collections.defaultdict(list)
    for (origin, provider_session_id), decisions in by_document.items():
        if f"{origin}:{provider_session_id}" in present_ids:
            continue
        # Classify why, so a fix's effect is attributable rather than a single
        # number that moves for unknown reasons.
        if decisions == {"ambiguous"}:
            cause = "ambiguous-only"
        elif "ambiguous" in decisions:
            cause = "mixed-ambiguous"
        else:
            cause = "settled-yet-absent"
        absent[(origin, cause)] += 1
        if len(samples[cause]) < 5:
            samples[cause].append(f"{origin}:{provider_session_id}")

    return {
        "documents_known": len(by_document),
        "documents_present": len(by_document) - sum(absent.values()),
        "absent_total": sum(absent.values()),
        "absent_by_origin_cause": {f"{o}/{c}": n for (o, c), n in sorted(absent.items(), key=lambda kv: -kv[1])},
        "samples": dict(samples),
    }


def audit_attachment_fidelity(index: sqlite3.Connection) -> dict[str, Any]:
    """Attachments the index reports unfetched.

    ``acquisition_status`` is the honest record of whether bytes were read, so
    an ``unfetched`` row is only a fidelity failure when the bytes are in fact
    obtainable. This reports the split by origin and upload_origin rather than
    a bare total: a Drive-hosted reference that was never fetched is
    actionable, a genuinely byte-less attachment kind is not.
    """
    rows = index.execute(
        """
        SELECT s.origin, COALESCE(f.upload_origin, '<none>'), a.acquisition_status, COUNT(*)
        FROM attachments AS a
        JOIN attachment_refs AS f USING (attachment_id)
        JOIN sessions AS s USING (session_id)
        GROUP BY 1, 2, 3
        """
    ).fetchall()
    breakdown: dict[str, int] = {}
    acquired = unfetched = 0
    for origin, upload_origin, status, count in rows:
        breakdown[f"{origin}/{upload_origin}/{status}"] = count
        if status == "acquired":
            acquired += count
        else:
            unfetched += count
    return {
        "refs_acquired": acquired,
        "refs_not_acquired": unfetched,
        "breakdown": breakdown,
    }


def audit_revision_fidelity(source: sqlite3.Connection, index: sqlite3.Connection) -> dict[str, Any]:
    """Documents whose indexed content is not the best revision on hand.

    Uses stored evidence only -- no reparsing, so this stays cheap enough to run
    as a gate. ``raw_session_memberships.message_count`` is the recorded size of
    each revision; when the indexed session carries less evidence than the
    largest revision recorded for that document, the archive is surfacing a
    smaller revision than it holds.

    Counts ``messages + session_events``, not messages alone. ``message_count``
    was recorded by whichever parser censused that raw, and index v46
    reclassified a large share of Codex/Claude Code rows from chat turns into
    typed session events -- a deliberate change, not loss. Comparing a stale
    message-only count against current message rows makes every such session
    look catastrophically truncated: measured before this correction, one Codex
    session read as 15 indexed vs 68,553 recorded, when it actually holds 15
    messages plus 84,612 events. That produced 474 false positives, 294 of them
    Codex, and no real ones.

    Two honest limits remain. It cannot see attachment-only enrichment (equal
    message_count, more acquired bytes) -- that is polylogue-bu1i's shape, so
    pair this with ``audit_attachment_fidelity``. And cross-generation counts
    are only approximately comparable at all, so treat a small residue here as
    a prompt to investigate rather than as proof of loss.
    """
    best: dict[tuple[str, str], int] = {}
    for origin, provider_session_id, message_count in source.execute(
        """
        SELECT r.origin, m.provider_session_id, m.message_count
        FROM raw_session_memberships AS m
        JOIN raw_sessions AS r USING (raw_id)
        WHERE m.message_count IS NOT NULL
        """
    ):
        key = (origin, provider_session_id)
        if message_count > best.get(key, -1):
            best[key] = message_count

    indexed: dict[str, int] = collections.defaultdict(int)
    for table in ("messages", "session_events"):
        for session_id, count in index.execute(
            f"SELECT session_id, COUNT(*) FROM {table} GROUP BY 1"  # table name is a fixed literal, not input
        ):
            indexed[session_id] += count
    for (session_id,) in index.execute("SELECT session_id FROM sessions"):
        indexed.setdefault(session_id, 0)

    shortfalls: collections.Counter[str] = collections.Counter()
    worst: list[dict[str, Any]] = []
    for (origin, provider_session_id), best_count in best.items():
        session_id = f"{origin}:{provider_session_id}"
        have = indexed.get(session_id)
        if have is None or have >= best_count:
            continue
        shortfalls[origin] += 1
        worst.append({"session_id": session_id, "indexed": have, "best_recorded": best_count})
    worst.sort(key=lambda item: item["indexed"] - item["best_recorded"])
    return {
        "documents_below_best_evidence": sum(shortfalls.values()),
        "by_origin": dict(shortfalls.most_common()),
        "worst": worst[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    source = _ro(args.archive_root / "source.db")
    index = _ro(args.archive_root / "index.db")

    report = {
        "archive_root": str(args.archive_root),
        "absences": audit_absences(source, index),
        "attachment_fidelity": audit_attachment_fidelity(index),
        "revision_fidelity": audit_revision_fidelity(source, index),
    }
    absent = report["absences"]["absent_total"]
    below = report["revision_fidelity"]["documents_below_best_evidence"]
    report["verdict"] = "PASS" if absent == 0 and below == 0 else "FAIL"

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))

    a = report["absences"]
    print(f"archive: {args.archive_root}")
    print(f"\nABSENCES  {a['absent_total']} of {a['documents_known']} known documents")
    for key, count in a["absent_by_origin_cause"].items():
        print(f"    {count:6d}  {key}")
    f = report["attachment_fidelity"]
    print(f"\nATTACHMENT FIDELITY  acquired={f['refs_acquired']}  not-acquired={f['refs_not_acquired']}")
    for key, count in sorted(f["breakdown"].items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {count:6d}  {key}")
    r = report["revision_fidelity"]
    print(f"\nREVISION FIDELITY  {r['documents_below_best_evidence']} document(s) below best recorded evidence")
    for key, count in r["by_origin"].items():
        print(f"    {count:6d}  {key}")

    print(f"\nVERDICT: {report['verdict']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
