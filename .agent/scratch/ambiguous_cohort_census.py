"""Read-only census of ambiguous membership cohorts, per origin.

Never opens the live archive for write. Only reads source.db (mode=ro) and
blob bytes under /realm/db/polylogue/blob/.
"""

from __future__ import annotations

import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/realm/project/polylogue/.claude/worktrees/agent-a9d80459cee6db3f3")

import io

from polylogue.archive.session_revision_membership import (
    MembershipRevision,
    _frontier,
    _strictly_dominates,
    classify_membership_revisions,
)
from polylogue.core.enums import Origin
from polylogue.core.sources import provider_from_origin
from polylogue.pipeline.ids import session_revision_projection
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload

SOURCE_DB = "file:/realm/db/polylogue/source.db?mode=ro"
BLOB_ROOT = Path("/realm/db/polylogue/blob")

random.seed(20260730)


def blob_path(blob_hash: bytes) -> Path:
    hex_hash = blob_hash.hex()
    return BLOB_ROOT / hex_hash[:2] / hex_hash[2:]


def load_payload(blob_hash: bytes) -> bytes:
    return blob_path(blob_hash).read_bytes()


def parse_raw(provider_str: str, raw_bytes: bytes, source_path: str):
    fallback_id = Path(source_path.split(":")[-1]).stem
    source_name = Path(source_path.split(":")[-1]).name
    records = list(_iter_json_stream(io.BytesIO(raw_bytes), source_name))
    return parse_payload(provider_str, records, fallback_id, source_path=source_path)


def main() -> None:
    con = sqlite3.connect(SOURCE_DB, uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """
        select rs.origin, m.logical_source_key, m.raw_id, m.message_count,
               rs.source_path, rs.blob_hash, rs.capture_mode, rs.acquired_at_ms
        from raw_session_memberships m join raw_sessions rs on rs.raw_id = m.raw_id
        where m.decision = 'ambiguous'
        """
    )
    rows = cur.fetchall()
    cohorts: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        cohorts[(row["origin"], row["logical_source_key"])].append(row)

    # Filter to "equal message_count" cohorts (all members equal) -- matches
    # the prompt's pre-measured population.
    equal_cohorts: dict[str, list[tuple[str, list[sqlite3.Row]]]] = defaultdict(list)
    for (origin, key), members in cohorts.items():
        counts = {m["message_count"] for m in members}
        if len(counts) == 1:
            equal_cohorts[origin].append((key, members))

    print("Population (equal-message-count ambiguous cohorts):")
    for origin, items in sorted(equal_cohorts.items()):
        print(f"  {origin}: {len(items)}")
    print()

    # Sampling plan per origin.
    sample_sizes = {
        "claude-ai-export": 40,
        "chatgpt-export": 35,
        "hermes-session": 4,  # full pop
        "claude-code-session": 6,  # full pop (low priority, different shape)
        "gemini-cli-session": 0,  # pop is 0 in equal-count bucket per prompt
        "grok-export": 1,  # full pop
        "unknown-export": 3,  # full pop, opportunistic
    }

    results: dict[str, list[dict]] = defaultdict(list)

    for origin, items in equal_cohorts.items():
        n = sample_sizes.get(origin, min(30, len(items)))
        n = min(n, len(items))
        sample = items if n >= len(items) else random.sample(items, n)
        print(f"=== {origin}: sampling {len(sample)} of {len(items)} ===")
        for key, members in sample:
            try:
                result = analyze_cohort(origin, key, members)
            except Exception as exc:
                result = {"key": key, "error": f"{type(exc).__name__}: {exc}"}
            results[origin].append(result)

    out_path = Path(
        "/realm/project/polylogue/.claude/worktrees/agent-a9d80459cee6db3f3/.agent/scratch/ambiguous_cohort_census_raw.json"
    )
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote raw results to {out_path}")


def analyze_cohort(origin: str, key: str, members: list[sqlite3.Row]) -> dict:
    origin_enum = Origin(origin)
    # Group by blob_hash first: identical bytes parse identically.
    by_blob: dict[bytes, list[sqlite3.Row]] = defaultdict(list)
    for m in members:
        by_blob[m["blob_hash"]].append(m)

    revisions: list[MembershipRevision] = []
    parse_errors: list[str] = []
    blob_group_info = []
    for blob_hash, rows in by_blob.items():
        rep = rows[0]
        provider = provider_from_origin(origin_enum, family_hint=rep["capture_mode"])
        try:
            raw_bytes = load_payload(blob_hash)
        except FileNotFoundError:
            parse_errors.append(f"blob missing for raw_id={rep['raw_id']}")
            continue
        try:
            sessions = parse_raw(str(provider), raw_bytes, rep["source_path"])
        except Exception as exc:
            parse_errors.append(f"parse failed raw_id={rep['raw_id']}: {type(exc).__name__}: {exc}")
            continue
        matches = [s for s in sessions if f"{s.source_name.value}:{s.provider_session_id}" == key]
        if len(matches) != 1:
            parse_errors.append(
                f"raw_id={rep['raw_id']} matched {len(matches)} sessions for key={key} (parsed {len(sessions)} total)"
            )
            continue
        session = matches[0]
        projection = session_revision_projection(session)
        blob_group_info.append(
            {
                "blob_hash": blob_hash.hex()[:16],
                "raw_ids": [r["raw_id"][:16] for r in rows],
                "n_raws_with_this_blob": len(rows),
                "source_paths": sorted({r["source_path"].split(":")[0].split("/")[-1] for r in rows}),
                "n_messages": len(session.messages),
                "n_attachments": len(session.attachments),
                "attachments": [
                    {
                        "id": a.provider_attachment_id,
                        "msg_id": a.message_provider_id,
                        "name": a.name,
                        "mime": a.mime_type,
                        "size_bytes": a.size_bytes,
                        "has_inline_bytes": a.inline_bytes is not None,
                    }
                    for a in session.attachments
                ],
                "title": session.title,
                "updated_at": session.updated_at,
                "raw_byte_size": len(raw_bytes),
            }
        )
        revisions.append(
            MembershipRevision(
                rep["raw_id"],
                projection,
                session.updated_at,
                observed_at_ms=rep["acquired_at_ms"],
                browser_snapshot_fidelity=None,
                provider_message_ids=frozenset(
                    m2.provider_message_id for m2 in session.messages if m2.provider_message_id is not None
                ),
                provider_attachment_ids=frozenset(a.provider_attachment_id for a in session.attachments),
            )
        )

    result: dict = {
        "origin": origin,
        "key": key,
        "n_members_db": len(members),
        "n_distinct_blob_hashes": len(by_blob),
        "parse_errors": parse_errors,
        "blob_groups": blob_group_info,
    }

    if len(revisions) < 2:
        result["classification"] = "insufficient_parsed_revisions"
        return result

    # Pairwise comparisons across distinct-content representatives.
    pair_comparisons = []
    for i in range(len(revisions)):
        for j in range(len(revisions)):
            if i == j:
                continue
            a, b = revisions[i], revisions[j]
            pa, pb = a.projection, b.projection
            pair_comparisons.append(
                {
                    "older_raw": a.raw_id[:16],
                    "newer_raw": b.raw_id[:16],
                    "message_hashes_equal": pa.message_hashes == pb.message_hashes,
                    "event_hashes_equal": pa.event_hashes == pb.event_hashes,
                    "attachment_hashes_equal": pa.attachment_hashes == pb.attachment_hashes,
                    "attachment_set_sizes": (len(pa.attachment_hashes), len(pb.attachment_hashes)),
                    "attachment_intersection": len(pa.attachment_hashes & pb.attachment_hashes),
                    "attachment_subset_a_le_b": pa.attachment_hashes <= pb.attachment_hashes,
                    "frontier_a": _frontier(pa),
                    "frontier_b": _frontier(pb),
                    "strictly_dominates_a_over_b": _strictly_dominates(pa, pb),
                }
            )
    result["pair_comparisons"] = pair_comparisons

    classification = classify_membership_revisions(revisions)
    result["classification"] = {
        "accepted": [r[:16] for r in classification.accepted_raw_ids],
        "equivalent": [r[:16] for r in classification.equivalent_raw_ids],
        "ambiguous": [r[:16] for r in classification.ambiguous_raw_ids],
    }
    return result


if __name__ == "__main__":
    main()
