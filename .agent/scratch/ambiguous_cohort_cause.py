"""Bucket every sampled ambiguous cohort into a minimal-delta cause.

Read-only against /realm/db/polylogue. Never stores message text, titles, or
attachment names in output -- aggregate/structural fields only.
"""

from __future__ import annotations

import io
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/realm/project/polylogue/.claude/worktrees/agent-a9d80459cee6db3f3")

from polylogue.archive.session_revision_membership import (
    MembershipRevision,
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


def cause_for_pair(sess_a, sess_b) -> tuple[str, dict]:
    """Classify the minimal delta between two ParsedSession revisions of the
    same logical conversation. Returns (cause_label, evidence_dict) where
    evidence_dict holds only counts/booleans, never content."""
    ids_a = [m.provider_message_id for m in sess_a.messages]
    ids_b = [m.provider_message_id for m in sess_b.messages]
    set_a, set_b = set(ids_a), set(ids_b)
    has_none_id = None in set_a or None in set_b

    atts_a = {(at.provider_attachment_id, at.message_provider_id): at for at in sess_a.attachments}
    atts_b = {(at.provider_attachment_id, at.message_provider_id): at for at in sess_b.attachments}
    att_key_set_equal = set(atts_a) == set(atts_b)
    att_bytes_delta = False
    if att_key_set_equal and atts_a:
        for k in atts_a:
            aa, bb = atts_a[k], atts_b[k]
            if (aa.inline_bytes is not None) != (bb.inline_bytes is not None):
                att_bytes_delta = True
            if aa.size_bytes != bb.size_bytes and (aa.size_bytes is None or bb.size_bytes is None):
                att_bytes_delta = True

    evidence = {
        "n_messages_a": len(ids_a),
        "n_messages_b": len(ids_b),
        "message_id_set_equal": (not has_none_id) and set_a == set_b,
        "message_order_equal": ids_a == ids_b,
        "n_attachments_a": len(atts_a),
        "n_attachments_b": len(atts_b),
        "attachment_key_set_equal": att_key_set_equal,
        "attachment_acquisition_delta": att_bytes_delta,
    }

    if not has_none_id and set_a == set_b and ids_a != ids_b:
        # Same message set, different sequence -- check whether per-id content
        # (role/text/timestamp) is identical; if so it's a pure reorder.
        map_a = {m.provider_message_id: (m.role, m.text, m.timestamp) for m in sess_a.messages}
        map_b = {m.provider_message_id: (m.role, m.text, m.timestamp) for m in sess_b.messages}
        n_content_diffs = sum(1 for k in map_a if map_a[k] != map_b.get(k))
        evidence["n_content_diffs_among_shared_ids"] = n_content_diffs
        if n_content_diffs == 0:
            return "message_order_nondeterminism", evidence
        return "same_ids_edited_content", evidence

    if att_bytes_delta and set_a == set_b and ids_a == ids_b:
        return "attachment_acquisition_state_in_identity_hash", evidence

    if len(ids_a) == len(ids_b) and set_a != set_b:
        return "different_message_id_sets_same_count", evidence

    return "other_or_genuine_divergence", evidence


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

    equal_cohorts: dict[str, list[tuple[str, list[sqlite3.Row]]]] = defaultdict(list)
    for (origin, key), members in cohorts.items():
        counts = {m["message_count"] for m in members}
        if len(counts) == 1:
            equal_cohorts[origin].append((key, members))

    sample_sizes = {
        "aistudio-drive": 30,
        "claude-ai-export": 40,
        "chatgpt-export": 35,
        "hermes-session": 4,
        "claude-code-session": 6,
        "gemini-cli-session": 0,
        "grok-export": 1,
        "unknown-export": 3,
    }

    bucket_counts: dict[str, Counter] = defaultdict(Counter)
    cohort_details: dict[str, list[dict]] = defaultdict(list)
    coverage: dict[str, dict] = {}

    for origin, items in equal_cohorts.items():
        n = min(sample_sizes.get(origin, min(30, len(items))), len(items))
        sample = items if n >= len(items) else random.sample(items, n)
        coverage[origin] = {"population": len(items), "sampled": len(sample)}
        for key, members in sample:
            try:
                cause, detail = classify_one_cohort(origin, key, members)
            except Exception as exc:
                cause = f"ERROR:{type(exc).__name__}"
                detail = {"error": str(exc)}
            bucket_counts[origin][cause] += 1
            cohort_details[origin].append({"cause": cause, "detail": detail})

    print(json.dumps({"coverage": coverage, "bucket_counts": {o: dict(c) for o, c in bucket_counts.items()}}, indent=2))

    out_path = Path(
        "/realm/project/polylogue/.claude/worktrees/agent-a9d80459cee6db3f3/.agent/scratch/ambiguous_cohort_causes.json"
    )
    out_path.write_text(
        json.dumps(
            {
                "coverage": coverage,
                "bucket_counts": {o: dict(c) for o, c in bucket_counts.items()},
                "cohort_details": cohort_details,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nWrote {out_path}")


def classify_one_cohort(origin: str, key: str, members: list[sqlite3.Row]) -> tuple[str, dict]:
    origin_enum = Origin(origin)
    by_blob: dict[bytes, sqlite3.Row] = {}
    for m in members:
        by_blob.setdefault(m["blob_hash"], m)

    parsed_sessions = []
    for blob_hash, rep in by_blob.items():
        provider = provider_from_origin(origin_enum, family_hint=rep["capture_mode"])
        raw_bytes = load_payload(blob_hash)
        sessions = parse_raw(str(provider), raw_bytes, rep["source_path"])
        matches = [s for s in sessions if f"{s.source_name.value}:{s.provider_session_id}" == key]
        if len(matches) != 1:
            return "PARSE_MISMATCH", {"n_matches": len(matches), "n_sessions": len(sessions)}
        parsed_sessions.append(matches[0])

    if len(parsed_sessions) < 2:
        return "single_distinct_content_group", {}

    # Confirm still classifies ambiguous with the production classifier
    # (sanity: the DB decision was recorded at a potentially earlier index
    # generation, so re-verify against current parser behavior).
    revisions = []
    for i, sess in enumerate(parsed_sessions):
        proj = session_revision_projection(sess)
        revisions.append(
            MembershipRevision(
                f"sample-{i}",
                proj,
                sess.updated_at,
                observed_at_ms=None,
                browser_snapshot_fidelity=None,
                provider_message_ids=frozenset(
                    m2.provider_message_id for m2 in sess.messages if m2.provider_message_id is not None
                ),
                provider_attachment_ids=frozenset(a.provider_attachment_id for a in sess.attachments),
            )
        )
    classification = classify_membership_revisions(revisions)
    still_ambiguous = len(classification.ambiguous_raw_ids) > 0

    # Pairwise cause across all distinct-content pairs; report the dominant
    # (most common) pairwise cause for this cohort, plus whether any pair
    # differs.
    causes = []
    evidence_all = []
    for i in range(len(parsed_sessions)):
        for j in range(i + 1, len(parsed_sessions)):
            cause, evidence = cause_for_pair(parsed_sessions[i], parsed_sessions[j])
            causes.append(cause)
            evidence_all.append(evidence)
    cause_counter = Counter(causes)
    dominant_cause = cause_counter.most_common(1)[0][0]
    return dominant_cause, {
        "n_distinct_content_groups": len(parsed_sessions),
        "still_ambiguous_under_current_classifier": still_ambiguous,
        "pairwise_causes": dict(cause_counter),
        "sample_evidence": evidence_all[0] if evidence_all else {},
    }


if __name__ == "__main__":
    main()
