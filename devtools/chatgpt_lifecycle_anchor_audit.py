"""Read-only ChatGPT lifecycle-anchor census through the production parser route.

This command audits whether quarantined ChatGPT revisions currently exhibit
the historical mapping-order failure: two exports with equal transcript and
lifecycle content but a different generation-lifecycle anchor.  It does not
change archive state.  The only optional write is a caller-selected,
sanitized JSON receipt outside the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

from polylogue.archive.session_revision_membership import MembershipRevision, _relation, classify_membership_revisions
from polylogue.core.enums import Provider
from polylogue.core.hashing import hash_payload
from polylogue.pipeline.ids import _event_content_payload, session_revision_projection
from polylogue.sources.parsers.base import ParsedSession, ParsedSessionEvent
from polylogue.sources.revision_backfill import _parse_one
from polylogue.storage.blob_store import BlobStore

_Relation = Literal["equal", "a_contains_b", "b_contains_a", "conflict"]

SCHEMA = "polylogue.chatgpt-lifecycle-anchor-audit.v2"
TARGET_PREDICATE = (
    "A pair in one persisted logical_source_key cohort where each parsed session has exactly one "
    "generation_lifecycle event (other session events are allowed), their source_message_provider_id "
    "anchors differ, message_contents and attachment_contents are equal, generation_lifecycle event "
    "content hashes after removing source_message_provider_id are equal, all normalized event content "
    "is equal after that same lifecycle-only exception, and the production _relation is conflict."
)
SELECTION_SQL = """
SELECT r.raw_id, r.source_path, lower(hex(r.blob_hash)) AS blob_hash,
       m.logical_source_key, m.provider_session_id
FROM raw_sessions AS r
JOIN raw_session_memberships AS m ON m.raw_id = r.raw_id
WHERE r.origin = 'chatgpt-export' AND r.revision_authority = 'quarantined'
ORDER BY m.logical_source_key, r.raw_id
""".strip()
POPULATION_SQL = """
SELECT raw_id
FROM raw_sessions
WHERE origin = 'chatgpt-export' AND revision_authority = 'quarantined'
ORDER BY raw_id
""".strip()


@dataclass(frozen=True, slots=True)
class _RawMember:
    raw_id: str
    source_path: str
    blob_hash: str
    logical_source_key: str
    provider_session_id: str


@dataclass(frozen=True, slots=True)
class _ParsedMember:
    revision: MembershipRevision
    session: ParsedSession


def _connect_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _database_provenance(conn: sqlite3.Connection, path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sqlite_schema_version": int(conn.execute("PRAGMA schema_version").fetchone()[0]),
        "sqlite_user_version": int(conn.execute("PRAGMA user_version").fetchone()[0]),
    }


def _git_provenance() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "-C", os.fspath(repo_root), "rev-parse", "--verify", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        status = subprocess.run(
            ["git", "-C", os.fspath(repo_root), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("ChatGPT lifecycle-anchor audit requires a readable git producer checkout") from error
    return {
        "git_revision": revision,
        "working_tree_clean": not bool(status),
        "working_tree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _generation_events(session: ParsedSession) -> list[ParsedSessionEvent]:
    return [event for event in session.session_events if event.event_type == "generation_lifecycle"]


def _anchor_independent_event_content(event: ParsedSessionEvent) -> bytes:
    """Hash one lifecycle event without its provider-message anchor."""
    payload = _event_content_payload(event)
    payload.pop("source_message_provider_id", None)
    return bytes.fromhex(hash_payload(payload))


def _event_content_signature(event: ParsedSessionEvent) -> bytes:
    """Hash normalized event content, retaining anchors except for lifecycle events."""
    if event.event_type == "generation_lifecycle":
        return _anchor_independent_event_content(event)
    return bytes.fromhex(hash_payload(_event_content_payload(event)))


def _session_event_content_signatures(session: ParsedSession) -> Counter[bytes]:
    """Return normalized event content as a multiset, independent of array order."""
    return Counter(_event_content_signature(event) for event in session.session_events)


def _blob_store_snapshot(blob_store: BlobStore) -> dict[str, object]:
    """Capture a deterministic, read-only identity and integrity scan of blobs."""
    snapshot_digest = hashlib.sha256()
    integrity_digest = hashlib.sha256()
    canonical_blob_count = 0
    canonical_blob_bytes = 0
    verified_blob_count = 0
    hash_mismatch_count = 0
    invalid_namespace_entry_count = 0

    for entry in blob_store.iter_namespace():
        if entry.hash_hex is None:
            invalid_namespace_entry_count += 1
            record = {
                "kind": entry.kind.value,
                "issue": entry.issue.value if entry.issue is not None else None,
                "relative_path": entry.relative_path,
            }
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
            snapshot_digest.update(encoded)
            snapshot_digest.update(b"\n")
            integrity_digest.update(encoded)
            integrity_digest.update(b"\n")
            continue

        size_bytes = entry.path.stat().st_size
        actual_digest = hashlib.sha256()
        with entry.path.open("rb") as blob:
            while chunk := blob.read(1024 * 1024):
                actual_digest.update(chunk)
        verified = actual_digest.hexdigest() == entry.hash_hex
        canonical_blob_count += 1
        canonical_blob_bytes += size_bytes
        verified_blob_count += int(verified)
        hash_mismatch_count += int(not verified)
        snapshot_record = {"hash": entry.hash_hex, "size_bytes": size_bytes}
        integrity_record = {
            **snapshot_record,
            "verified": verified,
            "observed_sha256": actual_digest.hexdigest(),
        }
        snapshot_encoded = json.dumps(snapshot_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        integrity_encoded = json.dumps(integrity_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        snapshot_digest.update(snapshot_encoded)
        snapshot_digest.update(b"\n")
        integrity_digest.update(integrity_encoded)
        integrity_digest.update(b"\n")

    return {
        "snapshot_sha256": snapshot_digest.hexdigest(),
        "canonical_blob_count": canonical_blob_count,
        "canonical_blob_bytes": canonical_blob_bytes,
        "integrity": {
            "scan": "full_read_only_namespace_and_content_hash",
            "verified_blob_count": verified_blob_count,
            "hash_mismatch_count": hash_mismatch_count,
            "invalid_namespace_entry_count": invalid_namespace_entry_count,
            "integrity_sha256": integrity_digest.hexdigest(),
        },
    }


def _matches_target(left: _ParsedMember, right: _ParsedMember, relation: _Relation) -> bool:
    left_generation_events = _generation_events(left.session)
    right_generation_events = _generation_events(right.session)
    if len(left_generation_events) != 1 or len(right_generation_events) != 1:
        return False
    left_event, right_event = left_generation_events[0], right_generation_events[0]
    left_projection = left.revision.projection
    right_projection = right.revision.projection
    return (
        left_event.source_message_provider_id != right_event.source_message_provider_id
        and left_projection.message_contents == right_projection.message_contents
        and left_projection.attachment_contents == right_projection.attachment_contents
        and _session_event_content_signatures(left.session) == _session_event_content_signatures(right.session)
        and _anchor_independent_event_content(left_event) == _anchor_independent_event_content(right_event)
        and relation == "conflict"
    )


def _load_existing_heads(index_conn: sqlite3.Connection) -> dict[str, str]:
    return {
        str(row[0]): str(row[1])
        for row in index_conn.execute("SELECT logical_source_key, accepted_raw_id FROM raw_revision_heads")
    }


def _parse_member(member: _RawMember, blob_store: BlobStore, archive_root: Path) -> _ParsedMember:
    sessions = _parse_one(
        Provider.CHATGPT,
        blob_store.read_all(member.blob_hash),
        member.source_path,
        archive_root=archive_root,
        fallback_id_override=member.provider_session_id,
    )
    matches = [session for session in sessions if session.provider_session_id == member.provider_session_id]
    if len(matches) != 1:
        raise RuntimeError(
            "ChatGPT lifecycle-anchor audit expected one parsed session for a persisted membership row, "
            f"got {len(matches)}"
        )
    session = matches[0]
    return _ParsedMember(MembershipRevision(member.raw_id, session_revision_projection(session)), session)


def _cohorts(rows: Iterable[_RawMember]) -> dict[str, list[_RawMember]]:
    grouped: dict[str, list[_RawMember]] = defaultdict(list)
    for row in rows:
        grouped[row.logical_source_key].append(row)
    return dict(grouped)


def run_audit(archive_root: Path) -> dict[str, object]:
    """Run the full current-corpus census without opening an archive writer."""
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    blob_store = BlobStore(archive_root / "blob")
    source_conn = _connect_read_only(source_db)
    index_conn = _connect_read_only(index_db)
    try:
        population_raw_ids = {str(row[0]) for row in source_conn.execute(POPULATION_SQL)}
        rows = [_RawMember(*map(str, row)) for row in source_conn.execute(SELECTION_SQL)]
        rows_by_raw_id: dict[str, list[_RawMember]] = defaultdict(list)
        for row in rows:
            rows_by_raw_id[row.raw_id].append(row)
        duplicated_membership_raw_count = sum(1 for members in rows_by_raw_id.values() if len(members) != 1)
        if duplicated_membership_raw_count:
            raise RuntimeError("ChatGPT lifecycle-anchor audit requires exactly one membership row per selected raw")
        cohorts = _cohorts(rows)
        relation_counts: Counter[str] = Counter()
        classifier_counts: Counter[str] = Counter()
        target_pair_count = 0
        parsed_raw_count = 0
        heads = _load_existing_heads(index_conn)
        blob_snapshot = _blob_store_snapshot(blob_store)
        producer = _git_provenance()
        for logical_source_key in sorted(cohorts):
            revisions = [
                _parse_member(member, blob_store, archive_root)
                for member in sorted(cohorts[logical_source_key], key=lambda member: member.raw_id)
            ]
            parsed_raw_count += len(revisions)
            for index, left in enumerate(revisions):
                for right in revisions[index + 1 :]:
                    relation = _relation(left.revision.projection, right.revision.projection)
                    relation_counts[relation] += 1
                    if _matches_target(left, right, relation):
                        target_pair_count += 1
            classification = classify_membership_revisions(
                [revision.revision for revision in revisions], existing_accepted_raw_id=heads.get(logical_source_key)
            )
            classifier_counts["cohorts_with_accepted_raw"] += bool(classification.accepted_raw_ids)
            classifier_counts["cohorts_with_equivalent_raw"] += bool(classification.equivalent_raw_ids)
            classifier_counts["cohorts_with_ambiguous_raw"] += bool(classification.ambiguous_raw_ids)
        cohort_sizes = Counter(len(members) for members in cohorts.values())
        return {
            "schema": SCHEMA,
            "provenance": {
                "archive_access": "SQLite source.db and index.db opened mode=ro; blob files read only; no archive writer created.",
                "producer_git_revision": producer["git_revision"],
                "producer_working_tree_clean": producer["working_tree_clean"],
                "producer_working_tree_status_sha256": producer["working_tree_status_sha256"],
                "production_route": [
                    "polylogue.sources.revision_backfill._parse_one",
                    "polylogue.pipeline.ids.session_revision_projection",
                    "polylogue.archive.session_revision_membership._relation",
                    "polylogue.archive.session_revision_membership.classify_membership_revisions",
                ],
                "source_db": _database_provenance(source_conn, source_db),
                "index_db": _database_provenance(index_conn, index_db),
                "blob_store": blob_snapshot,
            },
            "selection": {"sql": SELECTION_SQL, "population_sql": POPULATION_SQL},
            "target_predicate": TARGET_PREDICATE,
            "denominators": {
                "selected_quarantined_chatgpt_raw_count": len(population_raw_ids),
                "selected_membership_row_count": len(rows),
                "membershipless_selected_raw_count": len(population_raw_ids - set(rows_by_raw_id)),
                "logical_source_key_count": len(cohorts),
                "singleton_cohort_count": cohort_sizes[1],
                "multi_candidate_cohort_count": sum(count for size, count in cohort_sizes.items() if size > 1),
                "raws_in_multi_candidate_cohorts": sum(
                    size * count for size, count in cohort_sizes.items() if size > 1
                ),
                "parsed_and_projected_raw_count": parsed_raw_count,
            },
            "outcomes": {
                "pair_relation_counts": {
                    name: relation_counts[name] for name in ("equal", "a_contains_b", "b_contains_a", "conflict")
                },
                "target_pair_count": target_pair_count,
                "classifier_cohort_counts": dict(sorted(classifier_counts.items())),
            },
            "scope": {
                "sanitized": "No raw ids, native ids, source paths, blob hashes, titles, or payload content are emitted.",
                "conclusion_limit": (
                    "A zero target_pair_count describes only this current parser-and-corpus snapshot. It does not establish "
                    "the historical pre-fix replay required to reclassify or remove any graph gate."
                ),
            },
        }
    finally:
        index_conn.close()
        source_conn.close()


def _write_receipt(path: Path, receipt: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True, help="Archive root to inspect without mutation.")
    parser.add_argument("--receipt", type=Path, help="Optional worktree-local path for the sanitized JSON receipt.")
    args = parser.parse_args(argv)
    archive_root = args.archive_root.resolve()
    if args.receipt is not None:
        receipt_path = args.receipt.resolve()
        try:
            receipt_path.relative_to(archive_root)
        except ValueError:
            pass
        else:
            parser.error("--receipt must resolve outside --archive-root")
    receipt = run_audit(archive_root)
    if args.receipt is not None:
        _write_receipt(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), file=stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
