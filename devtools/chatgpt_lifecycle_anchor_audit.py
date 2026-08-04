"""Read-only ChatGPT lifecycle-anchor census through the production parser route.

This command audits whether quarantined ChatGPT revisions currently exhibit
the historical mapping-order failure: two exports with equal transcript and
lifecycle content but a different generation-lifecycle anchor.  It does not
change archive state.  The only optional write is a caller-selected,
sanitized JSON receipt outside the archive.
"""

from __future__ import annotations

import argparse
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
from polylogue.pipeline.ids import session_revision_projection
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.revision_backfill import _parse_one
from polylogue.storage.blob_store import BlobStore

_Relation = Literal["equal", "a_contains_b", "b_contains_a", "conflict"]

SCHEMA = "polylogue.chatgpt-lifecycle-anchor-audit.v1"
TARGET_PREDICATE = (
    "A pair in one persisted logical_source_key cohort where each parsed session has exactly one "
    "generation_lifecycle event, their source_message_provider_id anchors differ, message_contents and "
    "attachment_contents are equal, non-anchor lifecycle content hashes are equal, and the production _relation is conflict."
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


def _git_revision() -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _matches_target(left: _ParsedMember, right: _ParsedMember, relation: _Relation) -> bool:
    if len(left.session.session_events) != 1 or len(right.session.session_events) != 1:
        return False
    left_event, right_event = left.session.session_events[0], right.session.session_events[0]
    left_projection = left.revision.projection
    right_projection = right.revision.projection
    return (
        left_event.event_type == right_event.event_type == "generation_lifecycle"
        and left_event.source_message_provider_id != right_event.source_message_provider_id
        and left_projection.message_contents == right_projection.message_contents
        and left_projection.attachment_contents == right_projection.attachment_contents
        and left_projection.event_contents != right_projection.event_contents
        and {content for _, content in left_projection.event_contents}
        == {content for _, content in right_projection.event_contents}
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
                "producer_git_revision": _git_revision(),
                "production_route": [
                    "polylogue.sources.revision_backfill._parse_one",
                    "polylogue.pipeline.ids.session_revision_projection",
                    "polylogue.archive.session_revision_membership._relation",
                    "polylogue.archive.session_revision_membership.classify_membership_revisions",
                ],
                "source_db": _database_provenance(source_conn, source_db),
                "index_db": _database_provenance(index_conn, index_db),
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
    receipt = run_audit(args.archive_root)
    if args.receipt is not None:
        _write_receipt(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), file=stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
