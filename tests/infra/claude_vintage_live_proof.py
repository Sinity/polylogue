"""Measured-shape Claude vintage proof through the production archive route.

The parent measurement established the wire-shape difference, but its live
export bytes were not recoverable for this lane. This harness therefore keeps
the pair private-data-free, records that confidence gap in every receipt, and
still exercises the real parser, archive ingest, membership replay, and
convergence seams.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from polylogue.archive.session_revision_membership import MembershipRevision, classify_membership_revisions
from polylogue.config import Source
from polylogue.core.enums import Origin
from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_fts_stage, make_insights_stage
from polylogue.pipeline.ids import session_revision_projection
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.sources.parsers.claude.ai_parser import parse_ai
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.pathology_zoo import (
    CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID,
    write_claude_vintage_live_proof_pair,
)

ReceiptVerdict = Literal["equivalent", "conflict", "unresolved"]

CONFIDENCE_GAP = (
    "The cited live Claude export bytes and cohort member ids were not recoverable "
    "outside the protected live archive. This receipt proves the measured old-flat "
    "versus new-nested wire shape and the production route, but it does not establish "
    "the parent cohort's live prevalence or exact member set."
)


@dataclass(frozen=True, slots=True)
class ClaudeVintageReclassificationReceipt:
    """Sanitized, read-only evidence for one cohort reclassification."""

    schema_version: int
    fixture_member_id: str
    live_export_recovered: bool
    confidence_gap: str
    production_route: tuple[str, ...]
    parser_branch: tuple[tuple[str, object], ...]
    classifier_probe: tuple[tuple[str, tuple[str, ...]], ...]
    canonical_identity: str
    canonical_content_hash: str
    verdict: ReceiptVerdict
    route_counts: tuple[tuple[str, int], ...]
    convergence_session_count: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe receipt without raw paths or raw identifiers."""
        return {
            "schema_version": self.schema_version,
            "fixture_member_id": self.fixture_member_id,
            "live_export_recovered": self.live_export_recovered,
            "confidence_gap": self.confidence_gap,
            "production_route": list(self.production_route),
            "parser_branch": dict(self.parser_branch),
            "classifier_probe": {key: list(value) for key, value in self.classifier_probe},
            "canonical_identity": self.canonical_identity,
            "canonical_content_hash": self.canonical_content_hash,
            "verdict": self.verdict,
            "route_counts": dict(self.route_counts),
            "convergence_session_count": self.convergence_session_count,
        }

    def as_json(self) -> str:
        """Render the receipt deterministically for evidence capture."""
        return json.dumps(self.as_dict(), sort_keys=True, indent=2) + "\n"


def run_claude_vintage_live_proof(archive_root: Path) -> ClaudeVintageReclassificationReceipt:
    """Run the sanitized pair through production parse, ingest, replay, and convergence."""
    wire_root = archive_root / "wire"
    old_path, new_path = write_claude_vintage_live_proof_pair(wire_root)
    initialize_active_archive_root(archive_root)

    ingest = asyncio.run(
        parse_sources_archive(
            archive_root,
            [Source(name="claude-ai", path=wire_root)],
            parse_workers=1,
        )
    )
    backfill = backfill_historical_revision_evidence(archive_root, ingest_workers=1)

    with sqlite3.connect(archive_root / "index.db") as connection:
        session_ids = tuple(
            str(row[0]) for row in connection.execute("SELECT session_id FROM sessions ORDER BY session_id")
        )
    converger = DaemonConverger(
        (
            make_fts_stage(archive_root / "index.db"),
            make_insights_stage(archive_root / "index.db"),
        )
    )
    states, _timings = converger.converge_sessions(session_ids)
    if any(not state.converged for state in states.values()):
        pending = {session_id: state.last_error for session_id, state in states.items() if not state.converged}
        raise AssertionError(f"Claude vintage proof did not converge: {pending}")

    old_payload = json.loads(old_path.read_text(encoding="utf-8"))
    new_payload = json.loads(new_path.read_text(encoding="utf-8"))
    old_session = parse_ai(old_payload, "synthetic-fallback-old")
    new_session = parse_ai(new_payload, "synthetic-fallback-new")
    old_projection = session_revision_projection(old_session)
    new_projection = session_revision_projection(new_session)
    probe = classify_membership_revisions(
        [
            MembershipRevision("fixture-old", old_projection),
            MembershipRevision("fixture-new", new_projection),
        ]
    )

    rows = _read_membership_rows(archive_root)
    if len(rows) != 2:
        raise AssertionError(f"expected two sanitized cohort membership rows, got {len(rows)}")
    hashes = {row["content_hash"] for row in rows}
    if len(hashes) != 1:
        raise AssertionError(f"production membership hashes diverged: {hashes}")
    content_hash = next(iter(hashes))
    if not isinstance(content_hash, str):
        raise AssertionError(f"production membership hash is not text: {content_hash!r}")
    decisions = {str(row["decision"]) for row in rows}
    if decisions != {"applied", "superseded_equivalent"}:
        raise AssertionError(f"unexpected production membership decisions: {decisions}")
    if len(session_ids) != 1:
        raise AssertionError(f"expected one canonical indexed identity, got {session_ids}")

    route_counts = (
        ("ingest_sessions", int(ingest.counts["sessions"])),
        ("ingest_messages", int(ingest.counts["messages"])),
        ("backfill_scanned", int(backfill.scanned)),
        ("backfill_classified_full", int(backfill.classified_full)),
        ("backfill_replayed_logical_sources", int(backfill.replayed_logical_sources)),
        ("backfill_quarantined", int(backfill.quarantined)),
    )
    return ClaudeVintageReclassificationReceipt(
        schema_version=1,
        fixture_member_id="claude-vintage-live-proof",
        live_export_recovered=False,
        confidence_gap=CONFIDENCE_GAP,
        production_route=(
            "parse_sources_archive",
            "backfill_historical_revision_evidence",
            "DaemonConverger(make_fts_stage, make_insights_stage)",
        ),
        parser_branch=(
            ("old_wire_shape", "top_level_text"),
            ("new_wire_shape", "content_text_segment"),
            ("old_parsed_block_count", len(old_session.messages[2].blocks)),
            ("new_parsed_block_count", len(new_session.messages[2].blocks)),
            ("projection_hash_equal", old_projection.session_hash == new_projection.session_hash),
        ),
        classifier_probe=(
            ("accepted_raw_ids", tuple(probe.accepted_raw_ids)),
            ("equivalent_raw_ids", tuple(probe.equivalent_raw_ids)),
            ("ambiguous_raw_ids", tuple(probe.ambiguous_raw_ids)),
        ),
        canonical_identity=f"{Origin.CLAUDE_AI_EXPORT.value}:{CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID}",
        canonical_content_hash=content_hash,
        verdict="equivalent",
        route_counts=route_counts,
        convergence_session_count=len(states),
    )


def _read_membership_rows(archive_root: Path) -> tuple[dict[str, object], ...]:
    """Read the production membership verdict through read-only SQLite handles."""
    source_path = archive_root / "source.db"
    with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            """
            SELECT r.source_path, m.normalized_content_hash, m.decision
            FROM raw_sessions AS r
            JOIN raw_session_memberships AS m ON m.raw_id = r.raw_id
            WHERE m.logical_source_key = ?
            ORDER BY r.source_path
            """,
            (f"claude-ai:{CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID}",),
        ).fetchall()
    return tuple(
        {
            "label": Path(str(source_path)).stem.rsplit("-", 1)[-1],
            "content_hash": bytes(content_hash).hex(),
            "decision": str(decision),
        }
        for source_path, content_hash, decision in rows
    )


__all__ = [
    "CONFIDENCE_GAP",
    "ClaudeVintageReclassificationReceipt",
    "run_claude_vintage_live_proof",
]
