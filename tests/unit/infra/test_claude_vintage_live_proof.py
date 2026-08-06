"""Production-route proof for the sanitized Claude export-vintage cohort."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.archive.session_revision_membership import MembershipRevision, classify_membership_revisions
from polylogue.maintenance.pathology_zoo import PATHOLOGY_ZOO_MANIFEST
from polylogue.pipeline import ids
from polylogue.pipeline.ids import session_revision_projection
from polylogue.sources.parsers.base import ParsedMessage
from polylogue.sources.parsers.claude.ai_parser import parse_ai
from tests.infra.archive_canonical_snapshot import capture_canonical_snapshot
from tests.infra.claude_vintage_live_proof import (
    CONFIDENCE_GAP,
    run_claude_vintage_live_proof,
)
from tests.infra.pathology_zoo import (
    CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID,
    _claude_vintage_live_proof_payload,
)


def test_sanitized_pair_runs_the_real_route_and_emits_read_only_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = run_claude_vintage_live_proof(tmp_path / "archive")

    assert receipt.live_export_recovered is False
    assert receipt.confidence_gap == CONFIDENCE_GAP
    assert receipt.verdict == "equivalent"
    assert receipt.canonical_identity == f"claude-ai-export:{CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID}"
    assert receipt.canonical_content_hash and len(receipt.canonical_content_hash) == 64
    assert dict(receipt.parser_branch)["old_parsed_block_count"] == 0
    assert dict(receipt.parser_branch)["new_parsed_block_count"] == 1
    assert dict(receipt.parser_branch)["projection_hash_equal"] is True
    assert dict(receipt.classifier_probe) == {
        "accepted_raw_ids": ("fixture-new",),
        "equivalent_raw_ids": ("fixture-old",),
        "ambiguous_raw_ids": (),
    }
    assert dict(receipt.route_counts) == {
        "ingest_sessions": 1,
        "ingest_messages": 3,
        "backfill_scanned": 2,
        "backfill_classified_full": 2,
        "backfill_replayed_logical_sources": 1,
        "backfill_quarantined": 0,
    }
    assert receipt.convergence_session_count == 1

    rendered = receipt.as_json()
    assert json.loads(rendered)["live_export_recovered"] is False
    assert "raw_sessions" not in rendered
    print(rendered)
    assert json.loads(capsys.readouterr().out)["verdict"] == "equivalent"

    snapshot = capture_canonical_snapshot(
        tmp_path / "archive",
        session_ids=(receipt.canonical_identity,),
    )
    sessions = next(relation for relation in snapshot.canonical_rows if relation.relation == "sessions")
    assert sessions.rows
    assert any(key == f"summary:{receipt.canonical_identity}" for key, _value in snapshot.public_projections)


def test_red_mutation_restores_the_vintage_conflict_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    old_session = parse_ai(_claude_vintage_live_proof_payload(nested_target=False), "old-fallback")
    new_session = parse_ai(_claude_vintage_live_proof_payload(nested_target=True), "new-fallback")
    current_message_hash_payload = ids._message_hash_payload

    def legacy_message_hash_payload(message: ParsedMessage, message_id: str) -> dict[str, object]:
        payload: dict[str, object] = dict(current_message_hash_payload(message, message_id))
        if message.blocks:
            payload["content_blocks"] = [ids._content_block_payload(block) for block in message.blocks]
        return payload

    monkeypatch.setattr(ids, "_message_hash_payload", legacy_message_hash_payload)
    old_projection = session_revision_projection(old_session)
    new_projection = session_revision_projection(new_session)
    assert old_projection.session_hash != new_projection.session_hash

    result = classify_membership_revisions(
        [
            MembershipRevision("fixture-old", old_projection),
            MembershipRevision("fixture-new", new_projection),
        ]
    )
    assert result.accepted_raw_ids == ("fixture-old",)
    assert result.equivalent_raw_ids == ()
    assert result.ambiguous_raw_ids == ("fixture-new",)


def test_registry_records_the_unrecovered_live_evidence_gap() -> None:
    member = next(item for item in PATHOLOGY_ZOO_MANIFEST if item.member_id == "claude-vintage-live-proof")
    assert member.session_ids == (f"claude-ai-export:{CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID}",)
    assert member.evidence_note is not None
    assert "not recoverable" in member.evidence_note
