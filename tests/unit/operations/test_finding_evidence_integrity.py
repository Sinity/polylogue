"""Finding support is the shared evaluator's verdict, not an advisory string.

polylogue-rxdo.4. ``core/evidence_integrity.py`` declared
``FindingEvidenceAdapter`` and had no production caller; finding ref
resolution attached a staleness string as a caveat and reported success
regardless; no circular-ancestry detection existed anywhere in the finding or
assertion code. These tests pin the wiring and the three verdicts that were
previously unreachable.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.evidence_integrity import EvidenceIntegrityStatus
from polylogue.operations.finding_evidence import (
    build_finding_evidence_adapter,
    evaluate_finding_evidence,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    FindingAssertion,
    upsert_findings_as_assertions,
)
from polylogue.storage.sqlite.finding_provenance import compute_finding_provenance
from polylogue.storage.sqlite.query_objects import put_query, put_result_set


def _user_db(tmp_path: Path) -> Path:
    user_db = tmp_path / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    return user_db


def _grounded_finding(
    conn: sqlite3.Connection,
    *,
    claim_key: str = "evidence-integrity-claim",
    evidence_refs: tuple[str, ...] = (),
    detector_ref: str = "agent:integrity-detector",
) -> str:
    query = put_query(
        conn,
        {"field": "origin", "value": "codex-session"},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        created_at_ms=1,
    )
    result_set = put_result_set(
        conn,
        result_set_id=f"{claim_key}-rs",
        query_hash=query.query_hash,
        grain="session",
        corpus_epoch="index:g1",
        member_refs=("session:codex-session:one",),
        exactness="exact",
        persistence_class="finding",
        created_at_ms=1,
    )
    envelopes = upsert_findings_as_assertions(
        conn,
        [
            FindingAssertion(
                claim_key=claim_key,
                target_ref=f"query:{query.query_hash}",
                body_text="One session matched.",
                finding_kind="measure",
                statistic={"op": "count", "value": 1, "unit": "members"},
                n=1,
                query_ref=f"query:{query.query_hash}",
                result_set_ref=f"result-set:{result_set.result_set_id}",
                detector_ref=detector_ref,
                evidence_refs=evidence_refs,
            )
        ],
        now_ms=1,
    )
    return envelopes[0].assertion_id


def test_grounded_finding_is_supported(tmp_path: Path) -> None:
    """The opposite direction: a blanket refusal cannot pass this.

    Without it, returning ``NOT_SUPPORTED`` for everything would satisfy every
    other test in this module.
    """
    with sqlite3.connect(_user_db(tmp_path)) as conn:
        assertion_id = _grounded_finding(conn)
        conn.commit()
        provenance = compute_finding_provenance(conn, assertion_id)
        assert provenance is not None
        verdict = evaluate_finding_evidence(conn, provenance)

    assert verdict.status is EvidenceIntegrityStatus.SUPPORTED
    assert verdict.supported is True
    assert verdict.frame_ref == "index:g1"
    assert verdict.reason_codes == ()


def test_missing_evidence_is_unresolved(tmp_path: Path) -> None:
    """A declared ref that does not resolve cannot be current-supported.

    ANTI-VACUITY: give every cited ref ``ref_state="ok"`` in
    ``build_finding_evidence_adapter`` and this returns SUPPORTED for a finding
    whose cited assertion does not exist.
    """
    with sqlite3.connect(_user_db(tmp_path)) as conn:
        assertion_id = _grounded_finding(conn, evidence_refs=("assertion:does-not-exist",))
        conn.commit()
        provenance = compute_finding_provenance(conn, assertion_id)
        assert provenance is not None
        verdict = evaluate_finding_evidence(conn, provenance)

    assert verdict.supported is False
    assert verdict.status is EvidenceIntegrityStatus.UNRESOLVED
    assert "missing" in verdict.reason_codes


def test_citation_cycle_is_detected(tmp_path: Path) -> None:
    """A finding and a cited assertion citing each other is a cycle.

    "No circular-ancestry detection exists in the finding or assertion code"
    was this bead's confirmed defect. The loop is written through the ordinary
    assertion writer, not fabricated in memory.

    ANTI-VACUITY (executed): stop expanding cited assertions in
    ``build_finding_evidence_adapter`` (drop the ``pending.append`` branch) and
    the graph terminates at the first citation. The verdict then reports
    ``NOT_SUPPORTED`` with no witness naming the loop -- the right refusal for
    the wrong reason, and indistinguishable from an ordinary ungrounded
    ancestry, which is why the status is asserted and not just ``supported``.
    """
    from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        read_assertion_envelope,
        upsert_assertion,
    )

    with sqlite3.connect(_user_db(tmp_path)) as conn:
        finding_id = _grounded_finding(conn, claim_key="cycle-root")
        conn.commit()
        root = read_assertion_envelope(conn, finding_id)
        assert root is not None

        cited_id = "cycle-cited-assertion"
        upsert_assertion(
            conn,
            assertion_id=cited_id,
            scope_ref="agent:integrity-detector",
            target_ref=root.target_ref,
            key="cycle-cited",
            kind=AssertionKind.NOTE,
            value={"note": "cites the finding that cites it"},
            body_text="cites the finding that cites it",
            author_ref="agent:integrity-detector",
            author_kind="detector",
            evidence_refs=[f"assertion:{finding_id}"],
            status=AssertionStatus.ACTIVE,
            visibility=AssertionVisibility.PRIVATE,
            now_ms=2,
        )
        # Close the loop on the finding itself, preserving its identity.
        upsert_assertion(
            conn,
            assertion_id=finding_id,
            scope_ref=root.scope_ref,
            target_ref=root.target_ref,
            key=root.key,
            kind=AssertionKind.FINDING,
            value=root.value,
            body_text=root.body_text,
            author_ref=root.author_ref,
            author_kind="detector",
            evidence_refs=[*root.evidence_refs, f"assertion:{cited_id}"],
            status=AssertionStatus.ACTIVE,
            visibility=AssertionVisibility.PRIVATE,
            now_ms=3,
        )
        conn.commit()

        provenance = compute_finding_provenance(conn, finding_id)
        assert provenance is not None
        verdict = evaluate_finding_evidence(conn, provenance)

    assert verdict.supported is False
    assert verdict.status is EvidenceIntegrityStatus.CYCLE
    assert "cycle" in verdict.reason_codes


def test_closed_loop_on_the_detectors_own_output(tmp_path: Path) -> None:
    """A finding citing its own detector's output is not independent evidence."""
    with sqlite3.connect(_user_db(tmp_path)) as conn:
        assertion_id = _grounded_finding(
            conn,
            detector_ref="agent:self-citing-detector",
            evidence_refs=("agent:self-citing-detector",),
        )
        conn.commit()
        provenance = compute_finding_provenance(conn, assertion_id)
        assert provenance is not None
        verdict = evaluate_finding_evidence(conn, provenance)

    assert verdict.supported is False
    assert "closed_loop" in verdict.reason_codes


def test_ancestry_expansion_is_bounded(tmp_path: Path) -> None:
    """The graph population honours the evaluator's own node budget."""
    with sqlite3.connect(_user_db(tmp_path)) as conn:
        assertion_id = _grounded_finding(conn)
        conn.commit()
        provenance = compute_finding_provenance(conn, assertion_id)
        assert provenance is not None
        adapter = build_finding_evidence_adapter(
            conn,
            provenance,
            frame_hash="index:g1",
            definition_hash="definition",
            max_nodes=1,
        )

    assert len(adapter.nodes()) >= 1
    assert len(adapter.nodes()) <= 3


@pytest.mark.asyncio
async def test_resolution_reports_the_verdict(tmp_path: Path) -> None:
    """The finding ref stays addressable and carries the versioned verdict.

    ANTI-VACUITY: remove the ``evidence_integrity`` section and the
    not-current-supported caveats from ``_finding_ref_payload`` and this goes
    red -- the route returns to reporting an unresolved-ancestry finding with
    nothing but an advisory staleness string, which is the defect
    polylogue-rxdo.4 names.
    """
    from polylogue import Polylogue

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    initialize_archive_database(archive_root / "index.db", ArchiveTier.INDEX)
    with sqlite3.connect(_user_db(archive_root)) as conn:
        assertion_id = _grounded_finding(conn, evidence_refs=("assertion:does-not-exist",))
        conn.commit()

    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    try:
        payload = await archive.resolve_ref(f"finding:{assertion_id}")
    finally:
        await archive.close()

    # Addressable: AC3 keeps an unsupported finding reachable.
    assert payload.resolved is True
    assert payload.payload is not None
    integrity = payload.payload["evidence_integrity"]
    assert integrity["supported"] is False
    assert integrity["status"] == "unresolved"
    assert integrity["evaluator_version"] == "evidence-integrity-v1"
    assert payload.payload["staleness_verdict"] != "current"
    assert any("not current-supported" in caveat for caveat in payload.caveats)
