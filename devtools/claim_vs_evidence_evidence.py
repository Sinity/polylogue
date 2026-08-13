"""Represent a claim-vs-evidence report run as first-party archive evidence.

The classification/economy logic stays in ``devtools/claim_vs_evidence.py``
(the harness). This module only represents the harness's OUTPUT as durable
evidence: a content-addressed :class:`~polylogue.storage.sqlite.query_objects.
QueryObject` for the structured-failure selection (the AnalysisDefinition), a
:class:`~polylogue.storage.sqlite.query_objects.ResultSetManifest` for the
rows the run actually matched, an
:class:`~polylogue.storage.sqlite.query_objects.EvaluationReceipt` binding the
run to tier generations and the runtime build (the AnalysisRun), and
``AssertionKind.FINDING`` rows for the headline numbers (polylogue-rxdo.13).

It writes through the same production primitives the daemon's own
standing-query convergence stage uses
(``polylogue/daemon/convergence_standing_queries.py``) via
``open_daemon_connection`` -- not a new generic finding registry, not a
metric/pattern/cohort/experiment definition system, and not a scheduler. A
finding is written with ``public_claim=None`` (no ``PublicClaimDeclaration``)
unless the run's own construct-validity gates (``n_min``, non-zero classified
outcomes) are satisfied, so an unpublishable run still gets an honest private
evidence record without ever exposing a degenerate rate as a public claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from polylogue.archive.query.production_evaluator import (
    _index_epoch,  # planner-internal seam, reused for the same tier-generation identity
    _polylogue_runtime_build_ref,
    _tier_generation,
)
from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONValue
from polylogue.core.query_identity import JsonValue
from polylogue.core.query_identity import query_ref as _query_object_ref
from polylogue.core.query_identity import result_set_ref as _result_set_object_ref
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ArchiveAssertionEnvelope,
    FindingAssertion,
    PublicClaimDeclaration,
    upsert_findings_as_assertions,
)
from polylogue.storage.sqlite.connection_profile import open_daemon_connection
from polylogue.storage.sqlite.query_objects import (
    EvaluationReceipt,
    QueryObject,
    ResultSetManifest,
    get_result_set,
    membership_merkle_root,
    put_evaluation_receipt,
    put_query,
    put_result_set,
)

# Bump when the classifier's silent/acknowledged/ambiguous taxonomy or the
# handler-class split changes meaning, so the AnalysisDefinition identity
# (query_hash) changes with it instead of silently reusing a stale one.
CLASSIFIER_DEFINITION_VERSION = "2"
ANALYSIS_TARGET_REF = "analysis:claim-vs-evidence"
_QUERY_GRAIN = "structured-failure-followup"
_QUERY_LANE = "analysis"
_QUERY_RANK_POLICY = "origin,session_id,tool_id,tool_result_message_id"


class MaterializedEvidence(TypedDict):
    query_ref: str
    result_set_ref: str
    receipt_id: str
    finding_assertion_ids: list[str]
    public_claim_written: bool


def build_query_definition(report: dict[str, Any]) -> dict[str, JsonValue]:
    """Return the content-addressed AnalysisDefinition payload for one report run.

    Deliberately not a DSL-executable plan: the paired next-assistant-turn
    lookup with window-3 lookahead has no representation in the query
    predicate grammar today. This is provenance identity, mirroring the
    ``convergence_standing_queries`` doctrine that durable identity JSON is
    "provenance, not source syntax to reverse-compile" -- the harness in
    ``devtools/claim_vs_evidence.py`` remains the sole executor.
    """
    frame = report["sample_frame"]
    return {
        "kind": "analysis-selection",
        "analysis_id": "claim-vs-evidence",
        "classifier_module": "polylogue.archive.actions.followup",
        "classifier_function": "classify_failed_followup_evidence",
        "classifier_definition_version": CLASSIFIER_DEFINITION_VERSION,
        "failure_predicate": frame["failure_predicate"],
        "classification_scope": frame["classification_scope"],
        "sensitivity_scope": frame["sensitivity_scope"],
        "selection_strategy": frame["selection_strategy"],
        "n_min": frame["n_min"],
        "limit": report["limit"],
        "handler_class_definition": report["handler_class_definition"],
    }


def build_result_set_members(report: dict[str, Any]) -> tuple[str, ...]:
    """Return the sorted ``message:`` refs the run actually classified."""
    return tuple(report["evidence"]["member_refs"])


def build_evaluation_receipt(
    archive_root: Path,
    index_db: Path,
    *,
    query_hash: str,
    result_set_id: str,
    created_at_ms: int,
) -> EvaluationReceipt:
    """Bind one run to its source/user/index tier generations and runtime build.

    ``receipt_id`` is content-addressed (not a random UUID, unlike
    ``ArchiveCanonicalPlanEvaluator``'s per-execution telemetry receipts) over
    every field ``put_evaluation_receipt`` itself treats as significant,
    including ``created_at_ms`` -- that function already rejects reusing a
    receipt id with a changed ``created_at_ms``, so folding it into the hash
    is what lets two calls at the same ``created_at_ms`` collapse into one
    safe no-op while two calls at different times correctly get distinct
    receipts instead of a spurious conflict.
    """
    source_generation = _tier_generation(archive_root / "source.db", label="source")
    user_generation = _tier_generation(archive_root / "user.db", label="user")
    index_generation = _index_epoch(index_db)
    runtime_build_ref = _polylogue_runtime_build_ref()
    receipt_digest = hash_payload(
        [
            query_hash,
            result_set_id,
            source_generation,
            user_generation,
            index_generation,
            runtime_build_ref,
            created_at_ms,
        ]
    )
    receipt_id = f"receipt-{receipt_digest}"
    return EvaluationReceipt(
        receipt_id=receipt_id,
        source_generation=source_generation,
        user_generation=user_generation,
        index_generation=index_generation,
        runtime_build_ref=runtime_build_ref,
    )


def build_findings(
    report: dict[str, Any],
    *,
    query_reference: str,
    result_set_reference: str,
    receipt: EvaluationReceipt,
) -> list[FindingAssertion]:
    """Return the headline-number FindingAssertions for one report run.

    ``public_claim`` stays ``None`` (no PublicClaimDeclaration) unless the
    aggregate rate actually clears the run's own ``n_min``/classified-outcome
    gates -- an unpublishable run still gets an honest private evidence
    record, never a fabricated public rate.
    """
    frame = report["sample_frame"]
    totals = report["totals"]
    rates = report["rates"]
    run_ref = f"run:claim-vs-evidence-{report['captured_at']}"
    aggregate_publishable = rates["publication_status"] == "supported" and rates["silent_rate_lower_bound"] is not None
    statistic: dict[str, JSONValue] = {
        "op": "lower_bound",
        "value": rates["silent_rate_lower_bound"],
        "unit": "ratio",
        "numerator": totals["silent_proceed"],
        "denominator": totals["failed_outcomes"],
        "ambiguous": totals["ambiguous"],
        "classified_outcomes": totals.get("classified_outcomes"),
    }
    body_text = (
        f"Structured-failure follow-up classification over {frame['inspected_structured_failures']} "
        f"inspected failures: {totals['acknowledged']} acknowledged, {totals['silent_proceed']} silent, "
        f"{totals['ambiguous']} ambiguous."
    )
    public_claim: PublicClaimDeclaration | None = None
    if aggregate_publishable:
        body_text = (
            f"In one bounded private-archive sample, {totals['silent_proceed']} of "
            f"{totals['failed_outcomes']} inspected structured failures were followed by silent "
            f"continuation on the next assistant turn, a {rates['silent_rate_lower_bound']:.1%} lower bound."
        )
        public_claim = PublicClaimDeclaration(
            publication=body_text,
            scope=(
                f"One private archive; {frame['inspected_structured_failures']} inspected structured "
                "failures from the run's bounded sample frame; next assistant turn only."
            ),
            caveat=(
                "This is not a population estimate; ambiguous rows are excluded from the classified "
                "denominator, and support must be recomputed when the evidence epoch, definition, or "
                "frame changes."
            ),
            public_evidence_refs=("file:docs/findings/claim-vs-evidence.md",),
            disclosure="public",
        )
    return [
        FindingAssertion(
            claim_key="finding.silent-proceed-lower-bound",
            target_ref=ANALYSIS_TARGET_REF,
            body_text=body_text,
            finding_kind="claim-vs-evidence",
            statistic=statistic,
            n=totals["failed_outcomes"],
            query_ref=query_reference,
            result_set_ref=result_set_reference,
            detector_ref=run_ref,
            evidence_refs=("file:docs/findings/claim-vs-evidence.md",),
            source_epoch=report["captured_at"],
            evaluation_ref=f"receipt:{receipt.receipt_id}",
            frame_ref=query_reference,
            public_claim=public_claim,
        )
    ]


def materialize_claim_vs_evidence_evidence(
    report: dict[str, Any],
    *,
    archive_root: Path,
    now_ms: int,
) -> MaterializedEvidence:
    """Register one report run's query, result set, receipt, and findings.

    Writes through ``open_daemon_connection`` (the same connection helper the
    daemon's own standing-query convergence stage uses), so this coexists
    with the running daemon's single-writer discipline instead of bypassing
    it with a bare ``sqlite3.connect``.

    The AnalysisDefinition (query) and its matched-row ResultSetManifest are
    content-addressed: identical selection logic and identical matched rows
    always resolve to the same identity, at any ``now_ms``. The AnalysisRun
    receipt and its FindingAssertion are scoped to ``now_ms``: calling this
    twice with the same ``report`` and the same ``now_ms`` is a safe retry
    no-op, but calling it again at a later ``now_ms`` records a new run and a
    new finding row even if the numbers happen to match, because the archive
    should carry that a re-verification happened under a later tier state --
    not silently collapse repeated regenerations into one row.
    """
    index_db = Path(report["index_db"])
    query_definition = build_query_definition(report)
    member_refs = build_result_set_members(report)
    conn = open_daemon_connection(archive_root / "user.db", timeout=30.0)
    try:
        query: QueryObject = put_query(
            conn,
            query_definition,
            grain=_QUERY_GRAIN,
            lane=_QUERY_LANE,
            rank_policy=_QUERY_RANK_POLICY,
            created_at_ms=now_ms,
        )
        query_reference = _query_object_ref(query.query_hash).format()
        result_set_id = f"finding-{membership_merkle_root(member_refs)}"
        result_set: ResultSetManifest | None = get_result_set(conn, result_set_id)
        if result_set is None:
            result_set = put_result_set(
                conn,
                result_set_id=result_set_id,
                query_hash=query.query_hash,
                grain=_QUERY_GRAIN,
                corpus_epoch=_index_epoch(index_db),
                member_refs=member_refs,
                exactness="capped",
                persistence_class="finding",
                created_at_ms=now_ms,
            )
        result_set_reference = _result_set_object_ref(result_set.result_set_id).format()
        receipt = build_evaluation_receipt(
            archive_root,
            index_db,
            query_hash=query.query_hash,
            result_set_id=result_set.result_set_id,
            created_at_ms=now_ms,
        )
        put_evaluation_receipt(
            conn,
            query_hash=query.query_hash,
            receipt=receipt,
            result_set_id=result_set.result_set_id,
            created_at_ms=now_ms,
        )
        findings = build_findings(
            report,
            query_reference=query_reference,
            result_set_reference=result_set_reference,
            receipt=receipt,
        )
        envelopes: list[ArchiveAssertionEnvelope] = upsert_findings_as_assertions(conn, findings, now_ms=now_ms)
        conn.commit()
    finally:
        conn.close()
    return {
        "query_ref": query_reference,
        "result_set_ref": result_set_reference,
        "receipt_id": receipt.receipt_id,
        "finding_assertion_ids": [envelope.assertion_id for envelope in envelopes],
        "public_claim_written": any(finding.public_claim is not None for finding in findings),
    }
