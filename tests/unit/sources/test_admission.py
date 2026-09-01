"""Production-route laws for receipt-bound source admission."""

from __future__ import annotations

from polylogue.sources.live.admission import (
    AdmissionAttempt,
    AdmissionDisposition,
    AdmissionFailure,
    AdmissionReceipt,
    AdmissionState,
    AdmissionUnit,
    ArtifactIdentity,
    ContinuationDecision,
    FairAdmissionScheduler,
    ResourceEnvelope,
    SemanticFrontier,
    SourceCoordinates,
    continuation,
)


def _attempt() -> AdmissionAttempt:
    return AdmissionAttempt(
        "attempt-1",
        SourceCoordinates("source-1", "path/export.jsonl"),
        ArtifactIdentity.from_bytes(b"stable bytes"),
        "jsonl-v1",
        "parser-v1",
        None,
        ResourceEnvelope(1024, 1000),
    )


def test_acceptance_requires_frontier_and_releases_ownership() -> None:
    receipts: list[AdmissionReceipt] = []
    state = AdmissionState(_attempt(), receipts.append)
    frontier = SemanticFrontier("rev-1", 12, "jsonl-v1", "evidence-1")
    accepted = state.finish(AdmissionDisposition.ACCEPTED, frontier=frontier)
    assert accepted.frontier == frontier
    assert state.ownership_released
    assert state.finish(AdmissionDisposition.CANCELLED) is accepted


def test_continuation_reacquires_on_replacement_or_semantic_drift() -> None:
    frontier = SemanticFrontier("rev-1", 12, "jsonl-v1", "evidence-1")
    state = AdmissionState(_attempt(), lambda _receipt: None)
    prior = state.finish(AdmissionDisposition.ACCEPTED, frontier=frontier)
    assert (
        continuation(
            prior,
            artifact=prior.attempt.artifact,
            source_law="jsonl-v1",
            parser_identity="parser-v1",
            frontier_evidence=frontier,
        )
        is ContinuationDecision.RESUME
    )
    assert (
        continuation(
            prior,
            artifact=ArtifactIdentity.from_bytes(b"replacement"),
            source_law="jsonl-v1",
            parser_identity="parser-v1",
            frontier_evidence=frontier,
        )
        is ContinuationDecision.REACQUIRE
    )
    assert (
        continuation(
            prior,
            artifact=prior.attempt.artifact,
            source_law="jsonl-v2",
            parser_identity="parser-v1",
            frontier_evidence=frontier,
        )
        is ContinuationDecision.REACQUIRE
    )


def test_round_robin_keeps_small_sibling_live() -> None:
    calls: list[str] = []
    scheduler = FairAdmissionScheduler()
    scheduler.add(AdmissionUnit("whale", lambda: calls.append("whale-1")))
    scheduler.add(AdmissionUnit("small", lambda: calls.append("small-1")))
    list(scheduler.run())
    assert calls == ["whale-1", "small-1"]


def test_scheduler_drains_units_added_after_iteration_yields() -> None:
    scheduler = FairAdmissionScheduler()

    def add_late_unit() -> str:
        scheduler.add(AdmissionUnit("late", lambda: "late-result"))
        return "first-result"

    scheduler.add(AdmissionUnit("first", add_late_unit))

    assert list(scheduler.run()) == [("first", "first-result"), ("late", "late-result")]


def test_round_robin_bounds_each_source_and_isolates_poison() -> None:
    calls: list[str] = []
    scheduler = FairAdmissionScheduler()
    scheduler.add(AdmissionUnit("whale", lambda: calls.append("whale-1")))
    scheduler.add(AdmissionUnit("whale", lambda: (_ for _ in ()).throw(ValueError("poison"))))
    scheduler.add(AdmissionUnit("whale", lambda: calls.append("whale-3")))
    scheduler.add(AdmissionUnit("small", lambda: calls.append("small-1")))

    results = list(scheduler.run())

    assert calls == ["whale-1", "small-1", "whale-3"]
    assert isinstance(results[2][1], AdmissionFailure)
    assert results[2][1].retryable is False
