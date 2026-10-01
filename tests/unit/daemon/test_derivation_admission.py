"""Fair-intake verdicts for exact-key derivation passes (raw and hook carriers)."""

from __future__ import annotations

import pytest

from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.daemon.cli import _derivation_admission
from polylogue.daemon.derivation import (
    DerivationFrame,
    DerivationKey,
    DerivationReport,
    KeyOutcome,
    Outcome,
    PendingReason,
)
from polylogue.daemon.intake import (
    AdmissionOutcome,
    AdmissionResult,
    FairIntakeDispatcher,
    IntakeClassSpec,
    IntakeItem,
)
from polylogue.storage.derived.raw import RAW_OBSERVATION_DOMAIN

_FRAME = DerivationFrame(archive_root="/archive", source_revision="r1")


def _report(*outcomes: KeyOutcome) -> DerivationReport:
    counts: dict[Outcome, int] = {}
    for outcome in outcomes:
        counts[outcome.outcome] = counts.get(outcome.outcome, 0) + 1
    return DerivationReport(frame=_FRAME, outcomes=outcomes, counts=counts)


def _key(key: str) -> DerivationKey:
    return DerivationKey(RAW_OBSERVATION_DOMAIN, key)


def test_domain_level_discovery_failure_is_retryable_not_duplicate() -> None:
    """Filtering outcomes to the raw's own key (dropping ``"*"``) makes this DUPLICATE."""
    report = _report(KeyOutcome(key=_key("*"), outcome=Outcome.FAILED, error="discover: database is locked"))

    result = _derivation_admission(report, "raw-1", subject="raw observation")

    assert result.outcome is AdmissionOutcome.RETRYABLE
    assert result.reason == "discover: database is locked"


def test_truncated_failure_sample_still_counts() -> None:
    """A failure only in ``counts`` (sample truncated) must not read as a duplicate."""
    report = DerivationReport(frame=_FRAME, outcomes=(), counts={Outcome.FAILED: 1}, truncated=True)

    assert _derivation_admission(report, "raw-1", subject="raw observation").outcome is AdmissionOutcome.RETRYABLE


@pytest.mark.parametrize("key", ["raw-1", "*"])
def test_only_exact_typed_terminal_input_failure_is_excluded(key: str) -> None:
    report = _report(
        KeyOutcome(
            key=_key(key),
            outcome=Outcome.FAILED,
            error="retained decode refusal",
            terminal_refusal=RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
        )
    )
    expected = AdmissionOutcome.EXCLUDED if key == "raw-1" else AdmissionOutcome.RETRYABLE
    assert _derivation_admission(report, "raw-1", subject="raw observation").outcome is expected


def test_nontransient_infrastructure_failure_remains_retryable() -> None:
    report = _report(KeyOutcome(key=_key("raw-1"), outcome=Outcome.FAILED, error="schema unavailable", transient=False))
    assert _derivation_admission(report, "raw-1", subject="raw observation").outcome is AdmissionOutcome.RETRYABLE


def test_pending_and_done_and_duplicate_verdicts() -> None:
    pending = _report(KeyOutcome(key=_key("raw-1"), outcome=Outcome.PENDING, reason=PendingReason.BLOCKED))
    done = _report(KeyOutcome(key=_key("raw-1"), outcome=Outcome.DONE))

    assert _derivation_admission(pending, "raw-1", subject="raw observation").reason == "blocked"
    assert _derivation_admission(done, "raw-1", subject="raw observation").outcome is AdmissionOutcome.ADMITTED
    assert _derivation_admission(_report(), "raw-1", subject="raw observation").outcome is AdmissionOutcome.DUPLICATE


@pytest.mark.asyncio
async def test_admission_keeps_the_byte_estimate_as_its_cost() -> None:
    """Returning ``actual_cost=1`` reconciles a 4096-byte item down to one unit.

    That refund is returned to the class deficit, so the next pass admits
    more bytes than its share permits.
    """
    report = _report(KeyOutcome(key=_key("raw-a"), outcome=Outcome.DONE))

    class Adapter:
        async def discover(self, *, limit: int) -> list[IntakeItem]:
            return [IntakeItem("raw-a", class_name="raw", estimated_cost=4096)][:limit]

        async def admit(self, item: IntakeItem) -> AdmissionResult:
            return _derivation_admission(report, item.item_id, subject="raw observation")

        async def acknowledge(self, item: IntakeItem) -> None:
            return None

    dispatcher = FairIntakeDispatcher(
        (IntakeClassSpec(name="raw", adapter=Adapter(), page_size=1),),
        frame="test:derivation-admission",
    )
    result = await dispatcher.run_once()

    (report_row,) = result.classes
    assert report_row.admitted == 1
    assert report_row.actual_cost == report_row.estimated_cost == 4096
    assert report_row.reconciled_cost == 0
