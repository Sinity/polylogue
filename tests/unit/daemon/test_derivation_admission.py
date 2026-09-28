"""Fair-intake verdicts for exact-key derivation passes (raw and hook carriers)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.daemon.cli import _derivation_admission
from polylogue.daemon.derivation import (
    DerivationFrame,
    DerivationKey,
    DerivationReport,
    KeyOutcome,
    Outcome,
    PendingReason,
)
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.intake import (
    AdmissionOutcome,
    AdmissionResult,
    FairIntakeDispatcher,
    IntakeClassSpec,
    IntakeItem,
)
from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.operations.intake_adapters import RawMaterializationDiscovery
from polylogue.storage.derived.raw import RAW_OBSERVATION_DOMAIN
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root

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


@pytest.mark.asyncio
async def test_invalid_retained_payload_is_terminal_across_restart(tmp_path: Path) -> None:
    """A syntactically invalid payload converges to a durable terminal verdict.

    Red if the parse failure only raises (RETRYABLE in memory) instead of
    persisting a refusal: the second discovery, standing in for a daemon
    restart, would offer the same raw again.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b'[{"id": "broken", "mapping": ',
            source_path="broken.json",
            acquired_at_ms=1,
        )
    assert [item[0] for item in RawMaterializationDiscovery(tmp_path).discover_pending_raw_ids(8)] == [raw_id]
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    owner = RawObservationConvergenceOwner(
        tmp_path,
        compute_adapter=compute,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
    )
    try:
        report = await owner.converge_raw_id(raw_id)
        result = _derivation_admission(report, raw_id, subject="raw observation")
        assert result.outcome in (AdmissionOutcome.ADMITTED, AdmissionOutcome.DUPLICATE), result
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)

    assert RawMaterializationDiscovery(tmp_path).discover_pending_raw_ids(8) == ()
