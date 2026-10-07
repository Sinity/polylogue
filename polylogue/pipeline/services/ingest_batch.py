"""Publish one batch of acquired Raw IDs through the retained Raw owner.

Acquisition has physically settled before a batch reaches this module. The
caller's retained owner prepares, publishes and acknowledges each Raw; this
host only projects that owner's receipts into the parsing result.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from polylogue.core.raw_failure_evidence import CohortMembershipRefusalError, RetainedRawDecodeRefusalError
from polylogue.logging import WARNING, emit
from polylogue.pipeline.payload_types import ParseBatchObservation
from polylogue.sinex.material_adapter import PublicationEncodingError
from polylogue.sinex.models import PublicationMode

if TYPE_CHECKING:
    from polylogue.core.protocols import ProgressCallback
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.pipeline.services.parsing_models import ParseResult


async def process_ingest_batch(
    service: ParsingService,
    batch_ids: list[str],
    result: ParseResult,
    progress_callback: ProgressCallback | None,
) -> ParseBatchObservation:
    """Publish acquired Raw IDs through the caller's original retained owner.

    Canonical Raw replay owns Source census, Index publication and final
    acknowledgement; this host only projects its actual receipts into the
    parsing result.
    """
    if service.retained_runner is None:
        raise PermissionError("ingest publication requires its supplied retained Raw owner")
    from polylogue.config import load_polylogue_config

    publication_mode = PublicationMode.from_string(load_polylogue_config().sinex_mode)
    if publication_mode is not PublicationMode.OFF:
        raise PublicationEncodingError(
            "retained ingest requires its canonical accepted-marker and outbox producer before publication"
        )
    started = time.perf_counter()
    # A parse failure is a typed refusal the owner settled: a terminal decode
    # refusal or a refused cohort member. A membership-quarantined revision is
    # governance evidence, not a failed parse, so receipts' ``quarantined``
    # counts are not failures.
    refusals: list[object] = []

    def settle_terminal_refusal(_keys: tuple[str, ...], refusal: RetainedRawDecodeRefusalError) -> None:
        refusals.append(refusal)

    def settle_membership_refusal(refusal: CohortMembershipRefusalError) -> None:
        refusals.append(refusal)

    replay = await service.retained_runner(
        tuple(batch_ids),
        on_terminal_refusal=settle_terminal_refusal,
        on_membership_refusal=settle_membership_refusal,
    )
    receipts = replay.receipts
    # A raw whose preparation failed retryably stays retained and unpublished
    # for the next pass; it is this batch's failure, not a lost sibling page.
    for failure in replay.failures:
        emit(
            "ingest.retained_preparation_failed",
            level=WARNING,
            outcome="error",
            raw_id=failure.raw_id,
            error_type=type(failure.error).__name__,
        )
    result.parse_failures += len(refusals) + len(replay.failures)
    written: dict[str, None] = {}
    changed: dict[str, None] = {}
    for receipt in receipts:
        written.update(dict.fromkeys(receipt.written_session_ids))
        changed.update(dict.fromkeys(receipt.changed_session_ids))
        for key, count in receipt.written_counts.items():
            if key in result.counts:
                result.counts[key] += count
            if key in result.changed_counts and key != "sessions":
                result.changed_counts[key] += count
        for key, seconds in receipt.stage_timings_s.items():
            result.stage_timings_s[key] = result.stage_timings_s.get(key, 0.0) + seconds
    result.processed_ids.update(written)
    result._changed_session_ids.extend(key for key in changed if key not in result._changed_session_ids)
    result.changed_counts["sessions"] += len(changed)
    if progress_callback and receipts:
        progress_callback(len(batch_ids))
    return {
        "records": len(batch_ids),
        "sessions": len(written),
        "messages": sum(receipt.written_message_count for receipt in receipts),
        "changed_sessions": len(changed),
        "failed_raw_count": len(refusals) + len(replay.failures),
        "converged": not replay.failures and all(receipt.adoption_deferred == 0 for receipt in receipts),
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


__all__ = ["process_ingest_batch"]
