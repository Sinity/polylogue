"""Drive canonical retained preparation and record actual publication results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.stage_admission import admit_stage_write
from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation, raw_observation_frame
from polylogue.sources import revision_backfill
from polylogue.sources.revision_backfill import (
    PreparedMembershipReplay,
    PreparedRetainedAggregate,
    PreparedRetainedInput,
    PreparedRevisionReplayResult,
    RetainedPreparationRetryableError,
    RevisionCensusResult,
)
from polylogue.storage.sqlite.archive_tiers.revision_governance import PreparedRawRevisionClassification
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionWrite
from polylogue.storage.sqlite.connection_profile import readonly_connection_context


@dataclass(frozen=True, slots=True)
class RetainedReplayRun:
    """Actual prepared apply receipts emitted during one synthetic replay."""

    receipts: tuple[PreparedRevisionReplayResult | RevisionCensusResult, ...]

    @property
    def scanned(self) -> int:
        return sum(receipt.scanned for receipt in self.receipts)

    @property
    def classified_full(self) -> int:
        return sum(receipt.classified_full for receipt in self.receipts)

    @property
    def replayed_logical_sources(self) -> int:
        return sum(
            receipt.replayed_logical_sources
            for receipt in self.receipts
            if isinstance(receipt, PreparedRevisionReplayResult)
        )

    @property
    def quarantined(self) -> int:
        return sum(receipt.quarantined for receipt in self.receipts)

    @property
    def adoption_deferred(self) -> int:
        return sum(
            receipt.adoption_deferred for receipt in self.receipts if isinstance(receipt, PreparedRevisionReplayResult)
        )


def replay_retained_components(
    archive_root: Path,
    *,
    selected_raw_ids: Sequence[str] | None = None,
    active_index_path: Path | None = None,
) -> RetainedReplayRun:
    """Run the real captured preparation/publication route without fallback.

    Preparatory source census and byte-classification passes must change their
    durable input binding. A refused attempt with no progress is surfaced;
    this harness never adds a timeout, a retry count or a substitute result.
    """
    with readonly_connection_context(archive_root / "source.db") as source:
        retained = tuple(str(row[0]) for row in source.execute("SELECT raw_id FROM raw_sessions ORDER BY rowid"))
    selected = frozenset(selected_raw_ids) if selected_raw_ids is not None else None
    seeds = tuple(raw_id for raw_id in retained if selected is None or raw_id in selected)
    adapter = make_raw_observation_derivation(archive_root, index_db_path=active_index_path)
    frame = raw_observation_frame(archive_root, raw_ids=seeds, index_db_path=active_index_path)
    receipts: list[PreparedRevisionReplayResult | RevisionCensusResult] = []
    failures: list[RetainedPreparationRetryableError] = []
    original_apply = revision_backfill.apply_prepared_revision_replay
    original_census = revision_backfill.apply_prepared_revision_census

    def census(
        archive_root: Path,
        *,
        active_index_path: Path,
        selected_raw_ids: list[str],
        prepared_inputs: Mapping[str, PreparedRetainedInput] | None = None,
        classification_proofs: Mapping[str, PreparedRawRevisionClassification] | None = None,
    ) -> RevisionCensusResult:
        try:
            result = original_census(
                archive_root,
                active_index_path=active_index_path,
                selected_raw_ids=selected_raw_ids,
                prepared_inputs=prepared_inputs,
                classification_proofs=classification_proofs,
            )
        except RetainedPreparationRetryableError as failure:
            failures.append(failure)
            raise
        receipts.append(result)
        return result

    def apply(
        archive_root: Path,
        *,
        active_index_path: Path,
        selected_raw_ids: list[str],
        prepared_inputs: Mapping[str, PreparedRetainedInput],
        prepared_aggregates: Mapping[str, PreparedRetainedAggregate],
        prepared_writes: Mapping[tuple[str, str], PreparedSessionWrite],
        prepared_replay_plans: Mapping[str, tuple[str, ...]],
        prepared_membership_plans: Mapping[str, PreparedMembershipReplay],
        bulk_fts: bool = True,
        exact_fts_audit: bool = False,
    ) -> PreparedRevisionReplayResult:
        try:
            result = original_apply(
                archive_root,
                active_index_path=active_index_path,
                selected_raw_ids=selected_raw_ids,
                prepared_inputs=prepared_inputs,
                prepared_aggregates=prepared_aggregates,
                prepared_writes=prepared_writes,
                prepared_replay_plans=prepared_replay_plans,
                prepared_membership_plans=prepared_membership_plans,
                bulk_fts=bulk_fts,
                exact_fts_audit=exact_fts_audit,
            )
        except RetainedPreparationRetryableError as failure:
            failures.append(failure)
            raise
        receipts.append(result)
        return result

    visited: set[str] = set()
    with pytest.MonkeyPatch.context() as observe:
        observe.setattr(revision_backfill, "apply_prepared_revision_replay", apply)
        observe.setattr(revision_backfill, "apply_prepared_revision_census", census)
        for raw_id in seeds:
            if raw_id in visited:
                continue
            while True:
                check_compute_cancelled()
                replacement = adapter.compute(frame, raw_id, replay_current=True)
                before = adapter._binding(replacement.raw_ids)
                started = False
                previous_failures = len(failures)

                def publish(replacement=replacement) -> bool:
                    nonlocal started
                    started = True
                    return adapter.publish(frame, replacement)

                try:
                    published = admit_stage_write("synthetic-retained-replay", publish)
                finally:
                    if not started:
                        replacement.close()
                if published:
                    visited.update(replacement.raw_ids)
                    break
                if adapter._binding(replacement.raw_ids) == before:
                    if len(failures) > previous_failures:
                        raise failures[-1]
                    raise RetainedPreparationRetryableError("canonical retained publication refused without progress")
    return RetainedReplayRun(tuple(receipts))
