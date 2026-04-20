from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.pipeline.services.parsing import ParseResult
from polylogue.pipeline.stage_models import AcquireResult, ValidateResult
from polylogue.storage.run_state import DriftBucket, RenderFailurePayload, RunCounts, RunDrift


def _initial_counts() -> RunCounts:
    return RunCounts(
        conversations=0,
        messages=0,
        attachments=0,
        skipped_conversations=0,
        skipped_messages=0,
        skipped_attachments=0,
        materialized=0,
        rendered=0,
    )


def _initial_changed_counts() -> DriftBucket:
    return DriftBucket()


@dataclass(slots=True)
class RunExecutionState:
    """Mutable run bookkeeping shared across pipeline stages."""

    counts: RunCounts = field(default_factory=_initial_counts)
    changed_counts: DriftBucket = field(default_factory=_initial_changed_counts)
    processed_ids: set[str] = field(default_factory=set)
    render_failures: list[RenderFailurePayload] = field(default_factory=list)

    def record_acquire(self, acquire_result: AcquireResult) -> None:
        self.counts.acquired = acquire_result.acquired
        self.counts.skipped = acquire_result.skipped
        self.counts.acquire_errors = acquire_result.errors

    def record_validation(self, validation_result: ValidateResult | None) -> None:
        if validation_result is None:
            return
        self.counts.validated = validation_result.validated
        self.counts.validation_invalid = validation_result.invalid
        self.counts.validation_drift = validation_result.drift
        self.counts.validation_skipped_no_schema = validation_result.skipped_no_schema
        self.counts.validation_errors = validation_result.errors

    def record_parse(self, parse_result: ParseResult) -> None:
        for key, value in parse_result.counts.items():
            setattr(self.counts, key, value)
        if parse_result.parse_failures:
            self.counts.parse_failures = parse_result.parse_failures
        for key, value in parse_result.changed_counts.items():
            current = getattr(self.changed_counts, key)
            setattr(self.changed_counts, key, current + value)
        self.processed_ids = parse_result.processed_ids
        self.counts.conversations = len(self.processed_ids)

    def record_schema_generation(self, *, generated: int, failed: int) -> None:
        self.counts.schemas_generated = generated
        self.counts.schemas_failed = failed

    def record_materialize(self, *, materialized: int) -> None:
        self.counts.materialized = materialized

    def record_render(self, *, rendered: int, failures: list[RenderFailurePayload]) -> None:
        self.counts.rendered = rendered
        self.render_failures = failures
        if failures:
            self.counts.render_failures = len(failures)

    def finalize(self) -> RunDrift:
        new_counts = DriftBucket(
            conversations=max(self.counts.int_value("conversations") - self.changed_counts.conversations, 0),
            messages=max(self.counts.int_value("messages") - self.changed_counts.messages, 0),
            attachments=max(self.counts.int_value("attachments") - self.changed_counts.attachments, 0),
        )
        self.counts.new_conversations = new_counts.conversations
        self.counts.changed_conversations = self.changed_counts.conversations
        return RunDrift(
            new=new_counts,
            removed=DriftBucket(),
            changed=self.changed_counts.model_copy(deep=True),
        )


__all__ = ["RunExecutionState"]
