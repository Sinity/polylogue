from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.pipeline.services.parsing import ParseResult


def _initial_counts() -> dict[str, int]:
    return {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
        "rendered": 0,
    }


def _initial_changed_counts() -> dict[str, int]:
    return {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
    }


@dataclass(slots=True)
class RunExecutionState:
    """Mutable run bookkeeping shared across pipeline stages."""

    counts: dict[str, int] = field(default_factory=_initial_counts)
    changed_counts: dict[str, int] = field(default_factory=_initial_changed_counts)
    processed_ids: set[str] = field(default_factory=set)
    render_failures: list[dict[str, str]] = field(default_factory=list)

    def record_acquire(self, acquire_result) -> None:
        self.counts["acquired"] = acquire_result.counts["acquired"]
        self.counts["skipped"] = acquire_result.counts["skipped"]
        self.counts["acquire_errors"] = acquire_result.counts["errors"]

    def record_validation(self, validation_result) -> None:
        if validation_result is None:
            return
        self.counts["validated"] = validation_result.counts["validated"]
        self.counts["validation_invalid"] = validation_result.counts["invalid"]
        self.counts["validation_drift"] = validation_result.counts["drift"]
        self.counts["validation_skipped_no_schema"] = validation_result.counts["skipped_no_schema"]
        self.counts["validation_errors"] = validation_result.counts["errors"]

    def record_parse(self, parse_result: ParseResult) -> None:
        for key, value in parse_result.counts.items():
            self.counts[key] = value
        if parse_result.parse_failures:
            self.counts["parse_failures"] = parse_result.parse_failures
        self.changed_counts.update(parse_result.changed_counts)
        self.processed_ids = parse_result.processed_ids
        self.counts["conversations"] = len(self.processed_ids)

    def record_schema_generation(self, *, generated: int, failed: int) -> None:
        self.counts["schemas_generated"] = generated
        self.counts["schemas_failed"] = failed

    def record_render(self, *, rendered: int, failures: list[dict[str, str]]) -> None:
        self.counts["rendered"] = rendered
        self.render_failures = failures
        if failures:
            self.counts["render_failures"] = len(failures)

    def finalize(self) -> dict[str, dict[str, int]]:
        new_counts = {
            "conversations": max(
                self.counts["conversations"] - self.changed_counts["conversations"],
                0,
            ),
            "messages": max(self.counts["messages"] - self.changed_counts["messages"], 0),
            "attachments": max(
                self.counts["attachments"] - self.changed_counts["attachments"],
                0,
            ),
        }
        self.counts["new_conversations"] = new_counts["conversations"]
        self.counts["changed_conversations"] = self.changed_counts["conversations"]
        return {
            "new": new_counts,
            "removed": {"conversations": 0, "messages": 0, "attachments": 0},
            "changed": dict(self.changed_counts),
        }


__all__ = ["RunExecutionState"]
