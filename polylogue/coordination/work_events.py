"""Provider-neutral event payloads emitted by live agents."""

from __future__ import annotations

from typing import Final

WORK_EVENT_TYPES: Final[frozenset[str]] = frozenset({"tool_run", "subagent_spawn", "decision", "artifact_change"})


def validate_work_event_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in WORK_EVENT_TYPES:
        choices = ", ".join(sorted(WORK_EVENT_TYPES))
        raise ValueError(f"work event type must be one of: {choices}")
    return normalized


def validate_work_event_id(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("work event id must not be empty")
    return normalized


class WorkEventProvenanceRefusedError(ValueError):
    """The target session has no single retained acquisition provider."""

    def __init__(self, session_id: str, reason: str) -> None:
        self.session_id = session_id
        self.reason = reason
        super().__init__(f"work-event provenance refused for {session_id}: {reason}")
