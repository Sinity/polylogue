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
