"""Semantic ID types for polylogue."""

from __future__ import annotations

from typing import Literal, NewType, get_args

SessionId = NewType("SessionId", str)
MessageId = NewType("MessageId", str)
AttachmentId = NewType("AttachmentId", str)
ContentHash = NewType("ContentHash", str)
SessionEventId = NewType("SessionEventId", str)

# ``messages.identity_source`` records which durable message-ID path fired.
# Keep the closed domain vocabulary beside the semantic identity types so
# index DDL and identity-producing code can share one owner.
MessageIdentitySource = Literal["native", "content"]
AttachmentDirection = Literal["user_input", "model_output"]
AttachmentAcquisitionStatus = Literal["acquired", "unavailable", "unfetched"]
AttachmentUploadOrigin = Literal["drive", "paste", "url", "oauth"]
SessionTagSource = Literal["user", "auto"]
LineageInheritance = Literal["prefix-sharing", "spawned-fresh"]
SessionCommitDetectionType = Literal["time_window", "file_overlap", "explicit_ref", "origin_reported"]
ConvergenceDebtStatus = Literal["failed", "deferred"]
CursorLagSeverity = Literal["info", "warning", "error", "critical"]
JudgmentSchedulerStatus = Literal["completed", "parked", "failed"]
OperationRunStatus = Literal["running", "completed", "failed", "interrupted", "completed_with_failures"]
RouteObservationStatus = Literal["ok", "error", "degraded", "timed_out", "unavailable"]
RouteDaemonPath = Literal["daemon", "direct"]


def require_literal(value: object, vocabulary: object, *, name: str) -> str:
    """Validate a persisted token against its single ``Literal`` owner."""
    allowed = get_args(vocabulary)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"invalid {name} {value!r}; expected one of {', '.join(allowed)}")
    return value


__all__ = [
    "AttachmentId",
    "AttachmentAcquisitionStatus",
    "AttachmentDirection",
    "AttachmentUploadOrigin",
    "ConvergenceDebtStatus",
    "CursorLagSeverity",
    "JudgmentSchedulerStatus",
    "LineageInheritance",
    "ContentHash",
    "MessageId",
    "MessageIdentitySource",
    "OperationRunStatus",
    "RouteDaemonPath",
    "RouteObservationStatus",
    "SessionId",
    "SessionCommitDetectionType",
    "SessionEventId",
    "SessionTagSource",
    "require_literal",
]
