"""Closed evidence vocabulary for raw parse outcomes.

The source tier retains the original raw bytes and parser diagnostic. This
module adds the separate, machine-readable outcome needed to distinguish a
source that may progress from a payload that has reached a terminal refusal.
"""

from __future__ import annotations

from enum import StrEnum

from polylogue.core.enums import ArtifactSupportStatus


class RawFailureEvidenceKind(StrEnum):
    """Durable lifecycle evidence attached to a retained raw artifact."""

    DEFERRED_HOT_JSONL_CAPTURE = "deferred_hot_jsonl_capture"
    TERMINAL_CORRUPT_INPUT = "terminal_corrupt_input"
    TERMINAL_UNSUPPORTED_SHAPE = "terminal_unsupported_shape"

    @property
    def support_status(self) -> ArtifactSupportStatus:
        if self is RawFailureEvidenceKind.DEFERRED_HOT_JSONL_CAPTURE:
            return ArtifactSupportStatus.PARTIAL_DECODE
        if self is RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT:
            return ArtifactSupportStatus.DECODE_FAILED
        return ArtifactSupportStatus.UNSUPPORTED_PARSEABLE

    @property
    def lifecycle(self) -> str:
        return "deferred" if self is RawFailureEvidenceKind.DEFERRED_HOT_JSONL_CAPTURE else "terminal"


RAW_FAILURE_EVIDENCE_KINDS = frozenset(kind.value for kind in RawFailureEvidenceKind)
RAW_FAILURE_DEFERRED_EVIDENCE_KINDS = frozenset({RawFailureEvidenceKind.DEFERRED_HOT_JSONL_CAPTURE.value})
RAW_FAILURE_TERMINAL_EVIDENCE_KINDS = frozenset(
    {
        RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT.value,
        RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value,
    }
)


__all__ = [
    "RAW_FAILURE_DEFERRED_EVIDENCE_KINDS",
    "RAW_FAILURE_EVIDENCE_KINDS",
    "RAW_FAILURE_TERMINAL_EVIDENCE_KINDS",
    "RawFailureEvidenceKind",
]
