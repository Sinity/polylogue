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
    DEFERRED_CLAUDE_CODE_PARTIAL_JSONL = "deferred_claude_code_partial_jsonl"
    DEFERRED_CODEX_CAS_FRONTIER = "deferred_codex_cas_frontier"
    TERMINAL_CORRUPT_INPUT = "terminal_corrupt_input"
    TERMINAL_UNKNOWN_JSON_DECODE = "terminal_unknown_json_decode"
    TERMINAL_UNKNOWN_EXPORT_NO_SESSION = "terminal_unknown_export_no_session"
    TERMINAL_UNSUPPORTED_SHAPE = "terminal_unsupported_shape"

    @property
    def support_status(self) -> ArtifactSupportStatus:
        if self in {
            RawFailureEvidenceKind.DEFERRED_HOT_JSONL_CAPTURE,
            RawFailureEvidenceKind.DEFERRED_CLAUDE_CODE_PARTIAL_JSONL,
            RawFailureEvidenceKind.DEFERRED_CODEX_CAS_FRONTIER,
        }:
            return ArtifactSupportStatus.PARTIAL_DECODE
        if self in {
            RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
            RawFailureEvidenceKind.TERMINAL_UNKNOWN_JSON_DECODE,
        }:
            return ArtifactSupportStatus.DECODE_FAILED
        return ArtifactSupportStatus.UNSUPPORTED_PARSEABLE

    @property
    def lifecycle(self) -> str:
        return "deferred" if self in RAW_FAILURE_DEFERRED_EVIDENCE_KINDS else "terminal"


RAW_FAILURE_EVIDENCE_KINDS = frozenset(kind.value for kind in RawFailureEvidenceKind)
RAW_FAILURE_DEFERRED_EVIDENCE_KINDS = frozenset(
    {
        RawFailureEvidenceKind.DEFERRED_HOT_JSONL_CAPTURE.value,
        RawFailureEvidenceKind.DEFERRED_CLAUDE_CODE_PARTIAL_JSONL.value,
        RawFailureEvidenceKind.DEFERRED_CODEX_CAS_FRONTIER.value,
    }
)
RAW_FAILURE_TERMINAL_EVIDENCE_KINDS = frozenset(
    {
        RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT.value,
        RawFailureEvidenceKind.TERMINAL_UNKNOWN_JSON_DECODE.value,
        RawFailureEvidenceKind.TERMINAL_UNKNOWN_EXPORT_NO_SESSION.value,
        RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value,
    }
)


__all__ = [
    "RAW_FAILURE_DEFERRED_EVIDENCE_KINDS",
    "RAW_FAILURE_EVIDENCE_KINDS",
    "RAW_FAILURE_TERMINAL_EVIDENCE_KINDS",
    "RawFailureEvidenceKind",
]
