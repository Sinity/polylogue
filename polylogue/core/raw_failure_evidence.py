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
    DEFERRED_CAS_FRONTIER = "deferred_cas_frontier"
    # Historical rows written before CAS evidence was made provider-neutral.
    # Keep this token readable until a backup-gated migration or re-observation
    # receipt has converted every retained row.
    DEFERRED_CODEX_CAS_FRONTIER = "deferred_codex_cas_frontier"
    TERMINAL_CORRUPT_INPUT = "terminal_corrupt_input"
    TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER = "terminal_superseded_deferred_cas_frontier"
    TERMINAL_UNKNOWN_JSON_DECODE = "terminal_unknown_json_decode"
    TERMINAL_UNKNOWN_EXPORT_NO_SESSION = "terminal_unknown_export_no_session"
    TERMINAL_UNSUPPORTED_SHAPE = "terminal_unsupported_shape"

    @property
    def support_status(self) -> ArtifactSupportStatus:
        if self is RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER:
            return ArtifactSupportStatus.UNKNOWN
        if self in {
            RawFailureEvidenceKind.DEFERRED_HOT_JSONL_CAPTURE,
            RawFailureEvidenceKind.DEFERRED_CLAUDE_CODE_PARTIAL_JSONL,
            RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER,
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
        return "deferred" if self.value in RAW_FAILURE_DEFERRED_EVIDENCE_KINDS else "terminal"


def validated_raw_failure_evidence_kind(
    artifact_kind: object,
    support_status: object,
    *,
    validation_failed: bool,
) -> RawFailureEvidenceKind | None:
    """Return a typed kind only for a complete, self-consistent carrier.

    Decode failures are reported by the worker as validation failures because
    the payload cannot satisfy the input contract.  A matching terminal
    corrupt-input/decode carrier explains that state; deferred evidence still
    requires validation to have passed or been skipped.
    """
    if artifact_kind is None or support_status is None:
        return None
    try:
        evidence_kind = RawFailureEvidenceKind(str(artifact_kind))
    except ValueError:
        return None
    if evidence_kind.support_status.value != str(support_status):
        return None
    if validation_failed and evidence_kind not in {
        RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
        RawFailureEvidenceKind.TERMINAL_UNKNOWN_JSON_DECODE,
    }:
        return None
    return evidence_kind


RAW_FAILURE_EVIDENCE_KINDS = frozenset(kind.value for kind in RawFailureEvidenceKind)
RAW_FAILURE_DEFERRED_EVIDENCE_KINDS = frozenset(
    {
        RawFailureEvidenceKind.DEFERRED_HOT_JSONL_CAPTURE.value,
        RawFailureEvidenceKind.DEFERRED_CLAUDE_CODE_PARTIAL_JSONL.value,
        RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER.value,
        RawFailureEvidenceKind.DEFERRED_CODEX_CAS_FRONTIER.value,
    }
)
# Only frontier conflicts authorize retained-raw replay. Hot captures remain
# deferred until a complete source observation arrives; replaying their
# truncated blob would advance the cursor past the record that later bytes
# complete.
RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS = frozenset(
    {
        RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER.value,
        RawFailureEvidenceKind.DEFERRED_CODEX_CAS_FRONTIER.value,
    }
)
# Every deferred raw-failure carrier represents a partial decode. Consumers
# selecting retry authority must validate this companion field as well as the
# closed kind, or contradictory rows can authorize replay.
RAW_FAILURE_DEFERRED_SUPPORT_STATUS = ArtifactSupportStatus.PARTIAL_DECODE.value
RAW_FAILURE_EVIDENCE_SUPPORT_STATUS_PAIRS = tuple(
    sorted((kind.value, kind.support_status.value) for kind in RawFailureEvidenceKind)
)
RAW_FAILURE_TERMINAL_EVIDENCE_KINDS = frozenset(
    {
        RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT.value,
        RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.value,
        RawFailureEvidenceKind.TERMINAL_UNKNOWN_JSON_DECODE.value,
        RawFailureEvidenceKind.TERMINAL_UNKNOWN_EXPORT_NO_SESSION.value,
        RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value,
    }
)
RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS = tuple(
    sorted((kind.value, kind.support_status.value) for kind in RawFailureEvidenceKind if kind.lifecycle == "terminal")
)


__all__ = [
    "RAW_FAILURE_DEFERRED_EVIDENCE_KINDS",
    "RAW_FAILURE_DEFERRED_SUPPORT_STATUS",
    "RAW_FAILURE_EVIDENCE_KINDS",
    "RAW_FAILURE_EVIDENCE_SUPPORT_STATUS_PAIRS",
    "RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS",
    "RAW_FAILURE_TERMINAL_EVIDENCE_KINDS",
    "RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS",
    "RawFailureEvidenceKind",
    "validated_raw_failure_evidence_kind",
]
