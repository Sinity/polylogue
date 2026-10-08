"""Closed evidence vocabulary for raw parse outcomes.

The source tier retains the original raw bytes and parser diagnostic. This
module adds the separate, machine-readable outcome needed to distinguish a
source that may progress from a payload that has reached a terminal refusal.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from polylogue.core.enums import ArtifactSupportStatus


class MissingProfileIdentityError(ValueError):
    """Retained Hermes bytes have no acquisition-bound profile qualifier."""

    outcome_code = "missing_profile_identity"


class RetainedZipMembershipUnprovedError(ValueError):
    """Retained ZIP bytes lack a proved acquired namespace and complete member set."""

    outcome_code = "retained_zip_membership_unproved"


class RawFailureEvidenceKind(StrEnum):
    """Durable lifecycle evidence attached to a retained raw artifact."""

    DEFERRED_HOT_JSONL_CAPTURE = "deferred_hot_jsonl_capture"
    DEFERRED_CLAUDE_CODE_PARTIAL_JSONL = "deferred_claude_code_partial_jsonl"
    DEFERRED_CAS_FRONTIER = "deferred_cas_frontier"
    TERMINAL_CORRUPT_INPUT = "terminal_corrupt_input"
    TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER = "terminal_superseded_deferred_cas_frontier"
    TERMINAL_UNKNOWN_JSON_DECODE = "terminal_unknown_json_decode"
    TERMINAL_UNKNOWN_EXPORT_NO_SESSION = "terminal_unknown_export_no_session"
    TERMINAL_UNSUPPORTED_SHAPE = "terminal_unsupported_shape"
    TERMINAL_MISSING_SOURCE_COORDINATES = "terminal_missing_source_coordinates"
    TERMINAL_MISSING_PROFILE_IDENTITY = "terminal_missing_profile_identity"
    TERMINAL_RETAINED_ZIP_MEMBERSHIP_UNPROVED = "terminal_retained_zip_membership_unproved"

    @property
    def support_status(self) -> ArtifactSupportStatus:
        if self in {
            RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER,
            RawFailureEvidenceKind.TERMINAL_MISSING_PROFILE_IDENTITY,
            RawFailureEvidenceKind.TERMINAL_RETAINED_ZIP_MEMBERSHIP_UNPROVED,
            RawFailureEvidenceKind.TERMINAL_MISSING_SOURCE_COORDINATES,
        }:
            return ArtifactSupportStatus.UNKNOWN
        if self in {
            RawFailureEvidenceKind.DEFERRED_HOT_JSONL_CAPTURE,
            RawFailureEvidenceKind.DEFERRED_CLAUDE_CODE_PARTIAL_JSONL,
            RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER,
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
        if self is RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER:
            return "resolution"
        return "deferred" if self.value in RAW_FAILURE_DEFERRED_EVIDENCE_KINDS else "terminal"


#: Admitted only in part: a stable JSONL capture whose last record is
#: truncated, whose raw carries a deferred partial-decode carrier. Its
#: complete records are admitted; the unterminated tail is not a record yet,
#: and a later observation of the grown file admits it.
PARTIAL_TRUNCATED_TAIL = "truncated_tail"


@dataclass(frozen=True, slots=True)
class PartialAdmission:
    """What an admitted source left out, in typed, countable terms."""

    reason: str
    #: Records admitted from the complete prefix.
    complete_record_count: int
    #: Byte offset where the admitted prefix ends and the left-out tail begins.
    complete_prefix_bytes: int
    #: Size of the acquired bytes, tail included.
    source_bytes: int


RAW_FAILURE_TRUSTED_PROVENANCE = "worker-disposition-v1"
RAW_FAILURE_VALIDATION_FAILURE_KINDS = frozenset(
    {
        RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT.value,
        RawFailureEvidenceKind.TERMINAL_UNKNOWN_JSON_DECODE.value,
    }
)


class RetainedRawDecodeRefusalError(ValueError):
    """Current durable decode evidence refuses this retained input permanently."""

    def __init__(self, raw_id: str, kind: RawFailureEvidenceKind, diagnostic: str) -> None:
        if kind.value not in RAW_FAILURE_VALIDATION_FAILURE_KINDS:
            raise ValueError("retained decode refusal requires terminal decode evidence")
        self.raw_id = raw_id
        self.kind = kind
        super().__init__(diagnostic)


def raw_failure_classification_reason(
    *,
    diagnostic: str | None,
    evidence_ref: str | None,
    outcome_code: str,
    remediation: str | None,
    retryable: bool | None,
    trusted_validation_failure: bool,
) -> str:
    """Encode the typed carrier, including proof for a validation failure."""
    payload: dict[str, object] = {
        "diagnostic": diagnostic,
        "evidence_ref": evidence_ref,
        "outcome_code": outcome_code,
        "remediation": remediation,
        "retryable": retryable,
    }
    if trusted_validation_failure:
        payload["provenance"] = RAW_FAILURE_TRUSTED_PROVENANCE
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def has_trusted_raw_failure_provenance(
    classification_reason: object,
    *,
    artifact_kind: RawFailureEvidenceKind,
    outcome_code: object,
) -> bool:
    """Check the structural worker receipt required for corrupt evidence."""
    if artifact_kind.value not in RAW_FAILURE_VALIDATION_FAILURE_KINDS:
        return False
    if str(outcome_code) != "corrupt_input":
        return False
    if not isinstance(classification_reason, str):
        return False
    try:
        payload = json.loads(classification_reason)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("provenance") == RAW_FAILURE_TRUSTED_PROVENANCE


def raw_failure_outcome_code(classification_reason: object) -> object:
    """Read the typed outcome code from a structured failure carrier."""
    if not isinstance(classification_reason, str):
        return None
    try:
        payload = json.loads(classification_reason)
    except (TypeError, ValueError):
        return None
    return payload.get("outcome_code") if isinstance(payload, dict) else None


def validated_raw_failure_evidence_kind(
    artifact_kind: object,
    support_status: object,
    *,
    validation_failed: bool,
    classification_reason: object = None,
    outcome_code: object = None,
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
    if validation_failed and not has_trusted_raw_failure_provenance(
        classification_reason,
        artifact_kind=evidence_kind,
        outcome_code=outcome_code,
    ):
        return None
    return evidence_kind


RAW_FAILURE_EVIDENCE_KINDS = frozenset(kind.value for kind in RawFailureEvidenceKind)
RAW_FAILURE_DEFERRED_EVIDENCE_KINDS = frozenset(
    {
        RawFailureEvidenceKind.DEFERRED_HOT_JSONL_CAPTURE.value,
        RawFailureEvidenceKind.DEFERRED_CLAUDE_CODE_PARTIAL_JSONL.value,
        RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER.value,
    }
)
# Only frontier conflicts authorize retained-raw replay. Hot captures remain
# deferred until a complete source observation arrives; replaying their
# truncated blob would advance the cursor past the record that later bytes
# complete.
RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS = frozenset({RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER.value})
# Every deferred raw-failure carrier represents a partial decode. Consumers
# selecting retry authority must validate this companion field as well as the
# closed kind, or contradictory rows can authorize replay.
RAW_FAILURE_DEFERRED_SUPPORT_STATUS = ArtifactSupportStatus.PARTIAL_DECODE.value
RAW_FAILURE_EVIDENCE_SUPPORT_STATUS_PAIRS = tuple(
    sorted((kind.value, kind.support_status.value) for kind in RawFailureEvidenceKind)
)
RAW_FAILURE_LIFECYCLE_EVIDENCE_SUPPORT_STATUS_PAIRS = tuple(
    sorted(
        (kind.value, kind.support_status.value)
        for kind in RawFailureEvidenceKind
        if kind.lifecycle in {"deferred", "terminal"}
    )
)
RAW_FAILURE_TERMINAL_EVIDENCE_KINDS = frozenset(
    {
        RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT.value,
        RawFailureEvidenceKind.TERMINAL_UNKNOWN_JSON_DECODE.value,
        RawFailureEvidenceKind.TERMINAL_UNKNOWN_EXPORT_NO_SESSION.value,
        RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value,
        RawFailureEvidenceKind.TERMINAL_MISSING_SOURCE_COORDINATES.value,
        RawFailureEvidenceKind.TERMINAL_MISSING_PROFILE_IDENTITY.value,
        RawFailureEvidenceKind.TERMINAL_RETAINED_ZIP_MEMBERSHIP_UNPROVED.value,
    }
)
RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS = tuple(
    sorted((kind.value, kind.support_status.value) for kind in RawFailureEvidenceKind if kind.lifecycle == "terminal")
)


def terminal_carrier_overwrite_predicate() -> str:
    """SQL predicate that is true when an upsert would take back a terminal carrier.

    A raw refused with a typed terminal outcome owns its ``raw_artifacts``
    row: that carrier is the only durable statement that the bytes will
    never become a session, and the raw-frontier gate reads it to settle
    the path. Re-observing the same coordinate re-derives an ordinary path
    classification; only another failure-evidence write may change a
    carrier's kind. Written against the ``raw_artifacts``/``excluded``
    aliases an ``ON CONFLICT`` clause exposes.

    Scoped to the SAME raw. The observation id is stable by source/path/index,
    so when a path with a terminal carrier is replaced by different bytes, the
    NEW raw's ordinary classification conflicts with the OLD raw's carrier.
    Comparing kinds alone suppressed that update even though the timestamp and
    rowid ordering above had already accepted the newer raw, leaving the newest
    coordinate with no artifact receipt of its own -- which the raw-frontier
    check reports as ``source_raws_without_accepted_head`` and which blocks
    catch-up. A terminal refusal is a statement about the bytes it refused, not
    a permanent claim on the coordinate.
    """
    terminal = ", ".join(f"'{kind}'" for kind in sorted(RAW_FAILURE_TERMINAL_EVIDENCE_KINDS))
    evidence = ", ".join(f"'{kind}'" for kind in sorted(RAW_FAILURE_EVIDENCE_KINDS))
    return (
        "(raw_artifacts.raw_id = excluded.raw_id"
        f" AND raw_artifacts.artifact_kind IN ({terminal})"
        f" AND excluded.artifact_kind NOT IN ({evidence}))"
    )


__all__ = [
    "PARTIAL_TRUNCATED_TAIL",
    "PartialAdmission",
    "RAW_FAILURE_DEFERRED_EVIDENCE_KINDS",
    "RAW_FAILURE_DEFERRED_SUPPORT_STATUS",
    "RAW_FAILURE_EVIDENCE_KINDS",
    "RAW_FAILURE_EVIDENCE_SUPPORT_STATUS_PAIRS",
    "RAW_FAILURE_LIFECYCLE_EVIDENCE_SUPPORT_STATUS_PAIRS",
    "RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS",
    "RAW_FAILURE_TERMINAL_EVIDENCE_KINDS",
    "RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS",
    "RAW_FAILURE_TRUSTED_PROVENANCE",
    "RAW_FAILURE_VALIDATION_FAILURE_KINDS",
    "RawFailureEvidenceKind",
    "RetainedRawDecodeRefusalError",
    "has_trusted_raw_failure_provenance",
    "terminal_carrier_overwrite_predicate",
    "raw_failure_classification_reason",
    "raw_failure_outcome_code",
    "validated_raw_failure_evidence_kind",
]


def retained_raw_decode_refusal_from_row(
    raw_id: str, row: Sequence[object] | None
) -> RetainedRawDecodeRefusalError | None:
    """Validate the canonical current parser/artifact receipt projection."""
    if row is None:
        return None
    kind = validated_raw_failure_evidence_kind(
        row[0],
        row[2],
        validation_failed=row[3] == "failed",
        classification_reason=row[4],
        outcome_code=raw_failure_outcome_code(row[4]),
    )
    return None if kind is None else RetainedRawDecodeRefusalError(raw_id, kind, str(row[1]))


class RetainedRawDependencyRefusalError(ValueError):
    """A subject still requires an input with current durable decode refusal."""

    def __init__(
        self, subject_raw_id: str, logical_source_keys: tuple[str, ...], dependency: RetainedRawDecodeRefusalError
    ) -> None:
        self.subject_raw_id = subject_raw_id
        self.logical_source_keys = logical_source_keys
        self.dependency = dependency
        super().__init__(f"{subject_raw_id} requires refused retained input {dependency.raw_id}")


class CohortMembershipRefusalError(Exception):
    """One selector member cannot be resolved for one logical source key.

    This outcome names the original member and logical key that cannot be
    prepared. A callback owner records it while publishing healthy independent
    keys. A strict owner physically closes preparation and raises it before
    publishing (polylogue-163ku).
    """

    def __init__(self, logical_source_key: str, raw_id: str, reason: str) -> None:
        super().__init__(f"membership {raw_id}:{logical_source_key} refused: {reason}")
        self.logical_source_key = logical_source_key
        self.raw_id = raw_id
        self.reason = reason
