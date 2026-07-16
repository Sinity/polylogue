"""Typed vocabulary for durable Sinex publication and convergence.

The source-tier ledger stores both the obligation metadata and the exact
manifest/segment bytes needed to redrive it after a process crash.  Receipt
state, rather than successful invocation of a transport method, is the only
thing that may unlock primary-mode projection progress.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum


class PublicationMode(str, Enum):
    """Sinex-backed authority profile (polylogue-303r / polylogue-303r.2)."""

    OFF = "off"
    MIRROR = "mirror"
    PRIMARY = "primary"

    @classmethod
    def from_string(cls, value: str | PublicationMode) -> PublicationMode:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class ObligationStatus(str, Enum):
    """Lifecycle of one durable publication-obligation row."""

    PENDING = "pending"
    PUBLISHING = "publishing"
    CONFIRMED = "confirmed"
    DURABLE_DEBT = "durable_debt"
    REJECTED = "rejected"

    @classmethod
    def from_string(cls, value: str | ObligationStatus) -> ObligationStatus:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


TERMINAL_OBLIGATION_STATUSES = frozenset({ObligationStatus.CONFIRMED, ObligationStatus.REJECTED})
RETRYABLE_OBLIGATION_STATUSES = frozenset(
    {ObligationStatus.PENDING, ObligationStatus.PUBLISHING, ObligationStatus.DURABLE_DEBT}
)


class ReceiptState(str, Enum):
    """Durable-emission outcome vocabulary shared with the transport."""

    RAW_ACCEPTED = "raw_accepted"
    PERSISTED_CONFIRMED = "persisted_confirmed"
    DURABLE_DEBT = "durable_debt"
    SPOOL_ACCEPTED_LOSSLESS = "spool_accepted_lossless"
    REJECTED = "rejected"

    def unlocks_progress(self) -> bool:
        """Return whether this receipt may release primary-mode projection."""
        return self in (
            ReceiptState.PERSISTED_CONFIRMED,
            ReceiptState.DURABLE_DEBT,
            ReceiptState.SPOOL_ACCEPTED_LOSSLESS,
        )


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    """One transport attempt's outcome, keyed by the obligation request id."""

    request_id: str
    state: ReceiptState
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PublicationPayload:
    """Exact, restart-safe material bytes staged with an obligation.

    ``segments`` is an ordered tuple rather than a mutable mapping so the
    payload can cross the process-pool/async boundary without aliasing.  The
    segment name is the transport-visible protocol filename, not an invented
    database identifier.
    """

    object_id: str
    protocol_version: str
    revision_id: str
    manifest_digest: str
    manifest_bytes: bytes
    segments: tuple[tuple[str, bytes], ...]

    @property
    def segment_bytes(self) -> dict[str, bytes]:
        return dict(self.segments)

    @property
    def size_bytes(self) -> int:
        return len(self.manifest_bytes) + sum(len(payload) for _, payload in self.segments)


@dataclass(frozen=True, slots=True)
class PublicationObligation:
    """One durable ``sinex_publication_obligations`` row."""

    object_id: str
    protocol_version: str
    revision_id: str
    manifest_digest: str
    mode: PublicationMode
    status: ObligationStatus
    attempt_count: int
    last_attempt_at_ms: int | None
    last_receipt_state: ReceiptState | None
    last_error: str | None
    created_at_ms: int
    updated_at_ms: int
    retired_at_ms: int | None
    next_attempt_at_ms: int | None = None

    @property
    def request_id(self) -> str:
        """Deterministic transport idempotency key for this exact revision."""
        fields = (self.object_id, self.protocol_version, self.revision_id, self.manifest_digest)
        framed = json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return f"polylogue-publication-v1:{hashlib.sha256(framed).hexdigest()}"

    @property
    def progress_unlocked(self) -> bool:
        return self.last_receipt_state is not None and self.last_receipt_state.unlocks_progress()


@dataclass(frozen=True, slots=True)
class PublicationStatus:
    """Secret-safe status snapshot for daemon/operator observability."""

    mode: PublicationMode
    total: int = 0
    pending: int = 0
    publishing: int = 0
    confirmed: int = 0
    durable_debt: int = 0
    rejected: int = 0
    retry_due: int = 0
    blocking: int = 0
    active_lag: int = 0
    oldest_active_age_ms: int | None = None
    last_receipt_state: ReceiptState | None = None
    last_error_code: str | None = None

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "mode": self.mode.value,
            "total": self.total,
            "pending": self.pending,
            "publishing": self.publishing,
            "confirmed": self.confirmed,
            "durable_debt": self.durable_debt,
            "rejected": self.rejected,
            "retry_due": self.retry_due,
            "blocking": self.blocking,
            "active_lag": self.active_lag,
            "oldest_active_age_ms": self.oldest_active_age_ms,
            "last_receipt_state": self.last_receipt_state.value if self.last_receipt_state else None,
            "last_error_code": self.last_error_code,
        }


__all__ = [
    "ObligationStatus",
    "PublicationMode",
    "PublicationObligation",
    "PublicationPayload",
    "PublicationReceipt",
    "PublicationStatus",
    "ReceiptState",
    "RETRYABLE_OBLIGATION_STATUSES",
    "TERMINAL_OBLIGATION_STATUSES",
]
