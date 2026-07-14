"""Typed vocabulary for the Sinex publication obligation ledger and transport.

``ReceiptState`` mirrors the outcome vocabulary Sinex documents for
``DurableEmissionReceipt`` (sinex-r6d.11): progress may unlock only on
``PERSISTED_CONFIRMED`` or a documented terminal debt/lossless-spool outcome,
never on a bare in-memory accept. ``ObligationStatus`` is the Polylogue-side
publication-obligation lifecycle this package actually owns and persists in
``source.db``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PublicationMode(str, Enum):
    """Sinex-backed authority profile (design: polylogue-303r / 303r.2)."""

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


#: Terminal states: a drain loop must not keep retrying these automatically.
TERMINAL_OBLIGATION_STATUSES = frozenset({ObligationStatus.CONFIRMED, ObligationStatus.REJECTED})

#: Retryable states: durable_debt is terminal-for-this-attempt but explicitly
#: retryable (design: "configured failure is never a no-op").
RETRYABLE_OBLIGATION_STATUSES = frozenset(
    {ObligationStatus.PENDING, ObligationStatus.PUBLISHING, ObligationStatus.DURABLE_DEBT}
)


class ReceiptState(str, Enum):
    """Outcome vocabulary modeled on Sinex's DurableEmissionReceipt (r6d.11).

    Only :meth:`unlocks_progress` states may advance a local projection.
    ``RAW_ACCEPTED`` intentionally does NOT unlock progress -- it models the
    documented failure mode (mpsc/NATS-publish acceptance mistaken for a
    durable commit) that r6d.11 exists to close off.
    """

    RAW_ACCEPTED = "raw_accepted"
    PERSISTED_CONFIRMED = "persisted_confirmed"
    DURABLE_DEBT = "durable_debt"
    SPOOL_ACCEPTED_LOSSLESS = "spool_accepted_lossless"
    REJECTED = "rejected"

    def unlocks_progress(self) -> bool:
        """Whether this receipt state may unlock a local projection advance.

        Mirrors sinex-r6d.11's stated rule: PersistedConfirmed or a
        documented terminal DurableDebt/SpoolAcceptedLossless outcome only.
        RawAccepted (bare mpsc/NATS-publish acceptance) and Rejected never
        unlock progress.
        """
        return self in (
            ReceiptState.PERSISTED_CONFIRMED,
            ReceiptState.DURABLE_DEBT,
            ReceiptState.SPOOL_ACCEPTED_LOSSLESS,
        )


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    """One transport attempt's outcome, keyed by the obligation's request_id."""

    request_id: str
    state: ReceiptState
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PublicationObligation:
    """One durable ``sinex_publication_obligations`` row.

    The 4-tuple ``(object_id, protocol_version, revision_id,
    manifest_digest)`` is both the SQL primary key and the transport
    idempotency key (design: polylogue-303r.2, "idempotent by protocol
    version + stable object revision + manifest digest").
    """

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

    @property
    def request_id(self) -> str:
        """Deterministic transport idempotency key for this exact revision.

        Same revision -> same request_id -> a real transport can de-duplicate
        retried attempts by identity rather than by side-channel bookkeeping.
        """
        return "|".join((self.object_id, self.protocol_version, self.revision_id, self.manifest_digest))


__all__ = [
    "ObligationStatus",
    "PublicationMode",
    "PublicationObligation",
    "PublicationReceipt",
    "ReceiptState",
    "RETRYABLE_OBLIGATION_STATUSES",
    "TERMINAL_OBLIGATION_STATUSES",
]
