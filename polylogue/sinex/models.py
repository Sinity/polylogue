"""Typed vocabulary for durable Sinex publication and convergence.

The source-tier ledger stores both the obligation metadata and the exact
manifest/segment bytes needed to redrive it after a process crash.  Receipt
state, rather than successful invocation of a transport method, is the only
thing that may unlock primary-mode projection progress.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
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
class PublicationAcquisition:
    """Scope of the acquired representation, independently of its CAS digest."""

    kind: str = "unknown"
    byte_start: int | None = None
    byte_end: int | None = None
    addressing_mode: str | None = None
    tables: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"full_snapshot", "byte_window", "logical_snapshot", "unknown"}:
            raise ValueError("publication_acquisition_kind_invalid")
        if self.kind == "byte_window":
            if (
                type(self.byte_start) is not int
                or type(self.byte_end) is not int
                or not 0 <= self.byte_start <= self.byte_end
            ):
                raise ValueError("publication_acquisition_window_invalid")
        elif self.byte_start is not None or self.byte_end is not None:
            raise ValueError("publication_acquisition_window_unexpected")
        if self.addressing_mode is not None and not isinstance(self.addressing_mode, str):
            raise ValueError("publication_acquisition_address_invalid")
        if self.tables is not None and (
            self.kind != "logical_snapshot" or any(not isinstance(item, str) for item in self.tables)
        ):
            raise ValueError("publication_acquisition_tables_invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "byte_window": None if self.byte_start is None else {"start": self.byte_start, "end": self.byte_end},
            "addressing_mode": self.addressing_mode,
            "tables": None if self.tables is None else list(self.tables),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> PublicationAcquisition:
        if set(value) != {"kind", "byte_window", "addressing_mode", "tables"}:
            raise ValueError("publication_acquisition_fields_invalid")
        kind = value["kind"]
        window = value["byte_window"]
        address = value["addressing_mode"]
        tables = value["tables"]
        if not isinstance(kind, str) or (address is not None and not isinstance(address, str)):
            raise ValueError("publication_acquisition_identity_invalid")
        start = end = None
        if window is not None:
            if not isinstance(window, dict) or set(window) != {"start", "end"}:
                raise ValueError("publication_acquisition_window_invalid")
            start, end = window["start"], window["end"]
            if type(start) is not int or type(end) is not int:
                raise ValueError("publication_acquisition_window_invalid")
        if tables is not None and (not isinstance(tables, list) or any(not isinstance(item, str) for item in tables)):
            raise ValueError("publication_acquisition_tables_invalid")
        return cls(
            kind=kind,
            byte_start=start,
            byte_end=end,
            addressing_mode=address,
            tables=None if tables is None else tuple(tables),
        )


@dataclass(frozen=True, slots=True)
class PublicationOriginal:
    """Exact retained Source bytes published separately from normalized records."""

    role: str
    source_ref: str
    blob_hash: str
    size_bytes: int
    media_type: str
    acquisition: PublicationAcquisition = field(default_factory=PublicationAcquisition)

    def __post_init__(self) -> None:
        if self.role not in {"provider_native", "attachment"}:
            raise ValueError("publication_original_role_invalid")
        if type(self.source_ref) is not str or not self.source_ref.strip():
            raise ValueError("publication_original_source_ref_invalid")
        if (
            type(self.blob_hash) is not str
            or len(self.blob_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.blob_hash)
        ):
            raise ValueError("publication_original_hash_invalid")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("publication_original_size_invalid")
        if type(self.media_type) is not str or not self.media_type.strip():
            raise ValueError("publication_original_media_type_invalid")
        if not isinstance(self.acquisition, PublicationAcquisition):
            raise ValueError("publication_original_acquisition_invalid")
        if self.acquisition.kind == "byte_window" and (
            self.acquisition.byte_start is None
            or self.acquisition.byte_end is None
            or self.acquisition.byte_end - self.acquisition.byte_start != self.size_bytes
        ):
            raise ValueError("publication_original_window_size_mismatch")


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
    originals: tuple[PublicationOriginal, ...] = ()
    normalizer_version: str = "polylogue.material-protocol/v1;semantics=8;canonicalizer=2"

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

    def as_dict(self) -> dict[str, object]:
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
    "PublicationAcquisition",
    "PublicationOriginal",
    "PublicationPayload",
    "PublicationReceipt",
    "PublicationStatus",
    "ReceiptState",
    "RETRYABLE_OBLIGATION_STATUSES",
    "TERMINAL_OBLIGATION_STATUSES",
]
