"""Evidence-backed attachment availability for every read surface.

Availability is deliberately independent from acquisition status and from a
filesystem path.  A blob is available only when the selected CAS contains
bytes whose SHA-256 is the recorded attachment hash.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class AttachmentAvailabilityState(StrEnum):
    AVAILABLE = "available"
    UNFETCHED = "unfetched"
    MISSING = "missing"
    HASH_MISMATCH = "hash-mismatch"
    UNAUTHORIZED = "unauthorized"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AttachmentAvailability:
    state: AttachmentAvailabilityState
    reason: str
    can_fetch: bool
    generation_id: str | None = None

    @property
    def available(self) -> bool:
        return self.state is AttachmentAvailabilityState.AVAILABLE


def resolve_attachment_availability(
    *,
    blob_hash: bytes | str | None,
    acquisition_status: str | None,
    verify: Callable[[str], bool],
    exists: Callable[[str], bool] | None = None,
    authorized: bool = True,
    generation_id: str | None = None,
    expected_generation_id: str | None = None,
) -> AttachmentAvailability:
    """Resolve availability without returning a path or touching a surface."""
    if expected_generation_id is not None and generation_id != expected_generation_id:
        return AttachmentAvailability(
            AttachmentAvailabilityState.UNKNOWN,
            "wrong-generation",
            False,
            generation_id,
        )
    if not authorized:
        return AttachmentAvailability(
            AttachmentAvailabilityState.UNAUTHORIZED,
            "byte-fetch-not-authorized",
            False,
            generation_id,
        )
    if blob_hash is None:
        if acquisition_status == "unfetched":
            return AttachmentAvailability(
                AttachmentAvailabilityState.UNFETCHED,
                "bytes-not-requested",
                False,
                generation_id,
            )
        return AttachmentAvailability(
            AttachmentAvailabilityState.UNKNOWN,
            "no-canonical-blob-identity",
            False,
            generation_id,
        )
    hash_hex = blob_hash.hex() if isinstance(blob_hash, bytes) else str(blob_hash)
    try:
        readable = bool(verify(hash_hex))
    except (OSError, ValueError):
        readable = False
    if readable:
        return AttachmentAvailability(
            AttachmentAvailabilityState.AVAILABLE,
            "readable-and-hash-valid",
            True,
            generation_id,
        )
    # A recorded identity with no readable object is materially different from
    # an attachment that was never fetched; retain that evidence explicitly.
    if exists is not None:
        try:
            physical = bool(exists(hash_hex))
        except (OSError, ValueError):
            physical = False
    else:
        physical = False
    if physical:
        return AttachmentAvailability(
            AttachmentAvailabilityState.HASH_MISMATCH,
            "physical-bytes-hash-mismatch",
            False,
            generation_id,
        )
    reason = "missing-physical-bytes" if acquisition_status == "acquired" else "blob-not-readable"
    state = (
        AttachmentAvailabilityState.MISSING
        if reason == "missing-physical-bytes"
        else AttachmentAvailabilityState.UNKNOWN
    )
    return AttachmentAvailability(state, reason, False, generation_id)
