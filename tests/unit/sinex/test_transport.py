"""LocalReferenceTransport contract fidelity + NullTransport hard-fail."""

from __future__ import annotations

import pytest

from polylogue.sinex.models import ReceiptState
from polylogue.sinex.transport import LocalReferenceTransport, NullTransport, TransportUsedInOffModeError


async def test_null_transport_never_silently_no_ops() -> None:
    """Off mode's zero-transport-work guarantee: an accidental call is loud,
    not a graceful skip -- silent no-ops are exactly what let a real bug
    (transport wired despite off mode) go unnoticed.
    """
    transport = NullTransport()
    with pytest.raises(TransportUsedInOffModeError):
        await transport.publish_revision(request_id="req-1", manifest_bytes=b"{}", segment_bytes={})


async def test_local_reference_transport_default_confirms() -> None:
    transport = LocalReferenceTransport()
    receipt = await transport.publish_revision(request_id="req-1", manifest_bytes=b"{}", segment_bytes={"a": b"x"})
    assert receipt.state is ReceiptState.PERSISTED_CONFIRMED
    assert transport.call_count("req-1") == 1


async def test_local_reference_transport_is_idempotent_by_request_id_once_confirmed() -> None:
    """A retried publish for an ALREADY-confirmed request_id must not perform

    (or record) a second underlying attempt -- this is the exact property a
    real Sinex transport must have too (design: "same-revision retry is
    idempotent").
    """
    transport = LocalReferenceTransport()
    first = await transport.publish_revision(request_id="req-1", manifest_bytes=b"{}", segment_bytes={})
    second = await transport.publish_revision(request_id="req-1", manifest_bytes=b"{}", segment_bytes={})
    assert first == second
    assert transport.call_count("req-1") == 1

    # A DIFFERENT request_id is a fully independent attempt.
    await transport.publish_revision(request_id="req-2", manifest_bytes=b"{}", segment_bytes={})
    assert transport.call_count("req-1") == 1
    assert transport.call_count("req-2") == 1
    assert transport.call_count() == 2


async def test_local_reference_transport_fault_injection_does_not_cache_non_unlocking_states() -> None:
    """A non-unlocking outcome (RAW_ACCEPTED) must be retried for real on the

    next attempt, unlike a confirmed outcome which short-circuits.
    """
    attempts: list[int] = []

    def fault_fn(request_id: str, attempt_number: int) -> ReceiptState | None:
        attempts.append(attempt_number)
        return ReceiptState.RAW_ACCEPTED if attempt_number == 1 else None

    transport = LocalReferenceTransport(fault_fn=fault_fn)
    first = await transport.publish_revision(request_id="req-1", manifest_bytes=b"{}", segment_bytes={})
    assert first.state is ReceiptState.RAW_ACCEPTED
    assert not first.state.unlocks_progress()

    second = await transport.publish_revision(request_id="req-1", manifest_bytes=b"{}", segment_bytes={})
    assert second.state is ReceiptState.PERSISTED_CONFIRMED
    assert attempts == [1, 2]
    assert transport.call_count("req-1") == 2
