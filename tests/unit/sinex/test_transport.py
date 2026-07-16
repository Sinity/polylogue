"""Transport identity, composition, and off-mode contracts."""

from __future__ import annotations

import asyncio

import pytest

from polylogue.sinex.models import ReceiptState
from polylogue.sinex.transport import (
    LocalReferenceTransport,
    NullTransport,
    SinexTransportUnavailableError,
    TransportPayloadConflictError,
    TransportUsedInOffModeError,
    clear_configured_transport_factory,
    register_configured_transport_factory,
    resolve_configured_transport,
)


def test_null_transport_is_a_loud_bug_not_a_noop() -> None:
    with pytest.raises(TransportUsedInOffModeError):
        asyncio.run(NullTransport().publish_revision(request_id="req-1", manifest_bytes=b"{}", segment_bytes={}))


def test_reference_transport_replays_confirmed_request_without_duplicate_call() -> None:
    transport = LocalReferenceTransport()
    first = asyncio.run(
        transport.publish_revision(request_id="req-1", manifest_bytes=b"{}", segment_bytes={"head": b"x"})
    )
    second = asyncio.run(
        transport.publish_revision(request_id="req-1", manifest_bytes=b"{}", segment_bytes={"head": b"x"})
    )
    assert first == second
    assert first.state is ReceiptState.PERSISTED_CONFIRMED
    assert transport.call_count("req-1") == 1


def test_request_id_reuse_with_different_exact_bytes_is_rejected() -> None:
    transport = LocalReferenceTransport()
    asyncio.run(transport.publish_revision(request_id="req-1", manifest_bytes=b"a", segment_bytes={}))
    with pytest.raises(TransportPayloadConflictError):
        asyncio.run(transport.publish_revision(request_id="req-1", manifest_bytes=b"b", segment_bytes={}))


def test_payload_digest_frames_manifest_segment_names_and_bytes() -> None:
    transport = LocalReferenceTransport()
    asyncio.run(
        transport.publish_revision(
            request_id="req-framing",
            manifest_bytes=b"a",
            segment_bytes={"b": b"c"},
        )
    )
    with pytest.raises(TransportPayloadConflictError):
        asyncio.run(
            transport.publish_revision(
                request_id="req-framing",
                manifest_bytes=b"ab",
                segment_bytes={"": b"c"},
            )
        )


def test_deployment_transport_factory_is_explicit_and_resettable() -> None:
    clear_configured_transport_factory()
    with pytest.raises(SinexTransportUnavailableError):
        resolve_configured_transport()
    transport = LocalReferenceTransport()
    register_configured_transport_factory(lambda: transport)
    assert resolve_configured_transport() is transport
    clear_configured_transport_factory()
