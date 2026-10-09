"""The streamed accepted marker consumer accepts the producer's JSON numbers."""

from __future__ import annotations

import io

import pytest

from polylogue.core.enums import Provider, Role
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.revision_backfill import _accepted_marker_request_session_binding
from polylogue.storage.accepted_marker_inputs import (
    AcceptedMarkerInputReference,
    VerifiedAcceptedMarkerPayload,
    _validate_marker_payload,
)
from polylogue.storage.accepted_marker_producer import prepare_accepted_marker_carrier


@pytest.mark.parametrize("integer", [2**64, -(2**64), 2**128])
@pytest.mark.parametrize("cost", [1.0, 0.01, 1e-9])
def test_generated_marker_carrier_keeps_arbitrary_integers_and_float_identity(integer: int, cost: float) -> None:
    session = ParsedSession(
        source_name=Provider.DRIVE,
        provider_session_id="neutral",
        messages=[ParsedMessage(provider_message_id="m", role=Role.USER, text="neutral")],
        pending_drafts=[{"text": "unsent", "token_count": integer}],
        reported_cost_usd=cost,
    )
    binding = _accepted_marker_request_session_binding(session)
    facts = dict.fromkeys(
        ("blob_hash", "provider", "revision_kind", "source_path", "parser_fingerprint", "marker_recipe"), "neutral"
    )
    carrier = prepare_accepted_marker_carrier(
        raw_id="raw", request_facts=facts, request_sessions=lambda: [binding], prepared_sessions=[]
    )
    try:
        raw = b"".join(carrier.verified_chunks())
        reference = AcceptedMarkerInputReference(
            carrier.batch.raw_id, carrier.batch.identity, carrier.batch.payload_sha256
        )
        assert _validate_marker_payload(io.BytesIO(raw), reference) == (0, 0)
        verified = VerifiedAcceptedMarkerPayload(reference, io.BytesIO(raw))
        values = list(verified.iter_items("request_sessions.item"))
        assert values == [binding]
    finally:
        carrier.close()
