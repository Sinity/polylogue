"""Receipt and idempotency vocabulary invariants."""

from __future__ import annotations

import dataclasses

from polylogue.sinex.models import (
    ObligationStatus,
    PublicationMode,
    PublicationObligation,
    PublicationStatus,
    ReceiptState,
)


def _obligation() -> PublicationObligation:
    return PublicationObligation(
        object_id="claude-code-session:s1",
        protocol_version="polylogue.material-protocol/v1",
        revision_id="rev-1",
        manifest_digest="a" * 64,
        mode=PublicationMode.MIRROR,
        status=ObligationStatus.PENDING,
        attempt_count=0,
        last_attempt_at_ms=None,
        last_receipt_state=None,
        last_error=None,
        created_at_ms=1,
        updated_at_ms=1,
        retired_at_ms=None,
        next_attempt_at_ms=1,
    )


def test_only_documented_durable_receipts_unlock_primary_progress() -> None:
    allowed = {
        ReceiptState.PERSISTED_CONFIRMED,
        ReceiptState.DURABLE_DEBT,
        ReceiptState.SPOOL_ACCEPTED_LOSSLESS,
    }
    assert {state for state in ReceiptState if state.unlocks_progress()} == allowed


def test_request_id_is_stable_and_uses_every_identity_component() -> None:
    base = _obligation()
    assert dataclasses.replace(base).request_id == base.request_id
    variants = (
        dataclasses.replace(base, object_id="codex-session:s1"),
        dataclasses.replace(base, protocol_version="polylogue.material-protocol/v2"),
        dataclasses.replace(base, revision_id="rev-2"),
        dataclasses.replace(base, manifest_digest="b" * 64),
    )
    assert len({base.request_id, *(item.request_id for item in variants)}) == 5


def test_status_serialization_contains_only_bounded_operator_fields() -> None:
    status = PublicationStatus(
        mode=PublicationMode.PRIMARY,
        total=3,
        pending=1,
        confirmed=1,
        durable_debt=1,
        blocking=1,
        last_receipt_state=ReceiptState.DURABLE_DEBT,
        last_error_code="transport_timeout",
    )
    assert status.as_dict() == {
        "mode": "primary",
        "total": 3,
        "pending": 1,
        "publishing": 0,
        "confirmed": 1,
        "durable_debt": 1,
        "rejected": 0,
        "retry_due": 0,
        "blocking": 1,
        "active_lag": 0,
        "oldest_active_age_ms": None,
        "last_receipt_state": "durable_debt",
        "last_error_code": "transport_timeout",
    }
