"""ReceiptState/PublicationObligation vocabulary invariants."""

from __future__ import annotations

import dataclasses

from polylogue.sinex.models import (
    ObligationStatus,
    PublicationMode,
    PublicationObligation,
    ReceiptState,
)


def test_only_documented_states_unlock_progress() -> None:
    """Mirrors sinex-r6d.11: PersistedConfirmed/DurableDebt/SpoolAcceptedLossless

    unlock progress; a bare RawAccepted or an explicit Rejected must not --
    that is the exact bug class (mpsc/NATS-publish acceptance mistaken for a
    durable commit) r6d.11 exists to close off.
    """
    unlocking = {ReceiptState.PERSISTED_CONFIRMED, ReceiptState.DURABLE_DEBT, ReceiptState.SPOOL_ACCEPTED_LOSSLESS}
    for state in ReceiptState:
        assert state.unlocks_progress() == (state in unlocking), state


def test_request_id_is_deterministic_and_distinguishes_every_key_component() -> None:
    """The transport idempotency key must change if ANY of the 4 identity
    components changes -- otherwise two distinct revisions could collide on
    the same request_id and a real transport would treat them as duplicates.
    """
    base = PublicationObligation(
        object_id="claude-code-session:abc",
        protocol_version="polylogue.material-protocol/v1",
        revision_id="rev-1",
        manifest_digest="digest-1",
        mode=PublicationMode.MIRROR,
        status=ObligationStatus.PENDING,
        attempt_count=0,
        last_attempt_at_ms=None,
        last_receipt_state=None,
        last_error=None,
        created_at_ms=1,
        updated_at_ms=1,
        retired_at_ms=None,
    )
    same = dataclasses.replace(base)
    assert base.request_id == same.request_id

    variants = (
        dataclasses.replace(base, object_id="codex-session:xyz"),
        dataclasses.replace(base, protocol_version="polylogue.material-protocol/v2"),
        dataclasses.replace(base, revision_id="rev-2"),
        dataclasses.replace(base, manifest_digest="digest-2"),
    )
    request_ids = {base.request_id}
    for mutated in variants:
        assert mutated.request_id not in request_ids, mutated
        request_ids.add(mutated.request_id)
