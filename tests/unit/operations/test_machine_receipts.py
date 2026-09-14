"""Closed terminal machine-receipt boundaries."""

from __future__ import annotations

import pytest

from polylogue.operations.machine_receipts import (
    IngestHistoricalReceipt,
    IngestInputHistoricalReceipt,
    IngestInputPageHistoricalReceipt,
    IngestRefusedMembershipHistorical,
    IngestTerminalSummaryHistorical,
    decode_machine_receipt,
)


def test_ingest_history_keeps_known_unresolved_ids_in_bounded_input_page() -> None:
    """Known ids stay historical evidence; no reader may regenerate them later."""

    item = IngestInputHistoricalReceipt(
        source_item_id="source-item:fixture",
        logical_coordinate="fixture.json",
        denominator=2,
        raw_ids=["raw:complete", "raw:unresolved"],
        unresolved_raw_ids=["raw:unresolved"],
    )
    page = IngestInputPageHistoricalReceipt.from_items(0, [item])
    history = IngestHistoricalReceipt(
        source_generation_id="generation:fixture",
        final_sequence=1,
        input_count=1,
        input_pages=[page],
        summary=IngestTerminalSummaryHistorical(
            enumeration_complete=True,
            source_complete=False,
            confirmed_raw_count=1,
            unresolved_raw_count=1,
            profile_targets_observed=0,
        ),
    )

    replayed = decode_machine_receipt(history.model_dump(mode="json"))

    assert isinstance(replayed, IngestHistoricalReceipt)
    assert replayed.input_pages[0].items[0].unresolved_raw_ids == ["raw:unresolved"]


def test_ingest_history_refuses_unexplained_missing_attribution() -> None:
    """A terminal receipt cannot turn missing facts into a live-tier lookup."""

    with pytest.raises(ValueError, match="explicit unknown reason"):
        IngestInputHistoricalReceipt(
            source_item_id="source-item:fixture",
            logical_coordinate="fixture.json",
            denominator=0,
            raw_ids=None,
        )


def test_machine_receipt_decoder_rejects_generic_domain_mapping() -> None:
    """Arbitrary ``domain_receipt`` JSON is not a new audit payload protocol."""

    with pytest.raises(ValueError, match="not recognized"):
        decode_machine_receipt({"kind": "anything", "private": "not-a-receipt"})


def _one_input_page() -> IngestInputPageHistoricalReceipt:
    return IngestInputPageHistoricalReceipt.from_items(
        0,
        [
            IngestInputHistoricalReceipt(
                source_item_id="source-item:fixture",
                logical_coordinate="fixture.json",
                denominator=1,
                raw_ids=["raw:complete"],
                unresolved_raw_ids=[],
            )
        ],
    )


def _history(**summary: object) -> IngestHistoricalReceipt:
    base: dict[str, object] = {
        "enumeration_complete": True,
        "source_complete": False,
        "confirmed_raw_count": 1,
        "unresolved_raw_count": 0,
        "profile_targets_observed": 0,
    }
    base.update(summary)
    return IngestHistoricalReceipt(
        source_generation_id="generation:fixture",
        final_sequence=1,
        input_count=1,
        input_pages=[_one_input_page()],
        summary=IngestTerminalSummaryHistorical(**base),  # type: ignore[arg-type]
    )


def test_refused_memberships_are_counted_and_named_in_the_terminal_receipt() -> None:
    """A per-key cohort refusal survives serialization as a counted fact.

    Anti-vacuity: polylogue-163ku's fix makes one unparseable selector member
    skip its logical key instead of fencing the generation. A skip that nothing
    records would trade a loud failure for a silent one, so the terminal receipt
    the caller reads must carry both the count and the reason. Drop
    ``refused_membership_count``/``refused_memberships`` from
    ``IngestTerminalSummaryHistorical`` and this test fails at construction; keep
    the fields but stop populating them in ``daemon_ingest.historical_receipt``
    and the count reaching a reader is zero while keys were in fact dropped.
    """
    history = _history(
        refused_membership_count=1,
        refused_memberships=[
            IngestRefusedMembershipHistorical(
                logical_source_key="codex-session:unparseable",
                raw_id="raw:head",
                reason="selector member did not parse: synthetic unparseable retained payload",
            )
        ],
    )

    replayed = decode_machine_receipt(history.model_dump(mode="json"))

    assert isinstance(replayed, IngestHistoricalReceipt)
    assert replayed.summary.refused_membership_count == 1
    refused = replayed.summary.refused_memberships[0]
    assert refused.logical_source_key == "codex-session:unparseable"
    assert refused.raw_id == "raw:head"
    assert "synthetic unparseable retained payload" in refused.reason


def test_refused_memberships_cannot_coexist_with_a_source_complete_claim() -> None:
    """Refusing a key and claiming the source is complete is not a legal receipt.

    Anti-vacuity: without this validator a future caller could count refusals
    and still report ``source_complete=True``, reinstating exactly the
    refusal-reported-as-success shape polylogue-u1ww0 was filed for. Remove the
    validator and this ``pytest.raises`` stops raising.
    """
    with pytest.raises(ValueError, match="source-complete while memberships were refused"):
        _history(source_complete=True, refused_membership_count=1)


def test_refusal_count_may_not_understate_its_own_enumeration() -> None:
    """The count is a denominator, never smaller than the names it carries.

    Anti-vacuity: the enumerated list is capped at 256 entries, so the count may
    legitimately exceed it -- but a count *below* it would let a reader under-report
    dropped keys. Remove the check and this ``pytest.raises`` stops raising.
    """
    with pytest.raises(ValueError, match="refusal count is smaller"):
        _history(
            refused_membership_count=0,
            refused_memberships=[
                IngestRefusedMembershipHistorical(
                    logical_source_key="codex-session:unparseable",
                    raw_id="raw:head",
                    reason="selector member did not parse",
                )
            ],
        )
