"""Closed terminal machine-receipt boundaries."""

from __future__ import annotations

import pytest

from polylogue.operations.machine_receipts import (
    IngestHistoricalReceipt,
    IngestInputHistoricalReceipt,
    IngestInputPageHistoricalReceipt,
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
