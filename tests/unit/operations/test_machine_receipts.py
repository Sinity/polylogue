"""Closed terminal machine-receipt boundaries."""

from __future__ import annotations

import pytest

from polylogue.operations.machine_receipts import (
    IngestHistoricalReceiptV2,
    IngestInputHistoricalReceipt,
    IngestInputPageHistoricalReceipt,
    IngestInputRawMemberHistorical,
    IngestInputRawPageHistoricalReceipt,
    IngestInputRawPagesDigest,
    IngestInsightPageHistoricalReceipt,
    IngestRefusedMembershipHistorical,
    IngestTerminalSummaryHistorical,
    InsightCertifiedCountsHistorical,
    InsightTargetHistoricalReceipt,
    decode_machine_receipt,
    ingest_input_pages_digest,
    ingest_input_raw_pages_digest,
    ingest_insight_pages_digest,
)


def test_streamed_input_raw_pages_preserve_the_authenticated_historical_digest() -> None:
    import hashlib
    import json

    pages = [
        IngestInputRawPageHistoricalReceipt(
            source_item_id="source-item:non-ascii-α",
            ordinal=ordinal,
            raws=[IngestInputRawMemberHistorical(raw_id=f"raw:{ordinal}:α", unresolved=bool(ordinal))],
        )
        for ordinal in range(3)
    ]
    expected = hashlib.sha256(
        json.dumps([page.model_dump(mode="json") for page in pages], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    digest = IngestInputRawPagesDigest()
    for page in pages:
        digest.update(page)
    assert digest.hexdigest() == expected
    assert ingest_input_raw_pages_digest(iter(pages)) == expected
    assert IngestInputRawPagesDigest().hexdigest() == hashlib.sha256(b"[]").hexdigest()


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
    history = IngestHistoricalReceiptV2(
        source_generation_id="generation:fixture",
        final_sequence=1,
        input_count=1,
        input_pages_ref="operation:fixture",
        input_page_count=1,
        input_pages_digest=ingest_input_pages_digest([page]),
        summary=IngestTerminalSummaryHistorical(
            enumeration_complete=True,
            source_complete=False,
            confirmed_raw_count=1,
            unresolved_raw_count=1,
            profile_targets_observed=0,
        ),
    )

    replayed = decode_machine_receipt(history.model_dump(mode="json"))

    assert isinstance(replayed, IngestHistoricalReceiptV2)
    assert replayed.input_pages_digest == ingest_input_pages_digest([page])
    assert page.items[0].unresolved_raw_ids == ["raw:unresolved"]


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


def _history(**summary: object) -> IngestHistoricalReceiptV2:
    base: dict[str, object] = {
        "enumeration_complete": True,
        "source_complete": False,
        "confirmed_raw_count": 1,
        "unresolved_raw_count": 0,
        "profile_targets_observed": 0,
    }
    base.update(summary)
    return IngestHistoricalReceiptV2(
        source_generation_id="generation:fixture",
        final_sequence=1,
        input_count=1,
        input_pages_ref="operation:fixture",
        input_page_count=1,
        input_pages_digest=ingest_input_pages_digest([_one_input_page()]),
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

    assert isinstance(replayed, IngestHistoricalReceiptV2)
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
        # The refusal is named, so the count validator passes and only the
        # source-complete contradiction remains to reject the receipt.
        _history(source_complete=True, refused_membership_count=1, refused_memberships=[_refusal(0)])


def _refusal(index: int) -> IngestRefusedMembershipHistorical:
    return IngestRefusedMembershipHistorical(
        logical_source_key=f"codex-session:{index:05d}",
        raw_id=f"raw:{index:05d}",
        reason="selector member did not parse",
    )


def test_refusal_count_must_equal_the_refusals_it_names() -> None:
    """Every counted refusal is named, inline or in pages.

    Anti-vacuity: allow the count to exceed the inline names (the removed
    256-entry cap) and a receipt claiming 2 refusals while naming 1 decodes.
    """
    with pytest.raises(ValueError, match="refusal count does not match"):
        _history(refused_membership_count=2, refused_memberships=[_refusal(0)])


def test_more_refusals_than_one_page_are_named_in_referenced_pages() -> None:
    from polylogue.operations.machine_receipts import (
        MAX_PAGE_ITEMS,
        IngestRefusalPageHistoricalReceipt,
        ingest_refusal_pages_digest,
    )

    refusals = [_refusal(index) for index in range(MAX_PAGE_ITEMS * 2 + 1)]
    pages = [
        IngestRefusalPageHistoricalReceipt(ordinal=ordinal, refusals=refusals[offset : offset + MAX_PAGE_ITEMS])
        for ordinal, offset in enumerate(range(0, len(refusals), MAX_PAGE_ITEMS))
    ]
    history = _history(
        refused_membership_count=len(refusals),
        refused_membership_pages_ref="operation:fixture",
        refused_membership_page_count=len(pages),
        refused_memberships_digest=ingest_refusal_pages_digest(pages),
    )

    replayed = decode_machine_receipt(history.model_dump(mode="json"))
    assert isinstance(replayed, IngestHistoricalReceiptV2)
    assert replayed.summary.refused_membership_count == len(refusals)
    assert replayed.summary.refused_membership_page_count == 3
    with pytest.raises(ValueError, match="page count does not match"):
        _history(
            refused_membership_count=len(refusals),
            refused_membership_pages_ref="operation:fixture",
            refused_membership_page_count=2,
            refused_memberships_digest=ingest_refusal_pages_digest(pages),
        )


def test_large_ingest_projection_uses_a_page_reference_without_truncating_its_count() -> None:
    from polylogue.operations.machine_receipts import MAX_PAGE_ITEMS, ingest_session_ids_digest

    session_ids = [f"chatgpt:{index:05d}" for index in range(10_001)]
    history = _history(
        parse_projection_known=True,
        processed_session_id_pages_ref="operation:fixture",
        processed_session_id_page_count=(len(session_ids) + MAX_PAGE_ITEMS - 1) // MAX_PAGE_ITEMS,
        processed_session_ids_digest=ingest_session_ids_digest(session_ids),
        changed_session_count=len(session_ids),
    )

    replayed = decode_machine_receipt(history.model_dump(mode="json"))
    assert isinstance(replayed, IngestHistoricalReceiptV2)
    assert replayed.summary.changed_session_count == 10_001
    assert replayed.summary.processed_session_ids == []
    assert replayed.summary.processed_session_id_page_count == 40


def test_large_ingest_projection_rejects_a_missing_page() -> None:
    with pytest.raises(ValueError, match="page count does not match"):
        _history(
            parse_projection_known=True,
            processed_session_id_pages_ref="operation:fixture",
            processed_session_id_page_count=39,
            processed_session_ids_digest="a" * 64,
            changed_session_count=10_001,
        )


def test_ingest_insight_evidence_over_40_pages_uses_an_exact_audit_reference() -> None:
    pages = [
        IngestInsightPageHistoricalReceipt(
            ordinal=ordinal,
            targets=[
                InsightTargetHistoricalReceipt(
                    target_ref=f"session:fixture-{ordinal}",
                    disposition="published",
                    certified_counts=InsightCertifiedCountsHistorical(profiles=1),
                    publication_known_committed=True,
                )
            ],
        )
        for ordinal in range(41)
    ]
    history = IngestHistoricalReceiptV2(
        source_generation_id="generation:fixture",
        final_sequence=1,
        input_count=1,
        input_pages_ref="operation:fixture",
        input_page_count=1,
        input_pages_digest=ingest_input_pages_digest([_one_input_page()]),
        insight_pages_ref="operation:fixture",
        insight_page_count=len(pages),
        insight_pages_digest=ingest_insight_pages_digest(pages),
        summary=IngestTerminalSummaryHistorical(
            enumeration_complete=True,
            source_complete=True,
            confirmed_raw_count=1,
            unresolved_raw_count=0,
            profile_targets_observed=41,
        ),
    )
    replayed = decode_machine_receipt(history.model_dump(mode="json"))
    assert isinstance(replayed, IngestHistoricalReceiptV2)
    assert replayed.insight_page_count == 41
    assert replayed.insight_pages == []


def test_zip_like_input_raw_attribution_crosses_inline_threshold_without_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.operations import machine_receipts

    monkeypatch.setattr(machine_receipts, "MAX_INLINE_RAW_IDS_PER_INPUT", 3)
    raw_page = IngestInputRawPageHistoricalReceipt(
        source_item_id="source-item:zip",
        ordinal=0,
        raws=[IngestInputRawMemberHistorical(raw_id=f"raw:{index}", unresolved=index == 3) for index in range(4)],
    )
    item = IngestInputHistoricalReceipt(
        source_item_id="source-item:zip",
        logical_coordinate="export.zip",
        denominator=4,
        raw_id_pages_ref="operation:fixture",
        raw_id_page_count=1,
        raw_id_count=4,
        unresolved_raw_count=1,
        raw_ids_digest=ingest_input_raw_pages_digest([raw_page]),
    )
    history = IngestHistoricalReceiptV2(
        source_generation_id="generation:fixture",
        final_sequence=1,
        input_count=1,
        input_pages_ref="operation:fixture",
        input_page_count=1,
        input_pages_digest=ingest_input_pages_digest([IngestInputPageHistoricalReceipt.from_items(0, [item])]),
        summary=IngestTerminalSummaryHistorical(
            enumeration_complete=True,
            source_complete=False,
            confirmed_raw_count=3,
            unresolved_raw_count=1,
            profile_targets_observed=0,
        ),
    )
    replayed = decode_machine_receipt(history.model_dump(mode="json"))
    assert isinstance(replayed, IngestHistoricalReceiptV2)
    assert item.raw_id_count == 4
    with pytest.raises(ValueError, match="pages disagree"):
        IngestInputHistoricalReceipt.model_validate({**item.model_dump(mode="json"), "raw_id_count": 5})


@pytest.mark.parametrize(
    ("source_complete", "profile_convergence_complete", "outcome"),
    [
        (True, True, "completed"),
        # Committed rows whose profile/insight convergence stopped on a
        # retryable target: the ingest is not done (#5639 review).
        (True, False, "degraded"),
        (False, True, "degraded"),
        # Written before the fact was recorded: unknowable, left as it was.
        (True, None, "completed"),
        # ...unless the receipt already records incomplete source admission.
        (False, None, "degraded"),
    ],
)
def test_ingest_terminal_outcome_is_completed_only_when_converged(
    source_complete: bool, profile_convergence_complete: bool | None, outcome: str
) -> None:
    """An ingest whose derived convergence stopped is ``degraded``, not ``completed``.

    Anti-vacuity: returning ``completed`` unconditionally from
    ``ingest_terminal_outcome`` (the previous behaviour of
    ``execute_ingest_operation``) turns both degraded rows red.
    """
    from polylogue.operations.machine_receipts import ingest_terminal_outcome

    item = IngestInputHistoricalReceipt(
        source_item_id="source-item:fixture",
        logical_coordinate="fixture.json",
        denominator=1,
        raw_ids=["raw:complete"],
    )
    history = IngestHistoricalReceiptV2(
        source_generation_id="generation:fixture",
        final_sequence=1,
        input_count=1,
        input_pages_ref="operation:fixture",
        input_page_count=1,
        input_pages_digest=ingest_input_pages_digest([IngestInputPageHistoricalReceipt.from_items(0, [item])]),
        summary=IngestTerminalSummaryHistorical(
            enumeration_complete=True,
            source_complete=source_complete,
            confirmed_raw_count=1,
            unresolved_raw_count=0,
            profile_targets_observed=0,
            profile_convergence_complete=profile_convergence_complete,
        ),
    )

    assert ingest_terminal_outcome(decode_machine_receipt(history.model_dump(mode="json"))) == outcome  # type: ignore[arg-type]


@pytest.mark.parametrize("profile_convergence_complete", [True, False])
def test_ingest_result_contract_binds_its_outcome_to_the_receipt(profile_convergence_complete: bool) -> None:
    """The declared ingest result admits ``degraded`` only when the receipt records it.

    Anti-vacuity: restoring the completed-only validator rejects the degraded
    row; dropping the receipt binding accepts the mismatched outcome.
    """
    from polylogue.operations.daemon_protocol import OperationResultContractError, validate_operation_result
    from polylogue.operations.machine_receipts import ingest_terminal_outcome

    item = IngestInputHistoricalReceipt(
        source_item_id="source-item:fixture",
        logical_coordinate="fixture.json",
        denominator=1,
        raw_ids=["raw:complete"],
    )
    history = IngestHistoricalReceiptV2(
        source_generation_id="generation:fixture",
        final_sequence=1,
        input_count=1,
        input_pages_ref="operation:fixture",
        input_page_count=1,
        input_pages_digest=ingest_input_pages_digest([IngestInputPageHistoricalReceipt.from_items(0, [item])]),
        summary=IngestTerminalSummaryHistorical(
            enumeration_complete=True,
            source_complete=True,
            confirmed_raw_count=1,
            unresolved_raw_count=0,
            profile_targets_observed=0,
            profile_convergence_complete=profile_convergence_complete,
        ),
    )
    outcome = ingest_terminal_outcome(history)
    payload = {
        "source_generation_id": "generation:fixture",
        "sequence": 1,
        "historical_receipt": history.model_dump(mode="json"),
    }

    validate_operation_result("ingest", {**payload, "outcome": outcome})
    wrong = "completed" if outcome == "degraded" else "degraded"
    with pytest.raises(OperationResultContractError, match="recorded convergence"):
        validate_operation_result("ingest", {**payload, "outcome": wrong})


def test_unconverged_ingest_error_is_permanent_for_source_refusals_and_retryable_for_profiles() -> None:
    """A recorded admission refusal cannot change at this head; pending profiles can.

    Anti-vacuity: a blanket ``retryable=True`` error fails the first assertion.
    """
    from polylogue.operations.machine_receipts import ingest_unconverged_error

    def history(source_complete: bool) -> IngestHistoricalReceiptV2:
        item = IngestInputHistoricalReceipt(
            source_item_id="source-item:fixture",
            logical_coordinate="fixture.json",
            denominator=1,
            raw_ids=["raw:complete"],
        )
        return IngestHistoricalReceiptV2(
            source_generation_id="generation:fixture",
            final_sequence=1,
            input_count=1,
            input_pages_ref="operation:fixture",
            input_page_count=1,
            input_pages_digest=ingest_input_pages_digest([IngestInputPageHistoricalReceipt.from_items(0, [item])]),
            summary=IngestTerminalSummaryHistorical(
                enumeration_complete=True,
                source_complete=source_complete,
                confirmed_raw_count=1,
                unresolved_raw_count=0,
                profile_targets_observed=0,
                profile_convergence_complete=False,
            ),
        )

    refused = ingest_unconverged_error(history(False))
    pending = ingest_unconverged_error(history(True))
    assert (refused["code"], refused["retryable"]) == ("ingest_source_incomplete", False)
    assert (pending["code"], pending["retryable"]) == ("ingest_convergence_pending", True)


def test_await_state_of_a_degraded_ingest_satisfies_the_control_result_contract() -> None:
    """The lifecycle ``error`` is a declared field of the control result contract.

    Anti-vacuity: remove ``error`` from ``MutationResult`` and operation.await
    of a degraded ingest fails ``validate_operation_result``.
    """
    from polylogue.operations.daemon_protocol import validate_operation_result

    for control in ("operation.await", "operation.status", "operation.cancel"):
        validate_operation_result(
            control,
            {
                "outcome": "degraded",
                "sequence": 3,
                "error": {"code": "ingest_convergence_pending", "retryable": True},
            },
        )


def test_refusal_page_digest_streams_to_the_canonical_array_digest() -> None:
    """The incremental digest equals the digest of the canonical page array.

    Anti-vacuity: drop the separator or the closing bracket from the streamed
    form and the two digests differ.
    """
    import hashlib
    import json

    from polylogue.operations.machine_receipts import IngestRefusalPageHistoricalReceipt, IngestRefusalPagesDigest

    pages = [
        IngestRefusalPageHistoricalReceipt(ordinal=ordinal, refusals=[_refusal(ordinal * 2), _refusal(ordinal * 2 + 1)])
        for ordinal in range(3)
    ]
    streamed = IngestRefusalPagesDigest()
    for page in pages:
        streamed.update(page)
    canonical = json.dumps([page.model_dump(mode="json") for page in pages], sort_keys=True, separators=(",", ":"))
    assert streamed.hexdigest() == hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("suppressed,deleted,absent", [(0, 0, 0), (3, 1, 1), (3, 0, 0)])
def test_reset_history_round_trips_only_completed_apply_counts(suppressed: int, deleted: int, absent: int) -> None:
    from polylogue.operations.machine_receipts import IdentityResetHistoricalReceipt, encode_machine_receipt

    history = IdentityResetHistoricalReceipt(
        suppressed_count=suppressed, deleted_archive_rows=deleted, tombstoned_without_index_row_count=absent
    )
    assert decode_machine_receipt(encode_machine_receipt(history)) == history
    assert history.count_scope == "completing-apply"
    with pytest.raises(ValueError):
        decode_machine_receipt({**encode_machine_receipt(history), "session_ids": ["neutral-session"]})


@pytest.mark.parametrize("deleted,absent", [(4, 0), (0, 4), (-1, 0)])
def test_reset_history_refuses_impossible_completed_counts(deleted: int, absent: int) -> None:
    with pytest.raises(ValueError):
        decode_machine_receipt(
            {
                "kind": "identity-reset/v1",
                "suppressed_count": 3,
                "deleted_archive_rows": deleted,
                "tombstoned_without_index_row_count": absent,
            }
        )
