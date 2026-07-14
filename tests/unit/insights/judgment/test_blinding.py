"""Tests for blinded judgment projection (rxdo.9.6, mechanism F)."""

from __future__ import annotations

import pytest

from polylogue.insights.judgment.blinding import (
    DEFAULT_MASKED_PROVENANCE_FIELDS,
    assert_no_leak,
    blind_items,
    reveal,
)

_RECORDS = [
    {
        "assertion_id": "cand-1",
        "actor_ref": "agent:claude-sonnet-5",
        "model": "claude-sonnet-5",
        "detector_ref": "insight:detector-x",
        "body_text": "The refactor removes the dead branch.",
        "evidence_refs": ["session:s1"],
    },
    {
        "assertion_id": "cand-2",
        "actor_ref": "agent:gpt-5.6-terra",
        "model": "gpt-5.6-terra",
        "detector_ref": "insight:detector-y",
        "body_text": "The refactor introduces a race.",
        "evidence_refs": ["session:s2"],
    },
]

#: Hardcoded independently of DEFAULT_MASKED_PROVENANCE_FIELDS on purpose --
#: tests that derive their expectations FROM the production constant (e.g.
#: ``for field in DEFAULT_MASKED_PROVENANCE_FIELDS``) silently shrink their own
#: checklist if a field is ever accidentally deleted from production, so they
#: can never catch that deletion. test_default_masks_match_the_frozen_checklist
#: below is what actually guards the production constant against drift; every
#: other test in this file checks against THIS frozen literal.
_EXPECTED_MASKED_PROVENANCE_FIELDS: frozenset[str] = frozenset(
    {
        "actor_ref",
        "author_ref",
        "author_kind",
        "model",
        "provider",
        "detector_ref",
        "arm",
        "prior_score",
        "prior_rank",
        "judge_ref",
        "execution_context_id",
    }
)


def test_default_masks_match_the_frozen_checklist() -> None:
    """Guards DEFAULT_MASKED_PROVENANCE_FIELDS against silent shrinkage/growth.

    Every other test below asserts against the hardcoded checklist, not this
    constant, precisely so an accidental deletion from production is caught
    here directly rather than by a test whose own expectations quietly shrank
    along with it.
    """
    assert DEFAULT_MASKED_PROVENANCE_FIELDS == _EXPECTED_MASKED_PROVENANCE_FIELDS


def test_masked_fields_are_not_recoverable_from_the_visible_projection() -> None:
    blinded, receipt = blind_items(_RECORDS, order=[1, 0], rubric_ref="rubric:correctness@v1", sealed_at_ms=100)
    for item in blinded:
        for field in _EXPECTED_MASKED_PROVENANCE_FIELDS:
            assert field not in item.visible_fields
    # content required for the rubric survives
    assert {item.visible_fields["body_text"] for item in blinded} == {
        "The refactor removes the dead branch.",
        "The refactor introduces a race.",
    }
    assert receipt.revealed_at_ms is None


def test_order_is_receipted_and_reconstructible() -> None:
    blinded, receipt = blind_items(_RECORDS, order=[1, 0], rubric_ref="rubric:correctness@v1", sealed_at_ms=100)
    # position 0 in the blinded view is source index 1 (cand-2)
    assert blinded[0].visible_fields["assertion_id"] == "cand-2"
    assert blinded[1].visible_fields["assertion_id"] == "cand-1"
    assert receipt.rubric_ref == "rubric:correctness@v1"
    assert receipt.sealed_at_ms == 100


def test_order_must_be_a_permutation() -> None:
    with pytest.raises(ValueError, match="permutation"):
        blind_items(_RECORDS, order=[0, 0], rubric_ref="rubric:x", sealed_at_ms=1)


def test_reveal_requires_a_recorded_verdict() -> None:
    _, receipt = blind_items(_RECORDS, order=[0, 1], rubric_ref="rubric:x", sealed_at_ms=1)
    with pytest.raises(ValueError, match="only authorized after"):
        reveal(receipt, revealed_at_ms=200, verdict_recorded=False)


def test_reveal_preserves_receipt_fields_and_sets_revealed_at() -> None:
    _, receipt = blind_items(_RECORDS, order=[0, 1], rubric_ref="rubric:x", sealed_at_ms=1)
    revealed = reveal(receipt, revealed_at_ms=200, verdict_recorded=True)
    assert revealed.revealed_at_ms == 200
    assert revealed.item_order_hash == receipt.item_order_hash
    assert revealed.masked_fields == receipt.masked_fields
    assert revealed.rubric_ref == receipt.rubric_ref


@pytest.mark.parametrize("field", sorted(_EXPECTED_MASKED_PROVENANCE_FIELDS))
def test_removing_one_production_mask_leaks_a_sentinel_and_fails(field: str) -> None:
    """AC: removing one mask leaks a sentinel and fails the test.

    Parametrized over the hardcoded checklist, not DEFAULT_MASKED_PROVENANCE_FIELDS
    itself -- the prior version derived its "weakened" mask set by subtracting
    from the live production constant, so it only ever exercised whichever one
    field ("actor_ref") happened to be hardcoded elsewhere in the test, and 9 of
    the 11 fields had zero mutation coverage: deleting any of them from
    DEFAULT_MASKED_PROVENANCE_FIELDS passed every test in this file unchanged.
    """
    sentinel_value = f"SENTINEL-LEAK-{field}"
    sentinel_records = [
        {**_RECORDS[0], field: sentinel_value},
        _RECORDS[1],
    ]
    weakened_masks = _EXPECTED_MASKED_PROVENANCE_FIELDS - {field}
    blinded, _ = blind_items(
        sentinel_records,
        order=[0, 1],
        rubric_ref="rubric:x",
        sealed_at_ms=1,
        masked_fields=weakened_masks,
    )
    leaked = any(item.visible_fields.get(field) == sentinel_value for item in blinded)
    assert leaked, f"sentinel must be visible once the {field} mask is removed (proves the test is load-bearing)"
    with pytest.raises(ValueError, match="blinding leak"):
        assert_no_leak(blinded, masked_fields=DEFAULT_MASKED_PROVENANCE_FIELDS)


def test_production_masks_pass_the_leak_check() -> None:
    blinded, _ = blind_items(_RECORDS, order=[0, 1], rubric_ref="rubric:x", sealed_at_ms=1)
    assert_no_leak(blinded)
