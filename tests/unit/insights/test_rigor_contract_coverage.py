"""Every registered insight and public numeric field is rigor-contracted.

A product registered in ``INSIGHT_REGISTRY`` with no ``RigorContract`` row
vanishes from ``polylogue insights audit`` instead of reporting as uncovered.

Anti-vacuity: dropping a contract row from ``_RIGOR_MATRIX`` turns the second
test red.
"""

from __future__ import annotations

import pytest

from polylogue.insights import rigor as rigor_mod
from polylogue.insights.registry import INSIGHT_REGISTRY
from polylogue.insights.rigor import (
    RIGOR_EXEMPT,
    invalid_nullable_field_contracts,
    missing_numeric_field_coverage,
    missing_numeric_item_models,
    rigor_contract_names,
)


def _uncovered() -> tuple[str, ...]:
    return tuple(sorted(set(INSIGHT_REGISTRY) - (set(rigor_contract_names()) | set(RIGOR_EXEMPT))))


def test_every_registered_insight_is_contracted_or_exempt() -> None:
    assert _uncovered() == ()
    assert missing_numeric_field_coverage() == ()
    assert missing_numeric_item_models() == ()
    assert invalid_nullable_field_contracts() == ()


def test_removing_a_contract_reports_the_product_as_uncovered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rigor_mod,
        "_RIGOR_MATRIX",
        tuple(contract for contract in rigor_mod._RIGOR_MATRIX if contract.insight_name != "session_profiles"),
    )

    assert "session_profiles" in _uncovered()
