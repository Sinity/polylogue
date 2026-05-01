"""Serious proof claims must explain when their evidence becomes stale."""

from __future__ import annotations

import pytest

from polylogue.proof.catalog import build_verification_catalog

pytestmark = pytest.mark.proof_law


def test_serious_claims_explain_staleness() -> None:
    catalog = build_verification_catalog()
    missing = [
        claim.id
        for claim in catalog.claims
        if claim.severity == "serious" and not claim.staleness_conditions and not claim.tracked_exception
    ]
    assert not missing, f"serious claims missing staleness conditions: {missing}"
