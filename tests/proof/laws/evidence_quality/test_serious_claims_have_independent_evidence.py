"""Serious proof claims must not rely on ceremonial/self-attesting evidence."""

from __future__ import annotations

import pytest

from polylogue.proof.catalog import build_verification_catalog

pytestmark = pytest.mark.proof_law


def test_serious_claims_have_independent_evidence_metadata() -> None:
    catalog = build_verification_catalog()
    weak = [
        f"{claim.id}: {claim.oracle}/{claim.independence_level}"
        for claim in catalog.claims
        if claim.severity == "serious"
        and claim.tracked_exception is None
        and (
            claim.oracle == "ceremonial" or claim.independence_level in {"same_source", "self_attesting", "ceremonial"}
        )
    ]
    assert not weak, f"serious claims with weak oracle independence: {weak}"
