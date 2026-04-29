"""Every catalog Claim selects at least one subject.

Failure mode: a Claim is added (or its subject_query refactored) such
that nothing matches. The catalog still renders, the obligations table
shows zero for that claim, and there is no signal — silent dead code in
a structure that's supposed to be enforcement.

This law sits next to ``catalog.non_abstract_claim_subjects`` (the
existing catalog quality check). The quality check produces an
``OutcomeStatus.ERROR``; this law surfaces the same condition as a
pytest failure with the offending claim id named.
"""

from __future__ import annotations

from polylogue.proof.catalog import build_verification_catalog


def test_every_claim_selects_at_least_one_subject() -> None:
    catalog = build_verification_catalog()
    obligations_by_claim = catalog.obligations_by_claim()
    unbound = [
        claim.id for claim in catalog.claims if not claim.abstract and obligations_by_claim.get(claim.id, 0) == 0
    ]
    assert not unbound, f"non-abstract claims with zero subjects: {sorted(unbound)}"
