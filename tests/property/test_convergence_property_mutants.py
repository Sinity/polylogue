"""Anti-vacuity witnesses for every required convergence mutant."""

from __future__ import annotations

import pytest

from tests.infra.convergence_laws import (
    ConvergenceMutant,
    RouteObservation,
    SemanticProjection,
    assert_idempotence_law,
    assert_locality_law,
    assert_unchanged_publication_law,
)
from tests.infra.convergence_mutants import mutate_observation


def _healthy_observation() -> RouteObservation:
    return RouteObservation(
        projection=SemanticProjection(
            fts_membership=(("probe", ("block-a", "block-b")),),
            role_counts=(("assistant", 1), ("user", 1)),
        ),
        publication_count=1,
    )


@pytest.mark.parametrize(
    "mutant, reason",
    (
        (ConvergenceMutant.ORDER_SENSITIVE_OVERWRITE, "idempotence law"),
        (ConvergenceMutant.OMITTED_BATCH_MEMBER, "idempotence law"),
        (ConvergenceMutant.STALE_EXCESS_RETENTION, "idempotence law"),
    ),
)
def test_semantic_mutants_fail_for_their_localized_fts_reason(
    mutant: ConvergenceMutant,
    reason: str,
) -> None:
    healthy = _healthy_observation()
    mutated = mutate_observation(healthy, mutant)

    with pytest.raises(AssertionError, match=reason):
        assert_idempotence_law(healthy.projection, mutated.projection)


def test_unconditional_rewrite_mutant_fails_on_publication_seam() -> None:
    healthy = _healthy_observation()
    mutated = mutate_observation(healthy, ConvergenceMutant.UNCONDITIONAL_REWRITE)

    with pytest.raises(AssertionError, match="unchanged repetition law"):
        assert_unchanged_publication_law(healthy, mutated)


def test_over_broad_invalidation_mutant_fails_on_exact_locality() -> None:
    before = _healthy_observation().projection
    after = SemanticProjection(
        fts_membership=before.fts_membership,
        role_counts=(("assistant", 1), ("user", 2)),
    )
    mutated = mutate_observation(
        RouteObservation(projection=after),
        ConvergenceMutant.OVER_BROAD_INVALIDATION,
    )

    with pytest.raises(AssertionError, match="locality law"):
        assert_locality_law(before, mutated.projection, mutated.projection.affected_partitions)
