"""Ephemeral controlled seam mutants for the shared convergence laws.

These are observations supplied by a route test double. They do not create a
database, ledger, trace, scheduler record, or campaign state. A 04r9f variant
can install the same behavior at its smallest production seam and reuse the
same rejection assertions.
"""

from __future__ import annotations

from dataclasses import replace

from tests.infra.convergence_laws import ConvergenceMutant, RouteObservation


def mutate_observation(observation: RouteObservation, mutant: ConvergenceMutant) -> RouteObservation:
    """Apply one named fault to an otherwise valid route observation."""
    projection = observation.projection
    if mutant is ConvergenceMutant.ORDER_SENSITIVE_OVERWRITE:
        term, members = projection.fts_membership[0]
        replacement = (*members[:-1], "order-sensitive-overwrite")
        projection = replace(
            projection,
            fts_membership=((term, replacement), *projection.fts_membership[1:]),
        )
    elif mutant is ConvergenceMutant.OMITTED_BATCH_MEMBER:
        term, members = projection.fts_membership[0]
        projection = replace(projection, fts_membership=((term, members[1:]), *projection.fts_membership[1:]))
    elif mutant is ConvergenceMutant.UNCONDITIONAL_REWRITE:
        return replace(observation, publication_count=observation.publication_count + 1)
    elif mutant is ConvergenceMutant.STALE_EXCESS_RETENTION:
        term, members = projection.fts_membership[0]
        projection = replace(
            projection,
            fts_membership=((term, (*members, "stale-excess-block")), *projection.fts_membership[1:]),
        )
    elif mutant is ConvergenceMutant.OVER_BROAD_INVALIDATION:
        projection = replace(projection, affected_partitions=("assistant", "user"))
    else:
        raise AssertionError(f"unhandled convergence mutant: {mutant}")
    return replace(observation, projection=projection)


__all__ = ["mutate_observation"]
