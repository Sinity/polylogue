"""Derive the closed :class:`RawAuthorityVerdict` from proven revision evidence.

Phase 2 of the raw-authority redesign (polylogue-w6hql). This module adds
**no new source of truth**: it is a pure, read-only projection over the
already-proven per-raw evidence that :mod:`polylogue.archive.revision_authority`
computes (:class:`HistoricalRevisionDecision`), collapsing that richer
internal vocabulary (``authority`` x ``relation`` x ``duplicate_of_raw_id``)
down to the single small closed enum downstream consumers should read.

Deliberately out of scope for this phase (see polylogue-w6hql notes for the
full accounting): rewriting the write paths of ``raw_authority_blockers``,
``raw_authority_censuses``, ``raw_authority_census_plans``,
``raw_authority_post_plans``, ``raw_authority_parser_census``, and
``raw_membership_census``, or removing/rewriting the ~5,000-8,000 lines of
classification machinery in ``polylogue/storage/repair.py`` and
``polylogue/storage/sqlite/archive_tiers/revision_governance.py`` that
produce that evidence. Those tables and modules remain the system of record
this phase; this module only reads their *output* (via the
``HistoricalRevisionDecision`` shape) and re-expresses it as one verdict.
"""

from __future__ import annotations

from polylogue.archive.revision_authority import HistoricalRevisionDecision, RawRevisionAuthority
from polylogue.core.enums import RawAuthorityVerdict


def derive_raw_authority_verdict(
    decisions: list[HistoricalRevisionDecision],
) -> dict[str, RawAuthorityVerdict]:
    """Map one cohort's proven decisions onto the closed 5-value verdict.

    ``decisions`` must be the full output of
    :func:`polylogue.archive.revision_authority.classify_historical_full_revisions`
    (or its streaming twin) for a single logical-source cohort -- every raw_id
    that shares the cohort's ``logical_source_key``, including byte-identical
    duplicates. Passing a partial cohort silently mis-derives ``VERIFIED`` vs
    ``SOLE_COPY`` (the sole-copy check depends on seeing every member), so
    callers must not filter the list before calling this function.

    A raw_id that has never been classified at all (e.g.
    ``raw_sessions.revision_kind = 'unknown'``, the pre-governance default)
    is not represented in ``decisions`` -- callers derive ``UNCHECKED`` for
    those rows themselves, by absence, rather than through this function.
    See :func:`polylogue.storage.raw_authority_verdict_projection.project_raw_authority_verdicts`
    for the concrete read-only wiring against a live archive.
    """
    if not decisions:
        return {}

    successors: dict[str, bool] = {decision.raw_id: False for decision in decisions}
    for decision in decisions:
        if decision.predecessor_raw_id is not None:
            successors[decision.predecessor_raw_id] = True

    sole_copy = len(decisions) == 1

    verdicts: dict[str, RawAuthorityVerdict] = {}
    for decision in decisions:
        if decision.relation == "duplicate":
            verdicts[decision.raw_id] = RawAuthorityVerdict.SUPERSEDED
            continue
        if decision.authority is RawRevisionAuthority.QUARANTINED:
            verdicts[decision.raw_id] = RawAuthorityVerdict.DIVERGED
            continue
        # BYTE_PROVEN, relation in {"baseline", "predecessor"}.
        if successors[decision.raw_id]:
            verdicts[decision.raw_id] = RawAuthorityVerdict.SUPERSEDED
        elif sole_copy:
            verdicts[decision.raw_id] = RawAuthorityVerdict.SOLE_COPY
        else:
            verdicts[decision.raw_id] = RawAuthorityVerdict.VERIFIED
    return verdicts


__all__ = ["derive_raw_authority_verdict"]
