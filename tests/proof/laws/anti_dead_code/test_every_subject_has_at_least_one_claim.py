"""Every subject of a fully-bound kind is selected by at least one claim.

Failure mode: a SubjectRef is built but no claim's ``subject_query``
matches its kind/attrs. The subject sits in the catalog count but never
contributes an obligation — the count goes up, the verified surface
doesn't.

Fully-bound kinds are the ones where 100% of subjects must be selected
by definition. Partially-bound kinds (e.g. ``artifact.path``, where
the dependency-closure claim deliberately filters to durable+non_core
layers) are excluded — their partial coverage is intentional design.
Adding a kind to ``_FULLY_BOUND_KINDS`` is the mechanism by which a
catalog-construction guarantee becomes enforced.
"""

from __future__ import annotations

import pytest

from polylogue.proof.catalog import build_verification_catalog

_FULLY_BOUND_KINDS = (
    "cli.command",
    "cli.json_command",
    "provider.capability",
    "trace.operation",
    "diagnostic.observable",
)


@pytest.mark.parametrize("kind", _FULLY_BOUND_KINDS)
def test_every_subject_of_a_fully_bound_kind_is_selected(kind: str) -> None:
    catalog = build_verification_catalog()
    selected_ids = {obligation.subject.id for obligation in catalog.obligations}
    orphans = [subject.id for subject in catalog.subjects if subject.kind == kind and subject.id not in selected_ids]
    assert not orphans, f"{kind} subjects with no selecting claim: {sorted(orphans)}"
