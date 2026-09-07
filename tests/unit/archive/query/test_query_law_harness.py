"""The query-law declarations are complete, honest, and bound to production.

These tests execute no query. They pin the *generation* half of the
differential: a new query unit, terminal pipeline stage, read projection or
ref-shaped payload field has no place to hide, every exemption states a
reason, every declared mutation is actually wired, and every census family
names an SLO owner that exists. The executing half lives in
``tests/integration/test_query_law_differential.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from polylogue.archive.query.metadata import (
    PROJECTION_QUERY_UNITS,
    QUERY_UNIT_DESCRIPTORS,
)
from polylogue.scenarios.workload import BudgetSemantics
from polylogue.schemas.generation.archive_workload_profile import _shape_anchors
from tests.infra.query_contract import (
    CENSUS_FAMILIES,
    CORPUS_SHAPE_ANCHORS,
    NON_REF_FIELDS,
    PIPELINE_STAGE_EXEMPTIONS,
    PIPELINE_STAGE_KINDS,
    PIPELINE_STAGE_LAWS,
    PROJECTION_COVERAGE,
    QUERY_LAW_EXEMPTIONS,
    QUERY_LAWS,
    QUERY_LAWS_BY_ID,
    REF_FAMILIES_BY_UNIT,
    UNIT_IDENTITY_BY_UNIT,
    UNIT_PROBE_BY_UNIT,
    CensusFamily,
    ref_suspected_fields,
)
from tests.infra.query_differential import QUERY_LAW_MUTANTS

SLO_CATALOG = Path("docs/plans/slo-catalog.yaml")


def test_query_law_every_unit_declares_a_probe_and_an_identity() -> None:
    """A new query unit cannot join the grammar without a law coordinate.

    Anti-vacuity: deleting one ``UnitProbe`` or ``UnitIdentity`` entry fails
    here, so the differential can never silently skip a unit.
    """

    for descriptor in QUERY_UNIT_DESCRIPTORS:
        assert descriptor.unit in UNIT_PROBE_BY_UNIT, f"{descriptor.unit} declares no law probe"
        assert descriptor.unit in UNIT_IDENTITY_BY_UNIT, f"{descriptor.unit} declares no row identity"
        probe = UNIT_PROBE_BY_UNIT[descriptor.unit]
        assert probe.source == descriptor.plural_source
        if probe.group_field is None:
            assert not descriptor.aggregate_group_fields, (
                f"{descriptor.unit} declares aggregate group fields but its probe groups on nothing"
            )
        else:
            assert probe.group_field in descriptor.aggregate_group_fields


def test_query_law_every_pipeline_stage_is_covered_or_exempt() -> None:
    """Every terminal stage kind is constrained by a law or states why not."""

    for kind in PIPELINE_STAGE_KINDS:
        covered = PIPELINE_STAGE_LAWS.get(kind, ())
        exempt = PIPELINE_STAGE_EXEMPTIONS.get(kind, "")
        assert bool(covered) != bool(exempt), f"pipeline stage {kind!r} is neither covered nor exempt"
        for law_id in covered:
            assert law_id in QUERY_LAWS_BY_ID, f"stage {kind!r} names unknown law {law_id!r}"


def test_query_law_every_read_projection_is_covered() -> None:
    covered = {coverage.unit for coverage in PROJECTION_COVERAGE}
    assert covered == set(PROJECTION_QUERY_UNITS)
    for coverage in PROJECTION_COVERAGE:
        assert coverage.law_id in QUERY_LAWS_BY_ID


def test_query_law_every_ref_shaped_field_is_declared() -> None:
    """A payload field that looks like a reference is a ref family or is named.

    Anti-vacuity: adding a ``*_ref`` field to a row payload without a
    ``RefFamily`` or ``NON_REF_FIELDS`` entry fails here.
    """

    undeclared: list[str] = []
    for descriptor in QUERY_UNIT_DESCRIPTORS:
        declared = {family.field for family in REF_FAMILIES_BY_UNIT[descriptor.unit]}
        for name in ref_suspected_fields(descriptor):
            if name in declared or f"{descriptor.unit}.{name}" in NON_REF_FIELDS:
                continue
            undeclared.append(f"{descriptor.unit}.{name}")
    assert not undeclared, f"ref-shaped payload fields with no declaration: {undeclared}"


def test_query_law_declared_ref_families_exist_on_their_payload() -> None:
    """A stale ref declaration is as much a gap as a missing one."""

    for descriptor in QUERY_UNIT_DESCRIPTORS:
        available = set(ref_suspected_fields(descriptor))
        for family in REF_FAMILIES_BY_UNIT[descriptor.unit]:
            assert family.field in available, f"{family.unit}.{family.field} is not a field of its payload model"


def test_query_law_every_exemption_states_a_reason_for_a_known_coordinate() -> None:
    for exemption in QUERY_LAW_EXEMPTIONS:
        assert exemption.reason.strip(), f"{exemption.law_id} exemption states no reason"
        assert exemption.unit is not None or exemption.surface is not None, (
            f"{exemption.law_id} is exempted everywhere, which retires the law rather than scoping it"
        )


def test_query_law_every_declared_mutation_is_wired() -> None:
    """Each mutant names a law's own anti-vacuity sentence.

    Anti-vacuity: renaming a law's ``anti_vacuity`` without repointing its
    mutant fails here, so a law cannot claim a mutation nobody runs.
    """

    stated = {law.anti_vacuity for law in QUERY_LAWS}
    for mutant in QUERY_LAW_MUTANTS:
        assert mutant.mutation in stated, f"mutant {mutant.mutant_id} names an unstated mutation"
        for law_id in mutant.expected_violations:
            assert law_id in QUERY_LAWS_BY_ID


@pytest.mark.parametrize("family", CENSUS_FAMILIES, ids=lambda family: family.family_id)
def test_query_law_census_family_declares_a_gated_budget_from_an_slo_owner(family: CensusFamily) -> None:
    """A census family's budget comes from a real SLO surface and can fail.

    Anti-vacuity: making every budget measure-only, or naming a surface the
    catalog does not publish, fails here.
    """

    catalog = yaml.safe_load(SLO_CATALOG.read_text(encoding="utf-8"))
    surfaces = catalog["surfaces"]
    declared = family.slo_owner
    assert declared in surfaces, f"census family names SLO surface {declared!r}, which the catalog does not publish"
    gates = [budget for budget in family.budgets if budget.semantics is BudgetSemantics.REGRESSION_GATE]
    assert gates, "a census family with no regression gate measures nothing that can fail"
    assert family.pushdown_marker.strip(), "a census family must declare the restriction it pushes down"


def test_query_law_corpus_shape_anchors_name_the_profile_keys_production_emits() -> None:
    """The declared corpus shapes are quoted from the live profile builder.

    Anti-vacuity: renaming a distribution key in
    ``archive_workload_profile`` without repointing the anchor fails here,
    so a recorded profile can always replace the declared values.
    """

    quantiles = {"quantiles": {"p50": 1.0, "p95": 2.0, "max": 3.0}}
    emitted = _shape_anchors(
        {
            "session_shapes": {"message_count": quantiles},
            "action_shapes": {
                "tool_uses_per_session": quantiles,
                "tool_results_per_session": quantiles,
            },
            "topology": {"children_per_parent": quantiles},
        },
        {"blob_size": quantiles},
    )
    assert {str(anchor["distribution_ref"]) for anchor in emitted} == {
        anchor.distribution_ref for anchor in CORPUS_SHAPE_ANCHORS
    }
