"""Completeness laws for the public evidence-family inventory."""

from polylogue.core.evidence_families import (
    FACT_FAMILY_BY_NAME,
    FACT_FAMILY_DECLARATIONS,
    fact_family_schema,
)
from polylogue.core.evidence_value import audit_fact_family_completeness


def test_inventory_covers_required_fact_domains_once() -> None:
    required = {
        "temporal.value",
        "outcome.tool",
        "usage.tokens",
        "usage.price",
        "inference.profile_phase",
        "quota.observation",
        "metric.aggregate",
        "query.aggregate",
        "source.freshness",
    }
    assert required <= FACT_FAMILY_BY_NAME.keys()
    assert len(FACT_FAMILY_DECLARATIONS) == len(FACT_FAMILY_BY_NAME)


def test_inventory_declarations_are_complete_and_renderable() -> None:
    diagnostics = audit_fact_family_completeness(
        FACT_FAMILY_DECLARATIONS, (), required_families=tuple(FACT_FAMILY_BY_NAME)
    )
    assert diagnostics == ()
    schemas = fact_family_schema()
    assert [schema["family"] for schema in schemas] == sorted(FACT_FAMILY_BY_NAME)
    assert all("value_state" in schema["required"] for schema in schemas if isinstance(schema["required"], list))
