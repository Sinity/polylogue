"""The public fact-family declaration inventory.

This is a catalog of owners and wire requirements, not a storage table or a
second lifecycle.  A family is allowed to adopt :class:`EvidenceValue`
incrementally; until then its declaration still makes the intended public
contract visible to completeness checks and renderers.
"""

from __future__ import annotations

from typing import Final

from polylogue.archive.query.source_freshness import SOURCE_CURSOR_BYTE_LAG_FAMILY
from polylogue.core.evidence_value import (
    EvidenceAxis,
    FactFamilySpec,
    MeasurementAuthority,
    ValueState,
)
from polylogue.core.refs import ObjectRef
from polylogue.daemon.status_snapshot import STATUS_SNAPSHOT_STATE_FAMILY
from polylogue.storage.usage import (
    SESSION_USAGE_RECONCILED_COST_FAMILY,
    SESSION_USAGE_RECONCILED_TOKENS_FAMILY,
    USAGE_LANE_CATALOG_COST_FAMILY,
    USAGE_LANE_EXACT_TOKENS_FAMILY,
)

_COMMON_AXES: Final[frozenset[EvidenceAxis]] = frozenset(
    {
        "value_state",
        "measurement_authority",
        "evidence_refs",
        "definition_ref",
        "temporal",
        "enumeration",
        "coverage",
        "freshness",
    }
)
_STATES: Final[frozenset[ValueState]] = frozenset(
    {"known", "unknown", "unavailable", "skipped", "not_applicable", "redacted"}
)
_AUTHORITIES: Final[tuple[MeasurementAuthority, ...]] = (
    "structural",
    "provider-reported",
    "catalog-derived",
    "rule-derived",
    "model-derived",
    "agent-declared",
    "judged",
)


def _family(
    name: str,
    *,
    owner: str,
    field: str,
    schema: str,
    unit: str,
    grain: str,
    denominator: str,
    authorities: tuple[MeasurementAuthority, ...] = _AUTHORITIES,
) -> FactFamilySpec:
    return FactFamilySpec(
        family=name,
        owner=owner,
        source_adapter=f"{owner}.{field}",
        public_field=field,
        renderer_label=field.replace("_", " "),
        value_schema=schema,  # type: ignore[arg-type]
        unit=unit,
        grain=grain,
        denominator=denominator,
        definition_ref=ObjectRef(kind="insight", object_id=f"{name}:v1"),
        required_axes=_COMMON_AXES,
        allowed_states=_STATES,
        allowed_authorities=frozenset(authorities),
        authority_precedence=authorities,
    )


# Keep one declaration per public fact family.  These names intentionally
# describe the owning domain, so adding a producer cannot introduce a second
# family-specific value-state or authority vocabulary.
_DOMAIN_FAMILY_DECLARATIONS: Final[tuple[FactFamilySpec, ...]] = (
    _family(
        "temporal.value",
        owner="polylogue.core.temporal",
        field="temporal",
        schema="string",
        unit="time",
        grain="event",
        denominator="declared event frame",
        authorities=("structural", "provider-reported", "rule-derived"),
    ),
    _family(
        "outcome.tool",
        owner="polylogue.archive.actions",
        field="tool_outcome",
        schema="object",
        unit="outcome",
        grain="tool_action",
        denominator="declared tool-result frame",
        authorities=("structural", "provider-reported", "rule-derived"),
    ),
    _family(
        "usage.tokens",
        owner="polylogue.storage.usage",
        field="tokens",
        schema="number",
        unit="tokens",
        grain="session",
        denominator="declared usage frame",
        authorities=("provider-reported", "structural", "model-derived"),
    ),
    _family(
        "usage.price",
        owner="polylogue.storage.usage",
        field="price",
        schema="number",
        unit="USD",
        grain="session",
        denominator="declared priced usage frame",
        authorities=("provider-reported", "catalog-derived", "model-derived"),
    ),
    _family(
        "inference.profile_phase",
        owner="polylogue.insights.profile",
        field="profile_phase",
        schema="object",
        unit="classification",
        grain="session",
        denominator="declared session frame",
        authorities=("rule-derived", "model-derived", "judged"),
    ),
    _family(
        "quota.observation",
        owner="polylogue.operations.quota",
        field="quota",
        schema="object",
        unit="quota",
        grain="quota_window",
        denominator="declared quota frame",
    ),
    _family(
        "metric.aggregate",
        owner="polylogue.insights.measurement",
        field="metric",
        schema="number",
        unit="declared",
        grain="declared metric grain",
        denominator="declared metric frame",
    ),
    _family(
        "query.aggregate",
        owner="polylogue.archive.query",
        field="query_result",
        schema="object",
        unit="result",
        grain="declared query grain",
        denominator="declared query frame",
    ),
    _family(
        "source.freshness",
        owner="polylogue.archive.query.source_freshness",
        field="freshness",
        schema="object",
        unit="state",
        grain="source",
        denominator="declared source frame",
        authorities=("structural", "provider-reported"),
    ),
)

FACT_FAMILY_DECLARATIONS: Final[tuple[FactFamilySpec, ...]] = (
    *_DOMAIN_FAMILY_DECLARATIONS,
    SOURCE_CURSOR_BYTE_LAG_FAMILY,
    STATUS_SNAPSHOT_STATE_FAMILY,
    SESSION_USAGE_RECONCILED_COST_FAMILY,
    SESSION_USAGE_RECONCILED_TOKENS_FAMILY,
    USAGE_LANE_CATALOG_COST_FAMILY,
    USAGE_LANE_EXACT_TOKENS_FAMILY,
)

FACT_FAMILY_BY_NAME: Final[dict[str, FactFamilySpec]] = {family.family: family for family in FACT_FAMILY_DECLARATIONS}


def fact_family_declarations() -> tuple[FactFamilySpec, ...]:
    """Return the immutable declaration inventory in stable order."""

    return FACT_FAMILY_DECLARATIONS


def fact_family_schema() -> tuple[dict[str, object], ...]:
    """Return deterministic public schemas for discovery and renderers."""

    return tuple(family.public_schema() for family in sorted(FACT_FAMILY_DECLARATIONS, key=lambda item: item.family))


__all__ = [
    "FACT_FAMILY_BY_NAME",
    "FACT_FAMILY_DECLARATIONS",
    "fact_family_declarations",
    "fact_family_schema",
]
