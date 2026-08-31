"""The public fact-family declaration inventory.

This is a catalog of owners and wire requirements, not a storage table or a
second lifecycle.  A family is allowed to adopt :class:`EvidenceValue`
incrementally; until then its declaration still makes the intended public
contract visible to completeness checks and renderers.
"""

from __future__ import annotations

from typing import Final

from polylogue.core.evidence_value import (
    EvidenceAxis,
    FactFamilySpec,
    MeasurementAuthority,
    ValueState,
)
from polylogue.core.refs import ObjectRef

_SOURCE_CURSOR_BYTE_LAG_DEFINITION_REF: Final = ObjectRef(
    kind="insight",
    object_id="source-cursor-byte-lag:v1",
)
SOURCE_CURSOR_BYTE_LAG_FAMILY: Final = FactFamilySpec(
    family="archive.source_cursor_byte_lag",
    owner="polylogue.archive.query.source_freshness",
    source_adapter="project_named_source_freshness",
    public_field="byte_lag",
    renderer_label="cursor byte lag",
    value_schema="integer",
    unit="bytes",
    grain="source_path",
    denominator="declared exact source paths",
    definition_ref=_SOURCE_CURSOR_BYTE_LAG_DEFINITION_REF,
    required_axes=frozenset(
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
    ),
    allowed_states=frozenset({"known", "unknown", "unavailable"}),
    allowed_authorities=frozenset({"structural"}),
    authority_precedence=("structural",),
    requires_last_good_when_degraded=True,
)


_STATUS_SNAPSHOT_STATE_DEFINITION_REF = ObjectRef(
    kind="insight",
    object_id="daemon-status-snapshot-state:v1",
)
STATUS_SNAPSHOT_STATE_FAMILY = FactFamilySpec(
    family="daemon.status_snapshot_state",
    owner="polylogue.daemon.status_snapshot",
    source_adapter="StatusSnapshot.with_metadata",
    public_field="status_snapshot.state_evidence",
    renderer_label="status snapshot state",
    value_schema="string",
    unit="state",
    grain="status_snapshot",
    denominator="one cached daemon status snapshot",
    definition_ref=_STATUS_SNAPSHOT_STATE_DEFINITION_REF,
    required_axes=frozenset(
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
    ),
    allowed_states=frozenset({"known"}),
    allowed_authorities=frozenset({"structural"}),
    authority_precedence=("structural",),
    requires_last_good_when_degraded=True,
)


_USAGE_LANE_EXACT_TOKENS_DEFINITION_REF: Final = ObjectRef(
    kind="insight",
    object_id="usage-lane-exact-total-tokens:v1",
)
USAGE_LANE_EXACT_TOKENS_FAMILY: Final = FactFamilySpec(
    family="usage.pricing_lane_exact_total_tokens",
    owner="polylogue.storage.usage",
    source_adapter="_pricing_lane_reports",
    public_field="pricing_lanes[].exact_total_tokens_evidence",
    renderer_label="exact total tokens",
    value_schema="integer",
    unit="tokens",
    grain="pricing_lane",
    denominator="session_model_usage rows in the declared lane",
    definition_ref=_USAGE_LANE_EXACT_TOKENS_DEFINITION_REF,
    required_axes=frozenset(
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
    ),
    allowed_states=frozenset({"known", "unknown"}),
    allowed_authorities=frozenset({"provider-reported", "model-derived", "structural"}),
    authority_precedence=("provider-reported", "structural", "model-derived"),
)

_USAGE_LANE_CATALOG_COST_DEFINITION_REF: Final = ObjectRef(
    kind="insight",
    object_id="usage-lane-catalog-api-equivalent-cost:v1",
)
USAGE_LANE_CATALOG_COST_FAMILY: Final = FactFamilySpec(
    family="usage.pricing_lane_catalog_api_equivalent_cost",
    owner="polylogue.storage.usage",
    source_adapter="_pricing_lane_reports",
    public_field="pricing_lanes[].catalog_api_equivalent_evidence",
    renderer_label="catalog API-equivalent cost",
    value_schema="number",
    unit="USD",
    grain="pricing_lane",
    denominator="session_model_usage rows in the declared lane",
    definition_ref=_USAGE_LANE_CATALOG_COST_DEFINITION_REF,
    required_axes=frozenset(
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
    ),
    allowed_states=frozenset({"known", "unknown"}),
    allowed_authorities=frozenset({"catalog-derived"}),
    authority_precedence=("catalog-derived",),
)


_SESSION_USAGE_RECONCILED_TOKENS_DEFINITION_REF: Final = ObjectRef(
    kind="insight",
    object_id="session-usage-reconciled-total-tokens:v1",
)
SESSION_USAGE_RECONCILED_TOKENS_FAMILY: Final = FactFamilySpec(
    family="usage.session_reconciled_total_tokens",
    owner="polylogue.storage.usage",
    source_adapter="build_session_usage_reconciliation",
    public_field="session_usage_reconciliation.reconciled_tokens_evidence",
    renderer_label="reconciled session total tokens",
    value_schema="integer",
    unit="tokens",
    grain="session",
    denominator="canonical per-session token sources (session_model_usage rollup, session_profiles estimate)",
    definition_ref=_SESSION_USAGE_RECONCILED_TOKENS_DEFINITION_REF,
    required_axes=frozenset(
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
    ),
    allowed_states=frozenset({"known", "unknown"}),
    allowed_authorities=frozenset({"provider-reported", "structural", "model-derived"}),
    authority_precedence=("provider-reported", "structural", "model-derived"),
)

_SESSION_USAGE_RECONCILED_COST_DEFINITION_REF: Final = ObjectRef(
    kind="insight",
    object_id="session-usage-reconciled-catalog-cost:v1",
)
SESSION_USAGE_RECONCILED_COST_FAMILY: Final = FactFamilySpec(
    family="usage.session_reconciled_catalog_cost",
    owner="polylogue.storage.usage",
    source_adapter="build_session_usage_reconciliation",
    public_field="session_usage_reconciliation.reconciled_cost_evidence",
    renderer_label="reconciled session catalog-equivalent cost",
    value_schema="number",
    unit="USD",
    grain="session",
    denominator=(
        "canonical per-session cost sources (fresh catalog price of the reconciled tokens, "
        "legacy session_profiles/cost-insight total)"
    ),
    definition_ref=_SESSION_USAGE_RECONCILED_COST_DEFINITION_REF,
    required_axes=frozenset(
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
    ),
    allowed_states=frozenset({"known", "unknown"}),
    allowed_authorities=frozenset({"provider-reported", "catalog-derived", "model-derived"}),
    # A current catalog reprice supersedes a persisted cost, including rows
    # written before aggregate cost provenance was corrected.
    authority_precedence=("catalog-derived", "provider-reported", "model-derived"),
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
    "SESSION_USAGE_RECONCILED_COST_FAMILY",
    "SESSION_USAGE_RECONCILED_TOKENS_FAMILY",
    "SOURCE_CURSOR_BYTE_LAG_FAMILY",
    "STATUS_SNAPSHOT_STATE_FAMILY",
    "USAGE_LANE_CATALOG_COST_FAMILY",
    "USAGE_LANE_EXACT_TOKENS_FAMILY",
    "FACT_FAMILY_DECLARATIONS",
    "fact_family_declarations",
    "fact_family_schema",
]
