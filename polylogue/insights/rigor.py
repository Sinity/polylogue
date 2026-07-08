"""Per-product insight rigor contract matrix (#1275).

Each insight product declared in :mod:`polylogue.insights.archive` has a
``RigorContract`` row here. The contract is the durable, machine-readable
description of how to read a row of that product — which fields carry
direct evidence from the source archive, which fields carry probabilistic
inference, what fallback markers callers should consult, what readiness
semantics apply, and which consumer-facing fields are stable surface
contract.

The contract drives:

- ``polylogue insights audit`` (CLI): per-product rigor profile rollup
  with evidence/inference/fallback marker coverage, stale-version count,
  and confidence distribution.
- documentation: the rendered matrix lives at
  ``docs/insights-rigor-matrix.md`` (regenerated from this module).
- consumer self-discovery: future MCP and API surfaces can query the
  matrix instead of reading prose docs.

This module is contract-only; it does not query the archive. The audit
runner lives in :mod:`polylogue.insights.audit`.
"""

from __future__ import annotations

from collections.abc import Sequence

from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.insights.tool_usage import TOOL_USAGE_INSIGHT_VERSION
from polylogue.storage.runtime.store_constants import (
    SESSION_ENRICHMENT_VERSION,
    SESSION_INFERENCE_VERSION,
    SESSION_INSIGHT_MATERIALIZER_VERSION,
)


class RigorVersionField(ArchiveInsightModel):
    """One materialization version that a row carries."""

    name: str
    current_version: int
    payload_path: tuple[str, ...] = ()


class RigorFieldContract(ArchiveInsightModel):
    """Evidence contract for one quantitative (number-bearing) field (9e5.29).

    A rendered number is a claim. This contract makes the claim's grounding
    explicit so renderers and the audit can tell "true zero" (denominator
    present, value genuinely zero), "not-applicable" (no denominator to
    compute over -- e.g. an average with zero backing rows), and "absent"
    (evidence was never gathered) apart instead of collapsing all three to
    ``0.0``.

    Fields:
        field_path: dotted path to the numeric field on the insight payload.
        provenance_class: ``counted`` (direct SQL COUNT/SUM), ``derived``
            (computed from other counted fields, e.g. an average or ratio),
            or ``estimated`` (catalog/heuristic pricing or inference).
        denominator_field: dotted path to the field whose zero-ness makes
            this field not-applicable (e.g. ``session_count`` for an
            average-per-session field). Empty tuple means the field has
            no denominator (a plain count/sum, where zero is always a
            true measured zero).
        nullable_when_ungrounded: when True, the field must render ``None``
            (never ``0``/``0.0``) when its denominator is zero or its
            backing frame is empty.
        evidence_tier: short label for what grounds the field, e.g.
            ``"sql-aggregate"``, ``"catalog-priced"``, ``"heuristic"``.
    """

    field_path: tuple[str, ...]
    provenance_class: str
    denominator_field: tuple[str, ...] = ()
    nullable_when_ungrounded: bool = True
    evidence_tier: str = "sql-aggregate"


class RigorContract(ArchiveInsightModel):
    """Per-product rigor contract matrix entry.

    Fields:
        insight_name: matches the registry ``InsightType.name``.
        display_name: human label for tables and reports.
        evidence_payload: dotted path to the evidence payload field on the
            insight item (e.g. ``("evidence",)``). Empty tuple means the
            product is aggregate-only with no direct evidence payload.
        inference_payload: dotted path to the inference payload field.
            Empty tuple means the product is evidence-only or aggregate.
        fallback_markers: payload-dotted paths whose truthy value flags
            a row as having taken a fallback rather than a fully-grounded
            inference (e.g. ``("inference", "fallback_inference")``).
        confidence_field: payload-dotted path to a ``[0, 1]`` confidence
            score, when the product carries one. Empty tuple means no
            confidence score is exposed.
        readiness_semantics: short prose describing how callers should
            decide whether a row is consumable.
        consumer_fields: stable surface fields that consumers may rely
            on. Listed for documentation and contract review; not
            enforced at runtime.
        version_fields: materialization version fields the row carries.
            Used by the audit runner to count stale-version rows.
        field_contracts: per-field evidence contracts (9e5.29) for this
            product's quantitative fields. Not required to cover every
            numeric field on day one -- coverage grows incrementally --
            but any field listed here is a committed promise: it must
            render ``None``, never ``0``/``0.0``, when its declared
            denominator is zero or ungrounded.
        notes: optional free-form notes (deprecated fields, transition
            anchors, etc.).
    """

    insight_name: str
    display_name: str
    evidence_payload: tuple[str, ...] = ()
    inference_payload: tuple[str, ...] = ()
    fallback_markers: tuple[tuple[str, ...], ...] = ()
    confidence_field: tuple[str, ...] = ()
    readiness_semantics: str = ""
    consumer_fields: tuple[str, ...] = ()
    version_fields: tuple[RigorVersionField, ...] = ()
    field_contracts: tuple[RigorFieldContract, ...] = ()
    notes: str = ""


_RIGOR_MATRIX: tuple[RigorContract, ...] = (
    RigorContract(
        insight_name="session_profiles",
        display_name="Session Profiles",
        evidence_payload=("evidence",),
        inference_payload=("inference",),
        fallback_markers=(("enrichment", "fallback_reasons"),),
        confidence_field=("enrichment", "confidence"),
        readiness_semantics=(
            "Evidence payload is fully grounded in archive counts and timestamps. "
            "Inference payload is probabilistic — consult ``inference.support_level`` "
            "and ``inference.engaged_duration_source`` for grounding. "
            "Enrichment payload is also probabilistic and carries intent/outcome "
            "summaries plus ``enrichment.support_level`` / ``enrichment.confidence``. "
            "Profiles missing an ``inference`` payload should be treated as "
            "evidence-only."
        ),
        consumer_fields=(
            "session_id",
            "source_name",
            "title",
            "semantic_tier",
            "evidence",
            "inference",
            "enrichment",
            "provenance",
            "inference_provenance",
            "enrichment_provenance",
        ),
        version_fields=(
            RigorVersionField(name="materializer_version", current_version=SESSION_INSIGHT_MATERIALIZER_VERSION),
            RigorVersionField(name="inference_version", current_version=SESSION_INFERENCE_VERSION),
            RigorVersionField(name="enrichment_version", current_version=SESSION_ENRICHMENT_VERSION),
        ),
    ),
    RigorContract(
        insight_name="session_work_events",
        display_name="Work Events",
        evidence_payload=("evidence",),
        inference_payload=("inference",),
        fallback_markers=(("inference", "fallback_inference"),),
        confidence_field=("inference", "confidence"),
        readiness_semantics=(
            "Evidence payload describes the message-range and timing footprint "
            "of the event. Inference payload carries heuristic label/summary; rows with "
            "``inference.fallback_inference == True`` were emitted by the "
            "heuristic fallback and should be treated as low-rigor."
        ),
        consumer_fields=(
            "event_id",
            "session_id",
            "source_name",
            "event_index",
            "evidence",
            "inference",
        ),
        version_fields=(
            RigorVersionField(name="materializer_version", current_version=SESSION_INSIGHT_MATERIALIZER_VERSION),
            RigorVersionField(name="inference_version", current_version=SESSION_INFERENCE_VERSION),
        ),
    ),
    RigorContract(
        insight_name="session_phases",
        display_name="Session Phases",
        evidence_payload=("evidence",),
        inference_payload=(),
        fallback_markers=(),
        confidence_field=(),
        readiness_semantics=(
            "Evidence payload describes the phase's message-range timing and "
            "tool counts. Phases are deterministic time-gap intervals, not "
            "intent labels or probabilistic workflow classifications; consumers "
            "that need intent should use work-event heuristics or session-level "
            "workflow fields."
        ),
        consumer_fields=(
            "phase_id",
            "session_id",
            "source_name",
            "phase_index",
            "evidence",
        ),
        version_fields=(
            RigorVersionField(name="materializer_version", current_version=SESSION_INSIGHT_MATERIALIZER_VERSION),
        ),
    ),
    RigorContract(
        insight_name="threads",
        display_name="Work Threads",
        evidence_payload=("thread",),
        inference_payload=(),
        fallback_markers=(),
        confidence_field=(),
        readiness_semantics=(
            "Thread payload is a deterministic rollup over session "
            "parent/child links; there is no probabilistic inference layer. "
            "Rigor is governed by the underlying parent-link evidence."
        ),
        consumer_fields=("thread_id", "root_id", "dominant_repo", "thread"),
        version_fields=(
            RigorVersionField(name="materializer_version", current_version=SESSION_INSIGHT_MATERIALIZER_VERSION),
        ),
    ),
    RigorContract(
        insight_name="session_tag_rollups",
        display_name="Session Tag Rollups",
        evidence_payload=(),
        inference_payload=(),
        fallback_markers=(),
        confidence_field=(),
        readiness_semantics=(
            "Tag rollups aggregate explicit and auto tag counts. Auto-tag "
            "rows derive from probabilistic enrichment; explicit-tag rows "
            "are direct evidence. Inspect ``explicit_count`` vs "
            "``auto_count`` for rigor."
        ),
        consumer_fields=(
            "tag",
            "session_count",
            "explicit_count",
            "auto_count",
            "origin_breakdown",
            "repo_breakdown",
        ),
        version_fields=(
            RigorVersionField(name="materializer_version", current_version=SESSION_INSIGHT_MATERIALIZER_VERSION),
        ),
    ),
    RigorContract(
        insight_name="archive_coverage",
        display_name="Archive Coverage",
        evidence_payload=(),
        inference_payload=("work_event_breakdown",),
        fallback_markers=(),
        confidence_field=(),
        readiness_semantics=(
            "Session/message/word/cost/duration counts and origin/repo breakdowns are "
            "deterministic SQL aggregates over sessions, session_profiles, session_repos, "
            "and session_work_events. ``work_event_breakdown`` is the one probabilistic "
            "field: its keys are the heuristic ``work_event_type`` labels the session_work_events "
            "materializer assigns (see that contract's fallback_inference marker), so it is an "
            "aggregation over inferred labels, not raw evidence. ``provenance`` is only populated "
            "for day/week grouping (never for the default provider grouping), so consumers must "
            "not assume a materialization timestamp/version is always present."
        ),
        consumer_fields=(
            "bucket",
            "group_by",
            "source_name",
            "session_count",
            "message_count",
            "total_cost_usd",
            "tool_use_percentage",
            "thinking_percentage",
            "work_event_breakdown",
            "origin_breakdown",
            "repos_active",
        ),
        version_fields=(),
        field_contracts=(
            RigorFieldContract(
                field_path=("avg_messages_per_session",),
                provenance_class="derived",
                denominator_field=("session_count",),
            ),
            RigorFieldContract(
                field_path=("avg_user_words",),
                provenance_class="derived",
                denominator_field=("user_message_count",),
            ),
            RigorFieldContract(
                field_path=("avg_authored_user_words",),
                provenance_class="derived",
                denominator_field=("authored_user_message_count",),
            ),
            RigorFieldContract(
                field_path=("avg_assistant_words",),
                provenance_class="derived",
                denominator_field=("assistant_message_count",),
            ),
            RigorFieldContract(
                field_path=("tool_use_percentage",),
                provenance_class="derived",
                denominator_field=("session_count",),
            ),
            RigorFieldContract(
                field_path=("thinking_percentage",),
                provenance_class="derived",
                denominator_field=("session_count",),
            ),
        ),
        notes=(
            "provenance.materializer_version is a hardcoded literal 1 for day/week grouping "
            "(archive.py) with no dedicated store_constants entry, and absent entirely for "
            "provider grouping -- not declared as a version_field to avoid implying a real "
            "materialized-artifact version exists. The six field_contracts entries above "
            "(9e5.29) render None -- never 0.0 -- whenever their declared denominator is zero; "
            "day/week grouping does not compute avg_user_words/avg_authored_user_words/"
            "avg_assistant_words/tool_use_percentage/thinking_percentage at all (no per-type "
            "message counts fetched there), so those five render None for every day/week row "
            "today, not just zero-denominator ones -- a real, documented coverage gap, not a bug."
        ),
    ),
    RigorContract(
        insight_name="tool_usage",
        display_name="Tool Usage",
        evidence_payload=(),
        inference_payload=(),
        fallback_markers=(("has_coverage_gaps",),),
        confidence_field=(),
        readiness_semantics=(
            "Every field is a deterministic count, distinct-value count, or presence flag "
            "read straight from the canonical actions view; there is no heuristic/estimate "
            "layer. ``mcp_server`` is a deterministic string parse, not an inference. Consumers "
            "should check ``has_coverage_gaps`` (or the per-entry "
            "``provider_coverage[].data_available``) to distinguish a genuine zero tool-use "
            "count from an origin with no ingested action data at all."
        ),
        consumer_fields=(
            "entries",
            "provider_coverage",
            "total_call_count",
            "total_distinct_tools",
            "providers_with_data",
            "providers_without_data",
            "has_coverage_gaps",
        ),
        version_fields=(RigorVersionField(name="materializer_version", current_version=TOOL_USAGE_INSIGHT_VERSION),),
    ),
    RigorContract(
        insight_name="session_costs",
        display_name="Session Costs",
        evidence_payload=(),
        inference_payload=("estimate",),
        fallback_markers=(
            ("estimate", "missing_reasons"),
            ("estimate", "unavailable_reason"),
        ),
        confidence_field=("estimate", "confidence"),
        readiness_semantics=(
            "session_id/source_name/title/timestamps are direct archive facts. The nested "
            "``estimate`` payload carries the pricing outcome: ``estimate.status`` is one of "
            "exact/priced/partial/unavailable, ``estimate.confidence`` quantifies trust in a "
            "non-exact price (1.0 exact, 0.9 priced, 0.7 priced-and-flagged-estimated, 0.0 "
            "unavailable), and a non-empty ``estimate.missing_reasons`` or a set "
            "``estimate.unavailable_reason`` flags a fallback/unpriced row."
        ),
        consumer_fields=("session_id", "source_name", "title", "created_at", "updated_at", "estimate", "provenance"),
        version_fields=(
            RigorVersionField(name="materializer_version", current_version=SESSION_INSIGHT_MATERIALIZER_VERSION),
        ),
    ),
    RigorContract(
        insight_name="cost_rollups",
        display_name="Cost Rollups",
        evidence_payload=(),
        inference_payload=(),
        fallback_markers=(("unavailable_session_count",),),
        confidence_field=("confidence",),
        readiness_semantics=(
            "session/priced/unavailable counts, status_counts, total_usd, basis, and usage are "
            "grounded SQL sums/counts over stored per-session cost/usage rows. ``confidence`` is "
            "the one probabilistic signal -- a session-count-weighted average of the same "
            "per-row confidence heuristic session_costs uses -- and should be read alongside "
            "``unavailable_session_count``/``status_counts`` to judge how much of the rollup "
            "rests on exact vs. estimated pricing."
        ),
        consumer_fields=(
            "source_name",
            "model_name",
            "normalized_model",
            "session_count",
            "priced_session_count",
            "unavailable_session_count",
            "status_counts",
            "total_usd",
            "basis",
            "usage",
            "confidence",
            "per_model_breakdown",
        ),
        version_fields=(),
        notes=(
            "provenance.materializer_version is a hardcoded literal 0 (archive.py), a sentinel "
            "for 'computed live at query time', not a stored materialized artifact -- not "
            "declared as a version_field."
        ),
    ),
    RigorContract(
        insight_name="usage_timeline",
        display_name="Usage Timeline",
        evidence_payload=(),
        inference_payload=(),
        fallback_markers=(),
        confidence_field=(),
        readiness_semantics=(
            "session/event counts, token usage, and stored_cost_usd are grounded SQL sums over "
            "usage-event rows. ``subscription_credits`` is a catalog-rate estimate "
            "(compute_credit_cost) whenever stored credits are absent, indistinguishable in the "
            "payload from a genuinely stored credit figure -- inspect ``cost_provenance_counts`` "
            "(a count-of-labels dict: exact/priced/estimated/unknown) to judge how much of the "
            "bucket's cost basis is exact vs. estimated. Not declared as a fallback_markers entry "
            "since it does not follow the empty-then-truthy-on-fallback convention other "
            "contracts use."
        ),
        consumer_fields=(
            "bucket",
            "group_by",
            "source_name",
            "model_name",
            "normalized_model",
            "session_count",
            "event_count",
            "usage",
            "reasoning_output_tokens",
            "stored_cost_usd",
            "subscription_credits",
            "cost_provenance_counts",
        ),
        version_fields=(),
        notes=(
            "provenance.materializer_version is a hardcoded literal 0 (archive.py), the same "
            "live-aggregation sentinel as cost_rollups -- not declared as a version_field."
        ),
    ),
    RigorContract(
        insight_name="archive_debt",
        display_name="Archive Debt",
        evidence_payload=(),
        inference_payload=(),
        fallback_markers=(),
        confidence_field=(),
        readiness_semantics=(
            "Every row is a live, deterministic health-check result over current archive "
            "tables (FTS sync, orphaned profile rows, materialization staleness, etc.) with no "
            "inference or fallback layer. ``healthy`` is literally ``issue_count == 0``, not an "
            "estimate, and can be trusted at face value with no confidence caveat."
        ),
        consumer_fields=(
            "debt_name",
            "category",
            "maintenance_target",
            "issue_count",
            "healthy",
            "destructive",
            "detail",
        ),
        version_fields=(),
        notes=(
            "No materializer/inference/enrichment version field exists on this model at all -- "
            "every row is computed live from current archive tables, not read from a "
            "materialized artifact with a version to track."
        ),
    ),
)

#: Registered insight products deliberately excluded from the rigor matrix
#: because they carry no number-bearing/quantitative fields at all (37t.15
#: sibling, 9e5.28). Every entry needs an inline justification string; a
#: number-bearing product belongs in ``_RIGOR_MATRIX`` above, not here --
#: ``devtools lab policy insight-honesty`` fails on any registered insight
#: that is in neither set.
RIGOR_EXEMPT: dict[str, str] = {}


def list_rigor_contracts() -> tuple[RigorContract, ...]:
    """Return the immutable per-product rigor contract matrix."""

    return _RIGOR_MATRIX


def get_rigor_contract(insight_name: str) -> RigorContract | None:
    """Lookup a rigor contract by its insight registry name."""

    for contract in _RIGOR_MATRIX:
        if contract.insight_name == insight_name:
            return contract
    return None


def rigor_contract_names() -> tuple[str, ...]:
    """Return the insight names covered by the matrix in declaration order."""

    return tuple(contract.insight_name for contract in _RIGOR_MATRIX)


def rigor_exemption_reason(insight_name: str) -> str | None:
    """Return the justification for exempting ``insight_name``, or ``None``."""

    return RIGOR_EXEMPT.get(insight_name)


def resolve_payload(obj: object, path: Sequence[str]) -> object | None:
    """Walk a dotted attribute/key path against an insight item.

    The path can mix model attributes and dict keys. Returns ``None`` when
    any segment is missing or the result itself is ``None``.
    """

    current: object | None = obj
    for segment in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(segment)
            continue
        current = getattr(current, segment, None)
    return current


__all__ = [
    "RIGOR_EXEMPT",
    "RigorContract",
    "RigorFieldContract",
    "RigorVersionField",
    "get_rigor_contract",
    "list_rigor_contracts",
    "resolve_payload",
    "rigor_contract_names",
    "rigor_exemption_reason",
]
