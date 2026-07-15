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
- documentation: the human-readable matrix lives at
  ``docs/insights-rigor-matrix.md`` and is updated alongside this module.
- consumer self-discovery: future MCP and API surfaces can query the
  matrix instead of reading prose docs.

This module is contract-only; it does not query the archive. The audit
runner lives in :mod:`polylogue.insights.audit`.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import NoneType
from typing import get_args, get_origin

from pydantic import BaseModel

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


class RigorFieldExemption(ArchiveInsightModel):
    """Explicit justification for one numeric field whose zero is measured.

    This is deliberately field-level, rather than a product-level note: a
    newly exposed numeric field must receive its own declared rationale.
    """

    field_path: tuple[str, ...]
    reason: str


def _true_zero_fields(reason: str, *names: str) -> tuple[RigorFieldExemption, ...]:
    return tuple(RigorFieldExemption(field_path=(name,), reason=reason) for name in names)


def _true_zero_paths(reason: str, *paths: tuple[str, ...]) -> tuple[RigorFieldExemption, ...]:
    """Declare exact nested numeric paths whose zeros are meaningful values."""

    return tuple(RigorFieldExemption(field_path=path, reason=reason) for path in paths)


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
        field_exemptions: explicit field-level reasons that a numeric zero is
            a real measured value rather than absent evidence.
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
    field_exemptions: tuple[RigorFieldExemption, ...] = ()
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
        field_exemptions=(
            *_true_zero_paths(
                "Evidence counts, durations, and ratios are materialized archive measurements.",
                *(
                    ("evidence", name)
                    for name in (
                        "attachment_count",
                        "compaction_count",
                        "latency_percentiles_ms",
                        "message_count",
                        "output_duration_ms",
                        "substantive_count",
                        "thinking_count",
                        "thinking_duration_ms",
                        "timestamped_message_count",
                        "tool_active_duration_ms",
                        "tool_calls_per_minute",
                        "tool_categories",
                        "tool_duration_ms",
                        "tool_use_count",
                        "total_cache_read_tokens",
                        "total_cache_write_tokens",
                        "total_cost_usd",
                        "total_credit_cost",
                        "total_duration_ms",
                        "total_input_tokens",
                        "total_output_tokens",
                        "untimestamped_message_count",
                        "wall_duration_ms",
                        "word_count",
                    )
                ),
            ),
            *_true_zero_paths(
                "Inference scores and durations are explicit heuristic outputs, never missing-evidence sentinels.",
                *(
                    ("inference", name)
                    for name in (
                        "engaged_duration_ms",
                        "engaged_minutes",
                        "phase_count",
                        "terminal_state_confidence",
                        "tool_active_duration_ms",
                        "tool_active_minutes",
                        "work_event_count",
                        "workflow_shape_confidence",
                    )
                ),
            ),
            *_true_zero_paths(
                "Enrichment scores are explicit heuristic outputs, never missing-evidence sentinels.",
                ("enrichment", "confidence"),
                ("enrichment", "input_band_summary"),
            ),
        ),
        notes=(
            "Heuristic-tier inventory (polylogue-b0b): `inference.terminal_state` "
            "(`archive/session/runtime.py::_terminal_state`) now prefers a "
            "structural, session-wide `tool_id -> outcome` map "
            "(`_session_tool_results`) sourced from the keystone "
            "`blocks.tool_result_is_error`/`tool_result_exit_code` columns "
            "(index schema v16) over the prior prose `_ERROR_MARKERS` keyword "
            "scan for the mid-session error-action signal. The lookup is "
            "session-wide (mirroring `_pending_tool_blocks` and "
            "`insights/transforms.py::_extract_events`) rather than routed "
            "through the per-message `Action`/`ToolCall` pairing, because "
            "Claude/Codex-style transcripts near-always place a `tool_use` in "
            "one message and its `tool_result` in a later message -- "
            "per-message pairing alone would miss the common case. That "
            "structural signal is origin-gated, not universal "
            "(polylogue-9e5.3 audit): `tool_result_is_error` is well-populated "
            "only for claude-code-session (44.8%) and claude-ai-export (100% "
            "of a small volume), 0% for chatgpt-export/hermes-session/"
            "aistudio-drive; `tool_result_exit_code` is populated only for "
            "codex-session, and just 14.2% of even that. For origins/results "
            "with no structural coverage the code falls back to the tagged "
            "text scan rather than reporting a false negative. Every branch "
            "of `_terminal_state` now returns an `evidence_class` key in "
            "`inference.terminal_state_evidence` -- `raw_evidence` (tool-pairing "
            "counts, the structural action signal, the provider-emitted "
            "session-event status field, or message role) or `text_derived` "
            "(the last-message `_ERROR_MARKERS` scan and its `clean_finish` "
            "complement, and the structural-fallback text scan above). "
            "Consumers needing only grounded rows should filter on "
            "`inference.terminal_state_evidence.evidence_class == 'raw_evidence'`. "
            "`session_work_events`' sibling classifier's 50.5% (coin-flip) "
            "accuracy finding (9e5.9, noted below) is the reason this scan was "
            "not simply deleted: not measured as reliable, but not proven "
            "unreliable enough to remove entirely while the last-resort "
            "fallback path still has explicit provenance."
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
        field_exemptions=(
            *_true_zero_fields(
                "Event index is an ordinal identity component, not an aggregate claim.",
                "event_index",
            ),
            *_true_zero_paths(
                "Event evidence counts, offsets, and durations are materialized archive measurements.",
                ("evidence", "duration_ms"),
                ("evidence", "end_index"),
                ("evidence", "start_index"),
            ),
            *_true_zero_paths(
                "Event inference confidence is an explicit heuristic output, never a missing-evidence sentinel.",
                ("inference", "confidence"),
            ),
        ),
        notes=(
            "Heuristic-tier inventory (#b0b.1): the activity-type classifier "
            "(``inference.heuristic_label`` -- planning/debugging/testing/review/"
            "refactoring/documentation/configuration/data_analysis, "
            "archive/session/extraction.py _TEXT_SIGNAL_TABLE) has no structural "
            "signal to convert to -- unlike outcome/pathology fields "
            "(tool_result_is_error, tool_result_exit_code), there is no structural "
            "proxy for 'what category of work is this'; keyword text matching "
            "against user messages is the only available signal, and it is a "
            "fallback checked only after action-category (tool-use) evidence, per "
            "_classify_range. As of #b0b.1 the keyword match is word-boundary-"
            "anchored (previously a naive substring check that false-positived on "
            "unrelated words, e.g. 'fix' inside 'prefix', 'test' inside 'latest', "
            "'config' inside 'reconfigured') -- a correctness fix to the matching "
            "mechanism, not a claim about its predictive value. "
            "UNVERIFIED ACCURACY (9e5.9): this field's real-world precision has "
            "never been measured against ground truth -- 9e5.9's closing evidence "
            "found the sibling keyword heuristic in the same file (runtime.py "
            "_terminal_state's _ERROR_MARKERS fallback) scores only 50.5% "
            "agreement (coin-flip level) against structural ground truth "
            "(tool_result_is_error/exit_code) on 14,377 real runs. Do not treat "
            "``heuristic_label`` as a reliable signal until this classifier gets "
            "its own accuracy measurement; consumers should treat it as a weak, "
            "unverified prior, not a trustworthy label."
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
        field_exemptions=(
            *_true_zero_fields(
                "Phase index is an ordinal identity component, not an aggregate claim.",
                "phase_index",
            ),
            *_true_zero_paths(
                "Phase evidence counts, offsets, and durations are materialized archive measurements.",
                *(
                    ("evidence", name)
                    for name in (
                        "duration_ms",
                        "message_range",
                        "phase_idle_threshold_ms",
                        "tool_counts",
                        "word_count",
                    )
                ),
            ),
            *_true_zero_paths(
                "Phase inference confidence is an explicit heuristic output, never a missing-evidence sentinel.",
                ("inference", "confidence"),
            ),
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
        field_exemptions=(
            *_true_zero_paths(
                "Thread counts, depth, and confidence are deterministic rollups over resolved session links.",
                *(
                    ("thread", name)
                    for name in (
                        "branch_count",
                        "confidence",
                        "depth",
                        "origin_breakdown",
                        "session_count",
                        "total_cost_usd",
                        "total_messages",
                        "wall_duration_ms",
                        "work_event_breakdown",
                    )
                ),
                ("thread", "member_evidence", "confidence"),
                ("thread", "member_evidence", "depth"),
            ),
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
        field_exemptions=_true_zero_fields(
            "Tag counts and breakdowns are direct aggregate counts; zero means no matching tagged rows.",
            "session_count",
            "logical_session_count",
            "explicit_count",
            "auto_count",
            "origin_breakdown",
            "repo_breakdown",
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
        field_exemptions=_true_zero_fields(
            "Coverage counts, sums, and breakdowns are direct SQL aggregates; zero is a measured empty aggregate.",
            "session_count",
            "logical_session_count",
            "message_count",
            "user_message_count",
            "authored_user_message_count",
            "assistant_message_count",
            "total_cost_usd",
            "total_duration_ms",
            "total_tool_active_duration_ms",
            "total_wall_duration_ms",
            "total_words",
            "tool_use_count",
            "thinking_count",
            "total_sessions_with_tools",
            "total_sessions_with_thinking",
            "work_event_breakdown",
            "origin_breakdown",
        ),
        notes=(
            "provenance.materializer_version is a hardcoded literal 1 for day/week grouping "
            "(archive.py) with no dedicated store_constants entry, and absent entirely for "
            "origin grouping -- not declared as a version_field to avoid implying a real "
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
            "``origin_coverage[].data_available``) to distinguish a genuine zero tool-use "
            "count from an origin with no ingested action data at all."
        ),
        consumer_fields=(
            "entries",
            "origin_coverage",
            "total_call_count",
            "total_distinct_tools",
            "origins_with_data",
            "origins_without_data",
            "has_coverage_gaps",
        ),
        version_fields=(RigorVersionField(name="materializer_version", current_version=TOOL_USAGE_INSIGHT_VERSION),),
        field_exemptions=(
            *_true_zero_fields(
                "Tool-usage totals are direct action-view counts; zero means no matching action rows.",
                "total_call_count",
                "total_distinct_tools",
                "origins_with_data",
                "origins_without_data",
                "materializer_version",
            ),
            *_true_zero_paths(
                "Per-tool entries are direct action-view counts and coverage totals.",
                *(
                    ("entries", name)
                    for name in (
                        "affected_path_calls",
                        "call_count",
                        "distinct_tool_ids",
                        "message_count",
                        "output_text_calls",
                        "session_count",
                    )
                ),
            ),
            *_true_zero_paths(
                "Per-origin coverage values are direct action-view counts.",
                *(
                    ("origin_coverage", name)
                    for name in (
                        "action_count",
                        "distinct_action_kind_count",
                        "distinct_tool_count",
                        "session_count",
                    )
                ),
            ),
        ),
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
        field_exemptions=(
            *_true_zero_paths(
                "Pricing outcome values and confidence are explicit catalog-pricing results; zero confidence means unavailable pricing, not absent output.",
                *(("estimate", name) for name in ("confidence", "total_usd")),
                *(
                    ("estimate", "basis", name)
                    for name in (
                        "api_equivalent_usd",
                        "catalog_priced_usd",
                        "provider_reported_usd",
                        "subscription_equivalent_usd",
                        "tool_surcharge_usd",
                    )
                ),
                *(("estimate", "components", name) for name in ("tokens", "usd")),
                *(
                    ("estimate", "usage", name)
                    for name in (
                        "cache_read_tokens",
                        "cache_write_tokens",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                    )
                ),
                *(
                    ("estimate", "price", name)
                    for name in (
                        "cache_read_usd_per_1m",
                        "cache_write_usd_per_1m",
                        "input_usd_per_1m",
                        "output_usd_per_1m",
                    )
                ),
                ("estimate", "per_model_breakdown", "session_count"),
                ("estimate", "per_model_breakdown", "total_usd"),
                *(
                    ("estimate", "per_model_breakdown", "basis", name)
                    for name in (
                        "api_equivalent_usd",
                        "catalog_priced_usd",
                        "provider_reported_usd",
                        "subscription_equivalent_usd",
                        "tool_surcharge_usd",
                    )
                ),
                *(
                    ("estimate", "per_model_breakdown", "usage", name)
                    for name in (
                        "cache_read_tokens",
                        "cache_write_tokens",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                    )
                ),
            ),
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
        field_contracts=(
            RigorFieldContract(
                field_path=("confidence",),
                provenance_class="derived",
                denominator_field=("priced_session_count",),
                evidence_tier="cost-pricing-rollup",
            ),
        ),
        field_exemptions=(
            *_true_zero_fields(
                "Cost-rollup counts, cost sum, and reason/status breakdowns are direct aggregate measurements; zero is measured.",
                "session_count",
                "priced_session_count",
                "unavailable_session_count",
                "status_counts",
                "total_usd",
                "unavailable_reason_counts",
            ),
            *_true_zero_paths(
                "Cost-basis values are direct sums of recorded or catalog-priced costs.",
                *(
                    ("basis", name)
                    for name in (
                        "api_equivalent_usd",
                        "catalog_priced_usd",
                        "provider_reported_usd",
                        "subscription_equivalent_usd",
                        "tool_surcharge_usd",
                    )
                ),
            ),
            *_true_zero_paths(
                "Usage values are direct sums over stored usage rows.",
                *(
                    ("usage", name)
                    for name in (
                        "cache_read_tokens",
                        "cache_write_tokens",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                    )
                ),
            ),
            *_true_zero_paths(
                "Per-model breakdown values are direct grouped aggregates.",
                ("per_model_breakdown", "session_count"),
                ("per_model_breakdown", "total_usd"),
                *(
                    ("per_model_breakdown", "basis", name)
                    for name in (
                        "api_equivalent_usd",
                        "catalog_priced_usd",
                        "provider_reported_usd",
                        "subscription_equivalent_usd",
                        "tool_surcharge_usd",
                    )
                ),
                *(
                    ("per_model_breakdown", "usage", name)
                    for name in (
                        "cache_read_tokens",
                        "cache_write_tokens",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                    )
                ),
            ),
        ),
        notes=(
            "provenance.materializer_version is a hardcoded literal 0 (archive.py), a sentinel "
            "for 'computed live at query time', not a stored materialized artifact -- not "
            "declared as a version_field. ``confidence`` is null when no priced sessions "
            "provide a denominator; other numeric fields are direct counts or sums where "
            "zero is a measured aggregate."
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
        field_exemptions=(
            *_true_zero_fields(
                "Usage-timeline counts, token totals, cost totals, and provenance breakdowns are direct aggregate measurements.",
                "session_count",
                "event_count",
                "reasoning_output_tokens",
                "stored_cost_usd",
                "subscription_credits",
                "cost_provenance_counts",
            ),
            *_true_zero_paths(
                "Usage values are direct sums over usage-event rows.",
                *(
                    ("usage", name)
                    for name in (
                        "cache_read_tokens",
                        "cache_write_tokens",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                    )
                ),
            ),
        ),
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
        field_exemptions=_true_zero_fields(
            "Debt issue_count is a direct health-check count; zero means the check found no issues.",
            "issue_count",
        ),
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

_METADATA_FIELD_NAMES = frozenset(
    {"contract_version", "provenance", "inference_provenance", "enrichment_provenance", "materializer_version"}
)


def _numeric_paths_for_annotation(
    annotation: object,
    path: tuple[str, ...],
    seen_models: frozenset[type[BaseModel]],
) -> frozenset[tuple[str, ...]]:
    if annotation in (int, float):
        return frozenset({path})

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is dict:
        return _numeric_paths_for_annotation(args[1], path, seen_models) if len(args) == 2 else frozenset()
    if args:
        return frozenset().union(*(_numeric_paths_for_annotation(arg, path, seen_models) for arg in args))
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation in seen_models:
            return frozenset()
        return frozenset().union(
            *(
                _numeric_paths_for_annotation(field.annotation, (*path, field_name), seen_models | {annotation})
                for field_name, field in annotation.model_fields.items()
                if field_name not in _METADATA_FIELD_NAMES
            )
        )
    return frozenset()


def missing_numeric_item_models() -> tuple[str, ...]:
    """Return registered insight types that lack an inspectable item model."""

    from polylogue.insights.registry import INSIGHT_REGISTRY

    return tuple(sorted(name for name, insight_type in INSIGHT_REGISTRY.items() if insight_type.item_model is None))


def _model_type_for_annotation(annotation: object) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for argument in get_args(annotation):
        if model := _model_type_for_annotation(argument):
            return model
    return None


def _annotation_at_path(model: type[BaseModel], path: Sequence[str]) -> object | None:
    current: type[BaseModel] | None = model
    annotation: object | None = None
    for segment in path:
        if current is None or (field := current.model_fields.get(segment)) is None:
            return None
        annotation = field.annotation
        current = _model_type_for_annotation(annotation)
    return annotation


def _annotation_allows_none(annotation: object | None) -> bool:
    return annotation is NoneType or NoneType in get_args(annotation)


def invalid_nullable_field_contracts(
    contracts: Sequence[RigorContract] | None = None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return contracts that promise null-over-empty on a non-nullable field."""

    from polylogue.insights.registry import INSIGHT_REGISTRY

    invalid: list[tuple[str, tuple[str, ...]]] = []
    for contract in contracts or _RIGOR_MATRIX:
        item_model = INSIGHT_REGISTRY.get(contract.insight_name, None)
        model = item_model.item_model if item_model is not None else None
        for field_contract in contract.field_contracts:
            annotation = _annotation_at_path(model, field_contract.field_path) if model is not None else None
            if not field_contract.nullable_when_ungrounded or not _annotation_allows_none(annotation):
                invalid.append((contract.insight_name, field_contract.field_path))
    return tuple(sorted(invalid))


def numeric_insight_field_paths() -> frozenset[tuple[str, tuple[str, ...]]]:
    """Recursively discover public numeric leaves from registered item models."""

    from polylogue.insights.registry import INSIGHT_REGISTRY

    return frozenset(
        (insight_name, field_path)
        for insight_name, insight_type in INSIGHT_REGISTRY.items()
        if insight_type.item_model is not None
        for field_path in _numeric_paths_for_annotation(insight_type.item_model, (), frozenset())
    )


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


def missing_numeric_field_coverage(
    contracts: Sequence[RigorContract] | None = None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return public numeric fields lacking a contract or explicit rationale."""

    declared = {
        (contract.insight_name, field.field_path)
        for contract in (contracts or _RIGOR_MATRIX)
        for field in contract.field_contracts
    }
    exemptions = {
        (contract.insight_name, field.field_path)
        for contract in (contracts or _RIGOR_MATRIX)
        for field in contract.field_exemptions
    }
    return tuple(sorted(numeric_insight_field_paths() - declared - exemptions))


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
    "RigorFieldExemption",
    "RigorVersionField",
    "get_rigor_contract",
    "invalid_nullable_field_contracts",
    "list_rigor_contracts",
    "missing_numeric_item_models",
    "missing_numeric_field_coverage",
    "numeric_insight_field_paths",
    "resolve_payload",
    "rigor_contract_names",
    "rigor_exemption_reason",
]
