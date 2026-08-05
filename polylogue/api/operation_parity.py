"""Semantic-operation authority for the documented async Python facade.

The public :class:`polylogue.api.Polylogue` facade is intentionally broader
than any individual transport.  This declaration maps every callable on that
facade to a stable semantic operation, or records why an exported callable is
not a semantic operation.  Live inspection is used only to reject drift; it
does not assign operation IDs.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

API_PARITY_AUTHORITY = "polylogue-s1kr"

RouteClass = Literal[
    "lifecycle",
    "source-index-write",
    "source-read",
    "index-read",
    "index-write",
    "user-read",
    "user-write",
    "cross-tier",
    "embedding-status",
    "embedding-preflight",
    "embedding-read",
]


@dataclass(frozen=True, slots=True)
class SurfaceBinding:
    """A cross-surface name or an intentional, owned absence."""

    names: tuple[str, ...] = ()
    intentional_absence_authority: str | None = None

    def __post_init__(self) -> None:
        if bool(self.names) == bool(self.intentional_absence_authority):
            raise ValueError("surface binding needs names or an absence authority, exclusively")


@dataclass(frozen=True, slots=True)
class ApiOperation:
    """A semantic operation with explicit public-surface bindings."""

    operation_id: str
    section: str
    summary: str
    route_class: RouteClass
    python_bindings: tuple[str, ...]
    cli: SurfaceBinding
    mcp: SurfaceBinding

    def __post_init__(self) -> None:
        if not self.operation_id.startswith("api."):
            raise ValueError(f"operation ID must be stable and API-namespaced: {self.operation_id}")
        if not self.python_bindings:
            raise ValueError(f"{self.operation_id} must bind at least one Python callable")


@dataclass(frozen=True, slots=True)
class ApiExclusion:
    """A public callable/type deliberately outside the semantic operation map."""

    binding: str
    reason: str
    authority: str = API_PARITY_AUTHORITY


def _absence() -> SurfaceBinding:
    return SurfaceBinding(intentional_absence_authority=API_PARITY_AUTHORITY)


def _surface(*names: str) -> SurfaceBinding:
    return SurfaceBinding(names=names)


API_OPERATIONS: tuple[ApiOperation, ...] = (
    ApiOperation(
        "api.lifecycle.construct",
        "Lifecycle and builders",
        "Construct, open, and close a facade bound to one archive runtime.",
        "lifecycle",
        (
            "Polylogue",
            "Polylogue.__init__",
            "Polylogue.open",
            "Polylogue.__aenter__",
            "Polylogue.__aexit__",
            "Polylogue.close",
        ),
        _absence(),
        _absence(),
    ),
    ApiOperation(
        "api.embedding.status",
        "Embedding readiness",
        "Read the no-spend embedding readiness state.",
        "embedding-status",
        ("Polylogue.embedding_status",),
        _surface("ops embed status"),
        _surface("status"),
    ),
    ApiOperation(
        "api.embedding.preflight",
        "Embedding readiness",
        "Calculate a bounded no-provider-call embedding catch-up window.",
        "embedding-preflight",
        ("Polylogue.embedding_preflight",),
        _surface("ops embed preflight"),
        _surface("status"),
    ),
    ApiOperation(
        "api.embedding.search",
        "Embedding retrieval",
        "Search stored session vectors using the embeddings tier.",
        "embedding-read",
        ("Polylogue.search_similar_sessions",),
        _surface("find similar"),
        _surface("query"),
    ),
    ApiOperation(
        "api.ingest.parse",
        "Ingestion and derived maintenance",
        "Parse configured or explicit sources into source and index tiers.",
        "source-index-write",
        ("Polylogue.parse_file", "Polylogue.parse_sources"),
        _surface("import"),
        _surface("run"),
    ),
    ApiOperation(
        "api.index.rebuild",
        "Ingestion and derived maintenance",
        "Rebuild or update the derived index through the mutation executor.",
        "index-write",
        ("Polylogue.rebuild_index", "Polylogue.update_index", "Polylogue.rebuild_insights"),
        _surface("ops reset --index"),
        _surface("maintenance"),
    ),
    ApiOperation(
        "api.archive.session-read",
        "Archive reads",
        "Read sessions, summaries, messages, actions, and archive statistics from the index tier.",
        "index-read",
        (
            "Polylogue.get_session",
            "Polylogue.get_sessions",
            "Polylogue.get_actions_batch",
            "Polylogue.list_sessions",
            "Polylogue.list_summaries",
            "Polylogue.list_sessions_for_spec",
            "Polylogue.search_session_hits",
            "Polylogue.search",
            "Polylogue.search_envelope",
            "Polylogue.archive_count_sessions",
            "Polylogue.archive_get_session",
            "Polylogue.get_messages_paginated",
            "Polylogue.iter_messages",
            "Polylogue.bulk_get_messages",
            "Polylogue.query_sessions",
            "Polylogue.count_sessions",
            "Polylogue.get_session_summary",
            "Polylogue.get_session_stats",
            "Polylogue.get_stats_by",
            "Polylogue.get_index_status",
            "Polylogue.stats",
            "Polylogue.storage_stats",
            "Polylogue.facets",
            "Polylogue.health_check",
            "Polylogue.filter",
            "Polylogue.list_read_view_profiles",
        ),
        _surface("find", "read"),
        _surface("query", "read", "get", "status"),
    ),
    ApiOperation(
        "api.archive.query-analysis",
        "Archive reads",
        "Compile, explain, diagnose, and resolve archive query and reference projections.",
        "index-read",
        (
            "Polylogue.explain_query_expression",
            "Polylogue.query_units",
            "Polylogue.query_completions",
            "Polylogue.diagnose_query_miss",
            "Polylogue.resolve_ref",
            "Polylogue.export_otel",
            "Polylogue.neighbor_candidates",
            "Polylogue.neighbor_candidate_payloads",
            "Polylogue.session_correlation_payload",
            "Polylogue.origin_usage_report",
            "Polylogue.session_usage_reconciliation",
            "Polylogue.resume_brief",
            "Polylogue.find_resume_candidates",
        ),
        _surface("find", "read", "analyze"),
        _surface("query", "read", "get", "explain"),
    ),
    ApiOperation(
        "api.archive.source-evidence-read",
        "Source evidence reads",
        "Read raw artifacts and provider-side evidence retained in the durable source tier.",
        "source-read",
        (
            "Polylogue.explain_import",
            "Polylogue.get_raw_artifacts_for_session",
            "Polylogue.get_hook_event_summary_for_session",
            "Polylogue.get_session_events",
            "Polylogue.get_file_edits",
            "Polylogue.get_web_content_constructs",
            "Polylogue.get_agent_policies",
        ),
        _surface("read", "analyze"),
        _surface("read", "explain"),
    ),
    ApiOperation(
        "api.archive.insight-read",
        "Insights and topology",
        "Read materialized archive insights, topology, and derived archive health from the index tier.",
        "index-read",
        (
            "Polylogue.get_session_insight_status",
            "Polylogue.get_session_profile_insight",
            "Polylogue.get_session_profile_record",
            "Polylogue.list_session_profile_insights",
            "Polylogue.insight_readiness_report",
            "Polylogue.insight_rigor_audit",
            "Polylogue.archive_debt",
            "Polylogue.get_session_work_event_insights",
            "Polylogue.list_session_work_event_insights",
            "Polylogue.get_session_phase_insights",
            "Polylogue.list_session_phase_insights",
            "Polylogue.get_thread_insight",
            "Polylogue.list_thread_insights",
            "Polylogue.list_session_tag_rollup_insights",
            "Polylogue.list_archive_coverage_insights",
            "Polylogue.list_tool_usage_insights",
            "Polylogue.list_session_cost_insights",
            "Polylogue.get_session_latency_profile_insight",
            "Polylogue.list_session_latency_profile_insights",
            "Polylogue.find_stuck_session_latency_profile_insights",
            "Polylogue.list_cost_rollup_insights",
            "Polylogue.list_usage_timeline_insights",
            "Polylogue.list_archive_debt_insights",
            "Polylogue.cost_outlook",
            "Polylogue.aggregate_sessions",
            "Polylogue.workflow_shape_distribution",
            "Polylogue.find_abandoned_sessions",
            "Polylogue.tool_call_latency_distribution",
            "Polylogue.compare_sessions",
            "Polylogue.find_similar_sessions_by_metadata",
            "Polylogue.correlate_sessions",
            "Polylogue.get_session_topology",
            "Polylogue.get_ancestors",
            "Polylogue.get_descendants",
            "Polylogue.get_siblings",
            "Polylogue.get_thread",
            "Polylogue.get_logical_session",
            "Polylogue.get_session_tree",
            "Polylogue.postmortem_bundle",
            "Polylogue.pathology_report",
            "Polylogue.portfolio_bundle",
            "Polylogue.export_insight_bundle",
            "Polylogue.regenerate_private_fable_packet",
        ),
        _surface("analyze", "read"),
        _surface("query", "get", "status", "explain"),
    ),
    ApiOperation(
        "api.context.delivery",
        "Context and evidence",
        "Compile context and record or inspect durable delivery receipts.",
        "cross-tier",
        (
            "Polylogue.compile_context",
            "Polylogue.context_image_payload",
            "Polylogue.context_preamble_payload",
            "Polylogue.get_context_delivery",
            "Polylogue.list_context_deliveries",
            "Polylogue.record_context_delivery",
            "Polylogue.compile_and_record_context",
            "Polylogue.correlate_hermes_context_deliveries",
            "Polylogue.reconcile_hermes_session_lifecycle",
            "Polylogue.reconcile_codex_spawn_edges",
            "Polylogue.hermes_integration_health",
        ),
        _surface("continue", "read"),
        _surface("context", "get", "status"),
    ),
    ApiOperation(
        "api.assertion.review",
        "Assertions and judgments",
        "Read, capture, and judge durable assertions and comparative evidence.",
        "cross-tier",
        (
            "Polylogue.import_annotation_batch",
            "Polylogue.list_assertion_claims",
            "Polylogue.list_assertion_claim_payloads",
            "Polylogue.list_assertion_candidates",
            "Polylogue.list_assertion_candidate_reviews",
            "Polylogue.assertion_candidate_queue_health",
            "Polylogue.judge_assertion_candidate",
            "Polylogue.capture_assertion_candidate",
            "Polylogue.judge_assertion_candidates",
            "Polylogue.record_comparative_judgment",
            "Polylogue.list_comparative_judgments",
            "Polylogue.join_typed_annotations",
        ),
        _surface("mark", "read"),
        _surface("write", "judge", "read"),
    ),
    ApiOperation(
        "api.archive.session-delete",
        "Archive mutations",
        "Delete a session and its archive records through the shared mutation executor.",
        "cross-tier",
        ("Polylogue.delete_session", "Polylogue.delete_session_safe"),
        _surface("delete"),
        _surface("write"),
    ),
    ApiOperation(
        "api.user-state.read",
        "Durable user state",
        "Read tags, marks, annotations, views, recall packs, workspaces, corrections, notes, and settings from user.db.",
        "user-read",
        (
            "Polylogue.list_tags",
            "Polylogue.get_metadata",
            "Polylogue.list_marks",
            "Polylogue.get_annotation",
            "Polylogue.list_annotations",
            "Polylogue.get_view",
            "Polylogue.list_views",
            "Polylogue.get_recall_pack",
            "Polylogue.list_recall_packs",
            "Polylogue.get_workspace",
            "Polylogue.list_workspaces",
            "Polylogue.list_corrections",
            "Polylogue.list_blackboard_notes",
            "Polylogue.get_setting",
            "Polylogue.list_settings",
        ),
        _surface("read", "mark"),
        _surface("read", "get"),
    ),
    ApiOperation(
        "api.user-state.write",
        "Durable user state",
        "Mutate tags, metadata, marks, annotations, views, recall packs, workspaces, corrections, notes, and settings in user.db.",
        "user-write",
        (
            "Polylogue.add_tag",
            "Polylogue.remove_tag",
            "Polylogue.update_metadata",
            "Polylogue.set_metadata",
            "Polylogue.delete_metadata",
            "Polylogue.bulk_tag_sessions",
            "Polylogue.add_mark",
            "Polylogue.remove_mark",
            "Polylogue.save_annotation",
            "Polylogue.delete_annotation",
            "Polylogue.save_view",
            "Polylogue.delete_view",
            "Polylogue.create_recall_pack",
            "Polylogue.delete_recall_pack",
            "Polylogue.save_workspace",
            "Polylogue.delete_workspace",
            "Polylogue.record_correction",
            "Polylogue.delete_correction",
            "Polylogue.clear_corrections",
            "Polylogue.post_blackboard_note",
            "Polylogue.set_setting",
        ),
        _surface("mark", "delete"),
        _surface("write"),
    ),
)

API_EXCLUSIONS: tuple[ApiExclusion, ...] = (
    ApiExclusion("ArchiveStats", "Result data model, not an executable archive operation."),
    ApiExclusion(
        "select_pending_embedding_session_window",
        "Public adapter helper for daemon/CLI window selection. It is intentionally not a facade operation.",
    ),
    ApiExclusion("Polylogue.__repr__", "Diagnostic representation protocol, not an archive operation."),
)


def declared_python_bindings() -> dict[str, ApiOperation | ApiExclusion]:
    """Return the explicit binding map and reject duplicate authority."""

    bindings: dict[str, ApiOperation | ApiExclusion] = {}
    for declaration in API_OPERATIONS:
        for binding in declaration.python_bindings:
            if binding in bindings:
                raise ValueError(f"duplicate Python parity binding: {binding}")
            bindings[binding] = declaration
    for exclusion in API_EXCLUSIONS:
        if exclusion.binding in bindings:
            raise ValueError(f"binding is both operation and exclusion: {exclusion.binding}")
        bindings[exclusion.binding] = exclusion
    return bindings


def _facade_callable_names() -> set[str]:
    from polylogue.api import Polylogue

    names = {"Polylogue", "Polylogue.__init__", "Polylogue.__aenter__", "Polylogue.__aexit__", "Polylogue.__repr__"}
    for name in dir(Polylogue):
        if name.startswith("_"):
            continue
        descriptor = inspect.getattr_static(Polylogue, name)
        if isinstance(descriptor, property):
            continue
        if callable(getattr(Polylogue, name)):
            names.add(f"Polylogue.{name}")
    return names


def live_public_callable_names() -> set[str]:
    """Return the narrow documented facade surface, including public exports."""

    from polylogue import api

    names = _facade_callable_names()
    for name in api.__all__:
        if name == "Polylogue":
            continue
        if callable(getattr(api, name)):
            names.add(name)
    return names


def validate_live_facade() -> None:
    """Fail closed when declarations no longer classify the public facade."""

    declared = set(declared_python_bindings())
    live = live_public_callable_names()
    unknown = sorted(live - declared)
    stale = sorted(declared - live)
    if unknown or stale:
        details = []
        if unknown:
            details.append(f"unclassified live callables: {', '.join(unknown)}")
        if stale:
            details.append(f"declared non-callables: {', '.join(stale)}")
        raise ValueError("API parity declaration drift: " + "; ".join(details))


def facade_callable_records() -> tuple[tuple[str, str, bool], ...]:
    """Return declared facade methods with current signatures and asyncness."""

    from polylogue.api import Polylogue

    records: list[tuple[str, str, bool]] = []
    for binding in sorted(_facade_callable_names() - {"Polylogue", "Polylogue.__repr__"}):
        name = binding.removeprefix("Polylogue.")
        member = Polylogue if name == "__init__" else getattr(Polylogue, name)
        records.append((binding, str(inspect.signature(member)), inspect.iscoroutinefunction(member)))
    return tuple(records)


def operations_for_section(section: str) -> Iterable[ApiOperation]:
    return (operation for operation in API_OPERATIONS if operation.section == section)


__all__ = [
    "API_EXCLUSIONS",
    "API_OPERATIONS",
    "API_PARITY_AUTHORITY",
    "ApiExclusion",
    "ApiOperation",
    "RouteClass",
    "SurfaceBinding",
    "declared_python_bindings",
    "facade_callable_records",
    "live_public_callable_names",
    "operations_for_section",
    "validate_live_facade",
]
