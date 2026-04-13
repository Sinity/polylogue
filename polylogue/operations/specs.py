"""Typed runtime operation metadata shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
from typing import Any


class OperationKind(str, Enum):
    """High-level operation class over runtime artifacts."""

    PLANNING = "planning"
    MATERIALIZATION = "materialization"
    INDEXING = "indexing"
    PROJECTION = "projection"
    CLI = "cli"
    BENCHMARK = "benchmark"
    QUERY = "query"
    HEALTHCHECK = "healthcheck"


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """One named runtime operation over declared artifact nodes."""

    name: str
    kind: OperationKind
    description: str
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    path_targets: tuple[str, ...] = ()
    code_refs: tuple[str, ...] = ()
    surfaces: tuple[str, ...] = ()
    mutates_state: bool = False
    previewable: bool = False
    idempotent: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


@dataclass(frozen=True, slots=True)
class OperationCatalog:
    """Canonical operation registry with stable lookup and resolution helpers."""

    specs: tuple[OperationSpec, ...]

    def by_name(self) -> dict[str, OperationSpec]:
        return {spec.name: spec for spec in self.specs}

    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def resolve(self, names: tuple[str, ...]) -> tuple[OperationSpec, ...]:
        by_name = self.by_name()
        return tuple(by_name[name] for name in names if name in by_name)

    def to_dict(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self.specs]


RUNTIME_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    OperationSpec(
        name="plan-validation-backlog",
        kind=OperationKind.PLANNING,
        description="Select raw records that still require validation before normal parse planning.",
        consumes=("raw_validation_state",),
        produces=("validation_backlog",),
        path_targets=("raw-reparse-loop",),
        code_refs=(
            "polylogue.storage.raw_ingest_artifacts.validation_backlog_query_spec",
            "polylogue.pipeline.services.planning_backlog.collect_validation_backlog",
        ),
        surfaces=("run.parse", "reparse"),
        previewable=True,
    ),
    OperationSpec(
        name="plan-parse-backlog",
        kind=OperationKind.PLANNING,
        description="Select raw records that are eligible for parse planning under ordinary or force-reparse rules.",
        consumes=("raw_validation_state",),
        produces=("parse_backlog", "parse_quarantine"),
        path_targets=("raw-reparse-loop",),
        code_refs=(
            "polylogue.storage.raw_ingest_artifacts.parse_backlog_query_spec",
            "polylogue.pipeline.services.planning_backlog.collect_parse_backlog",
        ),
        surfaces=("run.parse", "reparse"),
        previewable=True,
    ),
    OperationSpec(
        name="index-message-fts",
        kind=OperationKind.INDEXING,
        description="Build or repair lexical message FTS rows from persisted archive messages.",
        consumes=("message_source_rows",),
        produces=("message_fts",),
        path_targets=("message-fts-health-loop", "conversation-query-loop"),
        code_refs=(
            "polylogue.storage.fts_lifecycle.rebuild_fts_index_sync",
            "polylogue.storage.fts_lifecycle.repair_fts_index_sync",
            "polylogue.storage.fts_lifecycle.message_fts_readiness_sync",
        ),
        surfaces=("run.index", "doctor", "repair", "query"),
        mutates_state=True,
    ),
    OperationSpec(
        name="materialize-action-events",
        kind=OperationKind.MATERIALIZATION,
        description="Build the action-event read model and trigger-maintained FTS projection from tool-use source blocks.",
        consumes=("tool_use_source_blocks",),
        produces=("action_event_rows", "action_event_fts"),
        path_targets=("action-event-repair-loop",),
        code_refs=(
            "polylogue.storage.action_event_rebuild_runtime.rebuild_action_event_read_model_sync",
            "polylogue.storage.backends.schema_ddl_actions.ACTION_FTS_DDL",
        ),
        surfaces=("index", "doctor", "repair", "retrieval_evidence"),
        mutates_state=True,
    ),
    OperationSpec(
        name="query-conversations",
        kind=OperationKind.QUERY,
        description="Resolve conversation-level query and search results from archive retrieval plans.",
        consumes=("message_fts",),
        produces=("conversation_query_results",),
        path_targets=("conversation-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveSearchMixin.query_conversations",
            "polylogue.operations.archive.ArchiveSearchMixin.search",
            "polylogue.lib.query_plan_execution",
        ),
        surfaces=("query", "facade", "mcp"),
        previewable=True,
    ),
    OperationSpec(
        name="project-action-event-health",
        kind=OperationKind.PROJECTION,
        description="Project health, debt, and repair semantics from action-event rows and FTS state.",
        consumes=("action_event_rows", "action_event_fts"),
        produces=("action_event_health",),
        path_targets=("action-event-repair-loop",),
        code_refs=(
            "polylogue.storage.derived_status",
            "polylogue.storage.repair",
            "polylogue.storage.embedding_stats_support",
        ),
        surfaces=("doctor", "archive_debt", "repair"),
        previewable=True,
    ),
    OperationSpec(
        name="materialize-session-products",
        kind=OperationKind.MATERIALIZATION,
        description="Build durable session-product rows and their trigger-maintained FTS projections from archive conversations.",
        consumes=("session_product_source_conversations",),
        produces=(
            "session_profile_rows",
            "session_profile_merged_fts",
            "session_profile_evidence_fts",
            "session_profile_inference_fts",
            "session_profile_enrichment_fts",
            "session_work_event_rows",
            "session_work_event_fts",
            "session_phase_rows",
            "work_thread_rows",
            "work_thread_fts",
            "session_tag_rollup_rows",
            "day_session_summary_rows",
            "session_product_rows",
            "session_product_fts",
        ),
        path_targets=("session-product-repair-loop",),
        code_refs=(
            "polylogue.storage.session_product_rebuild.rebuild_session_products_sync",
            "polylogue.storage.session_product_refresh.refresh_session_products_for_conversation_async",
        ),
        surfaces=("products", "doctor", "repair", "run.materialize"),
        mutates_state=True,
    ),
    OperationSpec(
        name="project-session-product-health",
        kind=OperationKind.PROJECTION,
        description="Project readiness, debt, and stale-surface semantics from durable session-product rows and FTS state.",
        consumes=("session_product_rows", "session_product_fts"),
        produces=("session_product_health",),
        path_targets=("session-product-repair-loop",),
        code_refs=(
            "polylogue.storage.session_product_status",
            "polylogue.storage.repair",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "doctor", "archive_debt", "repair"),
        previewable=True,
    ),
    OperationSpec(
        name="query-session-profiles",
        kind=OperationKind.QUERY,
        description="Resolve durable session-profile products from profile rows and merged profile FTS.",
        consumes=("session_profile_rows", "session_profile_merged_fts"),
        produces=("session_profile_results",),
        path_targets=("session-profile-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_session_profile_products",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "facade", "mcp"),
        previewable=True,
    ),
    OperationSpec(
        name="query-session-enrichments",
        kind=OperationKind.QUERY,
        description="Resolve durable session enrichments from profile rows and enrichment FTS.",
        consumes=("session_profile_rows", "session_profile_enrichment_fts"),
        produces=("session_enrichment_results",),
        path_targets=("session-enrichment-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_session_enrichment_products",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "facade", "mcp"),
        previewable=True,
    ),
    OperationSpec(
        name="query-session-work-events",
        kind=OperationKind.QUERY,
        description="Resolve durable session work-event products from work-event rows and work-event FTS.",
        consumes=("session_work_event_rows", "session_work_event_fts"),
        produces=("session_work_event_results",),
        path_targets=("session-work-event-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_session_work_event_products",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "facade", "mcp"),
        previewable=True,
    ),
    OperationSpec(
        name="query-session-phases",
        kind=OperationKind.QUERY,
        description="Resolve durable session-phase products from phase rows.",
        consumes=("session_phase_rows",),
        produces=("session_phase_results",),
        path_targets=("session-phase-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_session_phase_products",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "facade", "mcp"),
        previewable=True,
    ),
    OperationSpec(
        name="query-work-threads",
        kind=OperationKind.QUERY,
        description="Resolve durable work-thread products from work-thread rows and thread FTS.",
        consumes=("work_thread_rows", "work_thread_fts"),
        produces=("work_thread_results",),
        path_targets=("work-thread-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_work_thread_products",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "facade", "mcp"),
        previewable=True,
    ),
    OperationSpec(
        name="query-session-tag-rollups",
        kind=OperationKind.QUERY,
        description="Resolve durable session tag-rollup products from aggregate tag rows.",
        consumes=("session_tag_rollup_rows",),
        produces=("session_tag_rollup_results",),
        path_targets=("session-tag-rollup-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_session_tag_rollup_products",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "facade", "mcp"),
        previewable=True,
    ),
    OperationSpec(
        name="query-day-session-summaries",
        kind=OperationKind.QUERY,
        description="Resolve durable day session summaries from day-summary rows.",
        consumes=("day_session_summary_rows",),
        produces=("day_session_summary_results",),
        path_targets=("day-summary-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_day_session_summary_products",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "facade", "mcp"),
        previewable=True,
    ),
    OperationSpec(
        name="query-week-session-summaries",
        kind=OperationKind.QUERY,
        description="Resolve week session summaries from durable day-summary rows.",
        consumes=("day_session_summary_rows",),
        produces=("week_session_summary_results",),
        path_targets=("week-summary-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_week_session_summary_products",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "facade", "mcp"),
        previewable=True,
    ),
    OperationSpec(
        name="query-session-product-status",
        kind=OperationKind.QUERY,
        description="Resolve projected session-product status views from session-product health state.",
        consumes=("session_product_health",),
        produces=("session_product_status_results",),
        path_targets=("session-product-status-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveStatsMixin.get_session_product_status",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "facade", "mcp"),
        previewable=True,
    ),
    OperationSpec(
        name="query-provider-analytics",
        kind=OperationKind.QUERY,
        description="Resolve provider analytics from durable session-product aggregates.",
        consumes=("session_product_rows",),
        produces=("provider_analytics_results",),
        path_targets=("provider-analytics-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_provider_analytics_products",
            "polylogue.cli.commands.products",
            "polylogue.cli.helper_summary",
        ),
        surfaces=("products", "facade", "mcp", "helpers"),
        previewable=True,
    ),
    OperationSpec(
        name="query-archive-debt",
        kind=OperationKind.QUERY,
        description="Resolve archive debt views from projected derived-model health and maintenance state.",
        consumes=("action_event_health", "session_product_health", "archive_health"),
        produces=("archive_debt_results",),
        path_targets=("archive-debt-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductDebtMixin.list_archive_debt_products",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "facade", "mcp", "maintenance"),
        previewable=True,
    ),
    OperationSpec(
        name="project-archive-health",
        kind=OperationKind.PROJECTION,
        description="Project archive-wide health and debt semantics from message FTS and durable derived-model readiness.",
        consumes=("message_fts", "action_event_health", "session_product_health"),
        produces=("archive_health",),
        path_targets=("message-fts-health-loop",),
        code_refs=(
            "polylogue.health.run_archive_health",
            "polylogue.storage.derived_status.collect_derived_model_statuses_sync",
            "polylogue.storage.repair.collect_archive_debt_statuses_sync",
        ),
        surfaces=("doctor", "archive_debt", "maintenance"),
        previewable=True,
    ),
)

DECLARED_CONTROL_PLANE_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    OperationSpec(
        name="cli.help",
        kind=OperationKind.CLI,
        description="Render Click help for one command path without mutating archive state.",
        surfaces=("help", "showcase"),
        previewable=True,
    ),
    OperationSpec(
        name="cli.json-contract",
        kind=OperationKind.CLI,
        description="Exercise a machine-readable CLI JSON surface and verify its contract envelope.",
        surfaces=("doctor", "audit", "schema", "tags", "showcase"),
        previewable=True,
    ),
    OperationSpec(
        name="benchmark.query.search-filters",
        kind=OperationKind.BENCHMARK,
        description="Measure the canonical FTS and ConversationFilter query benchmark domain.",
        surfaces=("benchmark-campaign",),
        previewable=True,
    ),
    OperationSpec(
        name="benchmark.storage.crud",
        kind=OperationKind.BENCHMARK,
        description="Measure repository and backend CRUD latency for the storage benchmark domain.",
        surfaces=("benchmark-campaign",),
        previewable=True,
    ),
    OperationSpec(
        name="benchmark.pipeline.index-and-helpers",
        kind=OperationKind.BENCHMARK,
        description="Measure indexing and hot pipeline-helper throughput in the benchmark campaign domain.",
        surfaces=("benchmark-campaign",),
        previewable=True,
    ),
    OperationSpec(
        name="benchmark.repair.action-events",
        kind=OperationKind.BENCHMARK,
        description="Measure action-event repair throughput in focused benchmark scenarios.",
        surfaces=("benchmark-campaign",),
        previewable=True,
    ),
    OperationSpec(
        name="index.message-fts-rebuild",
        kind=OperationKind.INDEXING,
        description="Benchmark full message FTS rebuild over a synthetic archive.",
        surfaces=("synthetic-benchmark",),
        previewable=True,
    ),
    OperationSpec(
        name="index.message-fts-incremental",
        kind=OperationKind.INDEXING,
        description="Benchmark incremental message FTS updates over a synthetic archive.",
        surfaces=("synthetic-benchmark",),
        previewable=True,
    ),
    OperationSpec(
        name="query.filters.synthetic-scan",
        kind=OperationKind.QUERY,
        description="Benchmark common synthetic filter-query scans over generated archives.",
        surfaces=("synthetic-benchmark",),
        previewable=True,
    ),
    OperationSpec(
        name="health.startup.synthetic",
        kind=OperationKind.HEALTHCHECK,
        description="Benchmark startup health checks over a synthetic archive.",
        surfaces=("synthetic-benchmark",),
        previewable=True,
    ),
)

DECLARED_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    *RUNTIME_OPERATION_SPECS,
    *DECLARED_CONTROL_PLANE_OPERATION_SPECS,
)


@lru_cache(maxsize=1)
def build_runtime_operation_catalog() -> OperationCatalog:
    """Return the authored runtime operation catalog."""

    return OperationCatalog(specs=RUNTIME_OPERATION_SPECS)


@lru_cache(maxsize=1)
def build_declared_operation_catalog() -> OperationCatalog:
    """Return every authored operation target referenced across verification surfaces."""

    return OperationCatalog(specs=DECLARED_OPERATION_SPECS)


__all__ = [
    "DECLARED_CONTROL_PLANE_OPERATION_SPECS",
    "DECLARED_OPERATION_SPECS",
    "build_declared_operation_catalog",
    "build_runtime_operation_catalog",
    "OperationCatalog",
    "OperationKind",
    "OperationSpec",
    "RUNTIME_OPERATION_SPECS",
]
