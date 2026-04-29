"""Typed runtime operation metadata shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Literal

from polylogue.lib.json import JSONDocument, JSONDocumentList, json_document

Effect = Literal["Pure", "DbRead", "DbWrite", "FileWrite", "Network", "LiveArchive", "Destructive"]
"""Declared runtime effect of an operation.

Each effect implies specific guarantees that the verification catalog
must check:

  Pure        → deterministic, no_side_effect
  DbRead      → snapshot_consistent
  DbWrite     → preview, idempotent, rollback_safe, atomic
  FileWrite   → path_sanitized, atomic_rename, parent_exists
  Network     → timeout_bounded, retry_bounded
  LiveArchive → sampling_bounded, privacy_safe_evidence
  Destructive → explicit_dry_run_evidence, confirmed_before_execute
"""


class OperationKind(str, Enum):
    """High-level operation class over runtime artifacts."""

    PLANNING = "planning"
    MATERIALIZATION = "materialization"
    INDEXING = "indexing"
    PROJECTION = "projection"
    CLI = "cli"
    BENCHMARK = "benchmark"
    QUERY = "query"
    READINESSCHECK = "readinesscheck"


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
    effects: tuple[Effect, ...] = ()

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "name": self.name,
                "kind": self.kind.value,
                "description": self.description,
                "consumes": list(self.consumes),
                "produces": list(self.produces),
                "path_targets": list(self.path_targets),
                "code_refs": list(self.code_refs),
                "surfaces": list(self.surfaces),
                "mutates_state": self.mutates_state,
                "previewable": self.previewable,
                "idempotent": self.idempotent,
                "effects": list(self.effects),
            }
        )


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

    def to_dict(self) -> JSONDocumentList:
        return [spec.to_dict() for spec in self.specs]


RUNTIME_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    OperationSpec(
        name="acquire-raw-conversations",
        kind=OperationKind.MATERIALIZATION,
        description="Traverse configured sources, detect provider-shaped payloads, and persist raw conversation records plus artifact observations.",
        consumes=("configured_sources", "source_payload_stream"),
        produces=("raw_validation_state", "artifact_observation_rows"),
        path_targets=("source-acquisition-loop",),
        code_refs=(
            "polylogue.pipeline.run_stages.execute_acquire_stage",
            "polylogue.pipeline.services.acquisition.AcquisitionService.acquire_sources",
            "polylogue.sources.source_acquisition.iter_source_raw_data",
            "polylogue.pipeline.services.acquisition_persistence.persist_raw_record",
        ),
        surfaces=("run.acquire", "run.parse", "sources"),
        mutates_state=True,
        effects=("Network", "DbWrite", "LiveArchive"),
    ),
    OperationSpec(
        name="plan-validation-backlog",
        kind=OperationKind.PLANNING,
        description="Select raw records that still require validation before normal parse planning.",
        consumes=("raw_validation_state",),
        produces=("validation_backlog",),
        path_targets=("raw-reparse-loop", "raw-archive-ingest-loop"),
        code_refs=(
            "polylogue.storage.raw.artifacts.validation_backlog_query_spec",
            "polylogue.pipeline.services.planning_backlog.collect_validation_backlog",
        ),
        surfaces=("run.parse", "reparse"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="plan-parse-backlog",
        kind=OperationKind.PLANNING,
        description="Select raw records that are eligible for parse planning under ordinary or force-reparse rules.",
        consumes=("raw_validation_state",),
        produces=("parse_backlog", "parse_quarantine"),
        path_targets=("raw-reparse-loop", "raw-archive-ingest-loop"),
        code_refs=(
            "polylogue.storage.raw.artifacts.parse_backlog_query_spec",
            "polylogue.pipeline.services.planning_backlog.collect_parse_backlog",
        ),
        surfaces=("run.parse", "reparse"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="ingest-archive-runtime",
        kind=OperationKind.MATERIALIZATION,
        description=(
            "Decode, validate, parse, transform, and persist raw conversations into the durable archive runtime tables."
        ),
        consumes=("raw_validation_state", "validation_backlog", "parse_backlog"),
        produces=("raw_validation_state", "archive_conversation_rows"),
        path_targets=("raw-archive-ingest-loop",),
        code_refs=(
            "polylogue.pipeline.services.parsing_workflow.parse_from_raw",
            "polylogue.pipeline.prepare.prepare_records",
            "polylogue.pipeline.prepare.persist_prepared_bundle",
            "polylogue.storage.repository.archive.writes.conversations.save_via_backend",
            "polylogue.storage.repository.raw.repository_raw.RepositoryRawMixin.mark_raw_validated",
            "polylogue.storage.repository.raw.repository_raw.RepositoryRawMixin.mark_raw_parsed",
        ),
        surfaces=("run.parse", "reprocess", "ingest"),
        mutates_state=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="index-message-fts",
        kind=OperationKind.INDEXING,
        description="Build or repair lexical message FTS rows from persisted archive messages.",
        consumes=("message_source_rows",),
        produces=("message_fts",),
        path_targets=("message-fts-readiness-loop", "conversation-query-loop"),
        code_refs=(
            "polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync",
            "polylogue.storage.fts.fts_lifecycle.repair_fts_index_sync",
            "polylogue.storage.fts.fts_lifecycle.message_fts_readiness_sync",
        ),
        surfaces=("run.index", "doctor", "repair", "query"),
        mutates_state=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="materialize-transcript-embeddings",
        kind=OperationKind.MATERIALIZATION,
        description="Build or refresh transcript embedding metadata, conversation status rows, and semantic vector entries from archive conversations.",
        consumes=("archive_conversation_rows",),
        produces=("embedding_metadata_rows", "embedding_status_rows", "message_embedding_vectors"),
        path_targets=("embedding-materialization-loop",),
        code_refs=(
            "polylogue.cli.shared.embed_runtime.embed_batch",
            "polylogue.cli.shared.embed_runtime.embed_single",
            "polylogue.storage.search_providers.sqlite_vec_queries.SqliteVecQueryMixin.upsert",
        ),
        surfaces=("run.embed", "embed", "retrieval"),
        mutates_state=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="materialize-action-events",
        kind=OperationKind.MATERIALIZATION,
        description="Build the action-event read model and trigger-maintained FTS projection from tool-use source blocks.",
        consumes=("tool_use_source_blocks",),
        produces=("action_event_rows", "action_event_fts"),
        path_targets=("action-event-repair-loop",),
        code_refs=(
            "polylogue.storage.action_events.rebuild_runtime.rebuild_action_event_read_model_sync",
            "polylogue.storage.backends.schema_ddl_actions.ACTION_FTS_DDL",
        ),
        surfaces=("index", "doctor", "repair", "retrieval_evidence"),
        mutates_state=True,
        effects=("DbRead", "DbWrite"),
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
            "polylogue.lib.query.plan_execution",
        ),
        surfaces=("query", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="render-conversations",
        kind=OperationKind.MATERIALIZATION,
        description="Render repository conversation projections into durable filesystem artifacts for human-facing archive browsing.",
        consumes=("conversation_render_projection",),
        produces=("rendered_conversation_artifacts",),
        path_targets=("conversation-render-loop",),
        code_refs=(
            "polylogue.pipeline.services.rendering.RenderService.render_conversations",
            "polylogue.rendering.renderers.markdown.MarkdownRenderer.render",
            "polylogue.rendering.renderers.html.HTMLRenderer.render",
        ),
        surfaces=("run.render", "render"),
        mutates_state=True,
        effects=("DbRead", "FileWrite"),
    ),
    OperationSpec(
        name="publish-site",
        kind=OperationKind.MATERIALIZATION,
        description="Build static-site conversation pages, write the site publication manifest, and persist the latest publication record.",
        consumes=("conversation_render_projection",),
        produces=("site_conversation_pages", "site_publication_manifest", "publication_records"),
        path_targets=("site-publication-loop",),
        code_refs=(
            "polylogue.site.builder.SiteBuilder.build",
            "polylogue.site.conversation_pages.generate_conversation_page",
            "polylogue.site.publication_flow.build_site_publication_manifest",
            "polylogue.site.publication_flow.record_site_publication_manifest",
        ),
        surfaces=("run.site", "site", "maintenance"),
        mutates_state=True,
        effects=("DbRead", "DbWrite", "FileWrite"),
    ),
    OperationSpec(
        name="project-action-event-readiness",
        kind=OperationKind.PROJECTION,
        description="Project readiness, debt, and repair semantics from action-event rows and FTS state.",
        consumes=("action_event_rows", "action_event_fts"),
        produces=("action_event_readiness",),
        path_targets=("action-event-repair-loop",),
        code_refs=(
            "polylogue.storage.derived.derived_status",
            "polylogue.storage.repair",
            "polylogue.storage.embeddings.support",
        ),
        surfaces=("doctor", "archive_debt", "repair"),
        previewable=True,
        effects=("DbRead",),
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
            "polylogue.storage.products.session.rebuild.rebuild_session_products_sync",
            "polylogue.storage.products.session.refresh.refresh_session_products_for_conversation_async",
        ),
        surfaces=("products", "doctor", "repair", "run.materialize"),
        mutates_state=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="project-retrieval-band-readiness",
        kind=OperationKind.PROJECTION,
        description="Project transcript/evidence/inference/enrichment retrieval readiness from embeddings and durable read-model readiness.",
        consumes=(
            "embedding_metadata_rows",
            "embedding_status_rows",
            "message_embedding_vectors",
            "action_event_readiness",
            "session_product_readiness",
        ),
        produces=("retrieval_band_readiness",),
        path_targets=("retrieval-band-readiness-loop",),
        code_refs=(
            "polylogue.storage.embeddings.embedding_stats.read_embedding_stats_sync",
            "polylogue.storage.embeddings.support.build_retrieval_bands_from_status",
            "polylogue.storage.derived.derived_status.build_retrieval_statuses",
        ),
        surfaces=("embed", "doctor", "retrieval"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-embedding-status",
        kind=OperationKind.QUERY,
        description="Resolve operator-facing embedding coverage, freshness, and retrieval-band readiness status views.",
        consumes=(
            "embedding_metadata_rows",
            "embedding_status_rows",
            "message_embedding_vectors",
            "retrieval_band_readiness",
        ),
        produces=("embedding_status_results",),
        path_targets=("embedding-status-query-loop",),
        code_refs=(
            "polylogue.cli.shared.embed_stats.embedding_status_payload",
            "polylogue.cli.shared.embed_stats.render_embedding_stats",
        ),
        surfaces=("run.embed", "embed", "doctor", "retrieval"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="project-session-product-readiness",
        kind=OperationKind.PROJECTION,
        description="Project readiness, debt, and stale-surface semantics from durable session-product rows and FTS state.",
        consumes=("session_product_rows", "session_product_fts"),
        produces=("session_product_readiness",),
        path_targets=("session-product-repair-loop",),
        code_refs=(
            "polylogue.storage.products.session.status",
            "polylogue.storage.repair",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "doctor", "archive_debt", "repair"),
        previewable=True,
        effects=("DbRead",),
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
        effects=("DbRead",),
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
        effects=("DbRead",),
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
        effects=("DbRead",),
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
        effects=("DbRead",),
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
        effects=("DbRead",),
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
        effects=("DbRead",),
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
        effects=("DbRead",),
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
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-session-product-status",
        kind=OperationKind.QUERY,
        description="Resolve projected session-product status views from session-product readiness state.",
        consumes=("session_product_readiness",),
        produces=("session_product_status_results",),
        path_targets=("session-product-status-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveStatsMixin.get_session_product_status",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
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
            "polylogue.cli.shared.helper_summary",
        ),
        surfaces=("products", "facade", "mcp", "helpers"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-archive-debt",
        kind=OperationKind.QUERY,
        description="Resolve archive debt views from projected derived-model readiness and maintenance state.",
        consumes=("action_event_readiness", "session_product_readiness", "archive_readiness"),
        produces=("archive_debt_results",),
        path_targets=("archive-debt-query-loop",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductDebtMixin.list_archive_debt_products",
            "polylogue.cli.commands.products",
        ),
        surfaces=("products", "facade", "mcp", "maintenance"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="compile-inferred-corpus-specs",
        kind=OperationKind.PROJECTION,
        description="Compile inferred synthetic corpus specs from schema packages and cluster manifests.",
        consumes=("schema_packages", "schema_cluster_manifests"),
        produces=("inferred_corpus_specs",),
        path_targets=("inferred-corpus-compilation-loop",),
        code_refs=(
            "polylogue.scenarios.corpus.build_inferred_corpus_specs",
            "polylogue.schemas.operator.inference.list_inferred_corpus_specs",
        ),
        surfaces=("schema", "qa", "synthetic"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="compile-inferred-corpus-scenarios",
        kind=OperationKind.PROJECTION,
        description="Compile inferred corpus scenarios grouped from inferred corpus specs.",
        consumes=("inferred_corpus_specs",),
        produces=("inferred_corpus_scenarios",),
        path_targets=("inferred-corpus-compilation-loop",),
        code_refs=(
            "polylogue.scenarios.corpus.build_corpus_scenarios",
            "polylogue.schemas.operator.inference.list_inferred_corpus_scenarios",
        ),
        surfaces=("schema", "qa", "synthetic"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="query-schema-catalog",
        kind=OperationKind.QUERY,
        description="Resolve schema package catalogs, manifests, and inferred corpus projections for schema list surfaces.",
        consumes=("schema_packages", "schema_cluster_manifests", "inferred_corpus_specs", "inferred_corpus_scenarios"),
        produces=("schema_list_results",),
        path_targets=("schema-list-query-loop",),
        code_refs=(
            "polylogue.schemas.operator.inference.list_schemas",
            "polylogue.cli.commands.schema.schema_list",
        ),
        surfaces=("schema", "cli"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="query-schema-explanations",
        kind=OperationKind.QUERY,
        description="Resolve provider schema element explanations from versioned schema packages.",
        consumes=("schema_packages",),
        produces=("schema_explanation_results",),
        path_targets=("schema-explain-query-loop",),
        code_refs=(
            "polylogue.schemas.operator.resolution.explain_schema",
            "polylogue.cli.commands.schema.schema_explain",
        ),
        surfaces=("schema", "cli"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="project-archive-readiness",
        kind=OperationKind.PROJECTION,
        description="Project archive-wide readiness and debt semantics from message FTS and durable derived-model readiness.",
        consumes=("message_fts", "action_event_readiness", "session_product_readiness", "retrieval_band_readiness"),
        produces=("archive_readiness",),
        path_targets=("message-fts-readiness-loop", "retrieval-band-readiness-loop"),
        code_refs=(
            "polylogue.readiness.run_archive_readiness",
            "polylogue.storage.derived.derived_status.collect_derived_model_statuses_sync",
            "polylogue.storage.repair.collect_archive_debt_statuses_sync",
        ),
        surfaces=("doctor", "archive_debt", "maintenance"),
        previewable=True,
        effects=("DbRead",),
    ),
)

DECLARED_CONTROL_PLANE_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    OperationSpec(
        name="cli.help",
        kind=OperationKind.CLI,
        description="Render Click help for one command path without mutating archive state.",
        surfaces=("help", "showcase"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="cli.json-contract",
        kind=OperationKind.CLI,
        description="Exercise a machine-readable CLI JSON surface and verify its contract envelope.",
        surfaces=("doctor", "audit", "schema", "tags", "showcase"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="seed-archive-scenarios",
        kind=OperationKind.PROJECTION,
        description="Seed authored archive-scenario fixtures through typed storage-record helpers for verification lanes.",
        surfaces=("tests", "validation-lane"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="build-storage-record-fixtures",
        kind=OperationKind.PROJECTION,
        description="Build typed storage-record fixtures from JSON-validated helper inputs for verification lanes.",
        surfaces=("tests", "validation-lane"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="benchmark.query.search-filters",
        kind=OperationKind.BENCHMARK,
        description="Measure the canonical FTS and ConversationFilter query benchmark domain.",
        surfaces=("benchmark-campaign",),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="benchmark.storage.crud",
        kind=OperationKind.BENCHMARK,
        description="Measure repository and backend CRUD latency for the storage benchmark domain.",
        surfaces=("benchmark-campaign",),
        previewable=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="benchmark.pipeline.index-and-helpers",
        kind=OperationKind.BENCHMARK,
        description="Measure indexing and hot pipeline-helper throughput in the benchmark campaign domain.",
        surfaces=("benchmark-campaign",),
        previewable=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="benchmark.repair.action-events",
        kind=OperationKind.BENCHMARK,
        description="Measure action-event repair throughput in focused benchmark scenarios.",
        surfaces=("benchmark-campaign",),
        previewable=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="index.message-fts-rebuild",
        kind=OperationKind.INDEXING,
        description="Benchmark full message FTS rebuild over a synthetic archive.",
        surfaces=("synthetic-benchmark",),
        previewable=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="index.message-fts-incremental",
        kind=OperationKind.INDEXING,
        description="Benchmark incremental message FTS updates over a synthetic archive.",
        surfaces=("synthetic-benchmark",),
        previewable=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="query.filters.synthetic-scan",
        kind=OperationKind.QUERY,
        description="Benchmark common synthetic filter-query scans over generated archives.",
        surfaces=("synthetic-benchmark",),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="readiness.startup.synthetic",
        kind=OperationKind.READINESSCHECK,
        description="Benchmark startup readiness checks over a synthetic archive.",
        surfaces=("synthetic-benchmark",),
        previewable=True,
        effects=("DbRead",),
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
    "Effect",
    "build_declared_operation_catalog",
    "build_runtime_operation_catalog",
    "OperationCatalog",
    "OperationKind",
    "OperationSpec",
    "RUNTIME_OPERATION_SPECS",
]
