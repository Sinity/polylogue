"""Typed runtime artifact specifications shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ArtifactLayer(str, Enum):
    SOURCE = "source"
    DURABLE = "durable"
    DERIVED = "derived"
    INDEX = "index"
    PROJECTION = "projection"


@dataclass(frozen=True, slots=True)
class ArtifactNode:
    """One named artifact or projection in the Polylogue runtime graph."""

    name: str
    layer: ArtifactLayer
    description: str
    depends_on: tuple[str, ...] = ()
    code_refs: tuple[str, ...] = ()
    repair_targets: tuple[str, ...] = ()
    health_surfaces: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["layer"] = self.layer.value
        return data


@dataclass(frozen=True, slots=True)
class ArtifactPath:
    """One curated path through the artifact graph."""

    name: str
    description: str
    nodes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RUNTIME_ARTIFACT_NODES: tuple[ArtifactNode, ...] = (
    ArtifactNode(
        name="raw_validation_state",
        layer=ArtifactLayer.DURABLE,
        description="Persisted raw-conversation validation and parse state in raw_conversations.",
        code_refs=(
            "polylogue.storage.raw_ingest_artifacts.RawIngestArtifactState",
            "polylogue.storage.backends.queries.raw_state",
        ),
    ),
    ArtifactNode(
        name="validation_backlog",
        layer=ArtifactLayer.PROJECTION,
        description="Raw records that still require validation before ordinary parse planning can trust them.",
        depends_on=("raw_validation_state",),
        code_refs=(
            "polylogue.storage.raw_ingest_artifacts.validation_backlog_query_spec",
            "polylogue.pipeline.services.planning_backlog.collect_validation_backlog",
        ),
    ),
    ArtifactNode(
        name="parse_backlog",
        layer=ArtifactLayer.PROJECTION,
        description="Raw records eligible for parse planning, including force-reparse simulation.",
        depends_on=("raw_validation_state",),
        code_refs=(
            "polylogue.storage.raw_ingest_artifacts.parse_backlog_query_spec",
            "polylogue.pipeline.services.planning_backlog.collect_parse_backlog",
            "polylogue.pipeline.services.planning_runtime.build_ingest_plan",
        ),
    ),
    ArtifactNode(
        name="parse_quarantine",
        layer=ArtifactLayer.PROJECTION,
        description="Validation-failed unparsed raws that stay out of ordinary parse backlog but return under force-reparse.",
        depends_on=("raw_validation_state",),
        code_refs=("polylogue.storage.raw_ingest_artifacts.RawIngestArtifactState",),
    ),
    ArtifactNode(
        name="archive_conversation_rows",
        layer=ArtifactLayer.DURABLE,
        description="Durable conversation, message, attachment, and content-block rows written from parsed raw conversations.",
        depends_on=("raw_validation_state",),
        code_refs=(
            "polylogue.pipeline.services.parsing_workflow.parse_from_raw",
            "polylogue.pipeline.prepare.prepare_records",
            "polylogue.pipeline.prepare.persist_prepared_bundle",
            "polylogue.storage.repository_write_conversations.save_via_backend",
            "polylogue.storage.repository_raw.RepositoryRawMixin.mark_raw_parsed",
        ),
    ),
    ArtifactNode(
        name="message_source_rows",
        layer=ArtifactLayer.SOURCE,
        description="Persisted message rows that feed lexical FTS indexing and archive search.",
        depends_on=("archive_conversation_rows",),
        code_refs=(
            "polylogue.storage.fts_lifecycle.FTS_INDEXABLE_MESSAGE_COUNT_SQL",
            "polylogue.storage.fts_lifecycle.repair_fts_index_sync",
        ),
    ),
    ArtifactNode(
        name="message_fts",
        layer=ArtifactLayer.INDEX,
        description="Lexical FTS projection over persisted message rows.",
        depends_on=("message_source_rows",),
        code_refs=(
            "polylogue.storage.fts_lifecycle.message_fts_readiness_sync",
            "polylogue.storage.fts_lifecycle.rebuild_fts_index_sync",
            "polylogue.storage.fts_lifecycle.repair_fts_index_sync",
        ),
        repair_targets=("fts",),
        health_surfaces=("doctor", "archive_debt", "query"),
    ),
    ArtifactNode(
        name="tool_use_source_blocks",
        layer=ArtifactLayer.SOURCE,
        description="Tool-use content blocks anchored to valid conversations.",
        depends_on=("archive_conversation_rows",),
        code_refs=("polylogue.storage.action_event_status",),
    ),
    ArtifactNode(
        name="action_event_rows",
        layer=ArtifactLayer.DERIVED,
        description="Materialized action-event read model derived from tool-use source blocks.",
        depends_on=("tool_use_source_blocks",),
        code_refs=(
            "polylogue.storage.action_event_artifacts.ActionEventArtifactState",
            "polylogue.storage.action_event_status",
        ),
        repair_targets=("action_event_read_model",),
        health_surfaces=("doctor", "archive_debt"),
    ),
    ArtifactNode(
        name="action_event_fts",
        layer=ArtifactLayer.INDEX,
        description="FTS projection over the action-event read model.",
        depends_on=("action_event_rows",),
        code_refs=(
            "polylogue.storage.action_event_artifacts.ActionEventArtifactState",
            "polylogue.storage.derived_status_products.build_action_statuses",
        ),
        repair_targets=("action_event_read_model",),
        health_surfaces=("doctor", "archive_debt", "retrieval_evidence"),
    ),
    ArtifactNode(
        name="action_event_health",
        layer=ArtifactLayer.PROJECTION,
        description="Projected health, debt, and repair semantics over action-event rows and FTS.",
        depends_on=("action_event_rows", "action_event_fts"),
        code_refs=(
            "polylogue.storage.derived_status",
            "polylogue.storage.repair",
            "polylogue.storage.embedding_stats_support",
        ),
        repair_targets=("action_event_read_model",),
        health_surfaces=("doctor", "archive_debt", "retrieval_evidence"),
    ),
    ArtifactNode(
        name="session_product_source_conversations",
        layer=ArtifactLayer.SOURCE,
        description="Hydratable conversation/message/attachment/block rows that feed durable session-product rebuilds.",
        depends_on=("archive_conversation_rows",),
        code_refs=(
            "polylogue.storage.session_product_rebuild.rebuild_session_products_sync",
            "polylogue.storage.session_product_refresh.refresh_session_products_for_conversation_async",
        ),
    ),
    ArtifactNode(
        name="session_profile_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable session profile rows materialized from archive conversations.",
        depends_on=("session_product_source_conversations",),
        code_refs=(
            "polylogue.storage.session_product_rebuild.build_session_product_records",
            "polylogue.storage.session_product_profiles.build_session_profile_record",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_profile_merged_fts",
        layer=ArtifactLayer.INDEX,
        description="Merged session-profile FTS projection over durable session profile rows.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.session_product_status",
            "polylogue.storage.backends.schema_ddl_product_profiles",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_profile_evidence_fts",
        layer=ArtifactLayer.INDEX,
        description="Evidence-tier session-profile FTS projection over durable session profile rows.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.session_product_status",
            "polylogue.storage.backends.schema_ddl_product_profiles",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_profile_inference_fts",
        layer=ArtifactLayer.INDEX,
        description="Inference-tier session-profile FTS projection over durable session profile rows.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.session_product_status",
            "polylogue.storage.backends.schema_ddl_product_profiles",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_profile_enrichment_fts",
        layer=ArtifactLayer.INDEX,
        description="Enrichment-tier session-profile FTS projection over durable session profile rows.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.session_product_status",
            "polylogue.storage.backends.schema_ddl_product_profiles",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_work_event_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable session work-event rows materialized from session profile analysis.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.session_product_rebuild.build_session_product_records",
            "polylogue.storage.session_product_timeline_rows.build_session_work_event_records",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_work_event_fts",
        layer=ArtifactLayer.INDEX,
        description="Session work-event FTS projection over durable work-event rows.",
        depends_on=("session_work_event_rows",),
        code_refs=(
            "polylogue.storage.session_product_status",
            "polylogue.storage.backends.schema_ddl_product_timelines",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_phase_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable session phase rows materialized from session profile analysis.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.session_product_rebuild.build_session_product_records",
            "polylogue.storage.session_product_timeline_rows.build_session_phase_records",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="work_thread_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable work-thread rows materialized from session profile families.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.session_product_threads.build_thread_records_for_roots_sync",
            "polylogue.storage.session_product_threads.build_thread_records_for_roots_async",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="work_thread_fts",
        layer=ArtifactLayer.INDEX,
        description="Work-thread FTS projection over durable work-thread rows.",
        depends_on=("work_thread_rows",),
        code_refs=(
            "polylogue.storage.session_product_status",
            "polylogue.storage.backends.schema_ddl_product_aggregates",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_tag_rollup_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable provider/day/tag rollup rows aggregated from session profiles.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.session_product_aggregates.refresh_sync_provider_day_aggregates",
            "polylogue.storage.session_product_status",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="day_session_summary_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable provider/day summary rows aggregated from session profiles.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.session_product_aggregates.refresh_sync_provider_day_aggregates",
            "polylogue.storage.session_product_status",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_product_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable session-profile, timeline, thread, and aggregate rows derived from archive conversations.",
        depends_on=(
            "session_profile_rows",
            "session_work_event_rows",
            "session_phase_rows",
            "work_thread_rows",
            "session_tag_rollup_rows",
            "day_session_summary_rows",
        ),
        code_refs=(
            "polylogue.storage.session_product_rebuild.rebuild_session_products_sync",
            "polylogue.storage.session_product_status",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_product_fts",
        layer=ArtifactLayer.INDEX,
        description="The session-product FTS family over profiles, work events, and threads.",
        depends_on=(
            "session_profile_merged_fts",
            "session_profile_evidence_fts",
            "session_profile_inference_fts",
            "session_profile_enrichment_fts",
            "session_work_event_fts",
            "work_thread_fts",
        ),
        code_refs=(
            "polylogue.storage.session_product_status",
            "polylogue.storage.derived_status_products",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_product_health",
        layer=ArtifactLayer.PROJECTION,
        description="Projected readiness, debt, and stale-surface semantics for durable session products.",
        depends_on=("session_product_rows", "session_product_fts"),
        code_refs=(
            "polylogue.storage.session_product_status",
            "polylogue.storage.repair",
            "polylogue.cli.commands.products",
        ),
        repair_targets=("session_products",),
        health_surfaces=("doctor", "archive_debt", "products"),
    ),
    ArtifactNode(
        name="session_profile_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable session profiles.",
        depends_on=("session_profile_rows", "session_profile_merged_fts"),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_session_profile_products",
            "polylogue.cli.commands.products",
        ),
        health_surfaces=("products", "facade", "mcp"),
    ),
    ArtifactNode(
        name="session_enrichment_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable session enrichments.",
        depends_on=("session_profile_rows", "session_profile_enrichment_fts"),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_session_enrichment_products",
            "polylogue.cli.commands.products",
        ),
        health_surfaces=("products", "facade", "mcp"),
    ),
    ArtifactNode(
        name="session_work_event_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable session work events.",
        depends_on=("session_work_event_rows", "session_work_event_fts"),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_session_work_event_products",
            "polylogue.cli.commands.products",
        ),
        health_surfaces=("products", "facade", "mcp"),
    ),
    ArtifactNode(
        name="session_phase_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable session phases.",
        depends_on=("session_phase_rows",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_session_phase_products",
            "polylogue.cli.commands.products",
        ),
        health_surfaces=("products", "facade", "mcp"),
    ),
    ArtifactNode(
        name="work_thread_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable work threads.",
        depends_on=("work_thread_rows", "work_thread_fts"),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_work_thread_products",
            "polylogue.cli.commands.products",
        ),
        health_surfaces=("products", "facade", "mcp"),
    ),
    ArtifactNode(
        name="session_tag_rollup_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable session tag rollups.",
        depends_on=("session_tag_rollup_rows",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_session_tag_rollup_products",
            "polylogue.cli.commands.products",
        ),
        health_surfaces=("products", "facade", "mcp"),
    ),
    ArtifactNode(
        name="day_session_summary_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable day session summaries.",
        depends_on=("day_session_summary_rows",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_day_session_summary_products",
            "polylogue.cli.commands.products",
        ),
        health_surfaces=("products", "facade", "mcp"),
    ),
    ArtifactNode(
        name="week_session_summary_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for week session summaries derived from durable day summaries.",
        depends_on=("day_session_summary_rows",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_week_session_summary_products",
            "polylogue.cli.commands.products",
        ),
        health_surfaces=("products", "facade", "mcp"),
    ),
    ArtifactNode(
        name="session_product_status_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable session-product status views.",
        depends_on=("session_product_health",),
        code_refs=(
            "polylogue.operations.archive.ArchiveStatsMixin.get_session_product_status",
            "polylogue.cli.commands.products",
        ),
        health_surfaces=("products", "facade", "mcp"),
    ),
    ArtifactNode(
        name="provider_analytics_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for provider analytics derived from durable session products.",
        depends_on=("session_product_rows",),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductMixin.list_provider_analytics_products",
            "polylogue.cli.commands.products",
            "polylogue.cli.helper_summary",
        ),
        health_surfaces=("products", "facade", "mcp", "helpers"),
    ),
    ArtifactNode(
        name="archive_debt_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for archive debt views derived from projected health and maintenance state.",
        depends_on=("action_event_health", "session_product_health", "archive_health"),
        code_refs=(
            "polylogue.operations.archive.ArchiveProductDebtMixin.list_archive_debt_products",
            "polylogue.cli.commands.products",
        ),
        health_surfaces=("products", "facade", "mcp", "maintenance"),
    ),
    ArtifactNode(
        name="conversation_query_results",
        layer=ArtifactLayer.PROJECTION,
        description="Conversation-level query and search results resolved from lexical retrieval over the archive.",
        depends_on=("message_fts",),
        code_refs=(
            "polylogue.operations.archive.ArchiveSearchMixin.query_conversations",
            "polylogue.operations.archive.ArchiveSearchMixin.search",
            "polylogue.lib.query_plan_execution",
        ),
        health_surfaces=("query", "mcp", "facade"),
    ),
    ArtifactNode(
        name="conversation_render_projection",
        layer=ArtifactLayer.PROJECTION,
        description="Repository-owned render projection preserving canonical conversation, message, and attachment layout.",
        depends_on=("archive_conversation_rows",),
        code_refs=(
            "polylogue.storage.repository_archive_conversations.RepositoryArchiveConversationMixin.get_render_projection",
            "polylogue.rendering.core_formatter.ConversationFormatter.load_projection",
        ),
        health_surfaces=("render", "site"),
    ),
    ArtifactNode(
        name="rendered_conversation_artifacts",
        layer=ArtifactLayer.PROJECTION,
        description="Filesystem render outputs produced by run render for conversation pages.",
        depends_on=("conversation_render_projection",),
        code_refs=(
            "polylogue.pipeline.services.rendering.RenderService.render_conversations",
            "polylogue.rendering.renderers.markdown.MarkdownRenderer.render",
            "polylogue.rendering.renderers.html.HTMLRenderer.render",
        ),
        health_surfaces=("render",),
    ),
    ArtifactNode(
        name="site_conversation_pages",
        layer=ArtifactLayer.PROJECTION,
        description="Conversation page artifacts emitted by the static-site builder.",
        depends_on=("conversation_render_projection",),
        code_refs=(
            "polylogue.site.conversation_pages.generate_conversation_page",
            "polylogue.site.site_builder_archive.generate_conversation_page_for_builder",
        ),
        health_surfaces=("site",),
    ),
    ArtifactNode(
        name="site_publication_manifest",
        layer=ArtifactLayer.PROJECTION,
        description="Typed site publication manifest written to disk after a successful site build.",
        depends_on=("site_conversation_pages",),
        code_refs=(
            "polylogue.site.publication_flow.build_site_publication_manifest",
            "polylogue.site.publication_flow.write_site_publication_manifest",
        ),
        health_surfaces=("site", "maintenance"),
    ),
    ArtifactNode(
        name="publication_records",
        layer=ArtifactLayer.DURABLE,
        description="Persisted publication records storing the latest publication manifests in the archive database.",
        depends_on=("site_publication_manifest",),
        code_refs=(
            "polylogue.storage.repository_writes.RepositoryWritesMixin.record_publication",
            "polylogue.storage.repository_writes.RepositoryWritesMixin.get_latest_publication",
            "polylogue.storage.backends.queries.publications",
        ),
        health_surfaces=("site", "maintenance"),
    ),
    ArtifactNode(
        name="schema_packages",
        layer=ArtifactLayer.DURABLE,
        description="Versioned provider schema packages stored in the schema registry.",
        code_refs=(
            "polylogue.schemas.registry.SchemaRegistry",
            "polylogue.schemas.operator_inference.list_schemas",
            "polylogue.schemas.operator_resolution.explain_schema",
        ),
        health_surfaces=("schema", "qa"),
    ),
    ArtifactNode(
        name="schema_cluster_manifests",
        layer=ArtifactLayer.DURABLE,
        description="Persisted schema-cluster manifests describing observed structural families per provider.",
        code_refs=(
            "polylogue.schemas.registry.SchemaRegistry",
            "polylogue.schemas.operator_inference.list_schemas",
        ),
        health_surfaces=("schema", "qa"),
    ),
    ArtifactNode(
        name="inferred_corpus_specs",
        layer=ArtifactLayer.PROJECTION,
        description="Synthetic corpus specs distilled from schema packages and cluster manifests.",
        depends_on=("schema_packages", "schema_cluster_manifests"),
        code_refs=(
            "polylogue.scenarios.corpus.build_inferred_corpus_specs",
            "polylogue.schemas.operator_inference.list_inferred_corpus_specs",
        ),
        health_surfaces=("schema", "qa", "synthetic"),
    ),
    ArtifactNode(
        name="inferred_corpus_scenarios",
        layer=ArtifactLayer.PROJECTION,
        description="Compiled inferred corpus scenarios grouped from inferred corpus specs.",
        depends_on=("inferred_corpus_specs",),
        code_refs=(
            "polylogue.scenarios.corpus.build_corpus_scenarios",
            "polylogue.schemas.operator_inference.list_inferred_corpus_scenarios",
        ),
        health_surfaces=("schema", "qa", "synthetic"),
    ),
    ArtifactNode(
        name="schema_list_results",
        layer=ArtifactLayer.PROJECTION,
        description="Schema list/read results combining package catalogs, manifests, and inferred corpus projections.",
        depends_on=("schema_packages", "schema_cluster_manifests", "inferred_corpus_specs", "inferred_corpus_scenarios"),
        code_refs=(
            "polylogue.schemas.operator_inference.list_schemas",
            "polylogue.cli.commands.schema.schema_list",
        ),
        health_surfaces=("schema", "cli"),
    ),
    ArtifactNode(
        name="schema_explanation_results",
        layer=ArtifactLayer.PROJECTION,
        description="Schema explain/read results for provider package elements and annotations.",
        depends_on=("schema_packages",),
        code_refs=(
            "polylogue.schemas.operator_resolution.explain_schema",
            "polylogue.cli.commands.schema.schema_explain",
        ),
        health_surfaces=("schema", "cli"),
    ),
    ArtifactNode(
        name="archive_health",
        layer=ArtifactLayer.PROJECTION,
        description="Projected archive-wide health and maintenance view over message FTS and durable derived-model readiness.",
        depends_on=("message_fts", "action_event_health", "session_product_health"),
        code_refs=(
            "polylogue.health.run_archive_health",
            "polylogue.storage.derived_status.collect_derived_model_statuses_sync",
            "polylogue.storage.repair.collect_archive_debt_statuses_sync",
        ),
        repair_targets=("fts", "action_event_read_model", "session_products"),
        health_surfaces=("doctor", "archive_debt", "maintenance"),
    ),
)

RUNTIME_ARTIFACT_PATHS: tuple[ArtifactPath, ...] = (
    ArtifactPath(
        name="raw-reparse-loop",
        description="Raw validation state to validation/parse backlog and quarantine projections.",
        nodes=(
            "raw_validation_state",
            "validation_backlog",
            "parse_backlog",
            "parse_quarantine",
        ),
    ),
    ArtifactPath(
        name="raw-archive-ingest-loop",
        description="Raw validation and parse planning through durable archive runtime rows and their downstream source surfaces.",
        nodes=(
            "raw_validation_state",
            "validation_backlog",
            "parse_backlog",
            "parse_quarantine",
            "archive_conversation_rows",
            "message_source_rows",
            "tool_use_source_blocks",
            "session_product_source_conversations",
        ),
    ),
    ArtifactPath(
        name="action-event-repair-loop",
        description="Tool-use source blocks through action-event rows, FTS, and projected repair semantics.",
        nodes=(
            "tool_use_source_blocks",
            "action_event_rows",
            "action_event_fts",
            "action_event_health",
        ),
    ),
    ArtifactPath(
        name="session-product-repair-loop",
        description="Archive conversations through durable session-product rows, FTS, and projected repair semantics.",
        nodes=(
            "session_product_source_conversations",
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
            "session_product_health",
        ),
    ),
    ArtifactPath(
        name="session-profile-query-loop",
        description="Durable session profile rows through profile FTS and profile query results.",
        nodes=(
            "session_profile_rows",
            "session_profile_merged_fts",
            "session_profile_results",
        ),
    ),
    ArtifactPath(
        name="session-enrichment-query-loop",
        description="Durable session profile rows through enrichment FTS and enrichment query results.",
        nodes=(
            "session_profile_rows",
            "session_profile_enrichment_fts",
            "session_enrichment_results",
        ),
    ),
    ArtifactPath(
        name="session-work-event-query-loop",
        description="Durable session work-event rows through work-event FTS and query results.",
        nodes=(
            "session_work_event_rows",
            "session_work_event_fts",
            "session_work_event_results",
        ),
    ),
    ArtifactPath(
        name="session-phase-query-loop",
        description="Durable session phase rows through phase query results.",
        nodes=(
            "session_phase_rows",
            "session_phase_results",
        ),
    ),
    ArtifactPath(
        name="work-thread-query-loop",
        description="Durable work-thread rows through work-thread FTS and query results.",
        nodes=(
            "work_thread_rows",
            "work_thread_fts",
            "work_thread_results",
        ),
    ),
    ArtifactPath(
        name="session-tag-rollup-query-loop",
        description="Durable tag-rollup rows through tag-rollup query results.",
        nodes=(
            "session_tag_rollup_rows",
            "session_tag_rollup_results",
        ),
    ),
    ArtifactPath(
        name="day-summary-query-loop",
        description="Durable day-summary rows through day-summary query results.",
        nodes=(
            "day_session_summary_rows",
            "day_session_summary_results",
        ),
    ),
    ArtifactPath(
        name="week-summary-query-loop",
        description="Durable day-summary rows through week-summary query results.",
        nodes=(
            "day_session_summary_rows",
            "week_session_summary_results",
        ),
    ),
    ArtifactPath(
        name="session-product-status-query-loop",
        description="Projected session-product health through status query results.",
        nodes=(
            "session_product_health",
            "session_product_status_results",
        ),
    ),
    ArtifactPath(
        name="provider-analytics-query-loop",
        description="Durable session-product aggregates through provider-analytics query results.",
        nodes=(
            "session_product_rows",
            "provider_analytics_results",
        ),
    ),
    ArtifactPath(
        name="archive-debt-query-loop",
        description="Projected derived-model health through archive debt query results.",
        nodes=(
            "action_event_health",
            "session_product_health",
            "archive_health",
            "archive_debt_results",
        ),
    ),
    ArtifactPath(
        name="message-fts-health-loop",
        description="Persisted messages through lexical FTS and the archive-wide health projection.",
        nodes=(
            "message_source_rows",
            "message_fts",
            "action_event_health",
            "session_product_health",
            "archive_health",
        ),
    ),
    ArtifactPath(
        name="conversation-query-loop",
        description="Lexical message FTS through conversation-level query and search result projections.",
        nodes=(
            "message_fts",
            "conversation_query_results",
        ),
    ),
    ArtifactPath(
        name="conversation-render-loop",
        description="Durable archive conversations through repository render projections and rendered filesystem artifacts.",
        nodes=(
            "archive_conversation_rows",
            "conversation_render_projection",
            "rendered_conversation_artifacts",
        ),
    ),
    ArtifactPath(
        name="site-publication-loop",
        description="Durable archive conversations through site conversation pages, publication manifest output, and persisted publication records.",
        nodes=(
            "archive_conversation_rows",
            "conversation_render_projection",
            "site_conversation_pages",
            "site_publication_manifest",
            "publication_records",
        ),
    ),
    ArtifactPath(
        name="inferred-corpus-compilation-loop",
        description="Schema packages and cluster manifests through inferred corpus specs and compiled inferred corpus scenarios.",
        nodes=(
            "schema_packages",
            "schema_cluster_manifests",
            "inferred_corpus_specs",
            "inferred_corpus_scenarios",
        ),
    ),
    ArtifactPath(
        name="schema-list-query-loop",
        description="Schema registry packages and manifests through inferred corpus projections and schema list results.",
        nodes=(
            "schema_packages",
            "schema_cluster_manifests",
            "inferred_corpus_specs",
            "inferred_corpus_scenarios",
            "schema_list_results",
        ),
    ),
    ArtifactPath(
        name="schema-explain-query-loop",
        description="Schema registry packages through schema explanation results.",
        nodes=(
            "schema_packages",
            "schema_explanation_results",
        ),
    ),
)


def build_runtime_artifact_nodes() -> tuple[ArtifactNode, ...]:
    return RUNTIME_ARTIFACT_NODES


def build_runtime_artifact_paths() -> tuple[ArtifactPath, ...]:
    return RUNTIME_ARTIFACT_PATHS


__all__ = [
    "ArtifactLayer",
    "ArtifactNode",
    "ArtifactPath",
    "RUNTIME_ARTIFACT_NODES",
    "RUNTIME_ARTIFACT_PATHS",
    "build_runtime_artifact_nodes",
    "build_runtime_artifact_paths",
]
