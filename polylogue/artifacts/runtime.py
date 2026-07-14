"""Runtime artifact graph data and accessors."""

from __future__ import annotations

from polylogue.artifacts.descriptors import ArtifactLayer, ArtifactNode, ArtifactPath

RUNTIME_ARTIFACT_NODES: tuple[ArtifactNode, ...] = (
    ArtifactNode(
        name="configured_sources",
        layer=ArtifactLayer.SOURCE,
        description="Configured filesystem and Drive source descriptors selected for daemon ingestion.",
        code_refs=(
            "polylogue.config.Source",
            "polylogue.pipeline.services.planning.PlanningService.build_plan",
            "polylogue.cli.shared.helpers.resolve_sources",
        ),
        readiness_surfaces=("polylogued", "sources"),
    ),
    ArtifactNode(
        name="source_payload_stream",
        layer=ArtifactLayer.SOURCE,
        description="Traversed raw source payloads yielded as RawSessionData before durable persistence.",
        depends_on=("configured_sources",),
        code_refs=(
            "polylogue.sources.source_acquisition.iter_source_raw_data",
            "polylogue.pipeline.services.acquisition.AcquisitionService.visit_sources",
        ),
        readiness_surfaces=("polylogued", "sources"),
    ),
    ArtifactNode(
        name="raw_validation_state",
        layer=ArtifactLayer.DURABLE,
        description="Persisted raw-session validation and parse state in raw_sessions.",
        depends_on=("source_payload_stream",),
        code_refs=(
            "polylogue.storage.raw.artifacts.RawIngestArtifactState",
            "polylogue.storage.sqlite.queries.raw_state",
        ),
    ),
    ArtifactNode(
        name="artifact_observation_rows",
        layer=ArtifactLayer.DURABLE,
        description="Persisted artifact observations recorded alongside acquired raw sessions.",
        depends_on=("raw_validation_state",),
        code_refs=(
            "polylogue.pipeline.services.acquisition_persistence.persist_raw_record",
            "polylogue.storage.repository.raw.repository_raw.RepositoryRawMixin.save_artifact_observation",
            "polylogue.storage.artifacts.inspection.inspect_raw_artifact",
        ),
        readiness_surfaces=("run", "sources"),
    ),
    ArtifactNode(
        name="validation_backlog",
        layer=ArtifactLayer.PROJECTION,
        description="Raw records that still require validation before ordinary parse planning can trust them.",
        depends_on=("raw_validation_state",),
        code_refs=(
            "polylogue.storage.raw.artifacts.validation_backlog_query_spec",
            "polylogue.pipeline.services.planning_backlog.collect_validation_backlog",
        ),
    ),
    ArtifactNode(
        name="parse_backlog",
        layer=ArtifactLayer.PROJECTION,
        description="Raw records eligible for parse planning, including force-reparse simulation.",
        depends_on=("raw_validation_state",),
        code_refs=(
            "polylogue.storage.raw.artifacts.parse_backlog_query_spec",
            "polylogue.pipeline.services.planning_backlog.collect_parse_backlog",
            "polylogue.pipeline.services.planning_runtime.build_ingest_plan",
        ),
    ),
    ArtifactNode(
        name="parse_quarantine",
        layer=ArtifactLayer.PROJECTION,
        description="Validation-failed unparsed raws that stay out of ordinary parse backlog but return under force-reparse.",
        depends_on=("raw_validation_state",),
        code_refs=("polylogue.storage.raw.artifacts.RawIngestArtifactState",),
    ),
    ArtifactNode(
        name="archive_session_rows",
        layer=ArtifactLayer.DURABLE,
        description="Durable session, message, attachment, and content-block rows written from parsed raw sessions.",
        depends_on=("raw_validation_state",),
        code_refs=(
            "polylogue.pipeline.services.parsing_workflow.parse_from_raw",
            "polylogue.storage.sqlite.archive_tiers.write.write_parsed_session_to_archive",
            "polylogue.storage.repository.raw.repository_raw.RepositoryRawMixin.mark_raw_parsed",
        ),
    ),
    ArtifactNode(
        name="message_source_rows",
        layer=ArtifactLayer.SOURCE,
        description="Persisted message rows that feed lexical FTS indexing and archive search.",
        depends_on=("archive_session_rows",),
        code_refs=(
            "polylogue.storage.fts.fts_lifecycle.FTS_INDEXABLE_MESSAGE_COUNT_SQL",
            "polylogue.storage.fts.fts_lifecycle.repair_fts_index_sync",
        ),
    ),
    ArtifactNode(
        name="message_fts",
        layer=ArtifactLayer.INDEX,
        description="Lexical FTS projection over persisted message rows.",
        depends_on=("message_source_rows",),
        code_refs=(
            "polylogue.storage.fts.fts_lifecycle.message_fts_readiness_sync",
            "polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync",
            "polylogue.storage.fts.fts_lifecycle.repair_fts_index_sync",
        ),
        readiness_surfaces=("daemon", "archive_debt", "query"),
    ),
    ArtifactNode(
        name="embedding_metadata_rows",
        layer=ArtifactLayer.DURABLE,
        description="Durable embedding metadata rows storing model, dimension, timestamp, and provenance for embedded messages.",
        depends_on=("archive_session_rows",),
        code_refs=(
            "polylogue.storage.search_providers.sqlite_vec_runtime.SqliteVecRuntimeMixin._ensure_tables",
            "polylogue.storage.search_providers.sqlite_vec_queries.SqliteVecQueryMixin.upsert",
        ),
        readiness_surfaces=("embed", "doctor", "retrieval"),
    ),
    ArtifactNode(
        name="embedding_status_rows",
        layer=ArtifactLayer.DURABLE,
        description="Durable per-session embedding status rows recording embedded message counts and reindex needs.",
        depends_on=("archive_session_rows",),
        code_refs=(
            "polylogue.storage.search_providers.sqlite_vec_runtime.SqliteVecRuntimeMixin._ensure_tables",
            "polylogue.storage.search_providers.sqlite_vec_queries.SqliteVecQueryMixin.upsert",
        ),
        readiness_surfaces=("embed", "doctor", "retrieval"),
    ),
    ArtifactNode(
        name="message_embedding_vectors",
        layer=ArtifactLayer.INDEX,
        description="Semantic vector index over embedded archive messages.",
        depends_on=("archive_session_rows",),
        code_refs=(
            "polylogue.storage.search_providers.sqlite_vec_runtime.SqliteVecRuntimeMixin._ensure_tables",
            "polylogue.storage.search_providers.sqlite_vec_queries.SqliteVecQueryMixin.upsert",
            "polylogue.storage.search_providers.sqlite_vec_queries.SqliteVecQueryMixin.query",
        ),
        readiness_surfaces=("embed", "retrieval", "query"),
    ),
    ArtifactNode(
        name="session_insight_source_sessions",
        layer=ArtifactLayer.SOURCE,
        description="Hydratable session/message/attachment/block rows that feed durable session-insight rebuilds.",
        depends_on=("archive_session_rows",),
        code_refs=(
            "polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync",
            "polylogue.storage.insights.session.refresh.refresh_session_insights_for_session_async",
        ),
    ),
    ArtifactNode(
        name="session_profile_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable session profile rows materialized from archive sessions.",
        depends_on=("session_insight_source_sessions",),
        code_refs=(
            "polylogue.storage.insights.session.rebuild.build_session_insight_records",
            "polylogue.storage.insights.session.profiles.build_session_profile_record",
        ),
        repair_targets=("session_insights",),
        readiness_surfaces=("doctor", "archive_debt", "insights"),
    ),
    ArtifactNode(
        name="session_work_event_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable session work-event rows materialized from session profile analysis.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.insights.session.rebuild.build_session_insight_records",
            "polylogue.storage.insights.session.timeline_rows.build_session_work_event_records",
        ),
        repair_targets=("session_insights",),
        readiness_surfaces=("doctor", "archive_debt", "insights"),
    ),
    ArtifactNode(
        name="session_work_event_fts",
        layer=ArtifactLayer.INDEX,
        description="Session work-event FTS projection over durable work-event rows.",
        depends_on=("session_work_event_rows",),
        code_refs=(
            "polylogue.storage.insights.session.status",
            "polylogue.storage.sqlite.archive_tiers.index",
        ),
        repair_targets=("session_insights",),
        readiness_surfaces=("doctor", "archive_debt", "insights"),
    ),
    ArtifactNode(
        name="session_phase_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable session phase rows materialized from session profile analysis.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.insights.session.rebuild.build_session_insight_records",
            "polylogue.storage.insights.session.timeline_rows.build_session_phase_records",
        ),
        repair_targets=("session_insights",),
        readiness_surfaces=("doctor", "archive_debt", "insights"),
    ),
    ArtifactNode(
        name="thread_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable thread rows materialized from session profile families.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.insights.session.threads.build_thread_records_for_roots_sync",
            "polylogue.storage.insights.session.threads.build_thread_records_for_roots_async",
        ),
        repair_targets=("session_insights",),
        readiness_surfaces=("doctor", "archive_debt", "insights"),
    ),
    ArtifactNode(
        name="thread_fts",
        layer=ArtifactLayer.INDEX,
        description="Thread FTS projection over durable thread rows.",
        depends_on=("thread_rows",),
        code_refs=(
            "polylogue.storage.insights.session.status",
            "polylogue.storage.sqlite.archive_tiers.index",
        ),
        repair_targets=("session_insights",),
        readiness_surfaces=("doctor", "archive_debt", "insights"),
    ),
    ArtifactNode(
        name="session_tag_rollup_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable provider/day/tag rollup rows aggregated from session profiles.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.insights.session.aggregates.refresh_sync_provider_day_aggregates",
            "polylogue.storage.insights.session.status",
        ),
        repair_targets=("session_insights",),
        readiness_surfaces=("doctor", "archive_debt", "insights"),
    ),
    ArtifactNode(
        name="session_insight_rows",
        layer=ArtifactLayer.DERIVED,
        description="Durable session-profile, timeline, thread, and aggregate rows derived from archive sessions.",
        depends_on=(
            "session_profile_rows",
            "session_work_event_rows",
            "session_phase_rows",
            "thread_rows",
            "session_tag_rollup_rows",
        ),
        code_refs=(
            "polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync",
            "polylogue.storage.insights.session.status",
        ),
        repair_targets=("session_insights",),
        readiness_surfaces=("doctor", "archive_debt", "insights"),
    ),
    ArtifactNode(
        name="session_insight_fts",
        layer=ArtifactLayer.INDEX,
        description="The session-insight FTS family over profiles, work events, and threads.",
        depends_on=(
            "session_work_event_fts",
            "thread_fts",
        ),
        code_refs=(
            "polylogue.storage.insights.session.status",
            "polylogue.storage.derived.insights",
        ),
        repair_targets=("session_insights",),
        readiness_surfaces=("doctor", "archive_debt", "insights"),
    ),
    ArtifactNode(
        name="session_insight_readiness",
        layer=ArtifactLayer.PROJECTION,
        description="Projected readiness, debt, and stale-surface semantics for durable session insights.",
        depends_on=("session_insight_rows", "session_insight_fts"),
        code_refs=(
            "polylogue.storage.insights.session.status",
            "polylogue.storage.repair",
            "polylogue.cli.commands.insights",
        ),
        repair_targets=("session_insights",),
        readiness_surfaces=("doctor", "archive_debt", "insights"),
    ),
    ArtifactNode(
        name="retrieval_band_readiness",
        layer=ArtifactLayer.PROJECTION,
        description="Projected transcript/evidence/inference/enrichment retrieval readiness over embeddings and durable read models.",
        depends_on=(
            "embedding_metadata_rows",
            "embedding_status_rows",
            "message_embedding_vectors",
            "session_insight_readiness",
        ),
        code_refs=(
            "polylogue.storage.embeddings.embedding_stats.read_embedding_stats_sync",
            "polylogue.storage.embeddings.support.build_retrieval_bands_from_status",
            "polylogue.storage.derived.derived_status.build_retrieval_statuses",
        ),
        readiness_surfaces=("embed", "doctor", "retrieval"),
    ),
    ArtifactNode(
        name="embedding_status_results",
        layer=ArtifactLayer.PROJECTION,
        description="Operator-facing embedding status payload and retrieval-band readiness view.",
        depends_on=(
            "embedding_metadata_rows",
            "embedding_status_rows",
            "message_embedding_vectors",
            "retrieval_band_readiness",
        ),
        code_refs=(
            "polylogue.storage.embeddings.status_payload.embedding_status_payload",
            "polylogue.cli.shared.embed_stats.show_embedding_stats",
        ),
        readiness_surfaces=("embed", "doctor", "retrieval", "cli"),
    ),
    ArtifactNode(
        name="session_profile_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable session profiles.",
        depends_on=("session_profile_rows",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_session_profile_insights",
            "polylogue.cli.commands.insights",
        ),
        readiness_surfaces=("insights", "facade", "mcp"),
    ),
    ArtifactNode(
        name="session_work_event_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable session work events.",
        depends_on=("session_work_event_rows", "session_work_event_fts"),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_session_work_event_insights",
            "polylogue.cli.commands.insights",
        ),
        readiness_surfaces=("insights", "facade", "mcp"),
    ),
    ArtifactNode(
        name="session_phase_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable session phases.",
        depends_on=("session_phase_rows",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_session_phase_insights",
            "polylogue.cli.commands.insights",
        ),
        readiness_surfaces=("insights", "facade", "mcp"),
    ),
    ArtifactNode(
        name="thread_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable threads.",
        depends_on=("thread_rows", "thread_fts"),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_thread_insights",
            "polylogue.cli.commands.insights",
        ),
        readiness_surfaces=("insights", "facade", "mcp"),
    ),
    ArtifactNode(
        name="session_tag_rollup_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable session tag rollups.",
        depends_on=("session_tag_rollup_rows",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_session_tag_rollup_insights",
            "polylogue.cli.commands.insights",
        ),
        readiness_surfaces=("insights", "facade", "mcp"),
    ),
    ArtifactNode(
        name="session_insight_status_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for durable session-insight status views.",
        depends_on=("session_insight_readiness",),
        code_refs=(
            "polylogue.api.archive.PolylogueArchiveMixin.get_session_insight_status",
            "polylogue.cli.commands.insights",
        ),
        readiness_surfaces=("insights", "facade", "mcp"),
    ),
    ArtifactNode(
        name="archive_coverage_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for provider, day, or week archive coverage rollups.",
        depends_on=("archive_session_rows", "session_profile_rows"),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_archive_coverage_insights",
            "polylogue.cli.commands.insights",
            "polylogue.cli.shared.helper_summary",
        ),
        readiness_surfaces=("insights", "facade", "mcp", "helpers"),
    ),
    ArtifactNode(
        name="tool_usage_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for per-origin tool usage analytics from the canonical actions view.",
        depends_on=("archive_session_rows",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_tool_usage_insights",
            "polylogue.cli.commands.insights",
        ),
        readiness_surfaces=("insights", "facade", "mcp"),
    ),
    ArtifactNode(
        name="archive_debt_results",
        layer=ArtifactLayer.PROJECTION,
        description="Query/read results for archive debt views derived from projected readiness and maintenance state.",
        depends_on=("session_insight_readiness", "archive_readiness"),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_archive_debt_insights",
            "polylogue.cli.commands.insights",
        ),
        readiness_surfaces=("insights", "facade", "mcp", "maintenance"),
    ),
    ArtifactNode(
        name="session_query_results",
        layer=ArtifactLayer.PROJECTION,
        description="Session-level query and search results resolved from lexical retrieval over the archive.",
        depends_on=("message_fts",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_summaries",
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.search_summaries",
            "polylogue.archive.query.archive_execution",
        ),
        readiness_surfaces=("query", "mcp", "facade"),
    ),
    ArtifactNode(
        name="session_digest",
        layer=ArtifactLayer.PROJECTION,
        description="Deterministic digest envelope compiled from one archived session for successor context.",
        depends_on=("archive_session_rows", "message_source_rows"),
        code_refs=(
            "polylogue.insights.transforms.compile_session_digest",
            "polylogue.api.archive.PolylogueArchiveMixin._session_digest",
            "polylogue.cli.query_verbs.continue_verb",
        ),
        readiness_surfaces=("report", "cli"),
    ),
    ArtifactNode(
        name="forensic_index",
        layer=ArtifactLayer.PROJECTION,
        description="Evidence index attached to a session digest for command/tool/file/action reconstruction.",
        depends_on=("session_digest",),
        code_refs=(
            "polylogue.insights.transforms.compile_session_digest",
            "polylogue.insights.transforms.SessionDigest.forensic_index",
        ),
        readiness_surfaces=("report", "cli"),
    ),
    ArtifactNode(
        name="resume_bundle",
        layer=ArtifactLayer.PROJECTION,
        description="Compact continuation bundle produced with the session digest for session handoff.",
        depends_on=("session_digest",),
        code_refs=(
            "polylogue.insights.transforms.compile_session_digest",
            "polylogue.insights.transforms.render_resume_bundle",
        ),
        readiness_surfaces=("report", "cli"),
    ),
    ArtifactNode(
        name="session_report_markdown",
        layer=ArtifactLayer.PROJECTION,
        description="Rendered session report preset output derived from a compiled session digest.",
        depends_on=("session_digest", "forensic_index"),
        code_refs=(
            "polylogue.insights.transforms.render_session_report",
            "polylogue.insights.transforms.SessionDigest.report_markdown",
            "polylogue.cli.query_verbs.continue_verb",
        ),
        readiness_surfaces=("report", "cli", "mcp"),
    ),
    ArtifactNode(
        name="schema_packages",
        layer=ArtifactLayer.DURABLE,
        description="Versioned provider schema packages stored in the schema registry.",
        code_refs=(
            "polylogue.schemas.registry.SchemaRegistry",
            "polylogue.schemas.operator.inference.list_schemas",
            "polylogue.schemas.operator.resolution.explain_schema",
        ),
        readiness_surfaces=("schema", "verification-lab"),
    ),
    ArtifactNode(
        name="schema_cluster_manifests",
        layer=ArtifactLayer.DURABLE,
        description="Persisted schema-cluster manifests describing observed structural families per provider.",
        code_refs=(
            "polylogue.schemas.registry.SchemaRegistry",
            "polylogue.schemas.operator.inference.list_schemas",
        ),
        readiness_surfaces=("schema", "verification-lab"),
    ),
    ArtifactNode(
        name="inferred_corpus_specs",
        layer=ArtifactLayer.PROJECTION,
        description="Synthetic corpus specs distilled from schema packages and cluster manifests.",
        depends_on=("schema_packages", "schema_cluster_manifests"),
        code_refs=(
            "polylogue.scenarios.corpus.build_inferred_corpus_specs",
            "polylogue.schemas.operator.inference.list_inferred_corpus_specs",
        ),
        readiness_surfaces=("schema", "verification-lab", "synthetic"),
    ),
    ArtifactNode(
        name="inferred_corpus_scenarios",
        layer=ArtifactLayer.PROJECTION,
        description="Compiled inferred corpus scenarios grouped from inferred corpus specs.",
        depends_on=("inferred_corpus_specs",),
        code_refs=(
            "polylogue.scenarios.corpus.build_corpus_scenarios",
            "polylogue.schemas.operator.inference.list_inferred_corpus_scenarios",
        ),
        readiness_surfaces=("schema", "verification-lab", "synthetic"),
    ),
    ArtifactNode(
        name="schema_list_results",
        layer=ArtifactLayer.PROJECTION,
        description="Schema list/read results combining package catalogs, manifests, and inferred corpus projections.",
        depends_on=(
            "schema_packages",
            "schema_cluster_manifests",
            "inferred_corpus_specs",
            "inferred_corpus_scenarios",
        ),
        code_refs=(
            "polylogue.schemas.operator.inference.list_schemas",
            "polylogue.cli.shared.schema_rendering_results.render_schema_list_result",
        ),
        readiness_surfaces=("schema", "cli"),
    ),
    ArtifactNode(
        name="schema_explanation_results",
        layer=ArtifactLayer.PROJECTION,
        description="Schema explain/read results for provider package elements and annotations.",
        depends_on=("schema_packages",),
        code_refs=(
            "polylogue.schemas.operator.resolution.explain_schema",
            "polylogue.cli.shared.schema_rendering_explain.render_schema_explain_result",
        ),
        readiness_surfaces=("schema", "cli"),
    ),
    ArtifactNode(
        name="archive_readiness",
        layer=ArtifactLayer.PROJECTION,
        description="Projected archive-wide readiness and maintenance view over message FTS and durable derived-model readiness.",
        depends_on=("message_fts", "session_insight_readiness", "retrieval_band_readiness"),
        code_refs=(
            "polylogue.readiness.run_archive_readiness",
            "polylogue.storage.derived.derived_status.collect_derived_model_statuses_sync",
            "polylogue.storage.repair.collect_archive_debt_statuses_sync",
        ),
        repair_targets=("session_insights",),
        readiness_surfaces=("daemon", "archive_debt", "maintenance"),
    ),
)

RUNTIME_ARTIFACT_PATHS: tuple[ArtifactPath, ...] = (
    ArtifactPath(
        name="source-acquisition-loop",
        description="Configured sources through traversed raw payloads into durable raw session state and artifact observations.",
        nodes=(
            "configured_sources",
            "source_payload_stream",
            "raw_validation_state",
            "artifact_observation_rows",
        ),
    ),
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
            "configured_sources",
            "source_payload_stream",
            "raw_validation_state",
            "artifact_observation_rows",
            "validation_backlog",
            "parse_backlog",
            "parse_quarantine",
            "archive_session_rows",
            "message_source_rows",
            "session_insight_source_sessions",
        ),
    ),
    ArtifactPath(
        name="embedding-materialization-loop",
        description="Durable archive sessions through embedding metadata, status rows, and vector index materialization.",
        nodes=(
            "archive_session_rows",
            "embedding_metadata_rows",
            "embedding_status_rows",
            "message_embedding_vectors",
        ),
    ),
    ArtifactPath(
        name="session-insight-repair-loop",
        description="Archive sessions through durable session-insight rows, FTS, and projected repair semantics.",
        nodes=(
            "session_insight_source_sessions",
            "session_profile_rows",
            "session_work_event_rows",
            "session_work_event_fts",
            "session_phase_rows",
            "thread_rows",
            "thread_fts",
            "session_tag_rollup_rows",
            "session_insight_rows",
            "session_insight_fts",
            "session_insight_readiness",
        ),
    ),
    ArtifactPath(
        name="raw-session-insight-repair-loop",
        description=(
            "Raw validation and archive core rows through durable session-insight rows, FTS, and projected repair semantics."
        ),
        nodes=(
            "configured_sources",
            "source_payload_stream",
            "raw_validation_state",
            "archive_session_rows",
            "session_insight_source_sessions",
            "session_profile_rows",
            "session_work_event_rows",
            "session_phase_rows",
            "thread_rows",
            "session_tag_rollup_rows",
            "session_insight_rows",
            "session_insight_fts",
            "session_insight_readiness",
        ),
    ),
    ArtifactPath(
        name="retrieval-band-readiness-loop",
        description="Embedding state plus session-insight readiness through retrieval-band and archive readiness projections.",
        nodes=(
            "embedding_metadata_rows",
            "embedding_status_rows",
            "message_embedding_vectors",
            "session_insight_readiness",
            "retrieval_band_readiness",
            "archive_readiness",
        ),
    ),
    ArtifactPath(
        name="session-profile-query-loop",
        description="Durable session profile rows through profile FTS and profile query results.",
        nodes=(
            "session_profile_rows",
            "session_profile_results",
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
        name="thread-query-loop",
        description="Durable thread rows through thread FTS and query results.",
        nodes=(
            "thread_rows",
            "thread_fts",
            "thread_results",
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
        name="session-insight-status-query-loop",
        description="Projected session-insight readiness through status query results.",
        nodes=(
            "session_insight_readiness",
            "session_insight_status_results",
        ),
    ),
    ArtifactPath(
        name="archive-coverage-query-loop",
        description="Archive and session-profile rows through provider/day/week coverage query results.",
        nodes=(
            "archive_session_rows",
            "session_profile_rows",
            "archive_coverage_results",
        ),
    ),
    ArtifactPath(
        name="tool-usage-query-loop",
        description="Canonical actions view over archive rows through tool-usage analytics query results.",
        nodes=(
            "archive_session_rows",
            "tool_usage_results",
        ),
    ),
    ArtifactPath(
        name="embedding-status-query-loop",
        description="Embedding state and retrieval-band readiness through operator-facing embedding status results.",
        nodes=(
            "embedding_metadata_rows",
            "embedding_status_rows",
            "message_embedding_vectors",
            "retrieval_band_readiness",
            "embedding_status_results",
        ),
    ),
    ArtifactPath(
        name="archive-debt-query-loop",
        description="Projected derived-model readiness through archive debt query results.",
        nodes=(
            "session_insight_readiness",
            "archive_readiness",
            "archive_debt_results",
        ),
    ),
    ArtifactPath(
        name="message-fts-readiness-loop",
        description="Persisted messages through lexical FTS and the archive-wide readiness projection.",
        nodes=(
            "message_source_rows",
            "message_fts",
            "session_insight_readiness",
            "archive_readiness",
        ),
    ),
    ArtifactPath(
        name="session-query-loop",
        description="Lexical message FTS through session-level query and search result projections.",
        nodes=(
            "message_fts",
            "session_query_results",
        ),
    ),
    ArtifactPath(
        name="session-digest-transform-loop",
        description="Archived session/message rows through deterministic session digest and report projections.",
        nodes=(
            "archive_session_rows",
            "message_source_rows",
            "session_digest",
            "forensic_index",
            "resume_bundle",
            "session_report_markdown",
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
