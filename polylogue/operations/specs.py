"""Typed runtime operation metadata shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Literal

from polylogue.core.json import JSONDocument, JSONDocumentList, json_document

Effect = Literal["Pure", "DbRead", "DbWrite", "FileWrite", "Network", "LiveArchive", "Destructive"]
"""Declared runtime effect of an operation.

Each effect implies specific guarantees that the verification catalog
must check:

  Pure        → deterministic, no_side_effect
  DbRead      → snapshot_consistent
  DbWrite     → preview, idempotent, rollback_safe, atomic
  FileWrite   → reserved for future durable file-write operations
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
    IMPORT = "import"
    MAINTENANCE = "maintenance"


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
        name="acquire-raw-sessions",
        kind=OperationKind.MATERIALIZATION,
        description="Traverse configured sources, detect provider-shaped payloads, and persist raw session records plus artifact observations.",
        consumes=("configured_sources", "source_payload_stream"),
        produces=("raw_validation_state", "artifact_observation_rows"),
        path_targets=("source-acquisition-loop",),
        code_refs=(
            "polylogue.pipeline.run_stages.execute_acquire_stage",
            "polylogue.pipeline.services.acquisition.AcquisitionService.acquire_sources",
            "polylogue.sources.source_acquisition.iter_source_raw_data",
            "polylogue.pipeline.services.acquisition_persistence.persist_raw_record",
        ),
        surfaces=("daemon", "sources"),
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
        surfaces=("daemon", "reparse"),
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
        surfaces=("daemon", "reparse"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="ingest-archive-runtime",
        kind=OperationKind.MATERIALIZATION,
        description=(
            "Decode, validate, parse, transform, and persist raw sessions into the durable archive runtime tables."
        ),
        consumes=("raw_validation_state", "validation_backlog", "parse_backlog"),
        produces=("raw_validation_state", "archive_session_rows"),
        path_targets=("raw-archive-ingest-loop",),
        code_refs=(
            "polylogue.pipeline.services.parsing_workflow.parse_from_raw",
            "polylogue.storage.sqlite.archive_tiers.write.write_parsed_session_to_archive",
            "polylogue.storage.repository.raw.repository_raw.RepositoryRawMixin.mark_raw_validated",
            "polylogue.storage.repository.raw.repository_raw.RepositoryRawMixin.mark_raw_parsed",
        ),
        surfaces=("daemon", "reprocess", "ingest"),
        mutates_state=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="index-message-fts",
        kind=OperationKind.INDEXING,
        description="Build or repair lexical message FTS rows from persisted archive messages.",
        consumes=("message_source_rows",),
        produces=("message_fts",),
        path_targets=("message-fts-readiness-loop", "session-query-loop"),
        code_refs=(
            "polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync",
            "polylogue.storage.fts.fts_lifecycle.repair_fts_index_sync",
            "polylogue.storage.fts.fts_lifecycle.message_fts_readiness_sync",
        ),
        surfaces=("daemon", "doctor", "repair", "query"),
        mutates_state=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="materialize-transcript-embeddings",
        kind=OperationKind.MATERIALIZATION,
        description="Build or refresh transcript embedding metadata, session status rows, and semantic vector entries from archive sessions.",
        consumes=("archive_session_rows",),
        produces=("embedding_metadata_rows", "embedding_status_rows", "message_embedding_vectors"),
        path_targets=("embedding-materialization-loop",),
        code_refs=(
            "polylogue.cli.shared.embed_runtime.embed_batch",
            "polylogue.cli.shared.embed_runtime.embed_single",
            "polylogue.storage.search_providers.sqlite_vec_queries.SqliteVecQueryMixin.upsert",
        ),
        surfaces=("daemon", "embed", "retrieval"),
        mutates_state=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="query-sessions",
        kind=OperationKind.QUERY,
        description="Resolve session-level query and search results from archive retrieval plans.",
        consumes=("message_fts",),
        produces=("session_query_results",),
        path_targets=("session-query-loop",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_summaries",
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.search_summaries",
            "polylogue.archive.query.archive_execution",
        ),
        surfaces=("query", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="materialize-session-insights",
        kind=OperationKind.MATERIALIZATION,
        description="Build durable session-insight rows and their trigger-maintained FTS projections from archive sessions.",
        consumes=("session_insight_source_sessions",),
        produces=(
            "session_profile_rows",
            "session_work_event_rows",
            "session_work_event_fts",
            "session_phase_rows",
            "thread_rows",
            "thread_fts",
            "session_tag_rollup_rows",
            "session_insight_rows",
            "session_insight_fts",
        ),
        path_targets=("session-insight-repair-loop",),
        code_refs=(
            "polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync",
            "polylogue.storage.insights.session.refresh.refresh_session_insights_for_session_async",
        ),
        surfaces=("daemon", "insights", "doctor", "repair"),
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
            "session_insight_readiness",
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
            "polylogue.storage.embeddings.status_payload.embedding_status_payload",
            "polylogue.cli.shared.embed_stats.render_embedding_stats",
        ),
        surfaces=("daemon", "embed", "doctor", "retrieval"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="project-session-insight-readiness",
        kind=OperationKind.PROJECTION,
        description="Project readiness, debt, and stale-surface semantics from durable session-insight rows and FTS state.",
        consumes=("session_insight_rows", "session_insight_fts"),
        produces=("session_insight_readiness",),
        path_targets=("session-insight-repair-loop",),
        code_refs=(
            "polylogue.storage.insights.session.status",
            "polylogue.storage.repair",
            "polylogue.cli.commands.insights",
        ),
        surfaces=("insights", "doctor", "archive_debt", "repair"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-session-profiles",
        kind=OperationKind.QUERY,
        description="Resolve durable session-profile insights from profile rows and merged profile FTS.",
        consumes=("session_profile_rows",),
        produces=("session_profile_results",),
        path_targets=("session-profile-query-loop",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_session_profile_insights",
            "polylogue.cli.commands.insights",
        ),
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-session-work-events",
        kind=OperationKind.QUERY,
        description="Resolve durable session work-event insights from work-event rows and work-event FTS.",
        consumes=("session_work_event_rows", "session_work_event_fts"),
        produces=("session_work_event_results",),
        path_targets=("session-work-event-query-loop",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_session_work_event_insights",
            "polylogue.cli.commands.insights",
        ),
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-session-phases",
        kind=OperationKind.QUERY,
        description="Resolve durable session-phase insights from phase rows.",
        consumes=("session_phase_rows",),
        produces=("session_phase_results",),
        path_targets=("session-phase-query-loop",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_session_phase_insights",
            "polylogue.cli.commands.insights",
        ),
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-threads",
        kind=OperationKind.QUERY,
        description="Resolve durable thread insights from thread rows and thread FTS.",
        consumes=("thread_rows", "thread_fts"),
        produces=("thread_results",),
        path_targets=("thread-query-loop",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_thread_insights",
            "polylogue.cli.commands.insights",
        ),
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-session-tag-rollups",
        kind=OperationKind.QUERY,
        description="Resolve durable session tag-rollup insights from aggregate tag rows.",
        consumes=("session_tag_rollup_rows",),
        produces=("session_tag_rollup_results",),
        path_targets=("session-tag-rollup-query-loop",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_session_tag_rollup_insights",
            "polylogue.cli.commands.insights",
        ),
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-session-insight-status",
        kind=OperationKind.QUERY,
        description="Resolve projected session-insight status views from session-insight readiness state.",
        consumes=("session_insight_readiness",),
        produces=("session_insight_status_results",),
        path_targets=("session-insight-status-query-loop",),
        code_refs=(
            "polylogue.api.archive.PolylogueArchiveMixin.get_session_insight_status",
            "polylogue.cli.commands.insights",
        ),
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-archive-coverage",
        kind=OperationKind.QUERY,
        description="Resolve provider, day, or week archive coverage rollups from durable archive and session-profile rows.",
        consumes=("archive_session_rows", "session_profile_rows"),
        produces=("archive_coverage_results",),
        path_targets=("archive-coverage-query-loop",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_archive_coverage_insights",
            "polylogue.cli.commands.insights",
            "polylogue.cli.shared.helper_summary",
        ),
        surfaces=("insights", "facade", "mcp", "helpers"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-tool-usage",
        kind=OperationKind.QUERY,
        description="Resolve per-origin tool usage analytics from canonical archive actions.",
        consumes=("archive_session_rows",),
        produces=("tool_usage_results",),
        path_targets=("tool-usage-query-loop",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_tool_usage_insights",
            "polylogue.cli.commands.insights",
        ),
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-archive-debt",
        kind=OperationKind.QUERY,
        description="Resolve archive debt views from projected derived-model readiness and maintenance state.",
        consumes=("session_insight_readiness", "archive_readiness"),
        produces=("archive_debt_results",),
        path_targets=("archive-debt-query-loop",),
        code_refs=(
            "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.list_archive_debt_insights",
            "polylogue.cli.commands.insights",
        ),
        surfaces=("insights", "facade", "mcp", "maintenance"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="compile-recovery-digest",
        kind=OperationKind.PROJECTION,
        description="Compile one archived session into the deterministic recovery/digest artifact and evidence index.",
        consumes=("archive_session_rows", "message_rows"),
        produces=("recovery_digest", "forensic_index", "resume_bundle"),
        path_targets=("recovery-digest-transform-loop",),
        code_refs=(
            "polylogue.insights.transforms.compile_recovery_digest",
            "polylogue.api.archive.PolylogueArchive.recovery_digest",
            "polylogue.cli.query_verbs._render_recovery_read_view",
        ),
        surfaces=("api", "cli", "read-view"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="render-recovery-report",
        kind=OperationKind.PROJECTION,
        description="Render deterministic recovery report presets from a compiled recovery digest.",
        consumes=("recovery_digest", "forensic_index"),
        produces=("recovery_report_markdown",),
        path_targets=("recovery-digest-transform-loop",),
        code_refs=(
            "polylogue.insights.transforms.render_recovery_report",
            "polylogue.insights.transforms.RecoveryDigest.report_markdown",
            "polylogue.cli.query_verbs._render_recovery_read_view",
        ),
        surfaces=("cli", "read-view"),
        previewable=True,
        effects=("Pure",),
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
        name="mutate-add-tag",
        kind=OperationKind.MAINTENANCE,
        description="Add a tag to one session. Idempotent — returns unchanged when the tag is already present.",
        consumes=("session_metadata",),
        produces=("session_tags",),
        path_targets=("tag-mutation-loop",),
        code_refs=(
            "polylogue.storage.repository.archive.repository_writes.RepositoryWriteMixin.add_tag",
            "polylogue.api.archive.PolylogueArchiveMixin.add_tag",
            "polylogue.mcp.server_mutation_tools.add_tag",
        ),
        surfaces=("facade", "mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="mutate-remove-tag",
        kind=OperationKind.MAINTENANCE,
        description="Remove a tag from one session. Idempotent — returns not-found when the tag is absent.",
        consumes=("session_tags",),
        produces=("session_tags",),
        path_targets=("tag-mutation-loop",),
        code_refs=(
            "polylogue.api.archive.PolylogueArchiveMixin.remove_tag",
            "polylogue.mcp.server_mutation_tools.remove_tag",
        ),
        surfaces=("facade", "mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="mutate-bulk-tag-sessions",
        kind=OperationKind.MAINTENANCE,
        description="Apply tags to multiple sessions in one transaction. Returns affected and skipped counts.",
        consumes=("session_tags",),
        produces=("session_tags",),
        path_targets=("tag-mutation-loop",),
        code_refs=("polylogue.mcp.server_mutation_tools.bulk_tag_sessions",),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="mutate-set-metadata",
        kind=OperationKind.MAINTENANCE,
        description="Set a metadata key on one session. Idempotent — returns unchanged when the value matches.",
        consumes=("session_metadata",),
        produces=("session_metadata",),
        path_targets=("metadata-mutation-loop",),
        code_refs=(
            "polylogue.storage.repository.archive.repository_writes.RepositoryWriteMixin.update_metadata",
            "polylogue.api.archive.PolylogueArchiveMixin.update_metadata",
            "polylogue.mcp.server_mutation_tools.set_metadata",
        ),
        surfaces=("facade", "mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="mutate-delete-metadata",
        kind=OperationKind.MAINTENANCE,
        description="Delete a metadata key from one session. Idempotent — returns not-found when the key is absent.",
        consumes=("session_metadata",),
        produces=("session_metadata",),
        path_targets=("metadata-mutation-loop",),
        code_refs=("polylogue.mcp.server_mutation_tools.delete_metadata",),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="mutate-delete-session",
        kind=OperationKind.MAINTENANCE,
        description="Permanently delete one session and all associated data. Guarded by a confirm flag on all surfaces.",
        consumes=("archive_session_rows",),
        produces=("archive_deleted_session",),
        path_targets=("session-delete-loop",),
        code_refs=(
            "polylogue.storage.repository.archive.writes.sessions.delete_session_via_backend",
            "polylogue.api.archive.PolylogueArchiveMixin.delete_session",
            "polylogue.mcp.server_mutation_tools.delete_session",
        ),
        surfaces=("facade", "cli", "mcp", "daemon"),
        mutates_state=True,
        previewable=False,
        idempotent=True,
        effects=("DbRead", "DbWrite", "Destructive"),
    ),
    OperationSpec(
        name="project-archive-readiness",
        kind=OperationKind.PROJECTION,
        description="Project archive-wide readiness and debt semantics from message FTS and durable derived-model readiness.",
        consumes=("message_fts", "session_insight_readiness", "retrieval_band_readiness"),
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
        description="Measure the canonical FTS and SessionFilter query benchmark domain.",
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
        name="benchmark.reader.api",
        kind=OperationKind.BENCHMARK,
        description="Measure reader HTTP API list/get/facets/context-pack/cost-rollup latencies.",
        surfaces=("benchmark-campaign",),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="benchmark.daemon.convergence",
        kind=OperationKind.BENCHMARK,
        description="Measure daemon ingest convergence timing across synthetic scale tiers.",
        surfaces=("benchmark-campaign",),
        previewable=True,
        effects=("DbRead", "DbWrite"),
    ),
    OperationSpec(
        name="benchmark.transform.recovery-digest",
        kind=OperationKind.BENCHMARK,
        description="Measure deterministic recovery digest transform compilation over synthetic tool-heavy sessions.",
        surfaces=("benchmark-campaign",),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="benchmark.transform.recovery-report",
        kind=OperationKind.BENCHMARK,
        description="Measure deterministic recovery report rendering over synthetic tool-heavy sessions.",
        surfaces=("benchmark-campaign",),
        previewable=True,
        effects=("Pure",),
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
