"""Authored mutation campaigns shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.scenarios import NamedScenarioSource, ScenarioProjectionSourceKind


@dataclass(frozen=True, kw_only=True)
class MutationCampaign(NamedScenarioSource):
    paths_to_mutate: tuple[str, ...]
    tests: tuple[str, ...]
    notes: tuple[str, ...] = ()
    # A policy belongs to the obligation-owned campaign, not to the freshness
    # verifier. Catalog entries must declare a risk-shaped floor; ``None`` is
    # retained in the type so the catalog invariant can catch omissions.
    min_kill_rate: float | None = None
    consequence: str = "standard"
    budget_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.origin == "authored":
            object.__setattr__(self, "origin", "authored.mutation-campaign")
        if not self.tags:
            object.__setattr__(self, "tags", ("mutation",))

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.MUTATION_CAMPAIGN


MUTATION_CAMPAIGNS: dict[str, MutationCampaign] = {
    "filters": MutationCampaign(
        name="filters",
        description="SessionFilter semantics and summary/picker contracts",
        paths_to_mutate=("polylogue/archive/filter/filters.py",),
        tests=("tests/unit/core/test_filters_props.py",),
        min_kill_rate=0.6,
        consequence="query-filter-semantics",
        notes=(
            "Targets the historical largest no-test blind spot.",
            "Timeout tail is expected in filter pipeline helpers.",
        ),
    ),
    "models": MutationCampaign(
        name="models",
        description="Message/Session semantic helpers and pairing logic",
        paths_to_mutate=("polylogue/archive/models.py",),
        tests=(
            "tests/unit/core/test_models.py",
            "tests/unit/core/test_message_laws.py",
            "tests/unit/core/test_session_semantics.py",
        ),
        min_kill_rate=0.7,
        consequence="message-session-semantics",
    ),
    "json": MutationCampaign(
        name="json",
        description="JSON serialization and parser laws",
        paths_to_mutate=("polylogue/core/json.py",),
        tests=("tests/unit/core/test_json.py",),
        min_kill_rate=0.7,
        consequence="serialization-integrity",
    ),
    "hybrid": MutationCampaign(
        name="hybrid",
        description=(
            "Reciprocal Rank Fusion — the shared ranking primitive production hybrid "
            "retrieval composes directly (polylogue-a7xr.10 removed the unproven "
            "FTS5Provider/HybridSearchProvider classes and their fts5/hybrid campaigns; "
            "only reciprocal_rank_fusion in hybrid.py remains)."
        ),
        paths_to_mutate=("polylogue/storage/search_providers/hybrid.py",),
        tests=(
            "tests/unit/core/test_filters_props.py",
            "tests/unit/archive/test_query_search_runtime.py",
        ),
        min_kill_rate=0.7,
        consequence="retrieval-ranking",
    ),
    "schema-core": MutationCampaign(
        name="schema-core",
        description="Schema generation, privacy, verification, and safety contracts",
        paths_to_mutate=(
            "polylogue/schemas/operator/schema_inference.py",
            "polylogue/schemas/validator.py",
            "polylogue/schemas/operator/verification.py",
        ),
        tests=(
            "tests/unit/core/test_schema_validation.py",
            "tests/unit/core/test_schema_generation.py",
            "tests/unit/core/test_schema_annotation_contracts.py",
            "tests/unit/core/test_schema_laws.py",
            "tests/unit/core/test_schema_privacy.py",
            "tests/unit/core/test_verification.py",
            "tests/unit/storage/test_schema_safety.py",
        ),
        min_kill_rate=0.8,
        consequence="schema-contract",
        notes=("Larger campaign; use when law and privacy work are stable.",),
    ),
    "schema-inference": MutationCampaign(
        name="schema-inference",
        description="Schema inference and privacy heuristics",
        paths_to_mutate=("polylogue/schemas/operator/schema_inference.py",),
        tests=(
            "tests/unit/core/test_schema_generation.py",
            "tests/unit/core/test_schema_laws.py",
            "tests/unit/core/test_schema_privacy.py",
        ),
        min_kill_rate=0.75,
        consequence="schema-inference",
    ),
    "schema-validation": MutationCampaign(
        name="schema-validation",
        description="Schema validator and verification contracts",
        paths_to_mutate=(
            "polylogue/schemas/validator.py",
            "polylogue/schemas/operator/verification.py",
        ),
        tests=(
            "tests/unit/core/test_schema_validation.py",
            "tests/unit/core/test_schema_laws.py",
            "tests/unit/core/test_verification.py",
            "tests/unit/storage/test_schema_safety.py",
        ),
        min_kill_rate=0.8,
        consequence="schema-validation",
    ),
    "pipeline-services": MutationCampaign(
        name="pipeline-services",
        description="Acquire/validate/parse planning and stage contracts",
        paths_to_mutate=("polylogue/pipeline/services",),
        tests=(
            "tests/unit/pipeline/test_acquisition_streams.py",
            "tests/unit/pipeline/test_parsing_service.py",
            "tests/unit/pipeline/test_indexing.py",
            "tests/unit/pipeline/test_ingest_batch.py",
            "tests/unit/pipeline/test_stage_independence.py",
            "tests/unit/pipeline/test_resilience.py",
        ),
        min_kill_rate=0.75,
        consequence="ingest-pipeline",
        notes=("Likely to need more helper-level laws to reduce timeout noise.",),
    ),
    "cli-query": MutationCampaign(
        name="cli-query",
        description="Query command planning, action routing, and summary output contracts",
        paths_to_mutate=(
            "polylogue/cli/query.py",
            "polylogue/archive/query/plan.py",
            "polylogue/cli/query_output.py",
        ),
        tests=(
            "tests/unit/cli/test_query_verbs_runtime.py",
            "tests/unit/cli/test_query_exec_laws.py",
            "tests/unit/cli/test_query_fmt.py",
        ),
        min_kill_rate=0.7,
        consequence="query-routing",
    ),
    "ui-core": MutationCampaign(
        name="ui-core",
        description="UI prompt, progress, and facade interaction contracts",
        paths_to_mutate=(
            "polylogue/ui/__init__.py",
            "polylogue/ui/facade.py",
        ),
        tests=(
            "tests/unit/ui/test_ui.py",
            "tests/unit/ui/test_ui_visual.py",
            "tests/unit/ui/test_tui.py",
        ),
        min_kill_rate=0.6,
        consequence="user-interaction",
    ),
    "drive-client": MutationCampaign(
        name="drive-client",
        description="Drive auth, transport, JSON payload parsing, and ingest attachment contracts",
        paths_to_mutate=(
            "polylogue/sources/drive/source.py",
            "polylogue/sources/drive/gateway.py",
            "polylogue/sources/drive/auth.py",
            "polylogue/sources/drive/__init__.py",
        ),
        tests=(
            "tests/unit/sources/test_drive_source_client.py",
            "tests/unit/sources/test_drive_gateway.py",
            "tests/unit/sources/test_drive_auth.py",
            "tests/unit/sources/test_drive_ops.py",
        ),
        min_kill_rate=0.7,
        consequence="external-ingest",
        notes=("Targets the historical Drive not_checked cluster with focused tests.",),
    ),
    "repository": MutationCampaign(
        name="repository",
        description="Repository query, projection, and CRUD contracts",
        paths_to_mutate=("polylogue/storage/repository/__init__.py",),
        tests=(
            "tests/unit/storage/test_store_ops.py",
            "tests/unit/storage/test_tree_laws.py",
        ),
        min_kill_rate=0.75,
        consequence="storage-crud",
        notes=("Large surface; use to gauge storage law readiness before repository-law work.",),
    ),
    "source-detection": MutationCampaign(
        name="source-detection",
        description="Source detection, sniffing, and parser dispatch",
        paths_to_mutate=(
            "polylogue/sources/source_parsing.py",
            "polylogue/sources/source_acquisition.py",
            "polylogue/sources/dispatch.py",
            "polylogue/sources/decoders.py",
        ),
        tests=(
            "tests/unit/sources/test_source_laws.py",
            "tests/unit/sources/test_parsers_base.py",
            "tests/unit/sources/test_parsers_chatgpt.py",
            "tests/unit/sources/test_parsers_codex.py",
            "tests/unit/sources/test_parsers_props.py",
            "tests/unit/sources/test_parsers_drive.py",
        ),
        min_kill_rate=0.8,
        consequence="source-dispatch",
    ),
    "provider-parsers": MutationCampaign(
        name="provider-parsers",
        description="Provider parser semantic correctness — where message extraction and compaction detection live",
        paths_to_mutate=(
            "polylogue/sources/parsers/chatgpt.py",
            "polylogue/sources/parsers/claude/code_parser.py",
            "polylogue/sources/parsers/codex.py",
            "polylogue/sources/parsers/claude/index.py",
            "polylogue/pipeline/semantic_capture.py",
        ),
        tests=(
            "tests/unit/sources/test_parsers_chatgpt.py",
            "tests/unit/sources/test_parsers_codex.py",
            "tests/unit/sources/test_parsers_props.py",
            "tests/unit/sources/test_parser_crashlessness.py",
            "tests/unit/sources/test_compaction.py",
            "tests/unit/sources/test_assembly.py",
        ),
        min_kill_rate=0.8,
        consequence="provider-parsing",
        notes=("Focused on the parser modules where semantic correctness is most critical.",),
    ),
    "providers-semantics": MutationCampaign(
        name="providers-semantics",
        description="Provider semantic extraction, harmonization, and viewport contracts",
        paths_to_mutate=(
            "polylogue/sources/providers",
            "polylogue/schemas/registry.py",
        ),
        tests=(
            "tests/unit/sources/test_null_guard_properties.py",
            "tests/unit/sources/test_models.py",
            "tests/unit/sources/test_parsers_props.py",
            "tests/unit/sources/test_assembly.py",
        ),
        min_kill_rate=0.8,
        consequence="provider-semantics",
        notes=("Directly relevant to the next provider-law wave.",),
    ),
    "sources-parse": MutationCampaign(
        name="sources-parse",
        description="Provider detection, parsing, harmonization, and parser laws",
        paths_to_mutate=(
            "polylogue/sources",
            "polylogue/schemas/registry.py",
        ),
        tests=(
            "tests/unit/sources/test_parsers_props.py",
            "tests/unit/sources/test_source_laws.py",
            "tests/unit/sources/test_parsers_base.py",
            "tests/unit/sources/test_parsers_chatgpt.py",
            "tests/unit/sources/test_parsers_codex.py",
            "tests/unit/sources/test_parsers_drive.py",
            "tests/unit/sources/test_drive_source_client.py",
            "tests/unit/sources/test_drive_gateway.py",
            "tests/unit/sources/test_drive_auth.py",
            "tests/unit/sources/test_drive_ops.py",
            "tests/unit/sources/test_null_guard_properties.py",
            "tests/unit/sources/test_models.py",
            "tests/unit/sources/test_token_store.py",
        ),
        min_kill_rate=0.8,
        consequence="source-parsing",
        notes=("Broadest campaign here; best run after law-wave work lands.",),
    ),
    "daemon-http": MutationCampaign(
        name="daemon-http",
        description="Daemon HTTP API endpoint handler contracts",
        paths_to_mutate=("polylogue/daemon/http.py",),
        tests=("tests/unit/daemon/test_daemon_http.py",),
        min_kill_rate=0.75,
        consequence="http-api",
        notes=(
            "CI-compatible subset; run with fast timeout tiers.",
            "HTTP endpoint handlers should remain deterministic under test fixtures.",
        ),
    ),
    "repair-core": MutationCampaign(
        name="repair-core",
        description="Storage repair logic, preview/idempotence/failure state effects",
        paths_to_mutate=("polylogue/storage/repair.py",),
        tests=("tests/unit/storage/test_repair.py",),
        min_kill_rate=0.8,
        consequence="repair-safety",
        notes=(
            "Repair is a maintenance-critical path; preview/dry-run/execute contracts must hold.",
            "Keep timeout generous - repair tests may perform full FTS5 rebuilds.",
        ),
    ),
    "blob-liveness-delete": MutationCampaign(
        name="blob-liveness-delete",
        description="Blob liveness, durable GC intent, and irreversible deletion safety",
        paths_to_mutate=("polylogue/storage/blob_gc.py", "polylogue/storage/blob_liveness.py"),
        tests=(
            "tests/unit/storage/test_blob_gc.py",
            "tests/unit/storage/test_blob_gc_durable_intent.py",
            "tests/unit/storage/test_blob_liveness.py",
        ),
        min_kill_rate=0.8,
        consequence="irreversible-delete",
        budget_seconds=900,
    ),
    "cursor-publication": MutationCampaign(
        name="cursor-publication",
        description="Cursor publication only after the owning durable work commits",
        paths_to_mutate=("polylogue/sources/live/cursor.py", "polylogue/sources/live/batch.py"),
        tests=(
            "tests/unit/sources/test_live_cursor_locking.py",
            "tests/unit/sources/test_live_cursor_persistence.py",
            "tests/unit/sources/test_cursor_lifecycle.py",
        ),
        min_kill_rate=0.75,
        consequence="durable-cursor",
        budget_seconds=900,
    ),
    "lineage-identity-publication": MutationCampaign(
        name="lineage-identity-publication",
        description="Lineage and identity publication through the parsed-session write seam",
        paths_to_mutate=("polylogue/storage/sqlite/archive_tiers/write.py",),
        tests=(
            "tests/unit/storage/test_write_path_state_machine.py",
            "tests/unit/storage/test_archive_identity.py",
            "tests/unit/sources/test_revision_backfill.py",
        ),
        min_kill_rate=0.75,
        consequence="identity-lineage",
        budget_seconds=900,
    ),
    "durable-transition": MutationCampaign(
        name="durable-transition",
        description="Durable source-item and change-train transitions under interruption",
        paths_to_mutate=(
            "polylogue/storage/sqlite/archive_tiers/source_items.py",
            "polylogue/storage/sqlite/durable_change_train.py",
        ),
        tests=(
            "tests/unit/storage/test_source_items.py",
            "tests/unit/storage/test_durable_change_train.py",
            "tests/unit/storage/test_write_path_state_machine.py",
        ),
        min_kill_rate=0.75,
        consequence="durable-transition",
        budget_seconds=900,
    ),
    "authored-cost-accounting": MutationCampaign(
        name="authored-cost-accounting",
        description="Authoredness-preserving usage and cost aggregation",
        paths_to_mutate=("polylogue/storage/usage.py", "polylogue/storage/sqlite/archive_tiers/write.py"),
        tests=(
            "tests/unit/storage/test_session_usage_reconciliation.py",
            "tests/unit/storage/test_session_profile_model_usage_consistency.py",
            "tests/unit/storage/test_archive_tiers_write.py",
        ),
        min_kill_rate=0.8,
        consequence="accounting",
        budget_seconds=900,
    ),
}


__all__ = [
    "MUTATION_CAMPAIGNS",
    "MutationCampaign",
]
