"""Authored mutation campaigns shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.scenarios import NamedScenarioSource, ScenarioProjectionSourceKind


@dataclass(frozen=True, kw_only=True)
class MutationCampaign(NamedScenarioSource):
    paths_to_mutate: tuple[str, ...]
    tests: tuple[str, ...]
    notes: tuple[str, ...] = ()

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
        description="ConversationFilter semantics and summary/picker contracts",
        paths_to_mutate=("polylogue/lib/filter/filters.py",),
        tests=(
            "tests/unit/core/test_filters_schemas.py",
            "tests/unit/core/test_filters_props.py",
        ),
        notes=(
            "Targets the historical largest no-test blind spot.",
            "Timeout tail is expected in filter pipeline helpers.",
        ),
    ),
    "models": MutationCampaign(
        name="models",
        description="Message/Conversation semantic helpers and pairing logic",
        paths_to_mutate=("polylogue/lib/models.py",),
        tests=(
            "tests/unit/core/test_models.py",
            "tests/unit/core/test_message_laws.py",
            "tests/unit/core/test_conversation_semantics.py",
        ),
    ),
    "json": MutationCampaign(
        name="json",
        description="JSON serialization and parser laws",
        paths_to_mutate=("polylogue/lib/json.py",),
        tests=("tests/unit/core/test_json.py",),
    ),
    "hybrid": MutationCampaign(
        name="hybrid",
        description="Hybrid search fusion and ranked-conversation resolution",
        paths_to_mutate=("polylogue/storage/search_providers/hybrid.py",),
        tests=(
            "tests/unit/storage/test_hybrid.py",
            "tests/unit/storage/test_hybrid_laws.py",
        ),
    ),
    "fts5": MutationCampaign(
        name="fts5",
        description="FTS5 query escaping and conversation search semantics",
        paths_to_mutate=("polylogue/storage/search_providers/fts5.py",),
        tests=("tests/unit/storage/test_fts5.py",),
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
    ),
    "pipeline-services": MutationCampaign(
        name="pipeline-services",
        description="Acquire/validate/parse planning and stage contracts",
        paths_to_mutate=("polylogue/pipeline/services",),
        tests=(
            "tests/unit/pipeline/test_acquisition_streams.py",
            "tests/unit/pipeline/test_parsing_service.py",
            "tests/unit/pipeline/test_render_service.py",
            "tests/unit/pipeline/test_indexing.py",
            "tests/unit/pipeline/test_ingest_batch.py",
            "tests/unit/pipeline/test_stage_independence.py",
            "tests/unit/pipeline/test_resilience.py",
        ),
        notes=("Likely to need more helper-level laws to reduce timeout noise.",),
    ),
    "cli-query": MutationCampaign(
        name="cli-query",
        description="Query command planning, action routing, and summary output contracts",
        paths_to_mutate=(
            "polylogue/cli/query.py",
            "polylogue/archive/query/plan.py",
            "polylogue/cli/query_actions.py",
            "polylogue/cli/query_output.py",
        ),
        tests=(
            "tests/unit/cli/test_query_exec.py",
            "tests/unit/cli/test_query_exec_laws.py",
            "tests/unit/cli/test_query_fmt.py",
        ),
    ),
    "cli-run": MutationCampaign(
        name="cli-run",
        description="Run command execution, display, and watch contracts",
        paths_to_mutate=("polylogue/cli/commands/run.py",),
        tests=(
            "tests/unit/cli/test_run.py",
            "tests/unit/cli/test_run_int.py",
            "tests/unit/cli/test_run_laws.py",
        ),
        path_targets=("conversation-render-loop",),
        artifact_targets=("conversation_render_projection", "rendered_conversation_artifacts"),
        operation_targets=("render-conversations",),
        tags=("mutation", "run", "render"),
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
    ),
    "site-builder": MutationCampaign(
        name="site-builder",
        description="Static-site builder and CLI archive contracts",
        paths_to_mutate=("polylogue/site/builder.py",),
        tests=(
            "tests/integration/test_site.py",
            "tests/integration/test_site_laws.py",
        ),
        path_targets=("site-publication-loop",),
        artifact_targets=(
            "conversation_render_projection",
            "site_conversation_pages",
            "site_publication_manifest",
            "publication_records",
        ),
        operation_targets=("publish-site",),
        tags=("mutation", "site", "publication"),
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
            "tests/unit/sources/test_unified_semantic_laws.py",
            "tests/unit/sources/test_null_guard_properties.py",
            "tests/unit/sources/test_models.py",
            "tests/unit/sources/test_parsers_props.py",
            "tests/unit/sources/test_assembly.py",
        ),
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
            "tests/unit/sources/test_unified_semantic_laws.py",
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
        notes=("Broadest campaign here; best run after law-wave work lands.",),
    ),
}


__all__ = [
    "MUTATION_CAMPAIGNS",
    "MutationCampaign",
]
