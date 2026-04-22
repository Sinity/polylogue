"""Generated-world scenario families tracked by the proof catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from polylogue.lib.json import JSONDocument, require_json_value
from polylogue.proof.models import SourceSpan, SubjectRef

ScenarioFamilyStatus = Literal["implemented", "migration_task"]
SemanticClaimState = Literal["implemented", "mapped"]


@dataclass(frozen=True, slots=True)
class SemanticClaimMapping:
    """Semantic claim family protected or explicitly mapped by a generated scenario."""

    family: str
    state: SemanticClaimState
    claim_id: str | None = None
    issue: str | None = None

    def to_payload(self) -> JSONDocument:
        return _json_document(
            {
                "family": self.family,
                "state": self.state,
                "claim_id": self.claim_id,
                "issue": self.issue,
            }
        )


@dataclass(frozen=True, slots=True)
class GeneratedScenarioFamily:
    """One generated-world or generated-workload family."""

    name: str
    description: str
    status: ScenarioFamilyStatus
    generated_world: str
    workload_family: str
    local_deterministic: bool
    live_archive_dependency: bool
    reproducer: tuple[str, ...]
    semantic_claims: tuple[SemanticClaimMapping, ...]
    tags: tuple[str, ...]
    operation_targets: tuple[str, ...] = ()
    artifact_targets: tuple[str, ...] = ()
    issue: str = "#196"
    source_span: SourceSpan = SourceSpan(path="polylogue/proof/generated_scenarios.py")

    def to_subject(self) -> SubjectRef:
        return SubjectRef(
            kind="generated.scenario_family",
            id=f"generated.scenario_family.{self.name}",
            attrs=_json_document(
                {
                    "name": self.name,
                    "description": self.description,
                    "status": self.status,
                    "generated_world": self.generated_world,
                    "workload_family": self.workload_family,
                    "local_deterministic": self.local_deterministic,
                    "live_archive_dependency": self.live_archive_dependency,
                    "reproducer": list(self.reproducer),
                    "semantic_claims": [claim.to_payload() for claim in self.semantic_claims],
                    "tags": list(self.tags),
                    "operation_targets": list(self.operation_targets),
                    "artifact_targets": list(self.artifact_targets),
                    "issue": self.issue,
                }
            ),
            source_span=self.source_span,
        )


def _claim(
    family: str,
    state: SemanticClaimState,
    *,
    claim_id: str | None = None,
    issue: str | None = None,
) -> SemanticClaimMapping:
    return SemanticClaimMapping(family=family, state=state, claim_id=claim_id, issue=issue)


GENERATED_SCENARIO_FAMILIES: tuple[GeneratedScenarioFamily, ...] = (
    GeneratedScenarioFamily(
        name="cold-doctor-check",
        description="Cold doctor/check on a fresh generated archive state.",
        status="migration_task",
        generated_world="fresh generated archive workspace",
        workload_family="doctor/check startup readiness",
        local_deterministic=False,
        live_archive_dependency=False,
        reproducer=("devtools", "pipeline-probe", "--stage", "all", "--provider", "chatgpt", "--count", "1"),
        semantic_claims=(
            _claim("archive absence/degraded readiness", "mapped", issue="#217"),
            _claim("cold readiness diagnostics", "mapped", issue="#329"),
        ),
        operation_targets=("project-archive-readiness",),
        artifact_targets=("archive_readiness", "raw_validation_state"),
        tags=("generated", "cold", "readiness", "migration-task"),
    ),
    GeneratedScenarioFamily(
        name="cold-stats-search",
        description="Cold stats and search on a generated archive state.",
        status="migration_task",
        generated_world="small generated archive with indexed messages",
        workload_family="stats/search startup query",
        local_deterministic=False,
        live_archive_dependency=False,
        reproducer=("devtools", "pipeline-probe", "--stage", "all", "--provider", "codex", "--count", "3"),
        semantic_claims=(
            _claim(
                "filter algebra and count/list/search agreement",
                "implemented",
                claim_id="archive.query.provider_filter_consistency",
            ),
            _claim("slow-query diagnostics", "mapped", issue="#219"),
        ),
        operation_targets=("query-conversations",),
        artifact_targets=("conversation_query_results", "message_fts"),
        tags=("generated", "cold", "query", "migration-task"),
    ),
    GeneratedScenarioFamily(
        name="large-generated-search",
        description="Large generated archive search across FTS and hybrid paths.",
        status="migration_task",
        generated_world="large generated archive scale profile",
        workload_family="FTS and hybrid search",
        local_deterministic=False,
        live_archive_dependency=False,
        reproducer=("devtools", "benchmark-campaign", "run", "filter-scan"),
        semantic_claims=(
            _claim("comparative/growth-shape performance evidence", "mapped", issue="#195"),
            _claim("retrieval evidence provenance", "mapped", issue="#216"),
        ),
        operation_targets=("query-conversations", "query.filters.synthetic-scan"),
        artifact_targets=("conversation_query_results", "message_fts"),
        tags=("generated", "large", "search", "performance", "migration-task"),
    ),
    GeneratedScenarioFamily(
        name="pathological-raw-rerun",
        description="Pathological raw parse rerun for malformed JSONL, empty input, BOM, and giant records.",
        status="implemented",
        generated_world="repo-local malformed raw payload fixtures",
        workload_family="raw validation and parser quarantine",
        local_deterministic=True,
        live_archive_dependency=False,
        reproducer=(
            "pytest",
            "tests/unit/pipeline/test_quarantine_fixtures.py",
            "tests/unit/pipeline/test_ingestion_chaos.py",
        ),
        semantic_claims=(
            _claim(
                "parser crashlessness and explicit quarantine semantics",
                "implemented",
                claim_id="parser.quarantine.context_redaction",
            ),
            _claim("wrong-provider rejection without silent adoption", "mapped", issue="#333"),
        ),
        operation_targets=("acquire-raw-conversations",),
        artifact_targets=("raw_validation_state", "source_payload_stream"),
        tags=("generated", "raw", "pathological", "implemented"),
        source_span=SourceSpan(
            path="tests/unit/pipeline/test_quarantine_fixtures.py", symbol="test_validate_raw_ids_marks_invalid_records"
        ),
    ),
    GeneratedScenarioFamily(
        name="giant-grouped-jsonl-ingest",
        description="Giant grouped JSONL ingest with explicit fanout and batching expectations.",
        status="migration_task",
        generated_world="large grouped JSONL fixture",
        workload_family="batch ingest fanout",
        local_deterministic=False,
        live_archive_dependency=False,
        reproducer=("devtools", "pipeline-probe", "--stage", "all", "--provider", "codex", "--raw-batch-size", "10"),
        semantic_claims=(
            _claim("parser crashlessness and explicit quarantine semantics", "mapped", issue="#333"),
            _claim("pipeline batch fanout accounting", "mapped", issue="#323"),
        ),
        operation_targets=("acquire-raw-conversations",),
        artifact_targets=("raw_rows", "conversation_rows", "message_rows"),
        tags=("generated", "raw", "batching", "migration-task"),
    ),
    GeneratedScenarioFamily(
        name="repair-convergence-broken-generated",
        description="Repair convergence from known-broken generated archive states.",
        status="implemented",
        generated_world="seeded archive with orphaned message rows",
        workload_family="maintenance repair convergence",
        local_deterministic=True,
        live_archive_dependency=False,
        reproducer=("pytest", "tests/unit/proof/test_structural_error_evidence.py"),
        semantic_claims=(
            _claim(
                "repair convergence and explicit state effects",
                "implemented",
                claim_id="maintenance.repair.crash_consistency",
            ),
            _claim("destructive-preview safety", "implemented", claim_id="maintenance.repair.crash_consistency"),
        ),
        operation_targets=("repair-archive-state",),
        artifact_targets=("archive_conversation_rows", "archive_message_rows"),
        tags=("generated", "repair", "implemented"),
        source_span=SourceSpan(
            path="tests/unit/proof/test_structural_error_evidence.py",
            symbol="test_maintenance_repair_runner_exercises_synthetic_archive_transition",
        ),
    ),
    GeneratedScenarioFamily(
        name="action-event-rebuild-convergence",
        description="Action-event rebuild convergence over generated tool-use transcripts.",
        status="migration_task",
        generated_world="synthetic tool-use transcript archive",
        workload_family="action-event materialization",
        local_deterministic=False,
        live_archive_dependency=False,
        reproducer=("devtools", "benchmark-campaign", "run", "action-event-materialization"),
        semantic_claims=(
            _claim("action-event rebuild convergence", "mapped", issue="#196"),
            _claim("session product rebuild convergence", "mapped", issue="#322"),
        ),
        operation_targets=("materialize-action-events",),
        artifact_targets=("tool_use_source_blocks", "action_event_rows", "action_event_fts"),
        tags=("generated", "action-events", "migration-task"),
    ),
    GeneratedScenarioFamily(
        name="destructive-preview-safety",
        description="Destructive-preview safety against a deterministic generated broken archive state.",
        status="implemented",
        generated_world="seeded archive with previewable orphan cleanup",
        workload_family="dry-run versus destructive repair",
        local_deterministic=True,
        live_archive_dependency=False,
        reproducer=("pytest", "tests/unit/proof/test_structural_error_evidence.py"),
        semantic_claims=(
            _claim("destructive-preview safety", "implemented", claim_id="maintenance.repair.crash_consistency"),
            _claim("error machine/user context", "implemented", claim_id="error.machine_user_context"),
        ),
        operation_targets=("repair-archive-state",),
        artifact_targets=("archive_message_rows",),
        tags=("generated", "destructive-preview", "implemented"),
        source_span=SourceSpan(
            path="tests/unit/proof/test_structural_error_evidence.py",
            symbol="test_maintenance_repair_runner_exercises_synthetic_archive_transition",
        ),
    ),
    GeneratedScenarioFamily(
        name="archive-substrate-laws",
        description="Archive substrate laws for content hash stability, idempotent upsert, and NFC normalization.",
        status="implemented",
        generated_world="unit-generated conversation and text fixtures",
        workload_family="archive substrate laws",
        local_deterministic=True,
        live_archive_dependency=False,
        reproducer=("pytest", "tests/unit/core/test_properties.py", "tests/unit/storage/test_crud.py"),
        semantic_claims=(
            _claim("content hash stability", "implemented", issue="#196"),
            _claim("idempotent upsert and NFC normalization", "implemented", issue="#196"),
        ),
        operation_targets=("store-conversation",),
        artifact_targets=("archive_conversation_rows", "archive_message_rows"),
        tags=("generated", "substrate-law", "implemented"),
        source_span=SourceSpan(path="tests/unit/core/test_properties.py", symbol="test_content_hash_stable"),
    ),
)


def generated_scenario_subjects() -> tuple[SubjectRef, ...]:
    """Compile generated-world/workload families into proof subjects."""
    return tuple(family.to_subject() for family in GENERATED_SCENARIO_FAMILIES)


def _json_document(items: dict[str, object]) -> JSONDocument:
    return {key: require_json_value(value, context=key) for key, value in items.items()}


__all__ = [
    "GeneratedScenarioFamily",
    "GENERATED_SCENARIO_FAMILIES",
    "SemanticClaimMapping",
    "generated_scenario_subjects",
]
