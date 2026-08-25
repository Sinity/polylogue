"""Typed runtime operation metadata shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from typing import Literal

from polylogue.core.json import JSONDocument, JSONDocumentList, json_document
from polylogue.core.user_state_targets import TARGET_KIND_NAMES
from polylogue.operations.mutation_transaction import (
    IdempotencyPolicy,
    Surface,
    TargetAuthorityPolicy,
)

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

SafetyGuard = Literal["write_role_required", "confirmed_before_execute", "explicit_dry_run_evidence"]
"""Declared safety guard that a mutating operation's public surfaces must enforce."""

ExecutorStatus = Literal["executor-routed", "declared-not-routed", "typed-exemption"]
"""Whether a mutating operation uses ``OperationExecutor`` or carries typed debt."""


class OperationKind(str, Enum):
    """High-level class for a runtime operation."""

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
    """One named runtime operation and its control-plane policy."""

    name: str
    kind: OperationKind
    description: str
    surfaces: tuple[str, ...] = ()
    mutates_state: bool = False
    previewable: bool = False
    idempotent: bool = True
    effects: tuple[Effect, ...] = ()
    safety_guards: tuple[SafetyGuard, ...] = ()
    executor_status: ExecutorStatus | None = None
    """t46.9 AC1: required (non-``None``) whenever ``mutates_state`` is ``True``."""
    operation_version: int = 1
    capability_family: Literal["write", "judge", "maintenance"] = "write"
    allowed_surfaces: tuple[Surface, ...] = ()
    target_authority: tuple[TargetAuthorityPolicy, ...] = ()
    affected_tiers: tuple[str, ...] = ()
    idempotency: IdempotencyPolicy = "none"
    resumable: bool = False
    receipt_schema: str = "polylogue.mutation-receipt/v1"
    reconstructible: bool = False

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "name": self.name,
                "kind": self.kind.value,
                "description": self.description,
                "surfaces": list(self.surfaces),
                "mutates_state": self.mutates_state,
                "previewable": self.previewable,
                "idempotent": self.idempotent,
                "effects": list(self.effects),
                "safety_guards": list(self.safety_guards),
                "executor_status": self.executor_status,
                "operation_version": self.operation_version,
                "capability_family": self.capability_family,
                "allowed_surfaces": list(self.allowed_surfaces),
                "target_authority": [
                    {
                        "key": policy.key,
                        "target_kinds": list(policy.target_kinds),
                        "required_capabilities": list(policy.required_capabilities),
                        "destructive_class": policy.destructive_class,
                        "required_confirmation": policy.required_confirmation,
                        "allowed_durabilities": list(policy.allowed_durabilities),
                        "allowed_recovery": list(policy.allowed_recovery),
                    }
                    for policy in self.target_authority
                ],
                "affected_tiers": list(self.affected_tiers),
                "idempotency": self.idempotency,
                "resumable": self.resumable,
                "receipt_schema": self.receipt_schema,
                "reconstructible": self.reconstructible,
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
        surfaces=("daemon", "sources"),
        mutates_state=True,
        effects=("Network", "DbWrite", "LiveArchive"),
        executor_status="declared-not-routed",
    ),
    OperationSpec(
        name="plan-validation-backlog",
        kind=OperationKind.PLANNING,
        description="Select raw records that still require validation before normal parse planning.",
        surfaces=("daemon", "reparse"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="plan-parse-backlog",
        kind=OperationKind.PLANNING,
        description="Select raw records that are eligible for parse planning under ordinary or force-reparse rules.",
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
        surfaces=("daemon", "reprocess", "ingest"),
        mutates_state=True,
        effects=("DbRead", "DbWrite"),
        executor_status="declared-not-routed",
    ),
    OperationSpec(
        name="index-message-fts",
        kind=OperationKind.INDEXING,
        description="Build or repair lexical message FTS rows from persisted archive messages.",
        surfaces=("daemon", "doctor", "repair", "query"),
        mutates_state=True,
        effects=("DbRead", "DbWrite"),
        executor_status="declared-not-routed",
    ),
    OperationSpec(
        name="materialize-transcript-embeddings",
        kind=OperationKind.MATERIALIZATION,
        description="Build or refresh transcript embedding metadata, session status rows, and semantic vector entries from archive sessions.",
        surfaces=("daemon", "embed", "retrieval"),
        mutates_state=True,
        effects=("DbRead", "DbWrite"),
        executor_status="declared-not-routed",
    ),
    OperationSpec(
        name="query-sessions",
        kind=OperationKind.QUERY,
        description="Resolve session-level query and search results from archive retrieval plans.",
        surfaces=("query", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="materialize-session-insights",
        kind=OperationKind.MATERIALIZATION,
        description="Build durable session-insight rows and their trigger-maintained FTS projections from archive sessions.",
        surfaces=("daemon", "insights", "doctor", "repair"),
        mutates_state=True,
        effects=("DbRead", "DbWrite"),
        executor_status="declared-not-routed",
    ),
    OperationSpec(
        name="project-retrieval-band-readiness",
        kind=OperationKind.PROJECTION,
        description="Project transcript/evidence/inference/enrichment retrieval readiness from embeddings and durable read-model readiness.",
        surfaces=("embed", "doctor", "retrieval"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-embedding-status",
        kind=OperationKind.QUERY,
        description="Resolve operator-facing embedding coverage, freshness, and retrieval-band readiness status views.",
        surfaces=("daemon", "embed", "doctor", "retrieval"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="project-session-insight-readiness",
        kind=OperationKind.PROJECTION,
        description="Project readiness, debt, and stale-surface semantics from durable session-insight rows and FTS state.",
        surfaces=("insights", "doctor", "archive_debt", "repair"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-session-profiles",
        kind=OperationKind.QUERY,
        description="Resolve durable session-profile insights from profile rows and merged profile FTS.",
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-session-work-events",
        kind=OperationKind.QUERY,
        description="Resolve durable session work-event insights from work-event rows and work-event FTS.",
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-session-phases",
        kind=OperationKind.QUERY,
        description="Resolve durable session-phase insights from phase rows.",
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-threads",
        kind=OperationKind.QUERY,
        description="Resolve durable thread insights from thread rows and thread FTS.",
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-session-tag-rollups",
        kind=OperationKind.QUERY,
        description="Resolve durable session tag-rollup insights from aggregate tag rows.",
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-session-insight-status",
        kind=OperationKind.QUERY,
        description="Resolve projected session-insight status views from session-insight readiness state.",
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-archive-coverage",
        kind=OperationKind.QUERY,
        description="Resolve provider, day, or week archive coverage rollups from durable archive and session-profile rows.",
        surfaces=("insights", "facade", "mcp", "helpers"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-tool-usage",
        kind=OperationKind.QUERY,
        description="Resolve per-origin tool usage analytics from canonical archive actions.",
        surfaces=("insights", "facade", "mcp"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="query-archive-debt",
        kind=OperationKind.QUERY,
        description="Resolve unified archive debt rows from tier, convergence, embedding, FTS, and assertion readiness.",
        surfaces=("cli", "facade", "mcp", "daemon", "maintenance"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="compile-session-digest",
        kind=OperationKind.PROJECTION,
        description="Compile one archived session into the deterministic session digest artifact and evidence index.",
        surfaces=("cli", "report"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="render-session-report",
        kind=OperationKind.PROJECTION,
        description="Render deterministic session report presets from a compiled session digest.",
        surfaces=("cli", "mcp", "report"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="compile-inferred-corpus-specs",
        kind=OperationKind.PROJECTION,
        description="Compile inferred synthetic corpus specs from schema packages and cluster manifests.",
        surfaces=("schema", "verification-lab", "synthetic"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="compile-inferred-corpus-scenarios",
        kind=OperationKind.PROJECTION,
        description="Compile inferred corpus scenarios grouped from inferred corpus specs.",
        surfaces=("schema", "verification-lab", "synthetic"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="query-schema-catalog",
        kind=OperationKind.QUERY,
        description="Resolve schema package catalogs, manifests, and inferred corpus projections for schema list surfaces.",
        surfaces=("schema", "cli"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="query-schema-explanations",
        kind=OperationKind.QUERY,
        description="Resolve provider schema element explanations from versioned schema packages.",
        surfaces=("schema", "cli"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="mutate-add-tag",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Add a tag to one session. Idempotent — returns unchanged when the tag is already present. "
            "Routed through OperationExecutor/TagAddActuator (reversible class, role_only confirmation)."
        ),
        surfaces=("facade", "mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-remove-tag",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Remove a tag from one session. Idempotent — returns not-found when the tag is absent. "
            "Routed through OperationExecutor/TagRemoveActuator (reversible class, role_only confirmation)."
        ),
        surfaces=("facade", "mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-bulk-tag-sessions",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Apply tags to multiple sessions in one transaction. Returns affected and skipped counts. "
            "Routed through OperationExecutor/BulkTagActuator (reversible class, role_only confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-set-metadata",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Set a metadata key on one session. Idempotent — returns unchanged when the value matches. "
            "Routed through OperationExecutor/MetadataSetActuator (reversible class, role_only confirmation)."
        ),
        surfaces=("facade", "mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-delete-metadata",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Delete a metadata key from one session. Idempotent — returns not-found when the key is absent. "
            "Routed through OperationExecutor/MetadataDeleteActuator (reversible class, role_only confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-add-mark",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Add a mark (star/pin/archive) to a session, message, or block. Idempotent — returns unchanged "
            "when the mark is already present. Routed through OperationExecutor/MarkAddActuator (reversible "
            "class, role_only confirmation); the first MCP no-spec mutation family (t46.9 phase 2) to gain "
            "an OperationSpec and executor route."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-remove-mark",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Remove a mark from a session, message, or block. Idempotent — returns unchanged when the mark "
            "is absent. Routed through OperationExecutor/MarkRemoveActuator (reversible class, role_only "
            "confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-save-annotation",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Create or update an annotation on a session, message, or block. Every apply writes; "
            "created-vs-updated is a receipt detail, not an idempotency short-circuit. Routed through "
            "OperationExecutor/AnnotationSaveActuator (reversible class, role_only confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-delete-annotation",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Soft-delete an annotation (marks its assertion row deleted). Idempotent — returns not-found "
            "when already absent. Routed through OperationExecutor/AnnotationDeleteActuator (reversible "
            "class, role_only confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-blackboard-post",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Append one note to the persistent agent blackboard. Always inserts (a fresh note id "
            "is minted per call, so this is not idempotent). Reversible class -- there is no "
            "destructive counterpart today -- role_only confirmation. Routed through "
            "OperationExecutor/BlackboardPostActuator."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=False,
        effects=("DbWrite",),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-capture-assertion-candidate",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Capture one terminal assertion as a private, non-injected candidate. Resolution, idempotency, "
            "TTL, and the user-tier write are executed through OperationExecutor/"
            "CaptureAssertionCandidateActuator with role_only confirmation."
        ),
        surfaces=("facade", "cli", "mcp"),
        mutates_state=True,
        idempotent=False,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-import-annotation-batch",
        kind=OperationKind.IMPORT,
        description=(
            "Import a bounded JSONL annotation batch with durable schema, provenance, validation outcomes, and "
            "candidate assertions. Live reference validation stays in the import operation; its atomic user-tier "
            "write is routed through OperationExecutor/AnnotationBatchImportActuator with role_only confirmation."
        ),
        surfaces=("facade", "cli", "mcp"),
        mutates_state=True,
        idempotent=True,
        effects=("DbWrite",),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
        allowed_surfaces=("internal",),
        target_authority=(
            TargetAuthorityPolicy(
                key="annotation-import",
                target_kinds=("annotation-batch", "assertion"),
                required_capabilities=("archive.annotation.import_batch",),
                destructive_class="reversible",
                required_confirmation="role_only",
                allowed_durabilities=("durable",),
                allowed_recovery=("none",),
            ),
        ),
    ),
    OperationSpec(
        name="mutate-rebuild-index",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Rebuild the derived block-FTS index from persisted blocks. The operation is idempotent and "
            "routes its real ArchiveStore primitive through OperationExecutor/IndexRebuildActuator."
        ),
        surfaces=("facade", "mcp"),
        mutates_state=True,
        previewable=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
        allowed_surfaces=("api",),
        target_authority=(
            TargetAuthorityPolicy(
                key="index-rebuild",
                target_kinds=("source",),
                required_capabilities=("archive.rebuild_index",),
                destructive_class="maintenance",
                required_confirmation="role_only",
                allowed_durabilities=("derived",),
                allowed_recovery=("rebuild",),
            ),
        ),
    ),
    OperationSpec(
        name="mutate-update-index",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Reconcile the derived block-FTS index for the facade update route. The current storage "
            "primitive rebuilds the complete index, and OperationExecutor binds the caller scope before it runs."
        ),
        surfaces=("facade", "mcp"),
        mutates_state=True,
        previewable=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
        allowed_surfaces=("api",),
        target_authority=(
            TargetAuthorityPolicy(
                key="index-update",
                target_kinds=("source",),
                required_capabilities=("archive.update_index",),
                destructive_class="maintenance",
                required_confirmation="role_only",
                allowed_durabilities=("derived",),
                allowed_recovery=("rebuild",),
            ),
        ),
    ),
    OperationSpec(
        name="mutate-rebuild-insights",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Rebuild durable session-insight read models for the requested session set. The canonical "
            "materializer runs through OperationExecutor/InsightsRebuildActuator with a typed receipt."
        ),
        surfaces=("facade", "mcp"),
        mutates_state=True,
        previewable=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
        allowed_surfaces=("api",),
        target_authority=(
            TargetAuthorityPolicy(
                key="insights-rebuild",
                target_kinds=("session",),
                required_capabilities=("archive.rebuild_insights",),
                destructive_class="maintenance",
                required_confirmation="role_only",
                allowed_durabilities=("derived",),
                allowed_recovery=("rebuild",),
            ),
        ),
    ),
    OperationSpec(
        name="mutate-resolve-raw-authority-blocker",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Explicitly acknowledge current source/index evidence and reopen replanning for one "
            "unresolved raw-authority blocker (stale-plan or frontier-judgment). Routed through "
            "OperationExecutor/BlockerResolveActuator; requires --yes/confirm-flag. Frontier-judgment "
            "blockers additionally require their exact accepted assertion id and "
            "disposition=retain_canonical_authority, enforced by the primitive itself."
        ),
        surfaces=("cli",),
        mutates_state=True,
        previewable=True,
        idempotent=True,
        effects=("DbRead", "DbWrite", "Destructive"),
        safety_guards=("write_role_required", "confirmed_before_execute", "explicit_dry_run_evidence"),
        executor_status="executor-routed",
        allowed_surfaces=("cli",),
        target_authority=(
            TargetAuthorityPolicy(
                key="raw-authority-blocker",
                target_kinds=("raw-authority-blocker",),
                required_capabilities=("archive.raw_authority.resolve_blocker",),
                destructive_class="reset",
                required_confirmation="confirm_flag",
                allowed_durabilities=("durable",),
                allowed_recovery=("reconcile_required",),
            ),
        ),
    ),
    OperationSpec(
        name="mutate-reset-raw-authority-census",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Last-resort reset of poisoned raw-authority census bookkeeping. The route is dry-run first, "
            "requires an exact source-tier backup-attested plan and explicit offline operator-maintenance ownership "
            "with the daemon stopped, "
            "and never touches parser census, accepted raws, or blobs. Fresh applies run through OperationExecutor; "
            "only a durable already-authorized intent may resume receipt finalization offline."
        ),
        surfaces=("cli",),
        mutates_state=True,
        previewable=True,
        idempotent=True,
        effects=("DbRead", "DbWrite", "Destructive"),
        safety_guards=("write_role_required", "confirmed_before_execute", "explicit_dry_run_evidence"),
        executor_status="executor-routed",
        allowed_surfaces=("maintenance",),
        target_authority=(
            TargetAuthorityPolicy(
                key="raw-authority-recovery-source",
                target_kinds=("source",),
                required_capabilities=("archive.raw_authority_recovery",),
                destructive_class="reset",
                required_confirmation="confirm_flag",
                allowed_durabilities=("durable",),
                allowed_recovery=("none",),
            ),
        ),
    ),
    OperationSpec(
        name="mutate-prune-orphaned-index-revision-seeds",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Remove only active-index raw revision seed rows whose source raws are absent. The route binds "
            "the exact active generation, source snapshot, recoverable index backup, and stopped-daemon "
            "offline operator-maintenance ownership; it never performs a broad index reset. Fresh applies run through "
            "OperationExecutor; only a durable already-authorized intent may resume receipt finalization offline."
        ),
        surfaces=("cli",),
        mutates_state=True,
        previewable=True,
        idempotent=True,
        effects=("DbRead", "DbWrite", "Destructive"),
        safety_guards=("write_role_required", "confirmed_before_execute", "explicit_dry_run_evidence"),
        executor_status="executor-routed",
        allowed_surfaces=("maintenance",),
        target_authority=(
            TargetAuthorityPolicy(
                key="raw-authority-recovery-index",
                target_kinds=("index",),
                required_capabilities=("archive.raw_authority_recovery",),
                destructive_class="reset",
                required_confirmation="confirm_flag",
                allowed_durabilities=("derived",),
                allowed_recovery=("none",),
            ),
        ),
    ),
    OperationSpec(
        name="mutate-save-saved-view",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Create or update a named saved query view. Every apply writes; created-vs-updated is a "
            "receipt detail, not an idempotency short-circuit. Routed through OperationExecutor/"
            "SavedViewSaveActuator (reversible class, role_only confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-delete-saved-view",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Soft-delete a saved query view (marks its assertion row deleted). Idempotent — returns "
            "not-found when already absent. Routed through OperationExecutor/SavedViewDeleteActuator "
            "(reversible class, role_only confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-save-recall-pack",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Create or update a recall pack (a named, resolved set of session/message/annotation/mark "
            "references). Every apply writes. Routed through OperationExecutor/RecallPackSaveActuator "
            "(reversible class, role_only confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-delete-recall-pack",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Soft-delete a recall pack (marks its assertion row deleted). Idempotent — returns not-found "
            "when already absent. Routed through OperationExecutor/RecallPackDeleteActuator (reversible "
            "class, role_only confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-save-workspace",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Create or update a durable reader workspace (open targets, layout, active target). Every "
            "apply writes. Routed through OperationExecutor/WorkspaceSaveActuator (reversible class, "
            "role_only confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-delete-workspace",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Soft-delete a durable reader workspace (marks its assertion row deleted). Idempotent — "
            "returns not-found when already absent. Routed through OperationExecutor/"
            "WorkspaceDeleteActuator (reversible class, role_only confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-record-correction",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Insert or replace a typed learning correction for one session/kind pair. Every apply "
            "writes. Routed through OperationExecutor/CorrectionRecordActuator (reversible class, "
            "role_only confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-delete-correction",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Soft-delete one learning correction (marks its assertion row deleted). Idempotent — "
            "returns not-found when already absent. Routed through OperationExecutor/"
            "CorrectionDeleteActuator (reversible class, role_only confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-clear-corrections",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Soft-delete every learning correction for a session. The plan resolves the exact live "
            "set of correction kinds so a concurrent record-correction between preview and apply "
            "forces a replan. Routed through OperationExecutor/CorrectionsClearActuator (reversible "
            "class, role_only confirmation)."
        ),
        surfaces=("mcp", "api"),
        mutates_state=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required",),
        executor_status="executor-routed",
    ),
    OperationSpec(
        name="mutate-delete-session",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Permanently delete one session and all associated data. Routed through "
            "OperationExecutor/SessionDeleteActuator on every surface: PREPARE previews "
            "the exact target set, EXECUTE requires a confirm-flag-strength authorization "
            "bound to that plan's hash."
        ),
        surfaces=("facade", "cli", "mcp", "daemon"),
        mutates_state=True,
        # Previously declared False; the description said "confirm flag on all
        # surfaces" (implying a preview step) while this field said otherwise
        # -- a documented contradiction (t46.9 2026-07-18 note). Fixed: every
        # surface's delete route always runs OperationExecutor.prepare before
        # any mutation, and the CLI/dry-run path surfaces that preview.
        previewable=True,
        idempotent=True,
        effects=("DbRead", "DbWrite", "Destructive"),
        safety_guards=("write_role_required", "confirmed_before_execute", "explicit_dry_run_evidence"),
        executor_status="executor-routed",
        allowed_surfaces=("api", "cli"),
        target_authority=(
            TargetAuthorityPolicy(
                key="session-delete",
                target_kinds=("session",),
                required_capabilities=("archive.delete_session",),
                destructive_class="delete",
                required_confirmation="confirm_flag",
                allowed_durabilities=("derived",),
                allowed_recovery=("rebuild",),
            ),
        ),
    ),
    OperationSpec(
        name="mutate-session-excision",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Durable, cross-tier (source/index/embeddings/user) removal of one session that "
            "records a removed-hash marker so unmodified-source re-ingest cannot resurrect it. "
            "Routed through OperationExecutor/SessionExcisionActuator; requires --yes/confirm-flag."
        ),
        surfaces=("cli",),
        mutates_state=True,
        previewable=True,
        idempotent=True,
        effects=("DbRead", "DbWrite", "Destructive"),
        safety_guards=("write_role_required", "confirmed_before_execute", "explicit_dry_run_evidence"),
        executor_status="executor-routed",
        allowed_surfaces=("cli",),
        target_authority=(
            TargetAuthorityPolicy(
                key="session-excision",
                target_kinds=("session",),
                required_capabilities=("archive.excise_session",),
                destructive_class="excise",
                required_confirmation="confirm_flag",
                allowed_durabilities=("durable",),
                allowed_recovery=("none",),
            ),
        ),
    ),
    OperationSpec(
        name="mutate-session-lifecycle-request",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Create one durable mirror/primary session lifecycle-request outbox row through "
            "OperationExecutor so its intent, authorization, and receipt share audit continuity authority."
        ),
        surfaces=("cli",),
        mutates_state=True,
        previewable=True,
        idempotent=True,
        effects=("DbRead", "DbWrite", "Destructive"),
        safety_guards=("write_role_required", "confirmed_before_execute", "explicit_dry_run_evidence"),
        executor_status="executor-routed",
        allowed_surfaces=("cli",),
        affected_tiers=("user", "audit"),
        target_authority=(
            TargetAuthorityPolicy(
                key="session-lifecycle-request",
                target_kinds=("session",),
                required_capabilities=("archive.request_session_lifecycle",),
                destructive_class="additive",
                required_confirmation="confirm_flag",
                allowed_durabilities=("durable",),
                allowed_recovery=("retry_convergent",),
            ),
        ),
    ),
    OperationSpec(
        name="mutate-identity-reset",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Tombstone one or more sessions: durable user.db suppression plus rebuildable "
            "index.db row removal. Routed through OperationExecutor/IdentityResetActuator; "
            "requires --yes/confirm-flag. Distinct from excision: re-ingest of unmodified "
            "source content can still repopulate index.db rows, but the suppression hides them."
        ),
        surfaces=("cli",),
        mutates_state=True,
        previewable=True,
        idempotent=True,
        effects=("DbRead", "DbWrite", "Destructive"),
        safety_guards=("write_role_required", "confirmed_before_execute", "explicit_dry_run_evidence"),
        executor_status="executor-routed",
        allowed_surfaces=("cli",),
        target_authority=(
            TargetAuthorityPolicy(
                key="identity-reset",
                target_kinds=("session",),
                required_capabilities=("archive.identity_reset",),
                destructive_class="reset",
                required_confirmation="confirm_flag",
                allowed_durabilities=("durable",),
                allowed_recovery=("reconcile_required",),
            ),
        ),
    ),
    OperationSpec(
        name="mutate-maintenance-target-run",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Execute resumable maintenance targets through the CLI replay route or the MCP/daemon planner route. "
            "These routes retain operation ids, cursor checkpoints, failure isolation, and explicit confirmation, "
            "but do not yet run through OperationExecutor."
        ),
        surfaces=("cli", "mcp", "daemon"),
        mutates_state=True,
        previewable=True,
        idempotent=True,
        effects=("DbRead", "DbWrite"),
        safety_guards=("write_role_required", "confirmed_before_execute"),
        executor_status="declared-not-routed",
        resumable=True,
    ),
    OperationSpec(
        name="mutate-filesystem-reset",
        kind=OperationKind.MAINTENANCE,
        description=(
            "Delete selected archive databases, blob/assets/cache trees, or authentication state after an exact "
            "target preview and explicit confirmation. Session/source identity reset is a separate executor-routed operation."
        ),
        surfaces=("cli",),
        mutates_state=True,
        previewable=True,
        idempotent=True,
        effects=("FileWrite", "Destructive"),
        safety_guards=("confirmed_before_execute", "explicit_dry_run_evidence"),
        executor_status="declared-not-routed",
    ),
    OperationSpec(
        name="project-archive-readiness",
        kind=OperationKind.PROJECTION,
        description="Project archive-wide readiness and debt semantics from message FTS and durable derived-model readiness.",
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
        surfaces=("help", "cli"),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="cli.json-contract",
        kind=OperationKind.CLI,
        description="Exercise a machine-readable CLI JSON surface and verify its contract envelope.",
        surfaces=("doctor", "schema", "tags", "json-contract"),
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
        name="seed-demo-archive",
        kind=OperationKind.MATERIALIZATION,
        description="Build a deterministic demo archive from seeded fixtures through the public demo command.",
        surfaces=("cli", "tests", "validation-lane"),
        mutates_state=True,
        previewable=False,
        effects=("DbWrite",),
        executor_status="typed-exemption",
    ),
    OperationSpec(
        name="verify-demo-archive",
        kind=OperationKind.READINESSCHECK,
        description="Verify a seeded demo archive exposes the expected query, assertion, and session evidence.",
        surfaces=("cli", "tests", "validation-lane"),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="reader-visual-dom",
        kind=OperationKind.CLI,
        description="Exercise the daemon reader shell DOM through deterministic visual smoke tests.",
        surfaces=("daemon", "tests", "validation-lane"),
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
        description="Measure reader HTTP API list/get/facets/context-image/cost-rollup latencies.",
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
        name="benchmark.archive.backup-plan",
        kind=OperationKind.BENCHMARK,
        description="Measure archive backup-boundary planning without copying backup data.",
        surfaces=("benchmark-campaign",),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="benchmark.archive.blob-gc-dry-run",
        kind=OperationKind.BENCHMARK,
        description="Measure blob-GC candidate scanning through the dry-run path.",
        surfaces=("benchmark-campaign",),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="benchmark.archive.space-report",
        kind=OperationKind.BENCHMARK,
        description="Measure read-only archive SQLite space-report scans.",
        surfaces=("benchmark-campaign",),
        previewable=True,
        effects=("DbRead",),
    ),
    OperationSpec(
        name="benchmark.transform.session-digest",
        kind=OperationKind.BENCHMARK,
        description="Measure deterministic session digest transform compilation over synthetic tool-heavy sessions.",
        surfaces=("benchmark-campaign",),
        previewable=True,
        effects=("Pure",),
    ),
    OperationSpec(
        name="benchmark.transform.session-report",
        kind=OperationKind.BENCHMARK,
        description="Measure deterministic session report rendering over synthetic tool-heavy sessions.",
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


_USER_MUTATION_TARGET_KINDS = (
    "annotation",
    "assertion",
    "blackboard",
    "correction",
    "recall_pack",
    "saved_view",
    "workspace",
    *TARGET_KIND_NAMES,
)

_LEGACY_EXECUTOR_CAPABILITIES: dict[str, str] = {
    "mutate-add-tag": "archive.add_tag",
    "mutate-remove-tag": "archive.remove_tag",
    "mutate-bulk-tag-sessions": "archive.bulk_tag_sessions",
    "mutate-set-metadata": "archive.set_metadata",
    "mutate-delete-metadata": "archive.delete_metadata",
    "mutate-add-mark": "archive.add_mark",
    "mutate-remove-mark": "archive.remove_mark",
    "mutate-save-annotation": "archive.save_annotation",
    "mutate-delete-annotation": "archive.delete_annotation",
    "mutate-blackboard-post": "archive.post_blackboard_note",
    "mutate-capture-assertion-candidate": "archive.capture_assertion_candidate",
    "mutate-save-saved-view": "archive.save_view",
    "mutate-delete-saved-view": "archive.delete_view",
    "mutate-save-recall-pack": "archive.create_recall_pack",
    "mutate-delete-recall-pack": "archive.delete_recall_pack",
    "mutate-save-workspace": "archive.save_workspace",
    "mutate-delete-workspace": "archive.delete_workspace",
    "mutate-record-correction": "archive.record_correction",
    "mutate-delete-correction": "archive.delete_correction",
    "mutate-clear-corrections": "archive.clear_corrections",
}


def _declare_executor_authority(specs: tuple[OperationSpec, ...]) -> tuple[OperationSpec, ...]:
    """Give every executor route a specific capability and surface boundary."""

    declared: list[OperationSpec] = []
    for spec in specs:
        if spec.executor_status != "executor-routed" or spec.target_authority:
            declared.append(spec)
            continue
        capability = _LEGACY_EXECUTOR_CAPABILITIES.get(spec.name)
        if capability is None:
            raise ValueError(f"executor-routed operation lacks target authority: {spec.name}")
        declared.append(
            replace(
                spec,
                allowed_surfaces=("api",),
                target_authority=(
                    TargetAuthorityPolicy(
                        key=spec.name.removeprefix("mutate-"),
                        target_kinds=_USER_MUTATION_TARGET_KINDS,
                        required_capabilities=(capability,),
                        destructive_class="reversible",
                        required_confirmation="role_only",
                        allowed_durabilities=("durable",),
                        allowed_recovery=("none",),
                    ),
                ),
            )
        )
    return tuple(declared)


RUNTIME_OPERATION_SPECS = _declare_executor_authority(RUNTIME_OPERATION_SPECS)
DECLARED_OPERATION_SPECS = (
    *RUNTIME_OPERATION_SPECS,
    *DECLARED_CONTROL_PLANE_OPERATION_SPECS,
)


def _validate_executor_status() -> None:
    """t46.9 AC1: every mutating spec must declare an executor_status.

    A ``mutates_state=True`` spec with ``executor_status=None`` would be
    silent absence -- an operation the catalog knows mutates state but says
    nothing about whether it is executor-routed, still debt, or exempt.
    Phase 2 migration work narrows the ``declared-not-routed`` set; this
    check only guarantees the debt stays visible, never silent.
    """
    missing = [spec.name for spec in DECLARED_OPERATION_SPECS if spec.mutates_state and spec.executor_status is None]
    if missing:
        raise ValueError(
            "OperationSpec entries with mutates_state=True must declare executor_status "
            f"(missing: {', '.join(sorted(missing))})"
        )


_validate_executor_status()


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
    "ExecutorStatus",
    "build_declared_operation_catalog",
    "build_runtime_operation_catalog",
    "OperationCatalog",
    "OperationKind",
    "OperationSpec",
    "RUNTIME_OPERATION_SPECS",
    "SafetyGuard",
]
