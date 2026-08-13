from __future__ import annotations

from polylogue.core.user_state_targets import TARGET_KIND_NAMES
from polylogue.operations import (
    OperationKind,
    build_declared_operation_catalog,
    build_runtime_operation_catalog,
)


def test_runtime_operation_catalog_covers_the_current_runtime_paths() -> None:
    specs = build_runtime_operation_catalog().by_name()

    assert set(specs) == {
        "acquire-raw-sessions",
        "plan-validation-backlog",
        "plan-parse-backlog",
        "ingest-archive-runtime",
        "index-message-fts",
        "materialize-transcript-embeddings",
        "query-sessions",
        "materialize-session-insights",
        "project-retrieval-band-readiness",
        "query-embedding-status",
        "project-session-insight-readiness",
        "project-archive-readiness",
        "query-session-profiles",
        "query-session-work-events",
        "query-session-phases",
        "query-threads",
        "query-session-tag-rollups",
        "query-archive-coverage",
        "query-tool-usage",
        "query-session-insight-status",
        "query-archive-debt",
        "compile-session-digest",
        "render-session-report",
        "compile-inferred-corpus-specs",
        "compile-inferred-corpus-scenarios",
        "query-schema-catalog",
        "query-schema-explanations",
        # Mutation operations exposed via MCP and CLI write surfaces.
        "mutate-add-tag",
        "mutate-remove-tag",
        "mutate-set-metadata",
        "mutate-delete-metadata",
        "mutate-add-mark",
        "mutate-remove-mark",
        "mutate-save-annotation",
        "mutate-delete-annotation",
        "mutate-blackboard-post",
        "mutate-capture-assertion-candidate",
        "mutate-import-annotation-batch",
        "mutate-rebuild-index",
        "mutate-update-index",
        "mutate-rebuild-insights",
        "mutate-resolve-raw-authority-blocker",
        "mutate-reset-raw-authority-census",
        "mutate-prune-orphaned-index-revision-seeds",
        "mutate-save-saved-view",
        "mutate-delete-saved-view",
        "mutate-save-recall-pack",
        "mutate-delete-recall-pack",
        "mutate-save-workspace",
        "mutate-delete-workspace",
        "mutate-record-correction",
        "mutate-delete-correction",
        "mutate-clear-corrections",
        "mutate-delete-session",
        "mutate-bulk-tag-sessions",
        "mutate-session-excision",
        "mutate-session-lifecycle-request",
        "mutate-identity-reset",
        "mutate-maintenance-target-run",
        "mutate-filesystem-reset",
    }
    assert specs["acquire-raw-sessions"].kind is OperationKind.MATERIALIZATION
    assert specs["acquire-raw-sessions"].mutates_state is True
    assert specs["plan-validation-backlog"].kind is OperationKind.PLANNING
    assert specs["ingest-archive-runtime"].kind is OperationKind.MATERIALIZATION
    assert specs["ingest-archive-runtime"].mutates_state is True
    assert specs["materialize-transcript-embeddings"].kind is OperationKind.MATERIALIZATION
    assert specs["materialize-transcript-embeddings"].mutates_state is True
    assert specs["materialize-session-insights"].kind is OperationKind.MATERIALIZATION
    assert specs["materialize-session-insights"].mutates_state is True
    assert specs["project-retrieval-band-readiness"].previewable is True
    assert specs["project-session-insight-readiness"].previewable is True
    for operation in ("mutate-rebuild-index", "mutate-update-index", "mutate-rebuild-insights"):
        assert specs[operation].mutates_state is True
        assert specs[operation].previewable is True
        assert specs[operation].idempotent is True
        assert specs[operation].effects == ("DbRead", "DbWrite")
        assert specs[operation].executor_status == "executor-routed"
    assert specs["mutate-import-annotation-batch"].kind is OperationKind.IMPORT
    assert specs["mutate-import-annotation-batch"].mutates_state is True
    assert specs["mutate-import-annotation-batch"].idempotent is True
    assert specs["mutate-import-annotation-batch"].effects == ("DbWrite",)
    assert specs["mutate-import-annotation-batch"].executor_status == "executor-routed"


def test_runtime_operation_catalog_has_declared_surfaces() -> None:
    for spec in build_runtime_operation_catalog().specs:
        assert spec.surfaces


def test_raw_authority_recovery_specs_declare_their_exact_target_kinds() -> None:
    """Recovery target refs remain authorized by the production operation catalog."""

    specs = build_runtime_operation_catalog().by_name()

    reset_policy = specs["mutate-reset-raw-authority-census"].target_authority
    prune_policy = specs["mutate-prune-orphaned-index-revision-seeds"].target_authority

    assert [(policy.key, policy.target_kinds) for policy in reset_policy] == [
        ("raw-authority-recovery-source", ("source",))
    ]
    assert reset_policy[0].allowed_durabilities == ("durable",)
    assert [(policy.key, policy.target_kinds) for policy in prune_policy] == [
        ("raw-authority-recovery-index", ("index",))
    ]


def test_user_mutation_policy_tracks_the_supported_target_registry_and_identity_recovery() -> None:
    """Bound facade writes accept every user-state target the core registry admits."""

    specs = build_runtime_operation_catalog().by_name()
    add_mark_policy = specs["mutate-add-mark"].target_authority[0]
    identity_reset_policy = specs["mutate-identity-reset"].target_authority[0]

    assert set(TARGET_KIND_NAMES).issubset(add_mark_policy.target_kinds)
    assert identity_reset_policy.allowed_recovery == ("reconcile_required",)


def test_declared_operation_catalog_contains_runtime_and_control_plane_operations() -> None:
    catalog = build_declared_operation_catalog()

    assert "project-session-insight-readiness" in catalog.names()
    assert "compile-session-digest" in catalog.names()
    assert "render-session-report" in catalog.names()
    assert "benchmark.transform.session-digest" in catalog.names()
    assert "benchmark.transform.session-report" in catalog.names()
    assert "benchmark.archive.backup-plan" in catalog.names()
    assert "benchmark.archive.blob-gc-dry-run" in catalog.names()
    assert "benchmark.archive.space-report" in catalog.names()
    assert "benchmark.storage.crud" in catalog.names()
    assert "cli.json-contract" in catalog.names()


def test_operation_catalog_resolve_filters_unknown_names() -> None:
    catalog = build_declared_operation_catalog()

    assert tuple(spec.name for spec in catalog.resolve(("project-session-insight-readiness", "missing"))) == (
        "project-session-insight-readiness",
    )
