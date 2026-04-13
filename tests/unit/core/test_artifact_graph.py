from __future__ import annotations

from polylogue.artifact_graph import ArtifactLayer, build_artifact_graph
from polylogue.artifacts import build_runtime_artifact_nodes, build_runtime_artifact_paths
from polylogue.operations import OperationKind, build_runtime_operation_catalog


def test_artifact_graph_contains_the_current_runtime_paths() -> None:
    graph = build_artifact_graph()
    nodes = graph.by_name()
    operations = {operation.name: operation for operation in graph.operations}

    assert set(nodes) >= {
        "raw_validation_state",
        "validation_backlog",
        "parse_backlog",
        "parse_quarantine",
        "archive_conversation_rows",
        "message_source_rows",
        "message_fts",
        "tool_use_source_blocks",
        "action_event_rows",
        "action_event_fts",
        "action_event_health",
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
        "session_profile_results",
        "session_enrichment_results",
        "session_work_event_results",
        "session_phase_results",
        "work_thread_results",
        "session_tag_rollup_results",
        "day_session_summary_results",
        "week_session_summary_results",
        "provider_analytics_results",
        "session_product_status_results",
        "archive_debt_results",
        "conversation_render_projection",
        "rendered_conversation_artifacts",
        "site_conversation_pages",
        "site_publication_manifest",
        "publication_records",
        "schema_packages",
        "schema_cluster_manifests",
        "inferred_corpus_specs",
        "inferred_corpus_scenarios",
        "schema_list_results",
        "schema_explanation_results",
        "conversation_query_results",
        "archive_health",
    }
    assert nodes["raw_validation_state"].layer is ArtifactLayer.DURABLE
    assert nodes["archive_conversation_rows"].layer is ArtifactLayer.DURABLE
    assert nodes["archive_conversation_rows"].depends_on == ("raw_validation_state",)
    assert nodes["message_fts"].layer is ArtifactLayer.INDEX
    assert nodes["message_source_rows"].depends_on == ("archive_conversation_rows",)
    assert nodes["message_fts"].depends_on == ("message_source_rows",)
    assert nodes["tool_use_source_blocks"].depends_on == ("archive_conversation_rows",)
    assert nodes["action_event_fts"].layer is ArtifactLayer.INDEX
    assert nodes["action_event_fts"].depends_on == ("action_event_rows",)
    assert nodes["action_event_health"].depends_on == ("action_event_rows", "action_event_fts")
    assert nodes["session_profile_merged_fts"].depends_on == ("session_profile_rows",)
    assert nodes["session_work_event_fts"].depends_on == ("session_work_event_rows",)
    assert nodes["work_thread_fts"].depends_on == ("work_thread_rows",)
    assert nodes["session_product_fts"].layer is ArtifactLayer.INDEX
    assert "session_profile_merged_fts" in nodes["session_product_fts"].depends_on
    assert nodes["session_product_health"].depends_on == ("session_product_rows", "session_product_fts")
    assert nodes["parse_quarantine"].depends_on == ("raw_validation_state",)
    assert nodes["session_product_source_conversations"].depends_on == ("archive_conversation_rows",)
    assert nodes["conversation_query_results"].depends_on == ("message_fts",)
    assert nodes["conversation_render_projection"].depends_on == ("archive_conversation_rows",)
    assert nodes["rendered_conversation_artifacts"].depends_on == ("conversation_render_projection",)
    assert nodes["site_conversation_pages"].depends_on == ("conversation_render_projection",)
    assert nodes["site_publication_manifest"].depends_on == ("site_conversation_pages",)
    assert nodes["publication_records"].depends_on == ("site_publication_manifest",)
    assert nodes["session_profile_results"].depends_on == ("session_profile_rows", "session_profile_merged_fts")
    assert nodes["week_session_summary_results"].depends_on == ("day_session_summary_rows",)
    assert nodes["provider_analytics_results"].depends_on == ("session_product_rows",)
    assert nodes["inferred_corpus_specs"].depends_on == ("schema_packages", "schema_cluster_manifests")
    assert nodes["inferred_corpus_scenarios"].depends_on == ("inferred_corpus_specs",)
    assert nodes["schema_list_results"].depends_on == (
        "schema_packages",
        "schema_cluster_manifests",
        "inferred_corpus_specs",
        "inferred_corpus_scenarios",
    )
    assert nodes["schema_explanation_results"].depends_on == ("schema_packages",)
    assert nodes["archive_health"].depends_on == ("message_fts", "action_event_health", "session_product_health")
    assert operations["plan-validation-backlog"].produces == ("validation_backlog",)
    assert operations["plan-parse-backlog"].produces == ("parse_backlog", "parse_quarantine")
    assert operations["ingest-archive-runtime"].produces == ("raw_validation_state", "archive_conversation_rows")
    assert operations["index-message-fts"].produces == ("message_fts",)
    assert operations["materialize-action-events"].produces == ("action_event_rows", "action_event_fts")
    assert operations["query-conversations"].produces == ("conversation_query_results",)
    assert operations["render-conversations"].produces == ("rendered_conversation_artifacts",)
    assert operations["publish-site"].produces == (
        "site_conversation_pages",
        "site_publication_manifest",
        "publication_records",
    )
    assert operations["project-action-event-health"].consumes == ("action_event_rows", "action_event_fts")
    assert "session_product_rows" in operations["materialize-session-products"].produces
    assert "session_profile_rows" in operations["materialize-session-products"].produces
    assert "work_thread_fts" in operations["materialize-session-products"].produces
    assert operations["project-session-product-health"].consumes == ("session_product_rows", "session_product_fts")
    assert operations["query-session-profiles"].produces == ("session_profile_results",)
    assert operations["query-session-enrichments"].produces == ("session_enrichment_results",)
    assert operations["query-session-work-events"].produces == ("session_work_event_results",)
    assert operations["query-session-phases"].produces == ("session_phase_results",)
    assert operations["query-work-threads"].produces == ("work_thread_results",)
    assert operations["query-session-tag-rollups"].produces == ("session_tag_rollup_results",)
    assert operations["query-day-session-summaries"].produces == ("day_session_summary_results",)
    assert operations["query-week-session-summaries"].produces == ("week_session_summary_results",)
    assert operations["query-provider-analytics"].produces == ("provider_analytics_results",)
    assert operations["query-session-product-status"].produces == ("session_product_status_results",)
    assert operations["query-archive-debt"].produces == ("archive_debt_results",)
    assert operations["compile-inferred-corpus-specs"].produces == ("inferred_corpus_specs",)
    assert operations["compile-inferred-corpus-scenarios"].produces == ("inferred_corpus_scenarios",)
    assert operations["query-schema-catalog"].produces == ("schema_list_results",)
    assert operations["query-schema-explanations"].produces == ("schema_explanation_results",)
    assert operations["project-archive-health"].consumes == (
        "message_fts",
        "action_event_health",
        "session_product_health",
    )
    assert operations["plan-validation-backlog"].kind is OperationKind.PLANNING


def test_artifact_graph_operations_come_from_runtime_operation_catalog() -> None:
    graph = build_artifact_graph()
    authored = build_runtime_operation_catalog()

    assert tuple(operation.name for operation in graph.operations) == authored.names()
    assert graph.operations == authored.specs


def test_artifact_graph_paths_reference_only_declared_nodes() -> None:
    graph = build_artifact_graph()
    node_names = set(graph.by_name())

    assert {path.name for path in graph.paths} == {
        "raw-reparse-loop",
        "raw-archive-ingest-loop",
        "message-fts-health-loop",
        "conversation-query-loop",
        "action-event-repair-loop",
        "session-product-repair-loop",
        "session-profile-query-loop",
        "session-enrichment-query-loop",
        "session-work-event-query-loop",
        "session-phase-query-loop",
        "work-thread-query-loop",
        "session-tag-rollup-query-loop",
        "day-summary-query-loop",
        "week-summary-query-loop",
        "provider-analytics-query-loop",
        "session-product-status-query-loop",
        "archive-debt-query-loop",
        "conversation-render-loop",
        "site-publication-loop",
        "inferred-corpus-compilation-loop",
        "schema-list-query-loop",
        "schema-explain-query-loop",
    }
    for path in graph.paths:
        assert path.nodes
        assert set(path.nodes).issubset(node_names)


def test_artifact_graph_nodes_and_paths_come_from_runtime_artifact_specs() -> None:
    graph = build_artifact_graph()

    assert graph.nodes == build_runtime_artifact_nodes()
    assert graph.paths == build_runtime_artifact_paths()


def test_artifact_graph_serializes_layers_as_strings() -> None:
    payload = build_artifact_graph().to_dict()

    assert any(node["layer"] == "durable" for node in payload["nodes"])
    assert any(path["name"] == "raw-reparse-loop" for path in payload["paths"])
    assert any(operation["name"] == "plan-validation-backlog" for operation in payload["operations"])
    assert any(operation["kind"] == "planning" for operation in payload["operations"])


def test_artifact_graph_operations_reference_only_declared_nodes() -> None:
    graph = build_artifact_graph()
    node_names = set(graph.by_name())
    path_names = set(graph.path_names())

    for operation in graph.operations:
        assert set(operation.consumes).issubset(node_names)
        assert set(operation.produces).issubset(node_names)
        assert set(operation.path_targets).issubset(path_names)


def test_artifact_graph_resolves_runtime_targets() -> None:
    graph = build_artifact_graph()

    assert "action_event_rows" in graph.artifact_names()
    assert "materialize-session-products" in graph.operation_names()
    assert tuple(artifact.name for artifact in graph.resolve_artifacts(("action_event_rows", "missing"))) == (
        "action_event_rows",
    )
    assert tuple(
        operation.name for operation in graph.resolve_operations(("project-action-event-health", "missing"))
    ) == ("project-action-event-health",)


def test_artifact_graph_lists_operations_for_each_runtime_path() -> None:
    graph = build_artifact_graph()

    assert tuple(operation.name for operation in graph.operations_for_path("raw-reparse-loop")) == (
        "plan-validation-backlog",
        "plan-parse-backlog",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("raw-archive-ingest-loop")) == (
        "plan-validation-backlog",
        "plan-parse-backlog",
        "ingest-archive-runtime",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("message-fts-health-loop")) == (
        "index-message-fts",
        "project-archive-health",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("conversation-query-loop")) == (
        "index-message-fts",
        "query-conversations",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("conversation-render-loop")) == (
        "render-conversations",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("site-publication-loop")) == (
        "publish-site",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("action-event-repair-loop")) == (
        "materialize-action-events",
        "project-action-event-health",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("session-product-repair-loop")) == (
        "materialize-session-products",
        "project-session-product-health",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("session-profile-query-loop")) == (
        "query-session-profiles",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("session-enrichment-query-loop")) == (
        "query-session-enrichments",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("session-work-event-query-loop")) == (
        "query-session-work-events",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("session-phase-query-loop")) == (
        "query-session-phases",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("work-thread-query-loop")) == (
        "query-work-threads",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("session-product-status-query-loop")) == (
        "query-session-product-status",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("archive-debt-query-loop")) == (
        "query-archive-debt",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("inferred-corpus-compilation-loop")) == (
        "compile-inferred-corpus-specs",
        "compile-inferred-corpus-scenarios",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("schema-list-query-loop")) == (
        "query-schema-catalog",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("schema-explain-query-loop")) == (
        "query-schema-explanations",
    )
