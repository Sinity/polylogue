"""Tests for devtools.run_validation_lanes."""

from __future__ import annotations

import importlib

import pytest

from devtools.lane_models import LaneEntry
from devtools.run_validation_lanes import (
    LANES,
    VALID_LANES,
    build_lane_command,
    main,
    parse_lane,
)
from devtools.validation_lane_runtime import run_lane
from polylogue.scenarios import AssertionSpec, ExecutionKind, polylogue_execution


class TestValidationLanesImportable:
    def test_module_imports(self) -> None:
        assert importlib.import_module("devtools.run_validation_lanes") is not None

    def test_main_callable(self) -> None:
        assert callable(main)


class TestLaneParsing:
    @pytest.mark.parametrize("lane_name", sorted(VALID_LANES))
    def test_valid_lane_returns_config(self, lane_name: str) -> None:
        lane = parse_lane(lane_name)
        assert isinstance(lane, LaneEntry)
        assert lane.name == lane_name

    def test_invalid_lane_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid lane"):
            parse_lane("nope")

    def test_all_lanes_have_description_and_timeout(self) -> None:
        for lane in LANES.values():
            assert lane.description
            assert lane.timeout_s > 0

    def test_pytest_lanes_do_not_repeat_explicit_targets(self) -> None:
        for lane in LANES.values():
            if lane.execution is None or lane.execution.kind is not ExecutionKind.PYTEST:
                continue
            explicit_targets = tuple(arg for arg in lane.execution.argv if arg.startswith("tests/"))
            assert len(explicit_targets) == len(set(explicit_targets)), lane.name

    def test_machine_contract_lane_carries_shared_operation_metadata(self) -> None:
        lane = LANES["machine-contract"]

        assert lane.execution is not None
        assert lane.execution.kind is ExecutionKind.PYTEST
        assert lane.operation_targets == ("cli.json-contract",)
        assert lane.tags == ("contract", "json", "cli")

    def test_verification_substrate_lane_carries_fixture_contract_metadata(self) -> None:
        lane = LANES["verification-substrate-contracts"]

        assert lane.execution is not None
        assert lane.execution.kind is ExecutionKind.PYTEST
        assert lane.path_targets == ("verification-fixture-substrate",)
        assert lane.artifact_targets == (
            "archive_scenario_fixtures",
            "storage_record_fixtures",
            "json_contract_helpers",
        )
        assert lane.operation_targets == ("seed-archive-scenarios", "build-storage-record-fixtures")
        assert lane.tags == ("contract", "fixtures", "semantic-precision")

    def test_retrieval_checks_lane_carries_runtime_query_and_health_metadata(self) -> None:
        lane = LANES["retrieval-checks"]

        assert lane.path_targets == ("conversation-query-loop", "message-fts-readiness-loop")
        assert lane.artifact_targets == ("message_fts", "conversation_query_results", "archive_readiness")
        assert lane.operation_targets == ("query-conversations", "project-archive-readiness")
        assert lane.tags == ("contract", "retrieval", "readiness")

    def test_composite_lane_inherits_metadata_through_catalog_entries(self) -> None:
        from devtools.validation_catalog import build_composite_lane_entries

        frontier_local = next(entry for entry in build_composite_lane_entries() if entry.name == "frontier-local")
        archive_intelligence = next(
            entry for entry in build_composite_lane_entries() if entry.name == "archive-intelligence"
        )
        runtime_substrate = next(
            entry for entry in build_composite_lane_entries() if entry.name == "runtime-substrate-hardening"
        )

        assert "cli.json-contract" in frontier_local.operation_targets
        assert "cli.help" in frontier_local.operation_targets
        assert "query-conversations" in archive_intelligence.operation_targets
        assert "project-archive-readiness" in archive_intelligence.operation_targets
        assert runtime_substrate.family == "runtime-substrate"

    def test_live_health_json_lane_carries_archive_readiness_metadata(self) -> None:
        lane = LANES["live-readiness-json"]

        assert "message-fts-readiness-loop" in lane.path_targets
        assert "archive_readiness" in lane.artifact_targets
        assert "project-archive-readiness" in lane.operation_targets

    def test_live_products_status_lane_infers_status_query_metadata(self) -> None:
        lane = LANES["live-insights-status"]

        assert lane.path_targets == ("session-product-status-query-loop",)
        assert lane.artifact_targets == ("session_product_readiness", "session_product_status_results")
        assert lane.operation_targets == ("query-session-product-status",)

    def test_live_products_debt_lane_infers_archive_debt_metadata(self) -> None:
        lane = LANES["live-insights-debt"]

        assert lane.path_targets == ("archive-debt-query-loop",)
        assert lane.artifact_targets == (
            "action_event_readiness",
            "session_product_readiness",
            "archive_readiness",
            "archive_debt_results",
        )
        assert lane.operation_targets == ("query-archive-debt",)

    def test_live_embed_stats_lane_infers_embedding_status_metadata(self) -> None:
        lane = LANES["live-embed-stats"]

        assert lane.path_targets == ("retrieval-band-readiness-loop", "embedding-status-query-loop")
        assert lane.artifact_targets == (
            "embedding_metadata_rows",
            "embedding_status_rows",
            "message_embedding_vectors",
            "action_event_readiness",
            "session_product_readiness",
            "retrieval_band_readiness",
            "embedding_status_results",
        )
        assert lane.operation_targets == (
            "project-retrieval-band-readiness",
            "query-embedding-status",
            "cli.json-contract",
        )

    def test_pipeline_probe_chatgpt_lane_infers_parse_stage_runtime_metadata(self) -> None:
        lane = LANES["pipeline-probe-chatgpt"]

        assert lane.execution is not None
        assert lane.execution.kind is ExecutionKind.PIPELINE_PROBE
        assert lane.path_targets == (
            "source-acquisition-loop",
            "raw-reparse-loop",
            "raw-archive-ingest-loop",
        )
        assert lane.artifact_targets == (
            "configured_sources",
            "source_payload_stream",
            "raw_validation_state",
            "artifact_observation_rows",
            "validation_backlog",
            "parse_backlog",
            "parse_quarantine",
            "archive_conversation_rows",
        )
        assert lane.operation_targets == (
            "acquire-raw-conversations",
            "plan-validation-backlog",
            "plan-parse-backlog",
            "ingest-archive-runtime",
        )

    def test_memory_budget_lane_preserves_wrapped_runtime_metadata(self) -> None:
        lane = LANES["memory-budget"]

        assert lane.path_targets == ("conversation-query-loop",)
        assert lane.artifact_targets == ("message_fts", "conversation_query_results")
        assert lane.operation_targets == ("query-conversations",)


class TestCommandConstruction:
    def test_machine_contract_lane_uses_pytest_marker(self) -> None:
        cmd = build_lane_command(LANES["machine-contract"])
        assert cmd[0] == "pytest"
        assert "machine_contract" in cmd

    def test_query_routing_lane_uses_pytest_marker(self) -> None:
        cmd = build_lane_command(LANES["query-routing"])
        assert "query_routing" in cmd

    def test_verification_substrate_lane_uses_infra_contract_suite(self) -> None:
        cmd = build_lane_command(LANES["verification-substrate-contracts"])
        assert cmd[0] == "pytest"
        assert "tests/infra/test_storage_records.py" in cmd
        assert "tests/infra/test_archive_scenarios.py" in cmd

    def test_showcase_baselines_lane_uses_lab_scenario_baseline_action(self) -> None:
        cmd = build_lane_command(LANES["showcase-baselines"])
        assert cmd[:3] == ["devtools", "lab-scenario", "verify-baselines"]

    def test_pipeline_probe_chatgpt_lane_uses_probe_budgets(self) -> None:
        cmd = build_lane_command(LANES["pipeline-probe-chatgpt"])
        assert cmd[:2] == ["devtools", "pipeline-probe"]
        assert "--provider" in cmd
        assert "chatgpt" in cmd
        assert "--max-total-ms" in cmd
        assert "--max-peak-rss-mb" in cmd

    def test_live_archive_subset_parse_probe_lane_uses_medium_archive_subset_probe(self) -> None:
        lane = LANES["live-archive-subset-parse-probe"]
        assert lane.execution is not None
        cmd = build_lane_command(lane)
        assert lane.execution.kind is ExecutionKind.PIPELINE_PROBE
        assert cmd[:2] == ["devtools", "pipeline-probe"]
        assert "--input-mode" in cmd
        assert "archive-subset" in cmd
        assert "--stage" in cmd
        assert "parse" in cmd
        assert "--sample-per-provider" in cmd
        assert "50" in cmd
        assert "--workdir" in cmd
        assert "--json-out" in cmd

    def test_semantic_stack_lane_uses_explicit_semantic_suite(self) -> None:
        cmd = build_lane_command(LANES["semantic-stack"])
        assert cmd[0] == "pytest"
        assert "tests/unit/core/test_semantic_facts.py" in cmd
        assert "tests/unit/sources/test_unified_semantic_laws.py" in cmd

    def test_source_provider_fidelity_lane_uses_source_governance_suite(self) -> None:
        cmd = build_lane_command(LANES["source-provider-fidelity"])
        assert cmd[0] == "pytest"
        assert "tests/unit/sources/test_source_laws.py" in cmd
        assert "tests/unit/sources/test_drive_ops.py" in cmd
        assert "tests/integration/test_security.py" in cmd

    def test_source_provider_fidelity_lane_carries_source_acquisition_metadata(self) -> None:
        lane = LANES["source-provider-fidelity"]

        assert lane.path_targets == ("source-acquisition-loop",)
        assert lane.artifact_targets == (
            "configured_sources",
            "source_payload_stream",
            "raw_validation_state",
            "artifact_observation_rows",
        )
        assert lane.operation_targets == ("acquire-raw-conversations",)
        assert lane.tags == ("contract", "sources", "acquisition")

    def test_maintenance_workflows_lane_uses_health_and_check_suite(self) -> None:
        cmd = build_lane_command(LANES["maintenance-workflows"])
        assert cmd[0] == "pytest"
        assert "tests/unit/core/test_health_core.py" in cmd
        assert "tests/unit/cli/test_check.py" in cmd
        assert "tests/integration/test_health.py" in cmd

    def test_maintenance_workflows_lane_carries_publication_runtime_metadata(self) -> None:
        lane = LANES["maintenance-workflows"]

        assert lane.path_targets == ("site-publication-loop",)
        assert lane.artifact_targets == (
            "conversation_render_projection",
            "site_conversation_pages",
            "site_publication_manifest",
            "publication_records",
        )
        assert lane.operation_targets == ("publish-site",)
        assert lane.tags == ("contract", "maintenance", "publication")

    def test_archive_data_products_lane_uses_product_and_consumer_suite(self) -> None:
        cmd = build_lane_command(LANES["archive-data-insights"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_insights.py" in cmd
        assert "tests/unit/core/test_facade_api.py" in cmd
        assert "tests/unit/mcp/test_tool_contracts.py" in cmd
        assert "tests/integration/test_health.py" in cmd

    def test_semantic_product_normalization_lane_uses_normalization_toolchain_suite(self) -> None:
        cmd = build_lane_command(LANES["semantic-product-normalization"])
        assert cmd[0] == "pytest"
        assert "tests/unit/core/test_repo_identity.py" in cmd
        assert "tests/unit/cli/test_insights.py" in cmd
        assert "tests/integration/test_schema_operator_workflow.py" in cmd
        assert "tests/unit/sources/test_parsers_drive.py" in cmd

    def test_retrieval_checks_lane_uses_retrieval_suite(self) -> None:
        cmd = build_lane_command(LANES["retrieval-checks"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_query_exec.py" in cmd
        assert "tests/unit/core/test_health_core.py" in cmd

    def test_embeddings_coverage_lane_uses_embed_suite(self) -> None:
        cmd = build_lane_command(LANES["embeddings-coverage"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_embed.py" in cmd
        assert "tests/unit/storage/test_embedding_stats.py" in cmd

    def test_embeddings_coverage_lane_carries_embedding_runtime_metadata(self) -> None:
        lane = LANES["embeddings-coverage"]

        assert lane.path_targets == (
            "embedding-materialization-loop",
            "embedding-status-query-loop",
            "retrieval-band-readiness-loop",
        )
        assert lane.artifact_targets == (
            "archive_conversation_rows",
            "embedding_metadata_rows",
            "embedding_status_rows",
            "message_embedding_vectors",
            "retrieval_band_readiness",
            "embedding_status_results",
        )
        assert lane.operation_targets == (
            "materialize-transcript-embeddings",
            "project-retrieval-band-readiness",
            "query-embedding-status",
        )
        assert lane.tags == ("contract", "embeddings", "retrieval")

    def test_evidence_tier_contracts_lane_uses_evidence_suite(self) -> None:
        cmd = build_lane_command(LANES["evidence-tier-contracts"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_insights.py" in cmd
        assert "tests/unit/storage/test_backend.py" in cmd
        assert "tests/unit/pipeline/test_prepare_semantic.py" in cmd

    def test_inference_tier_contracts_lane_uses_inference_suite(self) -> None:
        cmd = build_lane_command(LANES["inference-tier-contracts"])
        assert cmd[0] == "pytest"
        assert "tests/unit/mcp/test_tool_contracts.py" in cmd
        assert "tests/unit/pipeline/test_prepare_semantic.py" in cmd

    def test_mixed_consumer_contracts_lane_uses_consumer_suite(self) -> None:
        cmd = build_lane_command(LANES["mixed-consumer-contracts"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_insights.py" in cmd
        assert "tests/unit/core/test_facade_api.py" in cmd
        assert "tests/integration/test_health.py" in cmd

    def test_retrieval_band_readiness_lane_uses_embed_health_suite(self) -> None:
        cmd = build_lane_command(LANES["retrieval-band-readiness"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_embed.py" in cmd
        assert "tests/unit/storage/test_embedding_stats.py" in cmd
        assert "tests/unit/core/test_health_core.py" in cmd

    def test_retrieval_band_readiness_lane_carries_retrieval_health_metadata(self) -> None:
        lane = LANES["retrieval-band-readiness"]

        assert lane.path_targets == (
            "embedding-status-query-loop",
            "retrieval-band-readiness-loop",
            "message-fts-readiness-loop",
        )
        assert lane.artifact_targets == (
            "embedding_metadata_rows",
            "embedding_status_rows",
            "message_embedding_vectors",
            "retrieval_band_readiness",
            "embedding_status_results",
            "archive_readiness",
        )
        assert lane.operation_targets == (
            "project-retrieval-band-readiness",
            "query-embedding-status",
            "project-archive-readiness",
        )
        assert lane.tags == ("contract", "retrieval", "embeddings", "readiness")

    def test_heuristic_inference_contracts_lane_uses_semantic_product_suite(self) -> None:
        cmd = build_lane_command(LANES["heuristic-inference-contracts"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_insights.py" in cmd
        assert "tests/unit/pipeline/test_prepare_semantic.py" in cmd

    def test_probabilistic_enrichment_contracts_lane_uses_enrichment_suite(self) -> None:
        cmd = build_lane_command(LANES["probabilistic-enrichment-contracts"])
        assert cmd[0] == "pytest"
        assert "tests/unit/storage/test_embedding_stats.py" in cmd
        assert "tests/unit/mcp/test_tool_contracts.py" in cmd

    def test_cleanup_contracts_lane_uses_health_and_check_suite(self) -> None:
        cmd = build_lane_command(LANES["cleanup-contracts"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_check.py" in cmd
        assert "tests/integration/test_health.py" in cmd

    def test_memory_budget_lane_uses_budget_runner(self) -> None:
        cmd = build_lane_command(LANES["memory-budget"])
        assert cmd[:2] == ["devtools", "query-memory-budget"]
        assert "--max-rss-mb" in cmd
        assert "polylogue" in cmd

    def test_maintenance_memory_budget_lane_uses_check_preview(self) -> None:
        cmd = build_lane_command(LANES["maintenance-memory-budget"])
        assert cmd[:2] == ["devtools", "query-memory-budget"]
        assert "--repair" in cmd
        assert "--cleanup" in cmd
        assert "--preview" in cmd

    def test_scale_fast_lane_uses_direct_pytest_execution(self) -> None:
        cmd = build_lane_command(LANES["scale-fast"])
        assert cmd[:2] == ["pytest", "-v"]
        assert "tests/unit/storage/test_scale.py" in cmd
        assert "--timeout=30" in cmd

    def test_scale_slow_lane_uses_direct_pytest_marker_filter(self) -> None:
        cmd = build_lane_command(LANES["scale-slow"])
        assert cmd[:2] == ["pytest", "-v"]
        assert "tests/unit/storage/" in cmd
        assert "--timeout=120" in cmd
        found = False
        for i, arg in enumerate(cmd[:-1]):
            if arg == "-m" and cmd[i + 1] == "slow":
                found = True
                break
        assert found, f"-m slow not found in command: {cmd}"

    def test_scale_stretch_lane_is_composite(self) -> None:
        lane = LANES["scale-stretch"]
        assert lane.is_composite
        assert lane.sub_lanes == ("scale-fast", "scale-slow")

    def test_long_haul_lane_uses_campaign_runner(self) -> None:
        cmd = build_lane_command(LANES["long-haul-small"])
        assert cmd[:2] == ["devtools", "run-benchmark-campaigns"]
        assert "--scale" in cmd
        assert "small" in cmd

    def test_live_lane_uses_module_entrypoint(self) -> None:
        cmd = build_lane_command(LANES["live-exercises"])
        assert cmd[:2] == ["devtools", "lab-scenario"]
        assert "archive-smoke" in cmd
        assert "--live" in cmd
        assert "--tier" in cmd

    def test_live_maintenance_preview_lane_uses_doctor_preview(self) -> None:
        cmd = build_lane_command(LANES["live-maintenance-preview"])
        assert cmd[:1] == ["polylogue"]
        assert "doctor" in cmd
        assert "--repair" in cmd
        assert "--cleanup" in cmd
        assert "--preview" in cmd

    def test_live_products_tags_lane_uses_products_entrypoint(self) -> None:
        cmd = build_lane_command(LANES["live-insights-tags"])
        assert cmd[:1] == ["polylogue"]
        assert "insights" in cmd
        assert "tags" in cmd
        assert "--format" in cmd
        assert "json" in cmd

    def test_live_products_day_summaries_lane_uses_products_entrypoint(self) -> None:
        cmd = build_lane_command(LANES["live-insights-day-summaries"])
        assert cmd[:1] == ["polylogue"]
        assert "insights" in cmd
        assert "day-summaries" in cmd
        assert "--format" in cmd
        assert "json" in cmd


class TestLaneAssertions:
    def test_run_lane_enforces_shared_assertion_spec(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        lane = LaneEntry(
            name="json-contract",
            description="JSON contract lane",
            timeout_s=30,
            category="contract",
            execution=polylogue_execution("doctor", "--format", "json"),
            assertion=AssertionSpec(stdout_is_valid_json=True),
        )

        class _Result:
            exit_code = 0
            stdout = "not-json"
            stderr = ""
            output = "not-json"

        monkeypatch.setattr("devtools.validation_lane_runtime.run_execution", lambda *_args, **_kwargs: _Result())

        exit_code = run_lane(lane)

        captured = capsys.readouterr()
        assert exit_code == 1
        assert "failed assertion" in captured.out

    def test_live_products_analytics_lane_uses_products_entrypoint(self) -> None:
        cmd = build_lane_command(LANES["live-insights-analytics"])
        assert cmd[:1] == ["polylogue"]
        assert "insights" in cmd
        assert "analytics" in cmd
        assert "--format" in cmd
        assert "json" in cmd

    def test_live_products_debt_lane_uses_products_entrypoint(self) -> None:
        cmd = build_lane_command(LANES["live-insights-debt"])
        assert cmd[:1] == ["polylogue"]
        assert "insights" in cmd
        assert "debt" in cmd
        assert "--format" in cmd
        assert "json" in cmd

    def test_live_products_profiles_evidence_lane_uses_tiered_products_entrypoint(self) -> None:
        cmd = build_lane_command(LANES["live-insights-profiles-evidence"])
        assert cmd[:1] == ["polylogue"]
        assert "insights" in cmd
        assert "profiles" in cmd
        assert "evidence" in cmd
        assert "--format" in cmd
        assert "json" in cmd

    def test_live_products_profiles_inference_lane_uses_tiered_products_entrypoint(self) -> None:
        cmd = build_lane_command(LANES["live-insights-profiles-inference"])
        assert cmd[:1] == ["polylogue"]
        assert "insights" in cmd
        assert "profiles" in cmd
        assert "inference" in cmd
        assert "--format" in cmd
        assert "json" in cmd

    def test_live_products_enrichments_lane_uses_enrichment_entrypoint(self) -> None:
        cmd = build_lane_command(LANES["live-insights-enrichments"])
        assert cmd[:1] == ["polylogue"]
        assert "insights" in cmd
        assert "enrichments" in cmd
        assert "--format" in cmd
        assert "json" in cmd

    def test_live_session_product_repair_lane_uses_doctor_repair_target(self) -> None:
        cmd = build_lane_command(LANES["live-session-product-repair"])
        assert cmd[:1] == ["polylogue"]
        assert "doctor" in cmd
        assert "--repair" in cmd
        assert "--target" in cmd
        assert "session_products" in cmd

    def test_composite_lane_has_no_direct_command(self) -> None:
        with pytest.raises(ValueError, match="composite"):
            build_lane_command(LANES["frontier-local"])

    def test_dry_run_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["--lane", "frontier-local", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "machine-contract" in captured.out
        assert "query-routing" in captured.out
        assert "showcase-baselines" in captured.out
        assert "semantic-stack" in captured.out

    def test_archive_intelligence_dry_run_includes_new_lanes(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["--lane", "archive-intelligence", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "retrieval-checks" in captured.out
        assert "embeddings-coverage" in captured.out

    def test_frontier_extended_dry_run_includes_pipeline_probe_lane(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "frontier-extended", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "pipeline-probe-chatgpt" in captured.out

    def test_source_runtime_alignment_dry_run_includes_new_lanes(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "source-runtime-alignment", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "source-provider-fidelity" in captured.out
        assert "maintenance-workflows" in captured.out

    def test_live_maintenance_small_dry_run_includes_preview_and_budget(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "live-maintenance-small", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "live-maintenance-preview" in captured.out
        assert "maintenance-memory-budget" in captured.out

    def test_archive_data_products_live_dry_run_includes_local_and_live_product_lanes(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "archive-data-insights-live", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "archive-data-insights" in captured.out
        assert "live-insights-small" in captured.out

    def test_domain_read_model_live_dry_run_includes_new_product_and_maintenance_lanes(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "domain-read-model-live", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "live-insights-small" in captured.out
        assert "live-insights-analytics" in captured.out
        assert "live-insights-debt" in captured.out
        assert "live-maintenance-small" in captured.out

    def test_domain_read_model_hardening_dry_run_expands_both_subtrees(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "domain-read-model-hardening", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "domain-read-model-contracts" in captured.out
        assert "domain-read-model-live" in captured.out

    def test_runtime_substrate_contracts_dry_run_includes_contract_lanes(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "runtime-substrate-contracts", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "query-routing" in captured.out
        assert "semantic-stack" in captured.out
        assert "maintenance-workflows" in captured.out
        assert "archive-data-insights" in captured.out

    def test_runtime_substrate_live_dry_run_includes_live_and_budget_lanes(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "runtime-substrate-live", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "live-archive-small" in captured.out
        assert "live-maintenance-small" in captured.out
        assert "memory-budget" in captured.out

    def test_runtime_substrate_hardening_dry_run_expands_both_subtrees(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "runtime-substrate-hardening", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "runtime-substrate-contracts" in captured.out
        assert "runtime-substrate-live" in captured.out

    def test_semantic_product_live_dry_run_includes_normalized_product_and_maintenance_lanes(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "semantic-product-live", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "live-insights-tags" in captured.out
        assert "live-insights-day-summaries" in captured.out
        assert "live-insights-debt" in captured.out
        assert "live-maintenance-small" in captured.out

    def test_semantic_product_hardening_dry_run_expands_both_subtrees(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "semantic-product-hardening", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "semantic-product-normalization" in captured.out
        assert "semantic-product-live" in captured.out

    def test_evidence_contracts_dry_run_includes_tier_and_retrieval_lanes(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "evidence-contracts", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "evidence-tier-contracts" in captured.out
        assert "inference-tier-contracts" in captured.out
        assert "mixed-consumer-contracts" in captured.out
        assert "retrieval-band-readiness" in captured.out

    def test_evidence_live_dry_run_includes_tiered_and_repair_lanes(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "evidence-live", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "live-insights-profiles-evidence" in captured.out
        assert "live-insights-profiles-inference" in captured.out
        assert "live-insights-work-events" in captured.out
        assert "live-insights-phases" in captured.out
        assert "live-session-product-repair" in captured.out
        assert "maintenance-memory-budget" in captured.out

    def test_evidence_hardening_dry_run_expands_both_subtrees(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["--lane", "evidence-hardening", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "evidence-contracts" in captured.out
        assert "evidence-live" in captured.out
