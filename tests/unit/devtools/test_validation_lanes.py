"""Tests for devtools.run_validation_lanes."""

from __future__ import annotations

import pytest

from devtools.run_validation_lanes import (
    LANES,
    VALID_LANES,
    LaneConfig,
    build_lane_command,
    main,
    parse_lane,
)


class TestValidationLanesImportable:
    def test_module_imports(self):
        import devtools.run_validation_lanes  # noqa: F401

    def test_main_callable(self):
        assert callable(main)


class TestLaneParsing:
    @pytest.mark.parametrize("lane_name", sorted(VALID_LANES))
    def test_valid_lane_returns_config(self, lane_name):
        lane = parse_lane(lane_name)
        assert isinstance(lane, LaneConfig)
        assert lane.name == lane_name

    def test_invalid_lane_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid lane"):
            parse_lane("nope")

    def test_all_lanes_have_description_and_timeout(self):
        for lane in LANES.values():
            assert lane.description
            assert lane.timeout_s > 0

    def test_machine_contract_lane_carries_shared_operation_metadata(self):
        lane = LANES["machine-contract"]

        assert lane.operation_targets == ("cli.json-contract",)
        assert lane.tags == ("contract", "json", "cli")

    def test_composite_lane_inherits_metadata_through_catalog_entries(self):
        from devtools.validation_catalog import build_composite_lane_entries

        frontier_local = next(entry for entry in build_composite_lane_entries() if entry.name == "frontier-local")

        assert "cli.json-contract" in frontier_local.operation_targets
        assert "cli.help" in frontier_local.operation_targets


class TestCommandConstruction:
    def test_machine_contract_lane_uses_pytest_marker(self):
        cmd = build_lane_command(LANES["machine-contract"])
        assert cmd[0] == "pytest"
        assert "machine_contract" in cmd

    def test_query_routing_lane_uses_pytest_marker(self):
        cmd = build_lane_command(LANES["query-routing"])
        assert "query_routing" in cmd

    def test_showcase_baselines_lane_uses_verify_showcase_module(self):
        cmd = build_lane_command(LANES["showcase-baselines"])
        assert cmd[:2] == ["devtools", "verify-showcase"]

    def test_pipeline_probe_chatgpt_lane_uses_probe_budgets(self):
        cmd = build_lane_command(LANES["pipeline-probe-chatgpt"])
        assert cmd[:2] == ["devtools", "pipeline-probe"]
        assert "--provider" in cmd
        assert "chatgpt" in cmd
        assert "--max-total-ms" in cmd
        assert "--max-peak-rss-mb" in cmd

    def test_live_archive_subset_parse_probe_lane_uses_medium_archive_subset_probe(self):
        cmd = build_lane_command(LANES["live-archive-subset-parse-probe"])
        assert cmd[:2] == ["devtools", "pipeline-probe"]
        assert "--input-mode" in cmd
        assert "archive-subset" in cmd
        assert "--stage" in cmd
        assert "parse" in cmd
        assert "--sample-per-provider" in cmd
        assert "50" in cmd
        assert "--workdir" in cmd
        assert "--json-out" in cmd

    def test_semantic_stack_lane_uses_explicit_semantic_suite(self):
        cmd = build_lane_command(LANES["semantic-stack"])
        assert cmd[0] == "pytest"
        assert "tests/unit/core/test_semantic_facts.py" in cmd
        assert "tests/unit/sources/test_unified_semantic_laws.py" in cmd

    def test_source_provider_fidelity_lane_uses_source_governance_suite(self):
        cmd = build_lane_command(LANES["source-provider-fidelity"])
        assert cmd[0] == "pytest"
        assert "tests/unit/sources/test_source_laws.py" in cmd
        assert "tests/unit/sources/test_drive_ops.py" in cmd
        assert "tests/integration/test_security.py" in cmd

    def test_maintenance_workflows_lane_uses_health_and_check_suite(self):
        cmd = build_lane_command(LANES["maintenance-workflows"])
        assert cmd[0] == "pytest"
        assert "tests/unit/core/test_health_core.py" in cmd
        assert "tests/unit/cli/test_check.py" in cmd
        assert "tests/integration/test_health.py" in cmd

    def test_archive_data_products_lane_uses_product_and_consumer_suite(self):
        cmd = build_lane_command(LANES["archive-data-products"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_products.py" in cmd
        assert "tests/unit/core/test_facade_api.py" in cmd
        assert "tests/unit/mcp/test_tool_contracts.py" in cmd
        assert "tests/integration/test_health.py" in cmd

    def test_semantic_product_normalization_lane_uses_normalization_toolchain_suite(self):
        cmd = build_lane_command(LANES["semantic-product-normalization"])
        assert cmd[0] == "pytest"
        assert "tests/unit/core/test_repo_identity.py" in cmd
        assert "tests/unit/cli/test_products.py" in cmd
        assert "tests/integration/test_schema_operator_workflow.py" in cmd
        assert "tests/unit/sources/test_parsers_drive.py" in cmd

    def test_retrieval_checks_lane_uses_retrieval_suite(self):
        cmd = build_lane_command(LANES["retrieval-checks"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_query_exec.py" in cmd
        assert "tests/unit/core/test_health_core.py" in cmd

    def test_embeddings_coverage_lane_uses_embed_suite(self):
        cmd = build_lane_command(LANES["embeddings-coverage"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_embed.py" in cmd
        assert "tests/unit/storage/test_embedding_stats.py" in cmd

    def test_evidence_tier_contracts_lane_uses_evidence_suite(self):
        cmd = build_lane_command(LANES["evidence-tier-contracts"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_products.py" in cmd
        assert "tests/unit/storage/test_backend.py" in cmd
        assert "tests/unit/pipeline/test_prepare_semantic.py" in cmd

    def test_inference_tier_contracts_lane_uses_inference_suite(self):
        cmd = build_lane_command(LANES["inference-tier-contracts"])
        assert cmd[0] == "pytest"
        assert "tests/unit/mcp/test_tool_contracts.py" in cmd
        assert "tests/unit/pipeline/test_prepare_semantic.py" in cmd

    def test_mixed_consumer_contracts_lane_uses_consumer_suite(self):
        cmd = build_lane_command(LANES["mixed-consumer-contracts"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_products.py" in cmd
        assert "tests/unit/core/test_facade_api.py" in cmd
        assert "tests/integration/test_health.py" in cmd

    def test_retrieval_band_readiness_lane_uses_embed_health_suite(self):
        cmd = build_lane_command(LANES["retrieval-band-readiness"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_embed.py" in cmd
        assert "tests/unit/storage/test_embedding_stats.py" in cmd
        assert "tests/unit/core/test_health_core.py" in cmd

    def test_heuristic_inference_contracts_lane_uses_semantic_product_suite(self):
        cmd = build_lane_command(LANES["heuristic-inference-contracts"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_products.py" in cmd
        assert "tests/unit/pipeline/test_prepare_semantic.py" in cmd

    def test_probabilistic_enrichment_contracts_lane_uses_enrichment_suite(self):
        cmd = build_lane_command(LANES["probabilistic-enrichment-contracts"])
        assert cmd[0] == "pytest"
        assert "tests/unit/storage/test_embedding_stats.py" in cmd
        assert "tests/unit/mcp/test_tool_contracts.py" in cmd

    def test_cleanup_contracts_lane_uses_health_and_check_suite(self):
        cmd = build_lane_command(LANES["cleanup-contracts"])
        assert cmd[0] == "pytest"
        assert "tests/unit/cli/test_check.py" in cmd
        assert "tests/integration/test_health.py" in cmd

    def test_memory_budget_lane_uses_budget_runner(self):
        cmd = build_lane_command(LANES["memory-budget"])
        assert cmd[:2] == ["devtools", "query-memory-budget"]
        assert "--max-rss-mb" in cmd
        assert "polylogue" in cmd

    def test_maintenance_memory_budget_lane_uses_check_preview(self):
        cmd = build_lane_command(LANES["maintenance-memory-budget"])
        assert cmd[:2] == ["devtools", "query-memory-budget"]
        assert "--repair" in cmd
        assert "--cleanup" in cmd
        assert "--preview" in cmd

    def test_long_haul_lane_uses_campaign_runner(self):
        cmd = build_lane_command(LANES["long-haul-small"])
        assert cmd[:2] == ["devtools", "run-benchmark-campaigns"]
        assert "--scale" in cmd
        assert "small" in cmd

    def test_live_lane_uses_module_entrypoint(self):
        cmd = build_lane_command(LANES["live-exercises"])
        assert cmd[:1] == ["polylogue"]
        assert "audit" in cmd
        assert "--live" in cmd

    def test_live_maintenance_preview_lane_uses_doctor_preview(self):
        cmd = build_lane_command(LANES["live-maintenance-preview"])
        assert cmd[:1] == ["polylogue"]
        assert "doctor" in cmd
        assert "--repair" in cmd
        assert "--cleanup" in cmd
        assert "--preview" in cmd

    def test_live_products_tags_lane_uses_products_entrypoint(self):
        cmd = build_lane_command(LANES["live-products-tags"])
        assert cmd[:1] == ["polylogue"]
        assert "products" in cmd
        assert "tags" in cmd
        assert "--json" in cmd

    def test_live_products_day_summaries_lane_uses_products_entrypoint(self):
        cmd = build_lane_command(LANES["live-products-day-summaries"])
        assert cmd[:1] == ["polylogue"]
        assert "products" in cmd
        assert "day-summaries" in cmd
        assert "--json" in cmd

    def test_live_products_analytics_lane_uses_products_entrypoint(self):
        cmd = build_lane_command(LANES["live-products-analytics"])
        assert cmd[:1] == ["polylogue"]
        assert "products" in cmd
        assert "analytics" in cmd
        assert "--json" in cmd

    def test_live_products_debt_lane_uses_products_entrypoint(self):
        cmd = build_lane_command(LANES["live-products-debt"])
        assert cmd[:1] == ["polylogue"]
        assert "products" in cmd
        assert "debt" in cmd
        assert "--json" in cmd

    def test_live_products_profiles_evidence_lane_uses_tiered_products_entrypoint(self):
        cmd = build_lane_command(LANES["live-products-profiles-evidence"])
        assert cmd[:1] == ["polylogue"]
        assert "products" in cmd
        assert "profiles" in cmd
        assert "evidence" in cmd
        assert "--json" in cmd

    def test_live_products_profiles_inference_lane_uses_tiered_products_entrypoint(self):
        cmd = build_lane_command(LANES["live-products-profiles-inference"])
        assert cmd[:1] == ["polylogue"]
        assert "products" in cmd
        assert "profiles" in cmd
        assert "inference" in cmd
        assert "--json" in cmd

    def test_live_products_enrichments_lane_uses_enrichment_entrypoint(self):
        cmd = build_lane_command(LANES["live-products-enrichments"])
        assert cmd[:1] == ["polylogue"]
        assert "products" in cmd
        assert "enrichments" in cmd
        assert "--json" in cmd

    def test_live_session_product_repair_lane_uses_doctor_repair_target(self):
        cmd = build_lane_command(LANES["live-session-product-repair"])
        assert cmd[:1] == ["polylogue"]
        assert "doctor" in cmd
        assert "--repair" in cmd
        assert "--target" in cmd
        assert "session_products" in cmd

    def test_composite_lane_has_no_direct_command(self):
        with pytest.raises(ValueError, match="composite"):
            build_lane_command(LANES["frontier-local"])

    def test_dry_run_returns_zero(self, capsys):
        exit_code = main(["--lane", "frontier-local", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "machine-contract" in captured.out
        assert "query-routing" in captured.out
        assert "showcase-baselines" in captured.out
        assert "semantic-stack" in captured.out

    def test_archive_intelligence_dry_run_includes_new_lanes(self, capsys):
        exit_code = main(["--lane", "archive-intelligence", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "retrieval-checks" in captured.out
        assert "embeddings-coverage" in captured.out

    def test_frontier_extended_dry_run_includes_pipeline_probe_lane(self, capsys):
        exit_code = main(["--lane", "frontier-extended", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "pipeline-probe-chatgpt" in captured.out

    def test_source_runtime_alignment_dry_run_includes_new_lanes(self, capsys):
        exit_code = main(["--lane", "source-runtime-alignment", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "source-provider-fidelity" in captured.out
        assert "maintenance-workflows" in captured.out

    def test_live_maintenance_small_dry_run_includes_preview_and_budget(self, capsys):
        exit_code = main(["--lane", "live-maintenance-small", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "live-maintenance-preview" in captured.out
        assert "maintenance-memory-budget" in captured.out

    def test_archive_data_products_live_dry_run_includes_local_and_live_product_lanes(self, capsys):
        exit_code = main(["--lane", "archive-data-products-live", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "archive-data-products" in captured.out
        assert "live-products-small" in captured.out

    def test_domain_read_model_live_dry_run_includes_new_product_and_maintenance_lanes(self, capsys):
        exit_code = main(["--lane", "domain-read-model-live", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "live-products-small" in captured.out
        assert "live-products-analytics" in captured.out
        assert "live-products-debt" in captured.out
        assert "live-maintenance-small" in captured.out

    def test_domain_read_model_hardening_dry_run_expands_both_subtrees(self, capsys):
        exit_code = main(["--lane", "domain-read-model-hardening", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "domain-read-model-contracts" in captured.out
        assert "domain-read-model-live" in captured.out

    def test_runtime_substrate_contracts_dry_run_includes_contract_lanes(self, capsys):
        exit_code = main(["--lane", "runtime-substrate-contracts", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "query-routing" in captured.out
        assert "semantic-stack" in captured.out
        assert "maintenance-workflows" in captured.out
        assert "archive-data-products" in captured.out

    def test_runtime_substrate_live_dry_run_includes_live_and_budget_lanes(self, capsys):
        exit_code = main(["--lane", "runtime-substrate-live", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "live-archive-small" in captured.out
        assert "live-maintenance-small" in captured.out
        assert "memory-budget" in captured.out

    def test_runtime_substrate_hardening_dry_run_expands_both_subtrees(self, capsys):
        exit_code = main(["--lane", "runtime-substrate-hardening", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "runtime-substrate-contracts" in captured.out
        assert "runtime-substrate-live" in captured.out

    def test_semantic_product_live_dry_run_includes_normalized_product_and_maintenance_lanes(self, capsys):
        exit_code = main(["--lane", "semantic-product-live", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "live-products-tags" in captured.out
        assert "live-products-day-summaries" in captured.out
        assert "live-products-debt" in captured.out
        assert "live-maintenance-small" in captured.out

    def test_semantic_product_hardening_dry_run_expands_both_subtrees(self, capsys):
        exit_code = main(["--lane", "semantic-product-hardening", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "semantic-product-normalization" in captured.out
        assert "semantic-product-live" in captured.out

    def test_evidence_contracts_dry_run_includes_tier_and_retrieval_lanes(self, capsys):
        exit_code = main(["--lane", "evidence-contracts", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "evidence-tier-contracts" in captured.out
        assert "inference-tier-contracts" in captured.out
        assert "mixed-consumer-contracts" in captured.out
        assert "retrieval-band-readiness" in captured.out

    def test_evidence_live_dry_run_includes_tiered_and_repair_lanes(self, capsys):
        exit_code = main(["--lane", "evidence-live", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "live-products-profiles-evidence" in captured.out
        assert "live-products-profiles-inference" in captured.out
        assert "live-products-work-events" in captured.out
        assert "live-products-phases" in captured.out
        assert "live-session-product-repair" in captured.out
        assert "maintenance-memory-budget" in captured.out

    def test_evidence_hardening_dry_run_expands_both_subtrees(self, capsys):
        exit_code = main(["--lane", "evidence-hardening", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "evidence-contracts" in captured.out
        assert "evidence-live" in captured.out
