"""Tests for devtools.run_validation_lanes."""

from __future__ import annotations

import sys

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


class TestCommandConstruction:
    def test_machine_contract_lane_uses_pytest_marker(self):
        cmd = build_lane_command(LANES["machine-contract"])
        assert cmd[:3] == [sys.executable, "-m", "pytest"]
        assert "machine_contract" in cmd

    def test_query_routing_lane_uses_pytest_marker(self):
        cmd = build_lane_command(LANES["query-routing"])
        assert "query_routing" in cmd

    def test_semantic_stack_lane_uses_explicit_semantic_suite(self):
        cmd = build_lane_command(LANES["semantic-stack"])
        assert cmd[:3] == [sys.executable, "-m", "pytest"]
        assert "tests/unit/core/test_semantic_facts.py" in cmd
        assert "tests/unit/sources/test_unified_semantic_laws.py" in cmd

    def test_source_provider_fidelity_lane_uses_source_governance_suite(self):
        cmd = build_lane_command(LANES["source-provider-fidelity"])
        assert cmd[:3] == [sys.executable, "-m", "pytest"]
        assert "tests/unit/sources/test_source_laws.py" in cmd
        assert "tests/unit/sources/test_drive_ops.py" in cmd
        assert "tests/integration/test_security.py" in cmd

    def test_maintenance_control_plane_lane_uses_health_and_check_suite(self):
        cmd = build_lane_command(LANES["maintenance-control-plane"])
        assert cmd[:3] == [sys.executable, "-m", "pytest"]
        assert "tests/unit/core/test_health_core.py" in cmd
        assert "tests/unit/cli/test_check.py" in cmd
        assert "tests/integration/test_health.py" in cmd

    def test_archive_data_products_lane_uses_product_and_consumer_suite(self):
        cmd = build_lane_command(LANES["archive-data-products"])
        assert cmd[:3] == [sys.executable, "-m", "pytest"]
        assert "tests/unit/cli/test_products.py" in cmd
        assert "tests/unit/core/test_facade_api.py" in cmd
        assert "tests/unit/mcp/test_tool_contracts.py" in cmd
        assert "tests/integration/test_health.py" in cmd

    def test_semantic_product_normalization_lane_uses_normalization_toolchain_suite(self):
        cmd = build_lane_command(LANES["semantic-product-normalization"])
        assert cmd[:3] == [sys.executable, "-m", "pytest"]
        assert "tests/unit/core/test_project_normalization.py" in cmd
        assert "tests/unit/cli/test_products.py" in cmd
        assert "tests/integration/test_schema_operator_workflow.py" in cmd
        assert "tests/unit/sources/test_parsers_drive.py" in cmd

    def test_retrieval_dogfood_lane_uses_retrieval_suite(self):
        cmd = build_lane_command(LANES["retrieval-dogfood"])
        assert cmd[:3] == [sys.executable, "-m", "pytest"]
        assert "tests/unit/cli/test_query_exec.py" in cmd
        assert "tests/unit/core/test_health_core.py" in cmd

    def test_embeddings_coverage_lane_uses_embed_suite(self):
        cmd = build_lane_command(LANES["embeddings-coverage"])
        assert cmd[:3] == [sys.executable, "-m", "pytest"]
        assert "tests/unit/cli/test_embed.py" in cmd
        assert "tests/unit/storage/test_embedding_stats.py" in cmd

    def test_memory_budget_lane_uses_budget_runner(self):
        cmd = build_lane_command(LANES["memory-budget"])
        assert cmd[:3] == [sys.executable, "-m", "devtools.query_memory_budget"]
        assert "--max-rss-mb" in cmd
        assert "polylogue" in cmd

    def test_maintenance_memory_budget_lane_uses_check_preview(self):
        cmd = build_lane_command(LANES["maintenance-memory-budget"])
        assert cmd[:3] == [sys.executable, "-m", "devtools.query_memory_budget"]
        assert "--repair" in cmd
        assert "--cleanup" in cmd
        assert "--preview" in cmd

    def test_long_haul_lane_uses_campaign_runner(self):
        cmd = build_lane_command(LANES["long-haul-small"])
        assert cmd[:3] == [sys.executable, "-m", "devtools.run_campaign"]
        assert "--scale" in cmd
        assert "small" in cmd

    def test_live_lane_uses_module_entrypoint(self):
        cmd = build_lane_command(LANES["live-exercises"])
        assert cmd[:3] == [sys.executable, "-m", "polylogue"]
        assert "qa" in cmd
        assert "--live" in cmd

    def test_live_maintenance_preview_lane_uses_check_preview(self):
        cmd = build_lane_command(LANES["live-maintenance-preview"])
        assert cmd[:3] == [sys.executable, "-m", "polylogue"]
        assert "check" in cmd
        assert "--repair" in cmd
        assert "--cleanup" in cmd
        assert "--preview" in cmd

    def test_live_products_tags_lane_uses_products_entrypoint(self):
        cmd = build_lane_command(LANES["live-products-tags"])
        assert cmd[:3] == [sys.executable, "-m", "polylogue"]
        assert "products" in cmd
        assert "tags" in cmd
        assert "--json" in cmd

    def test_live_products_day_summaries_lane_uses_products_entrypoint(self):
        cmd = build_lane_command(LANES["live-products-day-summaries"])
        assert cmd[:3] == [sys.executable, "-m", "polylogue"]
        assert "products" in cmd
        assert "day-summaries" in cmd
        assert "--json" in cmd

    def test_live_products_analytics_lane_uses_products_entrypoint(self):
        cmd = build_lane_command(LANES["live-products-analytics"])
        assert cmd[:3] == [sys.executable, "-m", "polylogue"]
        assert "products" in cmd
        assert "analytics" in cmd
        assert "--json" in cmd

    def test_live_products_debt_lane_uses_products_entrypoint(self):
        cmd = build_lane_command(LANES["live-products-debt"])
        assert cmd[:3] == [sys.executable, "-m", "polylogue"]
        assert "products" in cmd
        assert "debt" in cmd
        assert "--json" in cmd

    def test_composite_lane_has_no_direct_command(self):
        with pytest.raises(ValueError, match="composite"):
            build_lane_command(LANES["frontier-local"])

    def test_dry_run_returns_zero(self, capsys):
        exit_code = main(["--lane", "frontier-local", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "machine-contract" in captured.out
        assert "query-routing" in captured.out
        assert "semantic-stack" in captured.out

    def test_archive_intelligence_dry_run_includes_new_lanes(self, capsys):
        exit_code = main(["--lane", "archive-intelligence", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "retrieval-dogfood" in captured.out
        assert "embeddings-coverage" in captured.out
        assert "schema-roundtrip" in captured.out

    def test_source_runtime_governance_dry_run_includes_new_lanes(self, capsys):
        exit_code = main(["--lane", "source-runtime-governance", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "source-provider-fidelity" in captured.out
        assert "maintenance-control-plane" in captured.out

    def test_live_governance_small_dry_run_includes_preview_and_budget(self, capsys):
        exit_code = main(["--lane", "live-governance-small", "--dry-run"])
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

    def test_domain_read_model_live_dry_run_includes_new_product_and_governance_lanes(self, capsys):
        exit_code = main(["--lane", "domain-read-model-live", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "live-products-small" in captured.out
        assert "live-products-analytics" in captured.out
        assert "live-products-debt" in captured.out
        assert "live-governance-small" in captured.out

    def test_domain_read_model_stewardship_dry_run_expands_both_subtrees(self, capsys):
        exit_code = main(["--lane", "domain-read-model-stewardship", "--dry-run"])
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
        assert "maintenance-control-plane" in captured.out
        assert "archive-data-products" in captured.out

    def test_runtime_substrate_live_dry_run_includes_live_and_budget_lanes(self, capsys):
        exit_code = main(["--lane", "runtime-substrate-live", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "live-archive-small" in captured.out
        assert "live-governance-small" in captured.out
        assert "memory-budget" in captured.out

    def test_runtime_substrate_hardening_dry_run_expands_both_subtrees(self, capsys):
        exit_code = main(["--lane", "runtime-substrate-hardening", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "runtime-substrate-contracts" in captured.out
        assert "runtime-substrate-live" in captured.out

    def test_semantic_product_live_dry_run_includes_normalized_product_and_governance_lanes(self, capsys):
        exit_code = main(["--lane", "semantic-product-live", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "live-products-tags" in captured.out
        assert "live-products-day-summaries" in captured.out
        assert "live-products-debt" in captured.out
        assert "live-governance-small" in captured.out

    def test_semantic_product_hardening_dry_run_expands_both_subtrees(self, capsys):
        exit_code = main(["--lane", "semantic-product-hardening", "--dry-run"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "semantic-product-normalization" in captured.out
        assert "semantic-product-live" in captured.out
