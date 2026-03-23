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
