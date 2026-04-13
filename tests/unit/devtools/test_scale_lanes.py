"""Tests for devtools.run_scale_lanes — scale test lane routing."""

from __future__ import annotations

import pytest

from devtools.run_scale_lanes import (
    LANES,
    VALID_LANES,
    LaneConfig,
    build_pytest_command,
    parse_lane,
)


class TestScaleLanesImportable:
    """Verify the script is importable and has expected interface."""

    def test_module_imports(self):
        """devtools.run_scale_lanes is importable."""
        import devtools.run_scale_lanes  # noqa: F401

    def test_main_callable(self):
        """main() function exists and is callable."""
        from devtools.run_scale_lanes import main

        assert callable(main)


class TestLaneParsing:
    """Test lane parsing and validation."""

    @pytest.mark.parametrize("lane_name", sorted(VALID_LANES))
    def test_valid_lane_returns_config(self, lane_name):
        """Each valid lane name returns a LaneConfig."""
        config = parse_lane(lane_name)
        assert isinstance(config, LaneConfig)
        assert config.name == lane_name

    def test_invalid_lane_raises_value_error(self):
        """Invalid lane name raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Invalid lane"):
            parse_lane("nonexistent")

    def test_all_lanes_have_required_fields(self):
        """Every lane config has non-empty description and command."""
        for name, lane in LANES.items():
            assert lane.description, f"Lane {name!r} has empty description"
            assert lane.command, f"Lane {name!r} has empty command"
            assert lane.timeout_s > 0, f"Lane {name!r} has non-positive timeout"
            assert lane.origin == "authored.scale-lane"


class TestCommandConstruction:
    """Test pytest command building."""

    @pytest.mark.parametrize("lane_name", sorted(VALID_LANES))
    def test_command_starts_with_pytest(self, lane_name):
        """Built command starts with pytest -v."""
        lane = LANES[lane_name]
        cmd = build_pytest_command(lane)
        assert cmd[:2] == ["pytest", "-v"]
        assert "-v" in cmd

    def test_fast_lane_targets_scale_tests(self):
        """Fast lane targets the scale test file specifically."""
        cmd = build_pytest_command(LANES["fast"])
        assert any("test_scale.py" in arg for arg in cmd)

    def test_slow_lane_uses_mark_filter(self):
        """Slow lane uses -m slow marker filter."""
        cmd = build_pytest_command(LANES["slow"])
        # Find the -m that's followed by the "slow" marker (not the "-m pytest" prefix)
        found = False
        for i, arg in enumerate(cmd[:-1]):
            if arg == "-m" and "slow" in cmd[i + 1]:
                found = True
                break
        assert found, f"-m slow not found in command: {cmd}"

    def test_dry_run_returns_zero(self):
        """--dry-run prints command and returns 0."""
        from devtools.run_scale_lanes import main

        exit_code = main(["--lane", "fast", "--dry-run"])
        assert exit_code == 0
