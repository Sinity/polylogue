"""Command governance matrix enforcement.

These tests ensure every CLI command path has declared governance and
that the declarations are truthful.
"""

from __future__ import annotations

import click
import pytest

from polylogue.cli.click_app import cli as root_cli
from polylogue.cli.command_inventory import iter_command_paths
from polylogue.cli.governance import GOVERNANCE
from polylogue.showcase.exercises import EXERCISE_INDEX


class TestGovernanceCoverage:
    """Every command path in the Click tree must have a governance spec."""

    def test_every_command_path_has_governance_spec(self):
        paths = iter_command_paths(root_cli)
        ungoverned = [
            cp.display_name
            for cp in paths
            if cp.display_name not in GOVERNANCE
        ]
        assert ungoverned == [], (
            f"Command paths without governance specs: {ungoverned}. "
            f"Add entries to polylogue/cli/governance.py."
        )

    def test_no_stale_governance_specs(self):
        paths = iter_command_paths(root_cli)
        known = {cp.display_name for cp in paths}
        stale = sorted(set(GOVERNANCE) - known)
        assert stale == [], (
            f"Governance specs for non-existent command paths: {stale}. "
            f"Remove them from polylogue/cli/governance.py."
        )


class TestGovernanceTruthfulness:
    """Declared capabilities must match the actual command."""

    @pytest.fixture()
    def command_map(self) -> dict[str, click.Command]:
        return {cp.display_name: cp.command for cp in iter_command_paths(root_cli)}

    def test_has_json_matches_actual_command(self, command_map):
        for name, spec in GOVERNANCE.items():
            if not spec.has_json:
                continue
            cmd = command_map.get(name)
            if cmd is None:
                continue
            param_names = {p.name for p in cmd.params}
            assert "json_output" in param_names or "json_mode" in param_names, (
                f"Governance declares has_json=True for '{name}' but command "
                f"has no json_output/json_mode parameter. Params: {sorted(param_names)}"
            )

    def test_has_seeded_exercise_matches_actual_exercise(self):
        for name, spec in GOVERNANCE.items():
            if not spec.has_seeded_exercise:
                continue
            exercise_name = f"products-{name.split()[-1]}-json"
            assert exercise_name in EXERCISE_INDEX, (
                f"Governance declares has_seeded_exercise=True for '{name}' but "
                f"exercise '{exercise_name}' not found in EXERCISE_INDEX."
            )
