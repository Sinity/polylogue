"""Self-tests for the declaration-driven CLI interaction oracle."""

from __future__ import annotations

import shlex
import shutil
import subprocess

import pytest
from hypothesis import given, settings

from tests.infra.cli_interaction import (
    COMMAND_ROOT_OWNERS,
    INTERACTION_TRANSITIONS,
    QUERY_CAPABILITIES,
    RENDERERS,
    RESULT_STATES,
    assert_matrix_complete,
    coverage_gaps,
)
from tests.infra.strategies.cli import (
    CliInteractionCase,
    cli_interaction_case_strategy,
    executable_cli_example_strategy,
)


def test_generated_matrix_covers_every_declared_dimension() -> None:
    matrix = assert_matrix_complete()
    assert {cell.capability for cell in matrix} == set(QUERY_CAPABILITIES)
    assert {cell.renderer for cell in matrix} == set(RENDERERS)
    assert {cell.result_state for cell in matrix} == set(RESULT_STATES)
    assert {cell.transition for cell in matrix} == set(INTERACTION_TRANSITIONS)


def test_new_command_is_an_uncovered_cell_until_an_owner_is_declared() -> None:
    current = tuple({" ".join(cell.command.split()) for cell in assert_matrix_complete()})
    gaps = coverage_gaps(command_paths=(*current, "future-command"), owners=COMMAND_ROOT_OWNERS)
    assert "command:future-command" in gaps


@settings(max_examples=30, deadline=None)
@given(cli_interaction_case_strategy())
def test_property_cases_have_safe_cursor_and_representable_pagination(case: CliInteractionCase) -> None:
    assert case.cursor <= len(case.query)
    assert case.page_size > 0
    assert case.renderer in RENDERERS
    assert isinstance(case.argv, tuple)


@settings(max_examples=20, deadline=None)
@given(executable_cli_example_strategy())
def test_generated_examples_are_shell_quotable(example: tuple[str, ...]) -> None:
    command_line = shlex.join(example)
    assert shlex.split(command_line) == list(example)
    assert example[0] == "find"
    assert example[2] == "then"


@pytest.mark.parametrize("shell", ("bash", "zsh", "fish"))
def test_shell_matrix_has_one_authoritative_shell_set(shell: str) -> None:
    from tests.infra.cli_interaction import SUPPORTED_SHELLS

    assert shell in SUPPORTED_SHELLS


@pytest.mark.parametrize("shell", ("bash", "zsh", "fish"))
def test_completion_script_parses_in_each_installed_real_shell(shell: str) -> None:
    """The generated integration is checked by the real shell parser."""
    executable = shutil.which(shell)
    if executable is None:
        pytest.skip(f"{shell} is not installed")
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    result = CliRunner().invoke(cli, ["config", "completions", "--shell", shell])
    assert result.exit_code == 0
    check = [executable, "-n"] if shell in {"bash", "zsh"} else [executable, "--no-execute"]
    parsed = subprocess.run(check, input=result.stdout, text=True, capture_output=True, check=False)
    assert parsed.returncode == 0, parsed.stderr
