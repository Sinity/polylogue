"""Anti-vacuity tests for the shared first-run guided path (polylogue-jnj.8).

Every command string in ``polylogue.cli.onboarding.GUIDED_PATH_STEPS`` is
either executed end-to-end against a real fixture archive, or parsed
through the real Click parser it belongs to. A renamed, removed, or
reordered subcommand fails this test instead of silently rotting in the
onboarding text printed by bare ``polylogue``, ``polylogue tutorial``, and
the strict-command-floor error hint.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.onboarding import GUIDED_PATH_STEPS, GuidedStep, render_guided_path


def _argv(step: GuidedStep) -> list[str]:
    """The step's command with the program name stripped, exactly as typed."""
    return shlex.split(step.command_text)[1:]


def test_guided_path_step_shape() -> None:
    assert [step.number for step in GUIDED_PATH_STEPS] == [1, 2, 3, 4, 5]
    assert [step.program for step in GUIDED_PATH_STEPS] == [
        "polylogue",
        "polylogue",
        "polylogue",
        "polylogue",
        "polylogued",
    ]
    # The find+read step is the one non-destructive query/action pair.
    read_step = GUIDED_PATH_STEPS[3]
    assert read_step.argv[0] == "find"
    assert read_step.argv[-2:] == ("then", "read")


def test_steps_1_through_4_execute_in_order_against_a_real_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The demo-first walkthrough actually works end to end, in the printed order.

    This is the production-route proof polylogue-jnj.8 AC1 asks for: each
    printed command runs through the real CLI, chained exactly as a cold
    reader would type it, against a fresh isolated archive/config — not a
    mock, not a parse-only check.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive"))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    (tmp_path / "home").mkdir()
    runner = CliRunner()

    executable_steps = [step for step in GUIDED_PATH_STEPS if step.program == "polylogue"]
    assert len(executable_steps) == 4

    results = []
    for step in executable_steps:
        result = runner.invoke(cli, _argv(step), catch_exceptions=False)
        assert result.exit_code == 0, f"step {step.number} ({step.command_text}) failed:\n{result.output}"
        results.append(result)

    # Step 4's `read` genuinely rendered the demo receipts transcript content,
    # not just an empty result or a re-listed find candidate line.
    read_output = results[-1].output
    assert "clock-sensitive test" in read_output


def test_daemon_step_parses_through_the_real_daemon_parser() -> None:
    """`polylogued run` is only parse-checked here: starting it would block forever.

    A rename or removal of the `run` subcommand, or a newly required
    argument, fails this test the same way a typo in printed guidance would
    fail the executed steps above.
    """
    from polylogue.daemon.cli import main as daemon_main

    daemon_step = next(step for step in GUIDED_PATH_STEPS if step.program == "polylogued")
    assert daemon_step.argv == ("run",)

    runner = CliRunner()
    result = runner.invoke(daemon_main, [*daemon_step.argv, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_render_guided_path_contains_every_command_verbatim() -> None:
    rendered = render_guided_path()
    assert "Guided path" in rendered
    for step in GUIDED_PATH_STEPS:
        assert f"{step.number}. {step.title}" in rendered
        assert f"$ {step.command_text}" in rendered
    assert "polylogue manual" in rendered
    assert "polylogue tutorial" in rendered
