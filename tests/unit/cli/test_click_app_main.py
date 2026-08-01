"""Focused tests for the root CLI machine-error adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import click
import pytest

import polylogue.cli.click_app as click_app
from devtools.checkout_guard import CheckoutImportMismatchError
from polylogue.cli.click_app import main
from polylogue.core.json import JSONDocument
from tests.infra.json_contracts import parse_json_object

pytestmark = pytest.mark.machine_contract


def _system_exit_code(code: str | int | None) -> int:
    if isinstance(code, int):
        return code
    if code is None:
        return 0
    return 1


def _run_main_with_error(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    exc: BaseException,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, JSONDocument]:
    monkeypatch.setattr(sys, "argv", ["polylogue", *argv])
    with patch("polylogue.cli.click_app.cli", side_effect=exc):
        with pytest.raises(SystemExit) as exit_info:
            main()
    captured = capsys.readouterr()
    return _system_exit_code(exit_info.value.code), parse_json_object(captured.out, context="main stdout")


def test_main_wraps_usage_error_as_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, payload = _run_main_with_error(
        monkeypatch,
        ["ops", "doctor", "--format", "json", "--bad-flag"],
        click.NoSuchOption("--bad-flag"),
        capsys,
    )

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["code"] == "invalid_arguments"
    assert payload["details"] == {"option": "--bad-flag"}


def test_main_wraps_click_exception_as_runtime_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code, payload = _run_main_with_error(
        monkeypatch,
        ["ops", "doctor", "--format", "json"],
        click.ClickException("boom"),
        capsys,
    )

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["code"] == "runtime_error"
    assert payload["message"] == "boom"


def test_main_wraps_string_system_exit_as_invalid_arguments_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code, payload = _run_main_with_error(
        monkeypatch,
        ["ops", "doctor", "--format", "json"],
        SystemExit("doctor: --preview requires --repair"),
        capsys,
    )

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["code"] == "invalid_arguments"
    assert payload["message"] == "doctor: --preview requires --repair"


def test_main_wraps_unexpected_exception_as_runtime_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code, payload = _run_main_with_error(
        monkeypatch,
        ["ops", "doctor", "--format", "json"],
        RuntimeError("unexpected boom"),
        capsys,
    )

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["code"] == "runtime_error"
    assert payload["message"] == "unexpected boom"
    assert payload["details"] == {"exception_type": "RuntimeError"}


def test_main_without_json_converts_click_usage_to_systemexit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main()`` traps Click ``UsageError``, prints an actionable hint, and exits 2 (#1273).

    Previously the human path let ``click.NoSuchOption`` propagate so Click's
    own ``standalone_mode`` machinery would print and exit. With #1273 the
    human path runs in ``standalone_mode=False`` so we can decorate every
    Click ``UsageError`` with an actionable next-step hint; the resulting
    behavior is ``SystemExit(2)`` plus an extra ``Hint:`` line on stderr.
    """
    monkeypatch.setattr(sys, "argv", ["polylogue", "doctor", "--bad-flag"])
    with patch("polylogue.cli.click_app.cli", side_effect=click.NoSuchOption("--bad-flag")):
        with pytest.raises(SystemExit) as excinfo:
            main()
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "No such option" in captured.err and "--bad-flag" in captured.err
    assert "Hint:" in captured.err


def test_main_refuses_on_checkout_mismatch(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The bare ``polylogue``/``plg`` console script refuses before dispatching (polylogue-6mgg).

    Regression for the gap where ``devtools``/pytest were guarded against the
    shared-venv worktree-import hazard but the installed CLI entry point was
    not: an agent `cd`ing into a linked worktree and running `polylogue ...`
    got silently routed to the main checkout's code with no warning.
    """

    def _boom(repo_root: Path, *, context: str) -> Path:
        raise CheckoutImportMismatchError(f"{context}: mismatch against {repo_root}")

    monkeypatch.setattr(click_app, "find_git_worktree_root", lambda start: Path("/some/worktree"))
    monkeypatch.setattr(click_app, "assert_polylogue_matches_checkout", _boom)
    monkeypatch.delenv("POLYLOGUE_ALLOW_WORKTREE_ESCAPE", raising=False)
    monkeypatch.setattr(sys, "argv", ["polylogue", "status", "--json"])

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 125
    captured = capsys.readouterr()
    assert "mismatch against" in captured.err
    # And critically: it must not have produced the command's normal output --
    # the whole point is refusing *before* doing any real work.
    assert captured.out == ""


def test_main_skips_guard_when_cwd_has_no_git_ancestry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No git ancestor from cwd (e.g. a real installed-tool invocation) -- guard no-ops."""

    def _boom(repo_root: Path, *, context: str) -> Path:
        raise AssertionError("assert_polylogue_matches_checkout must not run without a git root")

    monkeypatch.setattr(click_app, "find_git_worktree_root", lambda start: None)
    monkeypatch.setattr(click_app, "assert_polylogue_matches_checkout", _boom)
    monkeypatch.delenv("POLYLOGUE_ALLOW_WORKTREE_ESCAPE", raising=False)
    monkeypatch.setattr(sys, "argv", ["polylogue", "ops", "doctor", "--format", "json"])

    with patch.object(click_app, "cli", side_effect=click.ClickException("boom")):
        with pytest.raises(SystemExit) as excinfo:
            main()

    # Reaches the normal machine-error path (not the 125 guard exit).
    assert excinfo.value.code == 1


def test_main_bypasses_guard_with_escape_env_var(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``POLYLOGUE_ALLOW_WORKTREE_ESCAPE=1`` skips the check entirely, even on a real mismatch."""

    def _boom(repo_root: Path, *, context: str) -> Path:
        raise AssertionError("assert_polylogue_matches_checkout must not run when escape is set")

    monkeypatch.setattr(click_app, "find_git_worktree_root", lambda start: Path("/some/worktree"))
    monkeypatch.setattr(click_app, "assert_polylogue_matches_checkout", _boom)
    monkeypatch.setenv("POLYLOGUE_ALLOW_WORKTREE_ESCAPE", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "ops", "doctor", "--format", "json"])

    with patch.object(click_app, "cli", side_effect=click.ClickException("boom")):
        with pytest.raises(SystemExit) as excinfo:
            main()

    assert excinfo.value.code == 1
