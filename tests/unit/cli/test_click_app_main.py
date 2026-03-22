"""Focused tests for the root CLI machine-error adapter."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import click
import pytest

from polylogue.cli.click_app import main

pytestmark = pytest.mark.machine_contract


def _run_main_with_error(monkeypatch: pytest.MonkeyPatch, argv: list[str], exc: BaseException, capsys) -> tuple[int, dict[str, object]]:
    monkeypatch.setattr(sys, "argv", ["polylogue", *argv])
    with patch("polylogue.cli.click_app.cli", side_effect=exc):
        with pytest.raises(SystemExit) as exit_info:
            main()
    captured = capsys.readouterr()
    return int(exit_info.value.code), json.loads(captured.out)


def test_main_wraps_usage_error_as_json(monkeypatch, capsys):
    exit_code, payload = _run_main_with_error(
        monkeypatch,
        ["check", "--json", "--bad-flag"],
        click.NoSuchOption("--bad-flag"),
        capsys,
    )

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["code"] == "invalid_arguments"
    assert payload["details"] == {"option": "--bad-flag"}


def test_main_wraps_click_exception_as_runtime_json(monkeypatch, capsys):
    exit_code, payload = _run_main_with_error(
        monkeypatch,
        ["check", "--json"],
        click.ClickException("boom"),
        capsys,
    )

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["code"] == "runtime_error"
    assert payload["message"] == "boom"


def test_main_wraps_string_system_exit_as_invalid_arguments_json(monkeypatch, capsys):
    exit_code, payload = _run_main_with_error(
        monkeypatch,
        ["check", "--json"],
        SystemExit("check: --preview requires --repair"),
        capsys,
    )

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["code"] == "invalid_arguments"
    assert payload["message"] == "check: --preview requires --repair"


def test_main_wraps_unexpected_exception_as_runtime_json(monkeypatch, capsys):
    exit_code, payload = _run_main_with_error(
        monkeypatch,
        ["check", "--json"],
        RuntimeError("unexpected boom"),
        capsys,
    )

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["code"] == "runtime_error"
    assert payload["message"] == "unexpected boom"
    assert payload["details"] == {"exception_type": "RuntimeError"}


def test_main_without_json_preserves_normal_click_failure(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["polylogue", "check", "--bad-flag"])
    with patch("polylogue.cli.click_app.cli", side_effect=click.NoSuchOption("--bad-flag")):
        with pytest.raises(click.NoSuchOption):
            main()
