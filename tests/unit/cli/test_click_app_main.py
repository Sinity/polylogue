"""Focused tests for the root CLI machine-error adapter."""

from __future__ import annotations

import json
import sys
from typing import cast
from unittest.mock import patch

import click
import pytest

from polylogue.cli.click_app import main

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
) -> tuple[int, dict[str, object]]:
    monkeypatch.setattr(sys, "argv", ["polylogue", *argv])
    with patch("polylogue.cli.click_app.cli", side_effect=exc):
        with pytest.raises(SystemExit) as exit_info:
            main()
    captured = capsys.readouterr()
    return _system_exit_code(exit_info.value.code), cast(dict[str, object], json.loads(captured.out))


def test_main_wraps_usage_error_as_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, payload = _run_main_with_error(
        monkeypatch,
        ["doctor", "--json", "--bad-flag"],
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
        ["doctor", "--json"],
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
        ["doctor", "--json"],
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
        ["doctor", "--json"],
        RuntimeError("unexpected boom"),
        capsys,
    )

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["code"] == "runtime_error"
    assert payload["message"] == "unexpected boom"
    assert payload["details"] == {"exception_type": "RuntimeError"}


def test_main_without_json_preserves_normal_click_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["polylogue", "doctor", "--bad-flag"])
    with patch("polylogue.cli.click_app.cli", side_effect=click.NoSuchOption("--bad-flag")):
        with pytest.raises(click.NoSuchOption):
            main()
