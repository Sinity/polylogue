# mypy: disable-error-code="no-untyped-def,call-arg,attr-defined"

from __future__ import annotations

from unittest.mock import patch

import click
import pytest

from polylogue.cli.machine_main import extract_option, run_machine_entry


def test_extract_option_returns_none_for_non_option_messages() -> None:
    assert extract_option("Something else happened") is None


def test_run_machine_entry_plain_success_calls_cli_once() -> None:
    called: list[str] = []

    def cli() -> None:
        called.append("plain")

    run_machine_entry(cli, ["stats"])

    assert called == ["plain"]


def test_run_machine_entry_json_bad_parameter_with_tuple_param_hint_emits_invalid_argument_envelope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class DummyUsageError(Exception):
        pass

    def bad_parameter(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise click.BadParameter("bad value", param_hint=("--provider", "--model"))

    with patch("polylogue.cli.machine_main.click.UsageError", DummyUsageError):
        with pytest.raises(SystemExit) as exc_info:
            run_machine_entry(bad_parameter, ["stats", "--json"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr().out
    assert '"code": "invalid_arguments"' in captured
    assert '"details"' in captured
    assert '"option": "--provider, --model"' in captured


def test_run_machine_entry_system_exit_string_emits_machine_invalid_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def exits_with_string(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise SystemExit("bad invocation")

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(exits_with_string, ["stats", "--json"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr().out
    assert '"code": "invalid_arguments"' in captured
    assert '"message": "bad invocation"' in captured


def test_run_machine_entry_reraises_nonzero_system_exit_for_json_mode() -> None:
    def exits_with_code(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise SystemExit(7)

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(exits_with_code, ["stats", "--json"])

    assert exc_info.value.code == 7
