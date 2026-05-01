from __future__ import annotations

import click
import pytest

from polylogue.cli.machine_main import run_machine_entry
from polylogue.errors import DatabaseError
from tests.infra.json_contracts import json_object, parse_json_object

TRACEBACK_SENTINEL = "Traceback (most recent call last)"


def test_run_machine_entry_plain_polylogue_error_emits_click_style_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom() -> None:
        raise DatabaseError("Database schema version 0 is incompatible with expected version 1.")

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(boom, ["stats"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert TRACEBACK_SENTINEL not in combined
    assert "Error: Database schema version 0 is incompatible with expected version 1." in combined


def test_run_machine_entry_json_polylogue_error_emits_runtime_envelope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise DatabaseError("Database schema version 0 is incompatible with expected version 1.")

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(boom, ["stats", "--format", "json"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert TRACEBACK_SENTINEL not in combined
    parsed = parse_json_object(captured.out, context="machine stdout")
    details = json_object(parsed["details"], context="details")
    assert parsed["status"] == "error"
    assert parsed["code"] == "runtime_error"
    assert parsed["message"] == "Database schema version 0 is incompatible with expected version 1."
    assert details["exception_type"] == "DatabaseError"


def test_run_machine_entry_format_json_polylogue_error_emits_runtime_envelope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise DatabaseError("Database schema version 0 is incompatible with expected version 1.")

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(boom, ["list", "--format", "json"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert TRACEBACK_SENTINEL not in combined
    parsed = parse_json_object(captured.out, context="machine stdout")
    assert parsed["status"] == "error"
    assert parsed["code"] == "runtime_error"
    assert parsed["message"] == "Database schema version 0 is incompatible with expected version 1."


def test_run_machine_entry_extracts_query_command_without_option_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def bad_args(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise click.UsageError("No such option: --limit")

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(bad_args, ["stats", "--by", "provider", "--format", "json", "--limit", "20"])

    assert exc_info.value.code == 2
    parsed = parse_json_object(capsys.readouterr().out, context="machine stdout")
    details = json_object(parsed["details"], context="details")
    assert parsed["status"] == "error"
    assert parsed["code"] == "invalid_arguments"
    assert parsed["command"] == ["stats"]
    assert details["option"] == "--limit"
