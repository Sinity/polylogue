from __future__ import annotations

import json

import click
import pytest

from polylogue.cli.machine_main import run_machine_entry
from polylogue.errors import DatabaseError
from polylogue.lib.json import JSONDocument, json_document

TRACEBACK_SENTINEL = "Traceback (most recent call last)"


def _parse_json_payload(stdout: str) -> JSONDocument:
    return json_document(json.loads(stdout))


def _json_object(value: object, label: str) -> JSONDocument:
    assert isinstance(value, dict), f"Expected {label} to be a JSON object"
    return value


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
        run_machine_entry(boom, ["stats", "--json"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert TRACEBACK_SENTINEL not in combined
    parsed = _parse_json_payload(captured.out)
    details = _json_object(parsed["details"], "details")
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
    parsed = json_document(json.loads(captured.out))
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
    parsed = _parse_json_payload(capsys.readouterr().out)
    details = _json_object(parsed["details"], "details")
    assert parsed["status"] == "error"
    assert parsed["code"] == "invalid_arguments"
    assert parsed["command"] == ["stats"]
    assert details["option"] == "--limit"
