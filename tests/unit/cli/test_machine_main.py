from __future__ import annotations

import json

import pytest

from polylogue.cli.machine_main import run_machine_entry
from polylogue.errors import DatabaseError

TRACEBACK_SENTINEL = "Traceback (most recent call last)"


def test_run_machine_entry_plain_polylogue_error_emits_click_style_error(capsys) -> None:
    def boom() -> None:
        raise DatabaseError("Database schema version 1 is incompatible with expected version 2.")

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(boom, ["stats"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert TRACEBACK_SENTINEL not in combined
    assert "Error: Database schema version 1 is incompatible with expected version 2." in combined


def test_run_machine_entry_json_polylogue_error_emits_runtime_envelope(capsys) -> None:
    def boom(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise DatabaseError("Database schema version 1 is incompatible with expected version 2.")

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(boom, ["stats", "--json"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert TRACEBACK_SENTINEL not in combined
    parsed = json.loads(captured.out)
    assert parsed["status"] == "error"
    assert parsed["code"] == "runtime_error"
    assert parsed["message"] == "Database schema version 1 is incompatible with expected version 2."
    assert parsed["details"]["exception_type"] == "DatabaseError"


def test_run_machine_entry_format_json_polylogue_error_emits_runtime_envelope(capsys) -> None:
    def boom(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise DatabaseError("Database schema version 1 is incompatible with expected version 2.")

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(boom, ["list", "--format", "json"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert TRACEBACK_SENTINEL not in combined
    parsed = json.loads(captured.out)
    assert parsed["status"] == "error"
    assert parsed["code"] == "runtime_error"
    assert parsed["message"] == "Database schema version 1 is incompatible with expected version 2."
