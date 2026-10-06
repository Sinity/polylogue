from __future__ import annotations

import click
import pytest

from polylogue.cli.machine_main import run_machine_entry
from polylogue.cli.operation_kernel import OperationUnavailableError
from polylogue.core.errors import DatabaseError
from tests.infra.json_contracts import json_object, parse_json_object

TRACEBACK_SENTINEL = "Traceback (most recent call last)"


@pytest.mark.parametrize("format_args", [["--format", "json"], ["--json"]])
def test_daemon_required_envelope_preserves_resolved_archive_root(
    capsys: pytest.CaptureFixture[str], format_args: list[str]
) -> None:
    """Machine refusal retains the archive route already resolved by dispatch.

    Anti-vacuity: dropping ``OperationUnavailableError.archive_root`` leaves
    multi-archive clients unable to select the archive the daemon must serve.
    """

    def unavailable(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise OperationUnavailableError("start daemon", operation="archive.facets", archive_root="/archives/old")

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(unavailable, ["facets", *format_args])

    # #5700 made a daemon-absent refusal exit with the failed-read code.
    from polylogue.cli.render.outcome import FAILED_READ_EXIT_CODE

    assert exc_info.value.code == FAILED_READ_EXIT_CODE
    parsed = parse_json_object(capsys.readouterr().out, context="machine stdout")
    details = json_object(parsed["details"], context="details")
    assert details["archive_root"] == "/archives/old"


def test_plain_daemon_absent_refusal_names_the_remedy_not_an_unexpected_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without a machine format, the daemon-absent refusal stays typed.

    Anti-vacuity: drop the plain-mode ``OperationUnavailableError`` branch and
    the generic handler prints ``unexpected error: OperationUnavailableError``
    and exits 1.
    """
    from polylogue.cli.render.outcome import FAILED_READ_EXIT_CODE

    def unavailable(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise OperationUnavailableError(
            "start polylogued run to serve this operation: insights.list", operation="insights.list"
        )

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(unavailable, ["analyze", "insights"])

    assert exc_info.value.code == FAILED_READ_EXIT_CODE
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert TRACEBACK_SENTINEL not in combined
    assert "unexpected error" not in combined
    assert "Error: start polylogued run to serve this operation: insights.list" in combined


def test_run_machine_entry_plain_polylogue_error_emits_click_style_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise DatabaseError("Database schema version 0 is not expected version 1.")

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(boom, ["analyze"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert TRACEBACK_SENTINEL not in combined
    assert "Error: Database schema version 0 is not expected version 1." in combined


def test_run_machine_entry_json_polylogue_error_emits_runtime_envelope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise DatabaseError("Database schema version 0 is not expected version 1.")

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(boom, ["analyze", "--format", "json"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert TRACEBACK_SENTINEL not in combined
    parsed = parse_json_object(captured.out, context="machine stdout")
    details = json_object(parsed["details"], context="details")
    assert parsed["status"] == "error"
    assert parsed["code"] == "runtime_error"
    assert parsed["message"] == "Database schema version 0 is not expected version 1."
    assert details["exception_type"] == "DatabaseError"


def test_run_machine_entry_format_json_polylogue_error_emits_runtime_envelope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise DatabaseError("Database schema version 0 is not expected version 1.")

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(boom, ["read", "--all", "--format", "json"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert TRACEBACK_SENTINEL not in combined
    parsed = parse_json_object(captured.out, context="machine stdout")
    assert parsed["status"] == "error"
    assert parsed["code"] == "runtime_error"
    assert parsed["message"] == "Database schema version 0 is not expected version 1."


def test_run_machine_entry_extracts_query_command_without_option_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def bad_args(*, standalone_mode: bool = False) -> None:
        del standalone_mode
        raise click.UsageError("No such option: --limit")

    with pytest.raises(SystemExit) as exc_info:
        run_machine_entry(bad_args, ["analyze", "--by", "provider", "--format", "json", "--limit", "20"])

    assert exc_info.value.code == 2
    parsed = parse_json_object(capsys.readouterr().out, context="machine stdout")
    details = json_object(parsed["details"], context="details")
    assert parsed["status"] == "error"
    assert parsed["code"] == "invalid_arguments"
    assert parsed["command"] == ["analyze"]
    assert details["option"] == "--limit"
