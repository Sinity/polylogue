"""Resident command ownership and retained daemon capabilities."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.cli.operation_kernel import OperationCancelledError, OperationFailedError, OperationUnavailableError
from polylogue.daemon import commands


@pytest.mark.parametrize("output", [[], ["--format", "json"]])
def test_absent_status_has_no_local_recomputation(output: list[str]) -> None:
    with (
        patch("polylogue.cli.operation_kernel.dispatch", side_effect=OperationUnavailableError("absent")) as dispatch,
        patch("polylogue.daemon.status.daemon_status_payload", side_effect=AssertionError("local status forbidden")),
    ):
        result = CliRunner().invoke(commands.main, ["status", *output])
    assert result.exit_code == 1
    assert dispatch.call_count == 1
    assert dispatch.call_args.kwargs == {"daemon_only": True}
    assert dispatch.call_args.args[1].operation == "status"
    if output:
        payload = json.loads(result.output)
        assert payload["ok"] is False
        assert payload["daemon_liveness"] is False
        assert payload["status_snapshot"]["reason"] == "daemon_absent"
    else:
        assert result.output == "Polylogue daemon: unavailable\nReason: daemon_absent\nDetail: absent\n"


@pytest.mark.parametrize(
    "error",
    [
        OperationFailedError("unauthorized", "refused", request_id="synthetic-request"),
        OperationCancelledError("status", request_id="synthetic-request"),
    ],
)
def test_status_retains_typed_refusal_and_cancellation(error: OperationFailedError | OperationCancelledError) -> None:
    with patch("polylogue.cli.operation_kernel.dispatch", side_effect=error):
        result = CliRunner().invoke(commands.main, ["status", "--format", "json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["daemon_liveness"] is None
    assert payload["status_snapshot"]["reason"] == error.code
    assert payload["status_snapshot"]["request_id"] == "synthetic-request"


def test_registered_commands_retain_original_options() -> None:
    context = commands.main.make_context("polylogued", [], resilient_parsing=True)
    assert commands.main.list_commands(context) == ["api", "browser-capture", "health", "run", "status", "watch"]
    for name in ["api", "browser-capture", "health", "run", "watch"]:
        command = commands.main.get_command(context, name)
        assert command is not None
        result = CliRunner().invoke(commands.main, [name, "--help"])
        assert result.exit_code == 0, result.output
        for parameter in command.params:
            if getattr(parameter, "hidden", False):
                continue
            for option in getattr(parameter, "opts", ()):
                assert option in result.output


@pytest.mark.parametrize("args", [["--help"], ["--version"], ["unrecognized-command"]])
def test_root_discovery_does_not_load_service_runtime(tmp_path: Path, args: list[str]) -> None:
    source = (
        "import json, sys; from click.testing import CliRunner; "
        "from polylogue.daemon.commands import main; "
        "result = CliRunner().invoke(main, " + repr(args) + "); "
        "print(json.dumps({'exit': result.exit_code, 'output': result.output, "
        "'service_loaded': 'polylogue.daemon.cli' in sys.modules}))"
    )
    result = subprocess.run([sys.executable, "-c", source], env=os.environ, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    observation = json.loads(result.stdout)
    assert observation["service_loaded"] is False
    assert observation["exit"] == (2 if args == ["unrecognized-command"] else 0)
    if args == ["--version"]:
        from polylogue.version import POLYLOGUE_VERSION

        assert observation["output"] == f"polylogued, version {POLYLOGUE_VERSION}\n"


def test_status_establishes_runtime_contract_before_dispatch() -> None:
    from polylogue.runtime import RuntimeContractError

    failure = RuntimeContractError("synthetic unsupported runtime")
    with (
        patch("polylogue.runtime.require_free_threaded_runtime", side_effect=failure),
        patch("polylogue.cli.operation_kernel.dispatch") as dispatch,
    ):
        result = CliRunner().invoke(commands.main, ["status", "--format", "json"])
    assert result.exception is failure
    dispatch.assert_not_called()


def test_plain_status_names_original_request_reference() -> None:
    with patch(
        "polylogue.cli.operation_kernel.dispatch",
        side_effect=OperationFailedError("unauthorized", "refused", request_id="synthetic-request"),
    ):
        result = CliRunner().invoke(commands.main, ["status"])
    assert result.exit_code == 1
    assert result.output == (
        "Polylogue daemon: unavailable\nReason: unauthorized\nDetail: unauthorized: refused\n"
        "Request: synthetic-request\n"
    )
