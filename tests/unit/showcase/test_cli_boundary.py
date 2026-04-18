from __future__ import annotations

from typing import cast

import pytest

from polylogue.scenarios import ExecutionResult, ExecutionSpec, polylogue_execution
from polylogue.showcase import cli_boundary


def test_invoke_showcase_cli_uses_public_polylogue_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(execution: ExecutionSpec, **kwargs: object) -> ExecutionResult:
        captured["execution"] = execution
        captured["kwargs"] = kwargs
        command = execution.command
        assert command is not None
        return ExecutionResult(execution=execution, command=command, exit_code=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(
        "polylogue.showcase.cli_boundary.shutil.which",
        lambda name: "/tmp/polylogue" if name == "polylogue" else None,
    )
    monkeypatch.setattr(cli_boundary, "run_execution", fake_run)

    result = cli_boundary.invoke_showcase_cli(polylogue_execution("--help"))
    kwargs = cast(dict[str, object], captured["kwargs"])
    execution = cast(ExecutionSpec, captured["execution"])

    assert result.exit_code == 0
    assert result.stdout == "ok\n"
    assert kwargs["binary_overrides"] == {"polylogue": "/tmp/polylogue"}
    assert execution.command == ("polylogue", "--plain", "--help")


def test_invoke_showcase_cli_passes_runtime_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(execution: ExecutionSpec, **kwargs: object) -> ExecutionResult:
        captured["execution"] = execution
        captured["kwargs"] = kwargs
        command = execution.command
        assert command is not None
        return ExecutionResult(execution=execution, command=command, exit_code=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(
        "polylogue.showcase.cli_boundary.shutil.which",
        lambda name: "/tmp/polylogue" if name == "polylogue" else None,
    )
    monkeypatch.setattr(cli_boundary, "run_execution", fake_run)

    cli_boundary.invoke_showcase_cli(
        polylogue_execution("doctor", "--json"),
        env={"POLYLOGUE_FORCE_PLAIN": "1"},
        cwd=None,
        timeout=30.0,
    )

    execution = cast(ExecutionSpec, captured["execution"])
    kwargs = cast(dict[str, object], captured["kwargs"])
    assert execution.display_command == ("polylogue", "doctor", "--json")
    assert kwargs["capture_output"] is True
    assert kwargs["timeout"] == 30.0


def test_invoke_showcase_cli_requires_public_command_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.showcase.cli_boundary.shutil.which", lambda _name: None)

    with pytest.raises(RuntimeError, match="requires `polylogue` on PATH"):
        cli_boundary.invoke_showcase_cli(polylogue_execution("--help"))
