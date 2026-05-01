from __future__ import annotations

import pytest

from polylogue.scenarios import ExecutionResult, ExecutionSpec, polylogue_execution
from polylogue.showcase import cli_boundary


def _captured_execution(captured: dict[str, object]) -> ExecutionSpec:
    execution = captured["execution"]
    if not isinstance(execution, ExecutionSpec):
        raise TypeError(f"expected ExecutionSpec, got {type(execution).__name__}")
    return execution


def _captured_kwargs(captured: dict[str, object]) -> dict[str, object]:
    kwargs = captured["kwargs"]
    if not isinstance(kwargs, dict):
        raise TypeError(f"expected kwargs dict, got {type(kwargs).__name__}")
    return dict(kwargs)


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
    kwargs = _captured_kwargs(captured)
    execution = _captured_execution(captured)

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
        polylogue_execution("doctor", "--format", "json"),
        env={"POLYLOGUE_FORCE_PLAIN": "1"},
        cwd=None,
        timeout=30.0,
    )

    execution = _captured_execution(captured)
    kwargs = _captured_kwargs(captured)
    assert execution.display_command == ("polylogue", "doctor", "--format", "json")
    assert kwargs["capture_output"] is True
    assert kwargs["timeout"] == 30.0


def test_invoke_showcase_cli_requires_public_command_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.showcase.cli_boundary.shutil.which", lambda _name: None)

    with pytest.raises(RuntimeError, match="requires `polylogue` on PATH"):
        cli_boundary.invoke_showcase_cli(polylogue_execution("--help"))
