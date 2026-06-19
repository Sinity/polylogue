from __future__ import annotations

from pathlib import Path

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


def _make_project_cli(project_root: Path) -> Path:
    cli_path = project_root / ".venv" / "bin" / "polylogue"
    cli_path.parent.mkdir(parents=True)
    cli_path.write_text("#!/bin/sh\n", encoding="utf-8")
    return cli_path


def test_invoke_showcase_cli_uses_current_project_polylogue_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    cli_path = _make_project_cli(tmp_path)

    def fake_run(execution: ExecutionSpec, **kwargs: object) -> ExecutionResult:
        captured["execution"] = execution
        captured["kwargs"] = kwargs
        command = execution.command
        assert command is not None
        return ExecutionResult(execution=execution, command=command, exit_code=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(cli_boundary, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli_boundary, "run_execution", fake_run)

    result = cli_boundary.invoke_showcase_cli(polylogue_execution("--help"))
    kwargs = _captured_kwargs(captured)
    execution = _captured_execution(captured)

    assert result.exit_code == 0
    assert result.stdout == "ok\n"
    assert kwargs["binary_overrides"] == {"polylogue": str(cli_path)}
    assert execution.command == ("polylogue", "--plain", "--help")


def test_invoke_showcase_cli_passes_runtime_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _make_project_cli(tmp_path)

    def fake_run(execution: ExecutionSpec, **kwargs: object) -> ExecutionResult:
        captured["execution"] = execution
        captured["kwargs"] = kwargs
        command = execution.command
        assert command is not None
        return ExecutionResult(execution=execution, command=command, exit_code=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(cli_boundary, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli_boundary, "run_execution", fake_run)

    cli_boundary.invoke_showcase_cli(
        polylogue_execution("ops", "doctor", "--format", "json"),
        env={"POLYLOGUE_FORCE_PLAIN": "1"},
        cwd=None,
        timeout=30.0,
    )

    execution = _captured_execution(captured)
    kwargs = _captured_kwargs(captured)
    assert execution.display_command == ("polylogue", "ops", "doctor", "--format", "json")
    assert kwargs["capture_output"] is True
    assert kwargs["timeout"] == 30.0


def test_invoke_showcase_cli_uses_python_fallback_without_project_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> object:
        captured["command"] = tuple(command)
        captured["kwargs"] = kwargs

        class Result:
            returncode = 0
            stdout = "Usage: polylogue [OPTIONS]\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(cli_boundary, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("polylogue.showcase.cli_boundary.subprocess.run", fake_run)

    result = cli_boundary.invoke_showcase_cli(polylogue_execution("--help"))

    command = captured["command"]
    assert isinstance(command, tuple)
    assert command[-1] == "--help"
    assert any("cli(prog_name='polylogue')" in part for part in command)
    assert result.exit_code == 0
    assert result.stdout == "Usage: polylogue [OPTIONS]\n"
