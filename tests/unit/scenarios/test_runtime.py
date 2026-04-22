from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.scenarios import (
    RunnerInvocation,
    dispatch_execution,
    dispatch_runner_execution,
    polylogue_execution,
    resolve_execution_command,
    run_execution,
    runner_execution,
)
from polylogue.scenarios.runtime import ExecutionRunner


def _string_env(value: object) -> dict[str, str]:
    assert isinstance(value, dict)
    env: dict[str, str] = {}
    for key, item in value.items():
        assert isinstance(key, str)
        assert isinstance(item, str)
        env[key] = item
    return env


def test_resolve_execution_command_applies_binary_overrides() -> None:
    execution = polylogue_execution("doctor", "--json")

    command = resolve_execution_command(execution, binary_overrides={"polylogue": "/tmp/polylogue"})

    assert command == ("/tmp/polylogue", "--plain", "doctor", "--json")


def test_run_execution_merges_env_and_captures_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_command: list[str] | None = None
    captured_kwargs: dict[str, object] | None = None

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal captured_command, captured_kwargs
        captured_command = command
        captured_kwargs = kwargs
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr("polylogue.scenarios.runtime.subprocess.run", fake_run)

    result = run_execution(
        polylogue_execution("--help"),
        env={"POLYLOGUE_FORCE_PLAIN": "1"},
        cwd=tmp_path,
        timeout=15.0,
        capture_output=True,
        binary_overrides={"polylogue": "/tmp/polylogue"},
    )

    assert result.exit_code == 0
    assert result.output == "ok\n"
    assert captured_command == ["/tmp/polylogue", "--plain", "--help"]
    assert captured_kwargs is not None
    assert captured_kwargs["cwd"] == tmp_path
    assert captured_kwargs["capture_output"] is True
    assert captured_kwargs["timeout"] == 15.0
    assert captured_kwargs["text"] is True
    env = _string_env(captured_kwargs["env"])
    assert env["POLYLOGUE_FORCE_PLAIN"] == "1"


def test_run_execution_rejects_composite_execution() -> None:
    from polylogue.scenarios import composite_execution

    with pytest.raises(ValueError, match="has no direct command"):
        run_execution(composite_execution("lane-a", "lane-b"))


@pytest.mark.asyncio
async def test_dispatch_execution_routes_runner_executions_through_resolver() -> None:
    captured: dict[str, object] = {}

    async def fake_runner(db_path: Path) -> dict[str, bool]:
        captured["db_path"] = db_path
        return {"ok": True}

    def resolve_runner(name: str) -> ExecutionRunner[dict[str, bool]]:
        assert name == "startup-readiness"
        return fake_runner

    result = await dispatch_execution(
        runner_execution("startup-readiness"),
        runner_resolver=resolve_runner,
        runner_args=(Path("/tmp/benchmark.db"),),
    )

    assert result == {"ok": True}
    assert captured["db_path"] == Path("/tmp/benchmark.db")


@pytest.mark.asyncio
async def test_dispatch_runner_execution_preserves_typed_result_and_kwargs() -> None:
    captured: dict[str, object] = {}

    def fake_runner(db_path: Path, *, scale: str) -> int:
        captured["db_path"] = db_path
        captured["scale"] = scale
        return 3

    def resolve_runner(name: str) -> ExecutionRunner[int]:
        assert name == "startup-readiness"
        return fake_runner

    result = await dispatch_runner_execution(
        runner_execution("startup-readiness"),
        runner_resolver=resolve_runner,
        invocation=RunnerInvocation(args=(Path("/tmp/benchmark.db"),), kwargs={"scale": "small"}),
    )

    assert result == 3
    assert captured == {"db_path": Path("/tmp/benchmark.db"), "scale": "small"}


@pytest.mark.asyncio
async def test_dispatch_execution_rejects_runner_without_resolver() -> None:
    with pytest.raises(ValueError, match="runner_resolver"):
        await dispatch_execution(runner_execution("startup-readiness"))
