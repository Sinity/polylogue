from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.scenarios import (
    dispatch_execution,
    polylogue_execution,
    resolve_execution_command,
    run_execution,
    runner_execution,
)


def test_resolve_execution_command_applies_binary_overrides() -> None:
    execution = polylogue_execution("doctor", "--json")

    command = resolve_execution_command(execution, binary_overrides={"polylogue": "/tmp/polylogue"})

    assert command == ("/tmp/polylogue", "--plain", "doctor", "--json")


def test_run_execution_merges_env_and_captures_output(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
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
    assert captured["command"] == ["/tmp/polylogue", "--plain", "--help"]
    assert captured["kwargs"]["cwd"] == tmp_path
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["timeout"] == 15.0
    assert captured["kwargs"]["text"] is True
    assert captured["kwargs"]["env"]["POLYLOGUE_FORCE_PLAIN"] == "1"


def test_run_execution_rejects_composite_execution() -> None:
    from polylogue.scenarios import composite_execution

    with pytest.raises(ValueError, match="has no direct command"):
        run_execution(composite_execution("lane-a", "lane-b"))


@pytest.mark.asyncio
async def test_dispatch_execution_routes_runner_executions_through_resolver() -> None:
    captured: dict[str, object] = {}

    async def fake_runner(db_path: Path):
        captured["db_path"] = db_path
        return {"ok": True}

    result = await dispatch_execution(
        runner_execution("startup-health"),
        runner_resolver=lambda name: fake_runner if name == "startup-health" else None,  # type: ignore[return-value]
        runner_args=(Path("/tmp/benchmark.db"),),
    )

    assert result == {"ok": True}
    assert captured["db_path"] == Path("/tmp/benchmark.db")


@pytest.mark.asyncio
async def test_dispatch_execution_rejects_runner_without_resolver() -> None:
    with pytest.raises(ValueError, match="runner_resolver"):
        await dispatch_execution(runner_execution("startup-health"))
