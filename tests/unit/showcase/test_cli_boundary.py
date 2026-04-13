from __future__ import annotations

from types import SimpleNamespace

import pytest

from polylogue.scenarios import polylogue_execution
from polylogue.showcase import cli_boundary


def test_invoke_showcase_cli_uses_public_polylogue_command(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(execution, **kwargs):
        captured["execution"] = execution
        captured["kwargs"] = kwargs
        return SimpleNamespace(exit_code=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(cli_boundary.shutil, "which", lambda name: "/tmp/polylogue" if name == "polylogue" else None)
    monkeypatch.setattr(cli_boundary, "run_execution", fake_run)

    result = cli_boundary.invoke_showcase_cli(polylogue_execution("--help"))

    assert result.exit_code == 0
    assert result.stdout == "ok\n"
    assert captured["kwargs"]["binary_overrides"] == {"polylogue": "/tmp/polylogue"}
    assert captured["execution"].command == ("polylogue", "--plain", "--help")


def test_invoke_showcase_cli_passes_runtime_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(execution, **kwargs):
        captured["execution"] = execution
        captured["kwargs"] = kwargs
        return SimpleNamespace(exit_code=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(cli_boundary.shutil, "which", lambda name: "/tmp/polylogue" if name == "polylogue" else None)
    monkeypatch.setattr(cli_boundary, "run_execution", fake_run)

    cli_boundary.invoke_showcase_cli(
        polylogue_execution("doctor", "--json"),
        env={"POLYLOGUE_FORCE_PLAIN": "1"},
        cwd=None,
        timeout=30.0,
    )

    assert captured["execution"].display_command == ("polylogue", "doctor", "--json")
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["timeout"] == 30.0


def test_invoke_showcase_cli_requires_public_command_on_path(monkeypatch) -> None:
    monkeypatch.setattr(cli_boundary.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="requires `polylogue` on PATH"):
        cli_boundary.invoke_showcase_cli(polylogue_execution("--help"))
