from __future__ import annotations

from types import SimpleNamespace

import pytest

from polylogue.showcase import cli_boundary


def test_invoke_showcase_cli_uses_public_polylogue_command(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(cli_boundary.shutil, "which", lambda name: "/tmp/polylogue" if name == "polylogue" else None)
    monkeypatch.setattr(cli_boundary.subprocess, "run", fake_run)

    result = cli_boundary.invoke_showcase_cli(["--help"])

    assert result.exit_code == 0
    assert result.stdout == "ok\n"
    assert captured["command"] == ["/tmp/polylogue", "--help"]


def test_invoke_showcase_cli_requires_public_command_on_path(monkeypatch) -> None:
    monkeypatch.setattr(cli_boundary.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="requires `polylogue` on PATH"):
        cli_boundary.invoke_showcase_cli(["--help"])
