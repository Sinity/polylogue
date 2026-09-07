"""Quick verification must not create a persistent type daemon."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from devtools import gate, mypy_gate


def test_quick_mypy_uses_a_foreground_checkout_local_process() -> None:
    """The checker stays in the managed task's process tree and exits with it.

    Anti-vacuity: changing this to ``dmypy`` makes the assertion red. A daemon
    is reparented outside the managed task and one accumulates per checkout.
    """
    assert gate.mypy_command() == [str(gate.ROOT / ".venv/bin/python"), "-m", "devtools.mypy_gate"]


def test_mypy_command_isolated_by_checkout(tmp_path: Path) -> None:
    """Each lane resolves its own environment through the checkout interpreter."""
    assert gate.mypy_command(root=tmp_path) == [str(tmp_path / ".venv/bin/python"), "-m", "devtools.mypy_gate"]


def test_shared_gate_uses_one_cache_and_returns_checker_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checker = tmp_path / ".venv/bin/mypy"
    checker.parent.mkdir(parents=True)
    checker.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    checker.chmod(0o755)
    common = tmp_path / ".git"
    common.mkdir()
    calls: list[list[str]] = []

    monkeypatch.setattr(mypy_gate, "_git_common_dir", lambda _root: common)

    def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(list(argv))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr("devtools.mypy_gate.subprocess.run", fake_run)

    assert mypy_gate.main(["--root", str(tmp_path)]) == 7
    assert calls == [[str(checker), "--cache-dir", str(common / "polylogue-mypy/cache")]]
    assert (common / "polylogue-mypy/lock").is_file()
