"""Quick verification must not create a persistent type daemon."""

from __future__ import annotations

from pathlib import Path

from devtools import gate


def test_quick_mypy_uses_a_foreground_checkout_local_process() -> None:
    """The checker stays in the managed task's process tree and exits with it.

    Anti-vacuity: changing this to ``dmypy`` makes the assertion red. A daemon
    is reparented outside the managed task and one accumulates per checkout.
    """
    assert gate.mypy_command() == [str(gate.ROOT / ".venv/bin/mypy")]


def test_mypy_command_isolated_by_checkout(tmp_path: Path) -> None:
    """Each lane resolves its own environment, leaving cache ownership to mypy."""
    assert gate.mypy_command(root=tmp_path) == [str(tmp_path / ".venv/bin/mypy")]
