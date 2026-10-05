"""Long-running Polylogue daemon exports.

The command registry and convergence types resolve lazily so importing a
small daemon utility does not initialize service or storage implementations.
The registry itself loads each service command only when selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.daemon.commands import main
    from polylogue.daemon.convergence import (
        ConvergenceStage,
        DaemonConverger,
        FileState,
        StageState,
    )

__all__ = [
    "ConvergenceStage",
    "DaemonConverger",
    "FileState",
    "StageState",
    "main",
]

_CONVERGENCE_EXPORTS = frozenset({"ConvergenceStage", "DaemonConverger", "FileState", "StageState"})


def __getattr__(name: str) -> object:
    if name == "main":
        from polylogue.daemon import commands

        return commands.main
    if name in _CONVERGENCE_EXPORTS:
        from polylogue.daemon import convergence

        return getattr(convergence, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
