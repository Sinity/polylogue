"""Long-running Polylogue daemon entrypoints."""

from __future__ import annotations

from polylogue.daemon.cli import main
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
