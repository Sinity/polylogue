"""Long-running Polylogue daemon entrypoints.

``main`` and the convergence types are exported lazily (PEP 562 module
``__getattr__``) rather than imported eagerly at package-import time. Python
executes a package's ``__init__.py`` before any of its submodules, so an eager
``from polylogue.daemon.cli import main`` here made ``polylogue.daemon.cli`` --
and with it the whole storage/convergence stack -- a dependency of importing
*any* daemon submodule. ``polylogue.daemon.socket_path`` is a few dozen lines
of path and permission arithmetic that a shell completer asks for on a
keystroke, and importing it cost seconds for that reason alone.

Exactly the same argument as ``polylogue.cli.__init__``; pinned by
``tests/unit/daemon/test_socket_path.py::test_socket_path_import_is_lightweight``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

_CONVERGENCE_EXPORTS = frozenset({"ConvergenceStage", "DaemonConverger", "FileState", "StageState"})


def __getattr__(name: str) -> object:
    if name == "main":
        from polylogue.daemon import cli

        return cli.main
    if name in _CONVERGENCE_EXPORTS:
        from polylogue.daemon import convergence

        return getattr(convergence, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
