"""MCP package exports."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.mcp import server as server


def __getattr__(name: str) -> ModuleType:
    if name == "server":
        return importlib.import_module(f"{__name__}.server")
    raise AttributeError(name)


__all__ = ["server"]
