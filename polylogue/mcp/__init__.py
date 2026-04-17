"""MCP package exports."""

from __future__ import annotations

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "server":
        return importlib.import_module(f"{__name__}.server")
    raise AttributeError(name)


__all__ = ["server"]
