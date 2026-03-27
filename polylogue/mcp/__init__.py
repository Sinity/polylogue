"""MCP package exports."""

from __future__ import annotations

import importlib


def __getattr__(name: str):
    if name == "server":
        return importlib.import_module(f"{__name__}.server")
    raise AttributeError(name)


__all__ = ["server"]
