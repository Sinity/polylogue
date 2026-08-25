"""Shared translation of Python CLI ``SystemExit`` values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemExitTranslation:
    """The process code and optional message represented by a ``SystemExit``."""

    code: int
    message: str | None = None


def translate_system_exit(exc: SystemExit) -> SystemExitTranslation:
    """Apply Python CLI semantics without treating printable values as success."""
    if exc.code is None:
        return SystemExitTranslation(code=0)
    if isinstance(exc.code, int):
        return SystemExitTranslation(code=exc.code)
    return SystemExitTranslation(code=1, message=str(exc.code))
