"""Shared translation of Python CLI ``SystemExit`` values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemExitTranslation:
    """The process code and optional message represented by a ``SystemExit``."""

    code: int
    message: str | None = None


def _safe_system_exit_message(code: object) -> str:
    """Render a non-status code without allowing presentation to fail."""
    try:
        message = str(code)
    except BaseException:
        try:
            message = repr(code)
        except BaseException:
            return "<unprintable SystemExit code>"
    if len(message) > 256:
        return message[:253] + "..."
    return message


def translate_system_exit(exc: SystemExit) -> SystemExitTranslation:
    """Apply Python CLI semantics without treating printable values as success."""
    if type(exc.code) is int:
        return SystemExitTranslation(code=exc.code)
    return SystemExitTranslation(code=1, message=_safe_system_exit_message(exc.code))
