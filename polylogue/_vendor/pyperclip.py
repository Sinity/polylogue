"""
Minimal in-memory pyperclip replacement for environments where the dependency
is not installed. This is not a full clipboard implementation but keeps the
Polylogue interface working for tests and headless environments.
"""

from __future__ import annotations

_CLIPBOARD = ""


class PyperclipException(RuntimeError):
    pass


def copy(text: str) -> None:
    global _CLIPBOARD
    _CLIPBOARD = text


def paste() -> str:
    return _CLIPBOARD
