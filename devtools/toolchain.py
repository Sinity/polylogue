"""Resolve verification tools from the invoking checkout's virtualenv."""

from __future__ import annotations

import os
from pathlib import Path


def venv_bin(name: str, *, root: Path) -> str:
    """Return an absolute path to a tool in ``root/.venv/bin``.

    The path is intentionally returned even when the file is missing.  The
    required-gate preflight then reports the missing checkout-local tool as a
    typed gate failure instead of silently falling through to ``PATH``.
    """
    return os.fspath((root / ".venv" / "bin" / name).resolve())


def venv_python(*, root: Path) -> str:
    """Return the checkout-local Python interpreter."""
    return venv_bin("python", root=root)


__all__ = ["venv_bin", "venv_python"]
