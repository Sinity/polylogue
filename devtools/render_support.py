"""Shared helpers for generated repository surfaces."""

from __future__ import annotations

from pathlib import Path


def write_if_changed(output_path: Path, content: str) -> None:
    """Write content atomically when it differs from the current file."""
    try:
        current = output_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        current = None
    if current == content:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(output_path)
