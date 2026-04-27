"""Persistent source-selection state helpers."""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.paths import state_home


def source_state_path() -> Path:
    return state_home() / "last-source.json"


def load_last_source() -> str | None:
    path = source_state_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        source = payload.get("source")
        if isinstance(source, str):
            return source
    return None


def save_last_source(source_name: str) -> None:
    path = source_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": source_name}), encoding="utf-8")


__all__ = ["load_last_source", "save_last_source", "source_state_path"]
