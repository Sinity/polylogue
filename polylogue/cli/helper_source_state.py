"""Persistent source-selection state helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path


def source_state_path() -> Path:
    raw_state_root = os.environ.get("XDG_STATE_HOME")
    state_root = Path(raw_state_root).expanduser() if raw_state_root else Path.home() / ".local/state"
    return state_root / "polylogue" / "last-source.json"


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
