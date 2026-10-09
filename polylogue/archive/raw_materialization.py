"""Shared raw-materialization classification helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def raw_jsonl_leading_objects(path: Path, *, limit: int) -> tuple[dict[str, Any], ...]:
    objects: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    return ()
                if not isinstance(payload, dict):
                    continue
                objects.append(payload)
                if len(objects) >= limit:
                    break
    except OSError:
        return ()
    return tuple(objects)


def source_path_native_id_candidates(source_path: str) -> tuple[str, ...]:
    """Return provider-native id candidates encoded in acquired source names."""
    if not source_path:
        return ()
    name = Path(source_path).name
    candidates: list[str] = []
    current = name
    for _ in range(4):
        stem = Path(current).stem
        if stem == current:
            break
        current = stem
        if current and current not in candidates:
            candidates.append(current)
        unsplit = re.sub(r"_\d+$", "", current)
        if unsplit and unsplit != current and unsplit not in candidates:
            candidates.append(unsplit)
    return tuple(candidates)


__all__ = ["raw_jsonl_leading_objects", "source_path_native_id_candidates"]
