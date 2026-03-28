"""Structural fingerprint helpers shared by schema sampling and generation."""

from __future__ import annotations

from typing import Any

from polylogue.schemas.field_stats import is_dynamic_key

_FINGERPRINT_MAX_DEPTH = 8
_FINGERPRINT_ARRAY_SAMPLE = 8


def _structure_fingerprint(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = _FINGERPRINT_MAX_DEPTH,
) -> Any:
    """Build a hashable structural fingerprint for schema-dedup heuristics."""
    if depth >= max_depth:
        return ("depth-limit", type(value).__name__)

    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool",)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("number",)
    if isinstance(value, str):
        return ("string",)

    if isinstance(value, list):
        item_shapes = {
            _structure_fingerprint(item, depth=depth + 1, max_depth=max_depth)
            for item in value[:_FINGERPRINT_ARRAY_SAMPLE]
        }
        return ("array", tuple(sorted(item_shapes, key=repr)))

    if isinstance(value, dict):
        props: list[tuple[str, Any]] = []
        for key in sorted(value):
            child = value[key]
            normalized_key = "*" if is_dynamic_key(key) else key
            props.append(
                (
                    normalized_key,
                    _structure_fingerprint(child, depth=depth + 1, max_depth=max_depth),
                )
            )
        return ("object", tuple(props))

    return ("other", type(value).__name__)


__all__ = ["_structure_fingerprint"]
