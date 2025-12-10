"""Shared helpers for schema/casing normalization across CLI outputs.

Provides a small utility to emit payloads with both snake_case and camelCase
keys while stamping the current schema/CLI versions. Keep this logic in one
place to avoid drift across commands.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .version import POLYLOGUE_VERSION, SCHEMA_VERSION


def _to_camel(s: str) -> str:
    if not s or "_" not in s:
        return s
    head, *rest = s.split("_")
    return head + "".join(part.capitalize() or "_" for part in rest)


def _dualize(obj: Any) -> Any:
    """Return a structure with both snake_case and camelCase keys.

    Dict keys are duplicated in camelCase form when a snake_case variant is
    present. Nested dicts/lists are processed recursively.
    """

    if isinstance(obj, Mapping):
        out = {}
        for key, value in obj.items():
            out[key] = _dualize(value)
            camel = _to_camel(key)
            if camel != key and camel not in out:
                out[camel] = out[key]
        return out
    if isinstance(obj, list):
        return [_dualize(item) for item in obj]
    return obj


def stamp_payload(payload: Mapping[str, Any], *, dualize_keys: bool = True, include_versions: bool = True) -> dict:
    """Clone a payload and add schema/version stamps with optional casing.

    Args:
        payload: Mapping to clone.
        dualize_keys: When True, ensure both snake_case and camelCase variants
            of each key are present.
        include_versions: When True, inject schemaVersion/polylogueVersion if
            missing.
    """

    cloned = deepcopy(payload)
    if include_versions:
        cloned.setdefault("schemaVersion", SCHEMA_VERSION)
        cloned.setdefault("polylogueVersion", POLYLOGUE_VERSION)
    return _dualize(cloned) if dualize_keys else cloned


__all__ = ["stamp_payload"]
