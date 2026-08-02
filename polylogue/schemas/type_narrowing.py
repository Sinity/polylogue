"""Structural type-narrowing detection shared by promotion-safety checks.

Extracted from ``tests/unit/schemas/test_promotion_monotonicity.py`` so the
same "did any leaf type shrink versus what was previously committed?" check
that guards ``SchemaRegistry.replace_provider_packages`` in tests can also run
as part of a real commit/promotion report (``polylogue.schemas.operator.commit``),
instead of only ever existing as test-only logic.
"""

from __future__ import annotations

from typing import Any

from polylogue.core.json import JSONDocument

_STRUCTURAL_KEYS = ("properties", "items", "additionalProperties", "anyOf", "oneOf", "allOf")


def types_by_path(schema: Any, path: str = "") -> dict[str, frozenset[str]]:
    """Every typed node in a schema, keyed by structural path.

    Two schemas produced for the "same" node are comparable by taking
    ``types_by_path(before)`` and ``types_by_path(after)`` and checking that
    every path's type set in ``before`` is a subset of the corresponding set
    in ``after`` -- see ``narrowed_paths`` for the comparison helper.
    """
    found: dict[str, frozenset[str]] = {}

    def merge(other: dict[str, frozenset[str]]) -> None:
        # A union (anyOf/oneOf) or a list of schemas can contribute distinct
        # type sets for the SAME path from different branches -- e.g. an
        # anyOf of {"type": "string"} and {"type": "number"} both project to
        # this node's own path. A plain dict.update() would let the last
        # branch silently overwrite an earlier branch's types at that path,
        # which could hide a real narrowing (the earlier-committed union
        # member's type would vanish from `found` even though it's still
        # legitimately part of this schema). Union the sets instead.
        for other_path, other_types in other.items():
            found[other_path] = found.get(other_path, frozenset()) | other_types

    if isinstance(schema, dict):
        declared = schema.get("type")
        if isinstance(declared, str):
            found[path] = frozenset({declared})
        elif isinstance(declared, list):
            found[path] = frozenset(item for item in declared if isinstance(item, str))
        for key, value in schema.items():
            if not isinstance(value, (dict, list)):
                continue
            structural = key in _STRUCTURAL_KEYS
            merge(types_by_path(value, path if structural else f"{path}.{key}"))
    elif isinstance(schema, list):
        for entry in schema:
            merge(types_by_path(entry, path))
    return found


def narrowed_paths(before: JSONDocument | None, after: JSONDocument | None) -> tuple[str, ...]:
    """Paths whose type set in ``before`` is not a subset of ``after``'s.

    An empty ``before`` (nothing previously committed) never narrows anything.
    A ``None`` ``after`` narrows every path ``before`` declared a type for.
    """
    if before is None:
        return ()
    before_types = types_by_path(before)
    after_types = types_by_path(after) if after is not None else {}
    return tuple(
        path
        for path, before_type_set in before_types.items()
        if not before_type_set <= after_types.get(path, frozenset())
    )


def added_paths(before: JSONDocument | None, after: JSONDocument | None) -> tuple[str, ...]:
    """Paths present in ``after`` with no corresponding typed node in ``before``."""
    if after is None:
        return ()
    after_types = types_by_path(after)
    before_types = types_by_path(before) if before is not None else {}
    return tuple(path for path in after_types if path not in before_types)


__all__ = ["added_paths", "narrowed_paths", "types_by_path"]
