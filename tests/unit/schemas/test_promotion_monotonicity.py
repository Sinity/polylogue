"""Promotion may only ever broaden a provider package, never narrow it.

A package records what a provider has been OBSERVED to emit. Regenerating from
the current sample window and registering that directly lets a thin window
silently replace a well-sampled package.

That happened on 2026-07-29: a promotion regenerated codex and claude-code from
a narrower window and, measured against the prior packages, narrowed 33 field
types and dropped 173 fields. Concretely, codex `timestamp` went from
`["string", "number"]` to `"string"` -- but the numeric form is real, which is
why the 151,159-sample inference emitted the union in the first place. Records
carrying it would then validate as drift.
"""

from __future__ import annotations

from typing import Any, cast

from polylogue.core.json import JSONDocument
from polylogue.schemas.generation.dynamic_keys import merge_observed_structure_schemas


def _types_by_path(schema: Any, path: str = "") -> dict[str, frozenset[str]]:
    """Every typed node in a schema, keyed by structural path."""
    found: dict[str, frozenset[str]] = {}
    if isinstance(schema, dict):
        declared = schema.get("type")
        if isinstance(declared, str):
            found[path] = frozenset({declared})
        elif isinstance(declared, list):
            found[path] = frozenset(item for item in declared if isinstance(item, str))
        for key, value in schema.items():
            if not isinstance(value, (dict, list)):
                continue
            structural = key in ("properties", "items", "additionalProperties", "anyOf", "oneOf", "allOf")
            found.update(_types_by_path(value, path if structural else f"{path}.{key}"))
    elif isinstance(schema, list):
        for entry in schema:
            found.update(_types_by_path(entry, path))
    return found


def test_merging_a_thin_window_cannot_narrow_a_union() -> None:
    """The exact codex timestamp regression: string|number must survive a string-only window."""
    promoted: JSONDocument = {
        "type": "object",
        "properties": {"timestamp": {"type": ["string", "number"]}, "id": {"type": "string"}},
    }
    thin_window: JSONDocument = {
        "type": "object",
        "properties": {"timestamp": {"type": "string"}, "id": {"type": "string"}},
    }

    merged = merge_observed_structure_schemas([promoted, thin_window])

    timestamp = cast("dict[str, Any]", cast("dict[str, Any]", merged["properties"])["timestamp"])
    assert set(timestamp["type"]) == {"string", "number"}


def test_merging_cannot_drop_a_field_the_package_already_carried() -> None:
    """173 fields vanished in the real incident; absence from a window is not evidence of removal."""
    promoted: JSONDocument = {
        "type": "object",
        "properties": {"kept": {"type": "string"}, "absent_from_new_window": {"type": "integer"}},
    }
    thin_window: JSONDocument = {"type": "object", "properties": {"kept": {"type": "string"}}}

    merged = merge_observed_structure_schemas([promoted, thin_window])

    properties = cast("dict[str, Any]", merged["properties"])
    assert "absent_from_new_window" in properties
    assert properties["absent_from_new_window"]["type"] == "integer"


def test_no_typed_node_narrows_under_merge_in_either_order() -> None:
    """Breadth is order-independent: merge is a union, not a last-writer-wins overwrite."""
    wide: JSONDocument = {
        "type": "object",
        "properties": {
            "a": {"type": ["string", "number"]},
            "nested": {"type": "object", "properties": {"b": {"type": ["integer", "null"]}}},
        },
    }
    narrow: JSONDocument = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "nested": {"type": "object", "properties": {"b": {"type": "integer"}}},
        },
    }

    for left, right in ((wide, narrow), (narrow, wide)):
        merged_types = _types_by_path(merge_observed_structure_schemas([left, right]))
        for path, wide_types in _types_by_path(wide).items():
            assert wide_types <= merged_types.get(path, frozenset()), (
                f"{path} narrowed: {sorted(wide_types)} not preserved in {sorted(merged_types.get(path, []))}"
            )
