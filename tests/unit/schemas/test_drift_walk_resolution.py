"""``detect_drift`` must observe the schema a node actually declares.

Two independent failures made the drift signal wrong in both directions: a
``$ref`` node was walked as an undeclared permissive object, so every
legitimate nested field was reported as ``Unexpected``; and a dynamic-key
container skipped its VALUES, so a genuinely new provider field inside one was
never observed at all.
"""

from __future__ import annotations

from polylogue.schemas.validator import detect_drift

_REF_SCHEMA = {
    "$defs": {
        "Provenance": {
            "type": "object",
            "properties": {"captured_at": {"type": "string"}},
            "additionalProperties": False,
        }
    },
    "type": "object",
    "properties": {"provenance": {"$ref": "#/$defs/Provenance"}},
    "additionalProperties": False,
}

_DYNAMIC_SCHEMA = {
    "type": "object",
    "properties": {
        "mapping": {
            "type": "object",
            "x-polylogue-dynamic-keys": True,
            "additionalProperties": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}


def test_referenced_nodes_are_resolved_before_walking() -> None:
    """A declared field behind a local ``$ref`` is not drift.

    Anti-vacuity: without ref resolution the first assertion reports
    ``provenance.captured_at``; the second pins that a genuinely new field
    behind the same ref is still reported, so resolution cannot become
    "stop walking referenced nodes".
    """
    assert detect_drift({"provenance": {"captured_at": "x"}}, _REF_SCHEMA, "") == []
    assert detect_drift({"provenance": {"captured_at": "x", "new_field": 1}}, _REF_SCHEMA, "") == [
        "Unexpected field: provenance.new_field"
    ]


def test_unresolvable_reference_is_not_reported_as_drift() -> None:
    """An external ref has no declarations in hand, so it observes nothing.

    Reporting the whole subtree would be the same false drift the local-ref
    fix removes, just from a different cause.

    Anti-vacuity: returning the bare ``{"$ref": ...}`` fragment instead of
    ``None`` makes this report ``payload.anything``.
    """
    external = {
        "type": "object",
        "properties": {"payload": {"$ref": "https://example.invalid/schema.json#/Thing"}},
        "additionalProperties": False,
    }
    assert detect_drift({"payload": {"anything": 1}}, external, "") == []


def test_dynamic_map_values_are_still_walked() -> None:
    """The KEY of a dynamic map is not a named field; its VALUE's fields are.

    ``additionalProperties`` declares the structured node under each dynamic
    key, so ``mapping..new_provider_field`` in a ChatGPT export is a real
    provider field ingest can silently discard.

    Anti-vacuity: restoring the unconditional ``continue`` for a dynamic
    container makes the second assertion return ``[]``; the first pins that
    the dynamic keys themselves are still suppressed.
    """
    assert detect_drift({"mapping": {"k1": {"id": "a"}, "k2": {"id": "b"}}}, _DYNAMIC_SCHEMA, "") == []
    assert detect_drift({"mapping": {"k1": {"id": "a", "new_provider_field": 1}}}, _DYNAMIC_SCHEMA, "") == [
        "Unexpected field: mapping.k1.new_provider_field"
    ]
