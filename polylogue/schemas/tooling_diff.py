"""Schema diff helpers for the tooling registry."""

from __future__ import annotations

from typing import Any

from polylogue.schemas.runtime_registry import SchemaProvider
from polylogue.schemas.tooling_models import PropertyChange, SchemaDiff


def _type_label(schema: dict[str, Any]) -> str:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return " | ".join(str(item) for item in schema_type)
    return str(schema_type)


def diff_schemas(
    provider: SchemaProvider,
    v1: str,
    v2: str,
    schema_a: dict[str, Any],
    schema_b: dict[str, Any],
) -> SchemaDiff:
    props_a = set(schema_a.get("properties", {}).keys())
    props_b = set(schema_b.get("properties", {}).keys())
    added = sorted(props_b - props_a)
    removed = sorted(props_a - props_b)
    changed: list[str] = []
    classified: list[PropertyChange] = []

    for prop in added:
        classified.append(
            PropertyChange(
                path=prop, kind="added", detail=f"new property (type: {_type_label(schema_b['properties'][prop])})"
            )
        )
    for prop in removed:
        classified.append(
            PropertyChange(
                path=prop,
                kind="removed",
                detail=f"removed property (was type: {_type_label(schema_a['properties'][prop])})",
            )
        )

    req_a = set(schema_a.get("required", []))
    req_b = set(schema_b.get("required", []))
    for prop in sorted(props_a & props_b):
        prop_a = schema_a["properties"][prop]
        prop_b = schema_b["properties"][prop]
        if prop_a.get("type") != prop_b.get("type"):
            changed.append(prop)
            classified.append(
                PropertyChange(
                    path=prop,
                    kind="type_mutation",
                    detail=f"type changed: {prop_a.get('type')} -> {prop_b.get('type')}",
                )
            )
        if (prop in req_a) != (prop in req_b):
            classified.append(
                PropertyChange(
                    path=prop,
                    kind="requiredness",
                    detail=f"{'required' if prop in req_b else 'optional'} (was {'required' if prop in req_a else 'optional'})",
                )
            )
        if prop_a.get("x-polylogue-semantic-role") != prop_b.get("x-polylogue-semantic-role"):
            classified.append(
                PropertyChange(
                    path=prop,
                    kind="semantic_role",
                    detail=f"semantic role changed: {prop_a.get('x-polylogue-semantic-role')!r} -> {prop_b.get('x-polylogue-semantic-role')!r}",
                )
            )
        if prop_a.get("x-polylogue-ref") != prop_b.get("x-polylogue-ref"):
            classified.append(
                PropertyChange(
                    path=prop,
                    kind="relational",
                    detail=f"reference changed: {prop_a.get('x-polylogue-ref')!r} -> {prop_b.get('x-polylogue-ref')!r}",
                )
            )

    for annotation_key in (
        "x-polylogue-foreign-keys",
        "x-polylogue-time-deltas",
        "x-polylogue-mutually-exclusive",
    ):
        val_a = schema_a.get(annotation_key)
        val_b = schema_b.get(annotation_key)
        if val_a != val_b:
            if val_a is None:
                detail = f"{annotation_key} added"
            elif val_b is None:
                detail = f"{annotation_key} removed"
            else:
                detail = f"{annotation_key} changed"
            classified.append(
                PropertyChange(
                    path="$",
                    kind="relational",
                    detail=detail,
                )
            )

    return SchemaDiff(
        provider=provider,
        version_a=v1,
        version_b=v2,
        added_properties=added,
        removed_properties=removed,
        changed_properties=sorted(set(changed)),
        classified_changes=classified,
    )


__all__ = ["diff_schemas"]
