"""Bounded, declaration-derived query capability discovery.

The MCP resource is intentionally only an index.  This module is the detail
route used by ``explain(subject="capability")``; it has no independent
catalogue or persistence layer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256

from polylogue.archive.query.fields import QUERY_FIELD_DESCRIPTORS
from polylogue.archive.query.metadata import query_unit_descriptors
from polylogue.sources.origin_specs import origin_specs

MAX_CAPABILITY_PAGE = 25


def _snapshot_id(stats: Mapping[str, object] | None) -> str:
    payload = json.dumps(dict(stats or {}), sort_keys=True, default=str).encode()
    return "archive:" + sha256(payload).hexdigest()[:16]


def _declaration_rows() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for field in sorted(QUERY_FIELD_DESCRIPTORS, key=lambda item: (item.stable_order, item.name)):
        rows.append(
            {
                "declaration_id": f"query.field.{field.name}",
                "kind": "field",
                "name": field.name,
                "meaning": field.name.replace("_", " "),
                "authority": field.authority,
                "applicability": field.applicability,
                "projection": list(field.projections),
                "binding": {
                    "spec": field.spec_attr,
                    "plan": field.plan_attr,
                    "storage": field.storage_names,
                    "mcp": field.mcp_names,
                    "api": field.api_names,
                },
                "operators": list(field.operators),
                "value_type": field.value_type,
                "cardinality": field.cardinality,
                "cost": {
                    "shape": field.cost_shape,
                    "pushdown": field.pushdown,
                    "stats_join": field.requires_stats_join,
                    "post_filter": field.requires_post_filter,
                    "content_loading": field.requires_content_loading,
                },
                "stable_order": field.stable_order,
                "examples": list(field.examples),
            }
        )
    for descriptor in query_unit_descriptors(terminal_supported=True):
        rows.append(
            {
                "declaration_id": f"query.unit.{descriptor.unit}",
                "kind": "unit",
                "name": descriptor.unit,
                "meaning": descriptor.description,
                "authority": "derived",
                "applicability": "unit",
                "projection": ["dsl", "mcp", "api"],
                "binding": {
                    "source": descriptor.plural_source,
                    "singular_source": descriptor.singular_source,
                    "lowerer": descriptor.lowerer_kind,
                    "sql": descriptor.sql_query_method,
                    "runtime": descriptor.runtime_query_method,
                },
                "operators": ["where", "exists"] if descriptor.exists_supported else ["where"],
                "value_type": "record",
                "cardinality": "many",
                "cost": {
                    "shape": "indexed" if descriptor.lowerer_kind == "sql" else "post_filter",
                    "pushdown": descriptor.lowerer_kind == "sql",
                },
                "stable_order": 10000 + len(rows),
                "examples": [descriptor.terminal_example or descriptor.example],
            }
        )
    return tuple(rows)


def _observed_count(name: str, kind: str, stats: Mapping[str, object] | None) -> int | None:
    if not stats:
        return None
    if kind == "unit":
        return {
            "message": stats.get("total_messages"),
            "action": None,
            "block": stats.get("total_messages"),
        }.get(name)
    if name in {"query_terms", "contains_terms", "exclude_text_terms"}:
        return stats.get("total_messages") if name == "query_terms" else None
    return stats.get("total_sessions")


def _status(*, supported: bool, observed: int | None, stale: bool = False) -> str:
    if not supported:
        return "unsupported"
    if stale:
        return "stale_or_degraded"
    if observed is None:
        return "unknown"
    if observed > 0:
        return "supported_and_observed"
    return "supported_but_absent"


def capability_detail_page(
    *,
    search: str | None = None,
    offset: int = 0,
    limit: int = MAX_CAPABILITY_PAGE,
    stats: Mapping[str, object] | None = None,
    readiness: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return one stable, bounded page of executable query declarations."""
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), MAX_CAPABILITY_PAGE))
    needle = search.strip().lower() if search else ""
    rows = _declaration_rows()
    if needle:
        rows = tuple(
            row
            for row in rows
            if needle
            in " ".join(str(row.get(key, "")) for key in ("declaration_id", "name", "meaning", "examples")).lower()
        )
    snapshot = _snapshot_id(stats)
    origins = [
        {
            "origin": spec.origin.value,
            "lifecycle": spec.lifecycle,
            "coverage_refs": list(spec.coverage_refs),
            "authority": "OriginSpec",
        }
        for spec in origin_specs()
    ]
    page: list[dict[str, object]] = []
    for row in rows[offset : offset + limit]:
        item = dict(row)
        observed = _observed_count(str(item["name"]), str(item["kind"]), stats)
        item["observed_count"] = observed
        item["status"] = _status(supported=True, observed=observed)
        item["evidence"] = {
            "authority": ["query declaration", "OriginSpec", "archive stats"],
            "archive_snapshot": snapshot,
            "freshness": "request-current" if stats is not None else "unknown",
            "readiness": dict(readiness or {}),
            "origins": origins,
        }
        item["next_narrowing"] = (
            "Search by declaration name or page with offset; use explain(subject='query') for a concrete plan."
        )
        page.append(item)
    next_offset = offset + len(page)
    return {
        "items": page,
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "search": search,
        "next_offset": next_offset if next_offset < len(rows) else None,
        "snapshot": {
            "id": snapshot,
            "authority": "archive stats",
            "freshness": "request-current" if stats is not None else "unknown",
        },
        "paging": "Repeat explain(subject='capability', search=..., offset=next_offset) until next_offset is null.",
    }


__all__ = ["MAX_CAPABILITY_PAGE", "capability_detail_page"]
