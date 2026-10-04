"""Bounded, declaration-derived query capability discovery.

The MCP resource is intentionally only an index.  This module is the detail
route used by ``explain(subject="capability")``; it has no independent
catalogue or persistence layer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256

from polylogue.archive.query.declarations import QUERY_CAPABILITY_DECLARATIONS
from polylogue.sources.origin_specs import origin_specs

MAX_CAPABILITY_PAGE = 25


def _snapshot_id(stats: Mapping[str, object] | None, readiness: Mapping[str, object] | None) -> str:
    payload = json.dumps({"counts": stats, "readiness": readiness}, sort_keys=True, default=str).encode()
    return "archive:" + sha256(payload).hexdigest()[:16]


def _declaration_rows() -> tuple[dict[str, object], ...]:
    """Return the served rows straight from the executable query declarations.

    The rows are *derived*, never re-assembled here: adding a query field or a
    terminal unit adds a declaration in
    :mod:`polylogue.archive.query.declarations`, and this route serves it with
    no second edit. Identity (``declaration_id``) has exactly one owner.
    """

    return tuple(dict(declaration.payload) for declaration in QUERY_CAPABILITY_DECLARATIONS)


def _stat_int(stats: Mapping[str, object], key: str) -> int | None:
    value = stats.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _observed_count(name: str, kind: str, stats: Mapping[str, object] | None) -> int | None:
    if stats is None:
        return None
    # Session totals cannot witness a field's values, and message totals do
    # not witness blocks/actions. Only the matching canonical unit is measured.
    return _stat_int(stats, "total_messages") if kind == "unit" and name == "message" else None


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
            in " ".join(
                str(row.get(key, ""))
                for key in ("declaration_id", "name", "meaning", "binding", "operators", "fields", "examples")
            ).lower()
        )
    snapshot = _snapshot_id(stats, readiness)
    origins = [
        {
            "origin": spec.origin.value,
            "lifecycle": spec.lifecycle,
            "coverage_refs": list(spec.coverage_refs),
            "authority": "OriginSpec",
        }
        for spec in origin_specs()
        if spec.public_filter
    ]
    readiness_state = str((readiness or {}).get("state", "unknown"))
    stale = readiness_state in {"stale", "rebuilding", "blocked", "degraded", "poisoned"}
    freshness = "unknown"
    if stats is not None:
        freshness = "stale_or_degraded" if stale else "request-current" if readiness_state == "ready" else "unknown"
    page: list[dict[str, object]] = []
    for row in rows[offset : offset + limit]:
        item = dict(row)
        observed = _observed_count(str(item["name"]), str(item["kind"]), stats)
        item["observed_count"] = observed
        item["status"] = _status(
            supported=True, observed=observed if freshness != "unknown" else None, stale=stale and stats is not None
        )
        item["evidence"] = {
            "authority": ["query declaration", "archive counts"],
            "archive_snapshot": snapshot,
            "freshness": freshness,
            "readiness_state": readiness_state,
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
            "authority": "archive counts and query readiness",
            "freshness": freshness,
        },
        "evidence": {
            "authority": ["OriginSpec", "query binding", "convergence debt"],
            "origins": origins,
            "readiness": dict(readiness or {}),
        },
        "paging": "Repeat explain(subject='capability', search=..., offset=next_offset) until next_offset is null.",
    }


__all__ = ["MAX_CAPABILITY_PAGE", "capability_detail_page"]
