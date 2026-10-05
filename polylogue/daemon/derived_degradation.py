"""Honest status metadata for a derived tier that cannot be served yet.

Derived tiers have one recovery owner: ordinary daemon convergence.  This
module only projects the read-side consequence of a schema refusal.  It does
not open a writer, create a replacement tier, or invent a second rebuild
route.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from polylogue.core.errors import SchemaRefusalError, SchemaSkewError, SchemaVersionMismatchError


def schema_refusal_lifecycle_action(exc: SchemaRefusalError) -> str | None:
    """The operator route a schema refusal declares, in one read vocabulary.

    ``SchemaVersionMismatchError`` carries its action. A ``SchemaSkew`` is
    translated here: a version stamp newer than this runtime needs a runtime
    upgrade, and any other Index skew (older stamp or stale derived identity)
    is reconverged by the daemon. Other tiers declare no lifecycle route.
    """
    if isinstance(exc, SchemaVersionMismatchError):
        return exc.lifecycle_action
    if isinstance(exc, SchemaSkewError):
        if type(exc.expected) is int and type(exc.found) is int and exc.found > exc.expected:
            return "upgrade_runtime"
        if exc.tier == "index":
            return "rebuild_index"
    return None


def schema_refusal_details(exc: SchemaRefusalError) -> dict[str, object]:
    """Return bounded, machine-readable degradation evidence for ``exc``.

    A refusal happens before a derived read snapshot can be pinned, so the
    completion estimate is deliberately unknown.  In particular, ``None`` is
    not changed into zero progress: no progress measurement was available at
    the refusal boundary.
    """

    if isinstance(exc, SchemaSkewError):
        code = exc.code
        expected_identity = _json_scalar(exc.expected)
        actual_identity = _json_scalar(exc.found)
        expected_version = None
        actual_version = None
        reason = "derived schema identity is stale; daemon convergence owns rebuild"
    elif isinstance(exc, SchemaVersionMismatchError):
        code = "schema_version_mismatch"
        expected_identity = None
        actual_identity = None
        expected_version = exc.expected_version
        actual_version = exc.current_version
        action = exc.lifecycle_action
        if action == "upgrade_runtime":
            reason = "runtime upgrade required before this index can be read"
        elif action == "retry":
            reason = "schema inspection failed transiently; retry the read"
        else:
            reason = "derived schema version is stale; daemon convergence owns rebuild"
    else:  # pragma: no cover - callers restrict this helper to SchemaRefusalError
        code = "schema_refusal"
        expected_identity = None
        actual_identity = None
        expected_version = None
        actual_version = None
        reason = "derived tier refused by the runtime schema contract"

    tier = str(getattr(exc, "tier", "index"))
    completion = {
        "state": "unknown",
        "completed": None,
        "total": None,
        "percent": None,
        "reason": "no convergence progress measurement was available at read refusal",
    }
    action = schema_refusal_lifecycle_action(exc)
    return {
        "code": code,
        "tier": tier,
        "affected_tier": tier,
        "expected_identity": expected_identity,
        "actual_identity": actual_identity,
        "expected_version": expected_version,
        "actual_version": actual_version,
        "state": "upgrade_required" if action == "upgrade_runtime" else "retry" if action == "retry" else "rebuilding",
        "route": "upgrade_runtime"
        if action == "upgrade_runtime"
        else "retry"
        if action == "retry"
        else "daemon_convergence",
        "retryable": action == "retry",
        "progress": completion,
        "completion_estimate": {
            "state": "unknown",
            "eta_s": None,
            "reason": reason,
        },
        "convergence_rate": {
            "state": "unknown",
            "value": None,
            "unit": "derived_units_per_second",
            "reason": "no measured derived-tier convergence sample was available at read refusal",
        },
    }


def _json_scalar(value: Any) -> object:
    """Keep refusal evidence JSON-safe without stringifying known values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def schema_refusal_status_component(details: Mapping[str, object]) -> dict[str, object]:
    """Project refusal evidence into the common readiness component shape."""

    tier = str(details.get("tier") or "derived")
    state = str(details.get("state") or "degraded")
    route = str(details.get("route") or "daemon convergence")
    return {
        "component": f"derived:{tier}",
        "scope": "derived-read",
        "state": "degraded",
        "summary": f"{tier} derived tier is {state}; reads are refused by the schema lifecycle contract",
        "counts": {"progress": details.get("progress")},
        "caveats": ["completion estimate is unknown until convergence records a measured sample"],
        "repair_hint": route.replace("_", " "),
        "evidence_refs": [f"schema_identity:{tier}"],
        "degradation": dict(details),
    }


__all__ = ["schema_refusal_details", "schema_refusal_status_component"]
