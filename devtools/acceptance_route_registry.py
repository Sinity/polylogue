"""Authoritative route registry for acceptance-contract dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from polylogue.core.json import JSONDecodeError
from polylogue.core.json import loads as json_loads

_REGISTRY_PATH = Path(__file__).parents[1] / "docs" / "plans" / "beads-acceptance-route-registry.json"


class AcceptanceRouteRegistryError(ValueError):
    """Raised when the committed route registry is missing or malformed."""


def load_registry(path: Path = _REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise AcceptanceRouteRegistryError(f"{path}: acceptance route registry is missing")
    try:
        document = json_loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise AcceptanceRouteRegistryError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise AcceptanceRouteRegistryError(f"{path}: unsupported acceptance route registry schema")
    entries = document.get("routes")
    if not isinstance(entries, list):
        raise AcceptanceRouteRegistryError(f"{path}: routes must be a list")
    registry: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise AcceptanceRouteRegistryError(f"{path}: route entries must be objects")
        identifier = entry.get("identifier")
        if not isinstance(identifier, str) or not identifier:
            raise AcceptanceRouteRegistryError(f"{path}: route entry has no identifier")
        if identifier in registry:
            raise AcceptanceRouteRegistryError(f"{path}: duplicate route identifier {identifier!r}")
        registry[identifier] = dict(entry)
    return registry


def resolve_route(
    identifier: object, *, registry: Mapping[str, Mapping[str, Any]] | None = None
) -> Mapping[str, Any] | None:
    if not isinstance(identifier, str):
        return None
    routes = load_registry() if registry is None else registry
    return routes.get(identifier)


def registry_digest(registry: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """Return the digest of the sorted route-entry payload."""
    import hashlib

    from polylogue.core.json import dumps as json_dumps

    routes = load_registry() if registry is None else registry
    payload = json_dumps(
        [routes[identifier] for identifier in sorted(routes)],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
