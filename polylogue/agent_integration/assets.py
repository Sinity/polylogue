"""Read packaged agent assets through ``importlib.resources``."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Final

from polylogue.agent_integration.spec import ASSET_VERSION, CAPABILITY_FAMILIES, RECIPES

_AGENT_ASSET_PACKAGE: Final = "polylogue.agent_integration.data"
TEXT_ASSETS: Final[tuple[str, ...]] = ("standing-manual.md", "deep-reference.md")
DATA_ASSETS: Final[tuple[str, ...]] = (
    "recipes.json",
    "integration-spec.json",
    "tool-contracts.json",
    "integration-manifest.json",
)
ALL_ASSETS: Final[tuple[str, ...]] = (*TEXT_ASSETS, *DATA_ASSETS)


def read_agent_asset(name: str) -> str:
    """Return one packaged UTF-8 agent asset."""
    if name not in ALL_ASSETS:
        raise ValueError(f"unknown agent asset: {name}")
    return files(_AGENT_ASSET_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def read_agent_json(name: str) -> dict[str, object]:
    """Return one packaged JSON asset."""
    if name not in DATA_ASSETS:
        raise ValueError(f"not an agent JSON asset: {name}")
    value = json.loads(read_agent_asset(name))
    if not isinstance(value, dict):
        raise ValueError(f"agent asset {name!r} is not a JSON object")
    return value


def agent_asset_digest() -> str:
    """Return a deterministic digest of every packaged agent asset."""
    digest = hashlib.sha256()
    for name in ALL_ASSETS:
        payload = read_agent_asset(name).encode()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def agent_asset_metadata() -> dict[str, object]:
    """Return measured package metadata used by status and evidence reports."""
    sizes = {name: len(read_agent_asset(name).encode()) for name in ALL_ASSETS}
    digest = agent_asset_digest()
    return {
        "content_version": ASSET_VERSION,
        "asset_digest": digest,
        "cache_key": f"polylogue-agent-{ASSET_VERSION}-{digest[:16]}",
        "asset_bytes": sizes,
        "standing_manual_bytes": sizes["standing-manual.md"],
        "deep_reference_bytes": sizes["deep-reference.md"],
        "capability_family_count": len(CAPABILITY_FAMILIES),
        "recipe_count": len(RECIPES),
    }


__all__ = [
    "ALL_ASSETS",
    "ASSET_VERSION",
    "DATA_ASSETS",
    "TEXT_ASSETS",
    "agent_asset_digest",
    "agent_asset_metadata",
    "read_agent_asset",
    "read_agent_json",
]
