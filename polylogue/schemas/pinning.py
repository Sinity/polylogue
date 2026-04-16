"""Human-reviewed semantic annotation pinning.

Pins are binary decisions: a human confirms an annotation is correct, or
rejects it. No confidence scores — pinning replaces confidence with certainty.

Pins are stored per-provider as ``pins.json`` alongside the schema package
catalog. On schema re-inference, pinned decisions survive: confirmed
annotations are preserved, rejected annotations are suppressed.

Usage::

    pins = load_pins(Provider.CLAUDE_CODE)
    paths = resolve_pinned_paths(schema, pins)
    # paths["message_role"] = ".type" (if pinned)
    # paths["message_body"] = None (if no pinned path for this role)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from polylogue.types import Provider

logger = logging.getLogger(__name__)

# Semantic roles that can be pinned
PINNABLE_ROLES = frozenset(
    {
        "message_role",
        "message_body",
        "message_timestamp",
        "message_container",
        "conversation_title",
    }
)


@dataclass(frozen=True, slots=True)
class PinDecision:
    """A human review decision about a semantic annotation."""

    path: str
    role: str
    action: str  # "confirm" or "reject"
    reason: str = ""

    def __post_init__(self) -> None:
        if self.action not in ("confirm", "reject"):
            raise ValueError(f"Pin action must be 'confirm' or 'reject', got {self.action!r}")
        if self.role not in PINNABLE_ROLES:
            raise ValueError(f"Role {self.role!r} is not pinnable. Valid: {sorted(PINNABLE_ROLES)}")


@dataclass
class PinSet:
    """Collection of pin decisions for one provider."""

    provider: str
    pins: list[PinDecision] = field(default_factory=list)

    def confirmed_path(self, role: str) -> str | None:
        """Return the pinned path for a role, or None if no confirmed pin."""
        for pin in self.pins:
            if pin.role == role and pin.action == "confirm":
                return pin.path
        return None

    def is_rejected(self, path: str, role: str) -> bool:
        """Check if a specific path+role combination was rejected."""
        return any(pin.path == path and pin.role == role and pin.action == "reject" for pin in self.pins)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "pins": [asdict(pin) for pin in self.pins],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PinSet:
        pins = [
            PinDecision(
                path=p["path"],
                role=p["role"],
                action=p["action"],
                reason=p.get("reason", ""),
            )
            for p in data.get("pins", [])
        ]
        return cls(provider=data.get("provider", ""), pins=pins)


# -------------------------------------------------------------------
# Storage
# -------------------------------------------------------------------


def _pins_path(provider: Provider | str) -> Path:
    """Path to the pins.json for a provider package."""
    from polylogue.schemas.runtime_registry import SCHEMA_DIR

    provider_str = str(provider)
    return SCHEMA_DIR / provider_str / "pins.json"


def load_pins(provider: Provider | str) -> PinSet:
    """Load pin decisions for a provider. Returns empty PinSet if no pins file."""
    path = _pins_path(provider)
    if not path.exists():
        return PinSet(provider=str(provider))
    try:
        data = json.loads(path.read_text())
        pin_set = PinSet.from_dict(data)
        pin_set.provider = str(provider)
        return pin_set
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to load pins from %s: %s", path, exc)
        return PinSet(provider=str(provider))


def save_pins(provider: Provider | str, pin_set: PinSet) -> None:
    """Save pin decisions for a provider."""
    path = _pins_path(provider)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pin_set.to_dict(), indent=2) + "\n")


# -------------------------------------------------------------------
# Schema integration
# -------------------------------------------------------------------


def resolve_pinned_paths(
    schema: dict[str, Any],
    pins: PinSet,
) -> dict[str, str | None]:
    """Resolve which schema paths to use for each semantic role.

    Returns a dict mapping role names to JSON paths. Only confirmed pins
    produce entries. If no pin exists for a role, it maps to None (and
    the extractor falls back to well-known-name heuristics).
    """
    result: dict[str, str | None] = dict.fromkeys(PINNABLE_ROLES)

    # First: apply confirmed pins directly
    for role in PINNABLE_ROLES:
        confirmed = pins.confirmed_path(role)
        if confirmed is not None:
            result[role] = confirmed

    return result


def apply_pins_to_schema(
    schema: dict[str, Any],
    pins: PinSet,
) -> dict[str, Any]:
    """Apply pin decisions to a schema in-place (for re-inference).

    - Confirmed annotations get ``x-polylogue-pinned: true``
    - Rejected annotations get their ``x-polylogue-semantic-role`` removed

    Returns the modified schema (mutated in place for efficiency).
    """
    _apply_pins_recursive(schema, "", pins)
    return schema


def _apply_pins_recursive(
    node: Any,
    path: str,
    pins: PinSet,
) -> None:
    """Recursively walk schema and apply pin decisions."""
    if not isinstance(node, dict):
        return

    role = node.get("x-polylogue-semantic-role")
    if role and pins.is_rejected(path or "$", role):
        # Remove rejected annotation
        node.pop("x-polylogue-semantic-role", None)
        node.pop("x-polylogue-evidence", None)
        node["x-polylogue-rejected"] = True
    elif role and pins.confirmed_path(role) == (path or "$"):
        node["x-polylogue-pinned"] = True

    # Recurse into schema structure
    props = node.get("properties")
    if isinstance(props, dict):
        for key, value in props.items():
            _apply_pins_recursive(value, f"{path}.{key}", pins)

    items = node.get("items")
    if isinstance(items, dict):
        _apply_pins_recursive(items, f"{path}[]", pins)

    add_props = node.get("additionalProperties")
    if isinstance(add_props, dict):
        _apply_pins_recursive(add_props, f"{path}.*", pins)

    for combiner in ("anyOf", "oneOf", "allOf"):
        variants = node.get(combiner)
        if isinstance(variants, list):
            for i, variant in enumerate(variants):
                _apply_pins_recursive(variant, f"{path}.{combiner}[{i}]", pins)


__all__ = [
    "PINNABLE_ROLES",
    "PinDecision",
    "PinSet",
    "apply_pins_to_schema",
    "load_pins",
    "resolve_pinned_paths",
    "save_pins",
]
