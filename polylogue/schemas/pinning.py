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
from typing import Literal, TypeAlias, cast

from polylogue.schemas.json_types import JSONDocument, json_document, json_document_list
from polylogue.types import Provider

logger = logging.getLogger(__name__)

PinAction: TypeAlias = Literal["confirm", "reject"]
PinnableRole: TypeAlias = Literal[
    "message_role",
    "message_body",
    "message_timestamp",
    "message_container",
    "conversation_title",
]

# Semantic roles that can be pinned
PINNABLE_ROLES: frozenset[PinnableRole] = frozenset(
    {
        "message_role",
        "message_body",
        "message_timestamp",
        "message_container",
        "conversation_title",
    }
)


def _required_string(record: JSONDocument, key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Pin field {key!r} must be a string")
    return value


def _optional_string(record: JSONDocument, key: str, default: str = "") -> str:
    value = record.get(key)
    return value if isinstance(value, str) else default


def _pin_action(value: object) -> PinAction:
    if value in ("confirm", "reject"):
        return cast(PinAction, value)
    raise ValueError(f"Pin action must be 'confirm' or 'reject', got {value!r}")


def _pin_role(value: object) -> PinnableRole:
    if value in PINNABLE_ROLES:
        return cast(PinnableRole, value)
    raise ValueError(f"Role {value!r} is not pinnable. Valid: {sorted(PINNABLE_ROLES)}")


@dataclass(frozen=True, slots=True)
class PinDecision:
    """A human review decision about a semantic annotation."""

    path: str
    role: PinnableRole
    action: PinAction
    reason: str = ""

    def __post_init__(self) -> None:
        _pin_action(self.action)
        _pin_role(self.role)

    def to_dict(self) -> JSONDocument:
        return json_document(asdict(self))

    @classmethod
    def from_dict(cls, data: object) -> PinDecision:
        record = json_document(data)
        return cls(
            path=_required_string(record, "path"),
            role=_pin_role(record.get("role")),
            action=_pin_action(record.get("action")),
            reason=_optional_string(record, "reason"),
        )


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

    def to_dict(self) -> JSONDocument:
        return json_document({"provider": self.provider, "pins": [pin.to_dict() for pin in self.pins]})

    @classmethod
    def from_dict(cls, data: object) -> PinSet:
        payload = json_document(data)
        return cls(
            provider=_optional_string(payload, "provider"),
            pins=[PinDecision.from_dict(pin) for pin in json_document_list(payload.get("pins"))],
        )


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
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
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
    schema: JSONDocument,
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
    schema: JSONDocument,
    pins: PinSet,
) -> JSONDocument:
    """Apply pin decisions to a schema in-place (for re-inference).

    - Confirmed annotations get ``x-polylogue-pinned: true``
    - Rejected annotations get their ``x-polylogue-semantic-role`` removed

    Returns the modified schema (mutated in place for efficiency).
    """
    _apply_pins_recursive(schema, "", pins)
    return schema


def _apply_pins_recursive(
    node: object,
    path: str,
    pins: PinSet,
) -> None:
    """Recursively walk schema and apply pin decisions."""
    if not isinstance(node, dict):
        return

    role = node.get("x-polylogue-semantic-role")
    if isinstance(role, str) and pins.is_rejected(path or "$", role):
        # Remove rejected annotation
        node.pop("x-polylogue-semantic-role", None)
        node.pop("x-polylogue-evidence", None)
        node["x-polylogue-rejected"] = True
    elif isinstance(role, str) and pins.confirmed_path(role) == (path or "$"):
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
