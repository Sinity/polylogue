"""Canonical target kinds for reader user-state (#1113).

User-state targets identify the unit that a mark, annotation, recall-pack
item, or workspace open-target points at. The original #867 implementation
admitted only ``session`` and ``message``; this module is the
authoritative registry of the additional kinds added by #1113.

Each kind declares:

- ``name``: the wire token used in CHECK constraints and on every surface
  (CLI, MCP, daemon HTTP, recall-pack items).
- ``unit``: human-readable role of the ``target_id`` for that kind.
- ``requires_message_id``: whether the kind stores ``message_id`` alongside
  ``session_id``. Only ``message`` and ``content_block`` do.
- ``identity_template``: format string for ``identity_key`` in recall packs
  and workspace open-targets; receives keyword args ``session_id``,
  ``target_id``, ``message_id``.

Kinds that still require external work to gain stable identity (currently
``topology_edge``, dependent on #866) are not listed here and remain
``unsupported`` at the recall-pack/workspace surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class TargetKind:
    """Declarative entry for a single user-state target kind."""

    name: str
    unit: str
    requires_message_id: bool
    identity_template: str


_KINDS: Final[tuple[TargetKind, ...]] = (
    TargetKind(
        name="session",
        unit="session_id",
        requires_message_id=False,
        identity_template="session:{target_id}",
    ),
    TargetKind(
        name="message",
        unit="message_id",
        requires_message_id=True,
        identity_template="message:{session_id}:{target_id}",
    ),
    TargetKind(
        name="work_event",
        unit="event_id from session_work_events",
        requires_message_id=False,
        identity_template="work_event:{session_id}:{target_id}",
    ),
    TargetKind(
        name="thread",
        unit="thread_id from threads",
        requires_message_id=False,
        identity_template="thread:{target_id}",
    ),
    TargetKind(
        name="content_block",
        unit="message_id:block_index",
        requires_message_id=True,
        identity_template="content_block:{session_id}:{target_id}",
    ),
    TargetKind(
        name="attachment",
        unit="attachment_id",
        requires_message_id=False,
        identity_template="attachment:{session_id}:{target_id}",
    ),
    TargetKind(
        name="paste_span",
        unit="paste_id",
        requires_message_id=False,
        identity_template="paste_span:{session_id}:{target_id}",
    ),
)

KINDS_BY_NAME: Final[dict[str, TargetKind]] = {kind.name: kind for kind in _KINDS}

TARGET_KIND_NAMES: Final[tuple[str, ...]] = tuple(kind.name for kind in _KINDS)
"""All supported user-state target kind names, in registry order."""


def is_supported(kind: str) -> bool:
    """Return ``True`` if ``kind`` is a registered user-state target kind."""

    return kind in KINDS_BY_NAME


def get_kind(name: str) -> TargetKind:
    """Look up a kind by name, raising ``ValueError`` for unknown names."""

    try:
        return KINDS_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown user-state target kind: {name!r}. Supported: {', '.join(TARGET_KIND_NAMES)}"
        ) from exc


def identity_key(
    kind_name: str,
    *,
    session_id: str,
    target_id: str,
    message_id: str | None = None,
) -> str:
    """Render the canonical ``identity_key`` for a recall-pack/workspace item."""

    kind = get_kind(kind_name)
    return kind.identity_template.format(
        session_id=session_id,
        target_id=target_id,
        message_id=message_id or "",
    )


__all__ = [
    "KINDS_BY_NAME",
    "TARGET_KIND_NAMES",
    "TargetKind",
    "get_kind",
    "identity_key",
    "is_supported",
]
