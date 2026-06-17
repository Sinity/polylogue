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
  ``session_id``. Only ``message`` and ``block`` do.
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

TARGET_SESSION: Final = "session"
TARGET_MESSAGE: Final = "message"
TARGET_WORK_EVENT: Final = "work_event"
TARGET_THREAD: Final = "thread"
TARGET_BLOCK: Final = "block"
TARGET_ATTACHMENT: Final = "attachment"
TARGET_PASTE_SPAN: Final = "paste_span"

MARK_STAR: Final = "star"
MARK_PIN: Final = "pin"
MARK_ARCHIVE: Final = "archive"


@dataclass(frozen=True)
class TargetKind:
    """Declarative entry for a single user-state target kind."""

    name: str
    unit: str
    requires_message_id: bool
    identity_template: str


_KINDS: Final[tuple[TargetKind, ...]] = (
    TargetKind(
        name=TARGET_SESSION,
        unit="session_id",
        requires_message_id=False,
        identity_template="session:{target_id}",
    ),
    TargetKind(
        name=TARGET_MESSAGE,
        unit="message_id",
        requires_message_id=True,
        identity_template="message:{session_id}:{target_id}",
    ),
    TargetKind(
        name=TARGET_WORK_EVENT,
        unit="event_id from session_work_events",
        requires_message_id=False,
        identity_template="work_event:{session_id}:{target_id}",
    ),
    TargetKind(
        name=TARGET_THREAD,
        unit="thread_id from threads",
        requires_message_id=False,
        identity_template="thread:{target_id}",
    ),
    TargetKind(
        name=TARGET_BLOCK,
        unit="message_id:block_index",
        requires_message_id=True,
        identity_template="block:{session_id}:{target_id}",
    ),
    TargetKind(
        name=TARGET_ATTACHMENT,
        unit="attachment_id",
        requires_message_id=False,
        identity_template="attachment:{session_id}:{target_id}",
    ),
    TargetKind(
        name=TARGET_PASTE_SPAN,
        unit="paste_id",
        requires_message_id=False,
        identity_template="paste_span:{session_id}:{target_id}",
    ),
)

KINDS_BY_NAME: Final[dict[str, TargetKind]] = {kind.name: kind for kind in _KINDS}

TARGET_KIND_NAMES: Final[tuple[str, ...]] = tuple(kind.name for kind in _KINDS)
"""All supported user-state target kind names, in registry order."""

STORAGE_TARGET_KIND_NAMES: Final[tuple[str, ...]] = (
    TARGET_SESSION,
    TARGET_MESSAGE,
    TARGET_BLOCK,
    TARGET_ATTACHMENT,
    TARGET_PASTE_SPAN,
    TARGET_WORK_EVENT,
    "phase",
    TARGET_THREAD,
)
"""Target tokens admitted by the user-tier CHECK constraints.

``phase`` remains storage-only until there is a stable public target identity
for it.
"""

MARK_TYPE_NAMES: Final[tuple[str, ...]] = (MARK_STAR, MARK_PIN, MARK_ARCHIVE)
"""All supported reader mark types, in public wire order."""


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


def validate_target_kind(kind: str) -> str:
    """Return ``kind`` or raise ``ValueError`` for unsupported target kinds."""

    if is_supported(kind):
        return kind
    raise ValueError(f"target_type must be one of: {', '.join(TARGET_KIND_NAMES)}")


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


def is_mark_type_supported(mark_type: str) -> bool:
    """Return ``True`` if ``mark_type`` is a supported reader mark token."""

    return mark_type in MARK_TYPE_NAMES


def validate_mark_type(mark_type: str) -> str:
    """Return ``mark_type`` or raise ``ValueError`` for unsupported values."""

    if is_mark_type_supported(mark_type):
        return mark_type
    raise ValueError(f"mark_type must be one of: {', '.join(MARK_TYPE_NAMES)}")


__all__ = [
    "KINDS_BY_NAME",
    "MARK_ARCHIVE",
    "MARK_PIN",
    "MARK_STAR",
    "MARK_TYPE_NAMES",
    "STORAGE_TARGET_KIND_NAMES",
    "TARGET_KIND_NAMES",
    "TARGET_ATTACHMENT",
    "TARGET_BLOCK",
    "TARGET_MESSAGE",
    "TARGET_PASTE_SPAN",
    "TARGET_SESSION",
    "TARGET_THREAD",
    "TARGET_WORK_EVENT",
    "TargetKind",
    "get_kind",
    "identity_key",
    "is_mark_type_supported",
    "is_supported",
    "validate_mark_type",
    "validate_target_kind",
]
