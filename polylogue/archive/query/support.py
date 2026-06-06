"""Shared support helpers for immutable session query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.core.enums import Origin
from polylogue.core.sources import provider_from_origin

if TYPE_CHECKING:
    from polylogue.archive.models import Session, SessionSummary


def _origins_as_provider_tokens(origins: tuple[str, ...]) -> list[str] | None:
    """Project origin tokens back to provider tokens for the archive repository
    search leg, which still filters the ``sessions.source_name`` provider column.

    Removed once the source_name→origin reconciliation (#1743 Phase 2) moves that
    SQL onto the ``origin`` column.
    """
    if not origins:
        return None
    return [provider_from_origin(Origin.from_string(token)).value for token in origins]


def session_has_branches(session: Session) -> bool:
    return any(message.branch_index > 0 for message in session.messages)


def session_to_summary(session: Session) -> SessionSummary:
    from polylogue.archive.models import SessionSummary

    return SessionSummary(
        id=session.id,
        origin=session.origin,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        provider_meta=session.provider_meta,
        metadata=session.metadata,
        parent_id=session.parent_id,
        branch_type=session.branch_type,
        message_count=len(session.messages),
        dialogue_count=sum(1 for message in session.messages if message.is_dialogue),
    )


__all__ = [
    "session_has_branches",
    "session_to_summary",
    "_origins_as_provider_tokens",
]
