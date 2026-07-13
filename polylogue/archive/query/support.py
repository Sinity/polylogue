"""Shared support helpers for immutable session query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.core.enums import Origin

if TYPE_CHECKING:
    from polylogue.archive.models import Session, SessionSummary


def _canonical_origins(origins: tuple[str, ...]) -> list[str] | None:
    """Normalize an optional origin scope for archive search queries."""
    if not origins:
        return None
    return [Origin(token).value for token in origins]


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
        metadata=session.metadata,
        working_directories=session.working_directories,
        git_branch=session.git_branch,
        git_repository_url=session.git_repository_url,
        provider_project_ref=session.provider_project_ref,
        parent_id=session.parent_id,
        branch_type=session.branch_type,
        message_count=len(session.messages),
        dialogue_count=sum(1 for message in session.messages if message.is_dialogue),
    )


__all__ = [
    "session_has_branches",
    "session_to_summary",
    "_canonical_origins",
]
