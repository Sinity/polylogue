"""Session and summary domain models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from polylogue.archive.attachment.models import Attachment
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.session.domain_runtime import SessionRuntimeMixin
from polylogue.archive.session.events import SessionEvent
from polylogue.archive.session.summary_runtime import SessionSummaryRuntimeMixin
from polylogue.core.enums import Origin, Provider, SessionKind
from polylogue.core.sources import origin_from_provider
from polylogue.core.web_urls import canonical_session_url, native_id_from_session_id
from polylogue.types import SessionId


def _coerce_origin(v: object) -> Origin:
    if isinstance(v, Origin):
        return v
    text = str(v) if v is not None else "unknown"
    try:
        return Origin(text)
    except ValueError:
        return origin_from_provider(Provider.from_string(text))


class SessionSummary(SessionSummaryRuntimeMixin, BaseModel):
    """Lightweight session metadata without messages."""

    id: SessionId
    origin: Origin
    title: str | None = None
    session_kind: SessionKind = SessionKind.STANDARD
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    working_directories: tuple[str, ...] = ()
    git_branch: str | None = None
    git_repository_url: str | None = None
    provider_project_ref: str | None = None
    parent_id: SessionId | None = None
    branch_type: BranchType | None = None
    message_count: int | None = None
    dialogue_count: int | None = None
    # #1240: tags are sourced from the M2M session_tags table when
    # hydrated through the repository. Empty by default so that legacy
    # constructors keep working.
    tags_m2m: tuple[str, ...] = ()

    @field_validator("origin", mode="before")
    @classmethod
    def coerce_origin(cls, v: object) -> Origin:
        return _coerce_origin(v)

    @field_validator("session_kind", mode="before")
    @classmethod
    def coerce_session_kind(cls, v: object) -> SessionKind:
        return SessionKind.normalize(v)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def canonical_url(self) -> str | None:
        """Public web URL for web-originated sessions; None for local origins."""
        return canonical_session_url(self.origin, native_id_from_session_id(self.id), self.provider_project_ref)


class Session(SessionRuntimeMixin, BaseModel):
    """Session with eagerly or lazily materialized message collection."""

    id: SessionId
    origin: Origin
    title: str | None = None
    session_kind: SessionKind = SessionKind.STANDARD
    messages: MessageCollection
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    working_directories: tuple[str, ...] = ()
    git_branch: str | None = None
    git_repository_url: str | None = None
    provider_project_ref: str | None = None
    session_events: tuple[SessionEvent, ...] = ()
    parent_id: SessionId | None = None
    branch_type: BranchType | None = None
    # #1240: tags hydrated from session_tags M2M; see SessionSummary.
    tags_m2m: tuple[str, ...] = ()
    # Session-level attachments not linked to a specific message (orphans).
    attachments: list[Attachment] = Field(default_factory=list)

    @field_validator("origin", mode="before")
    @classmethod
    def coerce_origin(cls, v: object) -> Origin:
        return _coerce_origin(v)

    @field_validator("session_kind", mode="before")
    @classmethod
    def coerce_session_kind(cls, v: object) -> SessionKind:
        return SessionKind.normalize(v)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def canonical_url(self) -> str | None:
        """Public web URL for web-originated sessions; None for local origins."""
        return canonical_session_url(self.origin, native_id_from_session_id(self.id), self.provider_project_ref)

    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = ["Session", "SessionSummary"]
