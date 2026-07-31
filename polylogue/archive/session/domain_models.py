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
from polylogue.core.enums import Origin, SessionKind, TitleSource
from polylogue.core.sources import source_name_to_origin
from polylogue.core.types import SessionId
from polylogue.core.web_urls import canonical_session_url, native_id_from_session_id


def _coerce_origin(v: object) -> Origin:
    if isinstance(v, Origin):
        return v
    return Origin.from_string(source_name_to_origin(v))


def _coerce_title_source(v: object) -> TitleSource | None:
    if v is None or isinstance(v, TitleSource):
        return v
    return TitleSource(str(v))


class SessionSummary(SessionSummaryRuntimeMixin, BaseModel):
    """Lightweight session metadata without messages."""

    id: SessionId
    origin: Origin
    title: str | None = None
    title_source: TitleSource | None = None
    # Specific provenance beyond title_source's coarse strategy label: exact
    # evidence reference plus a 0..1 confidence signal (polylogue-ih67).
    title_ref: str | None = None
    title_confidence: float | None = None
    session_kind: SessionKind = SessionKind.STANDARD
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    working_directories: tuple[str, ...] = ()
    git_branch: str | None = None
    git_repository_url: str | None = None
    provider_project_ref: str | None = None
    # Provider-assigned human-readable session name distinct from the
    # (possibly inferred) title -- e.g. Claude Code's "slug" wire field
    # ("greedy-squishing-hamming"), captured but previously dropped before
    # reaching any domain model (polylogue-cgfy: 1,500 sampled occurrences,
    # the fix for subagent rows displaying "<uuid>:agent-<suffix>" instead
    # of a human name).
    display_name: str | None = None
    parent_id: SessionId | None = None
    branch_type: BranchType | None = None
    message_count: int | None = None
    dialogue_count: int | None = None
    # #1240: tags are sourced from the M2M session_tags table when
    # hydrated through the repository. Empty by default so that legacy
    # constructors keep working.
    tags_m2m: tuple[str, ...] = ()
    # ``with <units>`` projection rows attached post-selection, keyed by unit
    # name → JSON-ready row payloads. Empty unless the query carried a
    # ``with`` clause (#2492).
    attached_units: dict[str, tuple[dict[str, object], ...]] = Field(default_factory=dict)

    @field_validator("origin", mode="before")
    @classmethod
    def coerce_origin(cls, v: object) -> Origin:
        return _coerce_origin(v)

    @field_validator("session_kind", mode="before")
    @classmethod
    def coerce_session_kind(cls, v: object) -> SessionKind:
        return SessionKind.normalize(v)

    @field_validator("title_source", mode="before")
    @classmethod
    def coerce_title_source(cls, v: object) -> TitleSource | None:
        return _coerce_title_source(v)

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
    title_source: TitleSource | None = None
    # Specific provenance beyond title_source's coarse strategy label: exact
    # evidence reference plus a 0..1 confidence signal (polylogue-ih67).
    title_ref: str | None = None
    title_confidence: float | None = None
    session_kind: SessionKind = SessionKind.STANDARD
    messages: MessageCollection
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    working_directories: tuple[str, ...] = ()
    git_branch: str | None = None
    git_repository_url: str | None = None
    provider_project_ref: str | None = None
    # See ``SessionSummary.display_name`` (polylogue-cgfy).
    display_name: str | None = None
    session_events: tuple[SessionEvent, ...] = ()
    parent_id: SessionId | None = None
    branch_type: BranchType | None = None
    # #1240: tags hydrated from session_tags M2M; see SessionSummary.
    tags_m2m: tuple[str, ...] = ()
    # Session-level attachments not linked to a specific message (orphans).
    attachments: list[Attachment] = Field(default_factory=list)
    # ``with <units>`` projection rows attached post-selection, keyed by unit
    # name → JSON-ready row payloads. Empty unless the query carried a
    # ``with`` clause (#2492).
    attached_units: dict[str, tuple[dict[str, object], ...]] = Field(default_factory=dict)
    # polylogue-gt1z: exact provider-reported session cost total (sessions.
    # reported_cost_usd, v49), when the origin's export carries one. None
    # means the origin never reports a session-level total, not a measured
    # zero. Feeds `_session_level_estimate`'s ``status == "exact"`` path.
    reported_cost_usd: float | None = None

    @field_validator("origin", mode="before")
    @classmethod
    def coerce_origin(cls, v: object) -> Origin:
        return _coerce_origin(v)

    @field_validator("session_kind", mode="before")
    @classmethod
    def coerce_session_kind(cls, v: object) -> SessionKind:
        return SessionKind.normalize(v)

    @field_validator("title_source", mode="before")
    @classmethod
    def coerce_title_source(cls, v: object) -> TitleSource | None:
        return _coerce_title_source(v)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def canonical_url(self) -> str | None:
        """Public web URL for web-originated sessions; None for local origins."""
        return canonical_session_url(self.origin, native_id_from_session_id(self.id), self.provider_project_ref)

    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = ["Session", "SessionSummary"]
