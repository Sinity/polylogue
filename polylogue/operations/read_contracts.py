"""Request/result models for the declared CLI-serving read operations.

These models deliberately live outside :mod:`polylogue.surfaces.payloads`.
That module is inside the derived-schema identity closure, so declaring a new
operation there would move the archive's schema identity and force a
reconvergence for a change that adds no derived semantics at all.  Keep this
module free of imports that reach the lowering, materializer or replay-routing
graphs; ``devtools schema closure`` is the check.

The models are self-contained for the same reason: importing the private
payload bases from :mod:`polylogue.operations.daemon_protocol` would be a
cycle, since that module imports these declarations.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ReadRequest(BaseModel):
    """Base for a declared read request payload."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _ReadResult(BaseModel):
    """Base for a declared read result; envelopes own authority metadata."""

    model_config = ConfigDict(extra="allow", frozen=True, strict=True)


AggregateMode = Literal["count", "stats", "stats_by"]


class QueryAggregateRequest(_ReadRequest):
    """One aggregate over the same selection vocabulary as ``cli.query``."""

    mode: AggregateMode
    group_by: str | None = None
    params: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def group_by_belongs_to_grouped_mode(self) -> QueryAggregateRequest:
        if self.mode == "stats_by" and not (self.group_by or "").strip():
            raise ValueError("stats_by requires a group_by field")
        if self.mode != "stats_by" and self.group_by is not None:
            raise ValueError("group_by is only meaningful for stats_by")
        return self


class QueryAggregateResult(_ReadResult):
    """Exactly one aggregate body, named by the mode that produced it."""

    outcome: dict[str, object]
    mode: AggregateMode
    count: int | None = Field(default=None, ge=0)
    stats: dict[str, object] | None = None
    group_by: str | None = None
    groups: dict[str, int] | None = None

    @model_validator(mode="after")
    def body_matches_mode(self) -> QueryAggregateResult:
        bodies = {
            "count": self.count is not None,
            "stats": self.stats is not None,
            "stats_by": self.groups is not None,
        }
        if not bodies[self.mode]:
            raise ValueError(f"{self.mode} result is missing its aggregate body")
        if any(present for mode, present in bodies.items() if mode != self.mode):
            raise ValueError("aggregate result carries a body from another mode")
        return self


#: What a ``session.read`` window is read *of*.
#:
#: ``transcript`` is the message window every read view was built on.  The
#: other kinds are per-session evidence relations that ride the same exact
#: reference but are not messages and have no query-grammar unit of their own
#: (design D3, ``cli/read_view_registry.py``): reading them through
#: ``session.read`` is what keeps a read view from opening an archive itself.
SessionReadKind = Literal["transcript", "hooks"]

#: Evidence kinds are bounded per-session read models, answered whole.  A kind
#: that later needs paging graduates to the windowed contract rather than
#: quietly truncating, which is why the result validator refuses a partial
#: evidence body instead of allowing one.
_WHOLE_EVIDENCE_KINDS: frozenset[str] = frozenset({"hooks"})


class SessionReadRequest(_ReadRequest):
    """One bounded read for an exact session reference.

    For ``kind="transcript"``, ``limit`` is a hard window, not a hint: a full
    transcript can exceed the 8 MiB bound on a single operation result, so the
    caller loops windows and the handler refuses a window it cannot deliver
    whole.  Evidence kinds are bounded by construction and ignore the window.
    """

    ref: str = Field(min_length=1)
    kind: SessionReadKind = "transcript"
    limit: int = Field(default=200, ge=1, le=2000)
    offset: int = Field(default=0, ge=0)
    projection: dict[str, object] | None = None
    continuation: str | None = None

    @model_validator(mode="after")
    def only_a_transcript_window_continues(self) -> SessionReadRequest:
        if self.kind != "transcript" and self.continuation is not None:
            raise ValueError(f"{self.kind} is answered whole and issues no continuation")
        return self


class SessionReadResult(_ReadResult):
    """One window plus the snapshot-bound token for the next one, if any."""

    outcome: dict[str, object]
    session: dict[str, object]
    session_id: str = Field(min_length=1)
    kind: SessionReadKind = "transcript"
    evidence: dict[str, object] | None = None
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    next_offset: int | None = Field(default=None, ge=0)
    continuation: str | None = None
    complete: bool

    @model_validator(mode="after")
    def continuation_accompanies_a_next_window(self) -> SessionReadResult:
        if (self.continuation is None) != (self.next_offset is None):
            raise ValueError("a next window and its continuation are issued together")
        if self.complete != (self.next_offset is None):
            raise ValueError("completeness must agree with the presence of a next window")
        return self

    @model_validator(mode="after")
    def the_body_matches_the_kind_that_was_read(self) -> SessionReadResult:
        if self.kind == "transcript":
            if self.evidence is not None:
                raise ValueError("a transcript window carries no evidence body")
            return self
        if self.evidence is None:
            raise ValueError(f"{self.kind} result is missing its evidence body")
        if self.kind in _WHOLE_EVIDENCE_KINDS and not self.complete:
            raise ValueError(f"{self.kind} is answered whole and cannot report a partial body")
        return self


class SessionReferenceRequest(_ReadRequest):
    """A bare ``from <ref>`` reference operand resolved against durable state."""

    expression: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=0)


class SessionReferenceResult(_ReadResult):
    """Resolved members of one reference operand, with its resolution lineage."""

    outcome: dict[str, object]
    source: str
    grain: str
    lineage: list[str]
    member_count: int = Field(ge=0)
    members: list[str]
    truncated: bool

    @model_validator(mode="after")
    def truncation_is_observable(self) -> SessionReferenceResult:
        if self.truncated != (len(self.members) < self.member_count):
            raise ValueError("truncation flag disagrees with the returned member count")
        return self


__all__ = [
    "AggregateMode",
    "QueryAggregateRequest",
    "QueryAggregateResult",
    "SessionReadKind",
    "SessionReadRequest",
    "SessionReadResult",
    "SessionReferenceRequest",
    "SessionReferenceResult",
]
