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
#: ``transcript`` and ``messages`` are two row vocabularies over the *same*
#: message window: ``transcript`` projects the archive's own identity fields
#: for a composed reading surface, while ``messages`` projects the full
#: message-row envelope the ``read --view messages`` document is made of
#: (token counts, model, timestamps, attachment refs).  They are declared as
#: two kinds rather than one projection flag because a continuation minted for
#: one row vocabulary must not resume the other.
#:
#: Every other kind is a per-session evidence relation that rides the same
#: exact reference but is not a message window and has no query-grammar unit
#: of its own (design D3, ``cli/read_view_registry.py``): reading them through
#: ``session.read`` is what keeps a read view from opening an archive itself.
SessionReadKind = Literal[
    "transcript",
    "messages",
    "hooks",
    "file-edits",
    "agent-policies",
    "web-content",
    "events",
    "raw",
]

#: Kinds that answer a bounded ``[offset, offset + limit)`` message window and
#: therefore take window coordinates and issue continuations.
WINDOWED_SESSION_READ_KINDS: frozenset[str] = frozenset({"transcript", "messages"})

#: Evidence kinds are bounded per-session read models, answered whole.  A kind
#: that later needs paging graduates to the windowed contract rather than
#: quietly truncating, which is why the result validator refuses a partial
#: evidence body instead of allowing one.
_WHOLE_EVIDENCE_KINDS: frozenset[str] = frozenset({"hooks", "file-edits", "agent-policies", "web-content"})

#: Evidence relations that are *larger than one answer*: they carry a row
#: bound that is declared by the caller and **reported back**, plus a
#: continuation for the next page.  This is the graduation path the whole
#: contract names -- the third body vocabulary, distinct from both the message
#: window (``messages``) and a relation answered whole (``evidence``).
#:
#: It exists because ``events`` accepted ``--limit`` and reported the
#: *truncated* row count as its ``total``: a caller could not tell a whole
#: relation from a clipped one, and lowering it onto a whole-evidence kind
#: would have made ``session.read`` report ``complete`` for a partial body.
#: A bound must be declared and reported, never a silent truncation, so the
#: body below carries ``total`` (the relation's own row count), ``returned``
#: (what this page holds) and ``complete``, and
#: :class:`EvidenceWindowBody` refuses any combination of the three that
#: would let a clipped body read as a whole one (polylogue-r3cuz).
WINDOWED_EVIDENCE_KINDS: frozenset[str] = frozenset({"events", "raw"})

#: Kinds that issue and accept a continuation.  Two families mint tokens here
#: and they are deliberately not interchangeable: the message window's
#: ``session-owner-v1`` projection (``operations/transcript_window.py``) and
#: the per-relation evidence-window projections
#: (``operations/evidence_window.py``).  Each refuses the other's token by
#: name rather than resuming a window the caller never asked for.
CONTINUABLE_SESSION_READ_KINDS: frozenset[str] = WINDOWED_SESSION_READ_KINDS | WINDOWED_EVIDENCE_KINDS


#: Kinds that accept ``around`` -- a *message* naming its own window instead
#: of a coordinate naming it.  It is sugar over ``offset``: the handler
#: resolves the message's ordinal index, aligns it onto the caller's declared
#: ``limit``, and reports the resolved coordinate back, so an ``around`` window
#: is byte-for-byte the window the same caller gets by asking for the offset it
#: reports.  Only the ``messages`` kind carries it, matching the HTTP messages
#: read-view capability that declared it first (polylogue-i5vqc).
ANCHORED_SESSION_READ_KINDS: frozenset[str] = frozenset({"messages"})


class EvidenceWindowBody(BaseModel):
    """One bounded page of a per-session evidence relation.

    The third body vocabulary on :class:`SessionReadResult`.  Its whole
    purpose is that a bound is *observable in the payload*: ``total`` is the
    relation's own row count, ``returned`` is what this page holds, and the
    validators below make "fewer rows than the relation holds, reported as
    complete" unrepresentable rather than merely discouraged.

    Surfaces that render the evidence body alone -- the CLI evidence read
    views print exactly this document -- therefore cannot lose the bound on
    the way out, which is why it is restated here rather than left to the
    envelope that wraps it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    #: The ``session.read`` kind these rows are rows *of*.  Two relations share
    #: this body, so a page that did not name its relation could be rendered
    #: under the wrong one.
    relation: str = Field(min_length=1)
    rows: list[dict[str, object]]
    #: The relation's own row count, *not* the returned count.
    total: int = Field(ge=0)
    #: How many rows this page actually carries.
    returned: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    next_offset: int | None = Field(default=None, ge=0)
    continuation: str | None = None
    complete: bool

    @model_validator(mode="after")
    def the_reported_count_is_the_delivered_count(self) -> EvidenceWindowBody:
        if self.returned != len(self.rows):
            raise ValueError("returned disagrees with the number of rows delivered")
        return self

    @model_validator(mode="after")
    def completeness_is_decided_by_the_rows_still_owed(self) -> EvidenceWindowBody:
        """A clipped body cannot claim to be whole.

        This is the validator the ``events`` view could not satisfy before the
        windowed-evidence contract existed: it reported the truncated count as
        ``total``, which made ``offset + returned >= total`` trivially true
        and every clipped read indistinguishable from a whole one.
        """

        delivered_through = self.offset + self.returned
        if self.complete != (delivered_through >= self.total):
            raise ValueError("a body that does not reach the relation's total is not complete")
        if (self.next_offset is None) != self.complete:
            raise ValueError("a next page and completeness are two spellings of one fact")
        if self.next_offset is not None and self.next_offset != delivered_through:
            raise ValueError("the next page must start where this one ended")
        if (self.continuation is None) != (self.next_offset is None):
            raise ValueError("a next page and its continuation are issued together")
        return self


class SessionReadRequest(_ReadRequest):
    """One bounded read for an exact session reference.

    For a windowed kind, ``limit`` is a hard window, not a hint: a full
    transcript can exceed the 8 MiB bound on a single operation result, so the
    caller loops windows and the handler refuses a window it cannot deliver
    whole.  Evidence kinds are bounded by construction and ignore the window.
    """

    ref: str = Field(min_length=1)
    kind: SessionReadKind = "transcript"
    limit: int = Field(default=200, ge=1, le=2000)
    offset: int = Field(default=0, ge=0)
    #: A message reference whose window is wanted.  Declared here rather than
    #: only on the HTTP route so the CLI, the MCP ``read`` tool and the Python
    #: API can ask the same question; before this they could only ask by
    #: coordinate, which made a deep link a page walk on three of the four
    #: surfaces (polylogue-idrej).
    around: str | None = Field(default=None, min_length=1)
    projection: dict[str, object] | None = None
    continuation: str | None = None

    @model_validator(mode="after")
    def only_a_continuable_kind_continues(self) -> SessionReadRequest:
        if self.kind not in CONTINUABLE_SESSION_READ_KINDS and self.continuation is not None:
            raise ValueError(f"{self.kind} is answered whole and issues no continuation")
        return self

    @model_validator(mode="after")
    def an_anchor_names_one_window(self) -> SessionReadRequest:
        if self.around is None:
            return self
        if self.kind not in ANCHORED_SESSION_READ_KINDS:
            raise ValueError(f"{self.kind} does not serve a window around a message")
        if self.continuation is not None:
            # An anchor asks the handler to *decide* the offset; a continuation
            # already carries one.  Honouring either silently would answer a
            # window the caller did not ask for.
            raise ValueError("around and continuation name two different windows")
        if self.offset:
            raise ValueError("around and offset name two different windows")
        return self


class SessionReadResult(_ReadResult):
    """One window plus the snapshot-bound token for the next one, if any."""

    outcome: dict[str, object]
    session: dict[str, object]
    session_id: str = Field(min_length=1)
    kind: SessionReadKind = "transcript"
    evidence: dict[str, object] | None = None
    #: The ``messages`` kind's window, as full message-row envelope documents.
    #: A transcript window carries its rows inside ``session`` instead, in the
    #: archive's identity vocabulary; the two never both appear.
    messages: list[dict[str, object]] | None = None
    #: The windowed-evidence kinds' page.  The third body vocabulary: neither
    #: a message window nor a relation answered whole, so it carries its own
    #: reported bound and its continuation is minted in its relation's own
    #: projection rather than the message family's.
    evidence_window: EvidenceWindowBody | None = None
    #: Whether the composed lineage the window was sliced from is the full
    #: logical transcript (polylogue-ppkj).  A short window behind a dangling
    #: branch point must not read as a short conversation.
    lineage_complete: bool = True
    lineage_truncation_reason: str | None = None
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
        if self.kind in WINDOWED_SESSION_READ_KINDS:
            if self.evidence is not None:
                raise ValueError(f"a {self.kind} window carries no evidence body")
            if self.evidence_window is not None:
                raise ValueError(f"a {self.kind} window carries no evidence-window body")
            if (self.messages is None) != (self.kind != "messages"):
                raise ValueError(f"a {self.kind} window must carry exactly its own row body")
            return self
        if self.messages is not None:
            raise ValueError(f"{self.kind} is not a message window and carries no message rows")
        if self.kind in WINDOWED_EVIDENCE_KINDS:
            if self.evidence is not None:
                raise ValueError(f"{self.kind} is windowed and carries no whole-evidence body")
            window = self.evidence_window
            if window is None:
                raise ValueError(f"{self.kind} result is missing its evidence-window body")
            if window.relation != self.kind:
                raise ValueError(f"{self.kind} result carries a {window.relation} page")
            # The envelope and the body must agree, because the two are read by
            # different callers: the operation kernel reads the envelope, and
            # the CLI evidence views render the body alone.  A bound reported
            # in only one of them is a bound one surface silently drops.
            envelope = (self.total, self.limit, self.offset, self.next_offset, self.continuation, self.complete)
            reported = (
                window.total,
                window.limit,
                window.offset,
                window.next_offset,
                window.continuation,
                window.complete,
            )
            if envelope != reported:
                raise ValueError(f"the {self.kind} window body and its envelope disagree about the bound")
            return self
        if self.evidence_window is not None:
            raise ValueError(f"{self.kind} is answered whole and carries no evidence-window body")
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
    "ANCHORED_SESSION_READ_KINDS",
    "CONTINUABLE_SESSION_READ_KINDS",
    "WINDOWED_EVIDENCE_KINDS",
    "WINDOWED_SESSION_READ_KINDS",
    "AggregateMode",
    "EvidenceWindowBody",
    "QueryAggregateRequest",
    "QueryAggregateResult",
    "SessionReadKind",
    "SessionReadRequest",
    "SessionReadResult",
    "SessionReferenceRequest",
    "SessionReferenceResult",
]
