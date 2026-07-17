"""Typed string reference DTOs for archive object and evidence pointers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Final, Literal, TypeAlias

ObjectRefKind: TypeAlias = Literal[
    "session",
    "message",
    "block",
    "attachment",
    "paste_span",
    "work_event",
    "phase",
    "thread",
    "file",
    "branch",
    "commit",
    "check-run",
    "workspace",
    "agent",
    "user",
    "repo",
    "insight",
    "run",
    "context-snapshot",
    "observed-event",
    "assertion",
    "saved_view",
    "recall_pack",
    "transform",
    "tool-call",
    "subagent-report",
    "github-issue",
    "github-pr",
    "github-review",
    # Analysis-provenance object kinds (polylogue-rxdo epic). Most refs land
    # here before their backing storage tiers exist; annotation-batch now
    # resolves through durable user.db storage (polylogue-rxdo.7.1). Do not append a
    # ``@content-hash`` anchor suffix onto these — that belongs to the
    # separate citation-anchor work (polylogue-bby.11).
    "query",
    "query-run",
    "result-set",
    "finding",
    "cohort",
    "analysis",
    "annotation-batch",
    # rxdo.9.11-.16 comparative-judgment mechanisms (K-O). ``ranker`` and
    # ``judgment-set`` follow the same content-address discipline as
    # ``query``/``metric``: two rankers citing different hashes are visibly
    # incomparable. ``elicitation-session`` and ``experiment-analysis`` are
    # durable receipt/analysis-artifact identities, not new lifecycle stores.
    "judgment-set",
    "ranker",
    "elicitation-session",
    "experiment-analysis",
    # polylogue-lph4: delegation attempt identity, reusing the polylogue-y964
    # `delegations` view vocabulary. Two id shapes share this one kind:
    #   - action-observed (resolved/unresolved/ambiguous): object_id is the
    #     parent-side dispatch `instruction_tool_use_block_id` verbatim. That
    #     block id already embeds its owning session_id as a structural
    #     prefix, so (parent_session_id, instruction_tool_use_block_id) is
    #     fully recoverable from the id alone -- no extra encoding needed.
    #   - edge-only (edge_only/quarantined, no parent-side dispatch action
    #     to key off): object_id is ``edge:<parent_session_id>::<child_session_id>``,
    #     a deterministic relation identity over the resolved session_links
    #     pair. ``::`` is used as the internal separator (matching
    #     EvidenceRef) because session ids themselves may contain ``:``.
    "delegation",
    # Provider-neutral work-evidence graph. These are topology and claim
    # identities, not provider-native workflow identifiers; source adapters
    # retain their native facts as evidence on the graph nodes and edges.
    "work-invocation",
    "work-call",
    "work-attempt",
    "work-session-segment",
    "work-result",
    "work-claim",
    "work-edge",
    "artifact",
    "execution-context",
    "actor",
]

EvidenceRefKind: TypeAlias = Literal["session", "message", "block"]
PublicRef: TypeAlias = "ObjectRef | EvidenceRef"
ActorKind: TypeAlias = Literal["user", "service", "agent", "model"]

_ACTOR_KINDS: Final[frozenset[ActorKind]] = frozenset({"user", "service", "agent", "model"})

_OBJECT_REF_KINDS: Final[dict[str, ObjectRefKind]] = {
    "session": "session",
    "message": "message",
    "block": "block",
    "attachment": "attachment",
    "paste_span": "paste_span",
    "work_event": "work_event",
    "phase": "phase",
    "thread": "thread",
    "file": "file",
    "branch": "branch",
    "commit": "commit",
    "check-run": "check-run",
    "workspace": "workspace",
    "agent": "agent",
    "user": "user",
    "repo": "repo",
    "insight": "insight",
    "run": "run",
    "context-snapshot": "context-snapshot",
    "observed-event": "observed-event",
    "assertion": "assertion",
    "saved_view": "saved_view",
    "recall_pack": "recall_pack",
    "transform": "transform",
    "tool-call": "tool-call",
    "subagent-report": "subagent-report",
    "github-issue": "github-issue",
    "github-pr": "github-pr",
    "github-review": "github-review",
    "query": "query",
    "query-run": "query-run",
    "result-set": "result-set",
    "finding": "finding",
    "cohort": "cohort",
    "analysis": "analysis",
    "annotation-batch": "annotation-batch",
    "delegation": "delegation",
    "judgment-set": "judgment-set",
    "ranker": "ranker",
    "elicitation-session": "elicitation-session",
    "experiment-analysis": "experiment-analysis",
    "work-invocation": "work-invocation",
    "work-call": "work-call",
    "work-attempt": "work-attempt",
    "work-session-segment": "work-session-segment",
    "work-result": "work-result",
    "work-claim": "work-claim",
    "work-edge": "work-edge",
    "artifact": "artifact",
    "execution-context": "execution-context",
    "actor": "actor",
}


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """Colon-formatted public object reference.

    The DTO intentionally keeps the object id opaque. Existing ids may contain
    colons (``codex-session:demo`` or stored message/block ids), so only the
    kind delimiter is globally significant. The block form may carry one
    explicit trailing qualifier for the block index.
    """

    kind: ObjectRefKind
    object_id: str
    qualifiers: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value: str) -> ObjectRef:
        """Parse ``kind:id[:qualifier...]`` into an ``ObjectRef``."""

        kind_value, separator, tail = value.partition(":")
        if not separator:
            raise ValueError("object ref must use 'kind:id' form")
        kind = _parse_object_ref_kind(kind_value)
        if kind == "block" and ":" in tail:
            object_id, qualifier = tail.rsplit(":", 1)
            qualifiers: tuple[str, ...] = (qualifier,)
        else:
            object_id = tail
            qualifiers = ()
        if not object_id:
            raise ValueError("object ref id cannot be empty")
        if object_id.endswith(":") or any(part == "" for part in qualifiers):
            raise ValueError("object ref qualifiers cannot be empty")
        return cls(kind=kind, object_id=object_id, qualifiers=qualifiers)

    def format(self) -> str:
        """Return the canonical colon-delimited string form."""

        return ":".join((self.kind, self.object_id, *self.qualifiers))


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """Archive evidence pointer using ``session_id[::message_id[::block]]``."""

    session_id: str
    message_id: str | None = None
    block_index: int | None = None

    @classmethod
    def parse(cls, value: str) -> EvidenceRef:
        """Parse the archive evidence id format.

        ``::`` is used because session ids themselves are often colon-bearing
        origin-prefixed strings such as ``codex-session:demo``.
        """

        parts = value.split("::")
        if not 1 <= len(parts) <= 3:
            raise ValueError("evidence ref must use session_id[::message_id[::block_index]] form")
        if any(part == "" for part in parts):
            raise ValueError("evidence ref segments cannot be empty")

        block_index: int | None = None
        if len(parts) == 3:
            try:
                block_index = int(parts[2])
            except ValueError as exc:
                raise ValueError("evidence ref block_index must be an integer") from exc
            if block_index < 0:
                raise ValueError("evidence ref block_index cannot be negative")

        return cls(
            session_id=parts[0],
            message_id=parts[1] if len(parts) >= 2 else None,
            block_index=block_index,
        )

    @property
    def ref_kind(self) -> EvidenceRefKind:
        """Return the most specific archive object kind addressed by this ref."""

        if self.block_index is not None:
            return "block"
        if self.message_id is not None:
            return "message"
        return "session"

    def format(self) -> str:
        """Return the canonical archive evidence id."""

        parts = [self.session_id]
        if self.message_id is not None:
            parts.append(self.message_id)
        if self.block_index is not None:
            parts.append(str(self.block_index))
        return "::".join(parts)

    def to_object_ref(self) -> ObjectRef:
        """Project the evidence pointer to the closest public object ref shape."""

        if self.block_index is not None:
            if self.message_id is None:
                raise ValueError("block evidence ref requires message_id")
            return ObjectRef(kind="block", object_id=self.message_id, qualifiers=(str(self.block_index),))
        if self.message_id is not None:
            return ObjectRef(kind="message", object_id=self.message_id)
        return ObjectRef(kind="session", object_id=self.session_id)


@dataclass(frozen=True, slots=True)
class ActorRef:
    """Stable identity for a human, service, agent persona, or model family.

    Runtime configuration, prompt material, and permissions intentionally do
    not belong here.  They identify an :class:`ExecutionContextRef` instead,
    so a single actor can be compared across explicitly distinct environments.
    """

    kind: ActorKind
    identity: str

    def __post_init__(self) -> None:
        if self.kind not in _ACTOR_KINDS:
            raise ValueError(f"unsupported actor kind: {self.kind!r}")
        if not self.identity.strip():
            raise ValueError("actor identity cannot be empty")

    @classmethod
    def parse(cls, value: str) -> ActorRef:
        kind, separator, identity = value.partition(":")
        if not separator:
            raise ValueError("actor ref must use 'kind:identity' form")
        if kind not in _ACTOR_KINDS:
            raise ValueError(f"unsupported actor kind: {kind!r}")
        return cls(kind=kind, identity=identity)

    def format(self) -> str:
        return f"{self.kind}:{self.identity}"


@dataclass(frozen=True, slots=True)
class ExecutionContextRef:
    """Content-addressed behavioral environment with explicit missing fields.

    ``from_observation`` is the construction path for new context identities.
    ``from_legacy_id`` exists only to retain already-persisted opaque values;
    callers cannot claim a legacy id is a complete content-addressed context.
    """

    context_id: str
    known_fields: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()
    content_addressed: bool = True

    def __post_init__(self) -> None:
        if not self.context_id.strip():
            raise ValueError("execution context id cannot be empty")
        known = tuple(sorted(set(self.known_fields)))
        unknown = tuple(sorted(set(self.unknown_fields)))
        if any(not field.strip() for field in (*known, *unknown)):
            raise ValueError("execution context field names cannot be empty")
        if set(known) & set(unknown):
            raise ValueError("execution context fields cannot be both known and unknown")
        if self.content_addressed and not self.context_id.startswith("sha256:"):
            raise ValueError("content-addressed execution contexts must use a sha256 id")
        object.__setattr__(self, "known_fields", known)
        object.__setattr__(self, "unknown_fields", unknown)

    @classmethod
    def from_observation(
        cls,
        fields: Mapping[str, object],
        *,
        unknown_fields: tuple[str, ...] = (),
    ) -> ExecutionContextRef:
        """Build one deterministic context id from observed environment fields."""

        known_fields = tuple(sorted(fields))
        canonical = json.dumps(
            {"fields": {key: fields[key] for key in known_fields}, "unknown_fields": sorted(set(unknown_fields))},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return cls(
            context_id=f"sha256:{sha256(canonical.encode('utf-8')).hexdigest()}",
            known_fields=known_fields,
            unknown_fields=unknown_fields,
        )

    @classmethod
    def from_legacy_id(cls, context_id: str) -> ExecutionContextRef:
        """Represent an old opaque context id without overstating its evidence."""

        return cls(context_id=context_id, content_addressed=False)

    @property
    def unknown_fraction(self) -> float:
        total = len(self.known_fields) + len(self.unknown_fields)
        return len(self.unknown_fields) / total if total else 1.0

    @property
    def is_complete(self) -> bool:
        return self.content_addressed and not self.unknown_fields


@dataclass(frozen=True, slots=True)
class WorkerProfileRef:
    """An explicit grouping over actor, exact context, and declared role."""

    actor: ActorRef
    execution_context: ExecutionContextRef
    role: str

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("worker profile role cannot be empty")

    @property
    def profile_id(self) -> str:
        canonical = json.dumps(
            {
                "actor": self.actor.format(),
                "execution_context": self.execution_context.context_id,
                "role": self.role,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return f"sha256:{sha256(canonical.encode('utf-8')).hexdigest()}"


def parse_public_ref(value: str) -> ObjectRef | EvidenceRef:
    """Parse a public object or raw evidence reference."""

    try:
        return ObjectRef.parse(value)
    except ValueError as object_exc:
        try:
            return EvidenceRef.parse(value)
        except ValueError:
            raise ValueError(f"unsupported public ref: {value!r}") from object_exc


def normalize_object_ref_text(value: str) -> str:
    """Return canonical object-ref text or raise ``ValueError``."""

    return ObjectRef.parse(value).format()


def normalize_public_ref_text(value: str) -> str:
    """Return canonical object/evidence-ref text or raise ``ValueError``."""

    return parse_public_ref(value).format()


def _parse_object_ref_kind(value: str) -> ObjectRefKind:
    try:
        return _OBJECT_REF_KINDS[value]
    except KeyError as exc:
        raise ValueError(f"unsupported object ref kind: {value!r}") from exc


_DELEGATION_EDGE_PREFIX: Final[str] = "edge:"


def delegation_edge_object_id(parent_session_id: str, child_session_id: str) -> str:
    """Return the deterministic edge-only delegation object id.

    Used when a resolved child has no discoverable parent-side dispatch
    action to key off (``mapping_state`` ``edge_only``/``quarantined`` in
    the ``delegations`` view) -- there is no ``instruction_tool_use_block_id``
    to anchor identity on, so identity falls back to the resolved
    ``session_links`` relation itself.
    """

    if not parent_session_id or not child_session_id:
        raise ValueError("delegation edge identity requires non-empty parent and child session ids")
    return f"{_DELEGATION_EDGE_PREFIX}{parent_session_id}::{child_session_id}"


def parse_delegation_edge_object_id(object_id: str) -> tuple[str, str] | None:
    """Split a delegation object id into ``(parent_session_id, child_session_id)``.

    Returns ``None`` when ``object_id`` is not the edge-only shape -- callers
    should then treat it as an action-observed
    ``instruction_tool_use_block_id`` instead.
    """

    if not object_id.startswith(_DELEGATION_EDGE_PREFIX):
        return None
    remainder = object_id[len(_DELEGATION_EDGE_PREFIX) :]
    parent_session_id, separator, child_session_id = remainder.partition("::")
    if not separator or not parent_session_id or not child_session_id:
        return None
    return parent_session_id, child_session_id


__all__ = [
    "ActorKind",
    "ActorRef",
    "EvidenceRef",
    "EvidenceRefKind",
    "ExecutionContextRef",
    "ObjectRef",
    "ObjectRefKind",
    "PublicRef",
    "WorkerProfileRef",
    "delegation_edge_object_id",
    "normalize_object_ref_text",
    "normalize_public_ref_text",
    "parse_delegation_edge_object_id",
    "parse_public_ref",
]
