"""Typed string reference DTOs for archive object and evidence pointers."""

from __future__ import annotations

from dataclasses import dataclass
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
]

EvidenceRefKind: TypeAlias = Literal["session", "message", "block"]
PublicRef: TypeAlias = "ObjectRef | EvidenceRef"

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
    """Recovery evidence pointer using ``session_id[::message_id[::block]]``."""

    session_id: str
    message_id: str | None = None
    block_index: int | None = None

    @classmethod
    def parse(cls, value: str) -> EvidenceRef:
        """Parse the recovery evidence id format.

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
        """Return the canonical recovery evidence id."""

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


__all__ = [
    "EvidenceRef",
    "EvidenceRefKind",
    "ObjectRef",
    "ObjectRefKind",
    "PublicRef",
    "normalize_object_ref_text",
    "normalize_public_ref_text",
    "parse_public_ref",
]
