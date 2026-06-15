"""Typed string reference DTOs for archive object and evidence pointers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

ObjectRefKind: TypeAlias = Literal[
    "session",
    "message",
    "block",
    "workspace",
    "agent",
    "github-issue",
    "github-pr",
]

EvidenceRefKind: TypeAlias = Literal["session", "message", "block"]

_OBJECT_REF_KINDS: Final[dict[str, ObjectRefKind]] = {
    "session": "session",
    "message": "message",
    "block": "block",
    "workspace": "workspace",
    "agent": "agent",
    "github-issue": "github-issue",
    "github-pr": "github-pr",
}


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """Colon-formatted public object reference.

    The DTO intentionally keeps trailing segments opaque. Existing assertion
    refs include both simple shapes (``session:s1``) and richer shapes such as
    ``github-issue:Sinity/polylogue#1883``; callers can type the head without
    reinterpreting domain-specific suffixes.
    """

    kind: ObjectRefKind
    object_id: str
    qualifiers: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value: str) -> ObjectRef:
        """Parse ``kind:id[:qualifier...]`` into an ``ObjectRef``."""

        parts = value.split(":")
        if len(parts) < 2:
            raise ValueError("object ref must use 'kind:id' form")
        kind = _parse_object_ref_kind(parts[0])
        object_id = parts[1]
        qualifiers = tuple(parts[2:])
        if not object_id:
            raise ValueError("object ref id cannot be empty")
        if any(part == "" for part in qualifiers):
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


def _parse_object_ref_kind(value: str) -> ObjectRefKind:
    try:
        return _OBJECT_REF_KINDS[value]
    except KeyError as exc:
        raise ValueError(f"unsupported object ref kind: {value!r}") from exc


__all__ = ["EvidenceRef", "EvidenceRefKind", "ObjectRef", "ObjectRefKind"]
