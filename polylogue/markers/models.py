"""Typed marker extraction records and declaration metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from polylogue.core.enums import AssertionKind


@dataclass(frozen=True, slots=True)
class MarkerProvenance:
    message_id: str
    block_id: str

    @property
    def evidence_refs(self) -> tuple[str, str]:
        return (f"message:{self.message_id}", f"block:{self.block_id}")


def marker_provenance(message_id: str, block_id: str) -> MarkerProvenance:
    """Construct the canonical provenance record for one marker block."""
    return MarkerProvenance(message_id, block_id)


@dataclass(frozen=True, slots=True)
class MarkerMatch:
    kind: str
    body: str
    arguments: Mapping[str, str]
    raw_text: str
    start: int
    end: int
    inline: bool = False
    malformed: bool = False


LoweringTarget = AssertionKind | None


@dataclass(frozen=True, slots=True)
class MarkerKindSpec:
    """The complete declaration for one authoring kind.

    ``lowering_target`` is deliberately an existing owner, never a marker
    storage object.  Adding a kind therefore changes this declaration only.
    """

    kind: str
    payload: str
    lowering_target: LoweringTarget
    description: str
    authority: str = "agent-declared"


@dataclass(frozen=True, slots=True)
class MarkerCandidate:
    match: MarkerMatch
    provenance: MarkerProvenance
    assertion_kind: AssertionKind | None
    authority: str = "agent-declared"

    @property
    def evidence_refs(self) -> tuple[str, str]:
        return self.provenance.evidence_refs
