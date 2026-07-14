"""Evidence ancestry walker: circularity, epoch skew, expired refs (rxdo.9.9).

A report renderer that walks a claim's cited-evidence graph and flags:
circular ancestry (including a claim that ultimately cites its own
detector's prior output -- an extension of the rxdo.4 laundering guard),
epoch skew between cited refs, definition-version incompatibility,
frame-coverage drift, and expired/stale/missing/ambiguous/quarantined/
private-or-excised refs. This is read-side analysis only: it never copies
evidence bytes, it only reports on the ref graph a caller supplies.

The graph itself (:class:`EvidenceNode`/:class:`EvidenceEdge`) is an
injected, in-memory representation. The real durable evidence graph this
eventually walks (findings/claims from rxdo.4, source anchors, judgments,
metric/ranker/classifier definitions, evaluation-world refs) has not landed
in this tree yet; wiring :func:`walk_evidence_ancestry` to a live storage
reader is deferred to whichever lane lands that durable graph (rxdo.4,
3tl.16, bby.15). The walker itself -- traversal, cycle/skew/state
detection, path witnesses, and the deterministic clean/flagged/blocked
status -- is complete and independently testable against any graph shape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

RefState = Literal["ok", "expired", "stale", "missing", "ambiguous", "quarantined", "private"]

AncestryFlagKind = Literal[
    "circular-ancestry",
    "epoch-skew",
    "definition-incompatible",
    "expired-ref",
    "stale-ref",
    "missing-ref",
    "ambiguous-ref",
    "quarantined-ref",
    "frame-coverage-drift",
    "private-or-excised",
]

AncestryStatus = Literal["clean", "flagged", "blocked"]

# Ref-resolution failures block an unqualified supported claim or cold-reader
# export outright (rxdo.9 program AC #3: "circularity/staleness/expired refs
# block current-supported claims and cold-reader export", extended here to
# the structurally identical missing/ambiguous/quarantined/private states --
# each is, like staleness and expiry, a case where the cited ref cannot be
# trusted to back the claim as written). Epoch skew, definition-version
# incompatibility, and frame-coverage drift are composition-quality
# warnings: visible in the report, but they describe a claim that *can*
# still be read honestly with its caveats attached, not one whose citation
# graph has failed to resolve.
_REF_STATE_FLAG_KINDS: dict[RefState, AncestryFlagKind] = {
    "expired": "expired-ref",
    "stale": "stale-ref",
    "missing": "missing-ref",
    "ambiguous": "ambiguous-ref",
    "quarantined": "quarantined-ref",
    "private": "private-or-excised",
}

_BLOCKING_FLAG_KINDS: frozenset[AncestryFlagKind] = frozenset(
    {
        "circular-ancestry",
        "expired-ref",
        "stale-ref",
        "missing-ref",
        "ambiguous-ref",
        "quarantined-ref",
        "private-or-excised",
    }
)


@dataclass(frozen=True, slots=True)
class EvidenceNode:
    """One node in the evidence citation graph."""

    ref: str
    kind: str
    epoch: str | None = None
    definition_version: str | None = None
    ref_state: RefState = "ok"
    frame_ref: str | None = None
    frame_coverage_complete: bool = True


@dataclass(frozen=True, slots=True)
class EvidenceEdge:
    """A directed citation: ``src_ref`` cites ``dst_ref``."""

    src_ref: str
    dst_ref: str
    edge_kind: str = "cites"


@dataclass(frozen=True, slots=True)
class AncestryFlag:
    """One offending condition, with the exact ref path that produced it."""

    kind: AncestryFlagKind
    path: tuple[str, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class AncestryReport:
    root_ref: str
    status: AncestryStatus
    flags: tuple[AncestryFlag, ...]

    def flags_of(self, kind: AncestryFlagKind) -> tuple[AncestryFlag, ...]:
        return tuple(flag for flag in self.flags if flag.kind == kind)


def walk_evidence_ancestry(
    root_ref: str,
    nodes: Mapping[str, EvidenceNode],
    edges: Sequence[EvidenceEdge],
    *,
    detector_output_refs: frozenset[str] = frozenset(),
) -> AncestryReport:
    """Walk the citation graph from ``root_ref`` and return every distinct flag.

    ``detector_output_refs`` names refs that are the *same detector's* own
    prior output, so a claim citing its own detector's result is flagged as
    circular even when the literal ref graph never repeats ``root_ref``
    (self-referential validation laundering, not a literal graph cycle).

    Distinct failures are never flattened into one "invalid" flag -- every
    offending node produces its own typed :class:`AncestryFlag` with a path
    witness back to the root, and the same node may legitimately carry more
    than one (e.g. a definition-incompatible ref that is also epoch-skewed).
    """

    adjacency: dict[str, list[EvidenceEdge]] = {}
    for edge in edges:
        adjacency.setdefault(edge.src_ref, []).append(edge)

    flags: list[AncestryFlag] = []
    root_node = nodes.get(root_ref)
    root_epoch = root_node.epoch if root_node else None
    root_version = root_node.definition_version if root_node else None
    root_frame = root_node.frame_ref if root_node else None

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(ref: str, path: tuple[str, ...]) -> None:
        if ref in visiting:
            flags.append(AncestryFlag("circular-ancestry", (*path, ref), f"{ref} is reachable from itself (cycle)"))
            return
        if ref in visited:
            return

        node = nodes.get(ref)
        if node is None:
            flags.append(AncestryFlag("missing-ref", (*path, ref), f"{ref} has no resolvable evidence node"))
            return

        if ref != root_ref and ref in detector_output_refs:
            flags.append(
                AncestryFlag(
                    "circular-ancestry",
                    (*path, ref),
                    f"{ref} is the citing detector's own prior output (self-referential validation)",
                )
            )

        flag_kind = _REF_STATE_FLAG_KINDS.get(node.ref_state)
        if flag_kind is not None:
            flags.append(AncestryFlag(flag_kind, (*path, ref), f"{ref} ref_state={node.ref_state!r}"))

        if root_epoch is not None and node.epoch is not None and node.epoch != root_epoch:
            flags.append(
                AncestryFlag(
                    "epoch-skew",
                    (*path, ref),
                    f"{ref} epoch {node.epoch!r} differs from root {root_ref!r} epoch {root_epoch!r}",
                )
            )

        if root_version is not None and node.definition_version is not None and node.definition_version != root_version:
            flags.append(
                AncestryFlag(
                    "definition-incompatible",
                    (*path, ref),
                    f"{ref} definition_version {node.definition_version!r} incompatible with root "
                    f"{root_ref!r} definition_version {root_version!r}",
                )
            )

        if root_frame is not None and node.frame_ref is not None and node.frame_ref != root_frame:
            flags.append(
                AncestryFlag(
                    "frame-coverage-drift",
                    (*path, ref),
                    f"{ref} frame {node.frame_ref!r} drifted from root {root_ref!r} frame {root_frame!r}",
                )
            )
        elif not node.frame_coverage_complete:
            flags.append(AncestryFlag("frame-coverage-drift", (*path, ref), f"{ref} has incomplete frame coverage"))

        visiting.add(ref)
        for edge in adjacency.get(ref, ()):
            visit(edge.dst_ref, (*path, ref))
        visiting.discard(ref)
        visited.add(ref)

    visit(root_ref, ())

    if not flags:
        status: AncestryStatus = "clean"
    elif any(flag.kind in _BLOCKING_FLAG_KINDS for flag in flags):
        status = "blocked"
    else:
        status = "flagged"
    return AncestryReport(root_ref=root_ref, status=status, flags=tuple(flags))


__all__ = [
    "AncestryFlag",
    "AncestryFlagKind",
    "AncestryReport",
    "AncestryStatus",
    "EvidenceEdge",
    "EvidenceNode",
    "RefState",
    "walk_evidence_ancestry",
]
