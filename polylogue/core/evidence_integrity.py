"""Bounded, provider-neutral evidence support evaluation.

The evaluator owns only graph semantics.  Assertions, findings, query results,
packets, and context images remain owners of their facts and adapt them to the
small node/edge protocol below.  This is deliberately pure so the same result
can be used by durable readers and in-memory packet validation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from polylogue.core.enums import PolylogueStrEnum


class EvidenceIntegrityStatus(PolylogueStrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"
    STALE = "stale"
    CLOSED_LOOP = "closed_loop"
    CYCLE = "cycle"
    UNRESOLVED = "unresolved"
    FRAME_INCOMPLETE = "frame_incomplete"
    HELD_PRIVATE = "held_private"


EvidenceRefState = Literal["ok", "stale", "missing", "ambiguous", "private", "quarantined"]
EvidenceAuthority = Literal["human", "tool", "raw", "git", "pr", "agent", "assertion", "unknown"]


@dataclass(frozen=True, slots=True)
class EvidenceGraphNode:
    ref: str
    kind: str
    authority: EvidenceAuthority = "unknown"
    ref_state: EvidenceRefState = "ok"
    content_hash: str | None = None
    definition_hash: str | None = None
    frame_hash: str | None = None
    as_of: str | None = None
    compatible: bool = True
    review_state: str = "approved"
    public: bool = True


@dataclass(frozen=True, slots=True)
class EvidenceGraphEdge:
    src_ref: str
    dst_ref: str
    purpose: str = "supports"
    compatible: bool = True


@dataclass(frozen=True, slots=True)
class EvidenceWitness:
    code: str
    path: tuple[str, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class EvidenceIntegrityVerdict:
    root_ref: str
    status: EvidenceIntegrityStatus
    witnesses: tuple[EvidenceWitness, ...] = ()
    supported_paths: tuple[tuple[str, ...], ...] = ()
    blind_spots: tuple[str, ...] = ()
    definition_ref: str | None = None
    frame_ref: str | None = None
    as_of: str | None = None
    evaluator_version: str = "evidence-integrity-v1"

    @property
    def supported(self) -> bool:
        return self.status in {EvidenceIntegrityStatus.SUPPORTED, EvidenceIntegrityStatus.PARTIALLY_SUPPORTED}

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.code for item in self.witnesses))


class EvaluationCancelledError(Exception):
    """Raised when a bounded evaluation is cancelled by its caller."""


class EvidenceAdapter(Protocol):
    def nodes(self) -> Sequence[EvidenceGraphNode]: ...
    def edges(self) -> Sequence[EvidenceGraphEdge]: ...


@dataclass(frozen=True, slots=True)
class EvidenceGraphAdapter:
    """Storage-free adapter shared by consumer-owned graph projections."""

    graph_nodes: tuple[EvidenceGraphNode, ...]
    graph_edges: tuple[EvidenceGraphEdge, ...]

    def nodes(self) -> Sequence[EvidenceGraphNode]:
        return self.graph_nodes

    def edges(self) -> Sequence[EvidenceGraphEdge]:
        return self.graph_edges


class AssertionEvidenceAdapter(EvidenceGraphAdapter):
    """Typed name for assertion ancestry projections."""


class FindingEvidenceAdapter(EvidenceGraphAdapter):
    """Typed name for finding provenance projections."""


class PublicClaimEvidenceAdapter(EvidenceGraphAdapter):
    """Typed name for public-claim projections."""


class PacketEvidenceAdapter(EvidenceGraphAdapter):
    """Typed name for analytical packet projections."""


class ContextEvidenceAdapter(EvidenceGraphAdapter):
    """Typed name for context compilation projections."""


def evaluate_evidence(
    root_ref: str,
    nodes: Mapping[str, EvidenceGraphNode] | Sequence[EvidenceGraphNode],
    edges: Sequence[EvidenceGraphEdge],
    *,
    definition_hash: str | None = None,
    frame_hash: str | None = None,
    as_of: str | None = None,
    detector_output_refs: frozenset[str] = frozenset(),
    max_nodes: int = 512,
    cancelled: Callable[[], bool] | None = None,
) -> EvidenceIntegrityVerdict:
    """Evaluate one claim with deterministic, fail-closed witnesses."""
    node_map = dict(nodes) if isinstance(nodes, Mapping) else {node.ref: node for node in nodes}
    adjacency: dict[str, list[EvidenceGraphEdge]] = {}
    for edge in edges:
        adjacency.setdefault(edge.src_ref, []).append(edge)
    witnesses: list[EvidenceWitness] = []
    supported_paths: list[tuple[str, ...]] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    authorities: set[EvidenceAuthority] = set()
    count = 0
    root = node_map.get(root_ref)

    def add(code: str, path: tuple[str, ...], detail: str) -> None:
        item = EvidenceWitness(code, path, detail)
        if item not in witnesses:
            witnesses.append(item)

    def visit(ref: str, path: tuple[str, ...]) -> None:
        nonlocal count
        if cancelled and cancelled():
            raise EvaluationCancelledError
        if count >= max_nodes:
            add("evaluation_budget_exhausted", path, f"node budget {max_nodes} exceeded")
            return
        if ref in visiting:
            add("cycle", (*path, ref), "evidence path revisits an active node")
            return
        if ref in visited:
            return
        count += 1
        node = node_map.get(ref)
        if node is None:
            add("missing_ref", (*path, ref), "referenced node is absent")
            return
        if ref != root_ref and ref in detector_output_refs:
            add("closed_loop", (*path, ref), "claim cites the detector's own output")
        if node.ref_state != "ok":
            add(node.ref_state, (*path, ref), f"ref_state={node.ref_state}")
        if not node.public or node.review_state in {"private", "held_private"}:
            add("held_private", (*path, ref), "node is private or held from publication")
        if not node.compatible:
            add("grounding_incompatible", (*path, ref), "node cannot ground this claim")
        if definition_hash and node.definition_hash and node.definition_hash != definition_hash:
            add("definition_drift", (*path, ref), "definition hash differs from evaluation")
        if frame_hash and node.frame_hash and node.frame_hash != frame_hash:
            add("frame_drift", (*path, ref), "frame hash differs from evaluation")
        if as_of and node.as_of and node.as_of > as_of:
            add("stale", (*path, ref), "node is newer than the evaluation as-of frame")
        # The claim node is not grounding evidence; only descendants decide
        # whether an assertion-only ancestry can launder itself into support.
        if ref != root_ref:
            authorities.add(node.authority)
        visiting.add(ref)
        outgoing = adjacency.get(ref, ())
        if not outgoing and ref != root_ref and node.compatible and node.ref_state == "ok":
            supported_paths.append((*path, ref))
        for edge in outgoing:
            if not edge.compatible:
                add("grounding_incompatible", (*path, ref, edge.dst_ref), "edge purpose is incompatible")
            visit(edge.dst_ref, (*path, ref))
        visiting.discard(ref)
        visited.add(ref)

    try:
        visit(root_ref, ())
    except EvaluationCancelledError:
        add("evaluation_cancelled", (root_ref,), "caller cancelled bounded evaluation")

    codes = {item.code for item in witnesses}
    if "held_private" in codes or "private" in codes:
        status = EvidenceIntegrityStatus.HELD_PRIVATE
    elif "cycle" in codes:
        status = EvidenceIntegrityStatus.CYCLE
    elif "evaluation_cancelled" in codes or "evaluation_budget_exhausted" in codes:
        status = EvidenceIntegrityStatus.UNRESOLVED
    elif "grounding_incompatible" in codes:
        status = EvidenceIntegrityStatus.NOT_SUPPORTED
    elif "closed_loop" in codes or authorities <= {"agent", "assertion"}:
        status = EvidenceIntegrityStatus.CLOSED_LOOP
    elif "missing_ref" in codes or "missing" in codes or "ambiguous" in codes:
        status = EvidenceIntegrityStatus.UNRESOLVED
    elif "stale" in codes or "definition_drift" in codes or "content_drift" in codes:
        status = EvidenceIntegrityStatus.STALE
    elif "frame_drift" in codes or root is None or not root.frame_hash:
        status = EvidenceIntegrityStatus.FRAME_INCOMPLETE
    elif root.ref_state != "ok":
        status = EvidenceIntegrityStatus.UNRESOLVED
    elif supported_paths and not witnesses:
        status = EvidenceIntegrityStatus.SUPPORTED
    elif supported_paths:
        status = EvidenceIntegrityStatus.PARTIALLY_SUPPORTED
    else:
        status = EvidenceIntegrityStatus.NOT_SUPPORTED
    return EvidenceIntegrityVerdict(
        root_ref=root_ref,
        status=status,
        witnesses=tuple(witnesses),
        supported_paths=tuple(supported_paths),
        blind_spots=tuple(sorted(codes - {item.code for item in witnesses})),
        definition_ref=definition_hash,
        frame_ref=frame_hash,
        as_of=as_of,
    )


def evaluate_adapter(root_ref: str, adapter: EvidenceAdapter, **kwargs: object) -> EvidenceIntegrityVerdict:
    """Evaluate any typed consumer adapter through the one algorithm."""
    return evaluate_evidence(root_ref, adapter.nodes(), adapter.edges(), **kwargs)  # type: ignore[arg-type]


__all__ = [
    "AssertionEvidenceAdapter",
    "ContextEvidenceAdapter",
    "EvaluationCancelledError",
    "EvidenceAdapter",
    "EvidenceAuthority",
    "EvidenceGraphAdapter",
    "EvidenceGraphEdge",
    "EvidenceGraphNode",
    "EvidenceIntegrityStatus",
    "EvidenceIntegrityVerdict",
    "FindingEvidenceAdapter",
    "PacketEvidenceAdapter",
    "PublicClaimEvidenceAdapter",
    "EvidenceRefState",
    "EvidenceWitness",
    "evaluate_adapter",
    "evaluate_evidence",
]
