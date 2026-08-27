"""Bounded multi-channel retrieval workflows.

The workflow deliberately composes existing query, vector, neighbor, and
topology reads. It owns no storage and never calls an embedding provider to
materialize new vectors.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from polylogue.archive.session.neighbor_candidates import (
    NeighborDiscoveryRequest,
    discover_neighbor_candidates,
)
from polylogue.core.protocols import NeighborStore, VectorProvider

REQUIRED_WORKFLOW_IDS = frozenset({"topic-pack"})


class TopicPackStore(NeighborStore, Protocol):
    async def search_similar(
        self, text: str, limit: int = 10, vector_provider: VectorProvider | None = None
    ) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class TopicPackRequest:
    query: str
    seed_limit: int = 8
    expansion_limit: int = 16
    neighbor_limit: int = 12
    max_sessions: int = 32
    max_messages: int = 64
    vector_provider: VectorProvider | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("topic-pack requires a non-empty query")
        for name in ("seed_limit", "expansion_limit", "neighbor_limit", "max_sessions", "max_messages"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class TopicPackEvidence:
    session_id: str
    reason: str
    evidence: dict[str, object] = field(default_factory=dict)
    citations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TopicPackResult:
    status: str
    query: str
    sessions: tuple[Any, ...]
    evidence: tuple[TopicPackEvidence, ...]
    timeline: tuple[dict[str, object], ...]
    context_pack: tuple[dict[str, object], ...]
    gaps: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        def dump(item: Any) -> Any:
            model_dump = getattr(item, "model_dump", None)
            return model_dump() if callable(model_dump) else item

        return {
            "status": self.status,
            "query": self.query,
            "sessions": [dump(item) for item in self.sessions],
            "evidence": [
                {
                    "session_id": item.session_id,
                    "reason": item.reason,
                    "evidence": item.evidence,
                    "citations": list(item.citations),
                }
                for item in self.evidence
            ],
            "timeline": list(self.timeline),
            "context_pack": list(self.context_pack),
            "gaps": list(self.gaps),
            "metadata": dict(self.metadata),
        }


def _citation(message: Any, session_id: str) -> str | None:
    for block in getattr(message, "blocks", ()):
        content_hash = block.get("content_hash") if isinstance(block, dict) else getattr(block, "content_hash", None)
        if content_hash:
            if isinstance(content_hash, (bytes, bytearray, memoryview)):
                digest = bytes(content_hash).hex()
            else:
                digest = str(content_hash).removeprefix("0x").lower()
            if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
                return f"{session_id}::{message.id}::block@sha256:{digest}"
    return None


def _session_id(value: Any) -> str:
    return str(getattr(value, "id", getattr(value, "session_id", value)))


def _signals(context_pack: list[dict[str, object]]) -> dict[str, list[str]]:
    """Extract bounded, non-semantic hints for later workflow stages."""
    text = "\n".join(str(item["text"]) for item in context_pack)
    patterns = {
        "files": r"(?<![\w/])(?:[\w.-]+/)+[\w.-]+|\b[\w.-]+\.(?:py|ts|tsx|js|json|md|nix)\b",
        "branches": r"\b(?:feature|bugfix|hotfix|release)/[\w./-]+\b",
        "issues": r"(?<!\w)#\d+\b|\b(?:issue|bead)[ -]?[\w.-]+\b",
    }
    return {
        name: sorted(set(re.findall(pattern, text, flags=re.IGNORECASE)))[:16] for name, pattern in patterns.items()
    }


async def build_topic_pack(store: TopicPackStore, request: TopicPackRequest) -> TopicPackResult:
    """Build a bounded topic pack with explainable, independent retrieval lanes."""
    query = " ".join(request.query.split())
    sessions: dict[str, Any] = {}
    evidence: dict[str, TopicPackEvidence] = {}
    gaps: list[str] = []

    seeds = await store.search_summary_hits(query, limit=min(request.seed_limit, request.max_sessions))
    for hit in seeds:
        summary = getattr(hit, "summary", hit)
        sid = _session_id(summary)
        sessions[sid] = summary
        evidence[sid] = TopicPackEvidence(sid, "fts", {"rank": getattr(hit, "rank", None), "lane": "text"})

    vector_status = "disabled" if request.vector_provider is None else "ready"
    retrieval_lanes = {"fts": len(seeds), "embedding": 0, "time": 0, "topology": 0, "content": 0}
    if request.vector_provider is not None and len(sessions) < request.max_sessions:
        try:
            vector_hits = await store.search_similar(
                query,
                limit=min(request.expansion_limit, request.max_sessions - len(sessions)),
                vector_provider=request.vector_provider,
            )
        except Exception as exc:
            vector_status = "unavailable"
            gaps.append(f"vector expansion failed: {type(exc).__name__}")
        else:
            for item in vector_hits:
                sid = _session_id(item)
                if sid not in sessions and len(sessions) >= request.max_sessions:
                    break
                sessions[sid] = item
                evidence[sid] = TopicPackEvidence(sid, "embedding", {"lane": "vector"})
                retrieval_lanes["embedding"] += 1
    elif request.vector_provider is None:
        gaps.append("vector expansion disabled; FTS, time, and topology lanes still ran")

    for sid in tuple(sessions)[: request.max_sessions]:
        try:
            neighbors = await discover_neighbor_candidates(
                store,
                NeighborDiscoveryRequest(session_id=sid, query=query, limit=request.neighbor_limit),
            )
        except Exception as exc:
            gaps.append(f"neighbor expansion failed for {sid}: {type(exc).__name__}")
            continue
        for candidate in neighbors:
            if len(sessions) >= request.max_sessions and candidate.session_id not in sessions:
                continue
            sessions.setdefault(candidate.session_id, candidate.summary)
            evidence.setdefault(
                candidate.session_id,
                TopicPackEvidence(
                    candidate.session_id,
                    "time/topology/content",
                    {"reasons": [reason.detail for reason in candidate.reasons]},
                ),
            )
            for reason in candidate.reasons:
                lane = {"nearby_time": "time", "content_search": "content", "content_similarity": "content"}.get(
                    reason.kind
                )
                if lane is not None:
                    retrieval_lanes[lane] += 1

    # A seed-free query still gets an independent recovery pass. This keeps an
    # empty FTS result from being treated as proof that the topic is absent.
    if not sessions:
        try:
            recovery = await discover_neighbor_candidates(
                store,
                NeighborDiscoveryRequest(query=query, limit=min(request.neighbor_limit, request.max_sessions)),
            )
        except Exception as exc:
            gaps.append(f"precursor recovery failed: {type(exc).__name__}")
        else:
            for candidate in recovery:
                if len(sessions) >= request.max_sessions:
                    break
                sessions[candidate.session_id] = candidate.summary
                evidence[candidate.session_id] = TopicPackEvidence(
                    candidate.session_id,
                    "precursor-recovery",
                    {"lane": "query-neighbor", "reasons": [reason.detail for reason in candidate.reasons]},
                )
                for reason in candidate.reasons:
                    if reason.kind == "content_similarity":
                        retrieval_lanes["content"] += 1

    ordered = tuple(sessions.values())[: request.max_sessions]
    timeline = tuple(
        {"session_id": _session_id(item), "title": getattr(item, "title", None), "position": index}
        for index, item in enumerate(ordered)
    )
    context_pack: list[dict[str, object]] = []
    message_count = 0
    for summary in ordered:
        session = await store.get(_session_id(summary))
        if session is None:
            gaps.append(f"session disappeared during read: {_session_id(summary)}")
            continue
        for message in getattr(session, "messages", ()):
            if message_count >= request.max_messages:
                break
            if not getattr(message, "text", None):
                continue
            citation = _citation(message, _session_id(session))
            context_item: dict[str, object] = {
                "session_id": _session_id(session),
                "message_id": str(message.id),
                "text": message.text,
            }
            if citation:
                context_item["citation"] = citation
                current = evidence.get(_session_id(session))
                if current is not None and citation not in current.citations:
                    evidence[_session_id(session)] = TopicPackEvidence(
                        current.session_id, current.reason, current.evidence, (*current.citations, citation)
                    )
            context_pack.append(context_item)
            message_count += 1
        if message_count >= request.max_messages:
            break

    topology_reader = getattr(store, "get_session_topology", None)
    if callable(topology_reader):
        for summary in ordered:
            try:
                topology = await topology_reader(_session_id(summary))
            except Exception as exc:
                gaps.append(f"topology expansion failed for {_session_id(summary)}: {type(exc).__name__}")
                continue
            if topology is None:
                continue
            retrieval_lanes["topology"] += len(getattr(topology, "nodes", ()))
            current = evidence.get(_session_id(summary))
            if current is not None:
                details = dict(current.evidence)
                details["topology"] = {
                    "root_id": str(getattr(topology, "root_id", "")),
                    "node_count": len(getattr(topology, "nodes", ())),
                    "edge_count": len(getattr(topology, "edges", ())),
                    "cycle_detected": bool(getattr(topology, "cycle_detected", False)),
                }
                evidence[_session_id(summary)] = TopicPackEvidence(
                    current.session_id, current.reason, details, current.citations
                )

    return TopicPackResult(
        status="ok" if ordered else "empty",
        query=query,
        sessions=ordered,
        evidence=tuple(evidence[sid] for sid in sessions if sid in evidence),
        timeline=timeline,
        context_pack=tuple(context_pack),
        gaps=tuple(gaps),
        metadata={
            "workflow_id": "topic-pack",
            "retrieval_reasons": sorted({item.reason for item in evidence.values()}),
            "vector_status": vector_status,
            "bounds": {"max_sessions": request.max_sessions, "max_messages": request.max_messages},
            "content_hash_citations": sum(len(item.citations) for item in evidence.values()),
            "retrieval_lanes": retrieval_lanes,
            "retrieval_channels_attempted": ["fts", "embedding", "time", "topology", "content"],
            "signals": _signals(context_pack),
            "quality_baseline": {
                "kind": "no-vector-fts",
                "session_count": len(seeds),
                "session_ids": [_session_id(item.summary if hasattr(item, "summary") else item) for item in seeds],
                "product_claims": False,
            },
            "query_digest": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        },
    )


__all__ = ["REQUIRED_WORKFLOW_IDS", "TopicPackEvidence", "TopicPackRequest", "TopicPackResult", "build_topic_pack"]
