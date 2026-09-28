"""Bounded multi-channel retrieval workflows.

The workflow deliberately composes existing query, vector, neighbor, and
topology reads. It owns no storage and never calls an embedding provider to
materialize new vectors.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator, Iterable
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
            if not callable(model_dump):
                return item
            try:
                return model_dump(mode="json")
            except TypeError:
                return model_dump()

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


async def _iter_messages(messages: Iterable[Any]) -> AsyncIterator[Any]:
    for message in messages:
        yield message


async def _session_messages(store: Any, session_id: str, page_size: int) -> AsyncIterator[Any] | None:
    """Stream one session's messages in bounded pages until the caller stops.

    The caller's bound counts text-bearing output, not raw rows, so paging
    continues past rows it discards; ``None`` means the session is gone.
    """
    pager = getattr(store, "get_messages_paginated", None)
    iterator = getattr(store, "iter_messages", None)
    if callable(pager):

        async def paged() -> AsyncIterator[Any]:
            offset = 0
            while True:
                page = await pager(session_id, limit=page_size, offset=offset)
                rows = tuple(page[0])
                for row in rows:
                    yield row
                if len(rows) < page_size:
                    return
                offset += len(rows)

        return paged()
    if callable(iterator):
        streamed: AsyncIterator[Any] = iterator(session_id)
        return streamed
    session = await store.get(session_id)
    if session is None:
        return None
    return _iter_messages(getattr(session, "messages", ()))


def _signals(context_pack: list[dict[str, object]]) -> dict[str, list[str]]:
    """Extract bounded, non-semantic hints for later workflow stages."""
    patterns = {
        "files": r"(?<![\w/])(?:[\w.-]+/)+[\w.-]+|\b[\w.-]+\.(?:py|ts|tsx|js|json|md|nix)\b",
        "branches": r"\b(?:feature|bugfix|hotfix|release)/[\w./-]+\b",
        "issues": r"(?<!\w)#\d+\b|\b(?:issue|bead)[ -]?(?:[\w]+-[\w]+(?:\.[\w]+)?|\d+)\b",
    }
    result: dict[str, list[str]] = {}
    for name, pattern in patterns.items():
        found: set[str] = set()
        for item in context_pack:
            for match in re.finditer(pattern, str(item["text"]), flags=re.IGNORECASE):
                found.add(match.group(0))
                if len(found) >= 16:
                    break
            if len(found) >= 16:
                break
        result[name] = sorted(found)
    return result


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
    vector_attempted = False
    retrieval_lanes = {"fts": len(seeds), "embedding": 0, "time": 0, "topology": 0, "content": 0}
    if request.vector_provider is not None and len(sessions) < request.max_sessions:
        vector_attempted = True
        try:
            vector_hits = await store.search_similar(
                query,
                limit=min(request.expansion_limit, request.max_sessions),
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
                if sid in sessions:
                    current = evidence[sid]
                    reasons = {current.reason, "embedding"}
                    evidence[sid] = TopicPackEvidence(
                        sid, "/".join(sorted(reasons)), {**current.evidence, "vector_lane": True}, current.citations
                    )
                    retrieval_lanes["embedding"] += 1
                else:
                    sessions[sid] = item
                    evidence[sid] = TopicPackEvidence(sid, "embedding", {"lane": "vector"})
                    retrieval_lanes["embedding"] += 1
    elif request.vector_provider is None:
        gaps.append("vector expansion disabled; FTS, time, and topology lanes still ran")

    neighbor_attempted = bool(sessions)
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
        neighbor_attempted = True
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
        sid = _session_id(summary)
        messages = await _session_messages(store, sid, max(1, request.max_messages - message_count))
        if messages is None:
            gaps.append(f"session disappeared during read: {_session_id(summary)}")
            continue
        async for message in messages:
            if not getattr(message, "text", None):
                continue
            citation = _citation(message, sid)
            context_item: dict[str, object] = {
                "session_id": sid,
                "message_id": str(message.id),
                "text": message.text,
            }
            if citation:
                context_item["citation"] = citation
                cited = evidence.get(sid)
                if cited is not None and citation not in cited.citations:
                    evidence[sid] = TopicPackEvidence(
                        cited.session_id, cited.reason, cited.evidence, (*cited.citations, citation)
                    )
            context_pack.append(context_item)
            message_count += 1
            if message_count >= request.max_messages:
                break
        if message_count >= request.max_messages:
            break

    attempted = ["fts"]
    if vector_attempted:
        attempted.append("embedding")
    if neighbor_attempted:
        attempted.extend(("time", "content"))
    topology_reader = getattr(store, "get_session_topology", None)
    if ordered and callable(topology_reader):
        attempted.append("topology")
        for summary in ordered:
            try:
                topology = await topology_reader(_session_id(summary))
            except Exception as exc:
                gaps.append(f"topology expansion failed for {_session_id(summary)}: {type(exc).__name__}")
                continue
            if topology is None:
                continue
            retrieval_lanes["topology"] += len(getattr(topology, "nodes", ()))
            expanded = evidence.get(_session_id(summary))
            if expanded is not None:
                details = dict(expanded.evidence)
                details["topology"] = {
                    "root_id": str(getattr(topology, "root_id", "")),
                    "node_count": len(getattr(topology, "nodes", ())),
                    "edge_count": len(getattr(topology, "edges", ())),
                    "cycle_detected": bool(getattr(topology, "cycle_detected", False)),
                }
                evidence[_session_id(summary)] = TopicPackEvidence(
                    expanded.session_id, expanded.reason, details, expanded.citations
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
            "retrieval_channels_attempted": attempted,
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
