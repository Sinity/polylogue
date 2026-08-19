"""Archive-level pathology composer (polylogue-amrpx, zoo v1 engine).

Pathologies are structures that emerge across a *corpus* -- a chain of
revisions, a lineage of forked sessions, a raw bundle of several sessions --
not something a single per-record schema-conformant generator can produce
(``tests/infra/strategies/schema_driven.py`` generates one record's worth of
data; nothing there decides how many revisions of a session exist, or
whether two sessions share a divergent tail).

This module is the missing layer: pure functions that take already-generated
per-record payloads or building blocks (a ``Session``/``Message`` pair from
``tests/infra/builders.py``, or a batch of ``schema_conformant_payload()``
draws) and *compose* them into the archive-level arrangements
``polylogue-yazae``'s zoo enumerates. Nothing here writes to a database --
callers feed the returned ``Session`` objects into ``SessionBuilder``
(``tests/infra/storage_records.py``) or ``write_parsed_session_to_archive``
when a live-archive integration test is wanted; the composed structure
itself is the deliverable and is fully inspectable without one.

Each function's docstring names the production pathology and bead/issue that
motivated it, per the bead's requirement that zoo members carry provenance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from polylogue.archive.models import Session
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.types import SessionId
from tests.infra.builders import make_conv, make_msg

JSONRecord = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ComposedPathology:
    """A named archive-level pathological arrangement plus provenance.

    ``sessions`` holds domain-level ``Session`` objects for pathologies that
    are naturally expressed post-parse (revision chains, lineage, whale
    components, quarantined heads). ``raw_payloads`` holds wire-level JSON
    structures for pathologies that are fundamentally about *raw acquisition
    shape* (multi-session bundles, vintage variant pairs) rather than
    anything a parsed ``Session`` could represent.
    """

    name: str
    pathology: str
    motivated_by: str
    description: str
    sessions: tuple[Session, ...] = ()
    raw_payloads: tuple[object, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    components: tuple[ComposedPathology, ...] = ()
    raw_ingestion_order: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Normalize and validate the raw sequence a harness will ingest."""
        order = self.raw_ingestion_order
        if order is None:
            order = tuple(range(len(self.raw_payloads)))
            object.__setattr__(self, "raw_ingestion_order", order)

        expected = tuple(range(len(self.raw_payloads)))
        if len(order) != len(expected) or not all(type(index) is int for index in order) or set(order) != set(expected):
            raise ValueError("raw_ingestion_order must be a permutation of raw_payloads indexes")

    @property
    def raw_payloads_in_ingestion_order(self) -> tuple[object, ...]:
        """Return raw payloads in the deterministic order a harness should ingest."""
        assert self.raw_ingestion_order is not None
        return tuple(self.raw_payloads[index] for index in self.raw_ingestion_order)

    def with_raw_ingestion_order(self, raw_ingestion_order: Sequence[int]) -> ComposedPathology:
        """Return this arrangement with a different deterministic raw ingest order."""
        return replace(self, raw_ingestion_order=tuple(raw_ingestion_order))

    def compose(
        self,
        *pathologies: ComposedPathology,
        name: str | None = None,
        raw_ingestion_order: Sequence[int] | None = None,
    ) -> ComposedPathology:
        """Nest this arrangement with others without introducing a corpus DSL."""
        return compose_pathologies(
            self,
            *pathologies,
            name=name,
            raw_ingestion_order=raw_ingestion_order,
        )


def compose_pathologies(
    *pathologies: ComposedPathology,
    name: str | None = None,
    raw_ingestion_order: Sequence[int] | None = None,
) -> ComposedPathology:
    """Combine existing arrangements and optionally permute their raw ingestion.

    The result retains each direct component for inspection while exposing its
    sessions and raw payloads as one flat corpus. ``raw_ingestion_order`` is a
    permutation over that flat raw-payload sequence, so a metamorphic test can
    feed the same artifacts through the real ingestion path under many orders.
    """
    if not pathologies:
        raise ValueError("compose_pathologies requires at least one pathology")

    component_names = tuple(pathology.name for pathology in pathologies)
    return ComposedPathology(
        name=name if name is not None else "+".join(component_names),
        pathology="+".join(pathology.pathology for pathology in pathologies),
        motivated_by="; ".join(pathology.motivated_by for pathology in pathologies),
        description=f"Composition of archive pathologies: {', '.join(component_names)}.",
        sessions=tuple(session for pathology in pathologies for session in pathology.sessions),
        raw_payloads=tuple(payload for pathology in pathologies for payload in pathology.raw_payloads),
        components=pathologies,
        raw_ingestion_order=None if raw_ingestion_order is None else tuple(raw_ingestion_order),
    )


# ---------------------------------------------------------------------------
# 1. Append revision chain
# ---------------------------------------------------------------------------


def compose_append_revision_chain(
    *,
    session_id: str = "revision-chain",
    revision_count: int = 4,
    messages_per_revision: int = 2,
    with_self_describing_identity: bool = True,
) -> ComposedPathology:
    """N growing revisions of one logical session sharing a single archive id.

    Motivated by #2467 (session lineage duplication) and the content-hash
    idempotency model (``CLAUDE.md`` "Content-hash idempotency"): re-ingesting
    the same ``native_id`` with more messages appended produces a differing
    content hash, which the writer treats as an *update* to the same session
    row, not a new one. Real providers append to the same on-disk JSONL
    across turns (Claude Code, Codex), so this is the single most common
    non-adversarial archive-level shape a per-record generator cannot
    express: it is a property of a *sequence* of writes, not one record.

    ``with_self_describing_identity`` toggles whether every revision embeds
    an explicit stable identity marker in its metadata (mirroring providers
    whose wire format repeats a conversation id inside every record) versus
    relying purely on the archive computing identity from session id/native
    id alone (mirroring providers where identity is positional/filename-only
    and no revision self-describes as "the same conversation").
    """
    if revision_count < 1:
        raise ValueError("revision_count must be >= 1")
    if messages_per_revision < 1:
        raise ValueError("messages_per_revision must be >= 1")

    revisions: list[Session] = []
    for revision_index in range(1, revision_count + 1):
        message_count = revision_index * messages_per_revision
        messages = [
            make_msg(
                id=f"m{i}",
                role="user" if i % 2 == 1 else "assistant",
                text=f"revision {revision_index} message {i}",
            )
            for i in range(1, message_count + 1)
        ]
        metadata: dict[str, object] = {"revision_index": revision_index}
        if with_self_describing_identity:
            metadata["source_identity"] = session_id
        revisions.append(
            make_conv(
                id=session_id,
                title=f"{session_id} (revision {revision_index})",
                messages=messages,
                metadata=metadata,
            )
        )

    return ComposedPathology(
        name=session_id,
        pathology="append-revision-chain",
        motivated_by="#2467 (session lineage duplication); content-hash idempotency model",
        description=(
            f"{revision_count} growing revisions of session {session_id!r}, "
            f"{'with' if with_self_describing_identity else 'without'} a self-describing "
            "identity marker in metadata."
        ),
        sessions=tuple(revisions),
        metadata={"revision_count": revision_count, "with_self_describing_identity": with_self_describing_identity},
    )


# ---------------------------------------------------------------------------
# 2. Fork / prefix-tail lineage
# ---------------------------------------------------------------------------


def compose_fork_prefix_tail_lineage(
    *,
    parent_id: str = "lineage-parent",
    child_id: str = "lineage-child",
    shared_prefix_len: int = 3,
    child_tail_len: int = 2,
    cycle_candidate: bool = False,
) -> ComposedPathology:
    """A parent session plus a child that replays the parent's prefix.

    Motivated by #2467 / ``docs/design/session-lineage-model.md``: forks,
    resumes, and auto-compactions physically replay the parent's prefix
    rather than referencing it, so the archive's lineage normalization
    (``session_links``, ``branch_point_message_id``, ``inheritance``) exists
    specifically to recompose parent-up-to-branch + child-tail on read
    instead of storing the shared prefix twice. This composer produces that
    exact shape at the domain level: the child's message id prefix is
    identical to the parent's (same ids, same text) up to the branch point,
    followed by a divergent tail.

    ``cycle_candidate=True`` additionally sets ``parent.parent_id =
    child.id``, producing a genuine 2-cycle -- the shape
    ``TopologyEdgeStatus.QUARANTINED`` (cycle-break, ``core/enums.py``)
    exists to detect and break.
    """
    if shared_prefix_len < 1:
        raise ValueError("shared_prefix_len must be >= 1")
    if child_tail_len < 1:
        raise ValueError("child_tail_len must be >= 1")

    shared_messages = [
        make_msg(id=f"m{i}", role="user" if i % 2 == 1 else "assistant", text=f"shared message {i}")
        for i in range(1, shared_prefix_len + 1)
    ]
    parent_tail = [
        make_msg(
            id=f"m{shared_prefix_len + i}",
            role="user" if (shared_prefix_len + i) % 2 == 1 else "assistant",
            text=f"parent-only message {shared_prefix_len + i}",
        )
        for i in range(1, 2)
    ]
    parent = make_conv(
        id=parent_id,
        title="Lineage parent",
        messages=[*shared_messages, *parent_tail],
    )

    child_tail = [
        make_msg(
            id=f"c{i}",
            role="user" if i % 2 == 1 else "assistant",
            text=f"child-divergent message {i}",
        )
        for i in range(1, child_tail_len + 1)
    ]
    child = make_conv(
        id=child_id,
        title="Lineage child",
        messages=[*shared_messages, *child_tail],
        parent_id=SessionId(parent_id),
        branch_type=BranchType.CONTINUATION,
    )

    if cycle_candidate:
        parent = parent.model_copy(update={"parent_id": SessionId(child_id)})

    return ComposedPathology(
        name=f"{parent_id}->{child_id}",
        pathology="fork-prefix-tail-lineage" + ("-cycle-candidate" if cycle_candidate else ""),
        motivated_by="#2467; docs/design/session-lineage-model.md; TopologyEdgeStatus.QUARANTINED",
        description=(
            f"Child {child_id!r} shares its first {shared_prefix_len} message ids/texts with "
            f"parent {parent_id!r}, then diverges for {child_tail_len} messages."
            + (
                f" parent_id is set on BOTH sessions ({parent_id!r}<->{child_id!r}), a genuine cycle."
                if cycle_candidate
                else ""
            )
        ),
        sessions=(parent, child),
        metadata={
            "shared_prefix_len": shared_prefix_len,
            "child_tail_len": child_tail_len,
            "cycle_candidate": cycle_candidate,
        },
    )


# ---------------------------------------------------------------------------
# 3. Multi-session bundle (grouped JSONL)
# ---------------------------------------------------------------------------


def compose_multi_session_bundle(
    records: Sequence[JSONRecord],
    *,
    session_count: int = 3,
) -> ComposedPathology:
    """Interleave generated per-record payloads into one grouped-JSONL raw.

    Motivated by the grouped-JSONL raw shape ``sources/dispatch.py``'s
    ``_lower_payload_specs`` explicitly handles: one raw acquisition unit
    (e.g. one file on disk) containing several sessions' worth of JSONL
    lines, distinguished only by a per-line ``sessionId`` field, not by
    file/document boundaries. No per-record generator produces this: it's a
    property of how many DIFFERENT session ids appear across a batch of
    records, and their relative ordering.

    Takes already-generated per-record payloads (e.g. draws from
    ``schema_conformant_payload``) and partitions them round-robin across
    ``session_count`` synthetic session ids, stamping each record's
    ``sessionId`` key, then interleaves them (preserving each record's
    original relative order within its assigned session) into one JSONL text.
    """
    if session_count < 2:
        raise ValueError("session_count must be >= 2 to be a genuine multi-session bundle")
    if not records:
        raise ValueError("records must be non-empty")

    session_ids = [f"bundle-session-{i}" for i in range(session_count)]
    lines: list[str] = []
    per_session_counts: dict[str, int] = dict.fromkeys(session_ids, 0)
    for index, record in enumerate(records):
        session_id = session_ids[index % session_count]
        stamped: JSONRecord = dict(record)
        stamped["sessionId"] = session_id
        lines.append(json.dumps(stamped))
        per_session_counts[session_id] += 1

    grouped_jsonl = "\n".join(lines) + "\n"

    return ComposedPathology(
        name="multi-session-bundle",
        pathology="multi-session-bundle",
        motivated_by="sources/dispatch.py _lower_payload_specs grouped-JSONL split by sessionId",
        description=(
            f"{len(records)} records interleaved across {session_count} session ids "
            f"({per_session_counts}) in one grouped-JSONL raw blob."
        ),
        raw_payloads=(grouped_jsonl,),
        metadata={"session_ids": session_ids, "lines_per_session": per_session_counts, "total_lines": len(lines)},
    )


# ---------------------------------------------------------------------------
# 4. Whale-scale component
# ---------------------------------------------------------------------------


def compose_whale_scale_component(
    *,
    session_id: str = "whale-component",
    declared_message_count: int = 50_000,
    declared_size_bytes: int = 2 * 1024**3,
    materialized_message_count: int = 32,
) -> ComposedPathology:
    """A session structurally describing a whale-scale component without allocating it.

    Motivated by t93b (whale-scale component handling). Real whale sessions
    are multi-GiB / tens-of-thousands-of-messages Claude Code JSONL exports;
    actually materializing that in a test fixture would itself be the
    resource-abuse anti-pattern the global operating contract forbids. This
    composer builds a small, bounded number of REAL messages
    (``materialized_message_count``, capped well below the declared scale)
    and carries the intended full scale as session metadata
    (``declared_message_count`` / ``declared_size_bytes``) so a harness that
    wants to exercise whale-scale *code paths* (streaming parse, chunked
    materialize) can recognize the pathology and apply its own scale-up
    strategy, without this composer ever holding gigabytes in memory.
    """
    if materialized_message_count < 1:
        raise ValueError("materialized_message_count must be >= 1")
    if materialized_message_count > 512:
        raise ValueError("materialized_message_count is meant to stay small; the whale is in the metadata, not here")
    if materialized_message_count > declared_message_count:
        raise ValueError("materialized_message_count must not exceed declared_message_count")

    messages = [
        make_msg(id=f"m{i}", role="user" if i % 2 == 1 else "assistant", text=f"whale message {i}")
        for i in range(1, materialized_message_count + 1)
    ]
    session = make_conv(
        id=session_id,
        title="Whale-scale component",
        messages=messages,
        metadata={
            "_whale_scale": {
                "declared_message_count": declared_message_count,
                "declared_size_bytes": declared_size_bytes,
                "materialized_message_count": materialized_message_count,
            }
        },
    )

    return ComposedPathology(
        name=session_id,
        pathology="whale-scale-component",
        motivated_by="t93b (whale-scale component handling)",
        description=(
            f"Session declares {declared_message_count} messages / {declared_size_bytes} bytes "
            f"but materializes only {materialized_message_count} real messages; the intended "
            "scale lives in metadata['_whale_scale'] for a harness to act on."
        ),
        sessions=(session,),
        metadata={
            "declared_message_count": declared_message_count,
            "declared_size_bytes": declared_size_bytes,
            "materialized_message_count": materialized_message_count,
        },
    )


# ---------------------------------------------------------------------------
# 5. Quarantined-head arrangement
# ---------------------------------------------------------------------------


def compose_quarantined_head_arrangement(
    *,
    child_id: str = "quarantined-child",
    missing_parent_id: str = "never-ingested-parent",
) -> ComposedPathology:
    """A child session whose parent reference is deliberately never resolved.

    Motivated by ``session_links``'s topology-edge persistence
    (``CLAUDE.md`` "Lineage normalization"): a parser can assert a parent
    reference before the parent itself is ever ingested (or the parent is
    permanently absent -- deleted export, never-captured session). Storage
    persists the edge as unresolved rather than dropping it, and
    ``TopologyEdgeStatus`` distinguishes ``unresolved`` from
    ``quarantined`` (a genuine cycle-break). This composer returns only the
    child half of that arrangement -- the missing parent is described, not
    included -- so a harness can ingest the child alone and assert the
    resulting topology edge is unresolved (or, if the harness later ingests
    a conflicting same-id parent forming a cycle, quarantined).
    """
    child = make_conv(
        id=child_id,
        title="Session with an unresolved parent reference",
        messages=[make_msg(id="m1", text="orphaned head")],
        parent_id=SessionId(missing_parent_id),
        branch_type=BranchType.CONTINUATION,
    )

    return ComposedPathology(
        name=child_id,
        pathology="quarantined-head-arrangement",
        motivated_by="session_links topology-edge persistence; TopologyEdgeStatus.unresolved/quarantined",
        description=(
            f"Session {child_id!r} references parent {missing_parent_id!r}, which is NOT "
            "included in this arrangement -- the harness controls whether/when the parent "
            "ever arrives."
        ),
        sessions=(child,),
        metadata={"missing_parent_native_id": missing_parent_id, "expected_topology_status": "unresolved"},
    )


# ---------------------------------------------------------------------------
# 6. Vintage-variant pair
# ---------------------------------------------------------------------------


def compose_vintage_variant_pair(
    *,
    turns: Sequence[tuple[str, str]] | None = None,
) -> ComposedPathology:
    """Two structurally different wire payloads encoding identical content.

    Motivated by 0qfy/uqwd (export-vintage variant pairs: same logical
    content, different wire shape across a provider's schema versions --
    e.g. gemini-cli v1 vs v2, claude-ai v1 vs v2 package schemas in
    ``polylogue/schemas/providers/*/versions/``). This is a wire-level
    pathology, not something a parsed ``Session`` can represent (parsing
    normalizes the shape difference away), so this composer returns two raw
    JSON documents directly: an "old" flat shape and a "new" nested shape,
    both encoding the same ``(role, text)`` turns. ``extract_old_shape`` /
    ``extract_new_shape`` mirror what two different parser versions would
    do, letting a test prove extracted content is equal despite the wire
    difference.

    Scope, deliberately narrow (``polylogue-0qfy`` composer-fidelity note,
    2026-08-04). This is a GENERIC illustrative shape pair. Its ``old``/``new``
    documents belong to no real provider schema, and the extractors above are
    hand-written stand-ins -- no production parser or classifier reads them. It
    therefore demonstrates the general "same content, different wire vintage"
    mechanism and nothing more.

    In particular it does NOT exercise the branch the measured claude-ai-export
    cohort hit (redundant single text ``content_blocks`` presence flipping
    ``_message_hash_payload``, which read as a real membership conflict). Do not
    treat this composer as evidence for that bug or for the ``ey4ro`` zoo's
    ``content-blocks-vintage`` row. The real-route proof for that cohort is
    ``tests/infra/claude_vintage_live_proof.py`` /
    ``tests/unit/infra/test_claude_vintage_live_proof.py``, built on
    ``pathology_zoo._claude_vintage_live_proof_payload``, which drives the
    production ``parse_ai`` -> ``session_revision_projection`` ->
    ``classify_membership_revisions`` route and carries its own red mutation.
    """
    resolved_turns = turns or (("user", "Hello"), ("assistant", "Hi there!"), ("user", "How are you?"))

    old_shape: JSONRecord = {
        "id": "vintage-pair",
        "messages": [{"role": role, "text": text} for role, text in resolved_turns],
    }
    new_shape: JSONRecord = {
        "id": "vintage-pair",
        "conversation": {
            "turns": [{"speaker": role, "content": {"parts": [text]}} for role, text in resolved_turns],
        },
    }

    return ComposedPathology(
        name="vintage-variant-pair",
        pathology="vintage-variant-pair",
        motivated_by="0qfy/uqwd (export-vintage variant pairs)",
        description=(
            f"Old (flat 'messages') and new (nested 'conversation.turns') wire shapes both "
            f"encoding the identical {len(resolved_turns)}-turn content."
        ),
        raw_payloads=(old_shape, new_shape),
        metadata={"turns": list(resolved_turns)},
    )


def extract_old_shape_turns(payload: JSONRecord) -> list[tuple[str, str]]:
    """Extract (role, text) turns from ``compose_vintage_variant_pair``'s old shape."""
    return [(entry["role"], entry["text"]) for entry in payload["messages"]]


def extract_new_shape_turns(payload: JSONRecord) -> list[tuple[str, str]]:
    """Extract (role, text) turns from ``compose_vintage_variant_pair``'s new shape."""
    return [(entry["speaker"], entry["content"]["parts"][0]) for entry in payload["conversation"]["turns"]]


__all__ = [
    "ComposedPathology",
    "compose_pathologies",
    "compose_append_revision_chain",
    "compose_fork_prefix_tail_lineage",
    "compose_multi_session_bundle",
    "compose_whale_scale_component",
    "compose_quarantined_head_arrangement",
    "compose_vintage_variant_pair",
    "extract_old_shape_turns",
    "extract_new_shape_turns",
]
