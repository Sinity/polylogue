"""Independent semantic oracle for the convergence-property corpus.

The generated workload comes from the existing pathology composer. This module
only reads those authoritative sessions to derive expected search-session
membership and message-role aggregates. It does not inspect archive tables,
write receipts, or a route implementation.

The fixture has text, tool-use, and tool-result blocks. Search membership is
derived from authored message text only, including an empty result for the
tool-use input probe that the FTS contract excludes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from polylogue.archive.models import Session
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_id as make_session_id
from tests.infra.pathology_composer import ComposedPathology


class ConvergenceLaw(StrEnum):
    """The executable convergence laws for the shared fixture."""

    PERMUTATION = "order-permutation-invariance"
    BATCHING = "incremental-equals-bulk-batching"
    IDEMPOTENCE = "idempotence"
    APPEND_PREFIX = "append-prefix-containment"


_PROVIDER = Provider.CODEX
_PROBE_TERMS = ("revision", "shared", "orphaned", "toolonly")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class AuthoritativeMessage:
    """The input fields used by the independent semantic oracle."""

    native_id: str
    role: str
    text: str


@dataclass(frozen=True, slots=True)
class AuthoritativeSession:
    """A selected logical session revision, independent of storage rows."""

    native_id: str
    messages: tuple[AuthoritativeMessage, ...]
    parent_native_id: str | None = None

    @property
    def session_id(self) -> str:
        return str(make_session_id(_PROVIDER, self.native_id))


@dataclass(frozen=True, slots=True)
class SemanticProjection:
    """Production-readable facts compared to the independent oracle."""

    fts_membership: tuple[tuple[str, tuple[str, ...]], ...]
    role_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ConvergenceDeclaration:
    """The convergence obligation owned by this domain.

    This is deliberately a declaration, not a universal test registry.  The
    execution layer consumes it without inventing case identities or a
    pathology catalogue.  ``candidate_applicability`` is explicit because a
    partial candidate cannot honestly claim the whole-archive laws.
    """

    declaration_id: str
    owner: str
    laws: tuple[ConvergenceLaw, ...]
    production_route: str
    candidate_applicability: str
    witness: str
    unsupported: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConvergenceRunPlan:
    """Small, serializable input contract for one law execution."""

    declaration_id: str
    workload_digest: str
    laws: tuple[ConvergenceLaw, ...]
    route_identity: str
    expected: SemanticProjection
    resource_profile: str
    mutants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GeneratedConvergenceWorkload:
    """The generated authoritative input for the convergence properties."""

    pathology: ComposedPathology
    probe_terms: tuple[str, ...]

    @property
    def authoritative_sessions(self) -> tuple[AuthoritativeSession, ...]:
        return authoritative_sessions(self.pathology)


def convergence_declaration() -> ConvergenceDeclaration:
    """Return the current domain-owned convergence obligation."""
    return ConvergenceDeclaration(
        declaration_id="archive.derived-convergence",
        owner="tests.infra.convergence_laws",
        laws=tuple(ConvergenceLaw),
        production_route="source admission -> parsed-session writer -> DaemonConverger",
        candidate_applicability="full-rewrite-only; partial candidates are unsupported",
        witness="generated_convergence_workload",
        unsupported=("provider/topology breadth beyond the bounded witness",),
    )


def authoritative_sessions(pathology: ComposedPathology) -> tuple[AuthoritativeSession, ...]:
    """Select the declared highest revision for each logical session."""
    selected: dict[str, Session] = {}
    revisions: dict[str, int] = {}
    for session in pathology.sessions:
        native_id = str(session.id)
        revision_value = session.metadata.get("revision_index", 0)
        revision = revision_value if isinstance(revision_value, int) else 0
        if native_id not in selected or revision >= revisions[native_id]:
            selected[native_id] = session
            revisions[native_id] = revision
    return tuple(
        AuthoritativeSession(
            native_id=native_id,
            parent_native_id=None if selected[native_id].parent_id is None else str(selected[native_id].parent_id),
            messages=tuple(
                AuthoritativeMessage(
                    native_id=str(message.id),
                    role=str(message.role),
                    text="" if message.text is None else str(message.text),
                )
                for message in selected[native_id].messages
            ),
        )
        for native_id in sorted(selected)
    )


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.casefold()))


def semantic_oracle(
    sessions: Sequence[AuthoritativeSession],
    *,
    probe_terms: Sequence[str] = _PROBE_TERMS,
) -> SemanticProjection:
    """Derive expected search-session membership and role counts from input."""
    session_by_native_id = {session.native_id: session for session in sessions}
    logical_messages: dict[str, tuple[AuthoritativeMessage, ...]] = {}
    for session in sessions:
        messages = session.messages
        parent = session_by_native_id.get(session.parent_native_id or "")
        if parent is not None:
            shared = 0
            for child_message, parent_message in zip(messages, parent.messages, strict=False):
                if child_message != parent_message:
                    break
                shared += 1
            messages = messages[shared:]
        logical_messages[session.native_id] = messages

    fts: list[tuple[str, tuple[str, ...]]] = []
    for term in probe_terms:
        members: set[str] = set()
        normalized_term = term.casefold()
        for session in sessions:
            if any(normalized_term in _tokens(message.text) for message in logical_messages[session.native_id]):
                members.add(session.session_id)
        fts.append((term, tuple(sorted(members))))
    counts = Counter(message.role for session in sessions for message in logical_messages[session.native_id])
    return SemanticProjection(fts_membership=tuple(fts), role_counts=tuple(sorted(counts.items())))


def build_convergence_run_plan(
    workload: GeneratedConvergenceWorkload | None = None,
    *,
    route_identity: str = "production-ingest-and-daemon-convergence",
    resource_profile: str = "focused-warm",
) -> ConvergenceRunPlan:
    """Compile the declaration and smallest faithful witness into a plan."""
    workload = generated_convergence_workload() if workload is None else workload
    declaration = convergence_declaration()
    payload = {
        "sessions": [
            {
                "native_id": session.native_id,
                "parent_native_id": session.parent_native_id,
                "messages": [asdict(message) for message in session.messages],
            }
            for session in workload.authoritative_sessions
        ],
        "probe_terms": workload.probe_terms,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return ConvergenceRunPlan(
        declaration_id=declaration.declaration_id,
        workload_digest=f"sha256:{digest}",
        laws=declaration.laws,
        route_identity=route_identity,
        expected=semantic_oracle(workload.authoritative_sessions, probe_terms=workload.probe_terms),
        resource_profile=resource_profile,
        mutants=(
            "order-sensitive-overwrite",
            "omitted-fts-batch-member",
            "unconditional-rewrite",
            "stale-excess-retention",
            "over-broad-invalidation",
        ),
    )


def execute_convergence_plan(
    plan: ConvergenceRunPlan,
    archive_roots: Sequence[str | Path],
    *,
    law: ConvergenceLaw,
) -> None:
    """Execute a plan through the production read route.

    The plan owns the expected typed meaning; this adapter owns only route
    execution and selection.  It intentionally does not inspect SQLite rows
    or compare schema-shaped snapshots.
    """
    if law not in plan.laws:
        raise ValueError(f"law {law.value!r} is not declared by {plan.declaration_id!r}")
    for root in archive_roots:
        assert_projection_matches_oracle(
            read_semantic_projection(root),
            plan.expected,
            law=law,
        )


def read_semantic_projection(
    archive_root: str | Path,
    *,
    probe_terms: Sequence[str] = _PROBE_TERMS,
) -> SemanticProjection:
    """Read search membership through the readiness-gated product search surface."""
    from polylogue.archive.query.predicate import QueryBoolPredicate
    from polylogue.storage.search import search_messages
    from polylogue.storage.search.runtime import search_messages_cached
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = Path(archive_root).resolve()
    # Hypothesis rebuilds fixed throwaway roots within one test invocation. The
    # production cache therefore cannot distinguish successive examples here.
    search_messages_cached.cache_clear()
    fts_membership = tuple(
        (
            term,
            tuple(
                sorted(
                    {
                        hit.session_id
                        for hit in search_messages(
                            term, archive_root=root, db_path=root / "index.db", limit=100_000
                        ).hits
                    }
                )
            ),
        )
        for term in probe_terms
    )
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        rows = archive.query_unit_counts(
            "message",
            QueryBoolPredicate(op="and", children=()),
            group_by="role",
            sort="key",
            sort_direction="asc",
            limit=100_000,
        )
    role_counts = tuple(sorted((str(row.group_key), int(row.count)) for row in rows if row.group_key is not None))
    return SemanticProjection(fts_membership=fts_membership, role_counts=role_counts)


def assert_projection_matches_oracle(
    observed: SemanticProjection,
    expected: SemanticProjection,
    *,
    law: ConvergenceLaw | None = None,
) -> None:
    if observed != expected:
        context = "production projection" if law is None else f"{law.value} law"
        raise AssertionError(f"{context} differs from the authoritative oracle")


@lru_cache(maxsize=1)
def generated_convergence_workload() -> GeneratedConvergenceWorkload:
    """Build the deterministic workload used by the convergence properties."""
    from tests.infra.convergence_harness import rich_convergence_pathology

    return GeneratedConvergenceWorkload(pathology=rich_convergence_pathology(), probe_terms=_PROBE_TERMS)


__all__ = [
    "AuthoritativeMessage",
    "AuthoritativeSession",
    "ConvergenceLaw",
    "ConvergenceDeclaration",
    "ConvergenceRunPlan",
    "GeneratedConvergenceWorkload",
    "SemanticProjection",
    "assert_projection_matches_oracle",
    "authoritative_sessions",
    "build_convergence_run_plan",
    "convergence_declaration",
    "execute_convergence_plan",
    "generated_convergence_workload",
    "read_semantic_projection",
    "semantic_oracle",
]
