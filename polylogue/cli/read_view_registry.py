"""Lightweight CLI read-view metadata.

This module is intentionally metadata-only.  It lets ``read --views`` and
Click option ownership checks run without importing executable read-view
handlers, the archive API bridge, or storage/query stacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from polylogue.archive.viewport import read_view_choices

ReadViewSessionPolicy = Literal["optional", "required", "query_or_session", "none"]
ReadViewOptionName = str

#: How a read view reaches durable evidence, decided once here (design D3).
#:
#: ``session-read-projection``
#:     A projection of the ``session.read`` operation over one exact session
#:     reference.
#: ``query-units-projection``
#:     A projection of ``query.units`` over a structural query unit, narrowed
#:     by a ``session:`` filter.  The view id must name a unit the grammar
#:     declares; see ``test_read_view_classification_is_total``.
#: ``distinct-operation``
#:     The view needs an operation of its own (``session.lineage``,
#:     ``context.compile``); those declarations land with S9, so these rows
#:     carry no operation name yet.
#: ``renderer``
#:     A client-side renderer composed from operations that already exist; it
#:     adds no operation of its own.
ReadViewExecutionKind = Literal[
    "session-read-projection",
    "query-units-projection",
    "distinct-operation",
    "renderer",
]

# Selection cardinality is shared by every read projection.  It is carried
# through the read command for query-set and body-window projections, so it is
# not a view-owned option even though some handlers consume the resulting
# value.
READ_VIEW_GLOBAL_OPTION_NAMES = frozenset({"limit", "offset"})


MESSAGE_READ_VIEW_OPTION_NAMES = frozenset({"full", "limit", "offset"})
CONTEXT_READ_VIEW_OPTION_NAMES = frozenset({"related_limit"})
CONTEXT_IMAGE_READ_VIEW_OPTION_NAMES = frozenset(
    {
        "max_sessions",
        "no_redact",
    }
)
NEIGHBOR_READ_VIEW_OPTION_NAMES = frozenset({"limit", "window_hours"})
CORRELATION_READ_VIEW_OPTION_NAMES = frozenset({"confidence_threshold", "github_api", "repo_path", "since_hours"})
CHRONICLE_READ_VIEW_OPTION_NAMES = frozenset({"limit"})
EVENTS_READ_VIEW_OPTION_NAMES = frozenset({"limit"})
EFFECTIVE_CONTEXT_READ_VIEW_OPTION_NAMES = frozenset({"at_position"})
LINEAGE_READ_VIEW_OPTION_NAMES = frozenset({"node_offset", "node_limit", "edge_offset", "edge_limit"})
# Topology pages by node window plus an edge bound. These reuse the Click
# params the lineage view already declares, so registering them adds no new
# positional surface to the query verbs.
TOPOLOGY_READ_VIEW_OPTION_NAMES = frozenset({"node_offset", "node_limit", "edge_limit"})


@dataclass(frozen=True, slots=True)
class ReadViewHandlerMetadata:
    """Executable handler metadata needed by static CLI surfaces."""

    view_id: str
    session_policy: ReadViewSessionPolicy
    accepted_options: frozenset[ReadViewOptionName] = frozenset()
    accepts_query_set: bool = False
    execution_kind: ReadViewExecutionKind = field(kw_only=True)
    operations: tuple[str, ...] = field(kw_only=True, default=())


# Every one of the six per-session evidence views below renders an index or
# source relation the query grammar does not declare as a structural unit:
# ``hook_events``, ``Session.session_events``, ``file_edits``,
# ``session_agent_policies``, ``web_content_constructs``, and the
# compaction-aware effective-context replay.  The two grammar units with
# similar names read different relations at a different grain -- the
# ``observed-event`` unit reads the materialized ``query_observed_events``
# projection, not ``Session.session_events``; the ``file`` unit reads affected
# file *paths* via ``query_files``, not the structured ``file_edits`` diffs --
# so none of them lowers to ``query.units`` with a ``session:`` filter.  They
# are ``session.read`` projections, and ``query-units-projection`` is
# currently unpopulated.
READ_VIEW_HANDLER_METADATA: dict[str, ReadViewHandlerMetadata] = {
    "summary": ReadViewHandlerMetadata(
        "summary",
        "optional",
        accepts_query_set=True,
        execution_kind="session-read-projection",
        operations=("session.read",),
    ),
    "transcript": ReadViewHandlerMetadata(
        "transcript",
        "optional",
        accepts_query_set=True,
        execution_kind="session-read-projection",
        operations=("session.read",),
    ),
    "dialogue": ReadViewHandlerMetadata(
        "dialogue",
        "required",
        accepts_query_set=True,
        execution_kind="session-read-projection",
        operations=("session.read",),
    ),
    "messages": ReadViewHandlerMetadata(
        "messages",
        "required",
        MESSAGE_READ_VIEW_OPTION_NAMES,
        execution_kind="session-read-projection",
        operations=("session.read",),
    ),
    "raw": ReadViewHandlerMetadata(
        "raw",
        "required",
        MESSAGE_READ_VIEW_OPTION_NAMES,
        execution_kind="session-read-projection",
        operations=("session.read",),
    ),
    "hooks": ReadViewHandlerMetadata(
        "hooks",
        "required",
        execution_kind="session-read-projection",
        operations=("session.read",),
    ),
    "events": ReadViewHandlerMetadata(
        "events",
        "required",
        EVENTS_READ_VIEW_OPTION_NAMES,
        execution_kind="session-read-projection",
        operations=("session.read",),
    ),
    "effective_context": ReadViewHandlerMetadata(
        "effective_context",
        "required",
        EFFECTIVE_CONTEXT_READ_VIEW_OPTION_NAMES,
        execution_kind="session-read-projection",
        operations=("session.read",),
    ),
    "lineage": ReadViewHandlerMetadata(
        "lineage",
        "required",
        LINEAGE_READ_VIEW_OPTION_NAMES,
        execution_kind="distinct-operation",
    ),
    "topology": ReadViewHandlerMetadata(
        "topology",
        "required",
        TOPOLOGY_READ_VIEW_OPTION_NAMES,
        execution_kind="distinct-operation",
    ),
    "file-edits": ReadViewHandlerMetadata(
        "file-edits",
        "required",
        execution_kind="session-read-projection",
        operations=("session.read",),
    ),
    "agent-policies": ReadViewHandlerMetadata(
        "agent-policies",
        "required",
        execution_kind="session-read-projection",
        operations=("session.read",),
    ),
    "web-content": ReadViewHandlerMetadata(
        "web-content",
        "required",
        execution_kind="session-read-projection",
        operations=("session.read",),
    ),
    "context": ReadViewHandlerMetadata(
        "context",
        "required",
        CONTEXT_READ_VIEW_OPTION_NAMES,
        execution_kind="distinct-operation",
    ),
    "context-image": ReadViewHandlerMetadata(
        "context-image",
        "none",
        CONTEXT_IMAGE_READ_VIEW_OPTION_NAMES,
        execution_kind="distinct-operation",
    ),
    "neighbors": ReadViewHandlerMetadata(
        "neighbors",
        "query_or_session",
        NEIGHBOR_READ_VIEW_OPTION_NAMES,
        execution_kind="renderer",
        operations=("cli.query",),
    ),
    "correlation": ReadViewHandlerMetadata(
        "correlation",
        "required",
        CORRELATION_READ_VIEW_OPTION_NAMES,
        execution_kind="renderer",
        operations=("session.read",),
    ),
    "temporal": ReadViewHandlerMetadata(
        "temporal",
        "optional",
        accepts_query_set=True,
        execution_kind="renderer",
        operations=("cli.query", "session.read"),
    ),
    "chronicle": ReadViewHandlerMetadata(
        "chronicle",
        "optional",
        CHRONICLE_READ_VIEW_OPTION_NAMES,
        accepts_query_set=True,
        execution_kind="renderer",
        operations=("cli.query", "session.read"),
    ),
}


def read_view_option_names() -> frozenset[ReadViewOptionName]:
    """Return every view-specific option name owned by read-view handlers."""

    return frozenset(
        option_name for metadata in READ_VIEW_HANDLER_METADATA.values() for option_name in metadata.accepted_options
    )


def validate_read_view_metadata_registry() -> None:
    """Fail fast if profile metadata and handler metadata drift."""

    profile_ids = set(read_view_choices())
    metadata_ids = set(READ_VIEW_HANDLER_METADATA)
    missing = sorted(profile_ids - metadata_ids)
    extra = sorted(metadata_ids - profile_ids)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing metadata: {', '.join(missing)}")
        if extra:
            details.append(f"metadata without profiles: {', '.join(extra)}")
        raise RuntimeError("read-view metadata registry drift: " + "; ".join(details))
    _validate_read_view_classification()


def _validate_read_view_classification() -> None:
    """Fail fast if a classification row cannot be executed as declared.

    A view that names an execution kind but no operation it can reach has not
    been classified; a renderer with no source operation would silently keep a
    local execution path.  Operation *names* are checked against the declared
    protocol by ``tests/unit/cli/test_query_verbs_runtime.py`` -- this module
    stays metadata-only and must not import the operations package.
    """

    offenders: list[str] = []
    for view_id, metadata in READ_VIEW_HANDLER_METADATA.items():
        kind = metadata.execution_kind
        if kind == "session-read-projection" and metadata.operations != ("session.read",):
            offenders.append(f"{view_id}: a session.read projection must name exactly session.read")
        elif kind == "renderer" and not metadata.operations:
            offenders.append(f"{view_id}: a renderer must name the operations it composes")
        elif kind == "distinct-operation" and metadata.operations:
            offenders.append(f"{view_id}: its operation is not declared yet, so it must name none")
    if offenders:
        raise RuntimeError("read-view classification drift: " + "; ".join(sorted(offenders)))


def read_views_by_execution_kind(kind: ReadViewExecutionKind) -> tuple[str, ...]:
    """Return the view ids classified under one execution kind."""

    return tuple(
        sorted(view_id for view_id, metadata in READ_VIEW_HANDLER_METADATA.items() if metadata.execution_kind == kind)
    )


validate_read_view_metadata_registry()


__all__ = [
    "CHRONICLE_READ_VIEW_OPTION_NAMES",
    "CONTEXT_IMAGE_READ_VIEW_OPTION_NAMES",
    "CONTEXT_READ_VIEW_OPTION_NAMES",
    "CORRELATION_READ_VIEW_OPTION_NAMES",
    "EVENTS_READ_VIEW_OPTION_NAMES",
    "LINEAGE_READ_VIEW_OPTION_NAMES",
    "TOPOLOGY_READ_VIEW_OPTION_NAMES",
    "EFFECTIVE_CONTEXT_READ_VIEW_OPTION_NAMES",
    "MESSAGE_READ_VIEW_OPTION_NAMES",
    "NEIGHBOR_READ_VIEW_OPTION_NAMES",
    "READ_VIEW_HANDLER_METADATA",
    "READ_VIEW_GLOBAL_OPTION_NAMES",
    "ReadViewExecutionKind",
    "ReadViewHandlerMetadata",
    "ReadViewOptionName",
    "ReadViewSessionPolicy",
    "read_view_option_names",
    "read_views_by_execution_kind",
    "validate_read_view_metadata_registry",
]
