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
#: ``in-process``
#:     Not classified yet: the handler reads the archive in this process
#:     (through the Python API facade or the archive store) without reaching a
#:     declared operation at all.  This is a *state*, not a design -- it names
#:     the work S8/S9 still owe.  Ten views claimed ``session-read-projection``
#:     while executing here, which made the table unfalsifiable
#:     (polylogue-dutav); :data:`IN_PROCESS_READ_VIEWS` is the shrink-only
#:     ratchet that keeps the honest count from growing back.
ReadViewExecutionKind = Literal[
    "session-read-projection",
    "query-units-projection",
    "distinct-operation",
    "renderer",
    "in-process",
]

# Selection cardinality is shared by every read projection.  It is carried
# through the read command for query-set and body-window projections, so it is
# not a view-owned option even though some handlers consume the resulting
# value.
READ_VIEW_GLOBAL_OPTION_NAMES = frozenset({"limit", "offset"})


#: The raw view windows *artifacts*, not messages, so it takes no message
#: anchor: accepting one it could only ignore would answer a window the caller
#: did not ask for.
RAW_READ_VIEW_OPTION_NAMES = frozenset({"full", "limit", "offset", "continuation"})
MESSAGE_READ_VIEW_OPTION_NAMES = RAW_READ_VIEW_OPTION_NAMES | {"around"}
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
#: A windowed-evidence view pages a relation whose bound is *reported*, so it
#: takes the continuation that resumes the page it minted.  ``limit``/``offset``
#: are the shared global coordinates.  ``events``, ``file-edits`` and
#: ``web-content`` share this set because they share the contract, not because
#: one of them owns it.
EVIDENCE_WINDOW_READ_VIEW_OPTION_NAMES = frozenset({"limit", "continuation"})
EVENTS_READ_VIEW_OPTION_NAMES = EVIDENCE_WINDOW_READ_VIEW_OPTION_NAMES
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


#: Views whose handler still reads the archive in this process.  The ratchet:
#: a view may leave this set (by moving onto a declared operation), never join
#: it.  ``tests/unit/cli/test_read_view_execution_routes.py`` proves membership
#: by dispatching each view with the daemon absent and watching which
#: operations reach the kernel, so a row cannot claim an operation it does not
#: execute -- which is the state ten of these rows were in (polylogue-dutav).
IN_PROCESS_READ_VIEWS: frozenset[str] = frozenset(
    {
        "chronicle",
        "correlation",
        "dialogue",
        "effective_context",
        "neighbors",
        "temporal",
    }
)

# The per-session evidence views render index or source relations the query
# grammar does not declare as structural units: ``hook_events``,
# ``Session.session_events``, ``file_edits``, ``session_agent_policies``,
# ``web_content_constructs``, and the compaction-aware effective-context
# replay.  The two grammar units with similar names read different relations at
# a different grain -- the ``observed-event`` unit reads the materialized
# ``query_observed_events`` projection, not ``Session.session_events``; the
# ``file`` unit reads affected file *paths* via ``query_files``, not the
# structured ``file_edits`` diffs -- so none of them lowers to ``query.units``
# with a ``session:`` filter, and ``query-units-projection`` is unpopulated.
# ``hooks``, ``file-edits``, ``agent-policies`` and ``web-content`` have been
# moved onto ``session.read`` as whole-evidence kinds
# (``daemon_reads._SESSION_EVIDENCE_READERS``), ``messages`` as the
# message-row window kind, and ``events``/``raw`` as the *windowed*-evidence
# kinds (``daemon_reads._WINDOWED_EVIDENCE_READERS``).  The last two needed a
# contract before they could move at all: ``events`` accepted ``--limit`` and
# reported the *truncated* row count as its ``total``, so a whole-evidence
# lowering would have reported ``complete`` for a clipped body; and ``raw``,
# though genuinely windowed, cannot mint artifact continuations into the
# message window's family.  ``read_contracts.EvidenceWindowBody`` supplies the
# reported bound and ``operations/evidence_window.py`` the per-relation
# continuation family.
#
# What remains is not a ``session.read`` projection at all.  ``dialogue``,
# ``temporal`` and ``chronicle`` accept query sets rather than one exact
# reference; ``effective_context`` is a compaction-aware replay with an
# ``--at-position`` parameter; ``neighbors`` and ``correlation`` reach other
# subsystems.  Each needs an operation declaration of its own, which is the
# work S8/S9 still owe.
READ_VIEW_HANDLER_METADATA: dict[str, ReadViewHandlerMetadata] = {
    "summary": ReadViewHandlerMetadata(
        "summary",
        "optional",
        accepts_query_set=True,
        execution_kind="renderer",
        operations=("cli.query",),
    ),
    "transcript": ReadViewHandlerMetadata(
        "transcript",
        "optional",
        accepts_query_set=True,
        execution_kind="renderer",
        operations=("cli.query",),
    ),
    "dialogue": ReadViewHandlerMetadata(
        "dialogue",
        "required",
        accepts_query_set=True,
        execution_kind="in-process",
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
        RAW_READ_VIEW_OPTION_NAMES,
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
        execution_kind="in-process",
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
        EVIDENCE_WINDOW_READ_VIEW_OPTION_NAMES,
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
        EVIDENCE_WINDOW_READ_VIEW_OPTION_NAMES,
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
        execution_kind="in-process",
    ),
    "correlation": ReadViewHandlerMetadata(
        "correlation",
        "required",
        CORRELATION_READ_VIEW_OPTION_NAMES,
        execution_kind="in-process",
    ),
    "temporal": ReadViewHandlerMetadata(
        "temporal",
        "optional",
        accepts_query_set=True,
        execution_kind="in-process",
    ),
    "chronicle": ReadViewHandlerMetadata(
        "chronicle",
        "optional",
        CHRONICLE_READ_VIEW_OPTION_NAMES,
        accepts_query_set=True,
        execution_kind="in-process",
    ),
}


def read_view_option_names() -> frozenset[ReadViewOptionName]:
    """Return every view-specific option name owned by read-view handlers."""

    return frozenset(
        option_name for metadata in READ_VIEW_HANDLER_METADATA.values() for option_name in metadata.accepted_options
    )


def read_view_specific_option_names(views: tuple[str, ...]) -> frozenset[ReadViewOptionName]:
    """Return the view-specific options admitted by a selected composition.

    The Click command owns the parameter objects and their shell completion,
    but this registry owns which view may advertise each one. Keeping the
    ownership calculation here means help validation and completion consume
    the same declared contract.
    """

    return (
        frozenset(
            option_name
            for view in views
            for option_name in READ_VIEW_HANDLER_METADATA.get(
                view, READ_VIEW_HANDLER_METADATA["summary"]
            ).accepted_options
        )
        - READ_VIEW_GLOBAL_OPTION_NAMES
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
        elif kind == "in-process" and metadata.operations:
            offenders.append(f"{view_id}: an in-process view reaches no operation, so it must name none")
    # The ratchet: the honest in-process set may shrink as views are migrated,
    # but a new row may not quietly appear in it.
    declared_in_process = {
        view_id for view_id, metadata in READ_VIEW_HANDLER_METADATA.items() if metadata.execution_kind == "in-process"
    }
    added = sorted(declared_in_process - IN_PROCESS_READ_VIEWS)
    if added:
        offenders.append(f"new in-process views are not allowed: {', '.join(added)}")
    stale = sorted(IN_PROCESS_READ_VIEWS - declared_in_process - set(READ_VIEW_HANDLER_METADATA))
    if stale:
        offenders.append(f"the in-process baseline names views that no longer exist: {', '.join(stale)}")
    migrated = sorted(IN_PROCESS_READ_VIEWS & set(READ_VIEW_HANDLER_METADATA) - declared_in_process)
    if migrated:
        offenders.append(f"remove the migrated views from IN_PROCESS_READ_VIEWS: {', '.join(migrated)}")
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
    "IN_PROCESS_READ_VIEWS",
    "MESSAGE_READ_VIEW_OPTION_NAMES",
    "NEIGHBOR_READ_VIEW_OPTION_NAMES",
    "RAW_READ_VIEW_OPTION_NAMES",
    "READ_VIEW_HANDLER_METADATA",
    "READ_VIEW_GLOBAL_OPTION_NAMES",
    "ReadViewExecutionKind",
    "ReadViewHandlerMetadata",
    "ReadViewOptionName",
    "ReadViewSessionPolicy",
    "read_view_option_names",
    "read_view_specific_option_names",
    "read_views_by_execution_kind",
    "validate_read_view_metadata_registry",
]
