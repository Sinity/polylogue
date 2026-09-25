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


@dataclass(frozen=True, slots=True)
class ReadViewOptionDeclaration:
    name: str
    flags: tuple[str, ...]
    help: str
    value_type: Literal["string", "integer", "float", "boolean", "bounded-integer"] = "string"
    default: str | int | float | bool | None = None
    show_default: bool = False
    minimum: int | None = None
    maximum: int | None = None


FULL_OPTION = ReadViewOptionDeclaration(
    "full", ("--full",), "Read a full single-session body for views that paginate.", "boolean", False
)
CONTINUATION_OPTION = ReadViewOptionDeclaration(
    "continuation",
    ("--continuation",),
    "Resume a snapshot-bound window; supersedes --limit/--offset and refuses a stale archive generation.",
)
AT_POSITION_OPTION = ReadViewOptionDeclaration(
    "at_position", ("--at-position",), "Message position for --view effective_context.", "integer"
)
WINDOW_HOURS_OPTION = ReadViewOptionDeclaration(
    "window_hours",
    ("--window-hours",),
    "Neighboring time window around the seed session.",
    "integer",
    24,
    True,
)
REPO_PATH_OPTION = ReadViewOptionDeclaration(
    "repo_path", ("--repo-path",), "Git repository path for correlation; defaults to the session's repo/cwd."
)
SINCE_HOURS_OPTION = ReadViewOptionDeclaration(
    "since_hours",
    ("--since-hours",),
    "Hours before/after the session to scan for commits.",
    "integer",
    2,
    True,
)
CONFIDENCE_THRESHOLD_OPTION = ReadViewOptionDeclaration(
    "confidence_threshold",
    ("--confidence-threshold",),
    "Minimum confidence for file-overlap commit detection.",
    "float",
    0.3,
    True,
)
GITHUB_API_OPTION = ReadViewOptionDeclaration(
    "github_api",
    ("--github-api/--no-github-api",),
    "Cross-reference issue/PR refs with the GitHub API via gh CLI.",
    "boolean",
    True,
    True,
)
RELATED_LIMIT_OPTION = ReadViewOptionDeclaration(
    "related_limit",
    ("--related-limit",),
    "Number of related sessions to include.",
    "integer",
    5,
    True,
)
MAX_SESSIONS_OPTION = ReadViewOptionDeclaration(
    "max_sessions",
    ("--max-sessions",),
    "Max sessions, 1-20.",
    "bounded-integer",
    5,
    True,
    1,
    20,
)
NO_REDACT_OPTION = ReadViewOptionDeclaration(
    "no_redact",
    ("--no-redact",),
    "Do not redact filesystem paths.",
    "boolean",
    False,
)
NODE_OFFSET_OPTION = ReadViewOptionDeclaration(
    "node_offset",
    ("--node-offset",),
    "Lineage node-page offset.",
    "integer",
    0,
)
NODE_LIMIT_OPTION = ReadViewOptionDeclaration(
    "node_limit",
    ("--node-limit",),
    "Lineage node-page size.",
    "integer",
    50,
    True,
)
EDGE_OFFSET_OPTION = ReadViewOptionDeclaration(
    "edge_offset",
    ("--edge-offset",),
    "Lineage edge-page offset.",
    "integer",
    0,
)
EDGE_LIMIT_OPTION = ReadViewOptionDeclaration(
    "edge_limit",
    ("--edge-limit",),
    "Lineage edge-page size.",
    "integer",
    50,
    True,
)


@dataclass(frozen=True, slots=True)
class ReadViewHandlerMetadata:
    """Executable handler metadata needed by static CLI surfaces."""

    view_id: str
    session_policy: ReadViewSessionPolicy
    accepted_options: frozenset[ReadViewOptionName] = frozenset()
    declared_options: tuple[ReadViewOptionDeclaration, ...] = ()
    accepts_query_set: bool = False
    execution_kind: ReadViewExecutionKind = field(kw_only=True)
    operations: tuple[str, ...] = field(kw_only=True, default=())
    example: str | None = field(kw_only=True, default=None)


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
        example="polylogue find id:codex-session:demo-receipts then read --view summary",
    ),
    "transcript": ReadViewHandlerMetadata(
        "transcript",
        "optional",
        accepts_query_set=True,
        execution_kind="renderer",
        operations=("cli.query",),
        example="polylogue find id:codex-session:demo-receipts then read --view transcript",
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
        frozenset({"limit", "offset"}),
        declared_options=(
            FULL_OPTION,
            CONTINUATION_OPTION,
            ReadViewOptionDeclaration(
                "around",
                ("--around",),
                "Read the window holding this message id (--view messages).",
            ),
        ),
        execution_kind="session-read-projection",
        operations=("session.read",),
        example="polylogue find id:codex-session:demo-receipts then read --view messages",
    ),
    "raw": ReadViewHandlerMetadata(
        "raw",
        "required",
        frozenset({"limit", "offset"}),
        declared_options=(FULL_OPTION, CONTINUATION_OPTION),
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
        frozenset({"limit"}),
        declared_options=(CONTINUATION_OPTION,),
        execution_kind="session-read-projection",
        operations=("session.read",),
    ),
    "effective_context": ReadViewHandlerMetadata(
        "effective_context",
        "required",
        declared_options=(AT_POSITION_OPTION,),
        execution_kind="in-process",
    ),
    "lineage": ReadViewHandlerMetadata(
        "lineage",
        "required",
        declared_options=(NODE_OFFSET_OPTION, NODE_LIMIT_OPTION, EDGE_OFFSET_OPTION, EDGE_LIMIT_OPTION),
        execution_kind="distinct-operation",
    ),
    "topology": ReadViewHandlerMetadata(
        "topology",
        "required",
        declared_options=(NODE_OFFSET_OPTION, NODE_LIMIT_OPTION, EDGE_LIMIT_OPTION),
        execution_kind="distinct-operation",
    ),
    "file-edits": ReadViewHandlerMetadata(
        "file-edits",
        "required",
        frozenset({"limit"}),
        declared_options=(CONTINUATION_OPTION,),
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
        frozenset({"limit"}),
        declared_options=(CONTINUATION_OPTION,),
        execution_kind="session-read-projection",
        operations=("session.read",),
    ),
    "context": ReadViewHandlerMetadata(
        "context",
        "required",
        declared_options=(RELATED_LIMIT_OPTION,),
        execution_kind="distinct-operation",
    ),
    "context-image": ReadViewHandlerMetadata(
        "context-image",
        "none",
        declared_options=(MAX_SESSIONS_OPTION, NO_REDACT_OPTION),
        execution_kind="distinct-operation",
    ),
    "neighbors": ReadViewHandlerMetadata(
        "neighbors",
        "query_or_session",
        frozenset({"limit"}),
        declared_options=(WINDOW_HOURS_OPTION,),
        execution_kind="in-process",
    ),
    "correlation": ReadViewHandlerMetadata(
        "correlation",
        "required",
        declared_options=(REPO_PATH_OPTION, SINCE_HOURS_OPTION, CONFIDENCE_THRESHOLD_OPTION, GITHUB_API_OPTION),
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
        frozenset({"limit"}),
        accepts_query_set=True,
        execution_kind="in-process",
    ),
}


def _view_option_names(view: str) -> frozenset[str]:
    metadata = READ_VIEW_HANDLER_METADATA[view]
    return metadata.accepted_options | {option.name for option in metadata.declared_options}


# Compatibility exports for read-view modules; the declaration remains the
# sole owner of option admission, Click binding, help and completion.
RAW_READ_VIEW_OPTION_NAMES = _view_option_names("raw")
MESSAGE_READ_VIEW_OPTION_NAMES = _view_option_names("messages")
CONTEXT_READ_VIEW_OPTION_NAMES = _view_option_names("context")
CONTEXT_IMAGE_READ_VIEW_OPTION_NAMES = _view_option_names("context-image")
NEIGHBOR_READ_VIEW_OPTION_NAMES = _view_option_names("neighbors")
CORRELATION_READ_VIEW_OPTION_NAMES = _view_option_names("correlation")
CHRONICLE_READ_VIEW_OPTION_NAMES = _view_option_names("chronicle")
EVIDENCE_WINDOW_READ_VIEW_OPTION_NAMES = _view_option_names("file-edits")
EVENTS_READ_VIEW_OPTION_NAMES = _view_option_names("events")
EFFECTIVE_CONTEXT_READ_VIEW_OPTION_NAMES = _view_option_names("effective_context")
LINEAGE_READ_VIEW_OPTION_NAMES = _view_option_names("lineage")
TOPOLOGY_READ_VIEW_OPTION_NAMES = _view_option_names("topology")


def read_view_option_names() -> frozenset[ReadViewOptionName]:
    """Return every view-specific option name owned by read-view handlers."""

    return frozenset(
        option_name
        for metadata in READ_VIEW_HANDLER_METADATA.values()
        for option_name in (*metadata.accepted_options, *(option.name for option in metadata.declared_options))
    )


def declared_read_view_options() -> tuple[ReadViewOptionDeclaration, ...]:
    """Return each dynamically bound Click option once, in declaration order."""

    unique: dict[str, ReadViewOptionDeclaration] = {}
    for metadata in READ_VIEW_HANDLER_METADATA.values():
        for option in metadata.declared_options:
            previous = unique.setdefault(option.name, option)
            if previous != option:
                raise RuntimeError(f"conflicting read option declaration: {option.name}")
    return tuple(unique.values())


def read_view_examples() -> tuple[str, ...]:
    """Return the examples carried by executable view declarations."""

    return tuple(metadata.example for metadata in READ_VIEW_HANDLER_METADATA.values() if metadata.example is not None)


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
            for option_name in (
                *READ_VIEW_HANDLER_METADATA.get(view, READ_VIEW_HANDLER_METADATA["summary"]).accepted_options,
                *(
                    option.name
                    for option in READ_VIEW_HANDLER_METADATA.get(
                        view, READ_VIEW_HANDLER_METADATA["summary"]
                    ).declared_options
                ),
            )
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
    "declared_read_view_options",
    "read_view_examples",
    "ReadViewExecutionKind",
    "ReadViewHandlerMetadata",
    "ReadViewOptionDeclaration",
    "ReadViewOptionName",
    "ReadViewSessionPolicy",
    "read_view_option_names",
    "read_view_specific_option_names",
    "read_views_by_execution_kind",
    "validate_read_view_metadata_registry",
]
