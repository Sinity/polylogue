"""The one SQL-pushable session-filter kwarg set every archive reader accepts.

``ArchiveStore.list_summaries``, ``search_summaries``, ``semantic_summaries``,
``count_sessions`` and ``count_search_sessions`` accept exactly the keys of
:class:`ArchiveFilterKwargs`.  A read surface lowers its request to a
:class:`~polylogue.archive.query.plan.SessionQueryPlan` and calls
:func:`plan_filter_kwargs`; the surface never spells the set itself, because a
second spelling drifts silently from the first — the ``root`` default alone
reads five plan fields, and a copy that reads two returns the empty page for
every branch-structure filter it was asked for.

Two shapes exist because the session scope is resolved differently by
different callers.  :class:`SessionFilterKwargs` is every filter *except* the
session scope, for a caller that resolved the scope itself and passes
``session_id=`` explicitly.  :class:`ArchiveFilterKwargs` adds the scope for a
caller that lets the reader resolve it.

``stats``/``stats_by`` aggregate over a resolved session-id set rather than a
page, so they accept the same keys minus :data:`AGGREGATE_UNSUPPORTED_KEYS`;
:func:`stats_filter_kwargs` performs that one narrowing.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, TypedDict

from polylogue.archive.message.types import MessageType
from polylogue.archive.query.spec import SessionQuerySpec, resolve_default_root_filter

if TYPE_CHECKING:
    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.archive.query.predicate import QueryPredicate


class SessionFilterKwargs(TypedDict):
    """Every SQL-pushable session filter except the session scope."""

    origins: tuple[str, ...]
    excluded_origins: tuple[str, ...]
    tags: tuple[str, ...]
    excluded_tags: tuple[str, ...]
    repo_names: tuple[str, ...]
    project_refs: tuple[str, ...]
    has_types: tuple[str, ...]
    has_tool_use: bool
    has_thinking: bool
    has_paste: bool
    tool_terms: tuple[str, ...]
    excluded_tool_terms: tuple[str, ...]
    action_terms: tuple[str, ...]
    excluded_action_terms: tuple[str, ...]
    action_sequence: tuple[str, ...]
    action_text_terms: tuple[str, ...]
    referenced_paths: tuple[str, ...]
    cwd_prefix: str | None
    typed_only: bool
    message_type: str | None
    title: str | None
    min_messages: int | None
    max_messages: int | None
    min_words: int | None
    max_words: int | None
    since_ms: int | None
    until_ms: int | None
    since_session_id: str | None
    boolean_predicate: QueryPredicate | None
    root: bool | None


class ArchiveFilterKwargs(SessionFilterKwargs):
    """The complete filter kwarg set shared by every ``ArchiveStore`` reader."""

    session_id: str | None


#: Keys the paged readers accept that ``stats``/``stats_by`` do not.
AGGREGATE_UNSUPPORTED_KEYS: frozenset[str] = frozenset({"boolean_predicate", "session_id"})


def datetime_to_ms(value: datetime | None) -> int | None:
    """Return an epoch-millisecond bound for a resolved plan datetime."""
    if value is None:
        return None
    return int(value.timestamp() * 1000)


def plan_session_filter_kwargs(plan: SessionQueryPlan) -> SessionFilterKwargs:
    """Translate a plan's SQL-pushable filters, leaving the session scope out."""
    message_type = MessageType.normalize(plan.message_type).value if plan.message_type is not None else None
    return {
        "origins": plan.origins,
        "excluded_origins": plan.excluded_origins,
        "tags": plan.tags,
        "excluded_tags": plan.excluded_tags,
        "repo_names": plan.repo_names,
        "project_refs": plan.project_refs,
        "has_types": plan.has_types,
        "has_tool_use": plan.filter_has_tool_use,
        "has_thinking": plan.filter_has_thinking,
        "has_paste": plan.filter_has_paste,
        "tool_terms": plan.tool_terms,
        "excluded_tool_terms": plan.excluded_tool_terms,
        "action_terms": plan.action_terms,
        "excluded_action_terms": plan.excluded_action_terms,
        "action_sequence": plan.action_sequence,
        "action_text_terms": plan.action_text_terms,
        "referenced_paths": plan.referenced_path,
        "cwd_prefix": plan.cwd_prefix,
        "typed_only": plan.typed_only,
        "message_type": message_type,
        "title": plan.title,
        "min_messages": plan.min_messages,
        "max_messages": plan.max_messages,
        "min_words": plan.min_words,
        "max_words": plan.max_words,
        "since_ms": datetime_to_ms(plan.since),
        "until_ms": datetime_to_ms(plan.until),
        "since_session_id": plan.since_session_id,
        "boolean_predicate": plan.boolean_predicate,
        "root": resolve_default_root_filter(
            plan.root,
            boolean_predicate=plan.boolean_predicate,
            parent_id=plan.parent_id,
            continuation=plan.continuation,
            sidechain=plan.sidechain,
            has_branches=plan.has_branches,
        ),
    }


def plan_filter_kwargs(plan: SessionQueryPlan) -> ArchiveFilterKwargs:
    """Translate the SQL-pushable subset of a plan into ``ArchiveStore`` kwargs."""
    return {**plan_session_filter_kwargs(plan), "session_id": plan.session_id}


def spec_session_filter_kwargs(spec: SessionQuerySpec) -> SessionFilterKwargs:
    """Lower a :class:`SessionQuerySpec` through the canonical plan.

    A surface that holds a spec uses this rather than reading spec attributes
    into a hand-built mapping: date parsing, the action-sequence/predicate
    reconciliation, and the ``root`` default all happen in the lowering, so a
    spec read field-by-field is a spec read incompletely.
    """
    return plan_session_filter_kwargs(spec.to_plan())


def stats_filter_kwargs(filters: SessionFilterKwargs) -> dict[str, object]:
    """Narrow the shared set to the keys ``stats``/``stats_by`` accept."""
    return {key: value for key, value in filters.items() if key not in AGGREGATE_UNSUPPORTED_KEYS}


__all__ = [
    "AGGREGATE_UNSUPPORTED_KEYS",
    "ArchiveFilterKwargs",
    "SessionFilterKwargs",
    "datetime_to_ms",
    "plan_filter_kwargs",
    "plan_session_filter_kwargs",
    "spec_session_filter_kwargs",
    "stats_filter_kwargs",
]
