"""Which tracked relations a lowered query-unit request actually reads.

A continuation is invalidated when a relation it read has moved. That is only
useful if the read set is *narrower* than "everything": the point of the
relation-scoped frame is that a session-profile sweep, which no plain
``messages where ...`` page reads, cannot 409 an ongoing read.

The declaration below mirrors two closed dispatches in
``archive_tiers/archive_query_reads.py``: ``_session_filter_clause`` (one
branch per session-scope keyword) and ``_field_predicate_clause`` (one branch
per Boolean session field, each funnelling into that same keyword set). Every
lookup is total by construction: an unrecognised keyword, field, unit or
predicate node yields the full relation set, so a new join that this module
has not learned about over-invalidates rather than serving stale rows.

``tests/unit/archive/query/test_frame_scope.py`` closes the loop from the
other side: it runs a real page for every declared keyword and unit under a
SQLite trace callback and fails if the SQL touches a tracked relation the
declaration omitted.
"""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryExistsPredicate,
    QueryFieldPredicate,
    QueryLineagePredicate,
    QueryNotPredicate,
    QueryPredicate,
    QuerySequencePredicate,
    QueryTextPredicate,
)
from polylogue.storage.sqlite.archive_tiers.query_unit_frame import ALL_FRAME_RELATIONS

FRAME_RELATIONS_ALL: frozenset[str] = frozenset(ALL_FRAME_RELATIONS)

# Every terminal row source joins ``sessions`` for its session scope, and
# every terminal row payload resolves message/block content.
_BASE_RELATIONS: frozenset[str] = frozenset({"sessions", "messages", "blocks"})

# The unit-specific relation each terminal row source adds on top of the base.
# A unit absent here is unknown to this module and yields the full set.
_UNIT_RELATIONS: dict[str, frozenset[str]] = {
    "message": frozenset(),
    "block": frozenset(),
    "action": frozenset(),
    "file": frozenset(),
    "run": frozenset(),
    "observed-event": frozenset(),
    "context-snapshot": frozenset(),
    "assertion": frozenset({"assertions"}),
    "delegation": frozenset({"delegation_facts"}),
}

# Tracked relations each ``_session_filter_clause`` keyword can read.
# ``session_repos``/``repos``/``session_working_dirs``/``action_pairs`` are
# read by some of these and carry no frame trigger at all; that predates this
# module and is recorded in the bead rather than silently widened here.
_SESSION_FILTER_RELATIONS: dict[str, frozenset[str]] = {
    "origin": frozenset({"sessions"}),
    "origins": frozenset({"sessions"}),
    "excluded_origins": frozenset({"sessions"}),
    "tags": frozenset({"sessions", "session_tags"}),
    "excluded_tags": frozenset({"sessions", "session_tags"}),
    "repo_names": frozenset({"sessions"}),
    "project_refs": frozenset({"sessions"}),
    "has_types": frozenset({"sessions", "blocks"}),
    "has_tool_use": frozenset({"sessions"}),
    "has_thinking": frozenset({"sessions"}),
    "has_paste": frozenset({"sessions"}),
    "typed_only": frozenset({"sessions"}),
    "tool_terms": frozenset({"sessions", "blocks"}),
    "excluded_tool_terms": frozenset({"sessions", "blocks"}),
    "action_terms": frozenset({"sessions", "blocks"}),
    "excluded_action_terms": frozenset({"sessions", "blocks"}),
    "action_sequence": frozenset({"sessions", "blocks"}),
    "action_text_terms": frozenset({"sessions", "blocks"}),
    "referenced_paths": frozenset({"sessions", "blocks"}),
    "cwd_prefix": frozenset({"sessions"}),
    "message_type": frozenset({"sessions", "messages"}),
    "title": frozenset({"sessions"}),
    "min_messages": frozenset({"sessions"}),
    "max_messages": frozenset({"sessions"}),
    "min_words": frozenset({"sessions"}),
    "max_words": frozenset({"sessions"}),
    "since_ms": frozenset({"sessions"}),
    "until_ms": frozenset({"sessions"}),
    "root": frozenset({"sessions"}),
}

# ``_field_predicate_clause`` maps a Boolean session field onto the keywords
# above; numeric and count fields read only ``sessions`` columns.
_SESSION_FIELD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "id": (),
    "session": (),
    "repo": ("repo_names",),
    "project": ("project_refs",),
    "origin": ("origins",),
    "tag": ("tags",),
    "path": ("referenced_paths",),
    "cwd": ("cwd_prefix",),
    "tool": ("tool_terms",),
    "action": ("action_terms",),
    "has": ("has_types", "has_paste", "has_tool_use", "has_thinking"),
    "title": ("title",),
    "date": ("since_ms", "until_ms"),
    "since": ("since_ms",),
    "until": ("until_ms",),
}

# Structural ``exists`` sub-queries, by the unit they range over.
_EXISTS_UNIT_RELATIONS: dict[str, frozenset[str]] = {
    "message": frozenset({"sessions", "messages"}),
    "action": frozenset({"sessions", "blocks"}),
    "block": frozenset({"sessions", "blocks"}),
    "file": frozenset({"sessions", "messages", "blocks"}),
    "run": frozenset({"sessions", "messages", "blocks"}),
    "observed-event": frozenset({"sessions", "messages", "blocks"}),
    "context-snapshot": frozenset({"sessions", "messages", "blocks"}),
    "assertion": frozenset({"sessions", "assertions"}),
    "delegation": frozenset({"sessions", "delegation_facts"}),
}


def _session_field_relations(field: str) -> frozenset[str] | None:
    """Relations one Boolean session field reads, or ``None`` when unknown."""
    keywords = _SESSION_FIELD_KEYWORDS.get(field)
    if keywords is None:
        # Count/numeric registry fields lower to a plain ``sessions`` column.
        from polylogue.archive.query.metadata import (
            COUNT_QUERY_FIELD_REGISTRY,
            NUMERIC_QUERY_FIELD_REGISTRY,
        )

        if field in COUNT_QUERY_FIELD_REGISTRY:
            return frozenset({"sessions"})
        numeric = NUMERIC_QUERY_FIELD_REGISTRY.get(field)
        if numeric is not None and numeric.unit_columns.get("session") is not None:
            return frozenset({"sessions"})
        return None
    relations = {"sessions"}
    for keyword in keywords:
        declared = _SESSION_FILTER_RELATIONS.get(keyword)
        if declared is None:
            return None
        relations |= declared
    return frozenset(relations)


def _predicate_relations(predicate: QueryPredicate | None) -> frozenset[str] | None:
    """Relations one predicate node reads, or ``None`` when unknown."""
    if predicate is None:
        return frozenset()
    if isinstance(predicate, QueryFieldPredicate):
        ref = predicate.field_ref
        field = ref.name if ref is not None and ref.scope == "session" else predicate.field
        if ref is not None and ref.scope != "session":
            # A unit-scoped field reads the unit's own row relation, already
            # covered by the request's base set.
            return frozenset()
        return _session_field_relations(field)
    if isinstance(predicate, QueryExistsPredicate):
        return _EXISTS_UNIT_RELATIONS.get(predicate.unit)
    if isinstance(predicate, QueryLineagePredicate):
        # ``logical:`` resolves the materialized logical family through
        # ``session_profiles``; plain lineage reads ``sessions.root_session_id``.
        return frozenset({"sessions", "session_profiles"}) if predicate.logical else frozenset({"sessions"})
    if isinstance(predicate, QueryTextPredicate):
        return frozenset({"sessions", "messages", "blocks"})
    if isinstance(predicate, QuerySequencePredicate):
        return frozenset({"sessions", "blocks"})
    if isinstance(predicate, QueryNotPredicate):
        return _predicate_relations(predicate.child)
    if isinstance(predicate, QueryBoolPredicate):
        merged: set[str] = set()
        for child in predicate.children:
            child_relations = _predicate_relations(child)
            if child_relations is None:
                return None
            merged |= child_relations
        return frozenset(merged)
    return None


def query_unit_frame_relations(source: object, session_filters: Mapping[str, object] | None) -> frozenset[str]:
    """Return the tracked relations one lowered query-unit page can read.

    Conservative in exactly one direction: anything this module does not
    recognise widens the answer to every tracked relation, so the caller
    over-invalidates instead of resuming over a relation that moved.
    """
    unit = getattr(source, "unit", None)
    if not isinstance(unit, str):
        return FRAME_RELATIONS_ALL
    unit_relations = _UNIT_RELATIONS.get(unit)
    if unit_relations is None:
        return FRAME_RELATIONS_ALL
    relations = set(_BASE_RELATIONS | unit_relations)

    # Result *shape* stages (group/aggregate/projection) are not modelled
    # here; an aggregate page carries no offset continuation anyway, so
    # widening costs nothing and keeps this declaration about relations only.
    if (
        getattr(source, "group_by", None) is not None
        or getattr(source, "aggregate", None) is not None
        or getattr(source, "agg_metrics", None)
        or getattr(source, "selected_fields", None)
        or getattr(source, "pipeline_stages", None)
    ):
        return FRAME_RELATIONS_ALL

    for predicate in (getattr(source, "predicate", None), getattr(source, "session_predicate", None)):
        declared = _predicate_relations(predicate)
        if declared is None:
            return FRAME_RELATIONS_ALL
        relations |= declared

    for keyword, value in (session_filters or {}).items():
        if value in (None, (), [], "", False):
            continue
        declared = _SESSION_FILTER_RELATIONS.get(keyword)
        if declared is None:
            return FRAME_RELATIONS_ALL
        relations |= declared

    return frozenset(relations)


__all__ = [
    "FRAME_RELATIONS_ALL",
    "query_unit_frame_relations",
]
