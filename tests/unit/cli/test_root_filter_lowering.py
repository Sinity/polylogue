"""Every root filter option lowers into the shared SQL filter set.

The root query pushes selection down through one kwarg set
(:mod:`polylogue.archive.query.filter_kwargs`).  An option that narrows
results but leaves that set unchanged is either silently dropped or applied to
the fetched page in Python, and a Python post-filter over a bounded page
returns a different, smaller answer than the same filter in SQL.

Anti-vacuity: :func:`test_every_root_option_is_classified` reads the live
Click tree, so a new root option that is neither declared a filter nor
declared not-a-filter fails; and :func:`test_declared_filters_reach_the_sql_set`
fails for any declared filter whose value leaves the lowered kwargs identical
to the unfiltered ones.
"""

from __future__ import annotations

import pytest

from polylogue.archive.query.filter_kwargs import SessionFilterKwargs, plan_session_filter_kwargs
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.click_app import cli

#: Root options that narrow the result set, with a value that must reach SQL.
_FILTER_SAMPLES: dict[str, object] = {
    "action": ("shell",),
    "action_sequence": "file_read,shell",
    "action_text": ("pytest",),
    "cwd_prefix": "/realm/project",
    "exclude_action": ("shell",),
    "exclude_origin": "chatgpt-export",
    "exclude_tag": ("stale",),
    "exclude_tool": ("Bash",),
    "filter_has_paste": True,
    "filter_has_thinking": True,
    "filter_has_tool_use": True,
    "has_type": ("tool_use",),
    "max_messages": 9,
    "min_messages": 2,
    "min_words": 3,
    "origin": "claude-ai-export",
    "project": ("some-project",),
    "referenced_path": ("README.md",),
    "repo": ("polylogue",),
    "root": False,
    "since": "2020-01-01",
    "since_session_id": "claude-ai-export:ext-conv-0",
    "tag": ("review",),
    "title": "Session",
    "tool": ("Bash",),
    "typed_only": True,
    "until": "2030-01-01",
}

#: Root options that do not narrow the SQL selection, and what carries them.
_NOT_A_SQL_FILTER: dict[str, str] = {
    "contains": "free text, carried in the FTS query string",
    "exclude_text": "free text, carried in the FTS query string",
    "conv_id": "session scope, passed to the reader as its own keyword",
    "cursor": "continuation state, not selection",
    "latest": "ordering and cardinality, not selection",
    "limit": "page size, not selection",
    "offset": "page offset, not selection",
    "sample": "page size and ordering, not selection",
    "sort": "ordering, not selection",
    "reverse": "ordering, not selection",
    "retrieval_lane": "chooses the retrieval lane, not a predicate",
    "lexical": "retrieval lane alias",
    "semantic": "retrieval lane alias",
    "similar_text": "vector seed, not a SQL predicate",
    "add_tag": "mutation, not selection",
    "set_meta": "mutation, not selection",
    "delete_matched": "mutation, not selection",
    "diagnose": "explains routing",
    "explain_query": "explains the compiled query",
    "why": "explains an empty result",
    "help_markdown": "help rendering",
    "output": "delivery destination",
    "output_as_json": "output format",
    "output_format": "output format",
    "plain": "output rendering",
    "stream": "output rendering",
    "verbose": "log level",
    "version": "prints the version",
    "no_daemon": "transport selection",
}


def _root_option_names() -> frozenset[str]:
    return frozenset(param.name for param in cli.params if param.name)


def _unfiltered() -> SessionFilterKwargs:
    return plan_session_filter_kwargs(SessionQuerySpec.from_params({}).to_plan())


def test_every_root_option_is_classified() -> None:
    """A new root option must declare whether it reaches the SQL filter set."""
    classified = set(_FILTER_SAMPLES) | set(_NOT_A_SQL_FILTER)
    unclassified = sorted(_root_option_names() - classified)
    assert not unclassified, f"root options with no declared lowering: {unclassified}"


def test_no_classified_option_has_disappeared() -> None:
    """The table cannot outlive the options it classifies."""
    stale = sorted((set(_FILTER_SAMPLES) | set(_NOT_A_SQL_FILTER)) - _root_option_names())
    assert not stale, f"classified options the root command no longer defines: {stale}"


def test_filter_and_non_filter_classifications_are_disjoint() -> None:
    """One option cannot be declared both a filter and not a filter."""
    assert not (set(_FILTER_SAMPLES) & set(_NOT_A_SQL_FILTER))


@pytest.mark.parametrize("option", sorted(_FILTER_SAMPLES))
def test_declared_filters_reach_the_sql_set(option: str) -> None:
    """Setting a declared filter changes the lowered kwargs."""
    spec = SessionQuerySpec.from_params({option: _FILTER_SAMPLES[option]})
    assert spec != SessionQuerySpec(), f"{option} did not reach the query spec at all"
    assert plan_session_filter_kwargs(spec.to_plan()) != _unfiltered(), (
        f"{option} narrows results but leaves the SQL filter set unchanged"
    )


#: Representative values for the not-a-filter options that accept one.
_NON_FILTER_SAMPLES: dict[str, object] = {
    "contains": ("alpha",),
    "exclude_text": ("omitted",),
    "conv_id": "claude-ai-export:ext-conv-0",
    "latest": True,
    "limit": 7,
    "offset": 4,
    "sample": 3,
    "sort": "messages",
    "reverse": True,
    "retrieval_lane": "hybrid",
    "similar_text": "vector seed",
}


@pytest.mark.parametrize("option", sorted(_NON_FILTER_SAMPLES))
def test_declared_non_filters_leave_the_sql_set_alone(option: str) -> None:
    """An option declared not-a-filter must not smuggle a predicate into SQL."""
    assert option in _NOT_A_SQL_FILTER
    spec = SessionQuerySpec.from_params({option: _NON_FILTER_SAMPLES[option]})
    assert spec != SessionQuerySpec(), f"{option} did not reach the query spec at all"
    assert plan_session_filter_kwargs(spec.to_plan()) == _unfiltered(), (
        f"{option} is declared not-a-filter but changed the SQL filter set"
    )
