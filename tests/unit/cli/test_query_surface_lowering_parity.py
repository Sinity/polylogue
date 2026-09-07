"""CLI, API, and MCP lower the same query expression to the same selection.

All three surfaces route a query string through
:func:`~polylogue.archive.query.expression.compile_expression_into`, so a token
the grammar gains is meant to reach every one of them at once.  What that
promise is worth depends on the *lowered* filter set rather than on the parse:
a surface that compiles the token and then drops it before calling a reader
selects a different row set while parsing identically.

Anti-vacuity: the comparison is over
:func:`~polylogue.archive.query.filter_kwargs.spec_session_filter_kwargs`, the
one kwarg set every ``ArchiveStore`` reader accepts.  Giving one surface its
own lowering, or dropping a field on the way to a reader, leaves the keys
unequal.  :func:`test_the_probed_expressions_exercise_a_live_filter_key` keeps
the corpus from degenerating into expressions that lower to nothing.
"""

from __future__ import annotations

import pytest

from polylogue.archive.query.filter_kwargs import spec_session_filter_kwargs
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.root_request import RootModeRequest
from polylogue.mcp.query_contracts import build_query_spec

#: ``(expression, the filter key it must reach)``.  The three
#: ``action_sequence``/``action_text``/``since_session`` rows are the filters
#: that were reachable only as CLI options before the grammar gained them.
_EXPRESSIONS: tuple[tuple[str, str], ...] = (
    ("action_sequence:file_edit>shell", "action_sequence"),
    ("action_text:pytest", "action_text_terms"),
    ("since_session:claude-code-session:abc123", "since_session_id"),
    ("repo:polylogue", "repo_names"),
    ("origin:claude-code-session", "origins"),
    ("tag:review", "tags"),
    ("-tag:stale", "excluded_tags"),
    ("tool:bash", "tool_terms"),
    ("action:file_edit", "action_terms"),
    ("path:polylogue/cli", "referenced_paths"),
    ("cwd:/realm/project", "cwd_prefix"),
    ("has:tools", "has_tool_use"),
    ("messages:>=10", "min_messages"),
    ("words:>=200", "min_words"),
    ("root:false", "root"),
    ("title:refactor", "title"),
)

_IDS = [expression for expression, _ in _EXPRESSIONS]


def _cli_lowering(expression: str) -> dict[str, object]:
    request = RootModeRequest.from_params({"query": (expression,)})
    return dict(spec_session_filter_kwargs(request.query_spec()))


def _api_lowering(expression: str) -> dict[str, object]:
    return dict(spec_session_filter_kwargs(SessionQuerySpec.from_expression(expression)))


def _mcp_lowering(expression: str) -> dict[str, object]:
    return dict(spec_session_filter_kwargs(build_query_spec(query=expression)))


@pytest.mark.parametrize(("expression", "key"), _EXPRESSIONS, ids=_IDS)
def test_the_three_surfaces_lower_one_expression_identically(expression: str, key: str) -> None:
    """One grammar, one selection — whichever surface accepted the string."""
    cli = _cli_lowering(expression)
    assert _api_lowering(expression) == cli
    assert _mcp_lowering(expression) == cli


@pytest.mark.parametrize(("expression", "key"), _EXPRESSIONS, ids=_IDS)
def test_the_probed_expressions_exercise_a_live_filter_key(expression: str, key: str) -> None:
    """An expression that lowers to the default would compare equal for free."""
    empty = dict(spec_session_filter_kwargs(SessionQuerySpec()))
    assert _cli_lowering(expression)[key] != empty[key]
