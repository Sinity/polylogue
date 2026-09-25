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

import click
import pytest
from hypothesis import given
from hypothesis import strategies as st

from polylogue.archive.query.filter_kwargs import spec_session_filter_kwargs
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.lowering import lower_cli_query
from polylogue.cli.root_request import RootModeRequest
from polylogue.mcp.query_contracts import build_query_spec
from polylogue.operations.daemon_reads import _cli_query_spec

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


# --- The lowering law -------------------------------------------------------
#
# ``RootModeRequest`` (the Click root) and ``_cli_query_spec`` (the daemon's
# in-process reads) must select the same rows for the same parameter map.
# They now share :mod:`polylogue.operations.query_lowering`; this law is what
# keeps a future surface-local shortcut from re-splitting them.

_QUERY_TERMS = st.sampled_from(
    [
        (),
        ("refactor",),
        ("repo:polylogue",),
        ("repo:polylogue since:7d",),
        ("tool:bash", "action:file_edit"),
        ('"exact phrase"',),
    ]
)

_FLAG_PARAMS = st.fixed_dictionaries(
    {
        "lexical": st.booleans(),
        "semantic": st.booleans(),
        "similar_text": st.sampled_from([None, "", "a prior session"]),
        "limit": st.sampled_from([None, 5, 50]),
        "origin": st.sampled_from([(), ("claude-code-session",)]),
        "tag": st.sampled_from([(), ("review",)]),
        "verbose": st.booleans(),
    }
)


@given(terms=_QUERY_TERMS, flags=_FLAG_PARAMS)
def test_root_request_and_daemon_reads_lower_identically(terms: tuple[str, ...], flags: dict[str, object]) -> None:
    """The lowering law: one parameter map, one selection, on both surfaces.

    Anti-vacuity: the generated corpus covers the ``lexical``/``semantic``/
    ``similar_text`` combinations that the desugaring exists for, so a surface
    keeping its own copy of it — a ``retrieval_lane`` set on one side only, a
    ``similar_text`` promoted on one side only, or a refusal raised on one
    side and not the other — turns this red.  Deleting the
    ``desugar_retrieval_flags`` delegation from ``root_request`` (or the
    ``lower_cli_query_params`` delegation from ``daemon_reads``) and changing
    either copy is exactly that mutation.
    """
    params: dict[str, object] = {**flags, "query": terms}

    cli_error: Exception | None = None
    cli_spec = None
    try:
        cli_spec = RootModeRequest.from_params(dict(params)).query_spec()
    except click.UsageError as exc:  # the CLI presentation of a refusal
        cli_error = exc

    daemon_error: Exception | None = None
    daemon_spec = None
    try:
        daemon_spec = _cli_query_spec(dict(params))
    except ValueError as exc:  # the protocol presentation of the same refusal
        daemon_error = exc

    assert (cli_error is None) == (daemon_error is None), (
        f"one surface refused and the other did not for {params!r}: {cli_error!r} vs {daemon_error!r}"
    )
    assert cli_spec == daemon_spec


def test_cli_and_daemon_query_selection_pass_through_read_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both executable adapters use the canonical read request after DSL lowering."""

    from polylogue.surfaces.read_contract import ReadRequest

    seen: list[SessionQuerySpec] = []
    normalize = ReadRequest.normalize

    def recording_normalize(params: dict[str, object], *, preset: str | None = None) -> ReadRequest:
        result = normalize(params, preset=preset)
        seen.append(result.selection)
        return result

    monkeypatch.setattr(ReadRequest, "normalize", staticmethod(recording_normalize))
    params: dict[str, object] = {"query": ("repo:polylogue", "typed_only:true"), "limit": 7}

    cli = RootModeRequest.from_params(params).query_spec()
    daemon = _cli_query_spec(params)

    assert seen == [cli, daemon]
    assert cli == daemon
    assert cli.repo_names == ("polylogue",)
    assert cli.typed_only is True


def test_semantic_lane_is_desugared_once_for_cli_and_daemon() -> None:
    request = RootModeRequest.from_params({"query": ("semantic evidence",), "retrieval_lane": "semantic"})
    cli = request.query_spec()
    daemon = _cli_query_spec(request.query_params())
    operation = lower_cli_query(request, limit=5, offset=0)

    assert cli == daemon
    assert cli.similar_text == "semantic evidence"
    assert cli.retrieval_lane == "auto"
    assert cli.query_terms == ()
    params = operation.payload["params"]
    assert isinstance(params, dict)
    assert params["similar_text"] == "semantic evidence"
