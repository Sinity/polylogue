from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from polylogue.cli import query_verbs
from polylogue.cli.root_request import RootModeRequest


def _context_pair(
    *,
    params: dict[str, object] | None = None,
    query_terms: tuple[str, ...] = (),
) -> tuple[click.Context, click.Context]:
    parent = click.Context(click.Command("query"))
    parent.params = {"query_term": query_terms, **(params or {})}
    parent.meta["polylogue_query_terms"] = query_terms
    child = click.Context(click.Command("verb"), parent=parent)
    child.obj = SimpleNamespace()
    return parent, child


def test_parent_helpers_require_parent_context_and_build_request() -> None:
    _, child = _context_pair(
        params={"provider": "chatgpt", "limit": 5},
        query_terms=("alpha", "beta"),
    )

    assert query_verbs._parent_query_terms(child) == ("alpha", "beta")
    request = query_verbs._parent_request(child)
    assert request.query_params()["provider"] == "chatgpt"
    assert request.query_params()["query"] == ("alpha", "beta")

    with pytest.raises(click.UsageError, match="Query verbs must be invoked"):
        query_verbs._require_parent_context(click.Context(click.Command("verb")))


def test_execute_query_verb_dispatches_typed_request() -> None:
    _, child = _context_pair()
    request = RootModeRequest.from_params({"query": ("alpha",)})

    with patch("polylogue.cli.query.execute_query_request") as execute:
        query_verbs._execute_query_verb(child, request)

    execute.assert_called_once_with(child.obj, request)


def test_list_and_count_verbs_update_parent_request() -> None:
    _, child = _context_pair(params={"provider": "chatgpt"}, query_terms=("alpha",))

    wrapped_list = getattr(query_verbs.list_verb.callback, "__wrapped__", None)
    assert callable(wrapped_list)
    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped_list(child, "json", "id,title", 7)

    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_params()["provider"] == "chatgpt"
    assert request.query_params()["output_format"] == "json"
    assert request.query_params()["fields"] == "id,title"
    assert request.query_params()["limit"] == 7
    assert request.query_params()["query"] == ("alpha",)

    wrapped_count = getattr(query_verbs.count_verb.callback, "__wrapped__", None)
    assert callable(wrapped_count)
    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped_count(child)

    count_request = execute.call_args.args[1]
    assert isinstance(count_request, RootModeRequest)
    assert count_request.query_params()["count_only"] is True


def test_stats_verb_toggles_stats_only_and_updates_grouping() -> None:
    _, child = _context_pair(query_terms=("alpha",))
    wrapped = getattr(query_verbs.stats_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped(child, None, None, None)

    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_params()["stats_only"] is True
    assert request.query_params()["query"] == ("alpha",)

    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped(child, "provider", "markdown", 3)

    grouped_request = execute.call_args.args[1]
    assert isinstance(grouped_request, RootModeRequest)
    assert grouped_request.query_params()["stats_only"] is False
    assert grouped_request.query_params()["stats_by"] == "provider"
    assert grouped_request.query_params()["output_format"] == "markdown"
    assert grouped_request.query_params()["limit"] == 3


def test_open_verb_routes_single_id_or_appends_target_terms() -> None:
    _, child = _context_pair(query_terms=())
    wrapped = getattr(query_verbs.open_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped(child, True, ("chatgpt:123",))

    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_params()["open_result"] is True
    assert request.query_params()["print_path"] is True
    assert request.query_params()["conv_id"] == "chatgpt:123"
    assert request.query_params()["query"] == ()

    _, child = _context_pair(query_terms=("existing",))
    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped(child, False, ("more", "terms"))

    appended_request = execute.call_args.args[1]
    assert isinstance(appended_request, RootModeRequest)
    assert appended_request.query_params()["query"] == ("existing", "more", "terms")
    assert appended_request.query_params()["open_result"] is True


def test_delete_verb_updates_force_and_dry_run_flags() -> None:
    _, child = _context_pair(query_terms=("alpha",))
    wrapped = getattr(query_verbs.delete_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped(child, True, False)

    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_params()["delete_matched"] is True
    assert request.query_params()["dry_run"] is True
    assert request.query_params()["force"] is False
    assert request.query_params()["query"] == ("alpha",)
