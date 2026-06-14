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
        params={"origin": "chatgpt-export", "limit": 5},
        query_terms=("alpha", "beta"),
    )

    assert query_verbs._parent_query_terms(child) == ("alpha", "beta")
    request = query_verbs._parent_request(child)
    assert request.query_params()["origin"] == "chatgpt-export"
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
    _, child = _context_pair(params={"origin": "chatgpt-export"}, query_terms=("alpha",))

    wrapped_list = getattr(query_verbs.list_verb.callback, "__wrapped__", None)
    assert callable(wrapped_list)
    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped_list(child, "json", "id,title", 7)

    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_params()["origin"] == "chatgpt-export"
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
        wrapped(child, "origin", "markdown", 3)

    grouped_request = execute.call_args.args[1]
    assert isinstance(grouped_request, RootModeRequest)
    assert grouped_request.query_params()["stats_only"] is False
    assert grouped_request.query_params()["stats_by"] == "origin"
    assert grouped_request.query_params()["output_format"] == "markdown"
    assert grouped_request.query_params()["limit"] == 3


def test_read_verb_summary_dispatches_to_execute_query_verb() -> None:
    """read --view summary (default) routes to _execute_query_verb."""
    _, child = _context_pair(params={"origin": "chatgpt-export"}, query_terms=("alpha",))
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped(
            child,
            "summary",  # view
            "terminal",  # destination
            None,  # output_format
            None,  # out_path
            False,  # export_all
            (),  # message_role
            None,  # message_type
            None,  # limit
            0,  # offset
            False,  # no_code_blocks
            False,  # no_tool_calls
            False,  # no_tool_outputs
            False,  # no_file_reads
            False,  # prose_only
            None,  # fields
        )

    execute.assert_called_once()
    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_params()["origin"] == "chatgpt-export"


def test_read_verb_all_invokes_run_bulk_export() -> None:
    """read --all routes to run_bulk_export."""
    _, child = _context_pair(params={"origin": "claude-code-session"}, query_terms=("alpha",))
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.bulk_export.run_bulk_export") as run_export:
        wrapped(
            child,
            "summary",  # view
            "terminal",  # destination
            "json",  # output_format
            None,  # out_path
            True,  # export_all
            (),  # message_role
            None,  # message_type
            None,  # limit
            0,  # offset
            False,  # no_code_blocks
            False,  # no_tool_calls
            False,  # no_tool_outputs
            False,  # no_file_reads
            False,  # prose_only
            None,  # fields
        )

    run_export.assert_called_once()
    assert run_export.call_args.kwargs["output_format"] == "json"


def test_resolve_target_session_id_uses_explicit_conv_id() -> None:
    request = RootModeRequest.from_params({"conv_id": "claude-code:abc123"})
    assert query_verbs._resolve_target_session_id(request) == "claude-code:abc123"


def test_resolve_target_session_id_returns_none_without_filters_or_latest() -> None:
    request = RootModeRequest.from_params({})
    assert query_verbs._resolve_target_session_id(request) is None


def test_resolve_target_session_id_resolves_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    """#1626: ``--latest`` must resolve to the most recent conv without an explicit id."""
    request = RootModeRequest.from_params({"latest": True})

    captured_limits: list[int | None] = []

    async def fake_list_summaries(self: object, repo: object) -> list[SimpleNamespace]:
        captured_limits.append(getattr(self, "limit", None))
        return [SimpleNamespace(id="claude-code:latest-conv-id")]

    monkeypatch.setattr(
        "polylogue.archive.query.spec.SessionQuerySpec.list_summaries",
        fake_list_summaries,
    )

    class _API:
        config = SimpleNamespace()

        async def __aenter__(self) -> _API:
            return self

        async def __aexit__(self, *exc: object) -> None: ...

    monkeypatch.setattr("polylogue.api.Polylogue.open", lambda **_: _API())

    result = query_verbs._resolve_target_session_id(request)

    assert result == "claude-code:latest-conv-id"
    assert captured_limits == [1]


def test_delete_verb_updates_force_and_dry_run_flags() -> None:
    _, child = _context_pair(query_terms=("alpha",))
    wrapped = getattr(query_verbs.delete_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    # Signature is delete_verb(ctx, dry_run, yes_flag, all_flag, force).
    # Dry-run previews the SAME full pre-resolved id set the real delete acts on
    # via execute_delete_by_session_ids(dry_run=True). It must NOT route through
    # _execute_query_verb, which re-runs the query at the default limit of 20 and
    # would preview fewer sessions than --yes --all deletes (#1873). force is True
    # so the preview never triggers the interactive confirmation prompt.
    with (
        patch(
            "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
            return_value=["alpha-id"],
        ) as resolve,
        patch("polylogue.cli.archive_query.execute_delete_by_session_ids") as execute,
        patch("polylogue.cli.query_verbs._execute_query_verb") as legacy,
    ):
        wrapped(child, True, False, False, False)

    legacy.assert_not_called()
    resolve.assert_called_once()
    args, kwargs = execute.call_args
    assert list(args[1]) == ["alpha-id"]
    assert kwargs.get("dry_run") is True
    assert kwargs.get("force") is True
