from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest

from polylogue.archive.session.domain_models import SessionSummary
from polylogue.archive.viewport import READ_VIEW_PROFILE_BY_ID, READ_VIEW_PROFILES, read_view_choices
from polylogue.cli import query_verbs, read_view_handlers
from polylogue.cli.read_view_handlers import ReadViewInvocation
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.context.compiler import ContextImage, ContextSegment, ContextSpec
from polylogue.surfaces.payloads import PublicRefResolutionPayload
from polylogue.surfaces.projection_spec import projection_from_views


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


def test_read_all_and_analyze_count_update_parent_request() -> None:
    _, child = _context_pair(params={"origin": "chatgpt-export"}, query_terms=("alpha",))

    wrapped_read = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped_read)
    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        with patch("polylogue.cli.query_verbs.run_query_set_read_view") as query_set_read:
            wrapped_read(
                child,
                **_read_verb_kwargs(
                    view="summary",
                    output_format="json",
                    all_matches=True,
                    limit=7,
                    fields="id,title",
                ),
            )

    query_set_read.assert_not_called()
    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_params()["origin"] == "chatgpt-export"
    assert request.query_params()["limit"] == 7
    assert request.query_params()["query"] == ("alpha",)
    assert request.query_params()["list_mode"] is True
    assert request.query_params()["output_format"] == "json"

    out_path = "/tmp/polylogue-sessions.json"
    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        with patch("polylogue.cli.query_verbs.run_query_set_read_view") as query_set_read:
            wrapped_read(
                child,
                **_read_verb_kwargs(
                    view="summary",
                    destination="file",
                    output_format="json",
                    out_path=out_path,
                    all_matches=True,
                ),
            )

    query_set_read.assert_not_called()
    file_request = execute.call_args.args[1]
    assert isinstance(file_request, RootModeRequest)
    assert file_request.query_params()["list_mode"] is True
    assert file_request.query_params()["output"] == out_path

    wrapped_analyze = getattr(query_verbs.analyze_verb.callback, "__wrapped__", None)
    assert callable(wrapped_analyze)
    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped_analyze(child, True, None, False, False, None, "linear", False, False, None, None)

    count_request = execute.call_args.args[1]
    assert isinstance(count_request, RootModeRequest)
    assert count_request.query_params()["count_only"] is True

    with pytest.raises(click.UsageError, match="does not support --limit"):
        wrapped_analyze(child, True, None, False, False, None, "linear", False, False, None, 5)


def test_read_direct_ref_emits_shared_resolution_payload(capsys: pytest.CaptureFixture[str]) -> None:
    _, child = _context_pair()
    child.obj = SimpleNamespace(polylogue=SimpleNamespace(resolve_ref=lambda ref: f"resolve:{ref}"))
    wrapped_read = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped_read)
    payload = PublicRefResolutionPayload(
        ref="session:abc",
        normalized_ref="session:abc",
        kind="session",
        resolved=True,
        payload_kind="session-summary",
        payload={"id": "abc", "title": "Resolved"},
    )

    with patch("polylogue.cli.query_verbs.run_coroutine_sync", return_value=payload) as run_sync:
        wrapped_read(
            ctx=child,
            view="summary",
            destination="terminal",
            output_format="json",
            out_path=None,
            all_matches=False,
            limit=None,
            offset=0,
            window_hours=24,
            repo_path=None,
            since_hours=2,
            confidence_threshold=0.3,
            github_api=True,
            otlp=False,
            related_limit=5,
            project_path=None,
            project_repo=None,
            since=None,
            until=None,
            context_origin=None,
            context_query=None,
            max_sessions=5,
            max_tokens=None,
            include_assertions=False,
            no_redact=False,
            fields=None,
            first_only=False,
            show_views=False,
            ref="session:abc",
        )

    run_sync.assert_called_once()
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["mode"] == "ref-resolution"
    assert emitted["normalized_ref"] == "session:abc"
    assert emitted["payload"]["id"] == "abc"


def test_analyze_verb_toggles_stats_only_and_updates_grouping() -> None:
    _, child = _context_pair(query_terms=("alpha",))
    wrapped = getattr(query_verbs.analyze_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped(child, False, None, False, False, None, "linear", False, False, None, None)

    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_params()["stats_only"] is True
    assert request.query_params()["query"] == ("alpha",)

    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped(child, False, "origin", False, False, None, "linear", False, False, "markdown", 3)

    grouped_request = execute.call_args.args[1]
    assert isinstance(grouped_request, RootModeRequest)
    assert grouped_request.query_params()["stats_only"] is False
    assert grouped_request.query_params()["stats_by"] == "origin"
    assert grouped_request.query_params()["output_format"] == "markdown"
    assert grouped_request.query_params()["limit"] == 3


def test_select_verb_invokes_query_backed_selector() -> None:
    _, child = _context_pair(params={"origin": "chatgpt-export"}, query_terms=("alpha",))
    wrapped = getattr(query_verbs.select_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.select.run_select") as run_select:
        wrapped(child, 7, "title", False)

    request = run_select.call_args.args[1]
    assert run_select.call_args.args[0] is child.obj
    assert isinstance(request, RootModeRequest)
    assert request.query_params()["origin"] == "chatgpt-export"
    assert request.query_params()["query"] == ("alpha",)
    assert run_select.call_args.kwargs == {"limit": 7, "print_field": "title"}


def test_select_verb_json_flag_overrides_print_field() -> None:
    _, child = _context_pair(query_terms=("alpha",))
    wrapped = getattr(query_verbs.select_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.select.run_select") as run_select:
        wrapped(child, 5, "id", True)

    assert run_select.call_args.kwargs == {"limit": 5, "print_field": "json"}


def test_read_view_click_choices_come_from_view_profiles() -> None:
    # --view accepts a comma-separated list of views (for multi-view composition),
    # so it is a free string validated against the profile registry at runtime
    # rather than a single-value click.Choice.
    option = next(param for param in query_verbs.read_verb.params if param.name == "view")

    assert not isinstance(option.type, click.Choice)
    assert read_view_choices() == query_verbs._READ_VIEWS


def test_read_view_handlers_cover_view_profiles() -> None:
    assert read_view_handlers.read_view_handler_ids() == read_view_choices()
    read_view_handlers.validate_read_view_handler_registry()


def test_read_view_completion_comes_from_view_profiles() -> None:
    option = next(param for param in query_verbs.read_verb.params if param.name == "view")

    items = option.shell_complete(click.Context(query_verbs.read_verb), "rec")

    assert [item.value for item in items] == []


def test_read_format_click_choices_come_from_view_profiles() -> None:
    option = next(param for param in query_verbs.read_verb.params if "--format" in param.opts)
    expected = tuple(sorted({fmt for profile in READ_VIEW_PROFILES for fmt in profile.formats}))

    assert isinstance(option.type, click.Choice)
    assert tuple(option.type.choices) == expected


def test_read_format_completion_comes_from_selected_view_profile() -> None:
    option = next(param for param in query_verbs.read_verb.params if "--format" in param.opts)
    context = click.Context(query_verbs.read_verb)
    context.params["view"] = "raw"

    items = option.shell_complete(context, "")

    assert [item.value for item in items] == list(READ_VIEW_PROFILE_BY_ID["raw"].formats)
    assert items[0].help == "Supported by read --view raw"


def _read_verb_kwargs(**overrides: object) -> dict[str, object]:
    """Full default kwargs for the read_verb callback (keyword-robust to signature growth)."""
    defaults: dict[str, object] = {
        "view": "summary",
        "destination": "terminal",
        "output_format": None,
        "out_path": None,
        "all_matches": False,
        "limit": None,
        "offset": 0,
        "window_hours": 24,
        "repo_path": None,
        "since_hours": 2,
        "confidence_threshold": 0.3,
        "github_api": True,
        "otlp": False,
        "related_limit": 5,
        "project_path": None,
        "project_repo": None,
        "since": None,
        "until": None,
        "context_origin": None,
        "context_query": None,
        "max_sessions": 5,
        "max_tokens": None,
        "include_assertions": False,
        "no_redact": False,
        "fields": None,
        "first_only": False,
        "show_views": False,
    }
    defaults.update(overrides)
    return defaults


def _continue_verb_kwargs(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "destination": "terminal",
        "out_path": None,
        "candidates": False,
        "repo_path": None,
        "cwd": None,
        "recent_files": (),
        "candidate_limit": 10,
        "output_format": None,
    }
    defaults.update(overrides)
    return defaults


def test_read_verb_summary_dispatches_to_execute_query_verb() -> None:
    """read --view summary (default) routes through the handler registry."""
    _, child = _context_pair(params={"origin": "chatgpt-export"}, query_terms=("alpha",))
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with (
        patch("polylogue.cli.query_verbs._resolve_query_action_session_id", return_value=None),
        patch("polylogue.cli.read_views.standard.execute_query_request") as execute,
    ):
        wrapped(child, **_read_verb_kwargs(view="summary"))

    execute.assert_called_once()
    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_params()["origin"] == "chatgpt-export"


def test_read_verb_summary_exact_ref_dispatches_direct_read() -> None:
    """Exact session refs read the selected session instead of re-running search."""
    _, child = _context_pair(query_terms=("session:codex-session:abc123",))
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.read_views.standard.execute_query_request") as execute:
        wrapped(child, **_read_verb_kwargs(view="summary", output_format="json"))

    execute.assert_called_once()
    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_terms == ()
    assert request.params["conv_id"] == "codex-session:abc123"
    assert request.params["output_format"] == "json"


def test_read_verb_summary_preserves_search_within_exact_ref() -> None:
    """Exact refs combined with FTS terms still search within the session."""
    _, child = _context_pair(query_terms=("session:codex-session:abc123", "needle"))
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with (
        patch("polylogue.cli.query_verbs._resolve_query_action_session_id", return_value="codex-session:abc123"),
        patch("polylogue.cli.read_views.standard.execute_query_request") as execute,
    ):
        wrapped(child, **_read_verb_kwargs(view="summary", output_format="json"))

    execute.assert_called_once()
    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_terms == ("session:codex-session:abc123", "needle")
    assert "conv_id" not in request.params
    assert request.params["output_format"] == "json"


def test_read_verb_all_non_summary_invokes_query_set_read_view() -> None:
    """read --all with a concrete non-summary view routes to query-set read."""
    _, child = _context_pair(params={"origin": "claude-code-session"}, query_terms=("alpha",))
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.query_verbs.run_query_set_read_view") as run_read_set:
        wrapped(child, **_read_verb_kwargs(view="transcript", output_format="json", all_matches=True))

    run_read_set.assert_called_once()
    assert run_read_set.call_args.kwargs["output_format"] == "json"


def test_read_verb_context_composes_preamble_not_passthrough() -> None:
    """read --view context routes to the context preamble composer."""
    _, child = _context_pair(params={"conv_id": "claude-code:abc123"}, query_terms=())
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with (
        patch("polylogue.context.preamble.compose_context_preamble", return_value="{}") as compose,
        patch("polylogue.cli.read_views.standard.execute_query_request") as execute,
        patch("polylogue.cli.read_views.context.deliver_content") as deliver,
    ):
        wrapped(child, **_read_verb_kwargs(view="context", related_limit=3))

    execute.assert_not_called()
    compose.assert_called_once()
    assert compose.call_args.kwargs["session_id"] == "claude-code:abc123"
    assert compose.call_args.kwargs["related_limit"] == 3
    deliver.assert_called_once()


def test_read_verb_context_image_invokes_pack_view() -> None:
    """read --view context-image compiles a ContextImage via context_image_payload."""
    from polylogue.context.compiler import ContextImage, ContextSpec

    _, child = _context_pair(query_terms=())
    child.obj.polylogue = SimpleNamespace(context_image_payload=MagicMock(name="context_image_payload"))
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    image = ContextImage(spec=ContextSpec(seed_query="cost", read_views=("messages",)), segments=())
    with (
        patch("polylogue.cli.query_verbs.run_coroutine_sync", return_value=image),
        patch("polylogue.cli.read_views.base.deliver_content") as deliver,
    ):
        wrapped(child, **_read_verb_kwargs(view="context-image", context_query="cost", max_sessions=3))

    child.obj.polylogue.context_image_payload.assert_called_once()
    kwargs = child.obj.polylogue.context_image_payload.call_args.kwargs
    assert kwargs["query"] == "cost"
    assert kwargs["max_sessions"] == 3
    deliver.assert_called_once()
    delivered = deliver.call_args.args[1]
    assert "- Selection query: cost" in delivered
    assert "- Selection limit: 3" in delivered
    assert "- Projection families: context, messages, assertions" in delivered
    assert "- Body policy: full" in delivered
    assert "- Render: markdown to terminal" in delivered
    assert "- Render layout: context-image" in delivered


def test_read_verb_context_image_projection_spec_records_resolved_refs() -> None:
    """Multi-view context images should expose resolved archive refs in the projection spec."""
    _, child = _context_pair(query_terms=("repo:polylogue",))
    child.obj.config = SimpleNamespace()
    child.obj.polylogue = SimpleNamespace(compile_context=MagicMock(name="compile_context"))
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    image = ContextImage(
        spec=ContextSpec(seed_refs=("session:codex-session:abc123",), read_views=("temporal", "chronicle")),
        segments=(),
    )

    with (
        patch(
            "polylogue.cli.query_verbs._resolve_query_action_session_ids",
            return_value=["codex-session:abc123"],
        ) as resolve_session_ids,
        patch("polylogue.cli.query_verbs.run_coroutine_sync", return_value=image),
        patch("polylogue.cli.read_views.base.deliver_content") as deliver,
    ):
        wrapped(child, **_read_verb_kwargs(view="temporal,chronicle", output_format="json", limit=2))

    assert resolve_session_ids.call_args.kwargs["limit"] == 2
    payload = json.loads(deliver.call_args.args[1])
    assert payload["projection_spec"]["selection"]["query"] == "repo:polylogue"
    assert payload["projection_spec"]["selection"]["limit"] == 2
    assert payload["projection_spec"]["selection"]["refs"] == ["session:codex-session:abc123"]
    assert payload["projection_spec"]["projection"]["families"] == [
        "temporal",
        "sessions",
        "chronicle",
        "messages",
    ]


def test_context_image_markdown_renderer_adds_document_structure() -> None:
    """Context-image Markdown should be readable as one composed packet."""
    from polylogue.context.compiler import ContextImage, ContextOmission, ContextSegment, ContextSpec

    projection_spec = projection_from_views(
        ("temporal", "chronicle"),
        format="json",
        destination="stdout",
        max_tokens=2000,
        query="repo:polylogue",
        origin="claude-code-session",
        since="2026-06-01",
        until="2026-06-30",
        project_path="/workspace/polylogue",
        project_repo="github.com/Sinity/polylogue",
        limit=2,
        edge_limit=3,
        body_limit=7,
        body_offset=2,
        neighbor_limit=4,
        neighbor_window_hours=12,
    )
    projection_spec = projection_spec.model_copy(
        update={
            "selection": projection_spec.selection.model_copy(
                update={"refs": ("session:codex-session:abc123", "session:codex-session:def456")}
            )
        }
    )
    image = ContextImage(
        spec=ContextSpec(purpose="handoff", seed_refs=("session:abc",), read_views=("temporal", "chronicle")),
        projection_spec=projection_spec,
        segments=(
            ContextSegment(
                segment_id="read-view:abc:temporal",
                kind="read_view",
                title="Temporal Evidence",
                markdown="# Temporal Evidence\n\n- Events: 2\n",
                payload_kind="temporal",
                token_estimate=5,
            ),
        ),
        omitted=(
            ContextOmission(
                ref="session:def",
                view="chronicle",
                reason="budget",
                detail="segment exceeded the requested context token budget",
            ),
        ),
        token_estimate=5,
    )

    rendered = query_verbs._render_context_image_markdown(image)

    assert rendered.startswith("# Context Image\n")
    assert "- Purpose: handoff" in rendered
    assert "- Views: temporal, chronicle" in rendered
    assert "- Selection query: repo:polylogue" in rendered
    assert "- Selection origin: claude-code-session" in rendered
    assert "- Selection since: 2026-06-01" in rendered
    assert "- Selection until: 2026-06-30" in rendered
    assert "- Selection project path: /workspace/polylogue" in rendered
    assert "- Selection project repo: github.com/Sinity/polylogue" in rendered
    assert "- Selection limit: 2" in rendered
    assert "- Selection refs: session:codex-session:abc123, session:codex-session:def456" in rendered
    assert "- Projection families: temporal, sessions, chronicle, messages" in rendered
    assert "- Body policy: authored-dialogue" in rendered
    assert "- Projection max tokens: 2000" in rendered
    assert "- Projection edge limit: 3" in rendered
    assert "- Projection body limit: 7" in rendered
    assert "- Projection body offset: 2" in rendered
    assert "- Projection neighbor limit: 4" in rendered
    assert "- Projection neighbor window hours: 12" in rendered
    assert "- Projection redact paths: true" in rendered
    assert "- Render: json to stdout" in rendered
    assert "- Render layout: standard" in rendered
    assert "- Segments: 1" in rendered
    assert "- Omissions: 1" in rendered
    assert "## 1. Temporal Evidence" in rendered
    assert "_(kind=temporal; tokens=5)_" in rendered
    assert "## Omitted" in rendered
    assert "session:def [budget]" in rendered


def test_continue_verb_compiles_context_from_query_unit_recipe() -> None:
    """``continue`` composes messages plus terminal query-unit DSL segments."""
    _, child = _context_pair(query_terms=("id:codex-session:abc123",))
    wrapped = getattr(query_verbs.continue_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    captured_specs: list[ContextSpec] = []
    image = ContextImage(
        spec=ContextSpec(seed_refs=("session:codex-session:abc123",), read_views=("messages",)), segments=()
    )

    async def compile_context(spec: ContextSpec) -> ContextImage:
        captured_specs.append(spec)
        return image

    async def get_session(session_id: str) -> object:
        assert session_id == "codex-session:abc123"
        return object()

    child.obj.polylogue = SimpleNamespace(compile_context=compile_context, get_session=get_session)
    with (
        patch("polylogue.cli.query_verbs.deliver_content") as deliver,
    ):
        wrapped(child, **_continue_verb_kwargs(destination="clipboard"))

    called_spec = captured_specs[0]
    assert called_spec.read_views == ("messages",)
    assert called_spec.unit_queries == (
        "runs where session.id:codex-session:abc123",
        "observed-events where session.id:codex-session:abc123",
        "context-snapshots where session.id:codex-session:abc123",
        "actions where session.id:codex-session:abc123",
    )
    deliver.assert_called_once()


def test_continue_verb_rejects_ambiguous_ranked_results() -> None:
    """``find QUERY then continue`` requires a singleton query result."""
    _, child = _context_pair(query_terms=("needle",))
    child.obj = SimpleNamespace(config=SimpleNamespace())
    wrapped = getattr(query_verbs.continue_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    def _close_and_return(coro: object) -> list[str]:
        close = getattr(coro, "close", None)
        if callable(close):
            close()
        return ["session-1", "session-2"]

    with patch("polylogue.cli.query_verbs.run_coroutine_sync", side_effect=_close_and_return):
        with pytest.raises(click.UsageError, match="Narrow the query to one session"):
            wrapped(child, **_continue_verb_kwargs())


def test_continue_verb_json_emits_context_image(capsys: pytest.CaptureFixture[str]) -> None:
    """``continue --format json`` exposes the shared ContextImage payload."""
    _, child = _context_pair(query_terms=("id:codex-session:abc123",))
    spec = ContextSpec(
        purpose="continue",
        seed_refs=("session:codex-session:abc123",),
        read_views=("messages",),
        unit_queries=("runs where session.id:codex-session:abc123",),
    )
    image = ContextImage(
        spec=spec,
        segments=(
            ContextSegment(
                segment_id="query-unit:abc123",
                kind="query_unit",
                title="Query: runs",
                markdown="Temporal rows.",
                token_estimate=3,
            ),
        ),
        token_estimate=3,
    )
    seen: dict[str, ContextSpec] = {}

    async def compile_context(spec: ContextSpec) -> ContextImage:
        seen["spec"] = spec
        return image

    async def get_session(session_id: str) -> object:
        assert session_id == "codex-session:abc123"
        return object()

    child.obj = SimpleNamespace(polylogue=SimpleNamespace(compile_context=compile_context, get_session=get_session))
    wrapped = getattr(query_verbs.continue_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    wrapped(child, **_continue_verb_kwargs(output_format="json"))

    emitted = json.loads(capsys.readouterr().out)
    assert emitted["spec"]["purpose"] == "continue"
    assert emitted["spec"]["seed_refs"] == ["session:codex-session:abc123"]
    assert emitted["spec"]["read_views"] == ["messages"]
    assert emitted["segments"][0]["kind"] == "query_unit"
    spec = seen["spec"]
    assert spec.purpose == "continue"
    assert spec.seed_refs == ("session:codex-session:abc123",)
    assert spec.read_views == ("messages",)
    assert spec.unit_queries == (
        "runs where session.id:codex-session:abc123",
        "observed-events where session.id:codex-session:abc123",
        "context-snapshots where session.id:codex-session:abc123",
        "actions where session.id:codex-session:abc123",
    )


def test_continue_candidates_ranks_context_without_session_resolution() -> None:
    """``continue --candidates`` moves candidate ranking onto the continuation action."""
    _, child = _context_pair(params={"output_format": "json"}, query_terms=())
    child.obj.polylogue = SimpleNamespace()
    wrapped = getattr(query_verbs.continue_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with (
        patch("polylogue.cli.query_verbs._emit_continue_candidates") as emit_candidates,
        patch("polylogue.cli.query_verbs._resolve_target_session_id") as resolve_target,
    ):
        wrapped(
            child,
            **_continue_verb_kwargs(
                candidates=True,
                repo_path="/workspace/polylogue",
                cwd="/workspace/polylogue",
                recent_files=("polylogue/cli/query_verbs.py",),
                candidate_limit=3,
                output_format="json",
            ),
        )

    resolve_target.assert_not_called()
    emit_candidates.assert_called_once_with(
        child.obj,
        query_verbs._parent_request(child),
        repo_path="/workspace/polylogue",
        cwd="/workspace/polylogue",
        recent_files=("polylogue/cli/query_verbs.py",),
        limit=3,
        output_format="json",
    )


def test_continue_candidate_options_require_candidate_mode() -> None:
    _, child = _context_pair(query_terms=("id:codex-session:abc123",))
    wrapped = getattr(query_verbs.continue_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with pytest.raises(click.UsageError, match="only valid with continue --candidates"):
        wrapped(child, **_continue_verb_kwargs(repo_path="/workspace/polylogue"))


def test_resolve_target_session_id_uses_query_terms(workspace_env: dict[str, Path]) -> None:
    """Single-session verbs resolve ``find id:... then ...`` through the query DSL."""
    from polylogue.config import Config
    from tests.infra.storage_records import SessionBuilder

    archive_root = workspace_env["archive_root"]
    stored = (
        SessionBuilder(archive_root / "index.db", "resolve-query-target").provider("codex").title("query target").save()
    )
    request = RootModeRequest.from_params(
        {
            "_config": Config(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
                render_root=archive_root / "render",
                sources=[],
            ),
            "query": ("title:query",),
        }
    )

    assert query_verbs._resolve_target_session_id(request) == f"{stored.origin.value}:{stored.native_id}"


def test_read_view_rejects_format_outside_selected_profile() -> None:
    env = SimpleNamespace(polylogue=SimpleNamespace())

    with pytest.raises(click.UsageError, match="read --view raw does not support --format markdown"):
        read_view_handlers.run_read_view(
            cast(AppEnv, env),
            RootModeRequest.from_params({}),
            ReadViewInvocation(
                view="raw",
                session_id="codex-session:abc",
                output_format="markdown",
                destination="terminal",
                out_path=None,
            ),
        )


def test_read_view_temporal_projects_selected_summaries(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    config = Config(
        archive_root=tmp_path,
        db_path=tmp_path / "index.db",
        render_root=tmp_path / "render",
        sources=[],
    )
    env = SimpleNamespace(config=config)
    summaries = [
        SessionSummary.model_validate(
            {
                "id": "codex-session:abc",
                "origin": "codex-session",
                "title": "Temporal slice",
                "created_at": datetime(2026, 6, 30, 8, 0, tzinfo=UTC),
            }
        ),
        SessionSummary.model_validate(
            {
                "id": "claude-code-session:def",
                "origin": "claude-code-session",
                "title": "Follow-up",
                "created_at": datetime(2026, 6, 30, 9, 0, tzinfo=UTC),
            }
        ),
    ]

    with (
        patch("polylogue.cli.query._create_query_vector_provider", return_value=None),
        patch(
            "polylogue.archive.query.spec.SessionQuerySpec.list_summaries",
            new=AsyncMock(return_value=summaries),
        ) as list_summaries,
        patch(
            "polylogue.cli.read_views.standard._message_temporal_events_for_summaries",
            return_value=([], ()),
        ) as message_events,
        patch(
            "polylogue.cli.read_views.standard._action_temporal_events_for_summaries",
            return_value=([], ()),
        ) as action_events,
    ):
        read_view_handlers.run_read_view(
            cast(AppEnv, env),
            RootModeRequest.from_params({"query": ("repo:polylogue",), "limit": 2}),
            ReadViewInvocation(
                view="temporal",
                session_id=None,
                output_format="json",
                destination="terminal",
                out_path=None,
            ),
        )

    list_summaries.assert_awaited_once()
    message_events.assert_called_once()
    action_events.assert_called_once()
    payload = json.loads(capsys.readouterr().out)
    window = payload["temporal_window"]
    assert window["event_count"] == 2
    assert window["family_counts"] == {"archive-session": 2}
    assert window["kind_counts"] == {"session": 2}
    assert [event["source_ref"] for event in window["events"]] == [
        "session:codex-session:abc",
        "session:claude-code-session:def",
    ]


def test_read_view_temporal_includes_bounded_message_events(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from polylogue.surfaces.temporal_evidence import TemporalEvidenceEvent

    config = Config(
        archive_root=tmp_path,
        db_path=tmp_path / "index.db",
        render_root=tmp_path / "render",
        sources=[],
    )
    env = SimpleNamespace(config=config)
    summaries = [
        SessionSummary.model_validate(
            {
                "id": "codex-session:abc",
                "origin": "codex-session",
                "title": "Temporal slice",
                "created_at": datetime(2026, 6, 30, 8, 0, tzinfo=UTC),
                "message_count": 12,
            }
        )
    ]
    message_event = TemporalEvidenceEvent(
        event_id="message:codex-session:abc:m1:message",
        occurred_at=datetime(2026, 6, 30, 8, 1, tzinfo=UTC),
        family="archive-message",
        kind="message",
        label="user message #1",
        source_ref="message:codex-session:abc:m1",
        evidence_refs=("session:codex-session:abc", "message:codex-session:abc:m1"),
        phase="user",
    )
    action_event = TemporalEvidenceEvent(
        event_id="action:codex-session:abc:m2:tool:action",
        occurred_at=datetime(2026, 6, 30, 8, 2, tzinfo=UTC),
        family="archive-action",
        kind="shell",
        label="shell via bash",
        source_ref="action:codex-session:abc:m2:tool",
        evidence_refs=(
            "session:codex-session:abc",
            "message:codex-session:abc:m2",
            "action:codex-session:abc:m2:tool",
        ),
        phase="shell",
    )

    with (
        patch("polylogue.cli.query._create_query_vector_provider", return_value=None),
        patch(
            "polylogue.archive.query.spec.SessionQuerySpec.list_summaries",
            new=AsyncMock(return_value=summaries),
        ),
        patch(
            "polylogue.cli.read_views.standard._message_temporal_events_for_summaries",
            return_value=([message_event], ("message_events_capped",)),
        ),
        patch(
            "polylogue.cli.read_views.standard._action_temporal_events_for_summaries",
            return_value=([action_event], ("action_events_capped",)),
        ),
    ):
        read_view_handlers.run_read_view(
            cast(AppEnv, env),
            RootModeRequest.from_params({"query": ("repo:polylogue",), "limit": 1}),
            ReadViewInvocation(
                view="temporal",
                session_id=None,
                output_format="json",
                destination="terminal",
                out_path=None,
            ),
        )

    window = json.loads(capsys.readouterr().out)["temporal_window"]
    assert window["event_count"] == 3
    assert window["family_counts"] == {"archive-action": 1, "archive-message": 1, "archive-session": 1}
    assert window["kind_counts"] == {"message": 1, "session": 1, "shell": 1}
    assert "message_events_capped" in window["caveats"]
    assert "action_events_capped" in window["caveats"]
    assert [event["source_ref"] for event in window["events"][1:]] == [
        "message:codex-session:abc:m1",
        "action:codex-session:abc:m2:tool",
    ]


def test_read_view_temporal_batches_session_scope_expression() -> None:
    from polylogue.cli.read_views.standard import _session_scope_for_summaries

    summaries = [
        SessionSummary.model_validate({"id": "codex-session:a", "origin": "codex-session"}),
        SessionSummary.model_validate({"id": "claude-code-session:b", "origin": "claude-code-session"}),
    ]

    assert _session_scope_for_summaries(summaries) == "session:codex-session:a OR session:claude-code-session:b"


def test_read_view_temporal_builder_records_phase_timings(tmp_path: Path) -> None:
    from polylogue.cli.read_views.standard import build_read_temporal_window

    config = Config(
        archive_root=tmp_path,
        db_path=tmp_path / "index.db",
        render_root=tmp_path / "render",
        sources=[],
    )
    summaries = [
        SessionSummary.model_validate(
            {
                "id": "codex-session:abc",
                "origin": "codex-session",
                "title": "Temporal slice",
                "created_at": datetime(2026, 6, 30, 8, 0, tzinfo=UTC),
            }
        )
    ]
    phases: list[tuple[str, float, Mapping[str, object]]] = []

    with (
        patch("polylogue.cli.query._create_query_vector_provider", return_value=None),
        patch(
            "polylogue.archive.query.spec.SessionQuerySpec.list_summaries",
            new=AsyncMock(return_value=summaries),
        ),
        patch(
            "polylogue.cli.read_views.standard._message_temporal_events_for_summaries",
            return_value=([], ()),
        ),
        patch(
            "polylogue.cli.read_views.standard._action_temporal_events_for_summaries",
            return_value=([], ()),
        ),
    ):
        window = build_read_temporal_window(
            config,
            RootModeRequest.from_params({"query": ("repo:polylogue",), "limit": 1}),
            phase_recorder=lambda name, elapsed_ms, details: phases.append((name, elapsed_ms, details)),
        )

    assert window.event_count == 1
    assert [name for name, _elapsed_ms, _details in phases] == [
        "prepare",
        "select_sessions",
        "project_sessions",
        "project_messages",
        "project_actions",
        "build_window",
    ]
    assert all(elapsed_ms >= 0 for _name, elapsed_ms, _details in phases)
    assert phases[1][2]["session_count"] == 1
    assert phases[-1][2]["family_counts"] == {"archive-session": 1}


def test_read_view_registry_builds_typed_view_options() -> None:
    options = read_view_handlers.read_view_options_for_view(
        "context-image",
        {
            "project_path": "/workspace/polylogue",
            "project_repo": "github.com/Sinity/polylogue",
            "context_query": "route contracts",
            "max_sessions": 3,
            "no_redact": True,
        },
    )

    assert isinstance(options, read_view_handlers.ReadViewContextImageOptions)
    assert options.project_path == "/workspace/polylogue"
    assert options.project_repo == "github.com/Sinity/polylogue"
    assert options.query == "route contracts"
    assert options.max_sessions == 3
    assert options.no_redact is True


def test_read_view_registry_builds_chronicle_edge_limit() -> None:
    options = read_view_handlers.read_view_options_for_view("chronicle", {"limit": 3})

    assert isinstance(options, read_view_handlers.ReadViewChronicleOptions)
    assert options.edge_limit == 3


def test_read_chronicle_uses_projection_spec_edge_limit() -> None:
    projection_spec = projection_from_views(("chronicle",), edge_limit=3)

    with (
        patch("polylogue.cli.read_views.chronicle.build_read_chronicle_payload") as build_payload,
        patch("polylogue.cli.read_views.chronicle.render_chronicle_markdown", return_value="chronicle\n"),
        patch("polylogue.cli.read_views.chronicle.deliver_content") as deliver,
    ):
        read_view_handlers.run_read_view(
            cast(AppEnv, SimpleNamespace(config=SimpleNamespace())),
            RootModeRequest.from_params({}),
            ReadViewInvocation(
                view="chronicle",
                session_id=None,
                output_format="markdown",
                destination="terminal",
                out_path=None,
                projection_spec=projection_spec,
            ),
        )

    assert build_payload.call_args.kwargs["edge_limit"] == 3
    deliver.assert_called_once()


def test_read_messages_uses_projection_spec_body_window() -> None:
    projection_spec = projection_from_views(("messages",), body_limit=7, body_offset=2)

    with patch("polylogue.cli.messages.run_messages") as run_messages:
        read_view_handlers.run_read_view(
            cast(AppEnv, SimpleNamespace(config=SimpleNamespace())),
            RootModeRequest.from_params({}),
            ReadViewInvocation(
                view="messages",
                session_id="session-1",
                output_format="json",
                destination="terminal",
                out_path=None,
                projection_spec=projection_spec,
            ),
        )

    assert run_messages.call_args.kwargs["session_id"] == "session-1"
    assert run_messages.call_args.kwargs["limit"] == 7
    assert run_messages.call_args.kwargs["offset"] == 2


def test_read_neighbors_uses_projection_spec_neighbor_policy() -> None:
    projection_spec = projection_from_views(("neighbors",), neighbor_limit=4, neighbor_window_hours=12)
    polylogue = SimpleNamespace(neighbor_candidates=MagicMock(name="neighbor_candidates"))

    with (
        patch("polylogue.api.sync.bridge.run_coroutine_sync", return_value=[]),
        patch("polylogue.cli.read_views.neighbors.deliver_content"),
    ):
        read_view_handlers.run_read_view(
            cast(AppEnv, SimpleNamespace(polylogue=polylogue)),
            RootModeRequest.from_params({}),
            ReadViewInvocation(
                view="neighbors",
                session_id="session-1",
                output_format="text",
                destination="terminal",
                out_path=None,
                projection_spec=projection_spec,
            ),
        )

    assert polylogue.neighbor_candidates.call_args.kwargs["session_id"] == "session-1"
    assert polylogue.neighbor_candidates.call_args.kwargs["limit"] == 4
    assert polylogue.neighbor_candidates.call_args.kwargs["window_hours"] == 12


def test_explicit_read_view_options_reports_command_line_values_only() -> None:
    ctx = click.Context(query_verbs.read_verb)
    ctx.set_parameter_source("related_limit", click.core.ParameterSource.COMMANDLINE)
    ctx.set_parameter_source("related_limit", click.core.ParameterSource.DEFAULT)

    assert query_verbs._explicit_read_view_options(ctx) == frozenset()


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


def test_delete_verb_updates_confirmation_and_dry_run_flags() -> None:
    _, child = _context_pair(query_terms=("alpha",))
    wrapped = getattr(query_verbs.delete_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    # Dry-run previews the SAME full pre-resolved id set the real delete acts on
    # via execute_delete_by_session_ids(dry_run=True). It must NOT route through
    # _execute_query_verb, which re-runs the query at the default limit of 20 and
    # would preview fewer sessions than --yes --all deletes (#1873). The preview
    # uses force=True internally so it never triggers an interactive prompt.
    with (
        patch(
            "polylogue.cli.verb_cardinality.probe_session_ids_for_verb",
            return_value=["alpha-id"],
        ) as probe_ids,
        patch(
            "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
            return_value=["alpha-id"],
        ) as resolve,
        patch("polylogue.cli.archive_query.execute_delete_by_session_ids") as execute,
        patch("polylogue.cli.query_verbs._execute_query_verb") as legacy,
    ):
        wrapped(child, True, False, False, "json")

    legacy.assert_not_called()
    probe_ids.assert_called_once()
    resolve.assert_called_once()
    args, kwargs = execute.call_args
    assert list(args[1]) == ["alpha-id"]
    assert kwargs.get("dry_run") is True
    assert kwargs.get("force") is True

    with (
        patch(
            "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
            return_value=["alpha-id"],
        ),
        patch("polylogue.cli.verb_cardinality.check_cardinality"),
        patch("polylogue.cli.archive_query.execute_delete_by_session_ids") as execute_confirmed,
    ):
        wrapped(child, False, True, False, None)

    _, confirmed_kwargs = execute_confirmed.call_args
    assert confirmed_kwargs.get("force") is True
    assert confirmed_kwargs.get("dry_run") is None


def test_terminal_facet_helpers_sort_and_bound_noisy_rows(capsys: pytest.CaptureFixture[str]) -> None:
    values = {f"value-{index:02d}": 20 - index for index in range(14)}

    query_verbs._emit_facet_bucket("Omitted/noisy facet counts (not canonical facets)", values, limit=12)

    output = capsys.readouterr().out
    assert "Omitted/noisy facet counts (not canonical facets):" in output
    assert "value-00: 20" in output
    assert "value-11: 9" in output
    assert "value-12: 8" not in output
    assert "… 2 more values omitted from terminal view; use --format json for full buckets." in output


def test_terminal_idf_helper_bounds_rows(capsys: pytest.CaptureFixture[str]) -> None:
    idf = {"tags": {f"tag-{index:02d}": float(30 - index) for index in range(13)}}

    query_verbs._emit_idf_buckets(idf, limit=12)

    output = capsys.readouterr().out
    assert "IDF (higher = rarer, partitions more strongly):" in output
    assert "tag-00: 30.000" in output
    assert "tag-11: 19.000" in output
    assert "tag-12: 18.000" not in output
    assert "… 1 more value omitted from terminal view; use --format json for full IDF." in output
