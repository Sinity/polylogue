from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import click
import pytest

from polylogue.archive.viewport import READ_VIEW_PROFILE_BY_ID, READ_VIEW_PROFILES, read_view_choices
from polylogue.cli import query_verbs, read_view_handlers
from polylogue.cli.read_view_handlers import ReadViewInvocation
from polylogue.cli.read_views.recovery import run_read_recovery
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.context.compiler import ContextImage, ContextSegment, ContextSpec
from polylogue.surfaces.payloads import PublicRefResolutionPayload


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
        with patch("polylogue.cli.query_verbs.run_bulk_export_view") as bulk_export:
            wrapped_read(
                child,
                "summary",
                "terminal",
                "json",
                None,
                True,
                (),
                None,
                7,
                0,
                24,
                None,
                2,
                0.3,
                True,
                False,
                5,
                None,
                None,
                None,
                None,
                None,
                None,
                5,
                20,
                False,
                False,
                False,
                False,
                False,
                False,
                "id,title",
                False,
            )

    bulk_export.assert_not_called()
    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_params()["origin"] == "chatgpt-export"
    assert request.query_params()["limit"] == 7
    assert request.query_params()["query"] == ("alpha",)
    assert request.query_params()["list_mode"] is True
    assert request.query_params()["output_format"] == "json"

    out_path = "/tmp/polylogue-sessions.json"
    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        with patch("polylogue.cli.query_verbs.run_bulk_export_view") as bulk_export:
            wrapped_read(
                child,
                "summary",
                "file",
                "json",
                out_path,
                True,
                (),
                None,
                None,
                0,
                24,
                None,
                2,
                0.3,
                True,
                False,
                5,
                None,
                None,
                None,
                None,
                None,
                None,
                5,
                20,
                False,
                False,
                False,
                False,
                False,
                False,
                None,
                False,
            )

    bulk_export.assert_not_called()
    file_request = execute.call_args.args[1]
    assert isinstance(file_request, RootModeRequest)
    assert file_request.query_params()["list_mode"] is True
    assert file_request.query_params()["output"] == out_path

    wrapped_analyze = getattr(query_verbs.analyze_verb.callback, "__wrapped__", None)
    assert callable(wrapped_analyze)
    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped_analyze(child, True, None, False, False, None, None)

    count_request = execute.call_args.args[1]
    assert isinstance(count_request, RootModeRequest)
    assert count_request.query_params()["count_only"] is True

    with pytest.raises(click.UsageError, match="does not support --limit"):
        wrapped_analyze(child, True, None, False, False, None, 5)


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
            export_all=False,
            message_role=(),
            message_type=None,
            limit=None,
            offset=0,
            window_hours=24,
            repo_path=None,
            since_hours=2,
            confidence_threshold=0.3,
            github_api=True,
            otlp=False,
            related_limit=5,
            recovery_report=None,
            project_path=None,
            project_repo=None,
            since=None,
            until=None,
            pack_origin=None,
            pack_query=None,
            max_sessions=5,
            max_messages=20,
            no_redact=False,
            no_code_blocks=False,
            no_tool_calls=False,
            no_tool_outputs=False,
            no_file_reads=False,
            prose_only=False,
            fields=None,
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
        wrapped(child, False, None, False, False, None, None)

    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_params()["stats_only"] is True
    assert request.query_params()["query"] == ("alpha",)

    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped(child, False, "origin", False, False, "markdown", 3)

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
    option = next(param for param in query_verbs.read_verb.params if param.name == "view")

    assert isinstance(option.type, click.Choice)
    assert tuple(option.type.choices) == read_view_choices()


def test_read_view_handlers_cover_view_profiles() -> None:
    assert read_view_handlers.read_view_handler_ids() == read_view_choices()
    read_view_handlers.validate_read_view_handler_registry()


def test_read_view_completion_comes_from_view_profiles() -> None:
    option = next(param for param in query_verbs.read_verb.params if param.name == "view")

    items = option.shell_complete(click.Context(query_verbs.read_verb), "rec")

    assert [item.value for item in items] == ["recovery"]
    assert items[0].help is not None
    assert "Recovery:" in items[0].help
    assert "successor-agent recovery transform" in items[0].help


def test_read_format_click_choices_come_from_view_profiles() -> None:
    option = next(param for param in query_verbs.read_verb.params if param.name == "output_format")
    expected = tuple(sorted({fmt for profile in READ_VIEW_PROFILES for fmt in profile.formats}))

    assert isinstance(option.type, click.Choice)
    assert tuple(option.type.choices) == expected


def test_read_format_completion_comes_from_selected_view_profile() -> None:
    option = next(param for param in query_verbs.read_verb.params if param.name == "output_format")
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
        "export_all": False,
        "message_role": (),
        "message_type": None,
        "limit": None,
        "offset": 0,
        "window_hours": 24,
        "repo_path": None,
        "since_hours": 2,
        "confidence_threshold": 0.3,
        "github_api": True,
        "otlp": False,
        "related_limit": 5,
        "recovery_report": None,
        "project_path": None,
        "project_repo": None,
        "since": None,
        "until": None,
        "pack_origin": None,
        "pack_query": None,
        "max_sessions": 5,
        "max_messages": 20,
        "no_redact": False,
        "no_code_blocks": False,
        "no_tool_calls": False,
        "no_tool_outputs": False,
        "no_file_reads": False,
        "prose_only": False,
        "fields": None,
    }
    defaults.update(overrides)
    return defaults


def test_read_verb_summary_dispatches_to_execute_query_verb() -> None:
    """read --view summary (default) routes through the handler registry."""
    _, child = _context_pair(params={"origin": "chatgpt-export"}, query_terms=("alpha",))
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.read_views.standard.execute_query_request") as execute:
        wrapped(child, **_read_verb_kwargs(view="summary"))

    execute.assert_called_once()
    request = execute.call_args.args[1]
    assert isinstance(request, RootModeRequest)
    assert request.query_params()["origin"] == "chatgpt-export"


def test_read_verb_all_non_summary_invokes_bulk_export_view() -> None:
    """read --all with a concrete non-summary view routes to bulk export."""
    _, child = _context_pair(params={"origin": "claude-code-session"}, query_terms=("alpha",))
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.query_verbs.run_bulk_export_view") as run_export:
        wrapped(child, **_read_verb_kwargs(view="transcript", output_format="json", export_all=True))

    run_export.assert_called_once()
    assert run_export.call_args.kwargs["output_format"] == "json"


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


def test_read_verb_context_pack_invokes_pack_view() -> None:
    """read --view context-pack routes to the context-pack view with pack options."""
    _, child = _context_pair(query_terms=())
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.context.pack.run_context_pack_view") as pack:
        wrapped(child, **_read_verb_kwargs(view="context-pack", pack_query="cost", max_sessions=3))

    pack.assert_called_once()
    assert pack.call_args.kwargs["query"] == "cost"
    assert pack.call_args.kwargs["max_sessions"] == 3


def test_read_verb_recovery_compiles_digest() -> None:
    """read --view recovery renders the facade-compiled recovery digest (#1880)."""
    _, child = _context_pair(params={"conv_id": "codex-session:abc123"}, query_terms=())
    child.obj.polylogue = SimpleNamespace()
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    async def recovery_digest(session_id: str) -> SimpleNamespace:
        assert session_id == "codex-session:abc123"
        return SimpleNamespace(resume_markdown="# Resume: demo\n")

    child.obj.polylogue.recovery_digest = recovery_digest

    with (
        patch("polylogue.cli.read_views.recovery.deliver_content") as deliver,
    ):
        wrapped(child, **_read_verb_kwargs(view="recovery"))

    deliver.assert_called_once_with(child.obj, "# Resume: demo\n", destination="terminal", out_path=None)


def test_read_verb_recovery_default_ignores_report_renderer() -> None:
    """Default recovery view stays the existing resume bundle, not a preset report."""
    _, child = _context_pair(params={"conv_id": "codex-session:abc123"}, query_terms=())
    child.obj.polylogue = SimpleNamespace()
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    digest = SimpleNamespace(
        resume_markdown="# Resume: demo\n",
        report_markdown=lambda preset: f"# {preset.title()}: demo [evidence: E1]\n",
    )

    async def recovery_digest(session_id: str) -> SimpleNamespace:
        assert session_id == "codex-session:abc123"
        return digest

    child.obj.polylogue.recovery_digest = recovery_digest

    with (
        patch("polylogue.cli.read_views.recovery.deliver_content") as deliver,
    ):
        wrapped(child, **_read_verb_kwargs(view="recovery"))

    deliver.assert_called_once_with(child.obj, "# Resume: demo\n", destination="terminal", out_path=None)


def test_read_verb_recovery_report_selector_renders_presets() -> None:
    """read --view recovery --report exposes distinct recovery reports."""
    _, child = _context_pair(params={"conv_id": "codex-session:abc123"}, query_terms=())
    child.obj.polylogue = SimpleNamespace()
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    reports = {
        "continue": "# Continue: demo [evidence: E1]\n",
        "blame": "# Blame: demo [evidence: E2]\n",
        "work-packet": "# Resume: demo\n\n## Evidence\n",
    }

    async def recovery_report(session_id: str, preset: str) -> str:
        assert session_id == "codex-session:abc123"
        return reports[preset]

    child.obj.polylogue.recovery_report = recovery_report

    with (
        patch("polylogue.cli.read_views.recovery.deliver_content") as deliver,
    ):
        wrapped(child, **_read_verb_kwargs(view="recovery", recovery_report="continue"))
        wrapped(child, **_read_verb_kwargs(view="recovery", recovery_report="blame"))
        wrapped(child, **_read_verb_kwargs(view="recovery", recovery_report="work-packet"))

    continue_report = deliver.call_args_list[0].args[1]
    blame_report = deliver.call_args_list[1].args[1]
    work_packet_report = deliver.call_args_list[2].args[1]
    assert continue_report.startswith("# Continue:")
    assert blame_report.startswith("# Blame:")
    assert work_packet_report.startswith("# Resume:")
    assert continue_report != blame_report
    assert work_packet_report not in {continue_report, blame_report}
    assert "[evidence: E1]" in continue_report
    assert "[evidence: E2]" in blame_report
    assert "## Evidence" in work_packet_report


def test_read_verb_recovery_report_selector_rejects_unknown() -> None:
    option = next(param for param in query_verbs.read_verb.params if param.name == "recovery_report")

    with pytest.raises(click.BadParameter, match="'incident' is not one of 'continue', 'blame', 'work-packet'"):
        option.type.convert("incident", option, click.Context(query_verbs.read_verb))


def test_continue_verb_renders_recovery_continue_report() -> None:
    """``find QUERY then continue`` reuses the recovery continue report surface."""
    _, child = _context_pair(query_terms=("id:codex-session:abc123",))
    wrapped = getattr(query_verbs.continue_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with (
        patch("polylogue.cli.query_verbs._resolve_target_session_id", return_value="codex-session:abc123"),
        patch("polylogue.cli.query_verbs.run_read_view") as run_read_view,
    ):
        wrapped(child, "clipboard", None, False, None, None, (), 10, None)

    invocation = run_read_view.call_args.args[2]
    assert invocation == ReadViewInvocation(
        view="recovery",
        session_id="codex-session:abc123",
        output_format=None,
        destination="clipboard",
        out_path=None,
        options=read_view_handlers.ReadViewRecoveryOptions(report="continue"),
    )


def test_continue_verb_json_emits_context_image(capsys: pytest.CaptureFixture[str]) -> None:
    """``continue --format json`` exposes the shared ContextImage payload."""
    _, child = _context_pair(query_terms=("id:codex-session:abc123",))
    spec = ContextSpec(
        purpose="continue",
        seed_refs=("session:codex-session:abc123",),
        read_views=("recovery",),
    )
    image = ContextImage(
        spec=spec,
        segments=(
            ContextSegment(
                segment_id="recovery:codex-session:abc123",
                kind="recovery",
                title="Recovery",
                markdown="Continue from here.",
                token_estimate=3,
            ),
        ),
        token_estimate=3,
    )
    seen: dict[str, ContextSpec] = {}

    async def compile_context(spec: ContextSpec) -> ContextImage:
        seen["spec"] = spec
        return image

    child.obj = SimpleNamespace(polylogue=SimpleNamespace(compile_context=compile_context))
    wrapped = getattr(query_verbs.continue_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.query_verbs._resolve_target_session_id", return_value="codex-session:abc123"):
        wrapped(child, "terminal", None, False, None, None, (), 10, "json")

    emitted = json.loads(capsys.readouterr().out)
    assert emitted["spec"]["purpose"] == "continue"
    assert emitted["spec"]["seed_refs"] == ["session:codex-session:abc123"]
    assert emitted["segments"][0]["markdown"] == "Continue from here."
    spec = seen["spec"]
    assert spec.purpose == "continue"
    assert spec.seed_refs == ("session:codex-session:abc123",)
    assert spec.read_views == ("recovery",)


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
            "terminal",
            None,
            True,
            "/workspace/polylogue",
            "/workspace/polylogue",
            ("polylogue/cli/query_verbs.py",),
            3,
            "json",
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
        wrapped(child, "terminal", None, False, "/workspace/polylogue", None, (), 10, None)


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


def test_read_view_recovery_json_uses_success_envelope(capsys: pytest.CaptureFixture[str]) -> None:
    """The recovery read view exposes the typed digest under the machine envelope."""

    class _API:
        async def recovery_digest(self, session_id: str) -> SimpleNamespace:
            assert session_id == "s1"
            return SimpleNamespace(resume_markdown="# Resume\n")

    env = SimpleNamespace(polylogue=_API())

    with (
        patch("polylogue.surfaces.payloads.model_json_document", return_value={"session_id": "s1"}),
    ):
        run_read_recovery(
            cast(AppEnv, env),
            RootModeRequest.from_params({}),
            ReadViewInvocation(
                view="recovery",
                session_id="s1",
                output_format="json",
                destination="terminal",
                out_path=None,
            ),
        )

    output = capsys.readouterr().out
    assert '"status": "ok"' in output
    assert '"recovery"' in output


def test_read_view_recovery_work_packet_json_uses_success_envelope(capsys: pytest.CaptureFixture[str]) -> None:
    """The recovery work-packet report exposes its DTO under machine output."""

    class _API:
        async def recovery_work_packet(self, session_id: str) -> SimpleNamespace:
            assert session_id == "s1"
            return SimpleNamespace(session_id=session_id, entries=())

    env = SimpleNamespace(polylogue=_API())

    with patch("polylogue.surfaces.payloads.model_json_document", return_value={"session_id": "s1"}):
        run_read_recovery(
            cast(AppEnv, env),
            RootModeRequest.from_params({}),
            ReadViewInvocation(
                view="recovery",
                session_id="s1",
                output_format="json",
                destination="terminal",
                out_path=None,
                options=read_view_handlers.ReadViewRecoveryOptions(report="work-packet"),
            ),
        )

    output = capsys.readouterr().out
    assert '"status": "ok"' in output
    assert '"recovery_work_packet"' in output


def test_read_view_recovery_honors_root_json_format() -> None:
    """Root --json applies to recovery output when the verb has no local format."""
    _, child = _context_pair(params={"conv_id": "codex-session:abc123", "output_format": "json"}, query_terms=())
    child.obj.polylogue = SimpleNamespace()
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    async def recovery_digest(session_id: str) -> SimpleNamespace:
        assert session_id == "codex-session:abc123"
        return SimpleNamespace(resume_markdown="# Resume\n")

    child.obj.polylogue.recovery_digest = recovery_digest

    with (
        patch("polylogue.surfaces.payloads.model_json_document", return_value={"session_id": "codex-session:abc123"}),
        patch("polylogue.cli.read_views.recovery.deliver_content") as deliver,
    ):
        wrapped(child, **_read_verb_kwargs(view="recovery", output_format=None))

    content = deliver.call_args.args[1]
    assert '"status": "ok"' in content
    assert '"recovery"' in content


def test_read_view_recovery_requires_session_id() -> None:
    env = SimpleNamespace(polylogue=SimpleNamespace())

    with pytest.raises(click.UsageError, match="requires a session ID"):
        read_view_handlers.run_read_view(
            cast(AppEnv, env),
            RootModeRequest.from_params({}),
            ReadViewInvocation(
                view="recovery",
                session_id=None,
                output_format=None,
                destination="terminal",
                out_path=None,
            ),
        )


def test_read_view_rejects_format_outside_selected_profile() -> None:
    env = SimpleNamespace(polylogue=SimpleNamespace())

    with pytest.raises(click.UsageError, match="read --view context-pack does not support --format json"):
        read_view_handlers.run_read_view(
            cast(AppEnv, env),
            RootModeRequest.from_params({}),
            ReadViewInvocation(
                view="context-pack",
                session_id=None,
                output_format="json",
                destination="terminal",
                out_path=None,
            ),
        )


def test_read_view_rejects_explicit_options_for_other_views() -> None:
    env = SimpleNamespace(polylogue=SimpleNamespace())

    with pytest.raises(click.UsageError, match="read --view recovery does not use --message-role"):
        read_view_handlers.run_read_view(
            cast(AppEnv, env),
            RootModeRequest.from_params({}),
            ReadViewInvocation(
                view="recovery",
                session_id="s1",
                output_format=None,
                destination="terminal",
                out_path=None,
                explicit_options=frozenset({"message_role"}),
            ),
        )


def test_read_view_accepts_explicit_options_owned_by_selected_view() -> None:
    read_view_handlers.READ_VIEW_HANDLERS["recovery"].validate(
        ReadViewInvocation(
            view="recovery",
            session_id="s1",
            output_format=None,
            destination="terminal",
            out_path=None,
            options=read_view_handlers.ReadViewRecoveryOptions(report="continue"),
            explicit_options=frozenset({"recovery_report"}),
        ),
        RootModeRequest.from_params({}),
    )


def test_read_view_registry_builds_typed_view_options() -> None:
    options = read_view_handlers.read_view_options_for_view(
        "context-pack",
        {
            "project_path": "/workspace/polylogue",
            "project_repo": "github.com/Sinity/polylogue",
            "pack_query": "route contracts",
            "max_sessions": 3,
            "max_messages": 8,
            "no_redact": True,
        },
    )

    assert isinstance(options, read_view_handlers.ReadViewContextPackOptions)
    assert options.project_path == "/workspace/polylogue"
    assert options.project_repo == "github.com/Sinity/polylogue"
    assert options.query == "route contracts"
    assert options.max_sessions == 3
    assert options.max_messages == 8
    assert options.no_redact is True


def test_explicit_read_view_options_reports_command_line_values_only() -> None:
    ctx = click.Context(query_verbs.read_verb)
    ctx.set_parameter_source("message_role", click.core.ParameterSource.COMMANDLINE)
    ctx.set_parameter_source("related_limit", click.core.ParameterSource.DEFAULT)

    assert query_verbs._explicit_read_view_options(ctx) == frozenset({"message_role"})


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
            "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
            return_value=["alpha-id"],
        ) as resolve,
        patch("polylogue.cli.archive_query.execute_delete_by_session_ids") as execute,
        patch("polylogue.cli.query_verbs._execute_query_verb") as legacy,
    ):
        wrapped(child, True, False, False)

    legacy.assert_not_called()
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
        wrapped(child, False, True, False)

    _, confirmed_kwargs = execute_confirmed.call_args
    assert confirmed_kwargs.get("force") is True
    assert confirmed_kwargs.get("dry_run") is None
