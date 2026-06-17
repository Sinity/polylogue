from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import click
import pytest

from polylogue.archive.viewport import READ_VIEW_PROFILE_BY_ID, READ_VIEW_PROFILES, read_view_choices
from polylogue.cli import query_verbs
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


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
    """read --view summary (default) routes to _execute_query_verb."""
    _, child = _context_pair(params={"origin": "chatgpt-export"}, query_terms=("alpha",))
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        wrapped(child, **_read_verb_kwargs(view="summary"))

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
        wrapped(child, **_read_verb_kwargs(output_format="json", export_all=True))

    run_export.assert_called_once()
    assert run_export.call_args.kwargs["output_format"] == "json"


def test_read_verb_context_composes_preamble_not_passthrough() -> None:
    """read --view context routes to run_context_compose, NOT a markdown passthrough (#1842 bug)."""
    _, child = _context_pair(params={"conv_id": "claude-code:abc123"}, query_terms=())
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with (
        patch("polylogue.cli.commands.context.run_context_compose", return_value="{}") as compose,
        patch("polylogue.cli.query_verbs._execute_query_verb") as execute,
        patch("polylogue.cli.query_verbs._deliver_content") as deliver,
    ):
        wrapped(child, **_read_verb_kwargs(view="context", related_limit=3))

    execute.assert_not_called()
    compose.assert_called_once()
    assert compose.call_args.kwargs["session_id"] == "claude-code:abc123"
    assert compose.call_args.kwargs["related_limit"] == 3
    deliver.assert_called_once()


def test_read_verb_context_pack_invokes_pack_view() -> None:
    """read --view context-pack routes to run_context_pack_view with pack options."""
    _, child = _context_pair(query_terms=())
    wrapped = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.commands.context_pack.run_context_pack_view") as pack:
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
        patch("polylogue.cli.query_verbs._deliver_content") as deliver,
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
        patch("polylogue.cli.query_verbs._deliver_content") as deliver,
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
        patch("polylogue.cli.query_verbs._deliver_content") as deliver,
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
        query_verbs._run_read_recovery(
            cast(AppEnv, env),
            session_id="s1",
            output_format="json",
            report=None,
            destination="terminal",
            out_path=None,
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
        query_verbs._run_read_recovery(
            cast(AppEnv, env),
            session_id="s1",
            output_format="json",
            report="work-packet",
            destination="terminal",
            out_path=None,
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
        patch("polylogue.cli.query_verbs._deliver_content") as deliver,
    ):
        wrapped(child, **_read_verb_kwargs(view="recovery", output_format=None))

    content = deliver.call_args.args[1]
    assert '"status": "ok"' in content
    assert '"recovery"' in content


def test_read_view_recovery_requires_session_id() -> None:
    env = SimpleNamespace(polylogue=SimpleNamespace())

    with pytest.raises(SystemExit):
        query_verbs._run_read_recovery(
            cast(AppEnv, env),
            session_id=None,
            output_format=None,
            report=None,
            destination="terminal",
            out_path=None,
        )


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
