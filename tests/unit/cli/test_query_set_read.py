"""Tests for the query-set read implementation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import click
from click.testing import CliRunner

from polylogue.archive.models import Session
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli import query_set_read, query_verbs
from polylogue.cli.click_app import cli
from polylogue.cli.read_view_handlers import ReadViewInvocation
from polylogue.cli.read_views.base import ReadViewMessageOptions
from polylogue.cli.read_views.query_set import run_query_set_read_view
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.surfaces.projection_spec import projection_from_views
from tests.infra.builders import make_conv, make_msg


def _stub_env(sessions: list[Session]) -> AppEnv:
    """Build an AppEnv-like stub whose list_sessions_for_spec returns the list."""

    async def _list_for_spec(spec, **_):  # type: ignore[no-untyped-def]
        return list(sessions)

    polylogue = SimpleNamespace(list_sessions_for_spec=_list_for_spec)
    return cast(AppEnv, SimpleNamespace(polylogue=polylogue))


def _capturing_env(sessions: list[Session], captured: dict[str, object]) -> AppEnv:
    async def _list_for_spec(spec, **_):  # type: ignore[no-untyped-def]
        captured["spec"] = spec
        return list(sessions)

    polylogue = SimpleNamespace(list_sessions_for_spec=_list_for_spec)
    return cast(AppEnv, SimpleNamespace(polylogue=polylogue))


def _request(**param_overrides: object) -> RootModeRequest:
    return RootModeRequest.from_params(param_overrides)


def _capture_run(env: AppEnv, request: RootModeRequest, output_format: str, fields: str | None) -> str:
    runner = CliRunner()
    with runner.isolation() as (out, _err, _term):
        query_set_read.run_query_set_read(env, request, output_format=output_format, fields=fields)
        return out.getvalue().decode("utf-8")


def test_run_query_set_read_emits_one_jsonl_line_per_session() -> None:
    convs = [
        make_conv(id="a", title="Alpha", messages=[make_msg(text="hello")]),
        make_conv(id="b", title="Beta", messages=[make_msg(text="world")]),
    ]
    output = _capture_run(_stub_env(convs), _request(), "jsonl", None)

    lines = output.strip().splitlines()
    assert len(lines) == 2
    titles = [json.loads(line).get("title") for line in lines]
    assert titles == ["Alpha", "Beta"]


def test_run_query_set_read_handles_zero_results() -> None:
    assert _capture_run(_stub_env([]), _request(), "jsonl", None) == ""


def test_run_query_set_read_separates_markdown_with_horizontal_rule() -> None:
    convs = [
        make_conv(id="a", title="Alpha", messages=[make_msg(text="hello")]),
        make_conv(id="b", title="Beta", messages=[make_msg(text="world")]),
    ]
    text = _capture_run(_stub_env(convs), _request(), "markdown", None)

    assert "Alpha" in text
    assert "Beta" in text
    # Separator appears between sessions, not before the first.
    assert text.count("\n---\n") == 1


def test_dialogue_query_set_selects_rows_then_uses_declared_view_operation() -> None:
    session = make_conv(id="session-1", messages=[make_msg(text="authored answer")])
    env = cast(AppEnv, SimpleNamespace(config=object()))
    with (
        patch(
            "polylogue.cli.session_rows.query_session_rows", return_value=[SimpleNamespace(session_id="session-1")]
        ) as select,
        patch("polylogue.cli.read_views.standard._read_dialogue_session", return_value=session) as read,
        patch("polylogue.cli.query_set_read.run_query_set_read", side_effect=AssertionError("local read")),
    ):
        runner = CliRunner()
        with runner.isolation() as (out, _err, _term):
            run_query_set_read_view(
                env,
                _request(limit=1),
                view="dialogue",
                output_format=None,
                fields=None,
                destination="terminal",
                out_path=None,
            )
            content = out.getvalue().decode("utf-8")
    assert "authored answer" in content
    select.assert_called_once()
    assert read.call_args.args[2] == "session-1"


def test_dialogue_query_set_applies_projection_spec_to_operation_result() -> None:
    session = make_conv(
        id="session-1",
        messages=[
            make_msg(id="user-1", role="user", text="one two", material_origin="human_authored"),
            make_msg(id="assistant-1", role="assistant", text="three four", material_origin="assistant_authored"),
        ],
    )

    env = cast(AppEnv, SimpleNamespace(config=object()))
    with (
        patch("polylogue.cli.session_rows.query_session_rows", return_value=[SimpleNamespace(session_id="session-1")]),
        patch("polylogue.cli.read_views.standard._read_dialogue_session", return_value=session) as read,
    ):
        runner = CliRunner()
        with runner.isolation() as (out, _err, _term):
            run_query_set_read_view(
                env,
                _request(limit=1),
                view="dialogue",
                output_format="json",
                fields=None,
                destination="terminal",
                out_path=None,
                projection_spec=projection_from_views(("dialogue",), max_tokens=3),
            )
            payload = json.loads(out.getvalue().decode("utf-8"))[0]
    assert read.call_args.args[3].max_tokens == 3
    assert payload["message_count"] == 2
    assert payload["rendered_message_count"] == 1
    assert payload["projection"]["max_tokens"] == 3
    assert [message["id"] for message in payload["messages"]] == ["user-1"]


def test_dialogue_query_set_without_limit_reads_every_selected_session() -> None:
    session_a = make_conv(id="session-a", messages=[make_msg(text="first")])
    session_b = make_conv(id="session-b", messages=[make_msg(text="second")])
    env = cast(AppEnv, SimpleNamespace(config=object()))
    with (
        patch(
            "polylogue.cli.session_rows.query_complete_session_ids",
            return_value=["session-a", "session-b"],
        ) as select,
        patch(
            "polylogue.cli.read_views.standard._read_dialogue_session",
            side_effect=[session_a, session_b],
        ) as read,
        patch("polylogue.cli.read_dispatch.daemon_route_disabled", return_value=False),
    ):
        runner = CliRunner()
        with runner.isolation() as (out, _err, _term):
            run_query_set_read_view(
                env,
                _request(),
                view="dialogue",
                output_format="json",
                fields=None,
                destination="terminal",
                out_path=None,
            )
            payload = json.loads(out.getvalue().decode("utf-8"))
    assert [item["id"] for item in payload] == ["session-a", "session-b"]
    select.assert_called_once()
    assert [call.args[2] for call in read.call_args_list] == ["session-a", "session-b"]


def test_read_all_registered_and_dispatches_via_root_cli() -> None:
    """Smoke: ``read --all`` routes to query-set read via the read verb."""
    convs = [make_conv(id="a", title="Smoke", messages=[make_msg(text="hi")])]

    captured: dict[str, object] = {}

    def _capture(
        env: object,
        request: RootModeRequest,
        *,
        output_format: str,
        fields: str | None = None,
        **_: object,
    ) -> None:
        captured["env"] = env
        captured["request"] = request
        captured["output_format"] = output_format
        captured["fields"] = fields

    # `--view messages` is a registered view, which the query-set path routes
    # through `_run_registered_view_query_set` rather than `run_query_set_read`.
    # Patching the latter intercepts nothing and the verb reaches a real
    # archive read.
    with patch("polylogue.cli.read_views.query_set._run_registered_view_query_set", side_effect=_capture):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--plain", "--origin", "claude-code-session", "read", "--all", "--view", "messages", "--format", "ndjson"],
            catch_exceptions=False,
        )

    # The CLI must accept the verb without error even though no archive exists in the test env.
    assert result.exit_code == 0, result.output
    # `messages` is a registered view, so the requested format reaches that
    # branch unchanged; the ndjson -> jsonl mapping belongs to the bulk
    # summary/transcript/dialogue path, and this route renders json instead.
    assert captured["output_format"] == "ndjson"
    assert isinstance(captured["request"], RootModeRequest)
    assert captured["request"].query_params()["origin"] == "claude-code-session"
    assert convs


def test_read_spec_emits_composed_projection_contract() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "--origin",
            "claude-code-session",
            "repo:polylogue",
            "read",
            "--view",
            "temporal,chronicle",
            "--format",
            "json",
            "--to",
            "stdout",
            "--max-tokens",
            "2000",
            "--limit",
            "8",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["selection"]["refs"] == []
    assert payload["selection"]["query"] == "repo:polylogue"
    assert payload["selection"]["origin"] == "claude-code-session"
    assert payload["selection"]["limit"] == 8
    assert payload["projection"]["families"] == ["temporal", "sessions", "chronicle", "messages"]
    assert payload["projection"]["body_policy"] == "authored-dialogue"
    assert payload["projection"]["max_tokens"] == 2000
    assert payload["render"] == {
        "format": "json",
        "destination": "stdout",
        "layout": "context-image",
        "timestamps": "include-available",
    }


def test_read_spec_accepts_explicit_render_layout() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "repo:polylogue",
            "read",
            "--view",
            "temporal,chronicle",
            "--render-layout",
            "standard",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["render"]["layout"] == "standard"
    assert payload["projection"]["families"] == ["temporal", "sessions", "chronicle", "messages"]


def test_read_spec_accepts_explicit_timestamp_policy() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "repo:polylogue",
            "read",
            "--view",
            "temporal",
            "--timestamps",
            "omit",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["render"]["timestamps"] == "omit"
    assert payload["projection"]["families"] == ["temporal", "sessions"]


def test_read_spec_accepts_render_expression() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "repo:polylogue",
            "read",
            "--view",
            "temporal,chronicle",
            "--render",
            "layout:context-image,timestamps:omit,format:json,destination:stdout",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["render"]["layout"] == "context-image"
    assert payload["render"]["timestamps"] == "omit"
    assert payload["render"]["format"] == "json"
    assert payload["render"]["destination"] == "stdout"
    assert payload["projection"]["families"] == ["temporal", "sessions", "chronicle", "messages"]


def test_read_spec_accepts_projection_expression() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "repo:polylogue",
            "read",
            "--view",
            "context-image",
            "--projection",
            "max-tokens:1234,redact-paths:false,include-assertions:true",
            "--spec",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["projection"]["max_tokens"] == 1234
    assert payload["projection"]["redact_paths"] is False
    assert payload["projection"]["include_assertions"] is True
    assert "assertions" in payload["projection"]["families"]


def test_read_spec_rejects_conflicting_projection_expression_alias() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "repo:polylogue",
            "read",
            "--view",
            "context-image",
            "--projection",
            "max-tokens:1234",
            "--max-tokens",
            "999",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    assert "Conflicting projection max_tokens" in result.output


def test_read_spec_rejects_conflicting_render_expression_alias() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "repo:polylogue",
            "read",
            "--view",
            "temporal",
            "--render",
            "timestamps:omit",
            "--timestamps",
            "include-available",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    assert "Conflicting render timestamps" in result.output


def test_read_spec_to_file_writes_composed_projection_contract(tmp_path: Path) -> None:
    out_path = tmp_path / "projection-spec.json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "--origin",
            "claude-code-session",
            "repo:polylogue",
            "read",
            "--view",
            "temporal,chronicle",
            "--format",
            "json",
            "--to",
            "file",
            "--out",
            str(out_path),
            "--limit",
            "8",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Wrote to" in result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["selection"]["query"] == "repo:polylogue"
    assert payload["render"]["destination"] == "file"
    assert payload["render"]["out"] == str(out_path)


def test_read_spec_moves_standalone_chronicle_limit_to_projection_policy() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "repo:polylogue",
            "read",
            "--view",
            "chronicle",
            "--format",
            "json",
            "--limit",
            "3",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["selection"]["query"] == "repo:polylogue"
    assert "limit" not in payload["selection"]
    assert payload["projection"]["edge_limit"] == 3
    assert payload["projection"]["body_policy"] == "authored-dialogue"


def test_read_spec_maps_dialogue_to_authored_dialogue_projection() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "--id",
            "codex-session:abc",
            "read",
            "--view",
            "dialogue",
            "--format",
            "markdown",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["projection"]["families"] == ["messages", "blocks"]
    assert payload["projection"]["body_policy"] == "authored-dialogue"
    assert payload["render"]["timestamps"] == "include-available"


def test_read_spec_moves_standalone_message_window_to_projection_policy() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "session:abc",
            "read",
            "--view",
            "messages",
            "--format",
            "json",
            "--limit",
            "7",
            "--offset",
            "2",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["selection"]["query"] == "session:abc"
    assert "limit" not in payload["selection"]
    assert payload["projection"]["body_limit"] == 7
    assert payload["projection"]["body_offset"] == 2
    assert payload["projection"]["families"] == ["messages", "blocks"]


def test_read_spec_moves_standalone_neighbor_options_to_projection_policy() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "session:abc",
            "read",
            "--view",
            "neighbors",
            "--format",
            "json",
            "--limit",
            "4",
            "--window-hours",
            "12",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["selection"]["query"] == "session:abc"
    assert "limit" not in payload["selection"]
    assert payload["projection"]["neighbor_limit"] == 4
    assert payload["projection"]["neighbor_window_hours"] == 12
    assert payload["projection"]["families"] == ["neighbors", "sessions"]


def test_read_spec_records_context_image_selector_fields() -> None:
    """polylogue-zok3: query predicates survive the public format/to route."""

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "find",
            "repo:github.com/Sinity/polylogue origin:claude-code-session since:2026-06-01 until:2026-06-30 cwd:/workspace/polylogue route contracts",
            "read",
            "--view",
            "context-image",
            "--format",
            "json",
            "--to",
            "stdout",
            "--max-sessions",
            "3",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["selection"] == {
        "refs": [],
        "query": "repo:github.com/Sinity/polylogue origin:claude-code-session since:2026-06-01 until:2026-06-30 cwd:/workspace/polylogue route contracts",
        "origin": "claude-code-session",
        "since": "2026-06-01",
        "until": "2026-06-30",
        "project_path": "/workspace/polylogue",
        "project_repo": "github.com/Sinity/polylogue",
        "limit": 3,
    }
    assert payload["projection"]["families"] == ["context", "messages"]
    assert payload["projection"]["body_policy"] == "authored-dialogue"
    assert payload["projection"]["redact_paths"] is True
    assert payload["projection"]["include_assertions"] is False
    assert {"tool_use", "tool_result", "function_call", "function_call_output"} <= set(
        payload["projection"]["exclude_block_kinds"]
    )
    assert payload["render"]["layout"] == "context-image"


def test_read_spec_records_context_image_redaction_policy() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "find",
            "route contracts",
            "read",
            "--view",
            "context-image",
            "--format",
            "json",
            "--no-redact",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["projection"]["redact_paths"] is False


def test_read_spec_records_context_image_assertion_policy() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "find",
            "route contracts",
            "read",
            "--view",
            "context-image",
            "--format",
            "json",
            "--include-assertions",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["projection"]["families"] == ["context", "messages", "assertions"]
    assert payload["projection"]["include_assertions"] is True


def test_read_handler_invocation_carries_projection_spec() -> None:
    captured: dict[str, object] = {}

    def _capture(env: object, request: RootModeRequest, invocation: object) -> None:
        captured["env"] = env
        captured["request"] = request
        captured["invocation"] = invocation

    with patch("polylogue.cli.query_verbs.run_read_view", side_effect=_capture):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--plain",
                "--origin",
                "claude-code-session",
                "repo:polylogue",
                "read",
                "--view",
                "temporal",
                "--format",
                "json",
                "--to",
                "stdout",
                "--limit",
                "8",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    invocation = cast(ReadViewInvocation, captured["invocation"])
    spec = invocation.projection_spec
    assert invocation.view == "temporal"
    assert invocation.output_format == "json"
    assert spec is not None
    assert spec.selection.query == "repo:polylogue"
    assert spec.selection.origin == "claude-code-session"
    assert spec.selection.limit == 8
    assert spec.projection.families == ("temporal", "sessions")
    assert spec.projection.body_policy.value == "full"
    assert spec.render.format == "json"
    assert spec.render.destination == "stdout"
    assert spec.render.layout == "standard"


def test_read_verb_in_verb_names() -> None:
    assert "read" in query_verbs.VERB_NAMES


def test_authored_content_query_set_read_passes_selection_filters_to_query_spec() -> None:
    captured: dict[str, object] = {}
    convs = [
        make_conv(
            id="authored",
            title="Authored workflow",
            provider="claude-code",
            messages=[
                make_msg(id="u1", role="user", text="typed authored content"),
                make_msg(id="a1", role="assistant", text="assistant content"),
            ],
        )
    ]
    request = _request(
        origin="claude-code-session",
        repo="__thoughtspace",
        filter_has_paste=True,
        typed_only=True,
    )

    output = _capture_run(_capturing_env(convs, captured), request, "jsonl", None)

    spec = cast(SessionQuerySpec, captured["spec"])
    assert spec.origins == ("claude-code-session",)
    assert spec.repo_names == ("__thoughtspace",)
    assert spec.filter_has_paste is True
    assert spec.typed_only is True
    exported = json.loads(output)
    assert exported["id"] == "authored"


def _two_sessions() -> list[Session]:
    return [
        make_conv(id="a", title="Alpha", messages=[make_msg(text="hello")]),
        make_conv(id="b", title="Beta", messages=[make_msg(text="world")]),
    ]


def test_query_set_native_view_dispatches_once_for_the_set() -> None:
    """``temporal``/``chronicle`` accept a query set, so they run once over it.

    Anti-vacuity: routing them back through ``_run_registered_view_query_set``
    calls the handler once per selected session with the request narrowed to a
    single ``conv_id``, which makes ``len(calls) == 2`` and gives each call a
    concrete ``session_id``.
    """
    calls: list[ReadViewInvocation] = []

    def _capture(env: object, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
        calls.append(invocation)

    for view in ("temporal", "chronicle"):
        calls.clear()
        with patch("polylogue.cli.read_view_handlers.run_read_view", side_effect=_capture):
            run_query_set_read_view(
                _stub_env(_two_sessions()),
                _request(query="repo:polylogue"),
                view=view,
                output_format="json",
                fields=None,
                destination="stdout",
                out_path=None,
            )
        assert len(calls) == 1, f"{view} fanned out into {len(calls)} per-session invocations"
        assert calls[0].view == view
        assert calls[0].session_id is None


def test_per_session_view_still_dispatches_once_per_session() -> None:
    """A view that does not accept a query set keeps its per-session fan-out."""
    calls: list[ReadViewInvocation] = []

    def _capture(env: object, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
        calls.append(invocation)

    with (
        patch("polylogue.cli.read_view_handlers.run_read_view", side_effect=_capture),
        patch("polylogue.cli.read_views.base.deliver_content"),
    ):
        run_query_set_read_view(
            _stub_env(_two_sessions()),
            _request(query="repo:polylogue"),
            view="messages",
            output_format="json",
            fields=None,
            destination="stdout",
            out_path=None,
        )

    assert [invocation.session_id for invocation in calls] == ["a", "b"]


def test_query_set_read_forwards_the_full_message_option() -> None:
    """``read --all --view messages --full`` must reach the handler as ``full``.

    Anti-vacuity: building options from the projection alone leaves
    ``ReadViewMessageOptions.full`` false, so ``run_read_messages`` applies its
    page fallback and truncates every selected session.
    """
    calls: list[ReadViewInvocation] = []

    def _capture(env: object, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
        calls.append(invocation)

    with (
        patch("polylogue.cli.read_view_handlers.run_read_view", side_effect=_capture),
        patch("polylogue.cli.read_views.base.deliver_content"),
    ):
        run_query_set_read_view(
            _stub_env(_two_sessions()),
            _request(query="repo:polylogue"),
            view="messages",
            output_format="json",
            fields=None,
            destination="stdout",
            out_path=None,
            option_values={"full": True, "limit": None, "offset": 0},
        )

    assert calls
    for invocation in calls:
        assert cast(ReadViewMessageOptions, invocation.options).full is True


def test_query_set_read_without_full_keeps_the_bounded_window() -> None:
    """The opposite direction: no ``--full`` must not become an unbounded read."""
    calls: list[ReadViewInvocation] = []

    def _capture(env: object, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
        calls.append(invocation)

    with (
        patch("polylogue.cli.read_view_handlers.run_read_view", side_effect=_capture),
        patch("polylogue.cli.read_views.base.deliver_content"),
    ):
        run_query_set_read_view(
            _stub_env(_two_sessions()),
            _request(query="repo:polylogue"),
            view="messages",
            output_format="json",
            fields=None,
            destination="stdout",
            out_path=None,
            option_values={"full": False, "limit": None, "offset": 0},
        )

    assert calls
    for invocation in calls:
        assert cast(ReadViewMessageOptions, invocation.options).full is False


def test_query_set_ndjson_read_keeps_per_line_framing() -> None:
    """``read --all --view messages --format ndjson`` stays line-framed.

    Anti-vacuity: collapsing ndjson into ``json`` makes the delivered body a
    pretty-printed JSON array, so it starts with ``[`` and ``json.loads`` of
    the first line fails.
    """
    delivered: dict[str, object] = {}

    def _deliver(env: object, content: str, **kwargs: object) -> None:
        delivered["content"] = content
        delivered["output_format"] = kwargs.get("output_format")

    def _emit(env: object, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
        assert invocation.output_format == "ndjson"
        click.echo(json.dumps({"session_id": invocation.session_id, "message_id": "m1"}))

    with (
        patch("polylogue.cli.read_view_handlers.run_read_view", side_effect=_emit),
        patch("polylogue.cli.read_views.base.deliver_content", side_effect=_deliver),
    ):
        run_query_set_read_view(
            _stub_env(_two_sessions()),
            _request(query="repo:polylogue"),
            view="messages",
            output_format="ndjson",
            fields=None,
            destination="stdout",
            out_path=None,
        )

    body = cast(str, delivered["content"])
    assert delivered["output_format"] == "ndjson"
    assert not body.lstrip().startswith("[")
    lines = [line for line in body.splitlines() if line]
    assert [json.loads(line)["session_id"] for line in lines] == ["a", "b"]


def test_query_set_json_read_still_emits_one_array() -> None:
    """The opposite direction: ``--format json`` keeps the aggregated array."""
    delivered: dict[str, object] = {}

    def _deliver(env: object, content: str, **kwargs: object) -> None:
        delivered["content"] = content

    def _emit(env: object, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
        click.echo(json.dumps({"session_id": invocation.session_id}))

    with (
        patch("polylogue.cli.read_view_handlers.run_read_view", side_effect=_emit),
        patch("polylogue.cli.read_views.base.deliver_content", side_effect=_deliver),
    ):
        run_query_set_read_view(
            _stub_env(_two_sessions()),
            _request(query="repo:polylogue"),
            view="messages",
            output_format="json",
            fields=None,
            destination="stdout",
            out_path=None,
        )

    payload = json.loads(cast(str, delivered["content"]))
    assert [item["session_id"] for item in payload] == ["a", "b"]
