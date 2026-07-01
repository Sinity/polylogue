"""Tests for the query-set read implementation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from click.testing import CliRunner

from polylogue.archive.models import Session
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.cli import query_set_read, query_verbs
from polylogue.cli.click_app import cli
from polylogue.cli.read_view_handlers import ReadViewInvocation
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


def test_dialogue_query_set_view_uses_prose_projection() -> None:
    captured: dict[str, object] = {}

    def _capture(*args: object, **kwargs: object) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    with patch("polylogue.cli.query_set_read.run_query_set_read", side_effect=_capture):
        runner = CliRunner()
        with runner.isolation():
            run_query_set_read_view(
                _stub_env([]),
                _request(limit=1),
                view="dialogue",
                output_format=None,
                fields=None,
                destination="terminal",
                out_path=None,
            )

    kwargs = cast(dict[str, object], captured["kwargs"])
    projection = kwargs["content_projection"]
    assert kwargs["output_format"] == "markdown"
    assert isinstance(projection, ContentProjectionSpec)
    assert projection.filters_content()
    assert kwargs["renderer"] is not None


def test_dialogue_query_set_renderer_applies_projection_spec() -> None:
    captured: dict[str, object] = {}
    session = make_conv(
        id="session-1",
        messages=[
            make_msg(id="user-1", role="user", text="one two", material_origin="human_authored"),
            make_msg(id="assistant-1", role="assistant", text="three four", material_origin="assistant_authored"),
        ],
    )

    def _capture(*args: object, **kwargs: object) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    with patch("polylogue.cli.query_set_read.run_query_set_read", side_effect=_capture):
        runner = CliRunner()
        with runner.isolation():
            run_query_set_read_view(
                _stub_env([session]),
                _request(limit=1),
                view="dialogue",
                output_format="json",
                fields=None,
                destination="terminal",
                out_path=None,
                projection_spec=projection_from_views(("dialogue",), max_tokens=3),
            )

    renderer = cast(dict[str, object], captured["kwargs"])["renderer"]
    assert callable(renderer)
    payload = json.loads(renderer(session, "json", None))
    assert payload["message_count"] == 2
    assert payload["rendered_message_count"] == 1
    assert payload["projection"]["max_tokens"] == 3
    assert [message["id"] for message in payload["messages"]] == ["user-1"]


def test_read_all_registered_and_dispatches_via_root_cli() -> None:
    """Smoke: ``read --all`` routes to query-set read via the read verb."""
    convs = [make_conv(id="a", title="Smoke", messages=[make_msg(text="hi")])]

    captured: dict[str, object] = {}

    def _capture(
        env: object,
        request: RootModeRequest,
        *,
        output_format: str,
        fields: str | None,
        **_: object,
    ) -> None:
        captured["env"] = env
        captured["request"] = request
        captured["output_format"] = output_format
        captured["fields"] = fields

    with patch("polylogue.cli.query_set_read.run_query_set_read", side_effect=_capture):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--plain", "--origin", "claude-code-session", "read", "--all", "--view", "messages", "--format", "ndjson"],
            catch_exceptions=False,
        )

    # The CLI must accept the verb without error even though no archive exists in the test env.
    assert result.exit_code == 0, result.output
    # ndjson is mapped to jsonl internally
    assert captured["output_format"] == "jsonl"
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
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "read",
            "--view",
            "context-image",
            "--format",
            "json",
            "--project-path",
            "/workspace/polylogue",
            "--project-repo",
            "github.com/Sinity/polylogue",
            "--query",
            "route contracts",
            "--context-origin",
            "claude-code-session",
            "--since",
            "2026-06-01",
            "--until",
            "2026-06-30",
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
        "query": "route contracts",
        "origin": "claude-code-session",
        "since": "2026-06-01",
        "until": "2026-06-30",
        "project_path": "/workspace/polylogue",
        "project_repo": "github.com/Sinity/polylogue",
        "limit": 3,
    }
    assert payload["projection"]["families"] == ["context", "messages", "assertions"]
    assert payload["projection"]["body_policy"] == "authored-dialogue"
    assert payload["projection"]["redact_paths"] is True
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
            "read",
            "--view",
            "context-image",
            "--format",
            "json",
            "--query",
            "route contracts",
            "--no-redact",
            "--spec",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["projection"]["redact_paths"] is False


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
