"""Tests for the bulk-export verb implementation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from click.testing import CliRunner

from polylogue.archive.models import Session
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli import bulk_export, query_verbs
from polylogue.cli.click_app import cli
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
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
        bulk_export.run_bulk_export(env, request, output_format=output_format, fields=fields)
        return out.getvalue().decode("utf-8")


def test_run_bulk_export_emits_one_jsonl_line_per_session() -> None:
    convs = [
        make_conv(id="a", title="Alpha", messages=[make_msg(text="hello")]),
        make_conv(id="b", title="Beta", messages=[make_msg(text="world")]),
    ]
    output = _capture_run(_stub_env(convs), _request(), "jsonl", None)

    lines = output.strip().splitlines()
    assert len(lines) == 2
    titles = [json.loads(line).get("title") for line in lines]
    assert titles == ["Alpha", "Beta"]


def test_run_bulk_export_handles_zero_results() -> None:
    assert _capture_run(_stub_env([]), _request(), "jsonl", None) == ""


def test_run_bulk_export_separates_markdown_with_horizontal_rule() -> None:
    convs = [
        make_conv(id="a", title="Alpha", messages=[make_msg(text="hello")]),
        make_conv(id="b", title="Beta", messages=[make_msg(text="world")]),
    ]
    text = _capture_run(_stub_env(convs), _request(), "markdown", None)

    assert "Alpha" in text
    assert "Beta" in text
    # Separator appears between sessions, not before the first.
    assert text.count("\n---\n") == 1


def test_read_all_registered_and_dispatches_via_root_cli() -> None:
    """Smoke: ``read --all`` routes to run_bulk_export via the read verb."""
    convs = [make_conv(id="a", title="Smoke", messages=[make_msg(text="hi")])]

    captured: dict[str, object] = {}

    def _capture(env: object, request: RootModeRequest, *, output_format: str, fields: str | None) -> None:
        captured["env"] = env
        captured["request"] = request
        captured["output_format"] = output_format
        captured["fields"] = fields

    with patch("polylogue.cli.bulk_export.run_bulk_export", side_effect=_capture):
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


def test_read_verb_in_verb_names() -> None:
    assert "read" in query_verbs.VERB_NAMES


def test_authored_content_bulk_export_workflow_filters_user_messages() -> None:
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
        message_role=("user",),
    )

    output = _capture_run(_capturing_env(convs, captured), request, "jsonl", None)

    spec = cast(SessionQuerySpec, captured["spec"])
    assert spec.origins == ("claude-code-session",)
    assert spec.repo_names == ("__thoughtspace",)
    assert spec.filter_has_paste is True
    assert spec.typed_only is True
    exported = json.loads(output)
    assert [message["role"] for message in exported["messages"]] == ["user"]
    assert exported["messages"][0]["text"] == "typed authored content"
