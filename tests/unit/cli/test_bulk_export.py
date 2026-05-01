"""Tests for the bulk-export verb implementation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import click
from click.testing import CliRunner

from polylogue.cli import bulk_export, query_verbs
from polylogue.cli.click_app import cli
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.lib.models import Conversation
from polylogue.lib.query.spec import ConversationQuerySpec
from polylogue.types import Provider
from tests.infra.builders import make_conv, make_msg


def _stub_env(conversations: list[Conversation]) -> AppEnv:
    """Build an AppEnv-like stub whose query_conversations returns the given list."""

    async def _query_conversations(spec, **_):  # type: ignore[no-untyped-def]
        return list(conversations)

    operations = SimpleNamespace(query_conversations=_query_conversations)
    return cast(AppEnv, SimpleNamespace(operations=operations))


def _capturing_env(conversations: list[Conversation], captured: dict[str, object]) -> AppEnv:
    async def _query_conversations(spec, **_):  # type: ignore[no-untyped-def]
        captured["spec"] = spec
        return list(conversations)

    operations = SimpleNamespace(query_conversations=_query_conversations)
    return cast(AppEnv, SimpleNamespace(operations=operations))


def _request(**param_overrides: object) -> RootModeRequest:
    return RootModeRequest.from_params(param_overrides)


def _capture_run(env: AppEnv, request: RootModeRequest, output_format: str, fields: str | None) -> str:
    runner = CliRunner()
    with runner.isolation() as (out, _err, _term):
        bulk_export.run_bulk_export(env, request, output_format=output_format, fields=fields)
        return out.getvalue().decode("utf-8")


def test_run_bulk_export_emits_one_jsonl_line_per_conversation() -> None:
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
    # Separator appears between conversations, not before the first.
    assert text.count("\n---\n") == 1


def test_bulk_export_verb_registered_and_dispatches_via_root_cli() -> None:
    """Smoke test: the CLI parser routes ``bulk-export`` as a query verb."""
    convs = [make_conv(id="a", title="Smoke", messages=[make_msg(text="hi")])]

    captured: dict[str, object] = {}

    def _capture(env: object, request: RootModeRequest, *, output_format: str, fields: str | None) -> None:
        captured["env"] = env
        captured["request"] = request
        captured["output_format"] = output_format
        captured["fields"] = fields

    with (
        patch("polylogue.cli.bulk_export.run_bulk_export", side_effect=_capture),
        patch.object(query_verbs, "_parent_request", wraps=query_verbs._parent_request),
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--plain", "--provider", "claude-code", "bulk-export", "--format", "jsonl"],
            catch_exceptions=False,
        )

    # The CLI must accept the verb without error even though no archive exists in the test env.
    assert result.exit_code == 0, result.output
    assert captured["output_format"] == "jsonl"
    assert isinstance(captured["request"], RootModeRequest)
    assert captured["request"].query_params()["provider"] == "claude-code"
    # Mark conversations referenced (lints the import path).
    assert convs


def test_bulk_export_verb_in_verb_names() -> None:
    assert "bulk-export" in query_verbs.VERB_NAMES


def test_bulk_export_verb_callback_passes_fields_through() -> None:
    parent = click.Context(click.Command("query"))
    parent.params = {"query_term": (), "provider": "codex"}
    parent.meta["polylogue_query_terms"] = ()
    child = click.Context(click.Command("verb"), parent=parent)
    child.obj = SimpleNamespace()

    wrapped = getattr(query_verbs.bulk_export_verb.callback, "__wrapped__", None)
    assert callable(wrapped)

    with patch("polylogue.cli.bulk_export.run_bulk_export") as run_export:
        wrapped(child, "json", "id,title")

    args, kwargs = run_export.call_args
    assert isinstance(args[1], RootModeRequest)
    assert kwargs == {"output_format": "json", "fields": "id,title"}


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
        provider="claude-code",
        repo="__thoughtspace",
        filter_has_paste=True,
        typed_only=True,
        message_role=("user",),
    )

    output = _capture_run(_capturing_env(convs, captured), request, "jsonl", None)

    spec = cast(ConversationQuerySpec, captured["spec"])
    assert spec.providers == (Provider.CLAUDE_CODE,)
    assert spec.repo_names == ("__thoughtspace",)
    assert spec.filter_has_paste is True
    assert spec.typed_only is True
    exported = json.loads(output)
    assert [message["role"] for message in exported["messages"]] == ["user"]
    assert exported["messages"][0]["text"] == "typed authored content"
