"""Thread-continue deep-link generator contracts (#1203).

The reader's per-message "continue in another agent" affordance is
backed by :mod:`polylogue.daemon.thread_continue`. Tests cover:

- the built-in template registry (always present, stable agent IDs);
- environment override semantics (add agent, override label, disable a
  built-in);
- URL substitution and percent-encoding (the prompt round-trips through
  the encoding without leaking control characters);
- the structural smoke contract for the reader shell so the action
  button and JS hooks ship with the daemon HTML.
"""

from __future__ import annotations

import json
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from typing import cast
from unittest.mock import MagicMock

from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer
from polylogue.daemon.thread_continue import (
    TEMPLATE_ENV_VAR,
    AgentTemplate,
    build_templates_envelope,
    build_url,
    list_templates,
)
from polylogue.daemon.web_shell import WEB_SHELL_HTML


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"


class _MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler(path: str) -> DaemonAPIHandler:
    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast(DaemonAPIHTTPServer, _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = "GET"
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.headers = cast(Message, _MockHeaders({"Content-Length": "0"}))
    handler.rfile = BytesIO(b"")
    handler.wfile = BytesIO()
    return handler


def _capture_responses(handler: DaemonAPIHandler) -> tuple[MagicMock, MagicMock]:
    send_error = MagicMock()
    send_json = MagicMock()
    handler._send_error = send_error  # type: ignore[method-assign]
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_error, send_json


class TestBuiltinRegistry:
    """The built-in registry ships with stable agent IDs."""

    def test_includes_claude_code_and_codex(self) -> None:
        agent_ids = {t.agent_id for t in list_templates(env={})}
        assert "claude-code" in agent_ids
        assert "codex" in agent_ids

    def test_clipboard_fallback_is_present(self) -> None:
        agent_ids = {t.agent_id for t in list_templates(env={})}
        assert "copy-prompt" in agent_ids

    def test_envelope_lists_count_and_env_var(self) -> None:
        env = build_templates_envelope(env={})
        assert cast(int, env["count"]) >= 3
        assert env["env_var"] == TEMPLATE_ENV_VAR
        templates = cast(list[dict[str, str]], env["templates"])
        assert all({"agent_id", "label", "url_template"} <= entry.keys() for entry in templates)


class TestEnvOverride:
    """``POLYLOGUE_READER_AGENT_TEMPLATES`` adds/overrides/disables agents."""

    def test_invalid_json_falls_back_to_builtins(self) -> None:
        env = {TEMPLATE_ENV_VAR: "not-json{{{"}
        agent_ids = {t.agent_id for t in list_templates(env=env)}
        assert "claude-code" in agent_ids

    def test_override_label_for_existing_agent(self) -> None:
        payload = json.dumps(
            [{"agent_id": "claude-code", "label": "Open Claude (custom)", "url_template": "claude://x"}]
        )
        env = {TEMPLATE_ENV_VAR: payload}
        templates = list_templates(env=env)
        match = next(t for t in templates if t.agent_id == "claude-code")
        assert match.label == "Open Claude (custom)"
        assert match.url_template == "claude://x"

    def test_disable_builtin_drops_agent(self) -> None:
        payload = json.dumps([{"agent_id": "codex", "enabled": False}])
        env = {TEMPLATE_ENV_VAR: payload}
        agent_ids = {t.agent_id for t in list_templates(env=env)}
        assert "codex" not in agent_ids
        assert "claude-code" in agent_ids  # other built-ins survive

    def test_add_new_agent(self) -> None:
        payload = json.dumps(
            [
                {
                    "agent_id": "my-cli",
                    "label": "Send to my CLI",
                    "url_template": "myscheme://prompt?p={prompt}",
                }
            ]
        )
        env = {TEMPLATE_ENV_VAR: payload}
        agents = {t.agent_id: t for t in list_templates(env=env)}
        assert "my-cli" in agents
        assert agents["my-cli"].label == "Send to my CLI"


class TestUrlBuilder:
    """``build_url`` substitutes the placeholders safely."""

    def test_prompt_is_percent_encoded(self) -> None:
        template = AgentTemplate(
            agent_id="x",
            label="X",
            url_template="https://example.test/?p={prompt}",
        )
        url = build_url(template, prompt="hello world & friends", session_id="c", message_id="m")
        assert "hello%20world%20%26%20friends" in url

    def test_session_and_message_id_are_quoted(self) -> None:
        template = AgentTemplate(
            agent_id="x",
            label="X",
            url_template="https://example.test/{session_id}/{message_id}",
        )
        url = build_url(template, prompt="", session_id="conv/with slash", message_id="m id")
        assert "conv%2Fwith%20slash" in url
        assert "m%20id" in url

    def test_prompt_plain_passes_through_unencoded(self) -> None:
        template = AgentTemplate(
            agent_id="x",
            label="X",
            url_template="custom://send?text={prompt_plain}",
        )
        url = build_url(template, prompt="hi & bye", session_id="c", message_id=None)
        # The placeholder receives the raw text; the operator opted in
        # by configuring ``{prompt_plain}`` deliberately.
        assert "hi & bye" in url

    def test_missing_message_id_substitutes_empty(self) -> None:
        template = AgentTemplate(
            agent_id="x",
            label="X",
            url_template="proto://?msg={message_id}",
        )
        url = build_url(template, prompt="x", session_id="c", message_id=None)
        assert url == "proto://?msg="


class TestEndpointDispatch:
    """``GET /api/thread-continue-templates`` returns the registry envelope."""

    def test_endpoint_returns_envelope(self) -> None:
        handler = _make_handler("/api/thread-continue-templates")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["count"] >= 3
        assert payload["env_var"] == TEMPLATE_ENV_VAR


class TestReaderShellThreadContinueHooks:
    """The shipped HTML must carry the action button and JS hooks."""

    def test_continue_action_button_present(self) -> None:
        assert 'data-act="continue-thread"' in WEB_SHELL_HTML

    def test_js_helpers_present(self) -> None:
        for hook in (
            "openThreadContinueMenu",
            "ensureThreadContinueTemplates",
            "activateThreadContinue",
            "_threadContinueFillTemplate",
        ):
            assert hook in WEB_SHELL_HTML, f"missing JS hook: {hook}"

    def test_endpoint_url_is_referenced(self) -> None:
        assert "/api/thread-continue-templates" in WEB_SHELL_HTML
