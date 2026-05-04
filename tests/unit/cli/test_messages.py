from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import cast
from unittest.mock import MagicMock, patch

from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.cli.messages import run_messages, run_raw
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config


class _FakeApi:
    def __init__(
        self,
        *,
        messages_result: tuple[list[dict[str, object]], int] | None = None,
        raw_result: tuple[list[dict[str, object]], int] = ([], 0),
    ) -> None:
        self.messages_result = messages_result
        self.raw_result = raw_result
        self.messages_kwargs: dict[str, object] = {}
        self.raw_kwargs: dict[str, object] = {}

    async def __aenter__(self) -> _FakeApi:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    async def get_messages_paginated(
        self, conversation_id: str, **kwargs: object
    ) -> tuple[list[dict[str, object]], int] | None:
        self.messages_kwargs = {"conversation_id": conversation_id, **kwargs}
        if self.messages_result is None:
            from polylogue.api.archive import ConversationNotFoundError

            raise ConversationNotFoundError("missing")
        msgs, total = self.messages_result
        # Convert dicts to fake objects with attribute access for Message compat
        objs = (
            [
                type(
                    "_FakeMsg", (), {**m, "message_type": type("_FakeMT", (), {"value": m.get("message_type", "")})()}
                )()
                for m in msgs
            ]
            if msgs
            else []
        )
        return objs, total

    async def get_raw_artifacts_for_conversation(
        self, conversation_id: str, **kwargs: object
    ) -> tuple[list[dict[str, object]], int]:
        self.raw_kwargs = {"conversation_id": conversation_id, **kwargs}
        return self.raw_result


def _env() -> AppEnv:
    ui = MagicMock()
    ui.print = MagicMock()
    ui.error = MagicMock()
    return AppEnv(ui=ui, services=MagicMock())


def _ui_print(env: AppEnv) -> MagicMock:
    return cast(MagicMock, env.ui.print)


def _ui_error(env: AppEnv) -> MagicMock:
    return cast(MagicMock, env.ui.error)


def _request(tmp_path: Path) -> RootModeRequest:
    return RootModeRequest.from_params(
        {
            "_config": Config(
                archive_root=tmp_path,
                render_root=tmp_path / "render",
                sources=[],
                db_path=tmp_path / "polylogue.db",
            )
        }
    )


def test_run_messages_emits_json_and_passes_projection(tmp_path: Path) -> None:
    env = _env()
    api = _FakeApi(
        messages_result=(
            [
                {
                    "id": "msg-1",
                    "role": "user",
                    "message_type": "message",
                    "text": "hello",
                }
            ],
            1,
        )
    )

    with patch("polylogue.api.Polylogue.open", return_value=api):
        run_messages(
            env,
            _request(tmp_path),
            conversation_id="conv-1",
            message_role=("user",),
            message_type="summary",
            limit=5,
            offset=2,
            no_code_blocks=True,
            prose_only=True,
            output_format="json",
        )

    payload = json.loads(_ui_print(env).call_args.args[0])
    assert payload["messages"][0]["text"] == "hello"
    assert api.messages_kwargs["conversation_id"] == "conv-1"
    assert api.messages_kwargs["message_role"] == ("user",)
    assert api.messages_kwargs["message_type"] == "summary"
    assert api.messages_kwargs["limit"] == 5
    assert api.messages_kwargs["offset"] == 2
    projection = cast(ContentProjectionSpec, api.messages_kwargs["content_projection"])
    assert projection.include_code is False
    assert projection.include_tool_calls is False


def test_run_messages_markdown_and_not_found_paths(tmp_path: Path) -> None:
    env = _env()
    api = _FakeApi(messages_result=([{"role": "assistant", "message_type": "message", "text": "x" * 501}], 1))

    with patch("polylogue.api.Polylogue.open", return_value=api):
        run_messages(env, _request(tmp_path), conversation_id="conv-1")

    assert _ui_print(env).call_args_list[0].args[0].endswith("...")
    assert _ui_print(env).call_args_list[1].args[0] == "---"

    missing_env = _env()
    with patch("polylogue.api.Polylogue.open", return_value=_FakeApi(messages_result=None)):
        run_messages(missing_env, _request(tmp_path), conversation_id="missing")

    _ui_error(missing_env).assert_called_once_with("Conversation not found: missing")


def test_run_raw_emits_json_yaml_and_empty_error(tmp_path: Path) -> None:
    api = _FakeApi(
        raw_result=(
            [
                {
                    "raw_id": "raw-1",
                    "provider_name": "codex",
                    "source_path": "/tmp/source.jsonl",
                    "blob_size": 42,
                }
            ],
            1,
        )
    )
    env = _env()

    with patch("polylogue.api.Polylogue.open", return_value=api):
        run_raw(env, _request(tmp_path), conversation_id="conv-raw", limit=3, offset=1)

    payload = json.loads(_ui_print(env).call_args.args[0])
    assert payload["artifacts"][0]["raw_id"] == "raw-1"
    assert api.raw_kwargs == {"conversation_id": "conv-raw", "limit": 3, "offset": 1}

    yaml_env = _env()
    with patch("polylogue.api.Polylogue.open", return_value=api):
        run_raw(yaml_env, _request(tmp_path), conversation_id="conv-raw", output_format="yaml")
    assert "raw-1" in _ui_print(yaml_env).call_args.args[0]

    empty_env = _env()
    with patch("polylogue.api.Polylogue.open", return_value=_FakeApi(raw_result=([], 0))):
        run_raw(empty_env, _request(tmp_path), conversation_id="missing")
    _ui_error(empty_env).assert_called_once_with("No raw artifacts found for conversation: missing")
