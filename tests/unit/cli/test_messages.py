from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import TracebackType
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli.messages import run_messages, run_raw
from polylogue.cli.read_views.messages import _write_messages_file
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.insights.topology import SessionTopology

SCHEMAS_DIR = Path("docs/schemas/cli-output")


def _load_schema(name: str) -> dict[str, object]:
    loaded = json.loads((SCHEMAS_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


class _FakeApi:
    def __init__(
        self,
        *,
        messages_result: tuple[list[dict[str, object]], int] | None = None,
        raw_result: tuple[list[dict[str, object]], int] = ([], 0),
        paginate_messages: bool = False,
        session_origin: str = "codex-session",
        topology: SessionTopology | None = None,
    ) -> None:
        self.messages_result = messages_result
        self.raw_result = raw_result
        self.paginate_messages = paginate_messages
        self.session_origin = session_origin
        self.topology = topology
        self.messages_kwargs: dict[str, object] = {}
        self.messages_calls: list[dict[str, object]] = []
        self.raw_kwargs: dict[str, object] = {}

    def _message_objects(self, msgs: list[dict[str, object]]) -> list[object]:
        defaults: dict[str, object] = {
            "blocks": [],
            "parent_id": None,
            "timestamp": None,
            "attachments": (),
            "branch_index": 0,
            "has_paste": False,
            "has_tool_use": False,
            "has_thinking": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "model_name": None,
        }
        return [
            type(
                "_FakeMsg",
                (),
                {
                    **defaults,
                    **m,
                    "message_type": type("_FakeMT", (), {"value": m.get("message_type", "")})(),
                },
            )()
            for m in msgs
        ]

    async def __aenter__(self) -> _FakeApi:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    async def get_messages_paginated(self, session_id: str, **kwargs: object) -> tuple[list[object], int] | None:
        self.messages_kwargs = {"session_id": session_id, **kwargs}
        self.messages_calls.append(self.messages_kwargs)
        if self.messages_result is None:
            from polylogue.api.archive import SessionNotFoundError

            raise SessionNotFoundError("missing")
        msgs, total = self.messages_result
        if self.paginate_messages:
            offset_value = kwargs.get("offset", 0)
            limit_value = kwargs.get("limit", len(msgs))
            assert isinstance(offset_value, int)
            assert isinstance(limit_value, int)
            offset = offset_value
            limit = limit_value
            msgs = msgs[offset : offset + limit]
        objs = self._message_objects(msgs) if msgs else []
        return objs, total

    async def iter_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        **_kwargs: object,
    ) -> AsyncIterator[object]:
        del session_id
        if self.messages_result is None:
            return
        msgs, _total = self.messages_result
        selected = msgs if limit is None else msgs[:limit]
        for obj in self._message_objects(selected):
            yield obj

    async def get_raw_artifacts_for_session(
        self, session_id: str, **kwargs: object
    ) -> tuple[list[dict[str, object]], int]:
        self.raw_kwargs = {"session_id": session_id, **kwargs}
        return self.raw_result

    async def get_session(self, session_id: str) -> object:
        del session_id
        return type("_FakeSession", (), {"origin": self.session_origin})()

    async def get_session_topology(self, session_id: str) -> SessionTopology | None:
        del session_id
        return self.topology


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
                db_path=tmp_path / "index.db",
            )
        }
    )


def test_run_messages_emits_json_and_passes_pagination(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
            session_id="conv-1",
            limit=5,
            offset=2,
            output_format="json",
        )

    payload = json.loads(capsys.readouterr().out)
    assert payload["messages"][0]["text"] == "hello"
    assert api.messages_kwargs["session_id"] == "conv-1"
    assert api.messages_kwargs["limit"] == 5
    assert api.messages_kwargs["offset"] == 2


def test_run_messages_full_rereads_with_total_limit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = _env()
    api = _FakeApi(
        messages_result=(
            [
                {"id": "msg-1", "role": "user", "message_type": "message", "text": "hello"},
                {"id": "msg-2", "role": "assistant", "message_type": "message", "text": "world"},
            ],
            2,
        ),
        paginate_messages=True,
    )

    with patch("polylogue.api.Polylogue.open", return_value=api):
        run_messages(
            env,
            _request(tmp_path),
            session_id="conv-1",
            limit=1,
            offset=0,
            full=True,
            output_format="json",
        )

    payload = json.loads(capsys.readouterr().out)
    assert len(payload["messages"]) == 2
    assert payload["limit"] == 2
    assert api.messages_calls == [
        {"session_id": "conv-1", "limit": 1, "offset": 0},
        {"session_id": "conv-1", "limit": 2, "offset": 0},
    ]


def test_run_messages_json_is_single_finite_document(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`read --view messages --format json` emits one finite JSON value (#1818)."""
    import jsonschema

    env = _env()
    api = _FakeApi(
        messages_result=(
            [
                # Rich-markup-like text must survive byte-for-byte: machine output
                # goes through raw click.echo, not the markup-interpreting console.
                {"id": "m1", "role": "user", "message_type": "message", "text": "[bold]first[/bold]"},
                {"id": "m2", "role": "assistant", "message_type": "message", "text": "second"},
            ],
            2,
        )
    )

    with patch("polylogue.api.Polylogue.open", return_value=api):
        run_messages(env, _request(tmp_path), session_id="conv-1", output_format="json")

    # Output is a single finite JSON value on stdout (one json.loads succeeds).
    payload = json.loads(capsys.readouterr().out)
    jsonschema.validate(instance=payload, schema=_load_schema("session-messages-response"))
    assert payload["session_id"] == "conv-1"
    assert [m["text"] for m in payload["messages"]] == ["[bold]first[/bold]", "second"]
    assert payload["total"] == 2


def test_write_messages_file_streams_json_payload(tmp_path: Path) -> None:
    env = _env()
    out = tmp_path / "messages.json"
    api = _FakeApi(
        messages_result=(
            [
                {"id": "m1", "role": "user", "message_type": "message", "text": "first"},
                {"id": "m2", "role": "assistant", "message_type": "message", "text": "second"},
            ],
            2,
        )
    )

    with patch("polylogue.api.Polylogue.open", return_value=api):
        _write_messages_file(
            env,
            _request(tmp_path),
            session_id="conv-1",
            limit=1,
            offset=1,
            full=False,
            output_format="json",
            out_path=out,
        )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["session_id"] == "conv-1"
    assert payload["total"] == 2
    assert payload["limit"] == 1
    assert payload["offset"] == 1
    assert [message["text"] for message in payload["messages"]] == ["second"]


def test_run_messages_ndjson_emits_one_json_document_per_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--format ndjson` streams one parseable JSON document per message (#1818)."""
    import jsonschema

    env = _env()
    api = _FakeApi(
        messages_result=(
            [
                {"id": "m1", "role": "user", "message_type": "message", "text": "[bold]first[/bold]"},
                {"id": "m2", "role": "assistant", "message_type": "message", "text": "second"},
            ],
            2,
        )
    )

    with patch("polylogue.api.Polylogue.open", return_value=api):
        run_messages(env, _request(tmp_path), session_id="conv-1", output_format="ndjson")

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2
    docs = [json.loads(line) for line in lines]  # each line parses independently
    schema = _load_schema("session-message-row")
    for doc in docs:
        jsonschema.validate(instance=doc, schema=schema)
    # Rich markup in text survives byte-for-byte (raw click.echo, no console markup).
    assert [d["text"] for d in docs] == ["[bold]first[/bold]", "second"]
    assert all(d["session_id"] == "conv-1" for d in docs)


def test_run_messages_markdown_and_not_found_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _env()
    api = _FakeApi(messages_result=([{"role": "assistant", "message_type": "message", "text": "x" * 501}], 1))

    with patch("polylogue.api.Polylogue.open", return_value=api):
        run_messages(env, _request(tmp_path), session_id="conv-1")

    rendered = capsys.readouterr().out
    assert "**assistant · message**" in rendered
    assert "x" * 501 in rendered
    _ui_print(env).assert_not_called()

    missing_env = _env()
    with patch("polylogue.api.Polylogue.open", return_value=_FakeApi(messages_result=None)):
        run_messages(missing_env, _request(tmp_path), session_id="missing")

    _ui_error(missing_env).assert_called_once_with("Session not found: missing")


def test_run_messages_text_alias_emits_human_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _env()
    api = _FakeApi(
        messages_result=(
            [{"id": "m1", "role": "user", "message_type": "message", "text": "text alias works"}],
            1,
        )
    )

    with patch("polylogue.api.Polylogue.open", return_value=api):
        run_messages(env, _request(tmp_path), session_id="conv-1", output_format="text")

    rendered = capsys.readouterr().out
    assert "**user · message**" in rendered
    assert "text alias works" in rendered
    _ui_print(env).assert_not_called()


def test_run_messages_markdown_uses_structural_shell_outcome(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _env()
    api = _FakeApi(
        messages_result=(
            [
                {
                    "id": "m-use",
                    "role": "assistant",
                    "message_type": "tool_use",
                    "text": None,
                    "blocks": [
                        {
                            "id": "b-use",
                            "type": "tool_use",
                            "tool_name": "exec_command",
                            "tool_id": "call-1",
                            "tool_input": {"command": "pytest -q"},
                            "semantic_type": "shell",
                        }
                    ],
                },
                {
                    "id": "m-result",
                    "role": "tool",
                    "message_type": "tool_result",
                    "text": None,
                    "blocks": [
                        {
                            "id": "b-result",
                            "type": "tool_result",
                            "tool_id": "call-1",
                            "text": "ERROR appears in output",
                            "tool_result_is_error": False,
                            "tool_result_exit_code": 0,
                        }
                    ],
                },
            ],
            2,
        )
    )

    with patch("polylogue.api.Polylogue.open", return_value=api):
        run_messages(env, _request(tmp_path), session_id="conv-1")

    rendered = capsys.readouterr().out
    assert "### Shell command · succeeded" in rendered
    assert "`is_error=false`" in rendered
    assert "`exit_code=0`" in rendered
    assert "ERROR appears in output" in rendered
    assert "FAILED" not in rendered


def test_run_raw_emits_json_yaml_and_empty_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    api = _FakeApi(
        raw_result=(
            [
                {
                    "raw_id": "raw-1",
                    "source_name": "codex",
                    "source_path": "/tmp/source.jsonl",
                    "blob_size": 42,
                }
            ],
            1,
        )
    )
    env = _env()

    with patch("polylogue.api.Polylogue.open", return_value=api):
        run_raw(env, _request(tmp_path), session_id="conv-raw", limit=3, offset=1)

    payload = json.loads(capsys.readouterr().out)
    assert payload["artifacts"][0]["raw_id"] == "raw-1"
    assert api.raw_kwargs == {"session_id": "conv-raw", "limit": 3, "offset": 1}
    _ui_print(env).assert_not_called()

    yaml_env = _env()
    with patch("polylogue.api.Polylogue.open", return_value=api):
        run_raw(yaml_env, _request(tmp_path), session_id="conv-raw", output_format="yaml")
    assert "raw-1" in capsys.readouterr().out
    _ui_print(yaml_env).assert_not_called()

    empty_env = _env()
    with patch("polylogue.api.Polylogue.open", return_value=_FakeApi(raw_result=([], 0))):
        run_raw(empty_env, _request(tmp_path), session_id="missing")
    _ui_error(empty_env).assert_called_once_with("No raw artifacts found for session: missing")
