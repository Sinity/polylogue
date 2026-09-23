from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli.messages import run_messages
from polylogue.cli.read_views.base import ReadViewInvocation
from polylogue.cli.read_views.messages import _write_messages_file
from polylogue.cli.read_views.session_evidence import run_read_hooks
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.storage.runtime import LineageCompleteness
from tests.infra.daemon_operations import cli_daemon_archive

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
        hook_summary_result: dict[str, object] | None = None,
        session_events_result: list[dict[str, object]] | None = None,
        paginate_messages: bool = False,
        session_origin: str = "codex-session",
        topology: object | None = None,
        lineage_completeness: LineageCompleteness | None = None,
        config: Config | None = None,
    ) -> None:
        # The read path builds its authority envelope from `api.config`; the
        # double carries one so it exercises the same route.
        self.config = (
            config
            if config is not None
            else Config(
                archive_root=Path("/nonexistent/fake-archive"),
                render_root=Path("/nonexistent/fake-archive/render"),
                sources=[],
            )
        )
        self.messages_result = messages_result
        self.raw_result = raw_result
        self.hook_summary_result = hook_summary_result
        self.session_events_result = session_events_result
        self.paginate_messages = paginate_messages
        self.session_origin = session_origin
        self.topology = topology
        self.lineage_completeness = lineage_completeness or LineageCompleteness()
        self.messages_kwargs: dict[str, object] = {}
        self.messages_calls: list[dict[str, object]] = []
        self.raw_kwargs: dict[str, object] = {}
        self.hook_summary_kwargs: dict[str, object] = {}
        self.session_events_kwargs: dict[str, object] = {}

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

    async def get_messages_paginated(
        self, session_id: str, **kwargs: object
    ) -> tuple[list[object], int, LineageCompleteness]:
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
        return objs, total, self.lineage_completeness

    async def read_transcript_window(self, session_id: str, **kwargs: object) -> object:
        """Mirror the production route over this fake's storage read.

        ``run_messages`` reaches the bound transcript-window route
        (polylogue-ijbwq), so the double must answer that shape; the window
        coordinates it reports are still computed from this fake's own page.
        """

        from polylogue.archive.query.transaction import QueryTransactionRequest
        from polylogue.operations.transcript_window import (
            TRANSCRIPT_WINDOW_ORDER,
            TRANSCRIPT_WINDOW_PROJECTION,
            TranscriptWindow,
        )

        kwargs.pop("continuation", None)
        rows, total, completeness = await self.get_messages_paginated(session_id, **kwargs)
        limit_value = kwargs.get("limit", 50)
        offset_value = kwargs.get("offset", 0)
        assert isinstance(limit_value, int)
        assert isinstance(offset_value, int)
        next_offset = offset_value + len(rows) if offset_value + len(rows) < total else None
        return TranscriptWindow(
            rows=list(rows),
            total=total,
            limit=limit_value,
            offset=offset_value,
            next_offset=next_offset,
            continuation="continuation-token" if next_offset is not None else None,
            lineage_complete=completeness.complete,
            lineage_truncation_reason=(
                str(completeness.truncation_reason) if completeness.truncation_reason is not None else None
            ),
            transaction=QueryTransactionRequest(
                operation="sessions.read",
                arguments={"ref": f"session:{session_id}"},
                page_size=limit_value,
                offset=offset_value,
                projection=TRANSCRIPT_WINDOW_PROJECTION,
                stable_order=TRANSCRIPT_WINDOW_ORDER,
            ),
        )

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

    async def get_hook_event_summary_for_session(self, session_id: str) -> dict[str, object] | None:
        self.hook_summary_kwargs = {"session_id": session_id}
        return self.hook_summary_result

    async def get_session_events(self, session_id: str, **kwargs: object) -> list[dict[str, object]] | None:
        self.session_events_kwargs = {"session_id": session_id, **kwargs}
        return self.session_events_result

    async def get_session(self, session_id: str) -> object:
        return type(
            "_FakeSession",
            (),
            {
                "id": session_id,
                "origin": self.session_origin,
                "parent_id": None,
                "branch_type": None,
            },
        )()

    async def get_session_topology(self, session_id: str) -> object | None:
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


def _seed_dangling_fork(tmp_path: Path) -> str:
    """Seed a prefix-sharing fork whose parent branch point was hard-deleted.

    ``session_links.branch_point_message_id`` is deliberately not a foreign
    key, so deleting the parent's messages leaves the edge in place and the
    composed read falls back to the child's own divergent tail -- the exact
    shape polylogue-ppkj reports as ``dangling_branch_point``.
    """

    import sqlite3

    from tests.infra.storage_records import SessionBuilder

    db_path = tmp_path / "index.db"
    SessionBuilder(db_path, "parent").provider("codex").title("parent").add_message(
        "p0", role="user", text="hello"
    ).add_message("p1", role="assistant", text="hi there").save()
    child = SessionBuilder(db_path, "child")
    child.provider("codex").title("child").parent_session("ext-parent").branch_type("fork")
    child.add_message("c0", role="user", text="hello")
    child.add_message("c1", role="assistant", text="hi there")
    child.add_message("cx", role="user", text="child diverges")
    child.save()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "DELETE FROM messages WHERE session_id = (SELECT session_id FROM sessions WHERE native_id = 'ext-parent')"
        )
        conn.commit()
    return child.native_session_id()


def _seeded_request(tmp_path: Path) -> RootModeRequest:
    """A root request pinned at ``tmp_path`` for the test daemon."""

    return RootModeRequest.from_params(
        {
            "_config": Config(
                archive_root=tmp_path,
                render_root=tmp_path / "render",
                sources=[],
                db_path=tmp_path / "index.db",
            ),
        }
    )


@pytest.fixture
def daemon_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Serve message reads through the production daemon operation route."""

    with cli_daemon_archive(tmp_path, monkeypatch):
        yield tmp_path


def _seed_messages(tmp_path: Path, *messages: dict[str, object], title: str = "Seeded") -> str:
    """Seed one real archive session and return its archive session id."""

    from tests.infra.storage_records import SessionBuilder

    builder = SessionBuilder(tmp_path / "index.db", "seeded")
    builder.provider("codex").title(title)
    for index, message in enumerate(messages):
        blocks = message.get("blocks")
        builder.add_message(
            cast(str, message.get("id") or f"m{index + 1}"),
            role=cast(str, message.get("role", "user")),
            text=cast(str, message.get("text", "")),
            blocks=blocks if blocks is not None else [],
        )
    builder.save()
    return builder.native_session_id()


def test_run_messages_emits_json_and_passes_pagination(
    tmp_path: Path, daemon_archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The verb's window coordinates reach the declared read, not a local page.

    Anti-vacuity: serve the window from offset 0 and the asserted row text
    changes, because the seeded rows are distinguishable by position.
    """

    session_id = _seed_messages(
        tmp_path,
        {"id": "m1", "role": "user", "text": "first"},
        {"id": "m2", "role": "assistant", "text": "second"},
        {"id": "m3", "role": "user", "text": "third"},
        {"id": "m4", "role": "assistant", "text": "fourth"},
    )

    run_messages(_env(), _seeded_request(tmp_path), session_id=session_id, limit=1, offset=2, output_format="json")

    payload = json.loads(capsys.readouterr().out)
    assert [message["text"] for message in payload["messages"]] == ["third"]
    assert payload["offset"] == 2
    assert payload["limit"] == 1
    assert payload["total"] == 4


def test_run_messages_json_names_the_executor_that_answered(
    tmp_path: Path, daemon_archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rendered authority names the executor the result reported.

    Anti-vacuity: hard-code ``server_identity="daemon"`` in ``run_messages``
    and this goes red, because the in-process executor answered this read.
    """

    session_id = _seed_messages(tmp_path, {"id": "m1", "role": "user", "text": "hi"})

    run_messages(_env(), _seeded_request(tmp_path), session_id=session_id, output_format="json")

    payload = json.loads(capsys.readouterr().out)
    assert payload["authority"]["server_identity"] == "daemon"


def test_run_messages_verbose_prints_the_serving_executor(
    tmp_path: Path, daemon_archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--verbose`` names which executor served the page, on stderr.

    Anti-vacuity: drop the ``verbose`` branch and no ``served-by:`` line is
    emitted at all.
    """

    session_id = _seed_messages(tmp_path, {"id": "m1", "role": "user", "text": "hi"})
    request = _seeded_request(tmp_path).with_param_updates(verbose=True)

    run_messages(_env(), request, session_id=session_id, output_format="json")

    captured = capsys.readouterr()
    assert captured.err.strip().startswith("served-by: daemon")
    assert "served-by:" not in captured.out


def test_run_messages_json_surfaces_truncated_lineage(
    tmp_path: Path, daemon_archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """polylogue-ppkj: a dangling branch point must not silently render a
    partial transcript as if it were the whole conversation.

    Anti-vacuity: leave the parent's messages in place and the composed read
    reports ``lineage_complete: true``, so the assertions below go red.
    """

    child_id = _seed_dangling_fork(tmp_path)

    run_messages(_env(), _seeded_request(tmp_path), session_id=child_id, output_format="json")

    payload = json.loads(capsys.readouterr().out)
    assert payload["lineage_complete"] is False
    assert payload["lineage_truncation_reason"] == "dangling_branch_point"
    assert payload["outcome"]["state"] == "degraded"


def test_run_messages_json_lineage_complete_by_default(
    tmp_path: Path, daemon_archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session_id = _seed_messages(tmp_path, {"id": "m1", "role": "user", "text": "hi"})

    run_messages(_env(), _seeded_request(tmp_path), session_id=session_id, output_format="json")

    payload = json.loads(capsys.readouterr().out)
    assert payload["lineage_complete"] is True
    assert "lineage_truncation_reason" not in payload


def test_run_messages_full_composes_every_remaining_window(
    tmp_path: Path, daemon_archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--full`` composes the rest of the transcript out of bounded windows.

    Anti-vacuity: stop the window loop after the first window and only one of
    the two seeded messages is rendered.
    """

    session_id = _seed_messages(
        tmp_path,
        {"id": "msg-1", "role": "user", "text": "hello"},
        {"id": "msg-2", "role": "assistant", "text": "world"},
    )

    run_messages(
        _env(),
        _seeded_request(tmp_path),
        session_id=session_id,
        limit=1,
        offset=0,
        full=True,
        output_format="json",
    )

    payload = json.loads(capsys.readouterr().out)
    assert [message["text"] for message in payload["messages"]] == ["hello", "world"]
    assert payload["limit"] == 2
    assert payload["total"] == 2
    assert "continuation" not in payload


def test_run_messages_json_is_single_finite_document(
    tmp_path: Path, daemon_archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`read --view messages --format json` emits one finite JSON value (#1818)."""
    import jsonschema

    session_id = _seed_messages(
        tmp_path,
        # Rich-markup-like text must survive byte-for-byte: machine output
        # goes through raw click.echo, not the markup-interpreting console.
        {"id": "m1", "role": "user", "text": "[bold]first[/bold]"},
        {"id": "m2", "role": "assistant", "text": "second"},
    )

    run_messages(_env(), _seeded_request(tmp_path), session_id=session_id, output_format="json")

    # Output is a single finite JSON value on stdout (one json.loads succeeds).
    payload = json.loads(capsys.readouterr().out)
    jsonschema.validate(instance=payload, schema=_load_schema("session-messages-response"))
    assert payload["session_id"] == session_id
    assert [m["text"] for m in payload["messages"]] == ["[bold]first[/bold]", "second"]
    assert payload["total"] == 2


def test_write_messages_file_streams_json_payload(tmp_path: Path, daemon_archive: Path) -> None:
    env = _env()
    out = tmp_path / "messages.json"
    session_id = _seed_messages(
        tmp_path,
        {"id": "m1", "role": "user", "text": "first"},
        {"id": "m2", "role": "assistant", "text": "second"},
    )

    _write_messages_file(
        env,
        _seeded_request(tmp_path),
        session_id=session_id,
        limit=1,
        offset=1,
        full=False,
        output_format="json",
        out_path=out,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["session_id"] == session_id
    assert payload["total"] == 2
    assert payload["limit"] == 1
    assert payload["offset"] == 1
    assert [message["text"] for message in payload["messages"]] == ["second"]


def test_write_messages_file_preserves_the_destination_when_the_read_fails(
    tmp_path: Path, daemon_archive: Path
) -> None:
    """A failed first window must not destroy the file it was going to replace.

    ``read_message_windows`` is a generator: nothing is read until the write
    loop pulls from it. Opening the destination with ``"w"`` therefore
    truncated the operator's previous export and wrote a partial JSON header
    *before* the read could fail, so a mistyped session reference reported a
    read failure and silently destroyed a good file, leaving malformed JSON.

    Anti-vacuity: restore ``with out_path.open("w", encoding="utf-8") as fh:``
    in ``_write_messages_file`` and this goes red -- the destination is then
    the truncated header ``{\n  "session_id": ...`` instead of the previous
    export.
    """
    env = _env()
    out = tmp_path / "messages.json"
    previous = json.dumps({"session_id": "kept", "messages": [{"text": "previous export"}]})
    out.write_text(previous, encoding="utf-8")
    _seed_messages(tmp_path, {"id": "m1", "role": "user", "text": "first"})

    _write_messages_file(
        env,
        _seeded_request(tmp_path),
        session_id="codex-session:does-not-exist",
        limit=1,
        offset=0,
        full=False,
        output_format="json",
        out_path=out,
    )

    assert out.read_text(encoding="utf-8") == previous
    assert _ui_error(env).called
    assert sorted(child.name for child in tmp_path.iterdir() if child.name.startswith(".messages.json")) == []


def test_write_messages_file_replaces_the_destination_on_success(tmp_path: Path, daemon_archive: Path) -> None:
    """The opposite direction: a successful read still rewrites the file.

    Anti-vacuity: leave the staged file in place and never ``os.replace`` it
    and this goes red with the previous export still on disk -- a fix that
    only ever preserved the old file would be no fix at all.
    """
    out = tmp_path / "messages.json"
    out.write_text(json.dumps({"session_id": "stale", "messages": []}), encoding="utf-8")
    session_id = _seed_messages(tmp_path, {"id": "m1", "role": "user", "text": "first"})

    _write_messages_file(
        _env(),
        _seeded_request(tmp_path),
        session_id=session_id,
        limit=10,
        offset=0,
        full=False,
        output_format="json",
        out_path=out,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["session_id"] == session_id
    assert [message["text"] for message in payload["messages"]] == ["first"]
    assert sorted(child.name for child in tmp_path.iterdir() if child.name.startswith(".messages.json")) == []


def test_run_messages_ndjson_emits_one_json_document_per_line(
    tmp_path: Path, daemon_archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--format ndjson` streams one parseable JSON document per message (#1818)."""
    import jsonschema

    session_id = _seed_messages(
        tmp_path,
        {"id": "m1", "role": "user", "text": "[bold]first[/bold]"},
        {"id": "m2", "role": "assistant", "text": "second"},
    )

    run_messages(_env(), _seeded_request(tmp_path), session_id=session_id, output_format="ndjson")

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2
    docs = [json.loads(line) for line in lines]  # each line parses independently
    schema = _load_schema("session-message-row")
    for doc in docs:
        jsonschema.validate(instance=doc, schema=schema)
    # Rich markup in text survives byte-for-byte (raw click.echo, no console markup).
    assert [d["text"] for d in docs] == ["[bold]first[/bold]", "second"]
    assert all(d["session_id"] == session_id for d in docs)


def test_run_messages_markdown_and_not_found_paths(
    tmp_path: Path,
    daemon_archive: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _env()
    session_id = _seed_messages(tmp_path, {"id": "m1", "role": "assistant", "text": "x" * 501})

    run_messages(env, _seeded_request(tmp_path), session_id=session_id)

    rendered = capsys.readouterr().out
    assert "**assistant · message**" in rendered
    assert "x" * 501 in rendered
    _ui_print(env).assert_not_called()

    missing_env = _env()
    run_messages(missing_env, _seeded_request(tmp_path), session_id="codex-session:missing")

    _ui_error(missing_env).assert_called_once_with("Session not found: codex-session:missing")


def test_run_messages_text_alias_emits_human_rows(
    tmp_path: Path,
    daemon_archive: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_id = _seed_messages(
        tmp_path,
        {"id": "m1", "role": "user", "text": "hello there"},
        {"id": "m2", "role": "assistant", "text": "general kenobi"},
    )

    run_messages(_env(), _seeded_request(tmp_path), session_id=session_id, output_format="text")

    rendered = capsys.readouterr().out
    assert "hello there" in rendered
    assert "general kenobi" in rendered


def test_run_messages_markdown_uses_structural_shell_outcome(
    tmp_path: Path,
    daemon_archive: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_id = _seed_messages(
        tmp_path,
        {
            "id": "m-use",
            "role": "assistant",
            "text": "",
            "blocks": [
                {
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
            "text": "",
            "blocks": [
                {
                    "type": "tool_result",
                    "tool_id": "call-1",
                    "text": "ERROR appears in output",
                    "tool_result_is_error": 0,
                    "tool_result_exit_code": 0,
                }
            ],
        },
    )

    run_messages(_env(), _seeded_request(tmp_path), session_id=session_id)

    rendered = capsys.readouterr().out
    assert "### Shell command · succeeded" in rendered
    assert "`is_error=false`" in rendered
    assert "`exit_code=0`" in rendered
    assert "ERROR appears in output" in rendered
    assert "FAILED" not in rendered


def _hooks_invocation(*, output_format: str = "json", destination: str = "terminal") -> ReadViewInvocation:
    return ReadViewInvocation(
        view="hooks",
        session_id="conv-hooks",
        output_format=output_format,
        destination=destination,
        out_path=None,
    )


def _hooks_result(evidence: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        value={
            "outcome": {"state": "ok"},
            "session": {"session_id": "conv-hooks", "messages": []},
            "session_id": "conv-hooks",
            "kind": "hooks",
            "evidence": evidence,
            "total": cast(int, evidence.get("total", 0)),
            "limit": 1,
            "offset": 0,
            "next_offset": None,
            "continuation": None,
            "complete": True,
        },
        envelope=None,
        authority={"mode": "direct"},
    )


_HOOK_EVIDENCE: dict[str, object] = {
    "session_id": "conv-hooks",
    "total": 3,
    "by_event_type": {"PostToolUse": 2, "PreToolUse": 1},
    "first_observed_at": "2026-07-10T10:00:00Z",
    "last_observed_at": "2026-07-10T10:00:02Z",
}


def test_read_hooks_renders_the_evidence_body_from_session_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The hooks view lowers onto ``session.read`` and renders what it returns.

    Anti-vacuity: reopening an archive in the view, or rendering the whole
    operation result instead of its evidence body, turns this red -- as does
    lowering without ``kind="hooks"``, which would return a transcript.
    """

    env = _env()
    with patch("polylogue.cli.operation_kernel.dispatch") as dispatch:
        dispatch.return_value = _hooks_result(_HOOK_EVIDENCE)
        run_read_hooks(env, _request(tmp_path), _hooks_invocation())

    request = dispatch.call_args.args[1]
    assert request.operation == "session.read"
    assert request.payload == {"ref": "conv-hooks", "kind": "hooks"}

    payload = json.loads(capsys.readouterr().out)
    assert payload == _HOOK_EVIDENCE
    _ui_print(env).assert_not_called()


def test_read_hooks_renders_yaml_when_asked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Anti-vacuity: render JSON regardless of format and this stops matching."""

    yaml_env = _env()
    with patch("polylogue.cli.operation_kernel.dispatch") as dispatch:
        dispatch.return_value = _hooks_result(_HOOK_EVIDENCE)
        run_read_hooks(yaml_env, _request(tmp_path), _hooks_invocation(output_format="yaml"))
    assert "PostToolUse" in capsys.readouterr().out
    _ui_print(yaml_env).assert_not_called()


def test_read_hooks_refuses_rather_than_rendering_an_empty_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refused read exits non-zero instead of reading as "no hook events".

    Anti-vacuity: swallow the refusal and print an empty body, let the
    exception escape untyped, or re-class it as ``click.UsageError`` (exit 2,
    the *empty* status), and this turns red.
    """

    from polylogue.cli.operation_kernel import OperationFailedError
    from polylogue.cli.render.outcome import EMPTY_EXIT_CODE, FAILED_READ_EXIT_CODE

    env = _env()
    with patch("polylogue.cli.operation_kernel.dispatch") as dispatch:
        dispatch.side_effect = OperationFailedError("invalid_request", "session not found: missing")
        with pytest.raises(SystemExit) as caught:
            run_read_hooks(env, _request(tmp_path), _hooks_invocation())
    assert caught.value.code == FAILED_READ_EXIT_CODE
    assert caught.value.code != EMPTY_EXIT_CODE
    assert "session not found: missing" in capsys.readouterr().err


def test_read_hooks_names_the_daemon_refusal_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Anti-vacuity: class an unavailable daemon as a usage mistake (exit 2,
    which is the empty status), or let it escape as a traceback, and this turns
    red."""

    from polylogue.cli.operation_kernel import OperationUnavailableError
    from polylogue.cli.render.outcome import EMPTY_EXIT_CODE, FAILED_READ_EXIT_CODE

    env = _env()
    with patch("polylogue.cli.operation_kernel.dispatch") as dispatch:
        dispatch.side_effect = OperationUnavailableError("daemon is unavailable for operation: session.read")
        with pytest.raises(SystemExit) as caught:
            run_read_hooks(env, _request(tmp_path), _hooks_invocation())
    assert caught.value.code == FAILED_READ_EXIT_CODE
    assert caught.value.code != EMPTY_EXIT_CODE
    err = capsys.readouterr().err
    assert "daemon is unavailable" in err
    assert "polylogued run" in err  # the remedy, not just the fault
    assert "Usage:" not in err
