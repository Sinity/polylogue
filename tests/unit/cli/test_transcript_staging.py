"""Complete CLI transcripts retain records and output until their delivery closes."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import ExitStack
from types import SimpleNamespace
from typing import TextIO, cast

import pytest

from polylogue.cli import archive_query
from polylogue.cli.operation_kernel import OperationEnvelopeError, OperationRequest
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.cli.transcript import TranscriptRead, TranscriptRecords, transcript_records
from polylogue.config import Config
from polylogue.surfaces.outcome import decide_outcome


@pytest.mark.parametrize("page_count", [0, 1, 3])
def test_transcript_pages_are_private_before_the_next_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], page_count: int
) -> None:
    stages: list[TranscriptRecords] = []
    original = TranscriptRecords.__init__

    def initialize(stage: TranscriptRecords, stream: TextIO) -> None:
        original(stage, stream)
        stages.append(stage)

    monkeypatch.setattr(TranscriptRecords, "__init__", initialize)
    monkeypatch.setattr(archive_query, "_SESSION_READ_WINDOW", 2)
    requests = 0

    def read(_config: Config, request: OperationRequest, **_kwargs: object) -> tuple[dict[str, object], None]:
        nonlocal requests
        assert len(stages) == 1 and stages[0].count == requests * 2
        assert os.fstat(stages[0].stream.fileno()).st_mode & 0o777 == 0o600
        start = requests * 2
        requests += 1
        messages = [{"position": index, "text": "neutral é"} for index in range(start, start + 2)] if page_count else []
        return {
            "session": {"session_id": "synthetic", "messages": messages},
            "outcome": decide_outcome(matched=len(messages)).to_dict(),
            "lineage_complete": True,
            "lineage_truncation_reason": None,
            "complete": requests >= page_count,
            "continuation": f"page-{requests}",
        }, None

    monkeypatch.setattr(archive_query, "dispatch_read", read)
    with ExitStack() as lifetime:
        result = archive_query._read_session_windows(
            cast(Config, object()), "synthetic", daemon_disabled=True, lifetime=lifetime
        )
        assert "messages" not in result.header
        assert result.records.count == page_count * 2
        if page_count:
            archive_query._emit_session_result(result, output_format="json", fields=None)
            document = json.loads(capsys.readouterr().out)
            assert [message["position"] for message in document["messages"]] == list(range(page_count * 2))
        else:
            assert list(result.records.records()) == []
    assert stages[0].stream.closed
    with pytest.raises(RuntimeError):
        next(stages[0].records())


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_late_transcript_render_failure_publishes_no_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: type[BaseException]
) -> None:
    captured: list[TranscriptRecords] = []

    def messages(_read: TranscriptRead) -> Iterator[dict[str, object]]:
        yield {"position": 0}
        raise failure("synthetic late render failure")

    def execute(_env: AppEnv, _request: RootModeRequest) -> None:
        with transcript_records() as records:
            captured.append(records)
            records.append({"position": 0})
            records.append({"position": 1})
            read = TranscriptRead(
                {
                    "session_id": "synthetic",
                    "native_id": None,
                    "origin": "codex",
                    "title": None,
                    "active_leaf_message_id": None,
                    "lineage_complete": True,
                    "lineage_truncation_reason": None,
                    "outcome": decide_outcome(matched=2).to_dict(),
                },
                records,
                decide_outcome(matched=2),
            )
            archive_query._emit_session_result(read, output_format="json", fields=None)

    monkeypatch.setattr(archive_query, "_projected_session_messages", messages)
    monkeypatch.setattr(archive_query, "_execute_archive_query_stdout", execute)
    env = cast(AppEnv, SimpleNamespace(begin_timing=lambda *_args: None, finish_timing=lambda *_args: None))
    with pytest.raises(failure):
        archive_query.execute_archive_query(env, RootModeRequest.from_params({"output_format": "json"}))
    assert capsys.readouterr().out == ""
    assert captured[0].stream.closed


@pytest.mark.parametrize(
    "rows,token", [([], "after-first"), ([{"id": "last"}], None), ([{"id": "last"}], "after-first")]
)
def test_complete_transcript_refuses_nonadvancing_windows(
    monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]], token: str | None
) -> None:
    calls = 0

    def read(_config: Config, _request: OperationRequest, **_kwargs: object) -> tuple[dict[str, object], None]:
        nonlocal calls
        calls += 1
        return {
            "session": {"session_id": "synthetic", "messages": [{"id": "first"}] if calls == 1 else rows},
            "outcome": decide_outcome(matched=1).to_dict(),
            "lineage_complete": True,
            "lineage_truncation_reason": None,
            "complete": False,
            "continuation": "after-first" if calls == 1 else token,
        }, None

    monkeypatch.setattr(archive_query, "dispatch_read", read)
    with ExitStack() as lifetime, pytest.raises(OperationEnvelopeError):
        archive_query._read_session_windows(
            cast(Config, object()), "synthetic", daemon_disabled=True, lifetime=lifetime
        )
    assert calls == 2


def test_incremental_csv_preserves_record_newlines_and_cell_whitespace() -> None:
    from polylogue.rendering.formatting import format_session
    from tests.infra.builders import make_conv, make_msg

    session = make_conv(id="s", messages=[make_msg(id="m", role="user", text=" trailing  ")])
    assert format_session(session, "csv", None) == (
        "session_id,message_id,role,timestamp,text\r\ns,m,user,, trailing  "
    )


def test_completed_output_descriptor_preserves_unicode_and_csv_newlines() -> None:
    from polylogue.cli.output_stream import output_chunks, output_stream, strip_final_newlines

    expected = "é" * 40000 + "\r\nrow, trailing  "
    with output_stream() as staged:
        staged.write(expected + "\n" * 70000)
        strip_final_newlines(staged)
        assert "".join(output_chunks(staged)) == expected
    assert staged.closed


def test_large_transcript_body_is_staged_and_rendered_one_page_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tracemalloc

    calls = 0
    size = 2 * 1024 * 1024
    page_count = 20
    emitted = 0

    def read(_config: Config, _request: OperationRequest, **_kwargs: object) -> tuple[dict[str, object], None]:
        nonlocal calls
        calls += 1
        return {
            "session": {"session_id": "synthetic", "messages": [{"position": calls, "text": "x" * size}]},
            "outcome": decide_outcome(matched=1).to_dict(),
            "lineage_complete": True,
            "lineage_truncation_reason": None,
            "complete": calls == page_count,
            "continuation": f"page-{calls}",
        }, None

    def echo(chunk: str | None = None, **_kwargs: object) -> None:
        nonlocal emitted
        emitted += len(chunk or "")

    monkeypatch.setattr(archive_query, "dispatch_read", read)
    monkeypatch.setattr("click.echo", echo)
    tracemalloc.start()
    try:
        with ExitStack() as lifetime:
            result = archive_query._read_session_windows(
                cast(Config, object()), "synthetic", daemon_disabled=True, lifetime=lifetime
            )
            assert result.records.count == page_count
            archive_query._emit_session_result(result, output_format="json", fields=None)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert emitted > size * page_count
    # Representative Python allocations only; this is neither RSS nor a bound
    # on branch positions/tool dependency identifiers for all possible inputs.
    assert peak < 24 * 1024 * 1024
