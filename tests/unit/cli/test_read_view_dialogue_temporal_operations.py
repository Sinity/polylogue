"""Focused contracts for the pinned dialogue and temporal read operations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.cli.operation_kernel import OperationFailedError, OperationKernel, OperationRequest
from polylogue.cli.read_views.base import ReadViewInvocation
from polylogue.cli.read_views.standard import _read_dialogue_session, run_read_dialogue, run_read_temporal
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.core.protocols import VectorProvider
from polylogue.operations.daemon_protocol import MAX_OPERATION_RESULT_BYTES
from polylogue.operations.read_view_dialogue_temporal import execute_dialogue_read, execute_temporal_read
from polylogue.rendering.formatting import format_session
from polylogue.surfaces.projection_spec import RenderDestination
from tests.infra.builders import make_conv, make_msg


def _dialogue_session(*, text: str = "answer") -> Session:
    return make_conv(
        id="codex-session:one",
        messages=[make_msg(id="m1", role="assistant", text=text, material_origin="assistant_authored")],
    )


def test_dialogue_operation_uses_supplied_archive_page_and_reports_continuation() -> None:
    session = _dialogue_session()
    archive = MagicMock()
    archive.resolve_session_id.return_value = "codex-session:one"
    envelope = SimpleNamespace(
        total_message_count=151,
        messages=(object(),),
        lineage_complete=True,
        lineage_truncation_reason=None,
    )
    archive.read_session_page.return_value = envelope
    archive.read_summary.return_value = SimpleNamespace(display_label=None, display_label_source=None)

    def _window(
        _archive: object,
        _request: object,
        *,
        read: Callable[[int, int], tuple[list[object], int, object]],
        **_kwargs: object,
    ) -> SimpleNamespace:
        rows, total, _completeness = read(100, 0)
        return SimpleNamespace(rows=rows, total=total, next_offset=100, continuation="bound-token")

    with (
        patch("polylogue.operations.read_view_dialogue_temporal.archive_envelope_to_session", return_value=session),
        patch("polylogue.operations.read_view_dialogue_temporal.read_transcript_window_sync", side_effect=_window),
        patch("polylogue.operations.read_view_dialogue_temporal.replace", return_value=envelope),
    ):
        result = execute_dialogue_read(
            {"session_id": "one", "params": {}, "projection": {}, "offset": 0, "limit": 100}, archive=archive
        )
    archive.read_session_page.assert_called_once_with("codex-session:one", limit=100, offset=0)
    body = result["payload"]
    assert isinstance(body, dict)
    assert body["next_offset"] == 100
    assert body["continuation"] == "bound-token"
    assert body["total_message_count"] == 151
    session_payload = body["session"]
    assert isinstance(session_payload, dict)
    messages = session_payload["messages"]
    assert isinstance(messages, list)
    assert messages[0]["text"] == "answer"


def test_dialogue_cli_composes_daemon_pages_before_projection() -> None:
    first = _dialogue_session(text="first")
    second = make_conv(
        id="codex-session:one",
        messages=[make_msg(id="m2", role="assistant", text="second", material_origin="assistant_authored")],
    )
    pages = [
        (
            {
                "view": "dialogue",
                "payload": {
                    "session": first.model_dump(mode="json"),
                    "total_message_count": 2,
                    "next_offset": 1,
                    "continuation": "bound-token",
                },
            },
            SimpleNamespace(line=lambda: "daemon"),
        ),
        (
            {
                "view": "dialogue",
                "payload": {"session": second.model_dump(mode="json"), "total_message_count": 2, "next_offset": None},
            },
            SimpleNamespace(line=lambda: "daemon"),
        ),
    ]
    env = cast(AppEnv, SimpleNamespace(config=object()))
    with patch("polylogue.cli.read_dispatch.dispatch_read", side_effect=pages) as dispatch:
        session = _read_dialogue_session(env, RootModeRequest.from_params({}), "codex-session:one", None)
    assert session is not None
    assert [message.text for message in session.messages] == ["first", "second"]
    assert dispatch.call_count == 2
    assert dispatch.call_args_list[1].args[1].payload["offset"] == 1
    assert dispatch.call_args_list[1].args[1].payload["continuation"] == "bound-token"


def test_dialogue_markdown_file_streams_typed_pages_with_exact_rendering(tmp_path: Path) -> None:
    first = make_conv(
        id="codex-session:one",
        title="Example",
        messages=[make_msg(id="m1", role="user", text="# heading", material_origin="human_authored")],
    )
    second = make_conv(
        id="codex-session:one",
        title="Example",
        messages=[make_msg(id="m2", role="assistant", text="answer", material_origin="assistant_authored")],
    )
    pages = [
        (
            {
                "view": "dialogue",
                "payload": {
                    "session": first.model_dump(mode="json"),
                    "total_message_count": 2,
                    "next_offset": 1,
                    "continuation": "bound-token",
                },
            },
            SimpleNamespace(line=lambda: "daemon"),
        ),
        (
            {
                "view": "dialogue",
                "payload": {"session": second.model_dump(mode="json"), "total_message_count": 2, "next_offset": None},
            },
            SimpleNamespace(line=lambda: "daemon"),
        ),
    ]
    env = cast(AppEnv, SimpleNamespace(config=object(), ui=MagicMock()))
    out = tmp_path / "dialogue.md"
    invocation = ReadViewInvocation(
        view="dialogue",
        session_id="codex-session:one",
        output_format=None,
        destination=RenderDestination.FILE,
        out_path=str(out),
    )
    original_format = format_session

    def bounded_format(session: Session, output_format: str, fields: str | None) -> str:
        assert len(session.messages) <= 1
        return original_format(session, output_format, fields)

    with (
        patch("polylogue.cli.read_dispatch.dispatch_read", side_effect=pages) as dispatch,
        patch("polylogue.cli.read_views.standard._read_dialogue_session", side_effect=AssertionError("eager read")),
        patch("polylogue.cli.read_views.standard.format_session", side_effect=bounded_format),
        patch("polylogue.cli.read_views.standard._warn_on_written_file_secret_candidates") as scan,
    ):
        run_read_dialogue(env, RootModeRequest.from_params({}), invocation)
    combined = first.model_copy(update={"messages": (*first.messages, *second.messages)})
    assert out.read_text(encoding="utf-8") == original_format(combined, "markdown", None)
    assert dispatch.call_count == 2
    assert dispatch.call_args_list[1].args[1].payload["continuation"] == "bound-token"
    scan.assert_called_once_with(env, str(out))


def test_dialogue_single_giant_message_has_typed_wire_refusal() -> None:
    oversized = {
        "view": "dialogue",
        "payload": {"session": {"messages": [{"text": "x" * (MAX_OPERATION_RESULT_BYTES + 1)}]}},
    }
    kernel = OperationKernel(lambda _request: {"operation": "read.dialogue", "result": oversized})
    with pytest.raises(OperationFailedError) as exc:
        kernel.execute(OperationRequest("read.dialogue", {"session_id": "codex-session:one", "params": {}}))
    assert exc.value.code == "result_too_large"


def test_temporal_operation_keeps_session_message_action_event_families() -> None:
    summary = SessionSummary.model_validate(
        {"id": "codex-session:one", "origin": "codex-session", "title": "Example", "created_at": "2026-01-01T12:00:00Z"}
    )
    archive = MagicMock()
    archive.resolve_session_id.return_value = "codex-session:one"
    archive.read_summary.return_value = object()
    archive.query_session_messages.return_value = []
    archive.query_session_action_occurrences.return_value = []
    with patch("polylogue.operations.read_view_dialogue_temporal.archive_summary_to_domain", return_value=summary):
        result = execute_temporal_read({"session_id": "one", "params": {}, "projection": {}}, archive=archive)
    body = result["payload"]
    assert isinstance(body, dict)
    window = body["temporal_window"]
    assert isinstance(window, dict)
    assert window["event_count"] == 1
    assert window["family_counts"] == {"archive-session": 1}
    archive.query_session_messages.assert_called_once_with(["codex-session:one"], limit=8, sort_direction="asc")
    archive.query_session_action_occurrences.assert_called_once_with(
        ["codex-session:one"], limit=4, sort_direction="asc"
    )


def test_temporal_operation_uses_pinned_vector_provider_for_semantic_selection() -> None:
    archive = MagicMock()
    provider = cast(VectorProvider, object())
    with patch("polylogue.operations.read_view_dialogue_temporal._archive_summaries", return_value=[]) as select:
        result = execute_temporal_read(
            {"session_id": None, "params": {"similar_text": "needle"}, "projection": {}},
            archive=archive,
            vector_provider=provider,
        )
    body = result["payload"]
    assert isinstance(body, dict)
    window = body["temporal_window"]
    assert isinstance(window, dict)
    assert window["event_count"] == 0
    assert select.call_args.args[0].vector_provider is provider


def test_temporal_cli_dispatches_declared_operation_without_local_builder() -> None:
    from polylogue.surfaces.temporal_evidence import build_temporal_evidence_window

    window = build_temporal_evidence_window([]).model_dump(mode="json")
    env = cast(AppEnv, SimpleNamespace(config=object()))
    invocation = ReadViewInvocation(
        view="temporal", session_id=None, output_format="json", destination=RenderDestination.STDOUT, out_path=None
    )
    with (
        patch(
            "polylogue.cli.read_dispatch.dispatch_read",
            return_value=(
                {"view": "temporal", "payload": {"temporal_window": window}},
                SimpleNamespace(line=lambda: "daemon"),
            ),
        ) as dispatch,
        patch("polylogue.cli.read_views.standard.build_read_temporal_window", side_effect=AssertionError("local read")),
        patch("polylogue.cli.read_views.standard.deliver_content") as deliver,
    ):
        run_read_temporal(env, RootModeRequest.from_params({"query": ("repo:example",)}), invocation)
    assert dispatch.call_args.args[1].operation == "read.temporal"
    assert dispatch.call_args.args[1].payload["params"]["query"] == ["repo:example"]
    assert "temporal_window" in deliver.call_args.args[1]
