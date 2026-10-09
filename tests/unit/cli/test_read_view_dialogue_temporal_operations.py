"""Focused contracts for the pinned dialogue and temporal read operations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.cli.operation_kernel import OperationKernel, OperationRequest
from polylogue.cli.read_views.base import ReadViewInvocation
from polylogue.cli.read_views.standard import _read_dialogue_session, run_read_dialogue, run_read_temporal
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.core.protocols import VectorProvider
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


def test_dialogue_single_giant_message_preserves_the_complete_value() -> None:
    text = "x" * (9 * 1024 * 1024)
    value = {"view": "dialogue", "payload": {"session": {"messages": [{"text": text}]}}}
    kernel = OperationKernel(lambda _request: {"operation": "read.dialogue", "result": value})
    result = kernel.execute(OperationRequest("read.dialogue", {"session_id": "codex-session:one", "params": {}}))
    assert result.value is value
    assert result.value["payload"]["session"]["messages"][0]["text"] == text


def test_temporal_operation_keeps_session_message_action_event_families() -> None:
    summary = SessionSummary.model_validate(
        {"id": "codex-session:one", "origin": "codex-session", "title": "Example", "created_at": "2026-01-01T12:00:00Z"}
    )
    archive = MagicMock()
    archive.resolve_session_id.return_value = "codex-session:one"
    archive.read_summary.return_value = object()
    archive.read_session_page.return_value = SimpleNamespace(
        messages=(), total_message_count=0, lineage_complete=True, lineage_truncation_reason=None
    )
    archive.query_session_action_occurrences.return_value = []
    with patch("polylogue.operations.read_view_dialogue_temporal.archive_summary_to_domain", return_value=summary):
        result = execute_temporal_read({"session_id": "one", "params": {}, "projection": {}}, archive=archive)
    body = result["payload"]
    assert isinstance(body, dict)
    window = body["temporal_window"]
    assert isinstance(window, dict)
    assert window["event_count"] == 1
    assert window["family_counts"] == {"archive-session": 1}
    archive.read_session_page.assert_called_once_with("codex-session:one", limit=256, offset=0)
    archive.query_session_action_occurrences.assert_not_called()


def test_temporal_operation_uses_pinned_vector_provider_for_semantic_selection() -> None:
    archive = MagicMock()
    provider = cast(VectorProvider, object())
    with patch(
        "polylogue.operations.read_view_dialogue_temporal.select_read_view_summaries", return_value=[]
    ) as select:
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
        patch("polylogue.cli.read_views.standard.deliver_content") as deliver,
    ):
        with pytest.raises(SystemExit) as exc:
            run_read_temporal(env, RootModeRequest.from_params({"query": ("repo:example",)}), invocation)
        assert exc.value.code == 2
    assert dispatch.call_args.args[1].operation == "read.temporal"
    assert dispatch.call_args.args[1].payload["params"]["query"] == ["repo:example"]
    assert "temporal_window" in deliver.call_args.args[1]


def test_temporal_query_set_filters_content_before_the_window(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.storage_records import SessionBuilder

    selected_ids = []
    for index, text in enumerate(("keep older", "keep newer", "needle excluded")):
        builder = (
            SessionBuilder(tmp_path / "index.db", f"temporal-scope-{index}")
            .provider("codex")
            .title("Content scope")
            .updated_at(f"2026-01-0{index + 1}T12:00:00Z")
            .add_message(text=text, timestamp=f"2026-01-0{index + 1}T12:00:00Z")
        )
        builder.save()
        selected_ids.append(builder.native_session_id())
    with ArchiveStore(tmp_path, read_only=True) as archive:
        result = execute_temporal_read(
            {
                "params": {
                    "title": "Content scope",
                    "exclude_text": ["needle"],
                    "reverse": False,
                    "sort": "date",
                    "offset": 1,
                    "limit": 1,
                }
            },
            archive=archive,
        )
    window = cast(dict[str, Any], result["payload"])["temporal_window"]
    sessions = [event["source_ref"] for event in window["events"] if event["family"] == "archive-session"]
    assert sessions == [f"session:{selected_ids[0]}"]
    assert not any("needle" in event["label"] for event in window["events"])


def test_temporal_pages_past_null_timestamps_and_action_counts(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.storage_records import SessionBuilder

    builder = SessionBuilder(tmp_path / "index.db", "late-evidence").provider("codex")
    for index in range(260):
        builder.add_message(message_id=f"null-{index}", text="untimestamped", timestamp=None)
    builder.add_message(
        message_id="recorded",
        text="recorded evidence",
        timestamp="2026-01-01T12:00:00Z",
        blocks=[
            {
                "block_id": f"action-{index}",
                "type": "tool_use",
                "tool_name": "Bash",
                "tool_id": f"call-{index}",
                "tool_input": {"command": f"printf synthetic-{index}"},
            }
            for index in range(6)
        ],
    )
    builder.save()
    with ArchiveStore(tmp_path, read_only=True) as archive:
        result = execute_temporal_read({"session_id": builder.native_session_id()}, archive=archive)
    window = cast(dict[str, Any], result["payload"])["temporal_window"]
    assert window["family_counts"]["archive-message"] == 1
    assert window["family_counts"]["archive-action"] == 6
    assert not any("capped" in caveat for caveat in window["caveats"])
    assert window["outcome"]["state"] == "ok"


def test_read_projections_preserve_a_physical_lineage_gap(tmp_path: Path) -> None:
    from polylogue.operations.read_view_chronicle import execute_chronicle_read
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.storage_records import seed_attachment_library_lineage_archive

    ids = seed_attachment_library_lineage_archive(tmp_path)
    with (
        write_lease("test.temporal.lineage-gap", archive_root=tmp_path),
        ArchiveStore.open_existing(tmp_path, read_only=False) as archive,
    ):
        archive._conn.execute(
            "UPDATE session_links SET branch_point_message_id='missing-branch-point', "
            "branch_point_content_address=NULL WHERE src_session_id=?",
            (ids["child"],),
        )
        archive._conn.commit()
    with ArchiveStore(tmp_path, read_only=True) as archive:
        page = archive.read_session_page(ids["child"], limit=100, offset=0)
        assert page.lineage_complete is False
        assert page.lineage_truncation_reason == "dangling_branch_point"
        chronicle = cast(
            dict[str, Any], execute_chronicle_read({"session_id": ids["child"]}, archive=archive)["payload"]
        )
        temporal = cast(
            dict[str, Any], execute_temporal_read({"session_id": ids["child"]}, archive=archive)["payload"]
        )["temporal_window"]
    for payload in (chronicle, temporal):
        assert payload["outcome"]["state"] == "degraded"
        assert payload["outcome"]["reason"] == "lineage_truncated:dangling_branch_point"
    assert "lineage_truncated:dangling_branch_point" in chronicle["sessions"][0]["caveats"]
    assert "lineage_truncated:dangling_branch_point" in temporal["caveats"]


def test_temporal_cli_delivers_the_supplied_degraded_outcome() -> None:
    from polylogue.surfaces.temporal_evidence import build_temporal_evidence_window

    window = build_temporal_evidence_window([], gaps=("lineage_truncated:dangling_branch_point",))
    env = cast(AppEnv, SimpleNamespace(config=object()))
    invocation = ReadViewInvocation(
        view="temporal", session_id=None, output_format="json", destination=RenderDestination.STDOUT, out_path=None
    )
    with (
        patch(
            "polylogue.cli.read_dispatch.dispatch_read",
            return_value=(
                {"view": "temporal", "payload": {"temporal_window": window.model_dump(mode="json")}},
                SimpleNamespace(line=lambda: "daemon"),
            ),
        ),
        patch("polylogue.cli.read_views.standard.deliver_content") as deliver,
    ):
        with pytest.raises(SystemExit) as exc:
            run_read_temporal(env, RootModeRequest.from_params({}), invocation)
    assert exc.value.code == 1
    assert '"degraded"' in deliver.call_args.args[1]


def test_temporal_selected_child_keeps_its_composed_prefix_only(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.storage_records import seed_attachment_library_lineage_archive

    ids = seed_attachment_library_lineage_archive(tmp_path)
    with ArchiveStore(tmp_path, read_only=True) as archive:
        page = archive.read_session_page(ids["child"], limit=100, offset=0)
        expected = {f"message:{message.message_id}" for message in page.messages if message.occurred_at is not None}
        result = execute_temporal_read({"session_id": ids["child"]}, archive=archive)
    window = cast(dict[str, Any], result["payload"])["temporal_window"]
    actual = {event["source_ref"] for event in window["events"] if event["family"] == "archive-message"}
    assert len(expected) == 2
    assert actual == expected
    assert window["outcome"]["state"] == "ok"
