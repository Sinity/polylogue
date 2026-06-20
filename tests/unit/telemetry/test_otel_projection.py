"""Tests for outbound OTel-style projection (#2183)."""

from __future__ import annotations

from polylogue.surfaces.payloads import (
    ActionQueryRowPayload,
    ContextSnapshotQueryRowPayload,
    MessageQueryRowPayload,
    ObservedEventQueryRowPayload,
    RunQueryRowPayload,
)
from polylogue.telemetry.otel_projection import project_query_unit_rows_to_otel


def test_project_query_unit_rows_to_otel_preserves_polylogue_refs() -> None:
    payload = project_query_unit_rows_to_otel(
        "session:codex-session:root",
        (
            RunQueryRowPayload(
                run_ref="run:codex-session:child",
                session_id="codex-session:root",
                origin="codex-session",
                title="OTel projection",
                native_session_id="child",
                native_parent_session_id="root",
                parent_run_ref=None,
                agent_ref="agent:codex/Explore",
                lineage_refs=("run:codex-session:root",),
                provider_origin="codex",
                harness="codex",
                role="subagent",
                cwd="/realm/project/polylogue",
                git_branch="feature/otel",
                status="completed",
                confidence="observed",
                transcript_ref="codex-session:root",
                evidence_refs=("codex-session:root::m1",),
                context_snapshot_ref="context-snapshot:codex-session:child:session_start",
            ),
            ActionQueryRowPayload(
                session_id="codex-session:root",
                message_id="m2",
                origin="codex-session",
                title="OTel projection",
                tool_use_block_id="tool-use-1",
                tool_result_block_id="tool-result-1",
                tool_name="Bash",
                semantic_type="shell",
                tool_command="pytest tests/unit/telemetry",
                tool_path="/realm/project/polylogue/secrets.txt",
                output_text="sensitive terminal output",
            ),
            ObservedEventQueryRowPayload(
                event_ref="observed-event:codex-session:root:permission",
                session_id="codex-session:root",
                origin="codex-session",
                title="OTel projection",
                kind="permission",
                summary="permission requested",
                delivery_state="delivered",
                subject_ref="run:codex-session:child",
                object_refs=("run:codex-session:child",),
                evidence_refs=("codex-session:root::m2",),
            ),
            ContextSnapshotQueryRowPayload(
                snapshot_ref="context-snapshot:codex-session:child:session_start",
                session_id="codex-session:root",
                origin="codex-session",
                title="OTel projection",
                run_ref="run:codex-session:child",
                boundary="session_start",
                inheritance_mode="explicit",
                segment_refs=("message:m1",),
                evidence_refs=("codex-session:root::m1",),
                metadata={"source": "fixture"},
            ),
        ),
    )

    assert payload.mode == "otel-projection"
    assert payload.trace_count == 1
    assert payload.span_count == 2
    assert payload.log_count == 2
    assert "run:codex-session:child" in payload.refs
    assert "context-snapshot:codex-session:child:session_start" in payload.refs

    run_span = next(span for span in payload.spans if span.attributes.get("polylogue.run.ref"))
    assert run_span.attributes["polylogue.run.ref"] == "run:codex-session:child"
    assert run_span.attributes["polylogue.run.cwd.redacted"] is True
    assert "codex-session:root::m1" in run_span.links

    action_span = next(span for span in payload.spans if span.attributes.get("polylogue.action.tool_name") == "Bash")
    assert action_span.attributes["polylogue.action.output_present"] is True
    assert action_span.attributes["polylogue.action.output_length"] == len("sensitive terminal output")
    assert action_span.attributes["polylogue.action.tool_path.redacted"] is True
    assert "sensitive terminal output" not in action_span.to_json()
    assert "/realm/project/polylogue/secrets.txt" not in action_span.to_json()

    assert any(log.attributes.get("polylogue.observed_event.kind") == "permission" for log in payload.logs)
    assert any(log.attributes.get("polylogue.context_snapshot.boundary") == "session_start" for log in payload.logs)


def test_message_text_is_omitted_unless_requested() -> None:
    row = MessageQueryRowPayload(
        message_id="m-secret",
        session_id="codex-session:root",
        origin="codex-session",
        title="OTel projection",
        role="assistant",
        message_type="assistant",
        position=1,
        word_count=3,
        text="private message text",
    )

    redacted = project_query_unit_rows_to_otel("session:codex-session:root", (row,))
    assert redacted.logs[0].body == "message assistant"
    assert redacted.logs[0].attributes["polylogue.message.text_included"] is False
    assert "private message text" not in redacted.to_json()

    explicit = project_query_unit_rows_to_otel("session:codex-session:root", (row,), include_message_text=True)
    assert explicit.logs[0].body == "private message text"
    assert explicit.logs[0].attributes["polylogue.message.text_included"] is True


def test_absolute_tool_paths_and_embedded_command_paths_are_redacted() -> None:
    rows = (
        ActionQueryRowPayload(
            session_id="codex-session:root",
            message_id="m-tmp",
            origin="codex-session",
            title="OTel projection",
            tool_use_block_id="tool-tmp",
            tool_name="Bash",
            semantic_type="shell",
            tool_command="cat /home/alice/secret.txt",
            tool_path="/tmp/secret.txt",
            output_text=None,
        ),
        ActionQueryRowPayload(
            session_id="codex-session:root",
            message_id="m-workspace",
            origin="codex-session",
            title="OTel projection",
            tool_use_block_id="tool-workspace",
            tool_name="Bash",
            semantic_type="shell",
            tool_command="pytest /workspace/polylogue/tests",
            tool_path="/workspace/polylogue/file.py",
            output_text=None,
        ),
        ActionQueryRowPayload(
            session_id="codex-session:root",
            message_id="m-mnt",
            origin="codex-session",
            title="OTel projection",
            tool_use_block_id="tool-mnt",
            tool_name="Read",
            semantic_type="file_read",
            tool_path="/mnt/data/export.json",
            output_text=None,
        ),
    )

    payload = project_query_unit_rows_to_otel("session:codex-session:root", rows)
    rendered = payload.to_json()

    assert "/home/alice/secret.txt" not in rendered
    assert "/tmp/secret.txt" not in rendered
    assert "/workspace/polylogue" not in rendered
    assert "/mnt/data/export.json" not in rendered
    for span in payload.spans:
        assert span.attributes.get("polylogue.action.tool_path.redacted") is True
        assert "polylogue.action.tool_path" not in span.attributes
        if span.attributes["polylogue.action.tool_use_block_id"] != "tool-mnt":
            assert span.attributes.get("polylogue.action.tool_command.redacted") is True
            assert "polylogue.action.tool_command" not in span.attributes
