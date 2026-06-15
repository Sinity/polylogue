from __future__ import annotations

import pytest
from pydantic import ValidationError

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin
from polylogue.insights.transforms import (
    RECOVERY_TRANSFORM,
    TRANSFORM_REGISTRY,
    RecoveryEvent,
    RunStateSummary,
    SubagentReport,
    ToolSummary,
    TransformRawRef,
    compile_recovery_digest,
)
from polylogue.types import SessionId


def _session() -> Session:
    return Session(
        id=SessionId("codex-session:demo"),
        origin=Origin.CODEX_SESSION,
        title="Ship the backlog",
        git_branch="feature/demo",
        working_directories=("/realm/project/polylogue",),
        messages=MessageCollection(
            messages=[
                Message(
                    id="m1",
                    role=Role.USER,
                    text=(
                        "Goal: burn down the backlog\n"
                        "Done:\n"
                        "- #1910 merged\n"
                        "- #1818 closed\n"
                        "In flight:\n"
                        "- #1913 CI\n"
                        "Blockers:\n"
                        "- none\n"
                        "Next: merge PR #1911"
                    ),
                ),
                Message(
                    id="m2",
                    role=Role.ASSISTANT,
                    text="Decision: keep benchmarks outside coverage-gate for scope, not flakiness.",
                    blocks=[
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "tool_input": {"command": "devtools verify --quick"},
                        },
                        {
                            "type": "tool_result",
                            "tool_id": "tool-1",
                            "text": "ruff check ... ok\n20 passed in 50.28s\nhttps://github.com/Sinity/polylogue/pull/1911",
                        },
                        {
                            "type": "tool_use",
                            "id": "tool-read",
                            "name": "Read",
                            "tool_input": {"file_path": "polylogue/insights/transforms.py"},
                        },
                        {
                            "type": "tool_result",
                            "tool_id": "tool-read",
                            "text": "class ToolSummary\nhandler_kind: Literal",
                        },
                    ],
                ),
                Message(
                    id="m3",
                    role=Role.ASSISTANT,
                    text="MERGED #1910 (machine mode)\n✓ Closed issue Sinity/polylogue#1818",
                    blocks=[
                        {
                            "type": "tool_use",
                            "id": "tool-2",
                            "name": "Task",
                            "tool_input": {
                                "subagent_type": "Explore",
                                "prompt": "Map the transform surface and report caveats.",
                            },
                        },
                        {
                            "type": "tool_result",
                            "tool_id": "tool-2",
                            "text": (
                                "Subagent done: opened https://github.com/Sinity/polylogue/pull/1912\n"
                                "6 passed in 0.72s\n"
                                "Caveat: storage persistence not included."
                            ),
                        },
                    ],
                ),
            ]
        ),
    )


def test_registry_declares_no_llm_recovery_transform() -> None:
    assert TRANSFORM_REGISTRY[RECOVERY_TRANSFORM.transform_id] == RECOVERY_TRANSFORM
    assert RECOVERY_TRANSFORM.deterministic is True
    assert RECOVERY_TRANSFORM.uses_llm is False


def test_compile_recovery_digest_extracts_small_evidence_linked_bundle() -> None:
    digest = compile_recovery_digest(_session())

    assert digest.transform.transform_id == "recovery_digest_v0"
    assert digest.transform.transform_version == 1
    assert digest.transform.input_session_id == "codex-session:demo"
    assert digest.size_metrics.message_count == 3
    assert digest.size_metrics.subagent_report_count == 1
    assert digest.size_metrics.run_state_count == 1
    assert digest.size_metrics.raw_bytes > digest.size_metrics.resume_bundle_bytes
    assert digest.size_metrics.resume_to_raw_ratio < 1
    assert digest.role_counts == {"user": 1, "assistant": 2}

    assert len(digest.tool_summaries) == 2
    tool = next(item for item in digest.tool_summaries if item.tool_name == "Bash")
    assert tool.tool_name == "Bash"
    assert tool.command == "devtools verify --quick"
    assert tool.handler_kind == "test"
    assert tool.status == "ok"
    assert tool.line_count == 3
    assert tool.pr_refs == ("#1911",)
    assert tool.test_evidence == ("ruff check ... ok", "20 passed in 50.28s")
    assert {ref.ref_kind for ref in tool.raw_refs} == {"block"}

    read_tool = next(item for item in digest.tool_summaries if item.tool_name == "Read")
    assert read_tool.handler_kind == "file_read"
    assert read_tool.command == "polylogue/insights/transforms.py"
    assert read_tool.line_count == 2
    assert read_tool.file_refs == ("polylogue/insights/transforms.py",)

    assert len(digest.subagent_reports) == 1
    subagent = digest.subagent_reports[0]
    assert subagent.subagent_type == "Explore"
    assert subagent.prompt == "Map the transform surface and report caveats."
    assert "Subagent done" in subagent.final_report_preview
    assert subagent.pr_refs == ("#1912",)
    assert subagent.test_evidence == ("6 passed in 0.72s",)
    assert subagent.caveats == ("Caveat: storage persistence not included.",)
    assert {ref.ref_kind for ref in subagent.raw_refs} == {"block"}

    assert digest.run_state is not None
    assert digest.run_state.goal == "burn down the backlog"
    assert digest.run_state.done == ("#1910 merged", "#1818 closed")
    assert digest.run_state.in_flight == ("#1913 CI",)
    assert digest.run_state.blockers == ("none",)
    assert digest.run_state.next_actions == ("merge PR #1911",)
    assert digest.run_state.raw_refs[0].message_id == "m1"

    event_summaries = {event.summary for event in digest.events}
    assert "PR #1911 opened" in event_summaries
    assert "PR #1910 merged" in event_summaries
    assert "Issue #1818 closed" in event_summaries
    assert "20 tests passed" in event_summaries
    assert "ruff check passed" in event_summaries

    candidates = {candidate.text for candidate in digest.decision_candidates}
    assert "goal: burn down the backlog" in candidates
    assert "next: merge PR #1911" in candidates
    assert "keep benchmarks outside coverage-gate for scope, not flakiness." in candidates

    assert "# Resume: Ship the backlog" in digest.resume_markdown
    assert "feature/demo" in digest.resume_markdown
    assert "## Subagents" in digest.resume_markdown
    assert "Explore — Map the transform surface and report caveats." in digest.resume_markdown
    assert "## Run State" in digest.resume_markdown
    assert "- done: #1910 merged" in digest.resume_markdown
    assert "- next: merge PR #1911" in digest.resume_markdown
    assert "Bash [test]" in digest.resume_markdown
    assert "Read [file_read]" in digest.resume_markdown
    assert "Raw refs are available" in digest.resume_markdown


def test_every_extracted_claim_carries_raw_refs() -> None:
    digest = compile_recovery_digest(_session())

    assert digest.raw_refs
    for tool in digest.tool_summaries:
        assert tool.raw_refs
        assert all(ref.session_id == digest.session_id for ref in tool.raw_refs)
    for report in digest.subagent_reports:
        assert report.raw_refs
        assert all(ref.session_id == digest.session_id for ref in report.raw_refs)
    assert digest.run_state is not None
    assert digest.run_state.raw_refs
    assert all(ref.session_id == digest.session_id for ref in digest.run_state.raw_refs)
    for event in digest.events:
        assert event.raw_refs
        assert all(ref.session_id == digest.session_id for ref in event.raw_refs)
    for candidate in digest.decision_candidates:
        assert candidate.raw_refs
        assert all(ref.session_id == digest.session_id for ref in candidate.raw_refs)


def test_github_cli_and_failed_check_events_are_extracted() -> None:
    session = Session(
        id=SessionId("codex-session:github-events"),
        origin=Origin.CODEX_SESSION,
        title="GitHub event extraction",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m-gh",
                    role=Role.ASSISTANT,
                    text=("✓ Created pull request #1930\nshowcase-verify ... FAILED (2.3s)\nAnalyze (python) ... ok"),
                )
            ]
        ),
    )

    digest = compile_recovery_digest(session)
    events = {(event.kind, event.summary) for event in digest.events}

    assert ("pr_opened", "PR #1930 opened") in events
    assert ("check_failed", "showcase-verify failed") in events
    assert ("check_passed", "Analyze (python) passed") in events


def test_claim_models_reject_missing_raw_refs() -> None:
    with pytest.raises(ValidationError):
        ToolSummary(tool_name="Bash", raw_refs=())
    with pytest.raises(ValidationError):
        SubagentReport(raw_refs=())
    with pytest.raises(ValidationError):
        RunStateSummary(raw_refs=())
    with pytest.raises(ValidationError):
        RecoveryEvent(kind="test_passed", summary="1 tests passed", raw_refs=())


def test_raw_refs_include_message_and_block_preview() -> None:
    digest = compile_recovery_digest(_session())

    tool_ref = digest.tool_summaries[0].raw_refs[0]
    assert tool_ref.message_id == "m2"
    assert tool_ref.block_index == 0
    assert tool_ref.preview == "devtools verify --quick"

    event_ref = next(event for event in digest.events if event.summary == "Issue #1818 closed").raw_refs[0]
    assert event_ref.message_id == "m3"
    assert event_ref.preview.startswith("MERGED #1910")


def test_raw_ref_requires_session_id() -> None:
    with pytest.raises(ValidationError):
        TransformRawRef(session_id="", message_id="m1")
