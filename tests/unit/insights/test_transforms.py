from __future__ import annotations

import pytest
from pydantic import ValidationError

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin
from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.run_projection import build_run_projection
from polylogue.insights.transforms import (
    RECOVERY_TRANSFORM,
    TRANSFORM_REGISTRY,
    ForensicIndexEntry,
    RecoveryEvent,
    RunStateSummary,
    SubagentReport,
    ToolSummary,
    TransformRawRef,
    compile_recovery_digest,
    render_recovery_report,
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
                    text=(
                        "Decision: keep benchmarks outside verify coverage for scope, not flakiness.\n"
                        # Prose claims about external state are deliberately present here.
                        # They must NOT become asserted events (#2482) — only the
                        # structured tool-result outcome below is a fact.
                        "Review posted on PR #1911\n"
                        "Read review on PR #1911\n"
                        "Addressed review on PR #1911"
                    ),
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
                            # Structured keystone outcome: exit 0 -> success.
                            "tool_result_exit_code": 0,
                        },
                        {
                            "type": "tool_use",
                            "id": "tool-close",
                            "name": "Bash",
                            "tool_input": {"command": "gh issue close 1818"},
                        },
                        {
                            "type": "tool_result",
                            "tool_id": "tool-close",
                            # No structured outcome -> status unknown, no event.
                            "text": "✓ Closed issue Sinity/polylogue#1818",
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
                        {
                            "type": "tool_use",
                            "id": "tool-commit",
                            "name": "Bash",
                            "tool_input": {"command": "git rev-parse HEAD"},
                        },
                        {
                            "type": "tool_result",
                            "tool_id": "tool-commit",
                            "text": "a8cd1c1516b29068ec9ce1493f262d663407ffa5",
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
                                "taskId": "task-42",
                                "child_session_id": "codex-session:child-42",
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


@pytest.mark.parametrize(
    ("origin", "expected_harness"),
    [
        ("codex-session", "codex"),
        ("claude-code-session", "claude-code"),
        ("chatgpt-export", "chatgpt"),
        ("gemini-cli-session", "local"),
        ("hermes-session", "local"),
        ("antigravity-session", "local"),
    ],
)
def test_run_projection_harness_uses_origin_predicate(origin: str, expected_harness: str) -> None:
    projection = build_run_projection(
        session_id=f"{origin}:demo",
        source_origin=origin,
        title="demo",
        git_branch=None,
        working_directories=(),
        session_raw_refs=(TransformRawRef(session_id=f"{origin}:demo"),),
        tool_summaries=(),
        subagent_reports=(),
        recovery_events=(),
    )

    [run] = projection.runs
    assert run.provider_origin == origin
    assert run.harness == expected_harness


def _sparse_session() -> Session:
    return Session(
        id=SessionId("codex-session:sparse"),
        origin=Origin.CODEX_SESSION,
        title="Sparse handoff",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m1",
                    role=Role.USER,
                    text="Please look into the archive behavior when you get back.",
                )
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
    assert digest.size_metrics.resume_bundle_bytes > 0
    assert digest.size_metrics.resume_bundle_bytes >= digest.size_metrics.normal_read_bytes
    assert digest.role_counts == {"user": 1, "assistant": 2}

    assert len(digest.tool_summaries) == 4
    tool = next(item for item in digest.tool_summaries if item.tool_name == "Bash")
    assert tool.tool_name == "Bash"
    assert tool.command == "devtools verify --quick"
    assert tool.handler_kind == "test"
    assert tool.status == "ok"
    assert tool.line_count == 3
    assert tool.pr_refs == ("#1911",)
    assert tool.test_evidence == ("ruff check ... ok", "20 passed in 50.28s")
    assert {ref.ref_kind for ref in tool.raw_refs} == {"block"}

    close_tool = next(item for item in digest.tool_summaries if item.command == "gh issue close 1818")
    # No structured outcome -> unknown, never inferred from output text (#2482).
    assert close_tool.status == "unknown"
    assert close_tool.pr_refs == ()
    assert close_tool.issue_refs == ("#1818",)

    read_tool = next(item for item in digest.tool_summaries if item.tool_name == "Read")
    assert read_tool.handler_kind == "file_read"
    assert read_tool.status == "unknown"
    assert read_tool.command == "polylogue/insights/transforms.py"
    assert read_tool.line_count == 2
    assert read_tool.file_refs == ("polylogue/insights/transforms.py",)

    commit_tool = next(item for item in digest.tool_summaries if item.command == "git rev-parse HEAD")
    assert commit_tool.handler_kind == "git"
    assert commit_tool.status == "unknown"
    assert commit_tool.commit_refs == ("a8cd1c1516b29068ec9ce1493f262d663407ffa5",)

    assert len(digest.subagent_reports) == 1
    subagent = digest.subagent_reports[0]
    assert subagent.subagent_type == "Explore"
    assert subagent.tool_id == "tool-2"
    assert subagent.task_id == "task-42"
    assert subagent.child_session_id == "codex-session:child-42"
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

    # Structured in-session outcomes only: the Bash test tool returned exit 0.
    # The prose "MERGED #1910" / "Closed issue #1818" / "Review ... on PR #1911"
    # produce NO events — external truths are not synthesized from prose (#2482).
    assert {(event.kind, event.summary) for event in digest.events} == {
        ("test_passed", "devtools verify --quick passed (exit 0)"),
    }
    [digest_event] = digest.events
    assert digest_event.tool_name == "Bash"
    assert digest_event.tool_id == "tool-1"
    assert digest_event.command == "devtools verify --quick"
    assert digest_event.handler_kind == "test"
    assert digest_event.status == "ok"

    projection = digest.run_projection
    assert projection.session_id == "codex-session:demo"
    assert [run.role for run in projection.runs] == ["main", "subagent"]
    main_run, child_run = projection.runs
    assert main_run.run_ref == ObjectRef(kind="run", object_id="codex-session:demo")
    assert main_run.agent_ref == ObjectRef(kind="agent", object_id="codex/main")
    assert main_run.lineage_refs == (main_run.run_ref,)
    assert main_run.provider_origin == "codex-session"
    assert main_run.harness == "codex"
    assert main_run.status == "completed"
    assert main_run.context_snapshot_ref == ObjectRef(
        kind="context-snapshot", object_id="codex-session:demo:session_start"
    )
    assert child_run.parent_run_ref == main_run.run_ref
    assert child_run.agent_ref == ObjectRef(kind="agent", object_id="codex/Explore")
    assert child_run.native_session_id == "codex-session:child-42"
    assert child_run.native_parent_session_id == "codex-session:demo"
    assert child_run.lineage_refs == (main_run.run_ref, child_run.run_ref)
    assert child_run.status == "completed"
    assert {snapshot.boundary for snapshot in projection.context_snapshots} == {"session_start", "subagent_start"}
    observed_kinds = {event.kind for event in projection.events}
    assert {
        "session_started",
        "tool_finished",
        "subagent_started",
        "subagent_finished",
        "test_passed",
    } <= observed_kinds
    # Fabricated event kinds never appear in the projection.
    assert observed_kinds.isdisjoint({"pr_merged", "pr_opened", "issue_closed", "review_posted", "check_passed"})
    test_event = next(event for event in projection.events if event.kind == "test_passed")
    assert test_event.delivery_state == "observed"
    assert test_event.object_refs == ()
    assert test_event.tool_name == "Bash"
    assert test_event.tool_id == "tool-1"
    assert test_event.command == "devtools verify --quick"
    assert test_event.handler_kind == "test"
    assert test_event.status == "ok"

    candidates = {candidate.text for candidate in digest.decision_candidates}
    assert "goal: burn down the backlog" in candidates
    assert "next: merge PR #1911" in candidates
    assert "keep benchmarks outside verify coverage for scope, not flakiness." in candidates

    assert "# Resume: Ship the backlog" in digest.resume_markdown
    assert "feature/demo" in digest.resume_markdown
    assert "## Subagents" in digest.resume_markdown
    assert "Explore — Map the transform surface and report caveats." in digest.resume_markdown
    assert "refs: tool_id=tool-2, task_id=task-42, child_session_id=codex-session:child-42" in digest.resume_markdown
    assert "## Run State" in digest.resume_markdown
    assert "- [assertion] done: #1910 merged" in digest.resume_markdown
    assert "- [assertion] next: merge PR #1911" in digest.resume_markdown
    assert "Bash [test]" in digest.resume_markdown
    assert "Read [file_read]" in digest.resume_markdown
    assert "Every packet row carries evidence refs and a support marker." in digest.resume_markdown


def test_every_extracted_claim_carries_raw_refs() -> None:
    digest = compile_recovery_digest(_session())

    assert digest.raw_refs
    assert digest.forensic_index.session_id == digest.session_id
    assert digest.forensic_index.entries
    assert all(entry.raw_ref.session_id == digest.session_id for entry in digest.forensic_index.entries)
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


def test_recovery_digest_enriches_subagent_reports_from_session_links() -> None:
    digest = compile_recovery_digest(
        _session(),
        session_links=(
            {
                "dst_origin": "codex-session",
                "dst_native_id": "child-42",
                "resolved_dst_session_id": "codex-session:child-42",
                "status": "resolved",
                "link_type": "subagent",
            },
        ),
    )

    [subagent] = digest.subagent_reports
    assert subagent.child_session_id == "codex-session:child-42"
    assert subagent.child_link_status == "resolved"
    assert subagent.child_link_type == "subagent"
    assert subagent.resolved_child_session_id == "codex-session:child-42"
    assert "child_link_status=resolved" in digest.resume_markdown
    assert "resolved_child_session_id=codex-session:child-42" in digest.resume_markdown


def test_forensic_index_groups_claims_by_raw_ref() -> None:
    digest = compile_recovery_digest(_session())
    entries = {entry.evidence_id: entry for entry in digest.forensic_index.entries}

    assert "codex-session:demo" in entries
    session_entry = entries["codex-session:demo"]
    assert session_entry.claim_kinds == ("digest",)
    assert session_entry.claim_labels == ("digest:session",)
    assert session_entry.raw_ref.ref_kind == "session"

    run_state_entry = entries["codex-session:demo::m1"]
    # m1 carries run_state + decision candidates, but no events: outcome events
    # come from tool-result blocks, not message prose (#2482).
    assert set(run_state_entry.claim_kinds) == {"run_state", "decision_candidate"}
    assert "run_state" in run_state_entry.claim_labels
    assert any(label.startswith("decision:run_state:") for label in run_state_entry.claim_labels)
    assert run_state_entry.raw_ref.preview.startswith("Goal: burn down")

    # The bash tool_use block carries both its tool summary and the structured
    # outcome event extracted from its paired tool_result.
    bash_call_entry = entries["codex-session:demo::m2::0"]
    assert set(bash_call_entry.claim_kinds) == {"tool_summary", "event"}
    assert "tool:Bash:tool-1" in bash_call_entry.claim_labels
    assert any(label.startswith("event:test_passed:") for label in bash_call_entry.claim_labels)
    assert bash_call_entry.raw_ref.preview == "devtools verify --quick"

    # Claims group by raw ref: the bash block carries more than one claim kind.
    assert digest.forensic_index.claim_count >= 1
    assert any(len(entry.claim_kinds) > 1 for entry in digest.forensic_index.entries)


def test_forensic_index_evidence_ids_round_trip_through_typed_refs() -> None:
    digest = compile_recovery_digest(_session())

    for entry in digest.forensic_index.entries:
        parsed = EvidenceRef.parse(entry.evidence_id)
        assert parsed.format() == entry.evidence_id
        assert parsed.ref_kind == entry.raw_ref.ref_kind
        assert parsed == entry.raw_ref.to_evidence_ref()


def test_work_packet_exposes_storage_free_continuation_bundle() -> None:
    digest = compile_recovery_digest(_session())

    packet = digest.work_packet()
    rendered = packet.render_markdown()

    assert packet.session_id == "codex-session:demo"
    assert packet.source_origin == "codex-session"
    assert packet.message_count == 3
    assert packet.target_refs == (
        ObjectRef(kind="session", object_id="codex-session:demo"),
        ObjectRef(kind="branch", object_id="feature/demo"),
    )
    assert {entry.section for entry in packet.entries} == {
        "execution",
        "events",
        "subagents",
        "run_state",
        "tools",
        "decisions",
    }
    assert all(entry.evidence_refs for entry in packet.entries)
    assert {entry.support for entry in packet.entries} >= {"raw_evidence", "assertion", "caveat", "inference"}
    assert any(ref.format() == "codex-session:demo::m2::0" for entry in packet.entries for ref in entry.evidence_refs)
    execution_entries = tuple(entry for entry in packet.entries if entry.section == "execution")
    assert any(
        entry.label == "run"
        and ObjectRef(kind="run", object_id="codex-session:demo") in entry.object_refs
        and ObjectRef(kind="agent", object_id="codex/main") in entry.object_refs
        and entry.metadata["status"] == "completed"
        for entry in execution_entries
    )
    assert any(
        entry.label == "run"
        and ObjectRef(kind="agent", object_id="codex/Explore") in entry.object_refs
        and entry.metadata["native_parent_session_id"] == "codex-session:demo"
        for entry in execution_entries
    )
    assert any(
        entry.label == "context_snapshot"
        and entry.metadata["boundary"] == "subagent_start"
        and entry.metadata["inheritance_mode"] == "summary"
        for entry in execution_entries
    )
    # The structured outcome is projected as an observed test_passed event.
    assert any(
        entry.label == "test_passed" and entry.metadata["delivery_state"] == "observed" for entry in execution_entries
    )

    # The "events" section carries only the structured outcome event — no
    # GitHub/issue/review object refs are synthesized.
    event_entries = tuple(entry for entry in packet.entries if entry.section == "events")
    assert [entry.label for entry in event_entries] == ["test_passed"]
    test_event = event_entries[0]
    assert test_event.text == "devtools verify --quick passed (exit 0)"
    assert test_event.metadata == {}
    assert test_event.object_refs == ()

    bash_entry = next(entry for entry in packet.entries if entry.section == "tools" and entry.label == "Bash")
    assert bash_entry.metadata == {
        "handler_kind": "test",
        "status": "ok",
        "pr_refs": "#1911",
        "test_evidence": "ruff check ... ok | 20 passed in 50.28s",
    }
    assert bash_entry.object_refs == (
        ObjectRef(kind="tool-call", object_id="codex-session:demo:tool-1"),
        ObjectRef(kind="github-pr", object_id="#1911"),
    )
    close_entry = next(
        entry for entry in packet.entries if entry.section == "tools" and entry.text == "gh issue close 1818"
    )
    assert close_entry.metadata == {
        "handler_kind": "github",
        "status": "unknown",
        "issue_refs": "#1818",
    }
    assert close_entry.object_refs == (
        ObjectRef(kind="tool-call", object_id="codex-session:demo:tool-close"),
        ObjectRef(kind="github-issue", object_id="#1818"),
    )
    read_entry = next(entry for entry in packet.entries if entry.section == "tools" and entry.label == "Read")
    assert read_entry.metadata == {
        "handler_kind": "file_read",
        "status": "unknown",
        "file_refs": "polylogue/insights/transforms.py",
    }
    assert read_entry.object_refs == (
        ObjectRef(kind="tool-call", object_id="codex-session:demo:tool-read"),
        ObjectRef(kind="file", object_id="polylogue/insights/transforms.py"),
    )
    commit_entry = next(
        entry for entry in packet.entries if entry.section == "tools" and entry.text == "git rev-parse HEAD"
    )
    assert commit_entry.metadata == {
        "handler_kind": "git",
        "status": "unknown",
        "commit_refs": "a8cd1c1516b29068ec9ce1493f262d663407ffa5",
    }
    assert commit_entry.object_refs == (
        ObjectRef(kind="tool-call", object_id="codex-session:demo:tool-commit"),
        ObjectRef(kind="commit", object_id="a8cd1c1516b29068ec9ce1493f262d663407ffa5"),
    )
    subagent_entry = next(
        entry for entry in packet.entries if entry.section == "subagents" and entry.label == "Explore"
    )
    assert subagent_entry.object_refs == (
        ObjectRef(kind="subagent-report", object_id="codex-session:demo:tool-2"),
        ObjectRef(kind="agent", object_id="codex/Explore"),
    )
    for entry in packet.entries:
        for ref in entry.object_refs:
            assert ObjectRef.parse(ref.format()) == ref
    assert "# Resume: Ship the backlog" in rendered
    assert "refs: session:codex-session:demo, branch:feature/demo" in rendered
    assert "## Execution Projection" in rendered
    assert "- [raw-evidence] run: Ship the backlog" in rendered
    assert (
        "refs: run:codex-session:demo, agent:codex/main, context-snapshot:codex-session:demo:session_start" in rendered
    )
    assert "details: role=main; status=completed; harness=codex; provider_origin=codex-session" in rendered
    assert "refs: run:codex-session:child-42, run:codex-session:demo, agent:codex/Explore" in rendered
    assert "native_parent_session_id=codex-session:demo" in rendered
    assert "- [raw-evidence] context_snapshot: subagent_start" in rendered
    assert "details: boundary=subagent_start; inheritance_mode=summary" in rendered
    assert "- [raw-evidence] test_passed: devtools verify --quick passed (exit 0)" in rendered
    assert "details: delivery_state=observed" in rendered
    assert "- [caveat] blocker: none" in rendered
    assert "refs: subagent-report:codex-session:demo:tool-2, agent:codex/Explore" in rendered
    assert "refs: tool_id=tool-2, task_id=task-42, child_session_id=codex-session:child-42" in rendered
    assert "- [raw-evidence] Bash [test] (ok) — devtools verify --quick" in rendered
    assert "refs: tool-call:codex-session:demo:tool-read, file:polylogue/insights/transforms.py" in rendered
    assert "refs: tool-call:codex-session:demo:tool-commit, commit:a8cd1c1516b29068ec9ce1493f262d663407ffa5" in rendered
    assert "details: pr_refs=#1911; test_evidence=ruff check ... ok | 20 passed in 50.28s" in rendered
    assert "details: file_refs=polylogue/insights/transforms.py" in rendered
    assert "details: commit_refs=a8cd1c1516b29068ec9ce1493f262d663407ffa5" in rendered
    # No fabricated GitHub/review event refs leak into the rendered packet.
    assert "github-review" not in rendered
    assert "pr_opened" not in rendered
    assert "issue_closed" not in rendered


def test_work_packet_marks_missing_evidence_explicitly() -> None:
    digest = compile_recovery_digest(_sparse_session())

    packet = digest.work_packet()
    rendered = packet.render_markdown()

    gap_entries = tuple(entry for entry in packet.entries if entry.section == "evidence_gaps")
    assert {entry.label for entry in gap_entries} == {"events", "subagents", "run_state", "tools", "decisions"}
    assert all(entry.support == "missing_evidence" for entry in gap_entries)
    assert all(entry.evidence_refs for entry in gap_entries)
    assert "## Evidence Gaps" in rendered
    assert "- [missing-evidence] run_state: No structured RunState section was extracted" in rendered
    assert "- [missing-evidence] tools: No tool execution summary was extracted." in rendered
    assert "- [missing-evidence] events: No structured tool or test outcome events were extracted." in rendered


def test_work_packet_does_not_promote_random_hex_to_commit_ref() -> None:
    session = Session(
        id=SessionId("codex-session:hex-output"),
        origin=Origin.CODEX_SESSION,
        title="Random hex output",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m-hex",
                    role=Role.ASSISTANT,
                    text="Tool output contained opaque ids.",
                    blocks=[
                        {
                            "type": "tool_use",
                            "id": "tool-hex",
                            "name": "Bash",
                            "tool_input": {"command": "cat artefact-id.txt"},
                        },
                        {
                            "type": "tool_result",
                            "tool_id": "tool-hex",
                            "text": "a8cd1c1516b29068ec9ce1493f262d663407ffa5",
                        },
                    ],
                )
            ]
        ),
    )

    [tool] = compile_recovery_digest(session).tool_summaries

    assert tool.commit_refs == ()


def _outcome_session(*, command: str, exit_code: int | None = None, is_error: bool | None = None) -> Session:
    result_block: dict[str, object] = {"type": "tool_result", "tool_id": "t", "text": "output"}
    if exit_code is not None:
        result_block["tool_result_exit_code"] = exit_code
    if is_error is not None:
        result_block["tool_result_is_error"] = 1 if is_error else 0
    return Session(
        id=SessionId("codex-session:outcome"),
        origin=Origin.CODEX_SESSION,
        title="Outcome fixture",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m",
                    role=Role.ASSISTANT,
                    text="",
                    blocks=[
                        {"type": "tool_use", "id": "t", "name": "Bash", "tool_input": {"command": command}},
                        result_block,
                    ],
                )
            ]
        ),
    )


def test_structured_outcomes_map_exit_code_to_event_kind() -> None:
    succeeded = compile_recovery_digest(_outcome_session(command="ls -la", exit_code=0))
    assert {(e.kind, e.summary) for e in succeeded.events} == {("command_succeeded", "ls -la succeeded (exit 0)")}

    failed = compile_recovery_digest(_outcome_session(command="ls /missing", exit_code=2))
    assert {(e.kind, e.summary) for e in failed.events} == {("command_failed", "ls /missing failed (exit 2)")}

    test_failed = compile_recovery_digest(_outcome_session(command="pytest tests/unit", exit_code=1))
    assert {(e.kind, e.summary) for e in test_failed.events} == {("test_failed", "pytest tests/unit failed (exit 1)")}
    # A failing test/command marks the run failed.
    assert test_failed.run_projection.runs[0].status == "failed"


def test_structured_outcomes_use_is_error_when_no_exit_code() -> None:
    ok = compile_recovery_digest(_outcome_session(command="gh pr view 1", is_error=False))
    assert {e.kind for e in ok.events} == {"command_succeeded"}
    assert "(exit" not in next(iter(ok.events)).summary

    bad = compile_recovery_digest(_outcome_session(command="gh pr view 1", is_error=True))
    assert {e.kind for e in bad.events} == {"command_failed"}


def test_unknown_outcome_yields_no_event() -> None:
    # tool_result with no structured fields -> unknown -> no fabricated event.
    digest = compile_recovery_digest(_outcome_session(command="ls"))
    assert digest.events == ()
    [tool] = digest.tool_summaries
    assert tool.status == "unknown"


def test_external_truths_are_not_synthesized_from_prose() -> None:
    # Prose asserting external state with NO structured tool outcome must produce
    # zero events — the regression that motivated #2482 (a fabricated pr_merged).
    session = Session(
        id=SessionId("codex-session:prose-only"),
        origin=Origin.CODEX_SESSION,
        title="Prose only",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m",
                    role=Role.ASSISTANT,
                    text=(
                        "MERGED #123\n"
                        "Created pull request #456\n"
                        "Closed issue #789\n"
                        "Review posted on PR #456\n"
                        "20 passed\ndeployment-smoke ... FAILED"
                    ),
                )
            ]
        ),
    )

    digest = compile_recovery_digest(session)
    assert digest.events == ()


def test_continue_report_renders_successor_boot_packet_with_evidence_refs() -> None:
    digest = compile_recovery_digest(_session())

    report = digest.report_markdown("continue")

    assert report.startswith("# Continue: Ship the backlog [evidence: codex-session:demo]")
    assert "## Boot Packet" in report
    assert "- goal: burn down the backlog [evidence: codex-session:demo::m1]" in report
    assert "- next: merge PR #1911 [evidence: codex-session:demo::m1]" in report
    # Structured outcomes are facts, not heuristic candidates (#2482).
    assert "## Recent Outcomes (structured tool/test results)" in report
    assert "- test_passed: devtools verify --quick passed (exit 0) [evidence: codex-session:demo::m2::0]" in report
    assert "heuristic" not in report
    assert (
        "Explore [tool_id=tool-2, task_id=task-42, child_session_id=codex-session:child-42] "
        "— Map the transform surface and report caveats. [evidence: codex-session:demo::m3::0"
    ) in report
    assert "Bash [test] (ok) — devtools verify --quick [evidence: codex-session:demo::m2::0" in report
    assert "## Evidence Index" in report
    assert "evidence_id: codex-session:demo::m2::0; raw: block message=m2 block=0" in report
    _assert_report_claim_lines_are_evidence_linked(report)


def test_blame_report_renders_forensic_evidence_report_with_raw_refs() -> None:
    digest = compile_recovery_digest(_session())

    report = render_recovery_report(digest, preset="blame")

    assert report.startswith("# Blame: Ship the backlog [evidence: codex-session:demo]")
    assert "## Forensic Summary" in report
    assert f"- extracted_claims: {digest.forensic_index.claim_count} [evidence: codex-session:demo]" in report
    assert "## Command And Test Outcomes" in report
    assert "- test_passed: devtools verify --quick passed (exit 0) [evidence: codex-session:demo::m2::0]" in report
    assert "- Bash [test] status=ok lines=3 — devtools verify --quick [evidence: codex-session:demo::m2::0" in report
    assert "output: ruff check ... ok 20 passed in 50.28s" in report
    assert "Explore [tool_id=tool-2, task_id=task-42, child_session_id=codex-session:child-42]:" in report
    assert "## Evidence Timeline" in report
    assert "raw: message message=m1" in report
    # No fabricated GitHub Events section.
    assert "## GitHub Events" not in report
    _assert_report_claim_lines_are_evidence_linked(report)


def _assert_report_claim_lines_are_evidence_linked(report: str) -> None:
    for line in report.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("##"):
            continue
        assert "[evidence: " in line, line


def test_claim_models_reject_missing_raw_refs() -> None:
    with pytest.raises(ValidationError):
        ToolSummary(tool_name="Bash", raw_refs=())
    with pytest.raises(ValidationError):
        SubagentReport(raw_refs=())
    with pytest.raises(ValidationError):
        RunStateSummary(raw_refs=())
    with pytest.raises(ValidationError):
        RecoveryEvent(kind="test_passed", summary="pytest passed (exit 0)", raw_refs=())
    with pytest.raises(ValidationError):
        ForensicIndexEntry(
            evidence_id="session::m1",
            raw_ref=TransformRawRef(session_id="session", message_id="m1"),
            claim_kinds=(),
            claim_labels=("run_state",),
        )


def test_raw_refs_include_message_and_block_preview() -> None:
    digest = compile_recovery_digest(_session())

    tool_ref = digest.tool_summaries[0].raw_refs[0]
    assert tool_ref.message_id == "m2"
    assert tool_ref.block_index == 0
    assert tool_ref.preview == "devtools verify --quick"

    # The structured outcome event points at the tool_use block, not message prose.
    [event] = digest.events
    event_ref = event.raw_refs[0]
    assert event_ref.ref_kind == "block"
    assert event_ref.message_id == "m2"
    assert event_ref.block_index == 0


def test_raw_ref_requires_session_id() -> None:
    with pytest.raises(ValidationError):
        TransformRawRef(session_id="", message_id="m1")


def test_run_projection_refs_round_trip() -> None:
    digest = compile_recovery_digest(_session())

    refs = {
        ref
        for entry in digest.work_packet().entries
        if entry.section == "execution"
        for ref in entry.object_refs
        if ref.kind in {"run", "context-snapshot", "observed-event"}
    }

    assert refs
    for ref in refs:
        assert ObjectRef.parse(ref.format()) == ref
