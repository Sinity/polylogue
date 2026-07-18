"""End-to-end Claude Workflow admission through configured source ingestion.

Production dependencies exercised:
- configured source walk and OriginSpec path classification;
- raw source/blob persistence and current ``raw_artifacts`` pointers;
- Claude session/event/authorship parsing;
- source-tier Workflow fact parsing;
- generic work-evidence graph replacement.

Anti-vacuity mutations: removing any Claude OriginSpec artifact rule changes the
224-member inventory; removing event extraction changes the four invocations;
using parent/child topology admits the 38 unrelated transcripts; dropping raw
ObjectRef evidence violates the non-empty evidence assertions; removing a
metadata member must degrade one attempt instead of fabricating a pair.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.insights.claude_workflow_materializer import (
    claude_workflow_materialization_needed,
    claude_workflow_reparse_plan,
    materialize_claude_workflow_archive,
)
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

RUN_ID = "wf_54d4fb2e-841"
ATTEMPT_COUNT = 91
UNRELATED_COUNT = 38


@pytest.mark.asyncio
async def test_configured_claude_workflow_admission_preserves_raw_revisions_and_rebuilds(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mandate fixture traverses the production configured-source route."""

    archive_root = workspace_env["archive_root"]
    claude_root, run_path, first_meta_path = _write_fixture(workspace_env["data_root"] / ".claude")
    monkeypatch.setenv("POLYLOGUE_INGEST_PARSE_WORKERS", "1")

    result = await parse_sources_archive(
        archive_root,
        [Source(name=Provider.CLAUDE_CODE.value, path=claude_root)],
    )
    assert result.parse_failures == 0

    summary = materialize_claude_workflow_archive(archive_root)
    assert summary.current_artifact_count == 224
    assert summary.retained_raw_revision_count == 224
    assert summary.artifact_counts == {
        "adopt_manifest": 1,
        "agent_sidecar_meta": 91,
        "agent_transcript": 129,
        "coordinator_session_stream": 1,
        "workflow_journal": 1,
        "workflow_run_snapshot": 1,
    }
    assert summary.graph_count == summary.run_count == 1
    assert summary.coordinator_invocation_count == 4
    assert summary.call_count == 50
    assert summary.attempt_count == 91
    assert summary.linked_session_count == 91
    assert summary.metadata_sidecar_count == 91
    assert summary.journal_result_count == 65
    assert summary.final_result_count == 1
    assert summary.unresolved_call_count == 1
    assert summary.excluded_session_count == 38
    assert summary.generated_prompt_count == 91
    assert summary.human_prompt_count == 1

    with sqlite3.connect(archive_root / "source.db") as source_conn:
        assert source_conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone()[0] == 224
        assert source_conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 224
    with sqlite3.connect(archive_root / "index.db") as index_conn:
        assert (
            index_conn.execute("SELECT COUNT(*) FROM sessions WHERE origin = 'claude-code-session'").fetchone()[0]
            == 130
        )
        node_counts = dict(
            index_conn.execute(
                """
                SELECT node_kind, COUNT(*)
                FROM work_evidence_nodes
                WHERE graph_id = ?
                GROUP BY node_kind
                """,
                (f"claude-workflow:{RUN_ID}",),
            ).fetchall()
        )
        assert {
            kind: node_counts.get(kind, 0)
            for kind in (
                "artifact",
                "attempt",
                "call",
                "invocation",
                "run",
                "session-segment",
                "structured-result",
            )
        } == {
            "artifact": 94,
            "attempt": 91,
            "call": 50,
            "invocation": 4,
            "run": 1,
            "session-segment": 91,
            "structured-result": 66,
        }
        assert (
            index_conn.execute(
                """
            SELECT COUNT(*)
            FROM work_evidence_nodes
            WHERE graph_id = ? AND evidence_refs_json = '[]'
            """,
                (f"claude-workflow:{RUN_ID}",),
            ).fetchone()[0]
            == 0
        )
        assert (
            index_conn.execute(
                """
            SELECT COUNT(*)
            FROM work_evidence_edges
            WHERE graph_id = ? AND evidence_refs_json = '[]'
            """,
                (f"claude-workflow:{RUN_ID}",),
            ).fetchone()[0]
            == 0
        )
        assert (
            index_conn.execute(
                """
            SELECT COUNT(*)
            FROM work_evidence_nodes
            WHERE graph_id = ? AND (node_ref LIKE '%unrelated%' OR label LIKE '%unrelated%')
            """,
                (f"claude-workflow:{RUN_ID}",),
            ).fetchone()[0]
            == 0
        )
        evidence_values = [
            value
            for (payload,) in index_conn.execute(
                "SELECT evidence_refs_json FROM work_evidence_nodes WHERE graph_id = ?",
                (f"claude-workflow:{RUN_ID}",),
            )
            for value in json.loads(payload)
        ]
        assert any(value.startswith("artifact:raw:") for value in evidence_values)

    plan = claude_workflow_reparse_plan(archive_root)
    assert plan.projector_only_raw_artifact_reads == 94
    assert plan.session_parser_raw_reads_if_authorship_or_event_semantics_change == 130
    assert plan.indexed_session_bindings_reused == 129
    assert plan.graphs_atomically_replaced == 1
    assert plan.retained_raw_revisions_preserved == 224
    assert plan.stale is False

    old_snapshot = summary.corpus_snapshot_ref
    revised_run = _run_snapshot(final_value="final result revision two")
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=json.dumps(revised_run, sort_keys=True).encode(),
            source_path=str(run_path),
            source_index=0,
            acquired_at_ms=2_000_000_000_000,
        )

    assert claude_workflow_materialization_needed(archive_root) is True
    revised = materialize_claude_workflow_archive(archive_root)
    assert revised.current_artifact_count == 224
    assert revised.retained_raw_revision_count == 225
    assert revised.graph_count == 1
    assert revised.final_result_count == 1
    assert revised.corpus_snapshot_ref != old_snapshot
    assert claude_workflow_materialization_needed(archive_root) is False

    with sqlite3.connect(archive_root / "source.db") as source_conn:
        assert (
            source_conn.execute(
                "SELECT COUNT(*) FROM raw_sessions WHERE source_path = ?",
                (str(run_path),),
            ).fetchone()[0]
            == 2
        )
        current_raw = source_conn.execute(
            "SELECT raw_id FROM raw_artifacts WHERE source_path = ?",
            (str(run_path),),
        ).fetchone()[0]
        assert (
            current_raw
            == source_conn.execute(
                """
            SELECT raw_id FROM raw_sessions
            WHERE source_path = ?
            ORDER BY acquired_at_ms DESC, rowid DESC LIMIT 1
            """,
                (str(run_path),),
            ).fetchone()[0]
        )
    with sqlite3.connect(archive_root / "index.db") as index_conn:
        assert (
            index_conn.execute(
                "SELECT COUNT(*) FROM work_evidence_graphs WHERE graph_id LIKE 'claude-workflow:%'"
            ).fetchone()[0]
            == 1
        )
        assert (
            index_conn.execute(
                """
            SELECT COUNT(*) FROM work_evidence_nodes
            WHERE graph_id = ? AND corpus_snapshot_ref != ?
            """,
                (f"claude-workflow:{RUN_ID}", revised.corpus_snapshot_ref),
            ).fetchone()[0]
            == 0
        )

    # Representative source-loss mutation: delete one retained metadata member.
    # The attempt must become explicit coverage debt and its transcript must not
    # be admitted through filename or parent-child inference.
    with sqlite3.connect(archive_root / "source.db") as source_conn:
        source_conn.execute("PRAGMA foreign_keys = ON")
        source_conn.execute(
            "DELETE FROM raw_artifacts WHERE source_path = ?",
            (str(first_meta_path),),
        )
        source_conn.execute(
            "DELETE FROM raw_sessions WHERE source_path = ?",
            (str(first_meta_path),),
        )
        source_conn.commit()
    degraded = materialize_claude_workflow_archive(archive_root)
    assert degraded.metadata_sidecar_count == 90
    assert degraded.linked_session_count == 90
    assert any("missing paired agent metadata sidecar" in gap for gap in degraded.gaps)


def _write_fixture(claude_root: Path) -> tuple[Path, Path, Path]:
    project = claude_root / "projects" / "fixture-project"
    subagents = project / "subagents"
    workflow_dir = subagents / "workflows" / RUN_ID
    workflows = project / "workflows"
    jobs = project / "jobs" / "coordinator-session"
    for directory in (subagents, workflow_dir, workflows, jobs):
        directory.mkdir(parents=True, exist_ok=True)

    coordinator_path = project / "coordinator-session.jsonl"
    coordinator_records: list[dict[str, object]] = [
        {
            "type": "user",
            "uuid": "coordinator-direct-prompt",
            "sessionId": "coordinator-session",
            "timestamp": "2026-07-18T00:00:00Z",
            "message": {"role": "user", "content": "Run the admitted Workflow fixture."},
        }
    ]
    for ordinal in range(4):
        tool_input: dict[str, object] = {
            "runId": RUN_ID,
            "taskId": "task-mandate-01",
            "workflowName": "admission-proof",
            "scriptPath": "/repo/scripts/workflow.py",
            "scriptHash": "sha256:fixture-script",
            "phases": ["plan", "execute", "collect"],
            "labels": ["mandate", "synthetic"],
        }
        if ordinal:
            tool_input["resumeFromRunId"] = RUN_ID
        coordinator_records.append(
            {
                "type": "assistant",
                "uuid": f"coordinator-invocation-{ordinal}",
                "sessionId": "coordinator-session",
                "timestamp": f"2026-07-18T00:0{ordinal + 1}:00Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": f"workflow-tool-{ordinal}",
                            "name": "Workflow",
                            "input": tool_input,
                        }
                    ],
                },
            }
        )
    _write_jsonl(coordinator_path, coordinator_records)

    journal_records: list[dict[str, object]] = []
    first_meta_path: Path | None = None
    for ordinal in range(ATTEMPT_COUNT):
        attempt_id = f"attempt-{ordinal:03d}"
        content_key = f"call-{ordinal % 50:02d}"
        transcript_path = subagents / f"agent-{attempt_id}.jsonl"
        meta_path = subagents / f"agent-{attempt_id}.meta.json"
        first_meta_path = first_meta_path or meta_path
        _write_jsonl(
            transcript_path,
            [
                {
                    "type": "user",
                    "uuid": f"worker-prompt-{ordinal:03d}",
                    "sessionId": f"agent-session-{ordinal:03d}",
                    "timestamp": "2026-07-18T01:00:00Z",
                    "message": {"role": "user", "content": f"Generated task pack for {content_key}."},
                }
            ],
        )
        meta_path.write_text(
            json.dumps(
                {
                    "runId": RUN_ID,
                    "attemptId": attempt_id,
                    "agentId": attempt_id,
                    "contentKey": content_key,
                    "transcriptPath": str(transcript_path),
                    "model": "claude-sonnet-fixture",
                    "status": "completed",
                    "phase": "execute",
                    "progress": 100,
                    "timing": {"startedAt": ordinal * 10, "completedAt": ordinal * 10 + 5},
                    "tokens": {"input": 100 + ordinal, "output": 20 + ordinal},
                    "tools": ["Read", "Write"],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        journal_records.append(
            {
                "runId": RUN_ID,
                "event": "attempt_started",
                "contentKey": content_key,
                "attemptId": attempt_id,
                "agentId": attempt_id,
                "metaPath": str(meta_path),
                "transcriptPath": str(transcript_path),
                "phase": "execute",
                "status": "completed",
                "unresolved": content_key == "call-49",
            }
        )

    for result_ordinal, content_index in enumerate([*range(49), *range(16)]):
        journal_records.append(
            {
                "runId": RUN_ID,
                "event": "result",
                "contentKey": f"call-{content_index:02d}",
                "attemptId": f"attempt-{content_index:03d}",
                "structuredResult": {"ok": True, "ordinal": result_ordinal},
                "status": "completed",
            }
        )
    _write_jsonl(workflow_dir / "journal.jsonl", journal_records)

    for ordinal in range(UNRELATED_COUNT):
        _write_jsonl(
            subagents / f"agent-unrelated-{ordinal:03d}.jsonl",
            [
                {
                    "type": "user",
                    "uuid": f"unrelated-prompt-{ordinal:03d}",
                    "sessionId": f"unrelated-session-{ordinal:03d}",
                    "timestamp": "2026-07-18T02:00:00Z",
                    "message": {"role": "user", "content": "Unrelated coordinator child."},
                }
            ],
        )

    run_path = workflows / f"{RUN_ID}.json"
    run_path.write_text(
        json.dumps(_run_snapshot(final_value="final result revision one"), sort_keys=True),
        encoding="utf-8",
    )
    (jobs / "adopt.json").write_text(
        json.dumps(
            {
                "runId": RUN_ID,
                "resumeFromRunId": RUN_ID,
                "adoptedSessionId": "coordinator-session",
                "entrypoint": "resume",
                "status": "adopted",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    assert first_meta_path is not None
    return claude_root, run_path, first_meta_path


def _run_snapshot(*, final_value: str) -> dict[str, object]:
    return {
        "runId": RUN_ID,
        "taskId": "task-mandate-01",
        "workflowName": "admission-proof",
        "scriptPath": "/repo/scripts/workflow.py",
        "scriptHash": "sha256:fixture-script",
        "phases": ["plan", "execute", "collect"],
        "labels": ["mandate", "synthetic"],
        "status": "completed_with_gap",
        "finalResult": {"status": "complete", "value": final_value},
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
