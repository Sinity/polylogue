"""Production incident-evidence materialization against a real SQLite archive.

Anti-vacuity: seeds one real session (via ``SessionBuilder``'s production
``write_parsed_session_to_archive`` write path) containing a real tool call
that mentions a GitHub PR and a real ``Task`` subagent invocation with a
self-reported result, then drives the actual repository route this module
uses: ``SessionRepository.get_many`` -> ``compile_session_run_projection`` ->
``materialize_incident_evidence_graph`` -> ``replace_work_evidence_graph`` ->
``get_work_evidence_graph``. Removing the ``compile_session_run_projection``
call (falling back to the thinner ``query_runs``/``query_observed_events``
read model) makes the claim/effect assertions below fail, because that read
model never emits subagent self-reports or mentioned commit/PR/issue refs
(see the module's own docstring for the source citation).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.operations.incident_evidence_materialization import (
    NoIncidentSessionsFoundError,
    materialize_incident_work_evidence,
)
from polylogue.storage.repository import SessionRepository


async def _seed_incident_session(db_path: Path) -> str:
    from tests.infra.storage_records import SessionBuilder

    builder = (
        SessionBuilder(db_path, "incident-op-demo")
        .provider("codex")
        .git_branch("feature/incident-op-demo")
        .title("Ship the incident-graph slice")
        .add_message(
            "m-tool",
            role="assistant",
            text="Ran verify and opened the tracking PR.",
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
                    "text": "ok\nhttps://github.com/Sinity/polylogue/pull/4343",
                    "tool_result_exit_code": 0,
                },
            ],
        )
        .add_message(
            "m-task",
            role="assistant",
            text="Dispatching a subagent to investigate the archive.",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "tool-2",
                    "name": "Task",
                    "tool_input": {
                        "subagent_type": "Explore",
                        "taskId": "task-incident-op-1",
                        "child_session_id": "codex-session:incident-op-child-1",
                        "prompt": "Investigate the incident evidence gap.",
                    },
                },
                {
                    "type": "tool_result",
                    "tool_id": "tool-2",
                    "text": "Subagent done: confirmed the gap and filed a follow-up.",
                },
            ],
        )
    )
    builder.save()
    return builder.native_session_id()


@pytest.mark.asyncio
async def test_materialize_incident_work_evidence_round_trips_real_archive_content(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    session_id = await _seed_incident_session(db_path)

    async with SessionRepository(db_path=db_path) as repository:
        result = await materialize_incident_work_evidence(
            repository,
            session_ids=[session_id],
            graph_id="incident:op-demo",
            apply=True,
        )

        assert result.applied is True
        assert result.summary.session_count == 1
        assert result.summary.run_count == 2
        assert result.summary.claim_count == 1
        assert result.summary.mentioned_effect_count == 1

        stored = await repository.get_work_evidence_graph("incident:op-demo")

    assert stored is not None
    assert {node.ref.object_id for node in stored.nodes if node.kind == "run"} == {
        session_id,
        f"{session_id}:subagent:0:tool-2",
    }
    claim = next(node for node in stored.nodes if node.kind == "claim")
    assert claim.claim_text is not None
    assert "confirmed the gap" in claim.claim_text
    effect = next(node for node in stored.nodes if node.kind == "effect")
    assert effect.ref.kind == "github-pr"
    assert effect.ref.object_id == "#4343"
    assert effect.association_state == "unresolved"


@pytest.mark.asyncio
async def test_materialize_incident_work_evidence_dry_run_does_not_persist(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    session_id = await _seed_incident_session(db_path)

    async with SessionRepository(db_path=db_path) as repository:
        result = await materialize_incident_work_evidence(
            repository,
            session_ids=[session_id],
            graph_id="incident:dry-run-demo",
            apply=False,
        )
        assert result.applied is False
        stored = await repository.get_work_evidence_graph("incident:dry-run-demo")

    assert stored is None


@pytest.mark.asyncio
async def test_materialize_incident_work_evidence_rejects_empty_selection(tmp_path: Path) -> None:
    async with SessionRepository(db_path=tmp_path / "index.db") as repository:
        with pytest.raises(NoIncidentSessionsFoundError):
            await materialize_incident_work_evidence(repository, session_ids=[], graph_id="incident:empty")


@pytest.mark.asyncio
async def test_materialize_incident_work_evidence_rejects_unknown_session_ids(tmp_path: Path) -> None:
    async with SessionRepository(db_path=tmp_path / "index.db") as repository:
        with pytest.raises(NoIncidentSessionsFoundError):
            await materialize_incident_work_evidence(
                repository,
                session_ids=["codex-session:does-not-exist"],
                graph_id="incident:unknown",
            )
