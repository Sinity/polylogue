"""CLI smoke tests for ``polylogue ops materialize-incident-evidence``.

Exercises the real command against a real seeded archive (``workspace_env``),
not a stubbed operation.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli import cli
from polylogue.insights.work_evidence import WorkEvidenceGraph
from polylogue.paths import active_index_db_path
from polylogue.storage.repository import SessionRepository


def _seed_incident_session(workspace_env: dict[str, Path]) -> str:
    from tests.infra.storage_records import SessionBuilder

    builder = (
        SessionBuilder(active_index_db_path(), "cli-incident-demo")
        .provider("codex")
        .git_branch("feature/cli-incident-demo")
        .title("Ship the incident-graph CLI slice")
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
                    "text": "ok\nhttps://github.com/Sinity/polylogue/pull/5151",
                    "tool_result_exit_code": 0,
                },
            ],
        )
    )
    builder.save()
    return builder.native_session_id()


def test_dry_run_reports_json_summary_without_persisting(workspace_env: dict[str, Path]) -> None:
    session_id = _seed_incident_session(workspace_env)

    result = CliRunner().invoke(
        cli,
        [
            "ops",
            "materialize-incident-evidence",
            "--session-id",
            session_id,
            "--graph-id",
            "incident:cli-demo",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["applied"] is False
    assert payload["session_count"] == 1
    assert payload["run_count"] == 1
    assert payload["mentioned_effect_count"] == 1

    async def _read() -> WorkEvidenceGraph | None:
        async with SessionRepository(db_path=active_index_db_path()) as repository:
            return await repository.get_work_evidence_graph("incident:cli-demo")

    assert run_coroutine_sync(_read()) is None


def test_yes_flag_persists_materialized_graph(workspace_env: dict[str, Path]) -> None:
    session_id = _seed_incident_session(workspace_env)

    result = CliRunner().invoke(
        cli,
        [
            "ops",
            "materialize-incident-evidence",
            "--session-id",
            session_id,
            "--graph-id",
            "incident:cli-apply-demo",
            "--yes",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["applied"] is True

    async def _read() -> WorkEvidenceGraph | None:
        async with SessionRepository(db_path=active_index_db_path()) as repository:
            return await repository.get_work_evidence_graph("incident:cli-apply-demo")

    stored = run_coroutine_sync(_read())
    assert stored is not None
    assert any(node.kind == "effect" for node in stored.nodes)


def test_unknown_session_id_reports_usage_error(workspace_env: dict[str, Path]) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "ops",
            "materialize-incident-evidence",
            "--session-id",
            "codex-session:does-not-exist",
            "--graph-id",
            "incident:cli-missing",
        ],
    )
    assert result.exit_code != 0
    assert "no sessions found" in str(result.output) or (
        result.exception is not None and "no sessions found" in str(result.exception)
    )
