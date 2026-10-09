"""CLI tests for ``polylogue ops materialize-incident-evidence``.

Exercises the real command against a real seeded archive (``workspace_env``),
not a stubbed operation. Building the graph is a read the CLI does itself;
``--yes`` persists it through the resident daemon's declared
``mutation.work_evidence.graph.replace`` operation, so the persisting test runs
a real daemon stack and the refusal test proves nothing is written without one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.analysis.work_evidence import WorkEvidenceGraph
from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli import cli
from polylogue.paths import archive_root
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.repository import SessionRepository
from tests.infra.daemon_operations import cli_daemon_archive


def _seed_incident_session(workspace_env: dict[str, Path]) -> str:
    from tests.infra.storage_records import SessionBuilder

    builder = (
        SessionBuilder(resolve_active_index_path(archive_root()), "cli-incident-demo")
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
    """Building the graph is a read, so it opens no writable archive tier.

    A writable open is refused outright beside a resident daemon, so a read
    that opened one could not run at all while ``polylogued`` serves the
    archive. Anti-vacuity: reading session tags through the backend's writable
    ``connection()`` would make ``opened`` nonempty.
    """
    from polylogue.maintenance.offline_guard import refuse_writable_tier_opens

    session_id = _seed_incident_session(workspace_env)

    opened: list[Path] = []
    with refuse_writable_tier_opens(opened.append):
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
    assert opened == []
    payload = json.loads(result.output)
    assert payload["applied"] is False
    assert "replacement" not in payload
    assert payload["session_count"] == 1
    assert payload["run_count"] == 1
    assert payload["mentioned_effect_count"] == 1

    async def _read() -> WorkEvidenceGraph | None:
        async with SessionRepository(db_path=resolve_active_index_path(archive_root())) as repository:
            return await repository.get_work_evidence_graph("incident:cli-demo")

    assert run_coroutine_sync(_read()) is None


def _stored(graph_id: str) -> WorkEvidenceGraph | None:
    async def _read() -> WorkEvidenceGraph | None:
        async with SessionRepository(db_path=resolve_active_index_path(archive_root())) as repository:
            return await repository.get_work_evidence_graph(graph_id)

    return run_coroutine_sync(_read())


def _apply_args(session_id: str, graph_id: str) -> list[str]:
    return [
        "ops",
        "materialize-incident-evidence",
        "--session-id",
        session_id,
        "--graph-id",
        graph_id,
        "--yes",
        "--format",
        "json",
    ]


def test_yes_flag_persists_materialized_graph_through_the_daemon(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = _seed_incident_session(workspace_env)

    with cli_daemon_archive(archive_root(), monkeypatch):
        result = CliRunner().invoke(cli, _apply_args(session_id, "incident:cli-apply-demo"), catch_exceptions=False)
        repeated = CliRunner().invoke(cli, _apply_args(session_id, "incident:cli-apply-demo"), catch_exceptions=False)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["applied"] is True
    assert payload["replacement"]["changed"] is True
    assert payload["replacement"]["previous_digest"] == "absent"
    # The same sessions build the same graph, so a second apply is a no-op
    # rather than a rewrite.
    assert repeated.exit_code == 0, repeated.output
    assert json.loads(repeated.output)["replacement"]["changed"] is False

    stored = _stored("incident:cli-apply-demo")
    assert stored is not None
    assert any(node.kind == "effect" for node in stored.nodes)
    assert payload["replacement"]["digest"] == json.loads(repeated.output)["replacement"]["digest"]


def test_yes_flag_refuses_without_a_daemon_and_writes_nothing(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: restore the in-process ``replace_work_evidence_graph``
    write and this exits 0 with the graph stored."""
    session_id = _seed_incident_session(workspace_env)
    monkeypatch.setenv("POLYLOGUE_NO_DAEMON", "1")

    result = CliRunner().invoke(cli, _apply_args(session_id, "incident:cli-no-daemon"))

    assert result.exit_code != 0, result.output
    assert "polylogued run" in f"{result.output}{result.exception}"
    assert _stored("incident:cli-no-daemon") is None


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
