"""Real CLI coverage for the private-data-free basic-usage demo archive.

These tests keep the captured basic-usage walkthroughs executable without a
second devtools command layer.  The CLI assertions cross the subprocess
boundary through ``tests.infra.cli_subprocess.run_cli``; the MCP assertion
uses the production stdio route.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from tests.infra.cli_subprocess import CliResult, run_cli, setup_isolated_workspace


@pytest.fixture
def seeded_demo_archive(tmp_path: Path) -> Path:
    from polylogue.demo import seed_demo_archive

    archive_root = tmp_path / "archive"
    asyncio.run(seed_demo_archive(archive_root, force=True, with_overlays=True))
    return archive_root


@pytest.fixture
def cli_env(tmp_path: Path, seeded_demo_archive: Path) -> dict[str, str]:
    return {
        "HOME": str(tmp_path / "home"),
        "POLYLOGUE_ARCHIVE_ROOT": str(seeded_demo_archive),
        "POLYLOGUE_DAEMON_URL": "http://127.0.0.1:1",
        "POLYLOGUE_FORCE_PLAIN": "1",
    }


def _run(args: list[str], *, env: dict[str, str]) -> CliResult:
    result = run_cli(args, env=env, timeout=120)
    assert result.success, f"polylogue {' '.join(args)} failed:\n{result.output}"
    return result


def test_find_query_covers_fielded_filter_and_pipeline_aggregate(cli_env: dict[str, str]) -> None:
    fielded = _run(
        [
            "--origin",
            "codex-session",
            "find",
            "sessions where origin:codex-session",
            "then",
            "select",
            "--format",
            "json",
        ],
        env=cli_env,
    )
    rows = json.loads(fielded.stdout)
    assert rows
    assert all(row["origin"] == "codex-session" for row in rows)

    aggregate = _run(["find", "actions where tool:bash | group by origin | count"], env=cli_env)
    assert "count=" in aggregate.output


def test_read_renders_the_seeded_transcript(cli_env: dict[str, str]) -> None:
    result = _run(["find", "id:codex-session:demo-receipts", "then", "read", "--view", "transcript"], env=cli_env)
    assert "codex-session:demo-receipts" in result.output
    assert "## user" in result.output
    assert "## assistant" in result.output


def test_search_spans_multiple_origins(cli_env: dict[str, str]) -> None:
    result = _run(["find", "clock", "then", "select", "--format", "json"], env=cli_env)
    rows = json.loads(result.stdout)
    assert rows
    assert len({row["origin"] for row in rows}) >= 2


@pytest.mark.parametrize(
    ("action", "post_verb_args"),
    [
        pytest.param(
            "select",
            ["find", "id:codex-session:demo-receipts", "then", "select"],
            id="select",
        ),
        pytest.param(
            "continue",
            ["find", "id:codex-session:demo-receipts", "then", "continue"],
            id="continue",
        ),
    ],
)
def test_root_json_matches_post_verb_format_json(
    action: str,
    post_verb_args: list[str],
    cli_env: dict[str, str],
) -> None:
    root_json = _run(["--json", *post_verb_args], env=cli_env)
    post_verb_format_json = _run([*post_verb_args, "--format", "json"], env=cli_env)

    assert json.loads(root_json.stdout) == json.loads(post_verb_format_json.stdout), action


def test_continue_generates_a_resume_command(cli_env: dict[str, str]) -> None:
    result = _run(["find", "id:codex-session:demo-receipts", "then", "continue"], env=cli_env)
    assert "resume" in result.output.lower()
    assert "demo-receipts" in result.output


def test_usage_reports_disjoint_token_lanes(cli_env: dict[str, str]) -> None:
    payload = json.loads(_run(["analyze", "usage", "--format", "json"], env=cli_env).stdout)
    lanes = payload["logical_pricing_lanes"]
    assert lanes
    for lane in lanes:
        assert {"input_tokens", "output_tokens", "cached_input_tokens"} <= set(lane["usage"])


def test_lineage_read_composes_parent_prefix_and_child_tail(cli_env: dict[str, str]) -> None:
    fork = _run(
        ["find", "id:codex-session:demo-lineage-fork", "then", "read", "--view", "transcript"], env=cli_env
    ).output
    parent = _run(
        ["find", "id:codex-session:demo-lineage-parent", "then", "read", "--view", "transcript"], env=cli_env
    ).output
    parent_line = "I have the base context and can branch the analysis."
    fork_only_line = "The fork diverges into demo corpus construct checks."
    assert parent_line in fork
    assert fork_only_line in fork
    assert parent_line in parent


def test_mcp_query_and_get_round_trip(seeded_demo_archive: Path) -> None:
    from devtools.continuity_replay import StdioMCPContinuityRoute

    async def _run() -> None:
        async with StdioMCPContinuityRoute(seeded_demo_archive) as route:
            search_payload = json.loads(
                await route.invoke("query", {"expression": "clock", "projection": "sessions", "limit": 3})
            )
            hits = search_payload["hits"]
            assert hits
            session_id = hits[0]["session"]["id"]

            summary_payload = json.loads(await route.invoke("get", {"ref": f"session:{session_id}"}))
            assert summary_payload["payload"]["id"] == session_id

    asyncio.run(_run())


def test_status_reports_direct_archive_fallback_when_daemon_is_unreachable(cli_env: dict[str, str]) -> None:
    result = _run(["status", "--daemon-url", "http://127.0.0.1:1"], env=cli_env)
    assert "Sessions:" in result.output
    assert "daemon not running" in result.output.lower()


@pytest.mark.parametrize("ledger_state", ["empty", "missing_table"])
def test_status_text_reports_authoritative_convergence_debt_state(tmp_path: Path, ledger_state: str) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    if ledger_state == "missing_table":
        with sqlite3.connect(workspace["paths"]["archive_root"] / "ops.db") as conn:
            conn.execute("DROP TABLE convergence_debt")
            conn.commit()

    result = _run(["--plain", "ops", "status"], env=workspace["env"])
    output = result.output.lower()

    assert "convergence debt:" in output
    if ledger_state == "empty":
        assert "none (ledger healthy)" in output
    else:
        assert "unavailable" in output
        assert "convergence debt table is unavailable" in output
