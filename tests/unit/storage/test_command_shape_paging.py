from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner, Result

from polylogue.analysis.command_shapes import CommandShapeUsage, CommandShapeUsageQuery, build_command_shape_usage
from polylogue.api import Polylogue
from polylogue.cli.click_app import cli
from polylogue.mcp.server import build_server
from polylogue.services import RuntimeServices
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.daemon_operations import cli_daemon_archive
from tests.infra.json_contracts import extract_json_result
from tests.infra.mcp import MCPServerUnderTest, installed_runtime_services, invoke_surface_async
from tests.infra.storage_records import seed_command_shape_archive


@pytest.mark.asyncio
async def test_command_shapes_repository_projection_and_paging_reach_api_cli_mcp(
    cli_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polylogue.storage.sqlite.archive_tiers.read_insights as readers

    original_fold = build_command_shape_usage
    streamed_reads = 0

    def require_stream(
        rows: Iterable[Mapping[str, object]],
        request: CommandShapeUsageQuery,
        *,
        materialized_at: str,
        checkpoint: Callable[[], None],
    ) -> list[CommandShapeUsage]:
        nonlocal streamed_reads
        assert iter(rows) is rows
        streamed_reads += 1
        return original_fold(rows, request, materialized_at=materialized_at, checkpoint=checkpoint)

    monkeypatch.setattr(readers, "build_command_shape_usage", require_stream)
    root = cli_workspace["archive_root"]
    # Seeding takes the archive writer's synchronous lease, off this event loop.
    session_id = run_off_event_loop(lambda: seed_command_shape_archive(root))
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        archive.begin_read_snapshot()
        try:
            unfiltered = archive.list_command_shape_usage(CommandShapeUsageQuery(limit=None))
            scoped = archive.list_command_shape_usage(CommandShapeUsageQuery(repository="B", limit=1))
            assert len(unfiltered) == 2
            assert {item.repository for item in unfiltered} == {"A"}
            assert [
                (item.command_shape, item.repository, item.execution_count, item.session_count) for item in scoped
            ] == [("foo bar", "B", 2, 1)]
            assert archive._conn.in_transaction
        finally:
            archive.end_read_snapshot()
    async with Polylogue(archive_root=root, db_path=root / "index.db") as poly:
        with installed_runtime_services(root):
            server = cast(MCPServerUnderTest, build_server(services=RuntimeServices(config=poly.config)))
            tool = server._tool_manager._tools["query"].fn
            for repository in ("A", "B"):
                for offset, shape in enumerate(("foo bar", "other status")):
                    request = CommandShapeUsageQuery(repository=repository, limit=1, offset=offset)
                    api_rows = await poly.list_command_shape_usage(request)
                    assert [(row.command_shape, row.repository, row.execution_count) for row in api_rows] == [
                        (shape, repository, 2)
                    ]
                    result = json.loads(
                        cast(
                            str,
                            await invoke_surface_async(
                                tool,
                                projection="command-shapes",
                                repo=repository,
                                limit=1,
                                offset=offset,
                            ),
                        )
                    )
                    assert [
                        (row["command_shape"], row["repository"], row["execution_count"])
                        for row in result["command_shapes"]
                    ] == [(shape, repository, 2)]
            assert await poly.list_command_shape_usage(CommandShapeUsageQuery(repository="absent")) == []

    def run_cli() -> Result:
        # The CLI daemon's bootstrap takes the synchronous lease, which must
        # not block this test's event loop.
        with cli_daemon_archive(root, monkeypatch):
            return CliRunner().invoke(
                cli,
                ["analyze", "insights", "command-shapes", "--repository", "B", "--limit", "1", "--format", "json"],
                catch_exceptions=False,
            )

    cli_result = run_off_event_loop(run_cli)
    assert cli_result.exit_code == 0
    payload = extract_json_result(cli_result.output)
    cli_rows = payload["command_shapes"]
    assert isinstance(cli_rows, list)
    assert isinstance(cli_rows[0], dict)
    assert cli_rows[0]["repository"] == "B"
    assert streamed_reads >= 11
    assert session_id
