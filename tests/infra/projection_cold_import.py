"""Cold-process probe for one-row CLI/MCP session projection registration."""

from __future__ import annotations

import asyncio
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints
from unittest.mock import AsyncMock, MagicMock, patch


def _literal_values(annotation: object) -> set[str]:
    if get_origin(annotation) is Literal:
        return {value for value in get_args(annotation) if isinstance(value, str)}
    return {value for argument in get_args(annotation) for value in _literal_values(argument)}


def main(root: Path) -> dict[str, int]:
    from polylogue.core.session_projections import SESSION_LIST_PROJECTIONS, SessionListProjection

    # This is a source declaration before either consumer imports, not a
    # monkeypatch of already-generated handler/profile tables.
    assert "polylogue.cli.read_view_handlers" not in sys.modules
    assert "polylogue.mcp.server_cutover" not in sys.modules
    projection = SessionListProjection("projection-fixture", "get_session_events", "events", "events")
    SESSION_LIST_PROJECTIONS[projection.name] = projection

    import click

    from polylogue.archive.viewport import READ_VIEW_PROFILE_BY_ID
    from polylogue.cli.query_verbs import read_verb
    from polylogue.cli.read_view_handlers import READ_VIEW_HANDLERS, run_read_view
    from polylogue.cli.read_view_registry import READ_VIEW_HANDLER_METADATA
    from polylogue.cli.read_views.base import ReadViewInvocation
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.config import Config
    from polylogue.mcp.server import build_server
    from polylogue.surfaces.projection_spec import EvidenceFamily, projection_from_view
    from tests.infra.mcp import invoke_surface_async, make_polylogue_mock

    assert projection.name in READ_VIEW_PROFILE_BY_ID
    assert projection.name in READ_VIEW_HANDLER_METADATA
    assert projection.name in READ_VIEW_HANDLERS
    assert projection_from_view(projection.name).projection.families == (EvidenceFamily.EVENTS,)
    context = click.Context(read_verb)
    read_verb.parse_args(context, ["--view", projection.name])
    assert context.params["view"] == projection.name

    rows = [{"kind": "synthetic-event"}]
    window: dict[str, Any] = {
        "rows": rows,
        "total": 2,
        "returned": 1,
        "limit": 1,
        "offset": 0,
        "next_offset": 1,
        "continuation": "synthetic-continuation",
        "complete": False,
    }
    session_id = "codex-session:projection-registry"
    config = Config(archive_root=root, db_path=root / "index.db", render_root=root / "render", sources=[])
    env = MagicMock(config=config, debug_timing=False)
    request = RootModeRequest.from_params({"_config": config, "id": session_id})
    invocation = ReadViewInvocation(
        view=projection.name,
        session_id=session_id,
        output_format="json",
        destination="stdout",
        out_path=None,
    )
    output = io.StringIO()
    with (
        patch(
            "polylogue.cli.read_dispatch.dispatch_read",
            return_value=({"session_id": session_id, "evidence_window": window}, None),
        ) as transport,
        redirect_stdout(output),
    ):
        run_read_view(env, request, invocation)
    assert transport.call_count == 1
    assert transport.call_args.args[1].operation == "session.read"
    cli = json.loads(output.getvalue())
    assert cli[projection.payload_key] == rows
    assert cli["complete"] is False

    server = build_server()
    read = server._tool_manager._tools["read"].fn
    get = server._tool_manager._tools["get"].fn
    assert projection.name in _literal_values(get_type_hints(read)["view"])
    poly = make_polylogue_mock()
    poly.read_session_evidence_window = AsyncMock(return_value=window)
    setattr(poly, projection.method, AsyncMock(side_effect=AssertionError("whole-list route bypassed paging")))

    async def exercise() -> tuple[dict[str, Any], dict[str, Any]]:
        with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
            read_body = json.loads(
                await invoke_surface_async(read, ref=f"session:{session_id}", view=projection.name, limit=1)
            )
            get_body = json.loads(
                await invoke_surface_async(get, ref=f"session:{session_id}", projection=projection.name)
            )
        return read_body, get_body

    read_body, get_body = asyncio.run(exercise())
    assert read_body[projection.payload_key] == get_body[projection.payload_key] == rows
    assert read_body["complete"] is get_body["complete"] is False
    assert poly.read_session_evidence_window.await_count == 2
    assert all(call.args[1] == projection.cli_handler for call in poly.read_session_evidence_window.await_args_list)
    return {
        "cli_rows": len(cli[projection.payload_key]),
        "mcp_read_rows": len(read_body[projection.payload_key]),
        "mcp_get_rows": len(get_body[projection.payload_key]),
        "window_calls": poly.read_session_evidence_window.await_count,
    }


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), sort_keys=True))
