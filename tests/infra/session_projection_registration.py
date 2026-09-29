"""Fresh-interpreter probe: one declaration must reach real CLI and MCP routes."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast, get_args, get_origin, get_type_hints
from unittest.mock import AsyncMock, patch

NAME = "fixture-projection"
MODULE = "polylogue.core.session_projections"


class _ProjectionLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module: ModuleType) -> None:
        source = self.get_source(self.name)
        assert source is not None
        anchor = "for projection in (\n"
        assert source.count(anchor) == 1
        source = source.replace(
            anchor,
            anchor
            + '        SessionListProjection("fixture-projection", "get_session_events", "events", cli_handler="events"),\n',
        )
        exec(compile(source, self.path, "exec"), module.__dict__)


class _ProjectionFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: object = None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, cast(Any, path), target)
        assert spec is not None and spec.origin is not None
        spec.loader = _ProjectionLoader(fullname, spec.origin)
        return spec


def _literal_values(annotation: object) -> set[str]:
    if get_origin(annotation) is Literal:
        return {value for value in get_args(annotation) if isinstance(value, str)}
    return {value for argument in get_args(annotation) for value in _literal_values(argument)}


def main(root: Path) -> None:
    assert MODULE not in sys.modules
    finder = _ProjectionFinder()
    sys.meta_path.insert(0, finder)
    try:
        from polylogue.core.session_projections import SESSION_LIST_PROJECTIONS
    finally:
        sys.meta_path.remove(finder)
    assert NAME in SESSION_LIST_PROJECTIONS

    import click

    from polylogue.archive.viewport import READ_VIEW_PROFILE_BY_ID
    from polylogue.cli.query_verbs import read_verb
    from polylogue.cli.read_view_handlers import READ_VIEW_HANDLERS, run_read_view
    from polylogue.cli.read_view_registry import READ_VIEW_HANDLER_METADATA
    from polylogue.cli.read_views.base import ReadViewInvocation
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.cli.shared.types import AppEnv
    from polylogue.config import Config
    from polylogue.mcp.server import build_server
    from polylogue.services import RuntimeServices
    from tests.infra.mcp import MCPServerUnderTest, invoke_surface, make_polylogue_mock

    context = click.Context(read_verb)
    read_verb.parse_args(context, ["--view", NAME])
    assert context.params["view"] == NAME
    assert NAME in READ_VIEW_HANDLERS and NAME in READ_VIEW_HANDLER_METADATA
    profile = READ_VIEW_PROFILE_BY_ID[NAME].to_payload()
    assert profile["projection_contract"] == READ_VIEW_PROFILE_BY_ID["events"].to_payload()["projection_contract"]
    config = Config(archive_root=root / "archive", render_root=root / "render", sources=[])
    session_id = "codex-session:registry-probe"
    rows = [{"kind": "fixture-event"}]
    window = {
        "rows": rows,
        "total": 1,
        "returned": 1,
        "limit": 1,
        "offset": 0,
        "next_offset": None,
        "continuation": None,
        "complete": True,
    }
    dispatch_payloads: list[dict[str, object]] = []

    def dispatch(_config: object, request: Any, **_kwargs: object) -> tuple[dict[str, object], object]:
        assert request.operation == "session.read"
        dispatch_payloads.append(dict(request.payload))
        return {"session_id": session_id, "evidence_window": window}, object()

    output = io.StringIO()
    with patch("polylogue.cli.read_dispatch.dispatch_read", side_effect=dispatch), redirect_stdout(output):
        run_read_view(
            AppEnv(),
            RootModeRequest.from_params({"_config": config, "id": session_id}),
            ReadViewInvocation(
                view=NAME, session_id=session_id, output_format="json", destination="stdout", out_path=None
            ),
        )
    cli = json.loads(output.getvalue())
    assert cli["events"] == rows and len(dispatch_payloads) == 1
    assert dispatch_payloads[0]["kind"] == "events"

    server = cast(MCPServerUnderTest, build_server(services=RuntimeServices(config=config)))
    tools = server._tool_manager._tools
    assert NAME in _literal_values(get_type_hints(tools["read"].fn)["view"])
    assert NAME in _literal_values(get_type_hints(tools["get"].fn)["projection"])
    poly = make_polylogue_mock()
    poly.get_session_events = AsyncMock(side_effect=AssertionError("windowed projection answered whole"))
    evidence = AsyncMock(return_value=window)
    poly.read_session_evidence_window = evidence
    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        read = json.loads(invoke_surface(tools["read"].fn, ref="session:" + session_id, view=NAME))
        get = json.loads(invoke_surface(tools["get"].fn, ref="session:" + session_id, projection=NAME))
    assert read["events"] == get["events"] == rows
    assert evidence.await_count == 2
    assert [call.args[1] for call in evidence.await_args_list] == ["events", "events"]
    print(json.dumps({"name": NAME, "cli": cli, "read": read, "get": get, "mcp_schema": True}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
