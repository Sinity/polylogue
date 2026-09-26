"""Read-family declarations must install the handlers that serve requests."""

from __future__ import annotations

import asyncio
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.daemon.route_families import read_detail, read_query


def test_read_family_routes_are_installed_on_the_production_router() -> None:
    from polylogue.daemon.http import DaemonAPIHandler, _parameterized_get_routes, _static_get_routes
    from polylogue.daemon.route_contracts import DAEMON_ROUTE_DECLARATIONS, route_contract_for_pattern

    expected = read_detail.ROUTES + read_query.ROUTES
    actual = {(route.method, route.path): route for route in DAEMON_ROUTE_DECLARATIONS}
    installed = {cast(Any, route).pattern: route for route in (*_static_get_routes(), *_parameterized_get_routes())}
    assert len(expected) == 22
    assert len(actual) == len(DAEMON_ROUTE_DECLARATIONS)
    for route in expected:
        assert actual[(route.method, route.path)] is route
        bindings = tuple(binding for binding in route.kernel.handlers if binding.surface == "daemon-http")
        assert len(bindings) == 1
        binding = bindings[0]
        assert binding.symbol == cast(Any, installed[route.path]).handler_name
        assert callable(getattr(DaemonAPIHandler, binding.symbol))
        assert route_contract_for_pattern(route.method, route.path).kind == route.kind
        assert route_contract_for_pattern(route.method, route.path).stability == route.stability


def test_query_completion_adapter_uses_the_shared_completion_product(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.archive.query import completions

    calls: list[tuple[str, str, str | None, str | None]] = []

    def completed(kind: str, *, incomplete: str, unit: str | None, field: str | None) -> dict[str, object]:
        calls.append((kind, incomplete, unit, field))
        return {"sentinel": "canonical-product-result"}

    monkeypatch.setattr(completions, "query_completion_payload", completed)

    class Handler:
        sent: tuple[HTTPStatus, object] | None = None

        def _get_param(self, params: dict[str, list[str]], key: str) -> str | None:
            values = params.get(key)
            return values[0] if values else None

        def _send_json(self, status: HTTPStatus, payload: object) -> None:
            self.sent = (status, payload)

    handler = Handler()
    read_query._handle_query_completions(handler, {"kind": ["field"], "incomplete": ["or"]})
    assert calls == [("field", "or", None, None)]
    assert handler.sent == (HTTPStatus.OK, {"query_completions": {"sentinel": "canonical-product-result"}})


def test_archive_overview_model_uses_one_bounded_recent_page() -> None:
    from polylogue.operations.http_read_models import read_archive_overview

    class Result:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self.rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self.rows

        def fetchone(self) -> tuple[object, ...]:
            return self.rows[0]

    class Connection:
        def execute(self, query: str) -> Result:
            if "GROUP BY origin" in query:
                return Result([("codex-session", 3), ("claude-code", 2)])
            assert query == "SELECT COUNT(*) FROM messages"
            return Result([(11,)])

    class Archive:
        _conn = Connection()
        requested_page: tuple[int, int] | None = None

        def count_sessions(self) -> int:
            return 5

        def list_summaries(self, *, limit: int, offset: int) -> list[Any]:
            self.requested_page = (limit, offset)
            return ["recent-1"]

    archive = Archive()
    payload = read_archive_overview(archive)  # type: ignore[arg-type]
    assert archive.requested_page == (6, 0)
    assert payload.total_sessions == 5
    assert payload.total_messages == 11
    assert payload.origins == {"codex-session": 3, "claude-code": 2}
    assert cast(tuple[object, ...], payload.recent) == ("recent-1",)


def test_pinned_read_models_resolve_session_without_facade(workspace_env: dict[str, Path]) -> None:
    from polylogue.operations.http_read_models import (
        read_session_cost,
        read_session_evidence,
        read_session_raw,
        read_session_topology,
    )
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.storage_records import SessionBuilder

    root = workspace_env["archive_root"]
    builder = (
        SessionBuilder(root / "index.db", "http-product")
        .provider("codex")
        .title("Pinned product read")
        .add_message(text="A bounded reader fixture.")
    )
    builder.save()
    session_id = builder.native_session_id()
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        raw = read_session_raw(archive, session_id)
        cost = read_session_cost(archive, session_id)
        evidence = read_session_evidence(archive, session_id)
        topology = asyncio.run(read_session_topology(archive, session_id, node_limit=1))

    assert raw is not None and raw["id"] == session_id
    assert cost is not None and cost.session_id == session_id
    assert evidence is not None and evidence.session_id == session_id
    assert evidence.tool_calls == 0
    assert topology is not None
    assert [str(node.session_id) for node in topology.nodes] == [session_id]
