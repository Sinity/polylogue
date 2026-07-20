"""Cross-surface continuation resume parity: HTTP, Python API, MCP (polylogue-z9gh.9.1).

The daemon HTTP ``/api/query-units`` route, the ``Polylogue.query_units()``
facade, and the MCP ``query`` tool all delegate to the same shared primitives
(``query_units_transaction_request``, ``QueryTransaction``,
``QueryContinuation``, ``validate_continuation_epoch`` in
``polylogue/archive/query/transaction.py`` and
``polylogue/archive/query/unit_results.py``). This module proves that
delegation holds end-to-end for a real corpus: paging the same expression
through all three surfaces returns identical rows in identical order, and
resuming a continuation issued before a write lands is rejected identically
by all three, rather than three independently-drifting per-surface
implementations (the exact failure mode the shared-transaction program
guards against, per polylogue-z9gh.9's #2472/#2470 partitioning-bug history).

These tests start a real HTTP server; they share an xdist group with the
other web-reader HTTP lane to avoid cross-worker port/event-loop
interference (tests/unit/daemon/test_web_reader.py).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from polylogue import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.archive.query.transaction import QueryContinuationStaleError
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async

pytestmark = pytest.mark.xdist_group("web-reader")


def _write_needle_message(archive_root: Path, native_id: str, text: str) -> None:
    with ArchiveStore(archive_root) as archive_db:
        archive_db.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=native_id,
                title="Continuation parity session",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                    )
                ],
            )
        )


@contextmanager
def _running_http_server() -> Iterator[str]:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    server.auth_token = ""
    server.api_host = "127.0.0.1"
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name="continuation-parity-test", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _get_json(base_url: str, path: str) -> object:
    with urlopen(Request(f"{base_url}{path}"), timeout=10) as resp:
        assert resp.status == HTTPStatus.OK, f"unexpected status {resp.status} for {path}"
        return json.loads(resp.read())


def _request_json(base_url: str, path: str) -> tuple[int, object]:
    try:
        with urlopen(Request(f"{base_url}{path}"), timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


@pytest.fixture
def mcp_server() -> MCPServerUnderTest:
    from polylogue.mcp.server import build_server

    return cast(MCPServerUnderTest, build_server())


async def test_continuation_pages_match_identically_across_http_api_mcp(
    mcp_server: MCPServerUnderTest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paging the same expression through HTTP/API/MCP yields identical pages.

    Production dependencies exercised: ``DaemonAPIHandler._handle_query_units``,
    ``Polylogue.query_units``, and the MCP ``query`` tool's delegation into
    that same facade method (``server_cutover.py``). Removing the shared
    ``query_units_transaction_request``/``QueryTransaction`` plumbing from any
    one surface would make its rows, ``query_ref``, or ``result_ref`` diverge
    from the other two.
    """
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    for index in range(3):
        _write_needle_message(archive_root, f"parity-{index}", f"paritycheck needle {index}")

    expression = "messages where text:paritycheck"
    archive = Polylogue(archive_root=archive_root)

    # --- API: page 1 then resume to page 2 ---
    api_first = await archive.query_units(expression, limit=2)
    assert api_first.continuation is not None
    api_second = await archive.query_units(continuation=api_first.continuation)
    assert api_second.continuation is None

    # --- HTTP: page 1 then resume to page 2 ---
    quoted_expression = quote(expression)
    with _running_http_server() as base_url:
        http_first = cast(
            dict[str, object], _get_json(base_url, f"/api/query-units?expression={quoted_expression}&limit=2")
        )
        http_second = cast(
            dict[str, object],
            _get_json(base_url, f"/api/query-units?continuation={quote(str(http_first['continuation']), safe='')}"),
        )

    # --- MCP: page 1 then resume to page 2 ---
    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        mcp_first = json.loads(
            await invoke_surface_async(mcp_server._tool_manager._tools["query"].fn, expression=expression, limit=2)
        )
        mcp_second = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["query"].fn, continuation=mcp_first["continuation"]
            )
        )

    def _message_ids(items: object) -> list[str]:
        return [cast(dict[str, object], item)["message_id"] for item in cast(list[object], items)]  # type: ignore[misc]

    api_first_ids = [cast(object, item).message_id for item in api_first.items]  # type: ignore[attr-defined]
    api_second_ids = [cast(object, item).message_id for item in api_second.items]  # type: ignore[attr-defined]
    http_first_ids = _message_ids(http_first["items"])
    http_second_ids = _message_ids(http_second["items"])
    mcp_first_ids = _message_ids(mcp_first["items"])
    mcp_second_ids = _message_ids(mcp_second["items"])

    assert api_first_ids == http_first_ids == mcp_first_ids
    assert api_second_ids == http_second_ids == mcp_second_ids
    assert len(set(api_first_ids) | set(api_second_ids)) == 3

    assert api_first.query_ref == http_first["query_ref"] == mcp_first["query_ref"]
    assert api_first.result_ref == http_first["result_ref"] == mcp_first["result_ref"]
    assert api_second.query_ref == http_second["query_ref"] == mcp_second["query_ref"]
    assert http_second["continuation"] is None
    assert mcp_second["continuation"] is None


async def test_continuation_stale_epoch_rejected_identically_across_http_api_mcp(
    mcp_server: MCPServerUnderTest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A continuation issued before a concurrent write lands is rejected the same way everywhere.

    Production dependency exercised: ``validate_continuation_epoch`` inside
    ``query_unit_envelope`` (``polylogue/archive/query/unit_results.py``) is
    the single call each surface's adapter reaches on resume. Bypassing it on
    any one surface would let a stale offset silently duplicate or skip rows
    across the newly-written session instead of raising
    ``query_continuation_stale``.
    """
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    for index in range(2):
        _write_needle_message(archive_root, f"stale-parity-{index}", "stale paritycheck needle")

    expression = "messages where text:paritycheck"
    archive = Polylogue(archive_root=archive_root)

    api_first = await archive.query_units(expression, limit=1)
    assert api_first.continuation is not None

    quoted_expression = quote(expression)
    with _running_http_server() as base_url:
        http_first = cast(
            dict[str, object], _get_json(base_url, f"/api/query-units?expression={quoted_expression}&limit=1")
        )

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        mcp_first = json.loads(
            await invoke_surface_async(mcp_server._tool_manager._tools["query"].fn, expression=expression, limit=1)
        )

    # A write after every surface issued its first page moves the shared
    # archive epoch; every resume attempt below must now be rejected.
    _write_needle_message(archive_root, "stale-parity-mutation", "stale paritycheck needle")

    with pytest.raises(QueryContinuationStaleError):
        await archive.query_units(continuation=api_first.continuation)

    with _running_http_server() as base_url:
        http_status, http_payload = _request_json(
            base_url, f"/api/query-units?continuation={quote(str(http_first['continuation']), safe='')}"
        )
    assert http_status == HTTPStatus.CONFLICT
    assert cast(dict[str, object], http_payload)["error"] == "query_continuation_stale"

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        mcp_stale = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["query"].fn, continuation=mcp_first["continuation"]
            )
        )
    assert mcp_stale["code"] == "query_continuation_stale"
