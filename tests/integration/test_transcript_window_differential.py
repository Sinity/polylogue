"""One transcript window, four public surfaces, one answer (polylogue-vclez).

The session-summary/list request already has an oracle-backed four-surface
differential (``tests/infra/surface_differential.py``).  The *transcript
window* — "give me messages ``[offset, offset + limit)`` of this session" —
did not, and that is the request where the surfaces had actually drifted: the
MCP ``read`` tool hard-coded ``offset=0`` and had no ``offset`` parameter at
all, so it could not express any window but the first page while the Python
API, the CLI ``read --view messages`` verb and the HTTP
``/api/sessions/:id/messages`` route all could.

This module pins the window as a cross-surface fact: the same
``(ref, limit, offset)`` must name the same messages, in the same order, with
the same reported total, on all four routes.

Anti-vacuity — what turns these red:

* ``test_transcript_window_is_identical_across_api_cli_mcp_http`` — drop the
  ``offset`` plumbing from any one surface (restore MCP's ``offset=0``, stop
  the CLI mapping ``--offset`` onto ``body_offset``, ignore HTTP's ``?offset=``)
  and that surface answers page one while the other three answer page two, so
  the tuple comparison fails.  Reporting a total for the page instead of the
  session fails the total comparison for the same reason.
* ``test_transcript_windows_tile_the_session_without_overlap_or_gap`` — an
  off-by-one in any surface's window arithmetic (``[offset, offset+limit]``,
  or an offset applied after the limit) makes the two halves overlap or skip,
  which the concatenation check catches even when each page looks plausible
  alone.
* ``test_mcp_messages_continuation_resumes_the_window`` — remove the
  ``message-offset:`` continuation branch and MCP refuses with
  ``invalid_continuation`` instead of returning the second page.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch
from urllib.request import Request, urlopen

import pytest
from click.testing import CliRunner

from polylogue import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.live_ingest import write_index_session
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async

pytestmark = pytest.mark.xdist_group("web-reader")

#: Enough messages that a window of two is neither the whole session nor its
#: last page, so an off-by-one cannot hide at either boundary.
_MESSAGE_COUNT = 6


def _seed_session(archive_root: Path) -> str:
    """Write one multi-message session through the production writer."""

    with ArchiveStore(archive_root) as archive_db:
        return write_index_session(
            archive_db,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="transcript-window",
                title="Transcript window parity",
                messages=[
                    ParsedMessage(
                        provider_message_id=f"message-{position}",
                        role=Role.USER if position % 2 == 0 else Role.ASSISTANT,
                        text=f"window body {position}",
                        timestamp=f"2026-01-01T00:00:{position:02d}Z",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"window body {position}")],
                    )
                    for position in range(_MESSAGE_COUNT)
                ],
            ),
        )


@contextmanager
def _running_http_server() -> Iterator[str]:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    server.auth_token = ""
    server.api_host = "127.0.0.1"
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name="transcript-window-test", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _get_json(base_url: str, path: str) -> dict[str, Any]:
    with urlopen(Request(f"{base_url}{path}"), timeout=10) as resp:
        assert resp.status == HTTPStatus.OK, f"unexpected status {resp.status} for {path}"
        return cast(dict[str, Any], json.loads(resp.read()))


def _row_ids(payload: dict[str, Any]) -> tuple[str, ...]:
    rows = payload.get("messages")
    assert isinstance(rows, list), f"no message rows in envelope: {sorted(payload)}"
    return tuple(str(row["id"]) for row in rows)


@pytest.fixture
def mcp_server() -> MCPServerUnderTest:
    from polylogue.mcp.server import build_server

    return cast(MCPServerUnderTest, build_server())


async def _api_window(archive_root: Path, session_id: str, *, limit: int, offset: int) -> tuple[tuple[str, ...], int]:
    archive = Polylogue(archive_root=archive_root)
    try:
        messages, total, _completeness = await archive.get_messages_paginated(session_id, limit=limit, offset=offset)
    finally:
        await archive.close()
    return tuple(str(message.id) for message in messages), total


def _cli_window(session_id: str, *, limit: int, offset: int) -> tuple[tuple[str, ...], int]:
    from polylogue.cli.click_app import cli

    result = CliRunner().invoke(
        cli,
        [
            "read",
            f"session:{session_id}",
            "--view",
            "messages",
            "--limit",
            str(limit),
            "--offset",
            str(offset),
            "--format",
            "json",
        ],
        catch_exceptions=True,
    )
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    assert result.exit_code == 0, f"CLI read failed ({result.exit_code}): {result.output}"
    payload = json.loads(result.output)
    return _row_ids(payload), int(payload["total"])


async def _mcp_window(
    mcp_server: MCPServerUnderTest,
    archive_root: Path,
    session_id: str,
    *,
    limit: int,
    offset: int,
) -> tuple[tuple[str, ...], int]:
    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["read"].fn,
            ref=f"session:{session_id}",
            view="messages",
            limit=limit,
            offset=offset,
        )
    payload = json.loads(raw)
    assert "error" not in payload, payload
    return _row_ids(payload), int(payload["total"])


def _http_window(base_url: str, session_id: str, *, limit: int, offset: int) -> tuple[tuple[str, ...], int]:
    payload = _get_json(base_url, f"/api/sessions/{session_id}/messages?limit={limit}&offset={offset}")
    return _row_ids(payload), int(payload["total"])


async def test_transcript_window_is_identical_across_api_cli_mcp_http(
    mcp_server: MCPServerUnderTest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same ``(ref, limit, offset)`` names the same messages on every surface."""

    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    session_id = _seed_session(archive_root)

    limit, offset = 2, 2
    api_ids, api_total = await _api_window(archive_root, session_id, limit=limit, offset=offset)
    cli_ids, cli_total = _cli_window(session_id, limit=limit, offset=offset)
    mcp_ids, mcp_total = await _mcp_window(mcp_server, archive_root, session_id, limit=limit, offset=offset)
    with _running_http_server() as base_url:
        http_ids, http_total = _http_window(base_url, session_id, limit=limit, offset=offset)

    assert len(api_ids) == limit, f"the window itself is wrong before parity matters: {api_ids}"
    assert api_ids == cli_ids == mcp_ids == http_ids
    assert api_total == cli_total == mcp_total == http_total == _MESSAGE_COUNT


async def test_transcript_windows_tile_the_session_without_overlap_or_gap(
    mcp_server: MCPServerUnderTest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consecutive windows concatenate into the whole transcript, on every surface."""

    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    session_id = _seed_session(archive_root)

    half = _MESSAGE_COUNT // 2
    whole, _total = await _api_window(archive_root, session_id, limit=_MESSAGE_COUNT, offset=0)
    assert len(whole) == _MESSAGE_COUNT

    first_mcp, _ = await _mcp_window(mcp_server, archive_root, session_id, limit=half, offset=0)
    second_mcp, _ = await _mcp_window(mcp_server, archive_root, session_id, limit=half, offset=half)
    assert first_mcp + second_mcp == whole

    first_cli, _ = _cli_window(session_id, limit=half, offset=0)
    second_cli, _ = _cli_window(session_id, limit=half, offset=half)
    assert first_cli + second_cli == whole

    with _running_http_server() as base_url:
        first_http, _ = _http_window(base_url, session_id, limit=half, offset=0)
        second_http, _ = _http_window(base_url, session_id, limit=half, offset=half)
    assert first_http + second_http == whole


async def test_mcp_messages_continuation_resumes_the_window(
    mcp_server: MCPServerUnderTest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP continues a transcript window by the shared decimal-offset vocabulary."""

    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    session_id = _seed_session(archive_root)

    half = _MESSAGE_COUNT // 2
    first, total = await _mcp_window(mcp_server, archive_root, session_id, limit=half, offset=0)
    assert total == _MESSAGE_COUNT

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["read"].fn,
            ref=f"session:{session_id}",
            view="messages",
            limit=half,
            continuation=f"message-offset:{half}",
        )
        bad = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["read"].fn,
                ref=f"session:{session_id}",
                view="messages",
                continuation="message-offset:not-a-number",
            )
        )

    resumed = json.loads(raw)
    assert "error" not in resumed, resumed
    assert int(resumed["offset"]) == half
    assert _row_ids(resumed) != first
    # A malformed continuation is refused typed, never silently read as page one.
    assert json.dumps(bad).find("invalid_continuation") >= 0, bad
