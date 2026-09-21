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
* ``test_transcript_window_rank_and_provenance_are_identical_across_surfaces``
  — rank (``position``) and provenance (``identity_source``,
  ``material_origin``, ``message_type``, and the lineage topology fields) are
  read off the domain message by every surface only because they all execute
  the one bound route (``polylogue/operations/transcript_window.py``).  Point
  any one surface back at ``Polylogue.get_messages_paginated`` with its own
  window arithmetic and it can still agree on ids while disagreeing on the
  window's rank origin; this test compares the fields themselves.
* ``test_every_surface_mints_a_snapshot_bound_continuation`` — drop the
  snapshot binding on any one surface (stop threading ``window.continuation``
  into its payload, or resume by re-asking an offset) and that surface reports
  ``continuation is None`` for a window that has a next page, while the other
  three carry a token.  This is the property that was false on three of four
  surfaces before polylogue-ijbwq.
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
from urllib.error import HTTPError
from urllib.parse import quote
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


#: Rank and provenance facts every surface must report identically for the
#: same window, as ``(public wire name, domain attribute)``.  ``position`` is
#: the rank within the composed transcript; the rest name where the row came
#: from and how its identity was decided.  The public spellings are the ones
#: ``MESSAGE_TOPOLOGY_MASK`` declares, which is why a surface cannot quietly
#: rename one.
_RANK_AND_PROVENANCE: tuple[tuple[str, str], ...] = (
    ("position", "position"),
    ("identity_source", "identity_source"),
    ("material_origin", "material_origin"),
    ("message_type", "message_type"),
    ("parent_message_id", "parent_id"),
    ("variant_index", "branch_index"),
    ("is_active_leaf", "is_active_leaf"),
)


def _scalar(value: object) -> object:
    """Normalise enum/str spellings so only real disagreement fails."""

    return None if value is None else str(value)


def _wire_facts(payload: dict[str, Any]) -> tuple[dict[str, object], ...]:
    """Read the rank/provenance facts off one surface's serialized rows."""

    rows = payload.get("messages")
    assert isinstance(rows, list), f"no message rows in envelope: {sorted(payload)}"
    return tuple({name: _scalar(row.get(name)) for name, _ in _RANK_AND_PROVENANCE} for row in rows)


def _domain_facts(messages: object) -> tuple[dict[str, object], ...]:
    """Read the same facts off the Python API's domain rows."""

    return tuple(
        {name: _scalar(getattr(message, attribute, None)) for name, attribute in _RANK_AND_PROVENANCE}
        for message in cast("list[Any]", messages)
    )


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


async def _api_window_payload(archive_root: Path, session_id: str, *, limit: int, offset: int) -> dict[str, Any]:
    """The Python API's own transcript-window entry point, serialized like the rest."""

    archive = Polylogue(archive_root=archive_root)
    try:
        window = await archive.read_transcript_window(session_id, limit=limit, offset=offset)
    finally:
        await archive.close()
    return {
        "messages": list(_domain_facts(window.rows)),
        "total": window.total,
        "limit": window.limit,
        "offset": window.offset,
        "next_offset": window.next_offset,
        "continuation": window.continuation,
    }


def _cli_window_payload(session_id: str, *, limit: int, offset: int) -> dict[str, Any]:
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
    return cast("dict[str, Any]", json.loads(result.output))


async def _mcp_window_payload(
    mcp_server: MCPServerUnderTest,
    archive_root: Path,
    session_id: str,
    *,
    limit: int,
    offset: int,
) -> dict[str, Any]:
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
    payload = cast("dict[str, Any]", json.loads(raw))
    assert "error" not in payload, payload
    return payload


async def test_transcript_window_rank_and_provenance_are_identical_across_surfaces(
    mcp_server: MCPServerUnderTest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same window, same rank and same provenance -- not merely the same ids.

    Anti-vacuity: agreeing on ids only proves the four surfaces selected the
    same rows.  These fields prove they were composed by the same route: point
    one surface back at its own ``get_messages_paginated`` call with its own
    offset arithmetic and it can still list the right ids while reporting a
    ``position`` relative to its page rather than to the transcript.
    """

    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    session_id = _seed_session(archive_root)

    limit, offset = 2, 2
    api = await _api_window_payload(archive_root, session_id, limit=limit, offset=offset)
    cli = _wire_facts(_cli_window_payload(session_id, limit=limit, offset=offset))
    mcp = _wire_facts(await _mcp_window_payload(mcp_server, archive_root, session_id, limit=limit, offset=offset))
    with _running_http_server() as base_url:
        http = _wire_facts(
            _get_json(base_url, f"/api/sessions/{session_id}/messages?limit={limit}&offset={offset}"),
        )

    api_facts = tuple(api["messages"])
    assert len(api_facts) == limit, f"the window itself is wrong before parity matters: {api_facts}"
    # Rank is the transcript position, not the position within the page: the
    # third and fourth messages of a six-message session.
    assert [fact["position"] for fact in api_facts] == ["2", "3"]
    assert api_facts == cli == mcp == http


async def test_every_surface_mints_a_snapshot_bound_continuation(
    mcp_server: MCPServerUnderTest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A window with a next page carries a resumable token on all four surfaces.

    Anti-vacuity: this is exactly the property that was false before
    polylogue-ijbwq -- only the CLI session-document route minted a
    snapshot-bound continuation and the other three resumed by re-asking an
    offset.  Drop the binding on any one surface (stop threading
    ``window.continuation`` into its payload) and that surface reports ``None``
    here while the other three carry a token.  The token is also proved
    *usable*, not merely present, by resuming it on MCP.
    """

    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    session_id = _seed_session(archive_root)

    half = _MESSAGE_COUNT // 2
    api = await _api_window_payload(archive_root, session_id, limit=half, offset=0)
    cli = _cli_window_payload(session_id, limit=half, offset=0)
    mcp = await _mcp_window_payload(mcp_server, archive_root, session_id, limit=half, offset=0)
    with _running_http_server() as base_url:
        http = _get_json(base_url, f"/api/sessions/{session_id}/messages?limit={half}&offset=0")

    for name, payload in (("api", api), ("cli", cli), ("mcp", mcp), ("http", http)):
        assert payload["next_offset"] == half, f"{name} reported next_offset={payload['next_offset']!r}"
        assert isinstance(payload["continuation"], str) and payload["continuation"], (
            f"{name} minted no snapshot-bound continuation for a window that has a next page"
        )

    # One vocabulary: every surface mints a token for the same transaction --
    # same operation, arguments, projection, order, window and bound archive
    # epoch.  (The encoded bytes carry the issuing wall clock, so they are
    # compared decoded rather than byte-for-byte.)
    from polylogue.archive.query.transaction import QueryContinuation

    decoded = {
        name: QueryContinuation.decode(str(payload["continuation"]))
        for name, payload in (("api", api), ("cli", cli), ("mcp", mcp), ("http", http))
    }
    requests = {
        name: (
            token.request.operation,
            json.dumps(dict(token.request.arguments), sort_keys=True, default=str),
            token.request.page_size,
            token.request.offset,
            token.request.projection,
            token.request.stable_order,
            token.request.archive_epoch,
        )
        for name, token in decoded.items()
    }
    assert len(set(requests.values())) == 1, requests
    assert len({token.result_ref for token in decoded.values()}) == 1

    first_ids = _row_ids(mcp)
    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        resumed = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["read"].fn,
                ref=f"session:{session_id}",
                view="messages",
                continuation=str(mcp["continuation"]),
            )
        )
    assert "error" not in resumed, resumed
    assert int(resumed["offset"]) == half
    assert _row_ids(resumed) != first_ids
    assert resumed["continuation"] is None and resumed["next_offset"] is None


# ----------------------------------------------------------------------
# polylogue-idrej — the anchored window ("around this message") on all four
# ----------------------------------------------------------------------


async def _api_anchor_window(
    archive_root: Path, session_id: str, *, limit: int, around: str
) -> tuple[tuple[str, ...], int, int]:
    archive = Polylogue(archive_root=archive_root)
    try:
        window = await archive.read_transcript_window(session_id, limit=limit, around=around)
    finally:
        await archive.close()
    return tuple(str(message.id) for message in window.rows), window.offset, window.total


def _cli_anchor_window(session_id: str, *, limit: int, around: str) -> tuple[tuple[str, ...], int, int]:
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
            "--around",
            around,
            "--format",
            "json",
        ],
        catch_exceptions=True,
    )
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    assert result.exit_code == 0, f"CLI read failed ({result.exit_code}): {result.output}"
    payload = json.loads(result.output)
    return _row_ids(payload), int(payload["offset"]), int(payload["total"])


async def _mcp_anchor_window(
    mcp_server: MCPServerUnderTest,
    archive_root: Path,
    session_id: str,
    *,
    limit: int,
    around: str,
) -> tuple[tuple[str, ...], int, int]:
    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["read"].fn,
            ref=f"session:{session_id}",
            view="messages",
            limit=limit,
            around=around,
        )
    payload = json.loads(raw)
    assert "error" not in payload, payload
    return _row_ids(payload), int(payload["offset"]), int(payload["total"])


def _http_anchor_window(base_url: str, session_id: str, *, limit: int, around: str) -> tuple[tuple[str, ...], int, int]:
    payload = _get_json(base_url, f"/api/sessions/{session_id}/messages?limit={limit}&around={quote(around, safe='')}")
    return _row_ids(payload), int(payload["offset"]), int(payload["total"])


async def test_anchored_window_is_identical_across_api_cli_mcp_http(
    mcp_server: MCPServerUnderTest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """polylogue-idrej: the same ``(ref, message, limit)`` names one window everywhere.

    ``around`` landed as an HTTP query parameter only, so "give me the window
    around this message" was inexpressible on the CLI, the MCP ``read`` tool
    and the Python API -- three surfaces that can otherwise answer the same
    question.  It is sugar over ``offset``, which is exactly why testing only
    the new surfaces could not see a divergence from the HTTP route that
    already worked: this compares all four against each other, and against the
    coordinate each of them reports resolving to.

    Anti-vacuity: drop the anchor plumbing from any one surface and it answers
    page zero (offset 0, ids ``message-0/1``) while the other three answer
    offset 2 -- the tuple comparison and the ``offset == 2`` assertion both
    fail.  Resolve the index against any ordering other than the one the page
    read windows with and the last assertion -- that the anchored window is
    byte-for-byte the window at the offset it reports -- fails while every
    surface still looks like it returned a plausible page.
    """

    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    session_id = _seed_session(archive_root)

    limit = 2
    anchor_index = 3
    whole, total = await _api_window(archive_root, session_id, limit=_MESSAGE_COUNT, offset=0)
    assert total == _MESSAGE_COUNT
    around = whole[anchor_index]
    expected_offset = anchor_index - (anchor_index % limit)
    assert expected_offset not in (0, _MESSAGE_COUNT - limit), (
        "the anchor must sit in neither the first nor the last window, or page-zero passes trivially"
    )

    api = await _api_anchor_window(archive_root, session_id, limit=limit, around=around)
    cli = _cli_anchor_window(session_id, limit=limit, around=around)
    mcp = await _mcp_anchor_window(mcp_server, archive_root, session_id, limit=limit, around=around)
    with _running_http_server() as base_url:
        http = _http_anchor_window(base_url, session_id, limit=limit, around=around)

    assert api == cli == mcp == http, {"api": api, "cli": cli, "mcp": mcp, "http": http}
    anchored_ids, resolved_offset, reported_total = api
    assert around in anchored_ids, "a resolvable reference must land inside the window it named"
    assert resolved_offset == expected_offset
    assert reported_total == _MESSAGE_COUNT

    # The anchor is sugar: the window it resolves to is the window the same
    # caller reaches by asking for the coordinate it reports back.
    by_offset, by_offset_total = await _api_window(archive_root, session_id, limit=limit, offset=resolved_offset)
    assert anchored_ids == by_offset
    assert reported_total == by_offset_total


async def test_anchored_window_is_refused_the_same_way_on_every_surface(
    mcp_server: MCPServerUnderTest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A message this session does not contain is refused, never answered with page zero.

    Anti-vacuity: answer page zero instead and every one of these assertions
    fails -- which is the substitution that silently hands back a *different*
    message's window under the caller's reference.
    """

    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    session_id = _seed_session(archive_root)
    missing = "not-a-message-in-this-session"

    from polylogue.cli.click_app import cli
    from polylogue.operations.message_locator import MessageNotInSessionError

    archive = Polylogue(archive_root=archive_root)
    try:
        with pytest.raises(MessageNotInSessionError) as raised:
            await archive.read_transcript_window(session_id, limit=2, around=missing)
    finally:
        await archive.close()
    assert raised.value.code == "message_not_found"

    result = CliRunner().invoke(
        cli,
        ["read", f"session:{session_id}", "--view", "messages", "--limit", "2", "--around", missing],
        catch_exceptions=True,
    )
    assert result.exit_code != 0, result.output

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        mcp = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["read"].fn,
                ref=f"session:{session_id}",
                view="messages",
                limit=2,
                around=missing,
            )
        )
    assert mcp.get("code") == "message_not_found", mcp

    with _running_http_server() as base_url:
        try:
            with urlopen(
                Request(f"{base_url}/api/sessions/{session_id}/messages?limit=2&around={quote(missing, safe='')}"),
                timeout=10,
            ) as response:
                raise AssertionError(f"expected a refusal, got {response.status}")
        except HTTPError as exc:
            assert exc.code == HTTPStatus.NOT_FOUND
            assert json.loads(exc.read())["error"] == "message_not_found"
