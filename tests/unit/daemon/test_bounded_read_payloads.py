"""Read payloads that grow with the requested window, not the session.

Two defects with the same shape, fixed together:

* **polylogue-i5vqc** — a deep link (``#msg-<id>``) was honoured by walking
  pages from the top until the target appeared. Past the walker's ceiling a
  **valid, resolvable** reference reported as unlocatable, and below it every
  preceding message was fetched anyway. The route now resolves the reference
  to the offset of the window holding it (``around=``) and serves that one
  window.
* **polylogue-o0zju** — ``/api/stack`` and ``/api/compare`` serialized every
  message of every referenced session for a view that renders a narrow
  reading window. They now serve a declared window and report it.

Anti-vacuity, per test:

* ``test_deep_link_serves_the_window_holding_a_target_past_the_old_ceiling``
  — the target sits at index 1,700 of a 2,000-message session, past the 50 ×
  30 ceiling the old walk carried. Drop the ``around`` plumbing from
  ``_do_archive_get_messages`` and the route answers offset 0, so the target
  is absent from the window and the offset assertion fails. A test that only
  checked "the route returns messages" would stay green.
* ``test_deep_link_window_matches_the_offset_window_it_resolves_to`` — resolve
  the index against any ordering other than the one the page read windows with
  (for instance ``ORDER BY occurred_at_ms``) and the two windows name
  different messages even though both look like plausible pages.
* ``test_deep_link_refuses_a_reference_this_session_does_not_contain`` —
  answer page zero instead of refusing and the 404 assertion fails; that
  substitution is exactly how a wrong message gets served under a caller's
  reference.
* ``test_stack_serves_only_the_requested_window`` /
  ``test_compare_serves_only_the_requested_window`` — remove the window from
  ``_do_archive_stack``/``_do_archive_compare`` and each item carries all
  2,000 messages instead of 25. The true totals are asserted alongside, so
  "bound it by truncating the reported count" is not a passing fix either.
* ``test_stack_reports_true_totals_for_an_explicit_whole_transcript_request``
  — turn the declared window into a hard ceiling and ``limit=0`` stops
  returning the whole transcript, which is the difference between a declared
  parameter and a cap.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.live_ingest import write_index_session

pytestmark = pytest.mark.xdist_group("web-reader")

#: Long enough that a served window is unmistakably not the whole transcript,
#: and that a target beyond ``50 * 30`` (the ceiling the retired sequential
#: deep-link walk carried) exists at all.
_MESSAGE_COUNT = 2000

#: Deliberately past ``50 * _PAGE`` — this is the index the old walk could not
#: reach and reported as "could not be located within the paged transcript".
_DEEP_INDEX = 1700

_PAGE = 30

#: A tenth of the long session, so a read whose cost tracks the session shows
#: it as a difference rather than as one unexplained number.
_SHORT_MESSAGE_COUNT = 200

#: Three sessions, because the stack payload's cost is per referenced session.
_STACK_NATIVE_IDS = ("stack-a", "stack-b", "stack-c")


def _seed_session(archive_root: Path, native_id: str, *, count: int = _MESSAGE_COUNT) -> str:
    with ArchiveStore(archive_root) as archive:
        return write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=native_id,
                title=f"Bounded read payloads · {native_id}",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T01:00:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id=f"m-{position}",
                        role=Role.USER if position % 2 == 0 else Role.ASSISTANT,
                        text=f"{native_id} body {position}",
                        timestamp="2026-01-01T00:00:00+00:00",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"{native_id} body {position}")],
                    )
                    for position in range(count)
                ],
            ),
        )


@pytest.fixture(scope="module")
def seeded_archive(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One archive holding three long sessions, seeded once for the module."""

    archive_root = tmp_path_factory.mktemp("bounded-read-payloads") / "archive"
    session_ids = [_seed_session(archive_root, native_id) for native_id in _STACK_NATIVE_IDS]
    return {"archive_root": archive_root, "session_ids": session_ids}


@pytest.fixture(scope="module")
def short_archive(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A much shorter session, so "bounded" can be measured against growth."""

    archive_root = tmp_path_factory.mktemp("bounded-read-short") / "archive"
    return {
        "archive_root": archive_root,
        "session_id": _seed_session(archive_root, "short", count=_SHORT_MESSAGE_COUNT),
    }


@pytest.fixture
def reader(seeded_archive: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(seeded_archive["archive_root"]))
    with _running_http_server() as base_url:
        yield {**seeded_archive, "base_url": base_url}


@contextmanager
def _running_http_server() -> Iterator[str]:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    server.auth_token = ""
    server.api_host = "127.0.0.1"
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name="bounded-read-payloads", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _get_json(base_url: str, path: str) -> dict[str, Any]:
    with urlopen(Request(f"{base_url}{path}"), timeout=30) as response:
        assert response.status == HTTPStatus.OK, f"unexpected status {response.status} for {path}"
        return cast("dict[str, Any]", json.loads(response.read()))


def _get_error(base_url: str, path: str) -> tuple[int, dict[str, Any]]:
    try:
        with urlopen(Request(f"{base_url}{path}"), timeout=30) as response:
            raise AssertionError(f"expected a refusal for {path}, got {response.status}")
    except HTTPError as exc:
        return exc.code, cast("dict[str, Any]", json.loads(exc.read()))


def _read_view(base_url: str, session_id: str, query: str) -> dict[str, Any]:
    """Unwrap the ``SessionReadViewEnvelope`` the ``/read`` route answers with."""

    envelope = _get_json(base_url, f"/api/sessions/{quote(session_id, safe='')}/read?view=messages&{query}")
    assert envelope["view"] == "messages"
    return cast("dict[str, Any]", envelope["payload"])


def _message_ids(payload: dict[str, Any]) -> list[str]:
    rows = payload["messages"]
    assert isinstance(rows, list)
    return [str(row["id"]) for row in rows]


def _deep_message_id(session_id: str, archive_root: Path) -> str:
    """The id of the message at ``_DEEP_INDEX`` in composed transcript order."""

    with ArchiveStore(archive_root) as archive:
        envelope = archive.read_session_page(session_id, limit=1, offset=_DEEP_INDEX)
    return str(envelope.messages[0].message_id)


# ----------------------------------------------------------------------
# polylogue-i5vqc — resolve a deep link by reference
# ----------------------------------------------------------------------


def test_deep_link_serves_the_window_holding_a_target_past_the_old_ceiling(reader: dict[str, Any]) -> None:
    session_id = reader["session_ids"][0]
    target = _deep_message_id(session_id, reader["archive_root"])

    payload = _read_view(reader["base_url"], session_id, f"limit={_PAGE}&around={quote(target, safe='')}")

    assert target in _message_ids(payload), "a resolvable reference must land inside the window it named"
    assert payload["offset"] == _DEEP_INDEX - (_DEEP_INDEX % _PAGE)
    assert len(payload["messages"]) == _PAGE, "the deep link costs one window, not the prefix before it"
    assert payload["total"] == _MESSAGE_COUNT, "the window is bounded; the reported total is not"


def test_deep_link_window_matches_the_offset_window_it_resolves_to(reader: dict[str, Any]) -> None:
    session_id = reader["session_ids"][0]
    target = _deep_message_id(session_id, reader["archive_root"])

    around = _read_view(reader["base_url"], session_id, f"limit={_PAGE}&around={quote(target, safe='')}")
    by_offset = _read_view(reader["base_url"], session_id, f"limit={_PAGE}&offset={around['offset']}")

    assert _message_ids(around) == _message_ids(by_offset)


def test_deep_link_is_answered_identically_by_both_message_window_routes(reader: dict[str, Any]) -> None:
    """``/read?view=messages`` and ``/messages`` are one window, two spellings."""

    session_id = reader["session_ids"][0]
    target = _deep_message_id(session_id, reader["archive_root"])
    encoded = quote(session_id, safe="")
    anchor = f"limit={_PAGE}&around={quote(target, safe='')}"

    read_view = _read_view(reader["base_url"], session_id, anchor)
    messages_route = _get_json(reader["base_url"], f"/api/sessions/{encoded}/messages?{anchor}")

    assert _message_ids(read_view) == _message_ids(messages_route)
    assert read_view["offset"] == messages_route["offset"] == _DEEP_INDEX - (_DEEP_INDEX % _PAGE)


def test_deep_link_window_continues_into_the_next_page(reader: dict[str, Any]) -> None:
    """A deep-linked window issues the same continuation coordinates a paged one would."""

    session_id = reader["session_ids"][0]
    target = _deep_message_id(session_id, reader["archive_root"])

    window = _read_view(reader["base_url"], session_id, f"limit={_PAGE}&around={quote(target, safe='')}")
    assert window["next_offset"] == window["offset"] + _PAGE
    following = _read_view(reader["base_url"], session_id, f"limit={_PAGE}&offset={window['next_offset']}")
    assert not set(_message_ids(window)) & set(_message_ids(following)), "windows must tile, not overlap"


def test_deep_link_refuses_a_reference_this_session_does_not_contain(reader: dict[str, Any]) -> None:
    session_id = reader["session_ids"][0]
    encoded = quote(session_id, safe="")

    status, body = _get_error(
        reader["base_url"],
        f"/api/sessions/{encoded}/read?view=messages&limit={_PAGE}&around=not-a-message-in-this-session",
    )

    assert status == HTTPStatus.NOT_FOUND
    assert body["error"] == "message_not_found"


def test_deep_link_refuses_to_be_given_two_different_windows(reader: dict[str, Any]) -> None:
    session_id = reader["session_ids"][0]
    encoded = quote(session_id, safe="")
    first = _read_view(reader["base_url"], session_id, f"limit={_PAGE}&offset=0")
    token = first["continuation"]
    assert token, "the fixture needs a session with a next window for this refusal to be reachable"

    status, body = _get_error(
        reader["base_url"],
        f"/api/sessions/{encoded}/read?view=messages&around=whatever&continuation={quote(str(token), safe='')}",
    )

    assert status == HTTPStatus.BAD_REQUEST
    assert body["error"] == "invalid_request"


def test_message_locator_indexes_against_composed_transcript_order(seeded_archive: dict[str, Any]) -> None:
    """The locator's index is the index the page read windows with."""

    from polylogue.operations.message_locator import (
        MessageNotInSessionError,
        locate_message_in_archive,
        window_offset_for_index,
    )

    session_id = seeded_archive["session_ids"][0]
    with ArchiveStore(seeded_archive["archive_root"]) as archive:
        page = archive.read_session_page(session_id, limit=5, offset=_DEEP_INDEX)
        for step, message in enumerate(page.messages):
            located = locate_message_in_archive(archive, session_id, str(message.message_id))
            assert located.index == _DEEP_INDEX + step
        with pytest.raises(MessageNotInSessionError):
            locate_message_in_archive(archive, session_id, "absent")

    assert window_offset_for_index(_DEEP_INDEX, _PAGE) == 1680
    assert window_offset_for_index(0, _PAGE) == 0


# ----------------------------------------------------------------------
# polylogue-o0zju — bound stack and compare to the requested window
# ----------------------------------------------------------------------


def test_stack_serves_only_the_requested_window(reader: dict[str, Any]) -> None:
    ids = ",".join(quote(session_id, safe="") for session_id in reader["session_ids"])

    payload = _get_json(reader["base_url"], f"/api/stack?ids={ids}&limit=25")

    assert payload["limit"] == 25
    assert payload["offset"] == 0
    assert len(payload["items"]) == len(_STACK_NATIVE_IDS)
    for item in payload["items"]:
        session = item["session"]
        assert len(session["messages"]) == 25, "the stack ships the window the view renders, not the session"
        assert session["total"] == _MESSAGE_COUNT, "the true length is still reported"
        assert session["message_count"] == _MESSAGE_COUNT


def test_stack_window_offset_selects_a_later_slice(reader: dict[str, Any]) -> None:
    ids = ",".join(quote(session_id, safe="") for session_id in reader["session_ids"][:1])

    first = _get_json(reader["base_url"], f"/api/stack?ids={ids}&limit=10&offset=0")
    later = _get_json(reader["base_url"], f"/api/stack?ids={ids}&limit=10&offset=500")

    assert later["offset"] == 500
    first_ids = [str(row["id"]) for row in first["items"][0]["session"]["messages"]]
    later_ids = [str(row["id"]) for row in later["items"][0]["session"]["messages"]]
    assert len(later_ids) == 10
    assert not set(first_ids) & set(later_ids)


def test_stack_reports_true_totals_for_an_explicit_whole_transcript_request(reader: dict[str, Any]) -> None:
    """``limit=0`` is the declared "whole transcript" request, so the default is not a cap."""

    ids = quote(reader["session_ids"][0], safe="")

    payload = _get_json(reader["base_url"], f"/api/stack?ids={ids}&limit=0")

    assert payload["limit"] is None
    session = payload["items"][0]["session"]
    assert len(session["messages"]) == _MESSAGE_COUNT
    assert session["total"] == _MESSAGE_COUNT


def test_compare_serves_only_the_requested_window(reader: dict[str, Any]) -> None:
    left, right = (quote(session_id, safe="") for session_id in reader["session_ids"][:2])

    payload = _get_json(reader["base_url"], f"/api/compare?left={left}&right={right}&align=prompt&limit=25")

    assert payload["limit"] == 25
    assert payload["offset"] == 0
    assert len(payload["pairs"]) == 25, "the diff covers the requested window, not two whole sessions"
    assert len(payload["left"]["messages"]) == 25
    assert len(payload["right"]["messages"]) == 25
    assert payload["left"]["total"] == payload["right"]["total"] == _MESSAGE_COUNT
    diff = payload["metadata_diff"]["message_count"]
    assert diff["left"] == diff["right"] == _MESSAGE_COUNT, "the header diff compares true lengths"


async def test_database_backed_session_payload_serializes_only_the_window(
    seeded_archive: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader the database-backed stack/compare routes call is bounded too.

    ``_do_get_session`` reads no handler state, so it is exercised directly
    rather than through a second HTTP server. Drop the slice and this returns
    2,000 message envelopes for a 25-message window.
    """

    from polylogue import Polylogue
    from polylogue.daemon.http import DaemonAPIHandler

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(seeded_archive["archive_root"]))
    session_id = seeded_archive["session_ids"][0]
    poly = Polylogue(archive_root=seeded_archive["archive_root"])
    try:
        payload = await DaemonAPIHandler._do_get_session(cast("Any", None), poly, session_id, limit=25, offset=100)
    finally:
        await poly.close()

    assert isinstance(payload, dict)
    assert len(payload["messages"]) == 25
    assert payload["total"] == _MESSAGE_COUNT
    assert payload["message_count"] == _MESSAGE_COUNT
    assert str(payload["messages"][0]["id"]).endswith(":m-100")


def _composed_row_counter(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record how many transcript rows each archive composition materialized.

    Rows composed -- not rows returned -- is the measurement these routes
    need: a payload can serve a 25-message window off a 2,000-message
    composition and look identical to one that composed 25. That is exactly
    the difference polylogue-2go3o is about, and a contents-only assertion
    cannot see it.
    """

    composed: list[int] = []

    def _instrument(name: str) -> None:
        original = getattr(ArchiveStore, name)

        def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            envelope = original(self, *args, **kwargs)
            composed.append(len(envelope.messages))
            return envelope

        monkeypatch.setattr(ArchiveStore, name, _wrapped)

    _instrument("read_session")
    _instrument("read_session_page")
    return composed


async def _db_backed(
    archive_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    call: str,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one database-backed reader handler against ``archive_root``.

    These handlers read no handler state, so they are exercised directly
    rather than through a second HTTP server -- and directly is the only way
    to reach them while an archive root the reader routes accept exists.
    """

    from polylogue import Polylogue
    from polylogue.daemon.http import DaemonAPIHandler

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    poly = Polylogue(archive_root=archive_root)
    try:
        handler = getattr(DaemonAPIHandler, call)
        payload = await handler(cast("Any", None), poly, *args, **kwargs)
    finally:
        await poly.close()
    assert isinstance(payload, dict)
    return cast("dict[str, Any]", payload)


async def test_db_backed_session_composes_only_the_window(
    seeded_archive: dict[str, Any],
    short_archive: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composition behind a windowed session payload is the window itself.

    Anti-vacuity: restore ``poly.get_session()`` plus a Python slice and the
    long session composes 2,000 rows to serve 25 -- the payload assertions
    below stay green under that, which is why the composed-row counts are the
    assertion.
    """

    composed = _composed_row_counter(monkeypatch)
    long_payload = await _db_backed(
        seeded_archive["archive_root"],
        monkeypatch,
        "_do_get_session",
        seeded_archive["session_ids"][0],
        limit=25,
        offset=100,
    )
    long_rows = sum(composed)
    composed.clear()
    short_payload = await _db_backed(
        short_archive["archive_root"],
        monkeypatch,
        "_do_get_session",
        short_archive["session_id"],
        limit=25,
        offset=100,
    )
    short_rows = sum(composed)

    assert long_rows == short_rows == 25, "the composition tracks the window, not the session"
    assert len(long_payload["messages"]) == len(short_payload["messages"]) == 25
    assert str(long_payload["messages"][0]["id"]).endswith(":m-100")
    # The window never becomes the reported session size.
    assert long_payload["total"] == long_payload["message_count"] == _MESSAGE_COUNT
    assert short_payload["total"] == short_payload["message_count"] == _SHORT_MESSAGE_COUNT
    assert long_payload["word_count"] > short_payload["word_count"] > 0


async def test_db_backed_window_composes_no_transcript(
    seeded_archive: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The database-backed message window composes no transcript at all.

    Its rows come from the paginated read and its header from the session
    row, so nothing on this route has to compose a transcript -- including
    the deep link, whose locate is answered from indexed counts.

    Anti-vacuity: restore either ``poly.get_session()`` call (the header read
    or the deep-link anchor) and the counter reports 2,000 composed rows for
    a 30-message window.
    """

    session_id = seeded_archive["session_ids"][0]
    target = _deep_message_id(session_id, seeded_archive["archive_root"])
    composed = _composed_row_counter(monkeypatch)

    paged = await _db_backed(seeded_archive["archive_root"], monkeypatch, "_do_get_messages", session_id, _PAGE, 0)
    paged_rows = sum(composed)
    composed.clear()
    deep = await _db_backed(
        seeded_archive["archive_root"], monkeypatch, "_do_get_messages", session_id, _PAGE, 0, None, target
    )
    deep_rows = sum(composed)

    assert paged_rows == deep_rows == 0
    assert len(paged["messages"]) == len(deep["messages"]) == _PAGE
    assert paged["total"] == deep["total"] == _MESSAGE_COUNT
    assert deep["offset"] == _DEEP_INDEX - (_DEEP_INDEX % _PAGE)
    assert target in _message_ids(deep), "a resolvable reference must land inside the window it named"
    assert _message_ids(paged) == [f"{session_id}:n:m-{position}" for position in range(_PAGE)]


def test_workspace_builders_thread_the_window_to_every_referenced_session() -> None:
    """The database-backed stack/compare routes carry the same window."""

    import asyncio

    from polylogue.daemon.workspace_routes import (
        MessageWindow,
        build_compare_payload,
        build_stack_payload,
    )

    seen: list[tuple[str, MessageWindow]] = []

    async def load_session(_poly: object, conv_id: str, window: MessageWindow) -> object:
        seen.append((conv_id, window))
        return {
            "id": conv_id,
            "messages": [
                {"id": f"{conv_id}:m{index}", "role": "user", "text": ""} for index in range(window.limit or 0)
            ],
            "message_count": _MESSAGE_COUNT,
            "total": _MESSAGE_COUNT,
        }

    window = MessageWindow(limit=7, offset=21)
    stack = asyncio.run(build_stack_payload(cast("Any", None), ["a", "b"], None, cast("Any", load_session), window))
    compare = asyncio.run(
        build_compare_payload(cast("Any", None), "a", "b", "prompt", cast("Any", load_session), window)
    )

    assert [conv_id for conv_id, _ in seen] == ["a", "b", "a", "b"]
    assert {carried for _, carried in seen} == {window}
    assert stack["limit"] == 7 and stack["offset"] == 21
    assert compare["limit"] == 7 and compare["offset"] == 21
    items = cast("list[dict[str, Any]]", stack["items"])
    assert len(cast("dict[str, Any]", items[0]["session"])["messages"]) == 7
    assert len(cast("list[Any]", compare["pairs"])) == 7
