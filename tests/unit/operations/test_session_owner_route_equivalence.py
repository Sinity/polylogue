"""The two declared read routes answer the same session question identically.

Polylogue serves indexed session reads through *two* declared executors, not
one:

* ``operations.daemon_reads.execute_read_operation`` -- the generic, dict-keyed
  read operation reached by the CLI, the daemon transport and the HTTP client;
* ``operations.session_reads.execute_session_operation`` -- the typed session
  owner family (``docs/session-operations.md``), reached by the MCP ``query``
  tool and by ``python -m polylogue.cli.session_operations execute``.

They are deliberately separate (the owner family declares per-operation
request/result/error contracts, and serves raw-JSONL reads the generic path has
no unit for), so "CLI/API/MCP parity" cannot mean "one shared chokepoint". What
it must mean is *behavioural equivalence* on the reads both routes claim to
answer: which sessions are selected, in what order, how many exist, and where
the next page starts.

Anti-vacuity: these tests call two different functions. Change a selection,
ordering or paging semantic on either route alone -- flip the default ordering
in ``session_reads.session_query``, drop a filter from one spec lowering, or
let one route decide ``next_offset`` differently -- and the assertions below go
red. A test that drove both surfaces through one function could not do that,
which is exactly why ``tests/unit/operations/test_session_read_parity.py`` (one
executor over two transports) does not discharge this obligation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.core.enums import BlockType, Origin, Provider, Role
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from polylogue.operations.session_contracts import SessionList, SessionRead, SessionSearch
from polylogue.operations.session_reads import execute_session_operation
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.live_ingest import write_index_session


def test_session_list_defaults_to_the_shared_surface_page_size() -> None:
    from polylogue.archive.query.spec import DEFAULT_SESSION_LIST_LIMIT

    assert SessionList().limit == DEFAULT_SESSION_LIST_LIMIT


def _seed(root: Path, count: int = 5, *, native_suffix: str = "") -> list[str]:
    """Seed sessions that differ in date and message count, so order and filters bite."""

    ids: list[str] = []
    with ArchiveStore(root) as archive:
        for index in range(count):
            ids.append(
                write_index_session(
                    archive,
                    ParsedSession(
                        source_name=Provider.CODEX,
                        provider_session_id=f"equivalence-{index}{native_suffix}",
                        title=f"Session {index}",
                        messages=[
                            ParsedMessage(
                                provider_message_id=f"m{message}",
                                role=Role.USER,
                                timestamp=f"2026-02-0{index + 1}T12:0{message}:00Z" if message == 0 else None,
                                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"needle {index} {message}")],
                            )
                            # Session N carries N + 1 messages.
                            for message in range(index + 1)
                        ],
                    ),
                )
            )
    return ids


#: Ordered session ids, reported total, page size, offset, next page offset.
Selection = tuple[list[str], int, int, int, int | None]


def _int(value: object) -> int:
    assert isinstance(value, int)
    return value


def _optional_int(value: object) -> int | None:
    assert value is None or isinstance(value, int)
    return value


async def _owner_list(root: Path, **fields: object) -> Selection:
    async with Polylogue(archive_root=root) as api:
        page = await execute_session_operation(api, SessionList.model_validate(fields))
    return (
        [str(item.id) for item in page.items],
        _int(page.total),
        _int(page.limit),
        _int(page.offset),
        _optional_int(page.next_offset),
    )


def _generic_list(root: Path, **params: object) -> Selection:
    with open_operation_read(root) as pinned:
        body = execute_read_operation(
            "cli.query",
            {"params": dict(params)},
            archive=pinned.archive,
            serving_identity="direct",
        )
    items = body["items"]
    assert isinstance(items, list)
    return (
        [str(row["id"]) for row in items],
        _int(body["total"]),
        _int(body["limit"]),
        _int(body["offset"]),
        _optional_int(body["next_offset"]),
    )


@pytest.mark.asyncio
async def test_session_listing_agrees_across_the_generic_and_owner_read_routes(tmp_path: Path) -> None:
    """Both executors select, order and page the same sessions for the same filter.

    Mutation: change the default ordering, the origin/min_messages lowering or
    the ``next_offset`` decision in ``session_reads`` alone and page one, the
    page boundary, or the second page diverges from the generic route here.
    """

    root = tmp_path / "archive"
    seeded = run_off_event_loop(lambda: _seed(root, count=25))

    # Anti-vacuity: 25 rows make the omitted-limit page boundary observable.
    first_owner = await _owner_list(root, origin=Origin.CODEX_SESSION)
    first_generic = _generic_list(root, origin="codex-session")
    assert first_owner == first_generic
    assert first_owner[2] == 20
    assert first_owner[4] == 20

    first_owner = await _owner_list(root, origin=Origin.CODEX_SESSION, limit=2)
    first_generic = _generic_list(root, origin="codex-session", limit=2)
    assert first_owner == first_generic

    ordered_ids, total, _limit, _offset, next_offset = first_owner
    assert total == len(seeded)
    assert next_offset == 2, "the page boundary itself is part of the compared semantics"

    second_owner = await _owner_list(root, origin=Origin.CODEX_SESSION, limit=2, offset=next_offset)
    second_generic = _generic_list(root, origin="codex-session", limit=2, offset=next_offset)
    assert second_owner == second_generic

    walked = ordered_ids + second_owner[0]
    assert len(walked) == len(set(walked)), "paging must not repeat a session on either route"


@pytest.mark.asyncio
async def test_session_filters_agree_across_the_generic_and_owner_read_routes(tmp_path: Path) -> None:
    """A message-count filter selects the same sessions through both executors.

    Mutation: drop or invert ``min_messages`` in one route's spec lowering and
    the selected id lists stop matching.
    """

    root = tmp_path / "archive"
    run_off_event_loop(lambda: _seed(root))

    owner = await _owner_list(root, min_messages=3, limit=50)
    generic = _generic_list(root, min_messages=3, limit=50)
    assert owner == generic
    assert owner[1] == 3, "sessions 2, 3 and 4 carry three or more messages"


@pytest.mark.asyncio
async def test_lexical_search_selects_the_same_sessions_on_both_read_routes(tmp_path: Path) -> None:
    """Lexical search agrees on selection and cardinality, not on envelope shape.

    The generic route answers a ranked search envelope and the owner route a
    typed page; what must not differ is *which* sessions matched and how many
    the archive reports. Mutation: change the owner route's distinct-by-session
    collapse or its lexical term extraction and the id sets diverge.
    """

    root = tmp_path / "archive"
    seeded = run_off_event_loop(lambda: _seed(root))

    async with Polylogue(archive_root=root) as api:
        owner_page = await execute_session_operation(api, SessionSearch(expression="needle", limit=50))
    owner_ids = [hit.session.id for hit in owner_page.items]

    with open_operation_read(root) as pinned:
        envelope = execute_read_operation(
            "cli.query",
            {"params": {"query": "needle", "limit": 50}},
            archive=pinned.archive,
            serving_identity="direct",
        )
    hits = envelope["hits"]
    assert isinstance(hits, list)
    generic_ids = [str(hit["session"]["id"]) for hit in hits]

    assert set(owner_ids) == set(generic_ids) == set(seeded)
    assert owner_page.total == envelope["total"]
    assert owner_page.next_offset == envelope["next_offset"]


def _seed_exclusion(root: Path, texts: dict[str, str] | None = None) -> dict[str, str]:
    """Seed sessions whose text makes a ``-secret`` exclusion observable."""

    if texts is None:
        texts = {"plain": "needle alpha", "secret": "needle secret", "other": "unrelated beta"}
    ids: dict[str, str] = {}
    with ArchiveStore(root) as archive:
        for name, text in texts.items():
            ids[name] = write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=f"exclusion-{name}",
                    title=f"Session {name}",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m0",
                            role=Role.USER,
                            timestamp="2026-02-01T12:00:00Z",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                        )
                    ],
                ),
            )
    return ids


async def _mcp_sessions(
    root: Path, expression: str, *, limit: int = 50, offset: int = 0, sort: str | None = None
) -> dict[str, object]:
    import json
    from typing import cast

    from polylogue.mcp.server import build_server
    from tests.infra.mcp import MCPServerUnderTest, installed_runtime_services, invoke_surface_async

    server = cast(MCPServerUnderTest, build_server())
    query_fn = server._tool_manager._tools["query"].fn
    with installed_runtime_services(root):
        result = json.loads(
            await invoke_surface_async(
                query_fn, expression=expression, projection="sessions", limit=limit, offset=offset, sort=sort
            )
        )
    assert isinstance(result, dict)
    return result


@pytest.mark.asyncio
async def test_ranked_search_with_text_exclusion_is_refused_on_mcp_and_generic_routes(tmp_path: Path) -> None:
    """``needle -secret`` is refused by MCP's post-filter fallback, as by the generic read.

    The post-filter fallback (``server_cutover._query_advanced_sessions``) is
    the third executor. Mutation: drop its exclusion refusal and MCP answers
    a ranked page that ignores ``-secret`` while the generic read refuses.
    """

    root = tmp_path / "archive"
    run_off_event_loop(lambda: _seed_exclusion(root))

    with open_operation_read(root) as pinned, pytest.raises(ValueError, match="text exclusions"):
        execute_read_operation(
            "cli.query",
            # The CLI hands its root query over as the words it was given.
            {"params": {"query": ("needle", "-secret"), "limit": 50}},
            archive=pinned.archive,
            serving_identity="direct",
        )

    result = await _mcp_sessions(root, "needle -secret")
    assert result.get("is_error") is True, result
    assert result.get("code") == "invalid_argument"


@pytest.mark.asyncio
async def test_text_exclusion_listing_agrees_between_mcp_and_generic_routes(tmp_path: Path) -> None:
    """A bare ``-secret`` listing drops the same sessions on MCP and the generic read.

    Mutation: send the raw expression to FTS again, or list without the
    content post-filter, and MCP's selection stops matching the generic one.
    """

    root = tmp_path / "archive"
    ids = run_off_event_loop(lambda: _seed_exclusion(root))

    generic = _generic_list(root, query="-secret", limit=50)
    result = await _mcp_sessions(root, "-secret")
    assert result.get("is_error") is not True, result
    items = result["items"]
    assert isinstance(items, list)
    mcp_ids = [str(item["id"]) for item in items]

    assert set(generic[0]) == {ids["plain"], ids["other"]}
    assert mcp_ids == generic[0]
    assert result["total"] == generic[1] == 2


@pytest.mark.asyncio
async def test_text_exclusion_listing_pages_a_scope_larger_than_one_hydration_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MCP pages a ``-secret`` listing whose candidates span many hydration chunks.

    The exclusion is valid at any archive size, so the MCP listing must page
    it with an exact total rather than refuse a large candidate scope. The
    chunk is shrunk to two so seven candidates cross four chunks.

    Mutation: reinstate a candidate-count refusal (the removed
    ``POST_FILTER_HYDRATION_CAP``) at or below seven in
    ``_archive_list_summaries_with_post_filters`` or
    ``_archive_count_sessions_for_spec``, stop the candidate stream after the
    first chunk, or estimate the total from one page, and the MCP call errors,
    loses the last-chunk survivor, or reports a total other than five.
    """

    from polylogue.api import archive as archive_api

    monkeypatch.setattr(archive_api, "POST_FILTER_HYDRATION_CHUNK", 2)
    root = tmp_path / "archive"
    texts = {f"s{index}": ("needle secret" if index in (1, 4) else f"needle plain {index}") for index in range(7)}
    ids = run_off_event_loop(lambda: _seed_exclusion(root, texts))
    survivors = {ids[name] for name in texts if "secret" not in texts[name]}

    generic = _generic_list(root, query="-secret", limit=50)
    assert set(generic[0]) == survivors
    assert generic[1] == len(survivors) == 5

    walked: list[str] = []
    offset: int | None = 0
    pages = 0
    while offset is not None:
        result = await _mcp_sessions(root, "-secret", limit=2, offset=offset)
        assert result.get("is_error") is not True, result
        assert result["total"] == 5, "the exclusion total must be exact, not a page estimate"
        items = result["items"]
        assert isinstance(items, list)
        assert len(items) == min(2, 5 - offset)
        walked.extend(str(item["id"]) for item in items)
        next_offset = result.get("next_offset")
        assert next_offset is None or isinstance(next_offset, int)
        offset = next_offset
        pages += 1
        assert pages <= 3, "paging must terminate after the last survivor"

    assert pages == 3
    assert walked == generic[0], "MCP pages must walk the generic route's order without gaps or repeats"


@pytest.mark.asyncio
async def test_transcript_windows_agree_across_the_generic_and_owner_read_routes(tmp_path: Path) -> None:
    """Both executors page one transcript into the same windows.

    Each route resumes only its own continuation: ``session.read`` keeps its
    ``session-read-v1`` dialect and ``sessions.read`` stamps
    ``session-owner-v1``, and replaying one against the other is a typed
    refusal (``test_session_read_parity``). What must agree is the window each
    page selects. Mutation: let either route decide its offset, page size,
    ``next_offset`` or row order outside ``read_transcript_window_sync`` and a
    page's message ids or coordinates diverge here.
    """

    root = tmp_path / "archive"
    session_id = run_off_event_loop(lambda: _seed(root))[-1]  # five messages: three windows of two

    owner_pages: list[tuple[list[str], int | None, int, int | None]] = []
    async with Polylogue(archive_root=root) as api:
        request = SessionRead(ref=f"session:{session_id}", limit=2)
        while True:
            page = await execute_session_operation(api, request)
            owner_pages.append(([str(item.id) for item in page.items], page.total, page.offset, page.next_offset))
            if page.continuation is None:
                break
            request = SessionRead(ref=f"session:{session_id}", continuation=page.continuation)

    generic_pages: list[tuple[list[str], int | None, int, int | None]] = []
    with open_operation_read(root) as pinned:
        payload: dict[str, object] = {"ref": f"session:{session_id}", "limit": 2}
        while True:
            body = execute_read_operation("session.read", payload, archive=pinned.archive, serving_identity="direct")
            session = body["session"]
            assert isinstance(session, dict)
            generic_pages.append(
                (
                    [str(message["message_id"]) for message in session["messages"]],
                    _optional_int(body["total"]),
                    _int(body["offset"]),
                    _optional_int(body["next_offset"]),
                )
            )
            if body["continuation"] is None:
                break
            payload = {"ref": f"session:{session_id}", "continuation": body["continuation"]}

    assert owner_pages == generic_pages
    assert [len(ids) for ids, *_ in owner_pages] == [2, 2, 1]
    walked = [message_id for ids, *_ in owner_pages for message_id in ids]
    assert len(walked) == len(set(walked)) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reference", ["equivalence-4", "codex-session:equivalence-4", "session:codex-session:equivalence-4"]
)
async def test_explicit_session_scope_resolves_on_both_routes(tmp_path: Path, reference: str) -> None:
    """Removing the shared resolver from typed query sends a literal outer namespace to SQL."""
    root = tmp_path / "archive"
    seeded = run_off_event_loop(lambda: _seed(root, native_suffix="-full"))
    owner = await _owner_list(root, expression=f"id:{reference}")
    generic = _generic_list(root, query=f"id:{reference}")
    assert owner == generic
    assert owner[0] == [seeded[4]]


@pytest.mark.asyncio
async def test_unique_prefix_read_preserves_full_identity_on_resume(tmp_path: Path) -> None:
    """Projection must use the resolved window identity on every page."""
    root = tmp_path / "archive"
    seeded = run_off_event_loop(lambda: _seed(root, native_suffix="-full"))
    reference = "session:codex-session:equivalence-4"
    async with Polylogue(archive_root=root) as api:
        first = await execute_session_operation(api, SessionRead(ref=reference, limit=1))
        assert first.continuation
        second = await execute_session_operation(api, SessionRead(ref=reference, continuation=first.continuation))
    assert first.items[0].session_id == second.items[0].session_id == seeded[4]
    assert second.offset == 1
    assert first.items[0].id != second.items[0].id


@pytest.mark.asyncio
async def test_action_lane_excludes_dialogue_and_reports_its_lane(tmp_path: Path) -> None:
    """Dropping actions_only admits the dialogue control and inflates total."""
    root = tmp_path / "archive"

    def seed_actions() -> str:
        ids: dict[str, str] = {}
        with ArchiveStore(root) as archive:
            for name, block in (
                ("dialogue", ParsedContentBlock(type=BlockType.TEXT, text="needle")),
                (
                    "action",
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE, tool_id="call-1", tool_name="run", tool_input={"command": "needle"}
                    ),
                ),
            ):
                ids[name] = write_index_session(
                    archive,
                    ParsedSession(
                        source_name=Provider.CODEX,
                        provider_session_id=name,
                        messages=[
                            ParsedMessage(
                                provider_message_id="m1",
                                role=Role.ASSISTANT,
                                timestamp="2026-02-01T12:00:00Z",
                                blocks=[block],
                            )
                        ],
                    ),
                )
            return ids["action"]

    action = run_off_event_loop(seed_actions)
    async with Polylogue(archive_root=root) as api:
        page = await execute_session_operation(api, SessionSearch(expression="needle lane:actions"))
        listed = await execute_session_operation(api, SessionList(expression="needle lane:actions"))
    with open_operation_read(root) as pinned:
        generic = execute_read_operation(
            "cli.query", {"params": {"query": "needle lane:actions"}}, archive=pinned.archive, serving_identity="direct"
        )
    assert [hit.session.id for hit in page.items] == [action]
    hits = generic["hits"]
    assert isinstance(hits, list)
    assert [hit["session"]["id"] for hit in hits] == [action]
    assert [item.id for item in listed.items] == [action]
    assert page.total == listed.total == 1
    assert generic["total"] is None  # Ranked action envelopes do not declare an exact total.
    mcp = await _mcp_sessions(root, "needle lane:actions")
    assert mcp["retrieval_lane"] == "actions"
    mcp_hits = mcp["hits"]
    assert isinstance(mcp_hits, list)
    assert [hit["session"]["id"] for hit in mcp_hits] == [action]
    assert mcp_hits[0]["match"]["retrieval_lane"] == "actions"

    assert page.items[0].match.retrieval_lane == "actions"


@pytest.mark.asyncio
async def test_transcript_epoch_uses_the_explicit_index_path(tmp_path: Path) -> None:
    """Opening the default root Index for epoch binding refuses a valid explicit Index."""
    root = tmp_path / "archive"
    seeded = run_off_event_loop(lambda: _seed(root))
    selected = root / ".index-generations" / "selected" / "index.db"
    selected.parent.mkdir(parents=True)
    (root / "index.db").rename(selected)
    async with Polylogue(archive_root=root, db_path=selected) as api:
        first = await execute_session_operation(api, SessionRead(ref=seeded[4], limit=1))
        assert first.continuation
        second = await execute_session_operation(api, SessionRead(ref=seeded[4], continuation=first.continuation))
    assert first.total == second.total == 5
    assert second.offset == 1
    assert first.items[0].session_id == second.items[0].session_id == seeded[4]
    assert not (root / "index.db").exists()


@pytest.mark.asyncio
async def test_missing_and_ambiguous_scopes_keep_the_generic_outcome(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    run_off_event_loop(lambda: _seed(root))
    assert await _owner_list(root, expression="id:session:missing") == _generic_list(root, query="id:session:missing")
    with pytest.raises(ValueError, match="ambiguous"):
        await _owner_list(root, expression="id:session:codex-session:equivalence-")
    with pytest.raises(ValueError, match="ambiguous"):
        _generic_list(root, query="id:session:codex-session:equivalence-")
    async with Polylogue(archive_root=root) as api:
        with pytest.raises(ValueError, match="ambiguous"):
            await execute_session_operation(api, SessionRead(ref="session:codex-session:equivalence-"))


@pytest.mark.asyncio
async def test_every_typed_indexed_page_keeps_the_explicit_index(tmp_path: Path) -> None:
    from polylogue.operations.session_contracts import SessionTimeline

    root = tmp_path / "archive"
    seeded = run_off_event_loop(lambda: _seed(root))
    selected = root / ".index-generations" / "selected" / "index.db"
    selected.parent.mkdir(parents=True)
    (root / "index.db").rename(selected)
    async with Polylogue(archive_root=root, db_path=selected) as api:
        for request in (SessionList(limit=1), SessionSearch(expression="needle", limit=1), SessionTimeline(limit=1)):
            first = await execute_session_operation(api, request)
            assert first.items and first.continuation
            second = await execute_session_operation(api, type(request)(continuation=first.continuation))
            assert second.offset == 1
            assert second.total == first.total == len(seeded)
    assert not (root / "index.db").exists()


@pytest.mark.asyncio
async def test_latest_expression_bounds_both_typed_selection_and_window(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    seeded = run_off_event_loop(lambda: _seed(root))
    async with Polylogue(archive_root=root) as api:
        listed = await execute_session_operation(api, SessionList(expression='{"latest":true}'))
        searched = await execute_session_operation(api, SessionSearch(expression='{"query":"needle","latest":true}'))
        window = await execute_session_operation(api, SessionList(expression='{"limit":2,"offset":1}'))
        assert window.continuation
        resumed = await execute_session_operation(api, SessionList(continuation=window.continuation, limit=1))
    generic = _generic_list(root, query='{"latest":true}')
    assert [item.id for item in listed.items] == generic[0] == [seeded[-1]]
    assert [item.session.id for item in searched.items] == [seeded[-1]]
    for page in (listed, searched):
        assert page.total == page.limit == 1
        assert page.continuation is None
    assert (window.limit, window.offset, window.next_offset) == (2, 1, 3)
    assert (resumed.limit, resumed.offset) == (1, 3)
    assert resumed.items[0].id == seeded[1]


@pytest.mark.asyncio
async def test_typed_continuation_refuses_an_equal_counter_other_archive(tmp_path: Path) -> None:
    from functools import partial

    from polylogue.archive.query.transaction import QueryContinuationStaleError

    roots = (tmp_path / "first", tmp_path / "second")
    for index, root in enumerate(roots):
        run_off_event_loop(partial(_seed, root, native_suffix=str(index)))
    async with Polylogue(archive_root=roots[0]) as api:
        first = await execute_session_operation(api, SessionList(limit=1))
    assert first.continuation
    async with Polylogue(archive_root=roots[1]) as api:
        with pytest.raises(QueryContinuationStaleError):
            await execute_session_operation(api, SessionList(continuation=first.continuation))


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_missing_date_actions_sort_last_on_both_read_routes(tmp_path: Path, reverse: bool) -> None:
    import json

    root = tmp_path / "archive"

    def seed() -> list[str]:
        with ArchiveStore(root) as archive:
            return [
                write_index_session(
                    archive,
                    ParsedSession(
                        source_name=Provider.CODEX,
                        provider_session_id=name,
                        messages=[
                            ParsedMessage(
                                provider_message_id="m",
                                role=Role.ASSISTANT,
                                timestamp=timestamp,
                                blocks=[
                                    ParsedContentBlock(
                                        type=BlockType.TOOL_USE,
                                        tool_id="call",
                                        tool_name="run",
                                        tool_input={"command": "needle"},
                                    )
                                ],
                            )
                        ],
                    ),
                )
                for name, timestamp in (("missing", None), ("dated", "2026-02-01T12:00:00Z"))
            ]

    missing, dated = run_off_event_loop(seed)
    expression = json.dumps({"query": "needle", "retrieval_lane": "actions", "sort": "date", "reverse": reverse})
    async with Polylogue(archive_root=root) as api:
        owner = await execute_session_operation(api, SessionSearch(expression=expression, limit=1))
    with open_operation_read(root) as pinned:
        generic = execute_read_operation(
            "cli.query",
            {
                "params": {
                    "query": "needle",
                    "retrieval_lane": "actions",
                    "sort": "date",
                    "reverse": reverse,
                    "limit": 1,
                }
            },
            archive=pinned.archive,
            serving_identity="direct",
        )
    assert [item.session.id for item in owner.items] == [dated]
    generic_hits = generic["hits"]
    assert isinstance(generic_hits, list)
    assert [hit["session"]["id"] for hit in generic_hits] == [dated]
    assert dated != missing


@pytest.mark.parametrize(
    ("payload", "expected_outcome", "exit_code"),
    [
        ({"operation": "sessions.list", "expression": "id:session:missing"}, "empty", 2),
        ({"operation": "sessions.timeline"}, "degraded", 1),
    ],
)
def test_machine_session_cli_returns_the_declared_terminal_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: dict[str, str],
    expected_outcome: str,
    exit_code: int,
) -> None:
    import io
    import json
    import sys

    import polylogue.api
    from polylogue.cli.session_operations import main

    root = tmp_path / "archive"
    _seed(root, count=2)  # The second session has a message with no recorded event time.
    facade = Polylogue
    monkeypatch.setattr(polylogue.api, "Polylogue", lambda: facade(archive_root=root))
    monkeypatch.setattr(sys, "argv", ["session_operations", "execute"])
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode())))
    assert main() == exit_code
    assert json.loads(capsys.readouterr().out)["outcome"] == expected_outcome


@pytest.mark.asyncio
async def test_root_only_facade_follows_active_index_and_explicit_shadow_stays_selected(tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "archive"
    seeded = run_off_event_loop(lambda: _seed(root))
    selected = root / ".index-generations" / "selected" / "index.db"
    selected.parent.mkdir(parents=True)
    shutil.copyfile(root / "index.db", selected)

    # Distinct populations prove the selected Index, independent of title rendering.
    def prepare_shadow() -> None:
        with ArchiveStore.open_existing(root, read_only=False) as shadow:
            shadow.delete_sessions(tuple(seeded[1:]))

    run_off_event_loop(prepare_shadow)
    (root / ".index-active-pointer").write_text(str(selected), encoding="utf-8")
    async with Polylogue(archive_root=root) as api:
        assert api.backend.db_path == selected
        page = await execute_session_operation(api, SessionList(limit=1))
        transcript = await execute_session_operation(api, SessionRead(ref=seeded[-1], limit=1))
    async with Polylogue(archive_root=root, db_path=root / "index.db") as api:
        assert api.backend.db_path == root / "index.db"
        explicit = await execute_session_operation(api, SessionList(limit=1))
    assert page.total == len(seeded)
    assert page.items[0].id == seeded[-1]
    assert explicit.total == 1
    assert explicit.items[0].id == seeded[0]
    assert transcript.items


@pytest.mark.asyncio
@pytest.mark.parametrize("outer_offset", [0, 2])
async def test_generic_read_executes_compiled_expression_window(tmp_path: Path, outer_offset: int) -> None:
    root = tmp_path / "archive"
    run_off_event_loop(lambda: _seed(root))
    expression = '{"limit":2,"offset":1}'
    owner = await _owner_list(root, expression=expression, limit=20, offset=outer_offset)
    generic = _generic_list(root, query=expression, limit=20, offset=outer_offset)
    assert generic == owner
    assert generic[2:4] == (2, outer_offset or 1)
    mcp = await _mcp_sessions(root, expression, limit=20, offset=outer_offset)
    mcp_items = mcp["items"]
    assert isinstance(mcp_items, list)
    assert [item["id"] for item in mcp_items] == generic[0]
    assert (mcp["limit"], mcp["offset"]) == generic[2:4]


@pytest.mark.asyncio
async def test_mcp_advanced_random_listing_retains_boolean_selection(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    run_off_event_loop(lambda: _seed(root))
    expression = 'title:"Session 0" OR title:"Session 1"'
    owner = await _owner_list(root, expression=expression)
    mcp = await _mcp_sessions(root, expression, limit=20, sort="random")
    assert mcp["total"] == owner[1] == 2
    mcp_items = mcp["items"]
    assert isinstance(mcp_items, list)
    assert {item["id"] for item in mcp_items} == set(owner[0])
