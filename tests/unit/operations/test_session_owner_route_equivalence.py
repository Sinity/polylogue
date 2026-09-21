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
from polylogue.operations.session_contracts import SessionList, SessionSearch
from polylogue.operations.session_reads import execute_session_operation
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.live_ingest import write_index_session


def _seed(root: Path, count: int = 5) -> list[str]:
    """Seed sessions that differ in date and message count, so order and filters bite."""

    ids: list[str] = []
    with ArchiveStore(root) as archive:
        for index in range(count):
            ids.append(
                write_index_session(
                    archive,
                    ParsedSession(
                        source_name=Provider.CODEX,
                        provider_session_id=f"equivalence-{index}",
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
    seeded = _seed(root)

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
    _seed(root)

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
    seeded = _seed(root)

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
