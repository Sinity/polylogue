"""A resumed transcript window reads the selection and page size its caller asked for.

Both declared read routes serve transcript windows: the typed session owner
(``sessions.read`` through ``execute_session_operation``) and the generic
``session.read`` operation the CLI and daemon transport reach. A continuation
carries the window it was minted for; a resume that states only the token must
read under the token's filters, and a resume that states a smaller ``limit``
must get a page of that size on either route.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.core.enums import BlockType, Provider, Role
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from polylogue.operations.read_contracts import SessionReadRequest
from polylogue.operations.session_contracts import SessionRead
from polylogue.operations.session_reads import execute_session_operation
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.live_ingest import write_index_session


def _seed(root: Path) -> str:
    """One session of six messages alternating user and assistant."""

    with ArchiveStore(root) as archive:
        return write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="resume-selection",
                title="Resume selection",
                messages=[
                    ParsedMessage(
                        provider_message_id=f"m{position}",
                        role=Role.USER if position % 2 == 0 else Role.ASSISTANT,
                        timestamp="2026-03-01T12:00:00Z" if position == 0 else None,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"turn {position}")],
                    )
                    for position in range(6)
                ],
            ),
        )


def _generic_read(root: Path, payload: dict[str, object]) -> dict[str, object]:
    # The operation transport validates the payload into its declared request
    # and forwards every field, defaults included; read through that shape.
    declared = SessionReadRequest.model_validate(payload).model_dump(mode="json")
    with open_operation_read(root) as pinned:
        return execute_read_operation("session.read", declared, archive=pinned.archive, serving_identity="direct")


def _transcript_ids(body: dict[str, object]) -> list[str]:
    session = body["session"]
    assert isinstance(session, dict)
    return [str(row["message_id"]) for row in session["messages"]]


def _message_ids(body: dict[str, object]) -> list[str]:
    rows = body["messages"]
    assert isinstance(rows, list)
    return [str(row["id"]) for row in rows]


@pytest.mark.asyncio
async def test_owner_resume_reads_under_the_tokens_filters(tmp_path: Path) -> None:
    """Fails if a token-only resume reads with default filters: the second page
    would be the assistant turn at unfiltered offset 1, with a total of six."""

    root = tmp_path / "archive"
    session_id = run_off_event_loop(lambda: _seed(root))
    async with Polylogue(archive_root=root) as api:
        first = await execute_session_operation(
            api, SessionRead(ref=f"session:{session_id}", message_role=(Role.USER,), limit=1)
        )
        assert first.continuation is not None
        second = await execute_session_operation(
            api, SessionRead(ref=f"session:{session_id}", continuation=first.continuation)
        )
    assert first.total == second.total == 3
    assert second.offset == 1
    assert [item.role for item in second.items] == [Role.USER]
    assert second.items[0].id != first.items[0].id


@pytest.mark.asyncio
async def test_generic_messages_read_refuses_a_filtered_owner_token(tmp_path: Path) -> None:
    """Fails if the generic messages window checks only its own payload for
    filters and serves unfiltered rows at a filtered token's offset."""

    root = tmp_path / "archive"
    session_id = run_off_event_loop(lambda: _seed(root))
    async with Polylogue(archive_root=root) as api:
        first = await execute_session_operation(
            api, SessionRead(ref=f"session:{session_id}", message_role=(Role.USER,), limit=1)
        )
    assert first.continuation is not None
    with pytest.raises(ValueError, match="filtered message window"):
        _generic_read(root, {"ref": f"session:{session_id}", "kind": "messages", "continuation": first.continuation})


@pytest.mark.asyncio
async def test_a_smaller_resume_limit_narrows_the_page_on_both_routes(tmp_path: Path) -> None:
    """Fails if either route drops the caller's narrowed limit beside a
    continuation (a two-row page), moves the token's offset, or reads the
    transport's serialized default limit as a request to widen the token."""

    root = tmp_path / "archive"
    session_id = run_off_event_loop(lambda: _seed(root))
    ref = f"session:{session_id}"
    async with Polylogue(archive_root=root) as api:
        owner_first = await execute_session_operation(api, SessionRead(ref=ref, limit=2))
        assert owner_first.continuation is not None
        owner_next = await execute_session_operation(
            api, SessionRead(ref=ref, limit=1, continuation=owner_first.continuation)
        )
    owner_ids = [str(item.id) for item in owner_next.items]

    for kind, ids in (("transcript", _transcript_ids), ("messages", _message_ids)):
        first = _generic_read(root, {"ref": ref, "kind": kind, "limit": 2})
        token = first["continuation"]
        assert isinstance(token, str)
        resumed = _generic_read(root, {"ref": ref, "kind": kind, "limit": 1, "continuation": token})
        assert (resumed["offset"], resumed["limit"]) == (2, 1), kind
        kept = _generic_read(root, {"ref": ref, "kind": kind, "continuation": token})
        assert (kept["offset"], kept["limit"]) == (2, 2), kind
        assert ids(resumed) == owner_ids, kind
        with pytest.raises(ValueError, match="cannot widen"):
            _generic_read(root, {"ref": ref, "kind": kind, "limit": 3, "continuation": token})

    assert (owner_next.offset, owner_next.limit, len(owner_ids)) == (2, 1, 1)
