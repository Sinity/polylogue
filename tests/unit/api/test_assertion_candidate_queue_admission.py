"""Admission contracts for the durable assertion candidate queue.

The queue is the actionable work list; the sibling review read model is an
audit surface that retains expired and judged rows on purpose. These tests hold
the two apart, and hold the durable user tier to a refusal rather than an empty
result when it cannot be read.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.core.enums import AssertionKind, BlockType, Provider, TitleSource
from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

_NOW_MS = 1_700_000_000_000


def _seed(root: Path) -> str:
    with ArchiveStore(root) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="queue-admission",
                title="Queue admission source",
                title_source=TitleSource.ORIGIN,
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="source message",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="source message")],
                    )
                ],
            )
        )
    with sqlite3.connect(root / "user.db") as conn:
        for assertion_id, expires_at_ms in (
            ("candidate-live", None),
            ("candidate-expired", _NOW_MS - 1),
        ):
            upsert_assertion(
                conn,
                assertion_id=assertion_id,
                target_ref=f"session:{session_id}",
                kind=AssertionKind.LESSON,
                body_text="a candidate awaiting judgment",
                author_ref="agent:standing-queries",
                author_kind="agent",
                status="candidate",
                staleness=None if expires_at_ms is None else {"expires_at_ms": expires_at_ms},
                now_ms=_NOW_MS - 10_000,
            )
    return session_id


async def test_candidate_queue_excludes_expired_claims_the_review_surface_retains(tmp_path: Path) -> None:
    """Anti-vacuity: route the queue back through the candidate-review read
    model (which pins ``include_expired=True``) and the expired claim reappears
    in the actionable queue.
    """

    session_id = _seed(tmp_path)
    archive = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        queue = await archive.list_assertion_candidates(target_ref=f"session:{session_id}")
        reviews = await archive.list_assertion_candidate_reviews(
            target_ref=f"session:{session_id}",
            statuses=("candidate",),
            limit=20,
        )
    finally:
        await archive.close()

    queued = {item.assertion_id for item in queue}
    reviewed = {item.candidate_ref.removeprefix("assertion:") for item in reviews.items}

    assert queued == {"candidate-live"}
    assert reviewed == {"candidate-live", "candidate-expired"}


async def test_unreadable_user_tier_refuses_instead_of_reporting_no_candidates(tmp_path: Path) -> None:
    """Anti-vacuity: swallow the tier error back into ``return []`` and this
    passes silently with an empty queue.

    The user tier is durable and irreplaceable. Reporting "no candidates" for
    an archive whose ``user.db`` cannot be read presents a read failure as a
    judged-empty queue.
    """

    session_id = _seed(tmp_path)
    (tmp_path / "user.db").unlink()

    archive = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        with pytest.raises(ArchiveTierUnavailableError) as missing:
            await archive.list_assertion_candidates(target_ref=f"session:{session_id}")
        assert missing.value.tier == "user"

        (tmp_path / "user.db").mkdir()
        with pytest.raises(ArchiveTierUnavailableError) as not_a_file:
            await archive.list_assertion_candidates(target_ref=f"session:{session_id}")
        assert "not a regular file" in not_a_file.value.reason
    finally:
        await archive.close()
