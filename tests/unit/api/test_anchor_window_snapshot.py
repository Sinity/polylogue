from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.operations import message_locator
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.frozen_clock import FrozenClock
from tests.infra.storage_records import db_setup, seed_anchor_session


@pytest.mark.asyncio
async def test_anchor_and_page_keep_the_original_snapshot_during_replacement(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, frozen_clock: FrozenClock
) -> None:
    db_path = db_setup(workspace_env)
    session_id = seed_anchor_session(db_path)
    async with Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path) as api:
        original = await api.read_transcript_window(session_id, limit=4)
        anchor = str(original.rows[3].id)
        locate = message_locator.window_offset_around
        reached = False

        def replace_after_location(archive: ArchiveStore, session_id: str, message_id: str, limit: int) -> int:
            nonlocal reached
            offset = locate(archive, session_id, message_id, limit)
            seed_anchor_session(db_path, insert_prefix=True)
            reached = True
            return offset

        monkeypatch.setattr(message_locator, "window_offset_around", replace_after_location)
        window = await api.read_transcript_window(session_id, around=anchor, limit=2)
        current = await api.read_transcript_window(session_id, limit=5)

    assert reached
    assert window.offset == 2 and window.total == 4
    assert [str(row.id) for row in window.rows] == [str(row.id) for row in original.rows[2:]]
    assert anchor in {str(row.id) for row in window.rows}
    assert current.total == 5 and str(current.rows[-1].id) == anchor
    assert window.transaction.archive_epoch != current.transaction.archive_epoch


@pytest.mark.asyncio
async def test_anchor_window_preserves_domain_rows_and_continuation(
    workspace_env: dict[str, Path], frozen_clock: FrozenClock
) -> None:
    from polylogue.operations.message_locator import MessageNotInSessionError

    db_path = db_setup(workspace_env)
    session_id = seed_anchor_session(db_path)
    async with Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path) as api:
        ordinary = await api.read_transcript_window(session_id, limit=2)
        anchored = await api.read_transcript_window(session_id, around=str(ordinary.rows[1].id), limit=2)
        with pytest.raises(MessageNotInSessionError) as refused:
            await api.read_transcript_window(session_id, around="absent-message", limit=2)
        assert refused.value.code == "message_not_found"

    anchored_rows = [row.model_dump(mode="json") for row in anchored.rows]
    ordinary_rows = [row.model_dump(mode="json") for row in ordinary.rows]
    assert ordinary_rows[0]["blocks"][0]["signature"] == "neutral-signature"
    assert ordinary_rows[0]["blocks"][0]["media_type"] == "text/plain"
    assert anchored_rows == ordinary_rows, [
        {key: (left[key], right[key]) for key in left if left[key] != right[key]}
        for left, right in zip(anchored_rows, ordinary_rows, strict=True)
    ]
    assert anchored.offset == ordinary.offset == 0
    assert anchored.total == ordinary.total == 4
    assert anchored.continuation == ordinary.continuation and anchored.continuation is not None
