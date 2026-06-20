from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.demo import seed_demo_archive, verify_demo_archive
from polylogue.scenarios import DEMO_CLAUDE_CODE_SESSION_ID, DEMO_SESSION_IDS


@pytest.mark.asyncio
async def test_seed_demo_archive_creates_ready_queryable_archive(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"

    seed = await seed_demo_archive(archive_root, force=True, with_overlays=True)
    verify = verify_demo_archive(archive_root, require_overlays=True)

    assert seed.archive_root == archive_root
    assert seed.session_count == 3
    assert seed.message_count == 19
    assert seed.session_ids == tuple(sorted(DEMO_SESSION_IDS))
    assert seed.overlays_seeded is True
    assert seed.assertion_count >= 4

    assert verify.ok is True
    assert verify.session_count == 3
    assert verify.message_count == 19
    assert DEMO_CLAUDE_CODE_SESSION_ID in verify.query_hits
    assert verify.overlays_present is True
    assert verify.absolute_path_leaks == ()
    assert verify.problems == ()


@pytest.mark.asyncio
async def test_demo_verify_reports_missing_overlays(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"

    await seed_demo_archive(archive_root, force=True, with_overlays=False)
    verify = verify_demo_archive(archive_root, require_overlays=True)

    assert verify.ok is False
    assert "expected demo overlays" in "\n".join(verify.problems)
