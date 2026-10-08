"""Manual continuation is rederived from User authority during retained replay."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.core.enums import Provider
from polylogue.sources.acquisition_boundary import bound_source_observation, open_bound_path
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_prepare, run_archive_fixture_write
from tests.infra.daemon_operations import async_daemon_serving_archive
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.source_builders import ChatGPTExportBuilder


@pytest.mark.asyncio
async def test_public_manual_continuation_survives_retained_rebuild(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    parent = "chatgpt-export:manual-parent"
    child = "chatgpt-export:manual-child"

    def acquire(native: str, body: str, acquired: int) -> str:
        bootstrap_archive_root(root)
        document = ChatGPTExportBuilder(native).add_node("user", body).build()
        document["update_time"] = 1704067200.0 + acquired
        payload = json.dumps(document).encode()
        path = tmp_path / f"{native}.json"
        path.write_bytes(payload)
        with ArchiveStore.open_existing(root, read_only=False) as archive, open_bound_path(path, None) as original:
            canonical, observation = bound_source_observation(original)
            assert canonical is not None and observation is not None
            return archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=original.read(),
                source_path=str(path),
                canonical_source_path=canonical,
                acquired_at_ms=acquired,
                file_mtime_ms=observation[3] // 1_000_000,
                post_parse=True,
            )

    raws = (
        await run_archive_fixture_write(root, lambda: acquire("manual-child", "child original", 1)),
        await run_archive_fixture_write(root, lambda: acquire("manual-parent", "parent original", 1)),
    )
    async with prepared_live_convergence_owner(root) as owner:
        (await owner.replay_retained_raw_ids(raws)).require_complete()
    async with async_daemon_serving_archive(root):
        async with Polylogue(archive_root=root) as api:
            await api.record_manual_continuation(child, parent)
            session = await api.get_session(child)
            assert session is not None and str(session.parent_id) == parent
    with closing(sqlite3.connect(root / "user.db")) as user:
        authority = user.execute(
            "SELECT assertion_id,value_json,status FROM assertions WHERE kind='handoff'"
        ).fetchall()
    assert len(authority) == 1
    assert json.loads(authority[0][1]) == {
        "_schema": "polylogue.manual-continuation.v1",
        "parent_session_id": parent,
    }
    # Neither acquisition path exists. The empty generation can use only
    # retained Source bytes and the same durable User assertion.
    (tmp_path / "manual-child.json").unlink()
    (tmp_path / "manual-parent.json").unlink()
    assert not (tmp_path / "manual-child.json").exists()
    assert not (tmp_path / "manual-parent.json").exists()
    generation = await run_archive_fixture_write(
        root,
        lambda: ColdBuildGeneration.begin(
            root,
            reason="test-manual-continuation",
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("neutral", tmp_path / "absent"),)),
        ),
    )
    register_cold_build_generation(generation)
    try:
        async with prepared_live_convergence_owner(root) as owner:
            (await owner.replay_retained_raw_ids(raws)).require_complete()
        await run_archive_fixture_prepare(generation.promote)
    finally:
        clear_cold_build_generation()
        if not generation.promoted:
            await run_archive_fixture_write(root, generation.discard)
    # Reopen both the daemon and public reader against the promoted generation.
    async with async_daemon_serving_archive(root):
        async with Polylogue(archive_root=root) as api:
            session = await api.get_session(child)
            assert session is not None and str(session.parent_id) == parent
            assert [
                block["text"] for message in session.messages for block in message.blocks if block["type"] == "text"
            ] == ["child original"], session
    with closing(sqlite3.connect(root / "user.db")) as user:
        assert (
            user.execute("SELECT assertion_id,value_json,status FROM assertions WHERE kind='handoff'").fetchall()
            == authority
        )
