"""Publication owns neither periodic WAL checkpoints nor forced allocator release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import pytest

from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.pipeline.services.ingest_batch import process_ingest_batch
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.pipeline.services.parsing_models import ParseResult
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.source_builders import ChatGPTExportBuilder


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["checkpoint", "memory_release"])
async def test_actual_ingest_publication_leaves_upkeep_to_its_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    root = tmp_path / "archive"
    payload = json.dumps(
        ChatGPTExportBuilder("publication-upkeep").add_node("user", "actual acquired publication").build()
    ).encode()

    def acquire() -> str:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path="conversations.json",
                canonical_source_path="conversations.json",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(root, acquire)

    def refuse(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("publication performed owner upkeep")

    if operation == "checkpoint":
        monkeypatch.setattr("polylogue.storage.sqlite.wal_checkpoint.checkpoint_wal", refuse)
        monkeypatch.setattr("polylogue.storage.sqlite.wal_checkpoint.checkpoint_archive_wals", refuse)
    else:
        monkeypatch.setattr("polylogue.core.memory.release_process_memory", refuse)
    config = Config(archive_root=root, render_root=tmp_path / "render", sources=[])
    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda **_kwargs: config)
    backend = SQLiteBackend(db_path=root / "index.db")
    repository = SessionRepository(backend=backend, archive_root=root)
    result = ParseResult()
    async with prepared_live_convergence_owner(root) as owner:
        service = ParsingService(repository, root, config, retained_runner=owner.replay_retained_raw_ids)
        observation = await process_ingest_batch(service, [raw_id], result, None)
    assert observation is not None
    assert observation["records"] == 1
    assert observation["sessions"] == 1
    assert observation["failed_raw_count"] == 0
    assert observation["converged"] is True
    assert result.processed_ids == {"chatgpt-export:publication-upkeep"}
