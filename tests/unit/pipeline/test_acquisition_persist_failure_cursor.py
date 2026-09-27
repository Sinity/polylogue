"""A raw that failed to persist must not get a stat cursor that skips it later."""

from __future__ import annotations

import errno
import json
from pathlib import Path
from typing import Any

import pytest

from polylogue.config import Source
from polylogue.pipeline.services.acquisition import AcquisitionService
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite import SQLiteBackend


def _write_session(path: Path) -> None:
    record = {
        "type": "user",
        "uuid": "message-1",
        "parentUuid": None,
        "sessionId": "persist-failure",
        "timestamp": "2026-07-10T00:00:00Z",
        "message": {"role": "user", "content": "stored on the second pass"},
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_failed_raw_persist_withholds_the_source_cursor(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity: before the failure was recorded against its path, the
    pass saved the stat cursor anyway and the second pass skipped the file
    (``acquired == 0``), so its bytes were never stored."""
    source_dir = workspace_env["data_root"] / "claude-projects"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "session.jsonl"
    _write_session(source_path)
    source = Source(name="claude-code", path=source_path)

    original = SessionRepository.admit_raw
    calls = 0

    async def fail_once(self: SessionRepository, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(SessionRepository, "admit_raw", fail_once)

    backend = SQLiteBackend(db_path=workspace_env["archive_root"] / "index.db")
    try:
        first = await AcquisitionService(backend).acquire_sources([source])
        assert first.errors == 1
        assert first.acquired == 0
        known = await SessionRepository(backend=backend).get_known_source_cursors()
        assert str(source_path) not in known

        second = await AcquisitionService(backend).acquire_sources([source])
        assert second.errors == 0
        assert second.acquired == 1
    finally:
        await backend.close()
