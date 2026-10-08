"""A raw that failed to persist must not get a stat cursor that skips it later."""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from polylogue.config import Source
from polylogue.pipeline.services.acquisition import AcquisitionService
from polylogue.sources.parsers.base import RawSessionData
from polylogue.sources.source_acquisition_components import (
    AcquisitionObservation,
    SourceReadContext,
    read_plain_source_file,
)
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite import SQLiteBackend


@asynccontextmanager
async def _admitted_acquisition(backend: SQLiteBackend, archive_root: Path) -> AsyncIterator[AcquisitionService]:
    """Acquisition publishes through the daemon's admitted writer, like configured catch-up."""
    from polylogue.daemon.drive_catchup import DriveCatchupExecution
    from tests.infra.live_ingest import prepared_live_convergence_owner

    async with prepared_live_convergence_owner(archive_root) as owner:
        execution = DriveCatchupExecution(owner._write_coordinator, compute_adapter=owner._compute_adapter)
        yield AcquisitionService(backend, execution=execution)


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
@pytest.mark.parametrize("directory", ["claude-projects", "run:1"])
async def test_failed_raw_persist_withholds_the_source_cursor(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    directory: str,
) -> None:
    """Anti-vacuity: before the failure was recorded against its path, the
    pass saved the stat cursor anyway and the second pass skipped the file
    (``acquired == 0``), so its bytes were never stored. The ``run:1``
    directory is a valid POSIX path whose colon a first-colon split would
    truncate to a prefix matching no file."""
    source_dir = workspace_env["data_root"] / directory
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
        async with _admitted_acquisition(backend, workspace_env["archive_root"]) as _service:
            first = await _service.acquire_sources([source])
        assert first.errors == 1
        assert first.acquired == 0
        known = await SessionRepository(backend=backend).get_known_source_cursors()
        assert str(source_path) not in known

        async with _admitted_acquisition(backend, workspace_env["archive_root"]) as _service:
            second = await _service.acquire_sources([source])
        assert second.errors == 0
        assert second.acquired == 1
    finally:
        await backend.close()


def _raw_payload_receipts(archive_root: Path, source_path: Path) -> list[tuple[str, int]]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        rows = conn.execute(
            "SELECT lower(hex(blob_hash)), size_bytes FROM blob_refs WHERE ref_type = 'raw_payload' AND source_path = ?",
            (str(source_path),),
        ).fetchall()
    return [(str(blob_hash), int(size)) for blob_hash, size in rows]


@pytest.mark.asyncio
async def test_acquisition_outcome_is_the_committed_raw_receipt(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file's accepted outcome is its committed raw receipt, nothing earlier.

    Acquisition hands back the retained artifact; only the source-tier raw
    commit accepts it. A failed commit leaves no receipt and no cursor, and
    the next pass -- a restart: a new service over the same archive --
    re-acquires the file and commits the receipt naming its exact bytes.

    Anti-vacuity: publish an admission receipt from acquisition itself and
    the observation stream carries a ``source-admission`` phase for bytes
    whose commit then fails, which the phase assertion rejects.
    """
    from polylogue.sources import source_acquisition

    source_dir = workspace_env["data_root"] / "claude-projects"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "session.jsonl"
    _write_session(source_path)
    source = Source(name="claude-code", path=source_path)
    artifact = (hashlib.sha256(source_path.read_bytes()).hexdigest(), source_path.stat().st_size)

    phases: list[object] = []

    def recording_read(context: SourceReadContext) -> RawSessionData:
        downstream = context.observation_callback

        def record(observation: AcquisitionObservation) -> None:
            phases.append(observation.get("phase"))
            if downstream is not None:
                downstream(observation)

        return read_plain_source_file(dataclasses.replace(context, observation_callback=record))

    monkeypatch.setattr(source_acquisition, "read_plain_source_file", recording_read)
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
        async with _admitted_acquisition(backend, workspace_env["archive_root"]) as _service:
            failed = await _service.acquire_sources([source])
        assert (failed.errors, failed.acquired) == (1, 0)
        assert _raw_payload_receipts(workspace_env["archive_root"], source_path) == []

        async with _admitted_acquisition(backend, workspace_env["archive_root"]) as _service:
            restarted = await _service.acquire_sources([source])
        assert (restarted.errors, restarted.acquired) == (0, 1)
        assert _raw_payload_receipts(workspace_env["archive_root"], source_path) == [artifact]
        known = await SessionRepository(backend=backend).get_known_source_cursors()
        assert str(source_path) in known
    finally:
        await backend.close()

    assert calls == 2
    assert "source-admission" not in phases


def test_a_colon_named_failure_does_not_withhold_its_prefix_sibling(tmp_path: Path) -> None:
    """``run.jsonl`` and ``run.jsonl:1/session.jsonl`` are different files; a
    failure of the second must leave the first's cursor alone. Anti-vacuity:
    indexing every colon prefix withholds ``run.jsonl`` as if it were the
    failed file's ZIP container."""
    import asyncio

    from polylogue.config import Source as ConfigSource

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    plain = source_dir / "run.jsonl"
    colon = source_dir / "run.jsonl:1" / "session.jsonl"
    colon.parent.mkdir()
    for path in (plain, colon):
        path.write_text("{}\n", encoding="utf-8")

    saved: list[str] = []

    class _Repository:
        async def upsert_source_file_cursor(self, path: str, **_stat: object) -> None:
            saved.append(path)

    service = AcquisitionService.__new__(AcquisitionService)
    service.repository = _Repository()  # type: ignore[assignment]
    cursor_state = {"failed_files": [{"path": str(colon), "error": "OSError"}]}
    asyncio.run(
        service._persist_source_cursors(
            ConfigSource(name="claude-code", path=source_dir),
            cursor_state=cursor_state,  # type: ignore[arg-type]
            observations={
                str(path): (
                    str(path.resolve()),
                    (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns),
                    None,
                )
                for path in (plain, colon)
                for st in (path.stat(),)
            },
        )
    )
    assert str(plain) in saved
    assert str(colon) not in saved


def test_a_failure_naming_no_resolved_file_withholds_every_cursor(tmp_path: Path) -> None:
    """A provenance path (a staged SQLite snapshot's original location) names
    no file the walk resolved; nothing can be proven safe to skip. Anti-vacuity:
    dropping an unmatched failure persists the staged file's cursor."""
    import asyncio

    from polylogue.config import Source as ConfigSource

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    staged = source_dir / "state.jsonl"
    staged.write_text("{}\n", encoding="utf-8")
    saved: list[str] = []

    class _Repository:
        async def upsert_source_file_cursor(self, path: str, **_stat: object) -> None:
            saved.append(path)

    service = AcquisitionService.__new__(AcquisitionService)
    service.repository = _Repository()  # type: ignore[assignment]
    asyncio.run(
        service._persist_source_cursors(
            ConfigSource(name="claude-code", path=source_dir),
            cursor_state={"failed_files": [{"path": "/elsewhere/original/state.db", "error": "OSError"}]},
            observations={
                str(staged): (
                    str(staged.resolve()),
                    (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns),
                    None,
                )
                for st in (staged.stat(),)
            },
        )
    )
    assert saved == []


def test_an_excision_refusal_is_not_a_retryable_persistence_failure() -> None:
    """Durably excised content is refused on purpose; recording it as a
    failure would withhold the cursor and retry forbidden bytes every pass.
    Anti-vacuity: appending every exception makes ``failures`` non-empty."""
    import asyncio
    from types import SimpleNamespace

    from polylogue.pipeline.services.acquisition_persistence import persist_raw_record
    from polylogue.pipeline.stage_models import AcquireResult
    from polylogue.security.excision_policy import ExcisionPolicyError

    class _Repository:
        async def admit_raw(self, _request: object) -> object:
            raise ExcisionPolicyError("content is excluded by excision policy")

    failures: list[Any] = []
    result = AcquireResult()
    record = SimpleNamespace(source_name="claude-code", source_path="/src/excised.jsonl")
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            "polylogue.pipeline.services.acquisition_persistence.pending_pre_parse_raw_admission_request",
            lambda record, policy_snapshot=None: object(),
        )
        asyncio.run(persist_raw_record(_Repository(), record, result=result, failures=failures))  # type: ignore[arg-type]
    assert result.errors == 1
    assert failures == []
