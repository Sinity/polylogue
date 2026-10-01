"""Probe and prototype scratch survives actual native statement close failure."""

from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite import connection_profile as profiles
from polylogue.storage.sqlite.archive_tiers import bootstrap
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.sqlite_cursor_settlement import BackupCursorFault


@pytest.mark.parametrize("fail_copy", [False, True])
def test_in_memory_probe_retains_actual_scratch_until_cursor_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_copy: bool
) -> None:
    import sqlite3

    handles: list[BackupCursorFault] = []

    actual_connect = connect_measured

    def connect(path: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if "polylogue-tier-probe-" in str(path):
            handle = sqlite3.connect(path, *args, factory=BackupCursorFault, **kwargs)
            assert isinstance(handle, BackupCursorFault)
            handle.fail_copy = fail_copy
            handles.append(handle)
            return handle
        return actual_connect(path, *args, **kwargs)

    monkeypatch.setattr(profiles, "connect_measured", connect)
    # Avoid unrelated process prototype reuse; the real tier/probe/migration
    # body still builds the isolated durable file and copies it into memory.
    monkeypatch.setattr(bootstrap, "_TIER_PROTOTYPES", {})
    monkeypatch.setattr(bootstrap, "_record_tier_prototype", lambda *_args: None)
    with closing(connect_measured(":memory:")) as destination:
        with pytest.raises(profiles.NativeConnectionSettlementError) as failed:
            bootstrap.initialize_runtime_tier_probe(destination, ArchiveTier.USER)
        owner = failed.value.owner
        try:
            assert owner.scratch_directory is not None
            directory = Path(owner.scratch_directory.name)
            assert directory.exists() and (directory / "user.db").exists()
            assert handles[0].retained_cursor is not None
            assert handles[0].retained_cursor.close_attempts == 1
        finally:
            for handle in handles:
                if handle.retained_cursor is not None:
                    handle.retained_cursor.allow_cleanup.set()
            owner.close()
        assert handles[0].retained_cursor is not None
        assert handles[0].retained_cursor.close_attempts == 2
        assert not directory.exists()


@pytest.mark.parametrize("fail_copy", [False, True])
def test_prototype_copy_retains_staging_and_directory_until_native_cursor_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_copy: bool
) -> None:
    import sqlite3

    directory = tmp_path / "prototypes"
    directory.mkdir()
    monkeypatch.setattr(bootstrap, "_TIER_PROTOTYPE_DIR", directory)
    monkeypatch.setattr(bootstrap, "_TIER_PROTOTYPES", {})
    with closing(sqlite3.connect(":memory:", factory=BackupCursorFault)) as connection:
        connection.execute("CREATE TABLE evidence(value INTEGER)")
        connection.commit()
        assert isinstance(connection, BackupCursorFault)
        source = connection
        source.on_target = True
        source.fail_copy = fail_copy
        with pytest.raises(profiles.NativeConnectionSettlementError) as failed:
            bootstrap._record_tier_prototype(source, ArchiveTier.USER, 1)
        owner = failed.value.owner
        try:
            assert source.retained_cursor is not None and source.retained_cursor.close_attempts == 1
            assert list(directory.glob("*.tmp"))
            with pytest.raises(RuntimeError):
                bootstrap._cleanup_tier_prototype_dir(directory)
            assert directory.exists()
        finally:
            if source.retained_cursor is not None:
                source.retained_cursor.allow_cleanup.set()
            owner.close()
            bootstrap._cleanup_tier_prototype_dir(directory)
        assert source.retained_cursor is not None
        assert source.retained_cursor.close_attempts == 2
        assert not directory.exists()
