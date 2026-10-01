"""Probe and prototype scratch survives actual native statement close failure."""

from contextlib import closing
from pathlib import Path
from typing import Any, cast

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
    actual_connect = profiles.connect_measured

    def connect(path: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = actual_connect(path, *args, **kwargs)
        if "polylogue-tier-probe-" in str(path):
            handle = BackupCursorFault(connection, on_target=False, fail_copy=fail_copy)
            handles.append(handle)
            return cast(sqlite3.Connection, handle)
        return connection

    monkeypatch.setattr(profiles, "connect_measured", connect)
    # Avoid unrelated process prototype reuse; the real tier/probe/migration
    # body still builds the isolated durable file and copies it into memory.
    monkeypatch.setattr(bootstrap, "_TIER_PROTOTYPES", {})
    monkeypatch.setattr(bootstrap, "_record_tier_prototype", lambda *_args: None)
    with closing(connect_measured(":memory:")) as destination:
        with pytest.raises(profiles.NativeConnectionSettlementError) as failed:
            bootstrap.initialize_runtime_tier_probe(destination, ArchiveTier.USER)
        owner = failed.value.owner
        assert owner.scratch_directory is not None
        directory = Path(owner.scratch_directory.name)
        assert directory.exists() and (directory / "user.db").exists()
        assert handles[0].cursor is not None
        assert handles[0].cursor.close_attempts == 1
        handles[0].cursor.allow_cleanup.set()
        owner.close()
        assert handles[0].cursor.close_attempts == 2
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
    with closing(connect_measured(":memory:")) as connection:
        connection.execute("CREATE TABLE evidence(value INTEGER)")
        connection.commit()
        source = BackupCursorFault(connection, on_target=True, fail_copy=fail_copy)
        with pytest.raises(profiles.NativeConnectionSettlementError) as failed:
            bootstrap._record_tier_prototype(cast(sqlite3.Connection, source), ArchiveTier.USER, 1)
        owner = failed.value.owner
        assert source.cursor is not None and source.cursor.close_attempts == 1
        assert list(directory.glob("*.tmp"))
        with pytest.raises(RuntimeError):
            bootstrap._cleanup_tier_prototype_dir(directory)
        assert directory.exists()
        source.cursor.allow_cleanup.set()
        owner.close()
        assert source.cursor.close_attempts == 2
        bootstrap._cleanup_tier_prototype_dir(directory)
        assert not directory.exists()
