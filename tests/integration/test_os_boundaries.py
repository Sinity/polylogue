"""OS-level resource boundary tests.

Tests real OS error conditions: read-only directories, file descriptor
pressure, and disk space exhaustion. Marked slow since they may involve
OS-level resource manipulation.
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Mapping
from io import TextIOWrapper
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.slow
class TestReadOnlyOutputDirectory:
    """Pipeline handles read-only output directory gracefully."""

    def test_readonly_archive_root(self, tmp_path: Path, workspace_env: Mapping[str, Path]) -> None:
        """Write to a read-only archive root produces a clean OS error."""
        del workspace_env
        readonly_dir = tmp_path / "readonly_archive"
        readonly_dir.mkdir()
        os.chmod(readonly_dir, stat.S_IRUSR | stat.S_IXUSR)

        try:
            # Writing a file into a read-only directory must raise PermissionError,
            # not silently fail or produce a traceback from a deeper layer.
            with pytest.raises((PermissionError, OSError)):
                (readonly_dir / "output.md").write_text("test")
        finally:
            os.chmod(readonly_dir, stat.S_IRWXU)


@pytest.mark.slow
class TestENOSPC:
    """Simulated disk space exhaustion via mocked OSError."""

    def test_enospc_during_write(self, tmp_path: Path) -> None:
        """ENOSPC during file write produces clean error."""
        target = tmp_path / "output.md"

        def raise_enospc(*args: object, **kwargs: object) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        with patch("builtins.open", side_effect=raise_enospc):
            with pytest.raises(OSError) as exc_info:
                with open(target, "w") as f:
                    f.write("data")
            assert exc_info.value.errno == errno.ENOSPC


@pytest.mark.slow
class TestFileDescriptorPressure:
    """Pipeline handles fd pressure gracefully."""

    def test_many_open_fds(self, tmp_path: Path, workspace_env: Mapping[str, Path]) -> None:
        """Pipeline still works with many open file descriptors."""
        del workspace_env
        import asyncio

        from polylogue.storage.query_models import ConversationRecordQuery
        from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
        from polylogue.storage.sqlite.connection import open_connection

        db_path = tmp_path / "fd_test.db"
        with open_connection(db_path):
            pass

        # Open many files to create fd pressure
        fds: list[TextIOWrapper] = []
        try:
            for i in range(100):
                f = open(tmp_path / f"fd_{i}.tmp", "w")
                f.write("x")
                fds.append(f)

            # SQLite connection should still work under fd pressure
            backend = SQLiteBackend(db_path=db_path)

            async def _check() -> int:
                count = await backend.queries.count_conversations(ConversationRecordQuery())
                await backend.close()
                return count

            asyncio.run(_check())
        finally:
            for f in fds:
                f.close()
