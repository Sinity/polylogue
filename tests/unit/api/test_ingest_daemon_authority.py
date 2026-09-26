"""The public facade cannot become an offline writer when the daemon is down."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.api.ingest import IngestDaemonRequiredError


@pytest.mark.asyncio
async def test_parse_file_refuses_without_resident_daemon_before_archive_write(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    source = tmp_path / "session.jsonl"
    source.write_text(
        '{"type":"user","uuid":"u1","sessionId":"s1","message":{"content":"hello"}}\n',
        encoding="utf-8",
    )
    archive = Polylogue(archive_root=root)
    try:
        with pytest.raises(IngestDaemonRequiredError, match="polylogued run"):
            await archive.parse_file(source, source_name="claude-code")
    finally:
        await archive.close()
    assert not (root / "source.db").exists()
    assert not (root / "index.db").exists()
