"""Archive availability helper tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from polylogue.api.archive import _archive_index_available
from polylogue.mcp.context_pack import archive_context_pack_active


def test_runtime_archive_helpers_use_configured_archive_index(tmp_path: Path) -> None:
    """CLI, API, and MCP helpers follow the configured archive root."""
    archive_root = tmp_path / "archive"
    override_root = tmp_path / "override"
    archive_root.mkdir()
    override_root.mkdir()
    db_path = override_root / "index.db"
    (archive_root / "index.db").write_text("index")

    config = SimpleNamespace(archive_root=archive_root, db_path=db_path)

    assert _archive_index_available(cast(Any, config))
    assert archive_context_pack_active(
        archive_root=archive_root,
        db_anchor_path=db_path,
    )


def test_runtime_archive_helpers_do_not_depend_on_polylogue_db(tmp_path: Path) -> None:
    """The old single-file database path is not part of archive availability."""
    archive_root = tmp_path / "archive"
    override_root = tmp_path / "override"
    archive_root.mkdir()
    override_root.mkdir()
    db_path = override_root / "index.db"

    config = SimpleNamespace(archive_root=archive_root, db_path=db_path)

    assert _archive_index_available(cast(Any, config))
    assert archive_context_pack_active(
        archive_root=archive_root,
        db_anchor_path=db_path,
    )
